from numbers import Integral
from collections import namedtuple
from mmcv.runner import BaseModule

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

__all__ = ['ResNet']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among torch modules.
    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super(ShapeSpec, cls).__new__(cls, channels, height, width, stride)


class NameAdapter(object):
    """Fix the backbones variable names for pretrained weight"""

    def __init__(self, model):
        super(NameAdapter, self).__init__()
        self.model = model

    @property
    def model_type(self):
        return getattr(self.model, '_model_type', '')

    @property
    def variant(self):
        return getattr(self.model, 'variant', '')

    def fix_conv_norm_name(self, name):
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        # the naming rule is same as pretrained weight
        if self.model_type == 'SEResNeXt':
            bn_name = name + "_bn"
        return bn_name

    def fix_shortcut_name(self, name):
        if self.model_type == 'SEResNeXt':
            name = 'conv' + name + '_prj'
        return name

    def fix_bottleneck_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            conv_name3 = 'conv' + name + '_x3'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            conv_name3 = name + "_branch2c"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, conv_name3, shortcut_name

    def fix_basicblock_name(self, name):
        if self.model_type == 'SEResNeXt':
            conv_name1 = 'conv' + name + '_x1'
            conv_name2 = 'conv' + name + '_x2'
            shortcut_name = name
        else:
            conv_name1 = name + "_branch2a"
            conv_name2 = name + "_branch2b"
            shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, shortcut_name

    def fix_layer_warp_name(self, stage_num, count, i):
        name = 'res' + str(stage_num)
        if count > 10 and stage_num == 4:
            if i == 0:
                conv_name = name + "a"
            else:
                conv_name = name + "b" + str(i)
        else:
            conv_name = name + chr(ord("a") + i)
        if self.model_type == 'SEResNeXt':
            conv_name = str(stage_num + 2) + '_' + str(i + 1)
        return conv_name

    def fix_c1_stage_name(self):
        return "res_conv1" if self.model_type == 'ResNeXt' else "conv1"


class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 act=None,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.act = act
        self.dcn_v2 = dcn_v2

        if not self.dcn_v2:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=False)
        else:
            self.offset_channel = 2 * filter_size ** 2
            self.mask_channel = filter_size ** 2

            self.conv_offset = nn.Conv2d(
                in_channels=ch_in,
                out_channels=3 * filter_size ** 2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                bias=True)
            self.conv = DeformConv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                dilation=1,
                groups=groups,
                bias=False)

        if norm_type == 'sync_bn':
            self.norm = nn.SyncBatchNorm(ch_out)
        else:
            self.norm = nn.BatchNorm2d(ch_out)

        if freeze_norm:
            norm_params = self.norm.parameters()
            for param in norm_params:
                param.require_grad = False

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset, mask = torch.split(
                offset_mask,
                split_size_or_sections=[self.offset_channel, self.mask_channel],
                dim=1)
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)

        if self.norm_type in ['bn', 'sync_bn']:
            out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out


class SELayer(nn.Module):
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        c_ = ch // reduction_ratio
        self.squeeze = nn.Linear(ch, c_, bias=True)

        self.extract = nn.Linear(c_, ch, bias=True)

    def forward(self, inputs):
        bs = inputs.shape[0]
        out = self.pool(inputs)
        out = out.view(bs, 1)
        out = self.squeeze(out)
        out = F.relu(out)
        out = self.extract(out)
        out = torch.sigmoid(out)
        out = out.view(bs, -1, 1, 1)
        scale = out * inputs
        return scale


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(BasicBlock, self).__init__()
        assert dcn_v2 is False, "Not implemented yet."
        assert groups == 1 and base_width == 64, 'BasicBlock only supports groups=1 and base_width=64'

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_module(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = out + short
        out = F.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 variant='b',
                 groups=1,
                 base_width=4,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(BottleNeck, self).__init__()
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        # ResNeXt
        width = int(ch_out * (base_width / 64.)) * groups

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential()
                self.short.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
                self.short.add_module(
                    'conv',
                    ConvNormLayer(
                        ch_in=ch_in,
                        ch_out=ch_out * self.expansion,
                        filter_size=1,
                        stride=1,
                        norm_type=norm_type,
                        norm_decay=norm_decay,
                        freeze_norm=freeze_norm,
                        lr=lr))
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out * self.expansion,
                    filter_size=1,
                    stride=stride,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=lr)

        self.branch2a = ConvNormLayer(
            ch_in=ch_in,
            ch_out=width,
            filter_size=1,
            stride=stride1,
            groups=1,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.branch2b = ConvNormLayer(
            ch_in=width,
            ch_out=width,
            filter_size=3,
            stride=stride2,
            groups=groups,
            act='relu',
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr,
            dcn_v2=dcn_v2)

        self.branch2c = ConvNormLayer(
            ch_in=width,
            ch_out=ch_out * self.expansion,
            filter_size=1,
            stride=1,
            groups=1,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            lr=lr)

        self.std_senet = std_senet
        if self.std_senet:
            self.se = SELayer(ch_out * self.expansion)

    def forward(self, inputs):

        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = out + short
        out = F.relu(out)

        return out


class Blocks(nn.Module):
    def __init__(self,
                 block,
                 ch_in,
                 ch_out,
                 count,
                 name_adapter,
                 stage_num,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False):
        super(Blocks, self).__init__()

        self.blocks_name = []
        for i in range(count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            self.blocks_name.append(conv_name)
            self.add_module(
                conv_name,
                block(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=dcn_v2,
                    std_senet=std_senet))

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        block_out = inputs
        for name in self.blocks_name:
            block_out = getattr(self, name)(block_out)
        return block_out


class ResNet(BaseModule):

    def __init__(self,
                 depth=50,
                 ch_in=64,
                 variant='d',
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0],
                 groups=1,
                 base_width=64,
                 norm_type='bn',
                 norm_decay=0,
                 freeze_norm=False,
                 freeze_at=-1,
                 out_indices=[1, 2, 3],
                 dcn_v2_stages=[3],
                 num_stages=4,
                 std_senet=False,
                 pretrained=None,
                 init_cfg=None):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385

        Args:
            depth (int): ResNet depth, should be 18, 34, 50, 101, 152.
            ch_in (int): output channel of first stage, default 64
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            lr_mult_list (list): learning rate ratio of different resnet stages(2,3,4,5),
                                 lower learning rate ratio is need for pretrained model
                                 got using distillation(default as [1.0, 1.0, 1.0, 1.0]).
            groups (int): group convolution cardinality
            base_width (int): base width of each group convolution
            norm_type (str): normalization type, 'bn', 'sync_bn' or 'affine_channel'
            norm_decay (float): weight decay for normalization layer weights
            freeze_norm (bool): freeze normalization layers
            freeze_at (int): freeze the backbone at which stage
            out_indices (list): index of the stages whose feature maps are returned
            dcn_v2_stages (list): index of stages who select deformable conv v2
            num_stages (int): total num of stages
            std_senet (bool): whether use senet, default True
        """
        super(ResNet, self).__init__()
        self._model_type = 'ResNet' if groups == 1 else 'ResNeXt'
        assert 1 <= num_stages <= 4
        self.depth = depth
        self.variant = variant
        self.groups = groups
        self.base_width = base_width
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.freeze_at = freeze_at
        if isinstance(out_indices, Integral):
            out_indices = [out_indices]
        assert max(out_indices) < num_stages, \
            'the maximum return index must smaller than num_stages, ' \
            'but received maximum return index is {} and num_stages ' \
            'is {}'.format(max(out_indices), num_stages)
        self.out_indices = out_indices
        self.num_stages = num_stages
        assert len(lr_mult_list) == 4, \
            "lr_mult_list length must be 4 but got {}".format(len(lr_mult_list))
        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages

        if isinstance(dcn_v2_stages, Integral):
            dcn_v2_stages = [dcn_v2_stages]
        assert max(dcn_v2_stages) < num_stages
        self.dcn_v2_stages = dcn_v2_stages

        block_nums = ResNet_cfg[depth]
        na = NameAdapter(self)

        conv1_name = na.fix_c1_stage_name()
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, conv1_name]]
        self.conv1 = nn.Sequential()
        for (c_in, c_out, k, s, _name) in conv_def:
            self.conv1.add_module(
                _name,
                ConvNormLayer(
                    ch_in=c_in,
                    ch_out=c_out,
                    filter_size=k,
                    stride=s,
                    groups=1,
                    act='relu',
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    lr=1.0))

        self.ch_in = ch_in
        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        self._out_channels = [block.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]

        self.res_layer_name = []
        for i in range(num_stages):
            lr_mult = lr_mult_list[i]
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            self.res_layer_name.append(res_name)
            # self.res_layers.add_module(
            self.add_module(
                res_name,
                Blocks(
                    block,
                    self.ch_in,
                    ch_out_list[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr_mult,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=(i in self.dcn_v2_stages),
                    std_senet=std_senet))

            self.ch_in = self._out_channels[i]

        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            print('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self._out_channels[i], stride=self._out_strides[i]) for i in self.out_indices]

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layer_name):
            x = getattr(self, stage)(x)
            if idx == self.freeze_at:
                x.require_grad = False
            if idx in self.out_indices:
                outs.append(x)
        return outs


if __name__ == "__main__":
    from thop import profile

    inputs = torch.rand(1, 3, 416, 416)
    # resnet-vd-DCN in ppyolov2
    # use 'tools/paddle_to_torch.py' to convert weight
    pretrain_weight = None  # "/media/ming/DATA2/project/weights/ResNet50_vd_ssld_pretrained.pth"
    m = ResNet(depth=50, variant='d', out_indices=[1, 2, 3], dcn_v2_stages=[3], pretrained=pretrain_weight)
    # original resent
    # m = ResNet(depth=50, out_indices=[1, 2, 3], dcn_v2_stages=[-1])
    # print(m)
    m.init_weights()
    m.eval()
    level_outputs = m(inputs)
    total_ops, total_params = profile(m, (inputs,))
    print("total_ops {}, total_params {}".format(total_ops, total_params))

    for level_out in level_outputs:
        print(tuple(level_out.shape))
