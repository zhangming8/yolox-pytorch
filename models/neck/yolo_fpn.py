import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ops import ConvBNLayer
from models.backbone.csp_darknet import BaseConv, CSPLayer, DWConv

__all__ = ['YOLOv3FPN', 'PPYOLOFPN', 'PPYOLOPAN', 'YOLOXPAFPN']


def add_coord(x, data_format):
    b = x.shape[0]
    if data_format == 'NCHW':
        h = x.shape[2]
        w = x.shape[3]
    else:
        h = x.shape[1]
        w = x.shape[2]

    gx = torch.arange(w, dtype=x.dtype, device=x.device) / (w - 1.) * 2.0 - 1.
    gy = torch.arange(h, dtype=x.dtype, device=x.device) / (h - 1.) * 2.0 - 1.

    if data_format == 'NCHW':
        gx = gx.reshape([1, 1, 1, w]).expand([b, 1, h, w])
        gy = gy.reshape([1, 1, h, 1]).expand([b, 1, h, w])
    else:
        gx = gx.reshape([1, 1, w, 1]).expand([b, h, w, 1])
        gy = gy.reshape([1, h, 1, 1]).expand([b, h, w, 1])

    return gx, gy


class YoloDetBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 channel,
                 norm_type,
                 freeze_norm=False,
                 name='',
                 data_format='NCHW'):
        """
        YOLODetBlock layer for yolov3, see https://arxiv.org/abs/1804.02767

        Args:
            ch_in (int): input channel
            channel (int): base channel
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        conv_def = [
            ['conv0', ch_in, channel, 1, '.0.0'],
            ['conv1', channel, channel * 2, 3, '.0.1'],
            ['conv2', channel * 2, channel, 1, '.1.0'],
            ['conv3', channel, channel * 2, 3, '.1.1'],
            ['route', channel * 2, channel, 1, '.2'],
        ]

        self.conv_module = nn.Sequential()
        for idx, (conv_name, ch_in, ch_out, filter_size, post_name) in enumerate(conv_def):
            self.conv_module.add_module(
                conv_name,
                ConvBNLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=filter_size,
                    padding=(filter_size - 1) // 2,
                    norm_type=norm_type,
                    freeze_norm=freeze_norm))

        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            padding=1,
            norm_type=norm_type,
            freeze_norm=freeze_norm)

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class SPP(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 norm_type,
                 freeze_norm=False,
                 name='',
                 act='leaky',
                 data_format='NCHW'):
        """
        SPP layer, which consist of four pooling layer follwed by conv layer

        Args:
            ch_in (int): input channel of conv layer
            ch_out (int): output channel of conv layer
            k (int): kernel size of conv layer
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            act (str): activation function
            data_format (str): data format, NCHW or NHWC
        """
        super(SPP, self).__init__()
        self.pool = nn.Sequential()
        self.data_format = data_format
        for size in pool_size:
            self.pool.add_module(
                '{}_pool{}'.format(name, size),
                nn.MaxPool2d(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    ceil_mode=False))

        self.conv = ConvBNLayer(
            ch_in,
            ch_out,
            k,
            padding=k // 2,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            act=act)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == "NCHW":
            y = torch.cat(outs, dim=1)
        else:
            y = torch.cat(outs, dim=-1)
        y = self.conv(y)
        return y


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (float): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size ** 2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = (torch.rand(x.shape, dtype=x.dtype, device=x.device) < gamma).float()
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class CoordConv(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 padding,
                 norm_type,
                 freeze_norm=False,
                 name='',
                 data_format='NCHW'):
        """
        CoordConv layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            name (str): layer name
            data_format (str): data format, NCHW or NHWC

        """
        super(CoordConv, self).__init__()
        self.conv = ConvBNLayer(
            ch_in + 2,
            ch_out,
            filter_size=filter_size,
            padding=padding,
            norm_type=norm_type,
            freeze_norm=freeze_norm)
        self.data_format = data_format

    def forward(self, x):
        gx, gy = add_coord(x, self.data_format)
        if self.data_format == 'NCHW':
            y = torch.cat([x, gx, gy], dim=1)
        else:
            y = torch.cat([x, gx, gy], dim=-1)
        y = self.conv(y)
        return y


class PPYOLODetBlock(nn.Module):
    def __init__(self, cfg, name, data_format='NCHW'):
        """
        PPYOLODetBlock layer

        Args:
            cfg (list): layer configs for this block
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlock, self).__init__()
        self.conv_module = nn.Sequential()
        for idx, (conv_name, layer, args, kwargs) in enumerate(cfg[:-1]):
            kwargs.update(name='{}_{}'.format(name, conv_name), data_format=data_format)
            self.conv_module.add_module(conv_name, layer(*args, **kwargs))

        conv_name, layer, args, kwargs = cfg[-1]
        kwargs.update(name='{}_{}'.format(name, conv_name), data_format=data_format)
        self.tip = layer(*args, **kwargs)

    def forward(self, inputs):
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class PPYOLOTinyDetBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 name,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 data_format='NCHW'):
        """
        PPYOLO Tiny DetBlock layer
        Args:
            ch_in (list): input channel number
            ch_out (int): output channel number
            name (str): block name
            drop_block: whether user DropBlock
            block_size: drop block size
            keep_prob: probability to keep block in DropBlock
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLOTinyDetBlock, self).__init__()
        self.drop_block_ = drop_block
        self.conv_module = nn.Sequential()

        cfgs = [
            # name, in channels, out channels, filter_size, stride, padding, groups
            ['_0', ch_in, ch_out, 1, 1, 0, 1],
            ['_1', ch_out, ch_out, 5, 1, 2, ch_out],
            ['_2', ch_out, ch_out, 1, 1, 0, 1],
            ['_route', ch_out, ch_out, 5, 1, 2, ch_out],
        ]
        for cfg in cfgs:
            conv_name, conv_ch_in, conv_ch_out, filter_size, stride, padding, groups = cfg
            self.conv_module.add_module(
                name + conv_name,
                ConvBNLayer(
                    ch_in=conv_ch_in,
                    ch_out=conv_ch_out,
                    filter_size=filter_size,
                    stride=stride,
                    padding=padding,
                    groups=groups))

        self.tip = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1)

        if self.drop_block_:
            self.drop_block = DropBlock(
                block_size=block_size,
                keep_prob=keep_prob,
                data_format=data_format,
                name=name + '_dropblock')

    def forward(self, inputs):
        if self.drop_block_:
            inputs = self.drop_block(inputs)
        route = self.conv_module(inputs)
        tip = self.tip(route)
        return route, tip


class PPYOLODetBlockCSP(nn.Module):
    def __init__(self,
                 cfg,
                 ch_in,
                 ch_out,
                 act,
                 norm_type,
                 name,
                 data_format='NCHW'):
        """
        PPYOLODetBlockCSP layer

        Args:
            cfg (list): layer configs for this block
            ch_in (int): input channel
            ch_out (int): output channel
            act (str): default mish
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlockCSP, self).__init__()
        self.data_format = data_format
        self.conv1 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type)
        self.conv2 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type)
        self.conv3 = ConvBNLayer(
            ch_out * 2,
            ch_out * 2,
            1,
            padding=0,
            act=act,
            norm_type=norm_type)
        self.conv_module = nn.Sequential()
        for idx, (layer_name, layer, args, kwargs) in enumerate(cfg):
            kwargs.update(name=name + layer_name, data_format=data_format)
            self.conv_module.add_module(str(layer_name), layer(*args, **kwargs))

    def forward(self, inputs):
        conv_left = self.conv1(inputs)
        conv_right = self.conv2(inputs)
        conv_left = self.conv_module(conv_left)
        if self.data_format == 'NCHW':
            conv = torch.cat([conv_left, conv_right], dim=1)
        else:
            conv = torch.cat([conv_left, conv_right], dim=-1)

        conv = self.conv3(conv)
        return conv, conv


class YOLOv3FPN(nn.Module):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 norm_type='bn',
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        YOLOv3FPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC

        """
        super(YOLOv3FPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)

        self._out_channels = []
        self.yolo_blocks = nn.Sequential()
        self.routes = nn.Sequential()
        self.data_format = data_format
        for i in range(self.num_blocks):
            name = 'yolo_block_{}'.format(i)
            in_channel = in_channels[-i - 1]
            if i > 0:
                in_channel += 512 // (2 ** i)
            self.yolo_blocks.add_module(
                name,
                YoloDetBlock(
                    in_channel,
                    channel=512 // (2 ** i),
                    norm_type=norm_type,
                    freeze_norm=freeze_norm,
                    data_format=data_format,
                    name=name))
            # tip layer output channel doubled
            self._out_channels.append(1024 // (2 ** i))

            if i < self.num_blocks - 1:
                name = 'yolo_transition_{}'.format(i)
                self.routes.add_module(
                    name,
                    ConvBNLayer(
                        ch_in=512 // (2 ** i),
                        ch_out=256 // (2 ** i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        freeze_norm=freeze_norm))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []

        # add embedding features output for multi-object tracking model
        if for_mot:
            emb_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], dim=1)
                else:
                    block = torch.cat([route, block], dim=-1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if for_mot:
                # add embedding features output
                emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        if for_mot:
            return {'yolo_feats': yolo_feats, 'emb_feats': emb_feats}
        else:
            return yolo_feats


class PPYOLOFPN(nn.Module):

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 norm_type='bn',
                 freeze_norm=False,
                 data_format='NCHW',
                 coord_conv=False,
                 conv_block_num=2,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 init_cfg=None,
                 ):
        """
        PPYOLOFPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            coord_conv (bool): whether use CoordConv or not
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not

        """
        super(PPYOLOFPN, self).__init__(init_cfg)
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.coord_conv = coord_conv
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.coord_conv:
            ConvLayer = CoordConv
        else:
            ConvLayer = ConvBNLayer

        if self.drop_block:
            dropblock_cfg = [['dropblock', DropBlock, [self.block_size, self.keep_prob], dict()]]
        else:
            dropblock_cfg = []

        self._out_channels = []
        self.yolo_blocks = nn.Sequential()
        self.routes = nn.Sequential()
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // (2 ** i)
            channel = 64 * (2 ** self.num_blocks) // (2 ** i)
            base_cfg = []
            c_in, c_out = ch_in, channel
            for j in range(self.conv_block_num):
                base_cfg += [
                    [
                        'conv{}'.format(2 * j), ConvLayer, [c_in, c_out, 1],
                        dict(
                            padding=0,
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
                    ],
                    [
                        'conv{}'.format(2 * j + 1), ConvBNLayer,
                        [c_out, c_out * 2, 3], dict(
                        padding=1,
                        norm_type=norm_type,
                        freeze_norm=freeze_norm)
                    ],
                ]
                c_in, c_out = c_out * 2, c_out

            base_cfg += [[
                'route', ConvLayer, [c_in, c_out, 1], dict(padding=0, norm_type=norm_type, freeze_norm=freeze_norm)
            ], [
                'tip', ConvLayer, [c_out, c_out * 2, 3], dict(padding=1, norm_type=norm_type, freeze_norm=freeze_norm)
            ]]

            if self.conv_block_num == 2:
                if i == 0:
                    if self.spp:
                        spp_cfg = [[
                            'spp', SPP, [channel * 4, channel, 1], dict(
                                pool_size=[5, 9, 13],
                                norm_type=norm_type,
                                freeze_norm=freeze_norm)
                        ]]
                    else:
                        spp_cfg = []
                    cfg = base_cfg[0:3] + spp_cfg + base_cfg[
                                                    3:4] + dropblock_cfg + base_cfg[4:6]
                else:
                    cfg = base_cfg[0:2] + dropblock_cfg + base_cfg[2:6]
            elif self.conv_block_num == 0:
                if self.spp and i == 0:
                    spp_cfg = [[
                        'spp', SPP, [c_in * 4, c_in, 1], dict(
                            pool_size=[5, 9, 13],
                            norm_type=norm_type,
                            freeze_norm=freeze_norm)
                    ]]
                else:
                    spp_cfg = []
                cfg = spp_cfg + dropblock_cfg + base_cfg
            name = 'yolo_block_{}'.format(i)
            self.yolo_blocks.add_module(name, PPYOLODetBlock(cfg, name))
            self._out_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = 'yolo_transition_{}'.format(i)
                self.routes.add_module(
                    name,
                    ConvBNLayer(
                        ch_in=channel,
                        ch_out=256 // (2 ** i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        freeze_norm=freeze_norm))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []

        # add embedding features output for multi-object tracking model
        if for_mot:
            emb_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], dim=1)
                else:
                    block = torch.cat([route, block], dim=-1)
            route, tip = self.yolo_blocks[i](block)
            yolo_feats.append(tip)

            if for_mot:
                # add embedding features output
                emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        if for_mot:
            return {'yolo_feats': yolo_feats, 'emb_feats': emb_feats}
        else:
            return yolo_feats


class PPYOLOPAN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 norm_type='bn',
                 data_format='NCHW',
                 act='mish',
                 conv_block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        """
        PPYOLOPAN layer with SPP, DropBlock and CSP connection.

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            act (str): activation function, default mish
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not

        """
        super(PPYOLOPAN, self).__init__(init_cfg)
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        if self.drop_block:
            dropblock_cfg = [['dropblock', DropBlock, [self.block_size, self.keep_prob], dict()]]
        else:
            dropblock_cfg = []

        # fpn
        self.fpn_block_names = []
        self.fpn_routes_names = []
        fpn_channels = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += 512 // (2 ** (i - 1))
            channel = 512 // (2 ** i)
            base_cfg = []
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    ['{}_0'.format(j), ConvBNLayer, [channel, channel, 1],
                     dict(padding=0, act=act, norm_type=norm_type)],
                    ['{}_1'.format(j), ConvBNLayer, [channel, channel, 3],
                     dict(padding=1, act=act, norm_type=norm_type)]
                ]

            if i == 0 and self.spp:
                base_cfg[3] = [
                    'spp', SPP, [channel * 4, channel, 1], dict(pool_size=[5, 9, 13], act=act, norm_type=norm_type)
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'fpn_{}'.format(i)
            fpn_block = PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name, data_format)
            self.add_module(name, fpn_block)
            self.fpn_block_names.append(name)

            fpn_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                route = ConvBNLayer(
                    ch_in=channel * 2,
                    ch_out=channel,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=act,
                    norm_type=norm_type)
                name = "route_{}".format(i)
                self.add_module(name, route)
                self.fpn_routes_names.append(name)
        # pan
        self.pan_blocks = []
        self.pan_routes = []
        self._out_channels = [512 // (2 ** (self.num_blocks - 2)), ]
        for i in reversed(range(self.num_blocks - 1)):
            route = ConvBNLayer(
                ch_in=fpn_channels[i + 1],
                ch_out=fpn_channels[i + 1],
                filter_size=3,
                stride=2,
                padding=1,
                act=act,
                norm_type=norm_type)
            self.pan_routes = [route, ] + self.pan_routes
            base_cfg = []
            ch_in = fpn_channels[i] + fpn_channels[i + 1]
            channel = 512 // (2 ** i)
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    [
                        '{}_0'.format(j), ConvBNLayer, [channel, channel, 1],
                        dict(
                            padding=0, act=act, norm_type=norm_type)
                    ],
                    [
                        '{}_1'.format(j), ConvBNLayer, [channel, channel, 3],
                        dict(
                            padding=1, act=act, norm_type=norm_type)
                    ]
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = 'pan_{}'.format(i)
            pan_block = PPYOLODetBlockCSP(cfg, ch_in, channel, act, norm_type, name, data_format)
            self.pan_blocks = [pan_block, ] + self.pan_blocks
            self._out_channels.append(channel * 2)

        self._out_channels = self._out_channels[::-1]
        self.pan_blocks = nn.Sequential(*self.pan_blocks)
        self.pan_routes = nn.Sequential(*self.pan_routes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        fpn_feats = []

        # add embedding features output for multi-object tracking model
        if for_mot:
            emb_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'NCHW':
                    block = torch.cat([route, block], dim=1)
                else:
                    block = torch.cat([route, block], dim=-1)
            fpn_block = getattr(self, self.fpn_block_names[i])
            route, tip = fpn_block(block)
            fpn_feats.append(tip)

            if for_mot:
                # add embedding features output
                emb_feats.append(route)

            if i < self.num_blocks - 1:
                fpn_route = getattr(self, self.fpn_routes_names[i])
                route = fpn_route(route)
                route = F.interpolate(route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[self.num_blocks - 1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            if self.data_format == 'NCHW':
                block = torch.cat([route, block], dim=1)
            else:
                block = torch.cat([route, block], dim=-1)

            route, tip = self.pan_blocks[i](block)
            pan_feats.append(tip)

        if for_mot:
            return {'yolo_feats': pan_feats[::-1], 'emb_feats': emb_feats}
        else:
            return pan_feats[::-1]


class YOLOXPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, blocks):
        assert len(blocks) == len(self.in_channels)
        [x2, x1, x0] = blocks

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs


if __name__ == "__main__":
    from thop import profile

    in_channels = [96, 192, 384]
    feats = [torch.rand([1, in_channels[0], 64, 64]), torch.rand([1, in_channels[1], 32, 32]),
             torch.rand([1, in_channels[2], 16, 16])]

    # fpn = PPYOLOPAN(in_channels, norm_type='bn', act='mish', conv_block_num=3, drop_block=True, block_size=3, spp=True)
    # fpn = PPYOLOFPN(in_channels, coord_conv=True, drop_block=True, block_size=3, keep_prob=0.9, spp=True)
    # fpn = YOLOv3FPN(in_channels)
    # fpn = PPYOLOTinyFPN(in_channels)
    fpn = YOLOXPAFPN(depth=0.33, width=0.375)
    fpn.init_weights()
    # print(fpn)
    fpn.eval()
    # total_ops, total_params = profile(fpn, (feats,))
    # print("total_ops {:.2f}G, total_params {:.2f}M".format(total_ops/1e9, total_params/1e6))
    output = fpn(feats)
    for o in output:
        print(o.size())
