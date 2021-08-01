# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 22:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import cv2
import time
import numpy as np
import torch
import torch.nn as nn

from models.backbone import CSPDarknet
from models.neck.yolo_fpn import YOLOXPAFPN
from models.head.yolo_head import YOLOXHead
from models.losses import YOLOXLoss
from models.post_process import yolox_post_process
from data.data_augment import preproc
from utils.model_utils import load_model


def get_model(opt):
    # define backbone
    backbone_cfg = {"nano": [0.33, 0.25],
                    "tiny": [0.33, 0.375],
                    "s": [0.33, 0.5],
                    "m": [0.67, 0.75],
                    "l": [1.0, 1.0],
                    "x": [1.33, 1.25]}
    depth, width = backbone_cfg[opt.backbone.split("-")[1]]  # "CSPDarknet-s
    in_channel = [256, 512, 1024]
    backbone = CSPDarknet(dep_mul=depth, wid_mul=width, out_indices=(3, 4, 5), depthwise=opt.depth_wise)
    # define neck
    neck = YOLOXPAFPN(depth=depth, width=width, in_channels=in_channel, depthwise=opt.depth_wise)
    # define head
    head = YOLOXHead(num_classes=opt.num_classes, reid_dim=opt.reid_dim, width=width, in_channels=in_channel,
                     depthwise=opt.depth_wise)
    # define loss
    loss = YOLOXLoss(opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums, strides=opt.stride,
                     in_channels=in_channel)
    # define network
    model = YOLOX(opt, backbone=backbone, neck=neck, head=head, loss=loss)
    return model


class YOLOX(nn.Module):
    def __init__(self, opt, backbone, neck, head, loss):
        super(YOLOX, self).__init__()
        self.opt = opt
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def forward(self, inputs, targets=None, return_loss=False, return_pred=True, vis_thresh=0.001, ratio=None,
                show_time=False):
        assert return_loss or return_pred

        loss, predicts = "", ""
        with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
            if show_time:
                torch.cuda.synchronize() if 'cpu' not in inputs.device.type else ""
                s1 = time.time()

            body_feats = self.backbone(inputs)
            neck_feats = self.neck(body_feats)
            yolo_outputs = self.head(neck_feats)
            # print('yolo_outputs:', [i.dtype for i in yolo_outputs])  # float16 when use_amp=True

            if show_time:
                torch.cuda.synchronize() if 'cpu' not in inputs.device.type else ""
                s2 = time.time()
                print("[inference] batch={} time: {}s".format("x".join([str(i) for i in inputs.shape]), s2 - s1))

            if return_loss:
                assert targets is not None
                loss = self.loss(yolo_outputs, targets)
                # for k, v in loss.items():
                #     print(k, v, v.dtype)  # always float32

        if return_pred:
            if ratio is None:
                return yolo_outputs
            assert ratio is not None
            if show_time:
                torch.cuda.synchronize() if 'cpu' not in inputs.device.type else ""
                s3 = time.time()

            predicts = yolox_post_process(yolo_outputs, self.opt.stride, self.opt.num_classes, vis_thresh,
                                          self.opt.nms_thresh, self.opt.label_name, ratio)
            if show_time:
                torch.cuda.synchronize() if 'cpu' not in inputs.device.type else ""
                s4 = time.time()
                print("[post_process] time: {}s".format(s4 - s3))
        if return_loss:
            return predicts, loss
        else:
            return predicts


class Detector(object):
    def __init__(self, cfg):
        self.opt = cfg
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt.pretrained = None
        self.model = get_model(self.opt)
        print("Loading model {}".format(self.opt.load_model))
        self.model = load_model(self.model, self.opt.load_model)
        self.model.to(self.opt.device)
        self.model.eval()

    @staticmethod
    def letterbox(img, dst_shape, color=(0, 0, 0)):
        if type(dst_shape) == int:
            dst_shape = [dst_shape, dst_shape]

        ratio_dst = dst_shape[0] / float(dst_shape[1])
        img_h, img_w = img.shape[0], img.shape[1]

        ratio_org = img_h / float(img_w)
        if ratio_dst > ratio_org:
            scale = dst_shape[1] / float(img_w)
        else:
            scale = dst_shape[0] / float(img_h)

        new_shape = (int(round(img_w * scale)), int(round(img_h * scale)))

        dw = (dst_shape[1] - new_shape[0]) / 2  # width padding
        dh = (dst_shape[0] - new_shape[1]) / 2  # height padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
        return img

    def run(self, images, vis_thresh, show_time=False):
        batch_img = True
        if np.ndim(images) == 3:
            images = [images]
            batch_img = False

        with torch.no_grad():
            if show_time:
                s1 = time.time()
            img_ratios = []
            inp_imgs = np.zeros([len(images), 3, self.opt.test_size[0], self.opt.test_size[1]], dtype=np.float32)
            for b_i, image in enumerate(images):
                img, r = preproc(image, self.opt.test_size, self.opt.rgb_means, self.opt.std)
                inp_imgs[b_i] = img
                img_ratios.append(r)

            if show_time:
                print("[preprocess] time {}".format(time.time() - s1))
            inp_imgs = torch.from_numpy(inp_imgs).to(self.opt.device)
            predicts = self.model(inp_imgs, vis_thresh=vis_thresh, ratio=img_ratios, show_time=show_time)

        if batch_img:
            return predicts
        else:
            return predicts[0]
