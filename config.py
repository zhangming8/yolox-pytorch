# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 20:24
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import sys
from easydict import EasyDict
from utils.util import merge_opt


def update_nano_tiny(cfg):
    cfg.scale = (0.5, 1.5)
    cfg.random_size = (10, 20)
    cfg.test_size = (416, 416)
    cfg.enable_mixup = False
    if 'nano' in cfg.backbone:
        cfg.depth_wise = True
    return cfg


opt = EasyDict()

opt.exp_id = "coco_CSPDarknet-s_640x640"  # experiment name, you can change it to any other name
opt.dataset_path = "/data/dataset/coco_dataset"  # COCO detection
# opt.dataset_path = r"D:\work\public_dataset\coco2017"  # Windows system
# opt.dataset_path = "/media/ming/DATA1/dataset/VisDrone"  # MOT tracking
opt.backbone = "CSPDarknet-s"  # CSPDarknet-nano, CSPDarknet-tiny, CSPDarknet-s, CSPDarknet-m, l, x
opt.input_size = (640, 640)
opt.random_size = (14, 26)  # None; multi-size train: from 448 to 800, random sample an int value and *32 as input size
opt.test_size = (640, 640)  # evaluate size
opt.gpus = "0"  # "-1" "0" "3,4,5" "0,1,2,3,4,5,6,7" # -1 for cpu
opt.batch_size = 24
opt.master_batch_size = -1  # batch size in first gpu. -1 means: master_batch_size=batch_size//len(gpus)
opt.num_epochs = 300

# coco 80 classes
opt.label_name = [
   'redheader','stamp']

# TODO: support MOT(multi-object tracking) like FairMot/JDE when reid_dim > 0
opt.reid_dim = 0  # 128  used in MOT, will add embedding branch if reid_dim>0
# opt.label_name = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
#                   'motor']
# tracking id number of label_name in MOT train dataset
opt.tracking_id_nums = None  # [1829, 853, 323, 3017, 295, 159, 215, 79, 55, 749]

# base params
opt.warmup_lr = 0  # start lr when warmup
opt.basic_lr_per_img = 0.01 / 64.0
opt.scheduler = "yoloxwarmcos"
opt.no_aug_epochs = 15  # close mixup and mosaic augments in the last 15 epochs
opt.accumulate = 1  # real batch size = accumulate * batch_size
opt.min_lr_ratio = 0.05
opt.weight_decay = 5e-4
opt.warmup_epochs = 5
opt.depth_wise = False  # depth_wise conv is used in 'CSPDarknet-nano'
opt.stride = [8, 16, 32]  # YOLOX down sample ratio: 8, 16, 32

# train augments
opt.degrees = 10.0  # rotate angle
opt.translate = 0.1
opt.scale = (0.1, 2)
opt.shear = 2.0
opt.perspective = 0.0
opt.enable_mixup = True
opt.seed = 0
opt.data_num_workers = 0

opt.momentum = 0.9
opt.vis_thresh = 0.3  # inference confidence, used in 'predict.py'
opt.load_model = ''
opt.ema = True  # False, Exponential Moving Average
opt.grad_clip = dict(max_norm=35, norm_type=2)  # None, clip gradient makes training more stable
opt.print_iter = 1  # print loss every 1 iteration
opt.metric = "loss"  # 'Ap' 'loss', a little slow when set 'Ap'
opt.val_intervals = 1  # evaluate(when metric='Ap') and save best ckpt every 1 epoch
opt.save_epoch = 1  # save check point every 1 epoch
opt.resume = False  # resume from 'model_last.pth' when set True
opt.use_amp = False  # True
opt.cuda_benchmark = True
opt.nms_thresh = 0.65

opt.rgb_means = [0.485, 0.456, 0.406]
opt.std = [0.229, 0.224, 0.225]

opt = merge_opt(opt, sys.argv[1:])
if opt.backbone.lower().split("-")[1] in ["tiny", "nano"]:
    opt = update_nano_tiny(opt)

# do not modify the following params
opt.train_ann = opt.dataset_path + "/annotations/instances_train2017.json"
opt.val_ann = opt.dataset_path + "/annotations/instances_val2017.json"
opt.data_dir = opt.dataset_path + "/images"
opt.num_classes = len(opt.label_name)
opt.gpus_str = opt.gpus
opt.metric = opt.metric.lower()
opt.gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
if opt.master_batch_size == -1:
    opt.master_batch_size = opt.batch_size // len(opt.gpus)
rest_batch_size = opt.batch_size - opt.master_batch_size
opt.chunk_sizes = [opt.master_batch_size]
for i in range(len(opt.gpus) - 1):
    slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
    if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
    opt.chunk_sizes.append(slave_chunk_size)
opt.root_dir = os.path.dirname(__file__)
opt.save_dir = os.path.join(opt.root_dir, 'exp', opt.exp_id)
if opt.resume and opt.load_model == '':
    opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
if opt.random_size is not None and (opt.random_size[1] - opt.random_size[0] > 1):
    opt.cuda_benchmark = False
if opt.reid_dim > 0:
    assert opt.tracking_id_nums is not None
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
