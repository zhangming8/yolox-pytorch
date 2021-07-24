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

opt.exp_id = "coco_CSPDarknet-s_640x640"
opt.dataset_path = "/media/ming/DATA1/dataset/coco2017"
# opt.dataset_path = r"D:\work\public_dataset\coco2017"
opt.backbone = "CSPDarknet-s"  # CSPDarknet-nano, CSPDarknet-tiny, CSPDarknet-s, CSPDarknet-m, l, x
opt.input_size = (640, 640)
opt.test_size = (640, 640)
opt.gpus = "-1"  # "-1" "0,1,2,3,4,5,6,7"
opt.batch_size = 2
opt.master_batch_size = -1  # batch size in first gpu
opt.num_epochs = 300
opt.random_size = (14, 26)  # None
opt.accumulate = 1  # real batch size = accumulate * batch_size

# TODO: support MOT(multi-object tracking) like FairMot/JDE when reid_dim > 0
opt.reid_dim = 0  # 128
opt.id_num = None  # tracking id number in train dataset

opt.warmup_lr = 0
opt.basic_lr_per_img = 0.01 / 64.0
opt.scheduler = "yoloxwarmcos"
opt.no_aug_epochs = 15
opt.min_lr_ratio = 0.05
opt.weight_decay = 5e-4
opt.warmup_epochs = 5
opt.depth_wise = False
opt.stride = [8, 16, 32]

# train augments
opt.degrees = 10.0
opt.translate = 0.1
opt.scale = (0.1, 2)
opt.shear = 2.0
opt.perspective = 0.0
opt.enable_mixup = True
opt.seed = 0
opt.data_num_workers = 0

opt.momentum = 0.9
opt.vis_thresh = 0.1  # inference confidence
opt.load_model = ''
opt.ema = True  # False
opt.grad_clip = dict(max_norm=35, norm_type=2)  # None
opt.print_iter = 1
opt.metric = "loss"  # 'Ap' or 'loss', slowly when set 'Ap'
opt.val_intervals = 1
opt.save_epoch = 1
opt.resume = False  # resume from 'model_last.pth'
opt.use_amp = False  # True
opt.cuda_benchmark = True
opt.nms_thresh = 0.65

opt.label_name = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
opt.rgb_means = [0.485, 0.456, 0.406]
opt.std = [0.229, 0.224, 0.225]

opt = merge_opt(opt, sys.argv[1:])
if opt.backbone.lower().split("-")[1] in ["tiny", "nano"]:
    opt = update_nano_tiny(opt)

# do not modify the flowing params
opt.train_ann = opt.dataset_path + "/annotations/instances_train2017.json"
opt.val_ann = opt.dataset_path + "/annotations/instances_val2017.json"
opt.data_dir = opt.dataset_path + "/images"
opt.num_classes = len(opt.label_name)
opt.gpus_str = opt.gpus
opt.metric = opt.metric.lower()
opt.gpus = [int(i) for i in opt.gpus.split(',')]
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
if opt.random_size is not None:
    opt.cuda_benchmark = False
