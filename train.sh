#!/usr/bin/env bash

# train from scratch
# yolox-s
python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=8
# yolox-tiny
#python train.py gpus='0' backbone="CSPDarknet-tiny" num_epochs=300 exp_id="coco_CSPDarknet-tiny_416x416" use_amp=False val_intervals=2 data_num_workers=8 metric="ap" random_size=None batch_size=128 input_size="(416,416)"

# load pre-train weight
#python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=8 load_model="../weights/yolox-s.pth" resume=False

# resume 'model_best.pth / model_num.pth', includes weight and epoch
#python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=8 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" resume=True

# resume 'model_last.pth', includes weight, optimizer, scaler and epoch
#python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=8 load_model="exp/coco_CSPDarknet-s_640x640/model_last.pth" resume=True
