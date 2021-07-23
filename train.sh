#!/usr/bin/env bash

# train
# rtx3090
python train.py gpus='0' backbone="CSPDarknet-s" batch_size=24 num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False dataset_path="/data/dataset/coco_dataset" val_intervals=1 data_num_workers=8

# load pretrain checkpoint
#python train.py gpus='0' backbone="CSPDarknet-s" batch_size=24 num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False dataset_path="/data/dataset/coco_dataset" val_intervals=1 data_num_workers=8 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" resume=False

# resume 'model_best.pth / model_num.pth', includes checkpoint and epoch
#python train.py gpus='0' backbone="CSPDarknet-s" batch_size=24 num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False dataset_path="/data/dataset/coco_dataset" val_intervals=1 data_num_workers=8 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" resume=True

# resume 'model_last.pth', includes checkpoint, optimizer, scaler and epoch
#python train.py gpus='0' backbone="CSPDarknet-s" batch_size=24 num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False dataset_path="/data/dataset/coco_dataset" val_intervals=1 data_num_workers=8 load_model="exp/coco_CSPDarknet-s_640x640/model_last.pth" resume=True
