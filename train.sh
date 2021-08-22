#!/usr/bin/env bash

# train from scratch
#python train.py gpus='0' backbone="CSPDarknet-nano" num_epochs=300 exp_id="coco_CSPDarknet-nano_416x416" use_amp=True data_num_workers=6 batch_size=128
#python train.py gpus='0' backbone="CSPDarknet-tiny" num_epochs=300 exp_id="coco_CSPDarknet-tiny_416x416" use_amp=True data_num_workers=6 batch_size=128
python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True data_num_workers=6 batch_size=52
#python train.py gpus='0,1' backbone="CSPDarknet-m" num_epochs=300 exp_id="coco_CSPDarknet-m_640x640" use_amp=True data_num_workers=6 batch_size=64
#python train.py gpus='0,1,2' backbone="CSPDarknet-l" num_epochs=300 exp_id="coco_CSPDarknet-l_640x640" use_amp=True data_num_workers=6 batch_size=64
#python train.py gpus='0,1,2' backbone="CSPDarknet-x" num_epochs=300 exp_id="coco_CSPDarknet-x_640x640" use_amp=True data_num_workers=6 batch_size=64

# disable multi-size
#python train.py gpus='0' backbone="CSPDarknet-tiny" num_epochs=300 exp_id="coco_CSPDarknet-tiny_416x416" use_amp=True data_num_workers=6 batch_size=128 input_size="(416,416)" random_size=None

# fine-tune, load pre-trained weight
#python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True data_num_workers=6 batch_size=48 load_model="../weights/yolox-s.pth" resume=False

# resume 'model_best.pth/model_num.pth', include weight and epoch
#python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True data_num_workers=6 batch_size=48 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" resume=True

# resume 'model_last.pth', include weight, optimizer, scaler and epoch
#python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True data_num_workers=6 batch_size=48 load_model="exp/coco_CSPDarknet-s_640x640/model_last.pth" resume=True

# GPU memory changes with the input size when multi-size training, which can be avoided by pre allocating memory
#python train.py gpus='0' backbone="CSPDarknet-tiny" num_epochs=300 exp_id="coco_CSPDarknet-tiny_416x416" use_amp=True data_num_workers=6 batch_size=128 occupy_mem=True
