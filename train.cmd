rem starting script for Windows.
rem  train from scratch
rem python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=8

rem load pre-train weight
python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=4 load_model="f:/weights/yolox-s.pth" resume=False

rem  resume 'model_best.pth / model_num.pth', includes weight and epoch
rem python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=8 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" resume=True

rem  resume 'model_last.pth', includes weight, optimizer, scaler and epoch
rem python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=False val_intervals=1 data_num_workers=8 load_model="exp/coco_CSPDarknet-s_640x640/model_last.pth" resume=True
