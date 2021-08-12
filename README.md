## A pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021"

## 1. Notes

    This is a pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021" [https://arxiv.org/abs/2107.08430]
    The repo is still under development

## 2. Environment

    pytorch>=1.7.0, python>=3.6, Ubuntu/Windows, see more in 'requirements.txt'
    
    cd /path/to/your/work
    git clone https://github.com/zhangming8/yolox-pytorch.git
    cd yolox-pytorch
    download pre-train weights in Model Zoo to /path/to/your/work/weights

## 3. Object Detection

#### Model Zoo

All weights can be downloaded
from [GoogleDrive](https://drive.google.com/drive/folders/1qEMLzikH5JwRNRoHpeCa6BJBeSQ6xXCH?usp=sharing)
or [BaiduDrive](https://pan.baidu.com/s/1UsbdnyVwRJhr9Vy1tmJLeQ) (code:bc72)

|Model      |test size  |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Params<br>(M) |
| ------    |:---:      |:---:                   | :---:                   |:---:          |
|yolox-nano |416        |25.4                    |25.7                     |0.91           |
|yolox-tiny |416        |33.1                    |33.2                     |5.06           |
|yolox-s    |640        |39.3                    |39.6                     |9.0            |
|yolox-m    |640        |46.2                    |46.4                     |25.3           |
|yolox-l    |640        |49.5                    |50.0                     |54.2           |
|yolox-x    |640        |50.5                    |51.1                     |99.1           |
|yolox-x    |800        |51.2                    |51.9                     |99.1           |

mAP was reevaluated on COCO val2017 and test2017, and some results are slightly better than the official
implement [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). You can reproduce them by scripts in 'evaluate.sh'

#### Dataset

    download COCO:
    http://images.cocodataset.org/zips/train2017.zip
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    
    unzip and put COCO dataset in following folders:
    /path/to/dataset/annotations/instances_train2017.json
    /path/to/dataset/annotations/instances_val2017.json
    /path/to/dataset/images/train2017/*.jpg
    /path/to/dataset/images/val2017/*.jpg
    
    change opt.dataset_path = "/path/to/dataset" in 'config.py'

#### Train

    See more example in 'train.sh'
    a. Train from scratch:(backbone="CSPDarknet-s" means using yolox-s, and you can change it, eg: CSPDarknet-nano, tiny, s, m, l, x)
    python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True val_intervals=2 data_num_workers=6 batch_size=48
    
    b. Finetune, download pre-trained weight on COCO and finetune on customer dataset:
    python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True val_intervals=2 data_num_workers=6 batch_size=48 load_model="../weights/yolox-s.pth"
    
    c. Resume, you can use 'resume=True' when your training is accidentally stopped:
    python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True val_intervals=2 data_num_workers=6 batch_size=48 load_model="exp/coco_CSPDarknet-s_640x640/model_last.pth" resume=True

#### Some Tips:

    a. You can also change params in 'train.sh'(these params will replace opt.xxx in config.py) and use 'nohup sh train.sh &' to train
    b. Multi-gpu train: set opt.gpus = "3,5,6,7" in 'config.py' or set gpus="3,5,6,7" in 'train.sh'
    c. If you want to close multi-size training, change opt.random_size = None in 'config.py' or set random_size=None in 'train.sh'
    d. random_size = (14, 26) means: Randomly select an integer from interval (14,26) and multiply by 32 as the input size
    e. Visualized log by tensorboard: 
        tensorboard --logdir exp/your_exp_id/logs_2021-08-xx-xx-xx and visit http://localhost:6006
       Your can also use the following shell scripts:
        (1) grep 'train epoch' exp/your_exp_id/logs_2021-08-xx-xx-xx/log.txt
        (2) grep 'val epoch' exp/your_exp_id/logs_2021-08-xx-xx-xx/log.txt

#### Evaluate

    Module weights will be saved in './exp/your_exp_id/model_xx.pth'
    change 'load_model'='weight/path/to/evaluate.pth' and backbone='backbone-type' in 'evaluate.sh'
    sh evaluate.sh

#### Predict/Inference/Demo

    a. Predict images, change img_dir and load_model
    python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" img_dir='/path/to/dataset/images/val2017'
    
    b. Predict video
    python predict.py gpus='0' backbone="CSPDarknet-s" vis_thresh=0.3 load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" video_dir='/path/to/your/video.mp4'
    
    You can also change params in 'predict.sh', and use 'sh predict.sh'

#### Train Customer Dataset(VOC format)

    1. put your annotations(.xml) and images(.jpg) into:
        /path/to/voc_data/images/train2017/*.jpg  # train images
        /path/to/voc_data/images/train2017/*.xml  # train xml annotations
        /path/to/voc_data/images/val2017/*.jpg  # val images
        /path/to/voc_data/images/val2017/*.xml  # val xml annotations

    2. change opt.label_name = ['your', 'dataset', 'label'] in 'config.py'
       change opt.dataset_path = '/path/to/voc_data' in 'config.py'

    3. python tools/voc_to_coco.py
       Converted COCO format annotation will be saved into:
        /path/to/voc_data/annotations/instances_train2017.json
        /path/to/voc_data/annotations/instances_val2017.json
    
    4. (Optional) you can visualize the converted annotations by:
        python tools/show_coco_anns.py
        Here is an analysis of the COCO annotation https://blog.csdn.net/u010397980/article/details/90341223?spm=1001.2014.3001.5501
    
    5. run train.sh, evaluate.sh, predict.sh (are the same as COCO)

## 4. Multi/One-class Multi-object Tracking(MOT)

#### one-class/single-class MOT Dataset

    DOING

#### Multi-class MOT Dataset

    DOING

#### Train

    DOING

#### Evaluate

    DOING

#### Predict/Inference/Demo

    DOING

## 5. Acknowledgement

    https://github.com/Megvii-BaseDetection/YOLOX
    https://github.com/PaddlePaddle/PaddleDetection
    https://github.com/open-mmlab/mmdetection
    https://github.com/xingyizhou/CenterNet
