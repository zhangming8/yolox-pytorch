## a Pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021"

## 1. Notes
    1. this is a Pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021"
    2. the repo is still under development

## 2. Environment
    pytorch>=1.7.0, python>=3.6, Ubuntu/Windows, see more in 'requirements.txt'

## 3. Object Detection
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
    sh train.sh
    
#### Evaluate
    sh evaluate.sh
    
#### Predict/Inference/Demo
    sh predict.sh

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
    1. download and unzip VisDrone dataset http://aiskyeye.com/download/multi-object-tracking_2021
    
    2. put train and val dataset into:
    /path/to/dataset/VisDrone/VisDrone2019-MOT-train  # This folder contains two subfolders, 'annotations' and 'sequences'
    /path/to/dataset/VisDrone/VisDrone2019-MOT-val  # This folder contains two subfolders, 'annotations' and 'sequences'
    
    3. change opt.dataset_path = "/path/to/dataset/VisDrone" in 'config.py'
    4. python tools/visdrone_mot_to_coco.py  # converted to COCO format
    5.(Optional) python tools/show_coco_anns.py  # visualized tracking id
    
    6. set class name and tracking id number
    change opt.label_name=['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    change opt.tracking_id_nums=[1829, 853, 323, 3017, 295, 159, 215, 79, 55, 749]
    change opt.reid_dim=128

#### Train
    DOING

#### Evaluate
    DOING

#### Predict/Inference/Demo
    DOING

## 5. Reference
    https://github.com/Megvii-BaseDetection/YOLOX
    https://github.com/PaddlePaddle/PaddleDetection
    https://github.com/open-mmlab/mmdetection
