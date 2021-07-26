## a Pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021"

## Notes
    1. this is a Pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021"
    2. the repo is still under development
    3. we needn't install apex, Pytorch(version >= 1.7.0) has supported it

## Environment
    pytorch>=1.7.0, python>=3.6, Ubuntu/Windows, see more in 'requirements.txt'

## Dataset
    put COCO dataset in following folders:

    /path/to/dataset/annotations/instances_train2017.json
    /path/to/dataset/annotations/instances_val2017.json
    /path/to/dataset/images/train2017/*.jpg
    /path/to/dataset/images/val2017/*.jpg
    
    change opt.dataset_path = "/path/to/dataset" in 'config.py'

## Train
    sh train.sh
    
## Evaluate
    sh evaluate.sh
    
## Predict/Inference/Demo
    sh predict.sh

## Train Customer Dataset(VOC format)
    
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
    
    5. run train.sh, evaluate.sh, predict.sh (are the same as COCO)

## Reference
    https://github.com/Megvii-BaseDetection/YOLOX
    https://github.com/PaddlePaddle/PaddleDetection
    https://github.com/open-mmlab/mmdetection
