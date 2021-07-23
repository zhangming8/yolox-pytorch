# a Pytorch easy re-implement of "YOLOX: Exceeding YOLO Series in 2021"


# environment
    pytorch>=1.7.0, python>=3.6

# dataset
    put your COCO dataset in fllowing folder:
    
    /path/to/dataset/annotations/instances_train2017.json
    /path/to/dataset/annotations/instances_val2017.json
    /path/to/dataset/images/train2017/*.jpg
    /path/to/dataset/images/val2017/*.jpg
    
    modify 'config.py'
    opt.dataset_path = "/path/to/dataset"

# train
    sh train.sh
    
# evaluate
    sh evaluate.sh
    
# predict/inference/demo
    sh predict.sh
   

# reference
    https://github.com/Megvii-BaseDetection/YOLOX
    https://github.com/PaddlePaddle/PaddleDetection
    https://github.com/open-mmlab/mmdetection
