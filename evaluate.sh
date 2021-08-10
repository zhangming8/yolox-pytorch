#!/usr/bin/env bash

# reproduce the results on COCO val2017 and test2017:
# evaluate COCO val2017
#python evaluate.py gpus='0' backbone="CSPDarknet-nano" load_model="../weights/yolox-nano.pth" dataset_path="/data/dataset/coco_dataset"
#python evaluate.py gpus='0' backbone="CSPDarknet-tiny" load_model="../weights/yolox-tiny.pth" dataset_path="/data/dataset/coco_dataset"
#python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="../weights/yolox-s.pth" dataset_path="/data/dataset/coco_dataset"
#python evaluate.py gpus='0' backbone="CSPDarknet-m" load_model="../weights/yolox-m.pth" dataset_path="/data/dataset/coco_dataset"
#python evaluate.py gpus='0' backbone="CSPDarknet-l" load_model="../weights/yolox-l.pth" dataset_path="/data/dataset/coco_dataset"
#python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset"
#python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset" test_size="(800,800)"
# evaluate COCO test2017, create upload file
#python evaluate.py gpus='0' backbone="CSPDarknet-nano" load_model="../weights/yolox-nano.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
#python evaluate.py gpus='0' backbone="CSPDarknet-tiny" load_model="../weights/yolox-tiny.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
#python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="../weights/yolox-s.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
#python evaluate.py gpus='0' backbone="CSPDarknet-m" load_model="../weights/yolox-m.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
#python evaluate.py gpus='0' backbone="CSPDarknet-l" load_model="../weights/yolox-l.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
#python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
#python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json" test_size="(800,800)"

# evaluate customer dataset
python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" batch_size=24

# fuse BN into Conv to speed up
#python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" batch_size=24 fuse=True
