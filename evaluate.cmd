rem !/usr/bin/env bash

rem  reproduce the results on COCO val2017 and test2017:
rem  evaluate COCO val2017
rem python evaluate.py gpus='0' backbone="CSPDarknet-nano" load_model="../weights/yolox-nano.pth" dataset_path="/data/dataset/coco_dataset"
rem python evaluate.py gpus='0' backbone="CSPDarknet-tiny" load_model="../weights/yolox-tiny.pth" dataset_path="/data/dataset/coco_dataset"
rem python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="../weights/yolox-s.pth" dataset_path="/data/dataset/coco_dataset"
rem python evaluate.py gpus='0' backbone="CSPDarknet-m" load_model="../weights/yolox-m.pth" dataset_path="/data/dataset/coco_dataset"
rem python evaluate.py gpus='0' backbone="CSPDarknet-l" load_model="../weights/yolox-l.pth" dataset_path="/data/dataset/coco_dataset"
rem python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset"
rem python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset" test_size="(800,800)"
rem  evaluate COCO test2017, create upload file
rem python evaluate.py gpus='0' backbone="CSPDarknet-nano" load_model="../weights/yolox-nano.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
rem python evaluate.py gpus='0' backbone="CSPDarknet-tiny" load_model="../weights/yolox-tiny.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
rem python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="../weights/yolox-s.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
rem python evaluate.py gpus='0' backbone="CSPDarknet-m" load_model="../weights/yolox-m.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
rem python evaluate.py gpus='0' backbone="CSPDarknet-l" load_model="../weights/yolox-l.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
rem python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json"
rem python evaluate.py gpus='0' backbone="CSPDarknet-x" load_model="../weights/yolox-x.pth" dataset_path="/data/dataset/coco_dataset" test_ann="/data/dataset/coco_dataset/annotations/image_info_test-dev2017.json" test_size="(800,800)"

rem  evaluate customer dataset
python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth"
