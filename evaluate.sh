#!/usr/bin/env bash

python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" exp_id="coco_CSPDarknet-s_640x640" dataset_path="/data/dataset/coco_dataset"
