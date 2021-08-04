rem !/usr/bin/env bash
rem  converting the best model to an onnx model
python pth2onnx.py  backbone="CSPDarknet-s" load_model="../../../exp/coco_CSPDarknet-s_640x640/model_best.pth" 
python pth2onnx.py  backbone="CSPDarknet-nano" load_model="../../../exp/coco_CSPDarknet-nano_640x640/model_best.pth" 

