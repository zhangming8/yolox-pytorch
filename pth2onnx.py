# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 21:50
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import cv2
import pycocotools.coco as coco_
from config import opt
from models.yolox import load_model,get_model
import pathlib as Path
import torch 
import onnx 


def convert(srcModel,strDstFile):
    
  
    inp_imgs = torch.randn(1, 3, 640, 640, device='cpu')

    model = get_model(opt)
    model= load_model(model,srcModel)
    #model(inp_imgs, vis_thresh=0.001, ratio=1, show_time=False)
    torch.onnx._export(model, inp_imgs, strDstFile, verbose=True, opset_version=12 )
    onnxmodel = onnx.load(strDstFile)
    onnx.checker.check_model(onnxmodel)
    onnx.helper.printable_graph(onnxmodel.graph)

if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")
    modelpath=opt.load_model.replace(".pth",".onnx")
    convert(opt.load_model,modelpath)
    print("=========done, onnx path:{}============",modelpath)
