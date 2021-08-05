# -*- coding: utf-8 -*-
# @Time    : 2021/08/05 06:16
# @Author  : Daniel Ma (znsoft)
# @Email   : znsoft@163.com

import os
import sys
sys.path.append("../../..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../../..") for name in dirs])
from config import opt
from models.yolox import load_model,get_model
import torch 
import onnx 


def convert(srcModel,strDstFile):
    
    inp_imgs = torch.randn(1, 3, 640, 640, device='cpu')
    model = get_model(opt)
    model= load_model(model,srcModel)
    torch.onnx._export(model, inp_imgs, strDstFile, verbose=True, do_constant_folding=True, opset_version=12 )
    onnxmodel = onnx.load(strDstFile)
    onnx.checker.check_model(onnxmodel)
    onnx.helper.printable_graph(onnxmodel.graph)

if __name__ == "__main__":
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")
    modelpath=opt.load_model.replace(".pth",".onnx")
    convert(opt.load_model,modelpath)
    print("=========done, onnx path:{}============",modelpath)
