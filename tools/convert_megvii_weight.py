# -*- coding: utf-8 -*-
# @Time    : 2021/7/30 19:55
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import sys
import torch

sys.path.append(".")
from config import opt
from models.yolox import get_model


def convert_weight(torch_model, megii_weight, save_weight):
    torch_model_dict = torch_model.state_dict()
    megii_params = torch.load(megii_weight, map_location=lambda storage, loc: storage)

    print(megii_params.keys())
    print("epoch:", megii_params["start_epoch"])
    for k, v in megii_params['model'].items():
        # print(k, v.shape)
        if k[:len("backbone.backbone.")] == "backbone.backbone.":
            new_k = k.replace("backbone.backbone.", "backbone.")
        elif k[:len("backbone.")] == "backbone.":
            new_k = k.replace("backbone.", "neck.")
        else:
            new_k = k
        # print("{} -> {}".format(k, new_k))
        assert new_k in torch_model_dict.keys(), "error key {}".format(new_k)
        assert v.shape == torch_model_dict[new_k].shape, "shape not match {} vs {}".format(v.shape, torch_model_dict[
            new_k].shape)
        torch_model_dict[new_k] = v

    torch_model.load_state_dict(torch_model_dict, strict=True)
    data = {'epoch': megii_params['start_epoch'], 'state_dict': torch_model.state_dict()}
    torch.save(data, save_weight)
    print("==>> save converted model to {}".format(save_weight))


def main():
    megii_weight_dir = r"D:\work\hc\weights"
    weights = ['yolox_s.pth.tar', 'yolox_m.pth.tar', 'yolox_l.pth.tar', 'yolox_x.pth.tar', 'yolox_tiny.pth.tar',
               'yolox_nano.pth.tar']

    for weight in weights:
        print("converting {}".format(weight))
        model_type = weight.split("_")[-1].split(".")[0]
        print("model type: {}".format(model_type))
        megii_weight = megii_weight_dir + "/" + weight

        save_converted_weight = megii_weight_dir + "/yolox-{}.pth".format(model_type)
        opt.backbone = "CSPDarknet-" + model_type
        if 'nano' in opt.backbone:
            opt.depth_wise = True
        else:
            opt.depth_wise = False

        model = get_model(opt)
        model.eval()
        convert_weight(model, megii_weight, save_converted_weight)
        del model
    print("done")


if __name__ == "__main__":
    main()
