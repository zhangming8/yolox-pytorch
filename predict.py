# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 19:59
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import cv2
import tqdm
import time

from config import opt
from models.yolox import Detector
from utils.util import mkdir, label_color, get_img_path


def detect():
    img_dir = "/data/dataset/coco_dataset/images/val2017" if "img_dir" not in opt else opt["img_dir"]
    output = "output"
    mkdir(output, rm=True)

    img_list = get_img_path(img_dir, extend=".jpg")
    assert len(img_list) != 0, "cannot find img in {}".format(img_dir)

    detector = Detector(opt)
    for index, image_path in enumerate(tqdm.tqdm(img_list)):
        print("------------------------------")
        print("{}/{}, {}".format(index, len(img_list), image_path))

        img = cv2.imread(image_path)
        s1 = time.time()
        results = detector.run(img, vis_thresh=opt.vis_thresh, show_time=True)
        print("[pre_process + inference + post_process] time cost: {}s".format(time.time() - s1))
        print(results)

        for res_i, res in enumerate(results):
            label, conf, bbox = res[:3]
            if len(res) > 3:
                reid_feat = res[4]
                print("reid feat dim {}".format(len(reid_feat)))

            color = label_color[opt.label_name.index(label)]
            # show box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # show label and conf
            txt = '{}:{:.2f}'.format(label, conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color, -1)
            cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        save_p = output + "/" + image_path.split("/")[-2]
        mkdir(save_p)
        cv2.imwrite(save_p + "/" + os.path.basename(image_path), img)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    detect()
