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


def vis_result(img, results):
    for res_i, res in enumerate(results):
        label, conf, bbox = res[:3]
        bbox = [int(i) for i in bbox]
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
    return img


def detect_video():
    detector = Detector(opt)
    video_dir = opt.video_dir
    save_folder = "output_video"

    assert os.path.isfile(video_dir), "cannot find {}".format(video_dir)
    cap = cv2.VideoCapture(video_dir)
    # cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(7))
    mkdir(save_folder)
    save_path = os.path.join(save_folder, os.path.basename(video_dir))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    idx = 1
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            print("detect frame {}/{}".format(idx, frame_num))
            results = detector.run(frame, vis_thresh=opt.vis_thresh, show_time=True)
            print(results)
            frame = vis_result(frame, results)
            vid_writer.write(frame)
            idx += 1
        else:
            break
    vid_writer.release()
    print("save video to {}".format(save_path))


def detect():
    img_dir = opt.dataset_path + "/images/val2017" if "img_dir" not in opt else opt["img_dir"]
    output = "output"
    mkdir(output, rm=True)

    img_list = get_img_path(img_dir, extend=".jpg")
    assert len(img_list) != 0, "cannot find img in {}".format(img_dir)

    detector = Detector(opt)
    for index, image_path in enumerate(tqdm.tqdm(img_list)):
        print("------------------------------")
        print("{}/{}, {}".format(index, len(img_list), image_path))

        assert os.path.isfile(image_path), "cannot find {}".format(image_path)
        img = cv2.imread(image_path)
        s1 = time.time()
        results = detector.run(img, vis_thresh=opt.vis_thresh, show_time=True)
        print("[pre_process + inference + post_process] time cost: {}s".format(time.time() - s1))
        print(results)
        img = vis_result(img, results)
        save_p = output + "/" + image_path.split("/")[-2]
        mkdir(save_p)
        save_img = save_p + "/" + os.path.basename(image_path)
        cv2.imwrite(save_img, img)
        print("save image to {}".format(save_img))


if __name__ == "__main__":
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    if 'video_dir' not in opt.keys():
        detect()
    else:
        detect_video()
