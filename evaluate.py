# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 21:50
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import cv2
import tqdm
import json
import pycocotools.coco as coco_
from pycocotools.cocoeval import COCOeval

from config import opt
from utils.util import NpEncoder
from models.yolox import Detector


def evaluate():
    detector = Detector(opt)
    gt_ann = opt.val_ann
    img_dir = opt.dataset_path + "/images/val2017"

    assert os.path.isfile(gt_ann), 'cannot find gt {}'.format(gt_ann)
    coco = coco_.COCO(gt_ann)
    images = coco.getImgIds()
    class_ids = sorted(coco.getCatIds())
    num_samples = len(images)
    print('find {} samples in {}'.format(num_samples, gt_ann))
    result_file = "result.json"
    coco_res = []
    for index in tqdm.tqdm(range(num_samples)):
        img_id = images[index]
        file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
        image_path = img_dir + "/" + file_name
        img = cv2.imread(image_path)

        results = detector.run(img, vis_thresh=0.01)
        for res in results:
            cls, conf, bbox = res[0], res[1], res[2]
            if len(res) > 3:
                reid_feat = res[4]
            cls_index = opt.label_name.index(cls)
            coco_res.append(
                {'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                 'category_id': class_ids[cls_index],
                 'image_id': int(img_id),
                 'score': conf})

    with open(result_file, 'w') as f_dump:
        json.dump(coco_res, f_dump, cls=NpEncoder)

    coco_det = coco.loadRes(result_file)
    coco_eval = COCOeval(coco, coco_det, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    os.remove(result_file)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    evaluate()
