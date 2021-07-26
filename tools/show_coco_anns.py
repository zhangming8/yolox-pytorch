# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 17:50
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import cv2
import sys
import pycocotools.coco as coco_

sys.path.append(".")
from config import opt
from utils.util import label_color


def x1y1wh_x1y1x2y2(box):
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]


def vis_coco_anns(data):
    gt_ann = opt.dataset_path + "/annotations/instances_{}.json".format(data)
    img_dir = opt.dataset_path + "/images/" + data

    assert os.path.isfile(gt_ann), 'cannot find gt {}'.format(gt_ann)
    coco = coco_.COCO(gt_ann)
    images = coco.getImgIds()
    class_ids = coco.getCatIds()
    cats = coco.loadCats(class_ids)
    classes = [c["name"] for c in cats]
    num_samples = len(images)

    print('find {} samples in {}'.format(num_samples, gt_ann))
    print("class_ids:", class_ids)
    print("classes:", classes)

    for index in range(num_samples):
        img_id = images[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ids=ann_ids)
        file_name = coco.loadImgs(ids=[img_id])[0]['file_name']

        image_path = img_dir + "/" + file_name
        assert os.path.isfile(image_path), 'cannot find {}'.format(image_path)
        img = cv2.imread(image_path)

        for ann in anns:
            area = ann["area"]
            is_crowd = ann["iscrowd"]
            bbox = ann["bbox"]
            bbox = x1y1wh_x1y1x2y2(bbox)
            category_id = ann["category_id"]
            label_index = class_ids.index(category_id)
            label = classes[label_index]
            # print("label {}, is_crowd {}".format(label, is_crowd))

            # draw bounding box
            color = label_color[label_index]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(img, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX, 1, color)
            cv2.circle(img, ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2), 4, color, -1)

        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == 27:
            exit()


if __name__ == "__main__":
    vis_coco_anns('train2017')
    # vis_coco_anns('val2017')
