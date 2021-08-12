#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import io
import os
import cv2
import json
import contextlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .datasets_wrapper import Dataset


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(self,
                 data_dir=None,
                 json_file="instances_train2017.json",
                 name="train2017",
                 img_size=(416, 416),
                 tracking=False,
                 preproc=None,
                 ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (tuple): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.tracking = tracking
        #################
        # self.name = "val2017"
        # self.json_file = self.json_file.replace("train", "val")
        #################
        assert os.path.isfile(json_file), 'cannot find {}'.format(json_file)
        print("==> Loading annotation {}".format(json_file))
        self.coco = COCO(self.json_file)
        self.ids = self.coco.getImgIds()
        print("images number {}".format(len(self.ids)))
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [c["name"] for c in cats]
        self.annotations = self._load_coco_annotations()

        if "val" in self.name:
            print("classes index:", self.class_ids)
            print("class names in dataset:", self.classes)

    def __len__(self):
        return len(self.ids)

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes.keys():
            one_img_res = all_bboxes[image_id]
            for res in one_img_res:
                cls, conf, bbox = res[0], res[1], res[2]
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    'category_id': self.class_ids[self.classes.index(cls)],
                    'image_id': int(image_id),
                    'score': float(conf)})
        return detections

    def run_coco_eval(self, results, save_dir):
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'))
        coco_det = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        str_result = redirect_string.getvalue()
        ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large = coco_eval.stats[:6]
        print(str_result)
        return ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, str_result

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6 if self.tracking else 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if self.tracking:
                assert "tracking_id" in obj.keys(), 'cannot find "tracking_id" in your dataset'
                res[ix, 5] = obj['tracking_id']
                # print('errorrrrrrrr: replace tracking_id to cls')
                # res[ix, 5] = cls

        img_info = (height, width)
        file_name = im_ann["file_name"]

        del im_ann, annotations

        return res, img_info, file_name

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = self.data_dir + "/" + self.name + "/" + file_name
        img = cv2.imread(img_file)
        assert img is not None, "error img {}".format(img_file)

        return img, res.copy(), img_info, id_

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
