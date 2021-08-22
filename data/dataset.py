# -*- coding: utf-8 -*-
# @Time    : 21-8-15 :05
# @Author  : MingZhang
# @Email   : zm19921120@126.com


import io
import os
import cv2
import json
import random
import contextlib
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data.data_augment import random_perspective, box_candidates, TrainTransform


class COCODataset(torch.utils.data.Dataset):

    def __init__(self, cfg, img_size=(640, 640), name="train2017", json_file="instances_train2017.json", preproc=None,
                 no_aug=True, tracking=False, logger=None):

        super(COCODataset, self).__init__()
        self.opt = cfg
        self.img_size = img_size
        self.name = name
        self.json_file = json_file
        self.preproc = preproc
        self.augment = not no_aug
        self.tracking = tracking
        self.logger = logger
        self.data_dir = self.opt.data_dir
        self.batch_size = self.opt.batch_size

        # data augment params
        self.random_size = self.opt.random_size
        self.degrees = self.opt.degrees
        self.translate = self.opt.translate
        self.scale = self.opt.scale
        self.shear = self.opt.shear
        self.perspective = self.opt.perspective
        self.mixup_scale = (0.5, 1.5)
        self.enable_mosaic = self.augment
        self.enable_mixup = self.opt.enable_mixup
        self.mosaic_prob = self.opt.mosaic_prob
        self.mixup_prob = self.opt.mixup_prob

        #################
        # self.json_file = self.json_file.replace("train", "val")
        # self.name = self.name.replace("train", "val")
        #################
        assert os.path.isfile(self.json_file), 'cannot find {}'.format(self.json_file)
        print("==> Loading {} annotation {}".format(self.name, self.json_file))
        self.coco = COCO(self.json_file)
        self.ids = self.coco.getImgIds()
        self.num_samples = len(self.ids)
        print("images number {}".format(self.num_samples))
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [c["name"] for c in cats]
        self.annotations = self._load_coco_annotations()
        self.samples_shapes = [self.img_size for _ in range(self.num_samples)]

        if 'val2017' == self.name:
            print("classes index:", self.class_ids)
            print("class names in dataset:", self.classes)

    def __len__(self):
        return self.num_samples

    def shuffle(self):
        np.random.shuffle(self.annotations)
        print("shuffle images list in {}".format(self.json_file))
        if self.logger:
            self.logger.write("shuffle {} images list...\n".format(self.json_file))

        if self.random_size is not None:
            self.samples_shapes = self.multi_shape()

    def multi_shape(self):
        size_factor = self.img_size[1] * 1. / self.img_size[0]

        multi_shapes = []
        for size in list(range(*self.random_size)):
            random_input_h, random_input_w = (int(32 * size), 32 * int(size * size_factor))
            multi_shapes.append([random_input_h, random_input_w])
        print("multi size training: {}".format(multi_shapes))
        if self.logger:
            self.logger.write("multi size training: {}\n".format(multi_shapes))

        iter_num = int(np.ceil(self.num_samples / self.batch_size))
        samples_shapes = []
        rand_idx = len(multi_shapes) - 1  # initialize with max size, in case of out of memory during training
        for it in range(iter_num):
            if it != 0 and it % 10 == 0:
                rand_idx = np.random.choice(list(range(len(multi_shapes))))
            for _ in range(self.batch_size):
                samples_shapes.append(multi_shapes[rand_idx])
        return samples_shapes

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

        return res, img_info, file_name, id_

    def pull_item(self, index):
        res, img_info, file_name, id_ = self.annotations[index]
        # load image and preprocess
        img_file = self.data_dir + "/" + self.name + "/" + file_name
        img = cv2.imread(img_file)
        assert img is not None, "error img {}".format(img_file)

        return img, res.copy(), img_info, id_

    def close_random_size(self):
        self.samples_shapes = [self.img_size for _ in range(self.num_samples)]
        print("close multi-size training")

    def __getitem__(self, idx):

        if self.enable_mosaic and self.augment and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_h, input_w = self.samples_shapes[idx]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, self.num_samples - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, _ = self.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove

            if self.enable_mixup and not len(mosaic_labels) == 0 and random.random() < self.mixup_prob:
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.samples_shapes[idx])
            img_info = (mosaic_img.shape[1], mosaic_img.shape[0])
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.samples_shapes[idx])
            return mix_img, padded_labels, img_info, -1
        else:
            img, label, img_info, img_id = self.pull_item(idx)
            img, label = self.preproc(img, label, self.samples_shapes[idx])
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_index = random.randint(0, self.__len__() - 1)
        img, cp_labels, _, _ = self.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            if self.tracking:
                tracking_id_labels = cp_labels[keep_list, 5:6].copy()
                labels = np.hstack((box_labels, cls_labels, tracking_id_labels))
            else:
                labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


def get_dataloader(opt, no_aug=False, logger=None, val_loader=True):
    do_tracking = opt.reid_dim > 0
    # train
    train_dataset = COCODataset(opt,
                                img_size=opt.input_size,
                                name='train2017',
                                json_file=opt.train_ann,
                                preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120,
                                                       tracking=do_tracking, augment=True),
                                no_aug=no_aug,
                                tracking=do_tracking,
                                logger=logger,
                                )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # do shuffle in dataset
        num_workers=opt.data_num_workers,
        pin_memory=True,
        drop_last=True
    )

    if not val_loader:
        return train_loader, None

    # val
    val_dataset = COCODataset(opt,
                              img_size=opt.test_size,
                              name='val2017',
                              json_file=opt.val_ann,
                              preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120,
                                                     tracking=do_tracking, augment=False),
                              no_aug=True,
                              tracking=do_tracking,
                              logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.data_num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader


def vis_inputs(inputs, targets, opt):
    from utils.util import label_color

    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    for b_i, inp in enumerate(inputs):
        target = targets[b_i]
        img = (((inp.transpose((1, 2, 0)) * opt.std) + opt.rgb_means) * 255).astype(np.uint8)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        gt_n = 0
        for t in target:
            if t.sum() > 0:
                if len(t) == 5:
                    cls, c_x, c_y, w, h = [int(i) for i in t]
                    tracking_id = None
                elif len(t) == 6:
                    cls, c_x, c_y, w, h, tracking_id = [int(i) for i in t]
                else:
                    raise ValueError("target shape != 5 or 6")
                bbox = [c_x - w // 2, c_y - h // 2, c_x + w // 2, c_y + h // 2]
                label = opt.label_name[cls]
                # print(label, bbox)
                color = label_color[cls]
                # show box
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                # show label and conf
                txt = '{}-{}'.format(label, tracking_id) if tracking_id is not None else '{}'.format(label)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color,
                              -1)
                cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)
                gt_n += 1

        print("img {}/{} gt number: {}".format(b_i, len(inputs), gt_n))
        cv2.namedWindow("input", 0)
        cv2.imshow("input", img)
        key = cv2.waitKey(0)

        if key == 27:
            exit()


def run_epoch(train_loader, e, opt):
    for batch_i, batch in enumerate(train_loader):
        inps, targets, img_info, ind = batch
        print("------------ epoch {} batch {}/{} ---------------".format(e, batch_i, len(train_loader)))
        print("batch img shape {}, target shape {}".format(inps.shape, targets.shape))
        if opt.show:
            vis_inputs(inps, targets, opt)
        if batch_i >= 21:
            break


def main():
    from config import opt

    # opt.dataset_path = r"D:\work\public_dataset\coco2017"
    opt.dataset_path = "/media/ming/DATA1/dataset/coco2017"
    opt.train_ann = opt.dataset_path + "/annotations/instances_train2017.json"
    opt.val_ann = opt.dataset_path + "/annotations/instances_val2017.json"
    opt.data_dir = opt.dataset_path + "/images"

    opt.input_size = (640, 640)
    opt.test_size = (640, 640)
    opt.batch_size = 2
    opt.data_num_workers = 2  # 0
    opt.reid_dim = 0  # 128
    opt.show = True  # False

    train_loader, val_loader = get_dataloader(opt, no_aug=False)

    # train_loader = val_loader
    dataset_label = train_loader.dataset.classes
    assert opt.label_name == dataset_label, "your class_name != dataset's {} {}".format(opt.label_name, dataset_label)
    for e in range(100):
        train_loader.dataset.shuffle()
        if e == 2:
            train_loader.dataset.enable_mosaic = False
        run_epoch(train_loader, e, opt)


if __name__ == "__main__":
    main()
