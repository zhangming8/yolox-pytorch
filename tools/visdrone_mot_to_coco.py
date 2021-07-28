# -*- coding: utf-8 -*-
# @Time    : 21-7-25 10:15
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import glob
import os
import cv2
import sys
import tqdm
import json
import shutil

sys.path.append('.')
from config import opt

label_map = {'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'tricycle': 7,
             'awning-tricycle': 8, 'bus': 9, 'motor': 10}
label_map_reverse = {v: k for k, v in label_map.items()}


def mkdir(path, rm=False):
    if os.path.exists(path):
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def get_img_path(path, extend=".jpg"):
    img_list = []
    for fpath, dirs, fs in os.walk(path):
        for f in fs:
            img_path = os.path.join(fpath, f)
            if os.path.dirname(img_path) == os.getcwd():
                continue
            if not os.path.isfile(img_path):
                continue
            file_name, file_extend = os.path.splitext(os.path.basename(img_path))
            if file_extend == extend:
                img_list.append(img_path)
    return img_list


def read_ann(txt, img_dir, img_folder, total_anns, data_idx):
    data_idx += 1
    with open(txt, "r") as f:
        for line in f.readlines():
            # <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            line = line.strip().split(",")
            frame_idx, tracking_id, x1, y1, w, h, score, object_category, truncation, occlusion = [int(i) for i in line]
            score = float(line[6])
            # object_category: ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11))

            relative_path = img_folder + "/" + '{0:07d}.jpg'.format(frame_idx)
            img_p = img_dir + "/" + relative_path
            assert os.path.isfile(img_p), "cannot find {}".format(img_p)

            if object_category not in label_map_reverse.keys():
                skip_label = {0: "ignored regions", 11: "others"}
                # print("skip label {} {}".format(object_category, skip_label[object_category]))
                continue

            if relative_path not in total_anns.keys():
                total_anns[relative_path] = []

            assert tracking_id < 10000000
            total_anns[relative_path].append(
                [label_map_reverse[object_category], 10000000 * data_idx + tracking_id, [x1, y1, w, h]])


def convert(data_path, data):
    print("start convert {}".format(data_path))
    annotation_path = data_path + "/annotations"
    image_path = data_path + "/sequences"
    assert os.path.exists(annotation_path), 'cannot find path {}'.format(annotation_path)
    assert os.path.exists(image_path), 'cannot find path {}'.format(image_path)

    save_anns = dataset_path + "/annotations/instances_{}.json".format(data)
    save_images = dataset_path + "/images/" + data
    if os.path.exists(save_images):
        shutil.rmtree(save_images)

    total_anns = {}
    ann_list = glob.glob(annotation_path + "/*.txt")
    for idx, ann_f in enumerate(ann_list):
        # print("reading annotation {}".format(ann_f))
        folder_name = os.path.basename(ann_f).replace(".txt", "")
        assert os.path.exists(image_path + "/" + folder_name)
        read_ann(ann_f, image_path, folder_name, total_anns, idx)

    print("==>> find {} images in {}".format(len(total_anns.keys()), data_path))
    uniq_tracking_id, ann_num = {}, {}
    for img_p, anns in total_anns.items():
        for ann in anns:
            label, tracking_id, box = ann

            if label not in ann_num:
                ann_num[label] = 0
            ann_num[label] += 1

            if label not in uniq_tracking_id:
                uniq_tracking_id[label] = []
            if tracking_id not in uniq_tracking_id[label]:
                uniq_tracking_id[label].append(tracking_id)

    assert set(ann_num.keys()) == set(uniq_tracking_id.keys())
    for label in label_map.keys():
        tracking_ids = uniq_tracking_id[label]
        print("label: {}, id: {}, bbox number: {}, tracking_id number: {}"
              "".format(label, label_map[label], ann_num[label], len(tracking_ids)))
    if data == "train2017":
        print('********** important **********')
        label_name = list(label_map.keys())
        tracking_id_nums = [len(uniq_tracking_id[lab]) for lab in label_name]
        print("change opt.label_name={}".format(label_name))
        print("change opt.tracking_id_nums={}".format(tracking_id_nums))
        print('********** important **********')

    converted_anns = {"images": [], "type": "instances", "annotations": [], "categories": []}
    image_id = 20210000001
    bnd_id = 1
    for img_p in tqdm.tqdm(sorted(total_anns.keys())):
        # print(img_p)
        anns = total_anns[img_p]
        img = cv2.imread(image_path + "/" + img_p)
        height, width, _ = img.shape
        image = {'file_name': img_p, 'height': height, 'width': width, 'id': image_id}
        converted_anns['images'].append(image)

        for label, tracking_id, box in anns:
            x1, y1, w, h = box
            # tracking_id should start with 0 in each class
            new_tracking_id = uniq_tracking_id[label].index(tracking_id)
            category_id = label_map[label]
            ann = {'area': w * h,
                   'iscrowd': 0,
                   'image_id': image_id,
                   'bbox': [x1, y1, w, h],
                   'category_id': category_id,
                   'id': bnd_id,
                   'ignore': 0,
                   'segmentation': [],
                   'tracking_id': new_tracking_id}
            converted_anns['annotations'].append(ann)
            bnd_id = bnd_id + 1

        image_id += 1

    for label, cid in label_map.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': label}
        converted_anns['categories'].append(cat)

    mkdir(os.path.dirname(save_anns))
    with open(save_anns, 'w') as f:
        json.dump(converted_anns, f, indent=2)
    print("save {}".format(save_anns))

    print("moving images, please waiting...")
    print("move {} to {}".format(image_path, save_images))
    shutil.move(image_path, save_images)
    print("convert {} done".format(data))
    shutil.rmtree(data_path)


if __name__ == '__main__':
    # dataset was downloaded from http://aiskyeye.com/download/multi-object-tracking_2021
    # the annotations information can be found here: http://aiskyeye.com/evaluate/results-format_2021/
    dataset_path = opt.dataset_path

    # don't change the following params
    train_dir = dataset_path + '/VisDrone2019-MOT-train'
    val_dir = dataset_path + '/VisDrone2019-MOT-val'
    assert os.path.exists(train_dir), "cannot find train path {}".format(train_dir)
    assert os.path.exists(val_dir), "cannot find val path {}".format(val_dir)
    convert(val_dir, "val2017")
    convert(train_dir, "train2017")
    print("convert done\nyou can use 'python tools/show_coco_anns.py' to visualized the annotations")
