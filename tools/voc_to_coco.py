# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 21:08
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import tqdm
import os
import shutil
import json
import sys
import xml.etree.ElementTree as ET  # pip install lxml

sys.path.append(".")
from config import opt

START_BOUNDING_BOX_ID = 1


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


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(img_list, json_file, message="converting"):
    if len(img_list) <= 0:
        print("empty img list, cannot {}".format(message))
        return
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories_id = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    dataset_categorie_num = {}
    for index, img_p in enumerate(tqdm.tqdm(img_list, message)):
        img_f = img_p.replace("\\", "/")
        xml_f = img_f.replace(img_format, ".xml")
        assert os.path.isfile(xml_f), "cannot find annotation ({}) of image {}".format(xml_f, img_p)
        tree = ET.parse(xml_f)
        root = tree.getroot()

        tmp_category = []
        for obj in get(root, 'object'):
            tmp_category.append(get_and_check(obj, 'name', 1).text)
        intersection = [i for i in tmp_category if i in classes]
        if only_care_pre_define_categories and len(intersection) == 0:
            continue
        filename = os.path.basename(img_f)
        image_id = 20210000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in dataset_categorie_num:
                dataset_categorie_num[category] += 1
            else:
                dataset_categorie_num[category] = 1
            if category not in categories_id:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories_id) + 1
                print("[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically"
                      "".format(category, pre_define_categories, new_id))
                categories_id[category] = new_id
            category_id = categories_id[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert xmax > xmin, "error xmax <= xmin, {}".format(xml_f)
            assert ymax > ymin, "error ymax <= ymin, {}".format(xml_f)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height,
                   'iscrowd': 0,
                   'image_id': image_id,
                   'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id,
                   'id': bnd_id,
                   'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories_id.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    mkdir(os.path.dirname(json_file))
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=2)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))

    print("find {} categories_id: {} in your dataset\nyour setting categories_id {}: {}"
          "".format(len(dataset_categorie_num), list(set(list(dataset_categorie_num.keys()))), len(classes), classes))
    if set(list(dataset_categorie_num.keys())) == set(classes):
        print("they are same")
    else:
        print("they are different")
    print("categories_id: {}".format(categories_id))
    print("save annotation to: {}".format(json_file))


if __name__ == '__main__':
    # convert VOC format dataset to COCO
    classes = opt.label_name
    train_img_xml_dir = opt.dataset_path + "/images/train2017"
    val_img_xml_dir = opt.dataset_path + "/images/val2017"
    save_json_train = opt.dataset_path + "/annotations/instances_train2017.json"
    save_json_val = opt.dataset_path + "/annotations/instances_val2017.json"

    img_format = ".jpg"
    only_care_pre_define_categories = True
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    train_list = get_img_path(train_img_xml_dir, img_format)
    print("find {} images in {}".format(len(train_list), train_img_xml_dir))
    val_list = get_img_path(val_img_xml_dir, img_format)
    print("find {} images in {}".format(len(val_list), val_img_xml_dir))

    print("start convert voc to coco ...")
    convert(train_list, save_json_train, "convert train")
    print("-" * 100)
    convert(val_list, save_json_val, "convert val")
    print("convert done\nyou can use 'python tools/show_coco_anns.py' to visualized the annotations")
