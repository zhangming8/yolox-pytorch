# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 20:42
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import cv2
import numpy as np
import torch
import sys

sys.path.append(".")
from data import (COCODataset, TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection)


def get_dataloader(opt, no_aug=False):
    # train
    do_tracking = opt.reid_dim > 0
    train_dataset = COCODataset(data_dir=opt.data_dir,
                                json_file=opt.train_ann,
                                img_size=opt.input_size,
                                tracking=do_tracking,
                                preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, tracking=do_tracking),
                                )
    train_dataset = MosaicDetection(
        train_dataset,
        mosaic=not no_aug,
        img_size=opt.input_size,
        preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120, tracking=do_tracking),
        degrees=opt.degrees,
        translate=opt.translate,
        scale=opt.scale,
        shear=opt.shear,
        perspective=opt.perspective,
        enable_mixup=opt.enable_mixup,
        tracking=do_tracking,
    )
    train_sampler = InfiniteSampler(len(train_dataset), seed=opt.seed)
    batch_sampler = YoloBatchSampler(
        sampler=train_sampler,
        batch_size=opt.batch_size,
        drop_last=False,
        input_dimension=opt.input_size,
        mosaic=not no_aug,
    )
    train_kwargs = {"num_workers": opt.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler}
    train_loader = DataLoader(train_dataset, **train_kwargs)

    # val
    val_dataset = COCODataset(
        data_dir=opt.data_dir,
        json_file=opt.val_ann,
        name="val2017",
        img_size=opt.test_size,
        tracking=do_tracking,
        preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=120, tracking=do_tracking,
                               augment=False))
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_kwargs = {"num_workers": opt.data_num_workers, "pin_memory": True, "sampler": val_sampler,
                  "batch_size": opt.batch_size}
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)

    return train_loader, val_loader


def memory_info():
    import psutil

    mem_total = psutil.virtual_memory().total / 1024 / 1024 / 1024
    mem_used = psutil.virtual_memory().used / 1024 / 1024 / 1024
    mem_percent = psutil.virtual_memory().percent
    return mem_percent, mem_used, mem_total


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


def run_epoch(data_iter, loader, total_iter, e, phase, opt):
    for batch_i in range(total_iter):
        batch = next(data_iter)
        inps, targets, img_info, ind = batch
        print("------------ epoch {} batch {}/{} ---------------".format(e, batch_i, total_iter))
        print("batch img shape {}, target shape {}".format(inps.shape, targets.shape))
        if opt.show:
            vis_inputs(inps, targets, opt)
        if batch_i == 0:
            print(ind)

        # random input
        if phase == 'train' and opt.random_size is not None and batch_i % 2 == 0:
            tensor = torch.LongTensor(2)
            size_factor = opt.input_size[1] * 1. / opt.input_size[0]
            size = np.random.randint(*opt.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]
            input_size = loader.change_input_dim(multiple=(tensor[0].item(), tensor[1].item()), random_range=None)


def main():
    from config import opt

    opt.input_size = (640, 640)
    opt.test_size = (640, 640)
    opt.batch_size = 2
    opt.data_num_workers = 0  # 0
    opt.reid_dim = 0  # 128
    opt.show = True  # False
    print(opt)
    train_loader, val_loader = get_dataloader(opt, no_aug=False)

    loader, phase = train_loader, 'train'
    # loader, phase = val_loader, 'val'

    dataset_label = loader.dataset._dataset.classes if phase == "train" else loader.dataset.classes
    assert opt.label_name == dataset_label, "your class_name != dataset's {} {}".format(opt.label_name, dataset_label)
    total_iter = len(loader)
    data_iter = iter(loader)
    for e in range(100):
        # train_loader.dataset.enable_mixup = False
        # train_loader.close_mosaic()
        run_epoch(data_iter, loader, total_iter, e, phase, opt)


if __name__ == "__main__":
    main()
