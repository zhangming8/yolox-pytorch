# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 20:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from __future__ import print_function, division

import os
import shutil
import cv2
import time
import numpy as np
from progress.bar import Bar
import torch
import torch.nn as nn

from config import opt
from data.coco_dataset import get_dataloader
from models.yolox import get_model
from utils.lr_scheduler import LRScheduler
from utils.util import AverageMeter, write_log, configure_module
from utils.model_utils import EMA, save_model, load_model, ensure_same, clip_grads
from utils.data_parallel import set_device
from utils.logger import Logger


def run_epoch(model_with_loss, optimizer, scaler, ema, phase, epoch, data_iter, num_iter, total_iter,
              train_loader=None, lr_scheduler=None, accumulate=1):
    if phase == 'train':
        model_with_loss.train()
    else:
        model_with_loss.eval()
        torch.cuda.empty_cache()

    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {}
    bar = Bar('{}'.format(opt.exp_id), max=num_iter)
    end = time.time()
    last_opt_iter = 0
    optimizer.zero_grad() if phase == 'train' else ""
    for iter_id in range(1, num_iter + 1):
        # load data
        inps, targets, img_info, ind = next(data_iter)
        inps = inps.to(device=opt.device, non_blocking=True)
        targets = targets.to(device=opt.device, non_blocking=True)
        data_time.update(time.time() - end)

        # inference and call loss
        return_pred = phase != 'train' and "ap" in opt.metric and opt.val_intervals > 0 and epoch % opt.val_intervals == 0
        img_ratio = [float(min(opt.test_size[0] / img_info[0][i], opt.test_size[1] / img_info[1][i])) for i in
                     range(inps.shape[0])] if return_pred else None
        preds, loss_stats = model_with_loss(inps, targets=targets, return_loss=True, return_pred=return_pred,
                                            ratio=img_ratio, vis_thresh=0.01)
        loss_stats = {k: v.mean() for k, v in loss_stats.items()}
        if phase == 'train':
            iteration = (epoch - 1) * num_iter + iter_id
            scaler.scale(loss_stats["loss"]).backward()

            if iteration - last_opt_iter >= accumulate or iter_id == num_iter:
                if opt.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    grad_normal = clip_grads(model_with_loss, opt.grad_clip)
                    if not opt.use_amp:
                        loss_stats['grad_normal'] = grad_normal
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update_params() if ema else ''
                last_opt_iter = iteration

            lr = lr_scheduler.update_lr(iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if (iteration - 1) % 50 == 0 and epoch <= 10:
                logger.scalar_summary("lr_iter_before_10_epoch", lr, iteration)
        else:
            iteration, total_iter = iter_id, num_iter
            lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - end)
        end = time.time()
        if return_pred:
            for img_id, pred in zip(ind.cpu().numpy().tolist(), preds):
                results[img_id] = pred

        shapes = "x".join([str(i) for i in inps.shape])
        Bar.suffix = '{phase}: total_epoch[{0}/{1}] total_batch[{2}/{3}] batch[{4}/{5}] |size: {6} |lr: {7} |Tot: ' \
                     '{total:} |ETA: {eta:} '.format(epoch, opt.num_epochs, iteration, total_iter, iter_id, num_iter,
                                                     shapes, "{:.8f}".format(lr), phase=phase, total=bar.elapsed_td,
                                                     eta=bar.eta_td)
        for l in loss_stats:
            if l not in avg_loss_stats:
                avg_loss_stats[l] = AverageMeter()
            avg_loss_stats[l].update(loss_stats[l], inps.size(0))
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if opt.print_iter > 0 and iter_id % opt.print_iter == 0:
            print('{}| {}'.format(opt.exp_id, Bar.suffix))
            logger.write('{}| {}\n'.format(opt.exp_id, Bar.suffix))
        else:
            bar.next()

        # random resizing
        if phase == 'train' and opt.random_size is not None and (iteration % 10 == 0 or iteration <= 20):
            tensor = torch.LongTensor(2).to(device=opt.device)
            size_factor = opt.input_size[1] * 1. / opt.input_size[0]
            size = np.random.randint(*opt.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0], tensor[1] = size[0], size[1]
            if iteration <= 10:
                # initialize with max size, in case of out of memory during training
                tensor[0], tensor[1] = int(max(opt.random_size) * 32), int(max(opt.random_size) * 32)
            input_size = train_loader.change_input_dim(multiple=(tensor[0].item(), tensor[1].item()), random_range=None)

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results


def train(model, scaler, train_loader, val_loader, optimizer, lr_scheduler, start_epoch, accumulate, no_aug):
    best = 1e10 if opt.metric == "loss" else -1
    iter_per_train_epoch = len(train_loader)
    iter_per_val_epoch = len(val_loader)

    # initialize data loader
    train_iter = iter(train_loader)
    total_train_iteration = opt.num_epochs * iter_per_train_epoch

    # exponential moving average
    ema = EMA(model) if opt.ema else None

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        if epoch == opt.num_epochs - opt.no_aug_epochs or no_aug:
            logger.write("--->No mosaic aug now! epoch {}\n".format(epoch))
            train_loader.close_mosaic()
            if isinstance(model, torch.nn.DataParallel):
                model.module.loss.use_l1 = True
            else:
                model.loss.use_l1 = True
            opt.val_intervals = 1
            logger.write("--->Add additional L1 loss now! epoch {}\n".format(epoch))

        logger.scalar_summary("lr_epoch", optimizer.param_groups[0]['lr'], epoch)
        loss_dict_train, _ = run_epoch(model, optimizer, scaler, ema, "train", epoch, train_iter, iter_per_train_epoch,
                                       total_train_iteration, train_loader, lr_scheduler, accumulate)
        logger.write('train epoch: {} |'.format(epoch))
        write_log(loss_dict_train, logger, epoch, "train")

        ema.apply_shadow() if ema is not None else ""
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model)
            logger.write('----------epoch {} evaluating----------\n'.format(epoch))
            with torch.no_grad():
                loss_dict_val, preds = run_epoch(model, optimizer, None, None, "val", epoch, iter(val_loader),
                                                 iter_per_val_epoch, iter_per_val_epoch)
            logger.write('----------epoch {} evaluate done----------\n'.format(epoch))
            logger.write('val epoch: {} |'.format(epoch))
            write_log(loss_dict_val, logger, epoch, "val")

            if "ap" in opt.metric.lower():
                ap, ap_0_5 = val_loader.dataset.run_coco_eval(preds, opt.save_dir)
                logger.write(
                    "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(ap))
                logger.write(
                    "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n".format(ap_0_5))
                logger.scalar_summary("val_AP", ap, epoch)
                logger.scalar_summary("val_AP_05", ap_0_5, epoch)
                if ap >= best:
                    best = ap
                    save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
            elif opt.metric == "loss":
                if loss_dict_val['loss'] <= best:
                    best = loss_dict_val['loss']
                    save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)

        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch,
                   model) if epoch % opt.save_epoch == 0 else ""
        save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer, scaler)
        ema.restore() if ema is not None else ""

    logger.close()


def main():
    # define model with loss
    model = get_model(opt)

    # define optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    lr = opt.warmup_lr if opt.warmup_epochs > 0 else opt.basic_lr_per_img * opt.batch_size
    optimizer = torch.optim.SGD(pg0, lr=lr, momentum=opt.momentum, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": opt.weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})

    # Automatic mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp, init_scale=2. ** 16)

    # fine-tune or resume
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch, scaler = load_model(model, opt.load_model, optimizer, scaler, opt.resume)

    # define loader
    no_aug = start_epoch >= opt.num_epochs - opt.no_aug_epochs
    train_loader, val_loader = get_dataloader(opt, no_aug=no_aug)
    dataset_label = val_loader.dataset.classes
    assert opt.label_name == dataset_label, "[ERROR] 'opt.label_name' should be the same as dataset's {} != {}".format(
        opt.label_name, dataset_label)
    # learning ratio scheduler
    base_lr = opt.basic_lr_per_img * opt.batch_size
    lr_scheduler = LRScheduler(opt.scheduler, base_lr, len(train_loader), opt.num_epochs,
                               warmup_epochs=opt.warmup_epochs, warmup_lr_start=opt.warmup_lr,
                               no_aug_epochs=opt.no_aug_epochs, min_lr_ratio=opt.min_lr_ratio)

    # DP
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    model, optimizer = set_device(model, optimizer, opt)
    train(model, scaler, train_loader, val_loader, optimizer, lr_scheduler, start_epoch, opt.accumulate, no_aug)


if __name__ == "__main__":
    configure_module()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = opt.cuda_benchmark

    logger = Logger(opt)
    shutil.copyfile("./config.py", logger.log_path + "/config.py")
    main()
