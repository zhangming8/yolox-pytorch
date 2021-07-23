# -*- coding: utf-8 -*-
# @Time    : 2021/7/23 23:06
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import torch
import torchvision


def yolo_post_process(outputs, down_strides, num_classes, conf_thre, nms_thre, label_name, img_ratios):
    hw = [i.shape[-2:] for i in outputs]
    # x,y,w,h,obj,cls
    for x in outputs:
        x[:, 4:, :, :].sigmoid_()
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

    grids = []
    strides = []
    for (hsize, wsize), stride in zip(hw, down_strides):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(outputs.dtype)
    strides = torch.cat(strides, dim=1).type(outputs.dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

    box_corner = outputs.new(outputs.shape)
    box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2
    box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2
    box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2
    box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2
    outputs[:, :, :4] = box_corner[:, :, :4]

    output = [[] for _ in range(len(outputs))]
    for i, image_pred in enumerate(outputs):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                                                    detections[:, 6], nms_thre)
        detections = detections[nms_out_index]

        detections[:, :4] = detections[:, :4] / img_ratios[i]

        for det in detections:
            x1, y1, x2, y2, obj_conf, class_conf, class_pred = det
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            conf = float(obj_conf * class_conf)
            label = label_name[int(class_pred)]
            output[i].append([label, conf, bbox])
    return output


def bbox_post_process(yolo_head_outs, anchors, downsamples, label_name, org_shapes, input_shape, conf_thres, iou_thres,
                      max_det, multi_label):
    outs = []
    for p, anchor, downsample in zip(yolo_head_outs, anchors, downsamples):
        bs, c, grid_h, grid_w = p.shape
        na = len(anchor)
        anchor = torch.tensor(anchor, dtype=p.dtype, device=p.device)
        anchor = anchor.view((1, na, 1, 1, 2))
        grid = make_grid(grid_h, grid_w, p.dtype, p.device).view((1, 1, grid_h, grid_w, 2))
        p = p.view((bs, na, -1, grid_h, grid_w)).permute((0, 1, 3, 4, 2))

        x, y = torch.sigmoid(p[:, :, :, :, 0:1]), torch.sigmoid(p[:, :, :, :, 1:2])
        w, h = torch.exp(p[:, :, :, :, 2:3]), torch.exp(p[:, :, :, :, 3:4])
        obj, pcls = torch.sigmoid(p[:, :, :, :, 4:5]), torch.sigmoid(p[:, :, :, :, 5:])

        # rescale to input size, eg: 512x512
        x = (x + grid[:, :, :, :, 0:1]) * downsample
        y = (y + grid[:, :, :, :, 1:2]) * downsample
        w = w * anchor[:, :, :, :, 0:1]
        h = h * anchor[:, :, :, :, 1:2]

        out = torch.cat([x, y, w, h, obj, pcls], -1).view(bs, na * grid_h * grid_w, -1)
        outs.append(out)
    outs = torch.cat(outs, 1)

    outs = non_max_suppression(outs, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
                               multi_label=multi_label)

    # rescale to original image size
    predict = []
    for batch_i, pred in enumerate(outs):
        one_img_res = []
        if pred.shape[0]:
            org_h, org_w = org_shapes[batch_i]
            pred = transform(pred, org_shapes[batch_i], input_shape)
            for p in pred:
                x1, y1, x2, y2, conf, cls_id = p
                label = label_name[int(cls_id)]
                x1 = int(max(0, min(x1, org_w - 1)))
                y1 = int(max(0, min(y1, org_h - 1)))
                x2 = int(max(0, min(x2, org_w - 1)))
                y2 = int(max(0, min(y2, org_h - 1)))
                one_img_res.append([label, float(conf), [x1, y1, x2, y2]])

        predict.append(one_img_res)
    return predict


def transform(box, org_img_shape, img_size):
    x1, y1, x2, y2 = box[:, 0:1], box[:, 1:2], box[:, 2:3], box[:, 3:4]
    org_h, org_w = org_img_shape
    model_h, model_w = img_size
    org_ratio = org_w / float(org_h)
    model_ratio = model_w / float(model_h)
    if org_ratio >= model_ratio:
        # pad h
        scale = org_w / float(model_w)
        x1 = x1 * scale
        x2 = x2 * scale
        pad = (scale * model_h - org_h) / 2
        y1 = scale * y1 - pad
        y2 = scale * y2 - pad
    else:
        # pad w
        scale = org_h / float(model_h)
        y1 = y1 * scale
        y2 = y2 * scale
        pad = (scale * model_w - org_w) / 2
        x1 = x1 * scale - pad
        x2 = x2 * scale - pad
    box[:, 0:1], box[:, 1:2], box[:, 2:3], box[:, 3:4] = x1, y1, x2, y2
    return box
