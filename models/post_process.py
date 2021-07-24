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
