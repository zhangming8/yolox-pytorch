# -*- coding: utf-8 -*-
# @Time    : 2021/7/23 23:06
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import torch
import torch.nn.functional as F
import torchvision


def yolox_post_process(outputs, down_strides, num_classes, conf_thre, nms_thre, label_name, img_ratios, img_shape):
    hw = [i.shape[-2:] for i in outputs]
    grids, strides = [], []
    for (hsize, wsize), stride in zip(hw, down_strides):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # bs, all_anchor, 85(+128)
    grids = torch.cat(grids, dim=1).type(outputs.dtype).to(outputs.device)
    strides = torch.cat(strides, dim=1).type(outputs.dtype).to(outputs.device)

    # x, y
    outputs[..., 0:2] = (outputs[..., 0:2] + grids) * strides
    # w, h
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # obj
    outputs[..., 4:5] = torch.sigmoid(outputs[..., 4:5])
    # 80 class
    outputs[..., 5:5 + num_classes] = torch.sigmoid(outputs[..., 5:5 + num_classes])
    # reid
    reid_dim = outputs.shape[2] - num_classes - 5
    if reid_dim > 0:
        outputs[..., 5 + num_classes:] = F.normalize(outputs[..., 5 + num_classes:], dim=2)

    box_corner = outputs.new(outputs.shape)
    box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2  # x1
    box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2  # y1
    box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2  # x2
    box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2  # y2
    outputs[:, :, :4] = box_corner[:, :, :4]

    output = [[] for _ in range(len(outputs))]
    for i, image_pred in enumerate(outputs):
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        if reid_dim > 0:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5 + num_classes:]),
                                   1)
        else:
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                                                    detections[:, 6], nms_thre)
        detections = detections[nms_out_index]

        detections[:, :4] = detections[:, :4] / img_ratios[i]

        img_h, img_w = img_shape[i]
        for det in detections:
            x1, y1, x2, y2, obj_conf, class_conf, class_pred = det[0:7]
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            conf = float(obj_conf * class_conf)
            label = label_name[int(class_pred)]
            # clip bbox
            bbox[0] = max(0, min(img_w, bbox[0]))
            bbox[1] = max(0, min(img_h, bbox[1]))
            bbox[2] = max(0, min(img_w, bbox[2]))
            bbox[3] = max(0, min(img_h, bbox[3]))

            if reid_dim > 0:
                reid_feat = det[7:].cpu().numpy().tolist()
                output[i].append([label, conf, bbox, reid_feat])
            else:
                output[i].append([label, conf, bbox])
    return output
