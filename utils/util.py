from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import time
import shutil
import json
import numpy as np
import torch


def merge_opt(opt, params):
    if len(params):
        print("==>> input params:", params)
    input_params = {}
    for arg in params:
        assert "=" in arg, "input format should be: python xxx.py param1=value1 param2=value2"
        name, value = arg.split("=")
        try:
            # string int, string float, string bool, string list, string tuple
            value = eval(value)
        except:
            # string others
            pass
        input_params[name] = value
        value_type = str(type(value)).split("class ")[1].split(">")[0]
        if name in opt:
            if opt[name] != value:
                print("[INFO] change param: {} {} -> {} ({})".format(name, opt[name], value, value_type))
            else:
                print("[INFO] same param: {}={} ({})".format(name, value, value_type))
        else:
            print("[INFO] add param: {}={} ({}) ".format(name, value, value_type))
        opt[name] = value

    def change_list_to_str(cfg, param):
        if param in cfg.keys():
            if isinstance(cfg[param], (list, tuple)):
                new_value = ",".join([str(i) for i in cfg[param]])
                value_t = str(type(new_value)).split("class ")[1].split(">")[0]
                print("[INFO] re-change param: {} {} to {} {} ".format(param, cfg[param], new_value, value_t))
                cfg[param] = new_value
            elif isinstance(cfg[param], int):
                new_value = str(cfg[param])
                value_t = str(type(new_value)).split("class ")[1].split(">")[0]
                print("[INFO] re-change param: {} {} to {} {} ".format(param, cfg[param], new_value, value_t))
                cfg[param] = new_value

        return cfg

    opt = change_list_to_str(opt, "gpus")
    # opt = change_list_to_str(opt, "lr_decay_epoch")
    return opt, input_params


def sync_time(inputs):
    torch.cuda.synchronize() if 'cpu' not in inputs.device.type else ""
    return time.time()


def get_total_and_free_memory_in_mb(cuda_device):
    devices_info_str = os.popen("nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader")
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_mb(0)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.FloatTensor(256, 1024, block_mem).to(cuda_device)
    del x
    time.sleep(5)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def configure_module(ulimit_value=8192):
    """
    Configure pytorch module environment. setting of ulimit and cv2 will be set.

    Args:
        ulimit_value(int): default open file number on linux. Default value: 8192.
    """
    # system setting
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # Exception might be raised in Windows OS or rlimit reaches max limit number.
        # However, set rlimit value might not be necessary.
        pass

    # cv2
    # multiprocess might be harmful on performance of torch dataloader
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # cv2 version mismatch might rasie exceptions.
        pass


def write_log(loss_dict, logger, epoch, phase):
    for k, v in loss_dict.items():
        logger.scalar_summary('{}_{}'.format(phase, k), v, epoch)
        logger.write('{} {:4f} | '.format(k, v))
    logger.write('\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, mean_val, n=1):
        if isinstance(mean_val, torch.Tensor):
            mean_val = mean_val.item()
        self.val = mean_val
        self.sum += mean_val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


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
                img_list.append(img_path.replace("\\", "/"))
    return img_list


def mkdir(path, rm=False):
    if os.path.exists(path):
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def sort_fun(x):
    day, hour = os.path.basename(x).split("_")[1:3]
    hour = hour.replace("-", "")
    return int(day + hour)


def vis_result(img, results, label_list, tracking_result=[], draw_class_num=True):
    class_num = {}
    for res_i, res in enumerate(results):
        cls, conf, bbox = res[0], res[1], res[2]
        class_num[cls] = 1 if cls not in class_num else class_num[cls] + 1
        color = label_color[label_list.index(cls)]
        # show box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # show label and conf
        txt = '{}:{:.2f}'.format(cls, conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(img, (bbox[0], bbox[1] - txt_size[1] - 2), (bbox[0] + txt_size[0], bbox[1] - 2), color, -1)
        cv2.putText(img, txt, (bbox[0], bbox[1] - 2), font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for tracking_r in tracking_result:
        for (x1, y1, x2, y2, cls, det_conf, track_id) in tracking_r:
            color = label_color[label_list.index(cls)]
            cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 5, color, -1)

    if draw_class_num:
        for i, (k, v) in enumerate(class_num.items()):
            cv2.putText(img, k + ":" + str(v), (5, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        label_color[label_list.index(k)], thickness=2)
    return img, class_num


def crop_img(img_org, results, output, img_name, expand=0):
    for res_i, res in enumerate(results):
        cls, conf, bbox = res[0], res[1], res[2]
        bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        crop_x1 = max(0, bbox[0] - int(expand * bbox_w))
        crop_y1 = max(0, bbox[1] - int(expand * bbox_h))
        crop_x2 = min(img_org.shape[1], bbox[2] + int(expand * bbox_w))
        crop_y2 = min(img_org.shape[0], bbox[3] + int(expand * bbox_h))

        crop_img = img_org[crop_y1: crop_y2, crop_x1: crop_x2, :]
        mkdir(output + "/crop/" + str(cls))
        cv2.imwrite(output + "/crop/" + str(cls) + "/" + img_name.split(".")[0] + str(res_i) + ".jpg", crop_img)


def draw_bboxes(image, bboxes, color=(0, 255, 0)):
    for (x1, y1, x2, y2, cls, det_conf, track_id) in bboxes:
        track_text = '{} ID:{} max_conf:{:.2f}'.format(cls, track_id, det_conf)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(image, track_text, (x1 + 2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=1,
                    lineType=cv2.LINE_AA)
        cv2.circle(image, ((x1 + x2) // 2, (y1 + y2) // 2), 5, color, -1)
    return image


label_color = [[31, 0, 255], [0, 159, 255], [255, 95, 0], [255, 19, 0], [255, 0, 0], [255, 38, 0], [0, 255, 25],
               [255, 0, 133],
               [255, 172, 0], [108, 0, 255], [0, 82, 255], [0, 255, 6], [255, 0, 152], [223, 0, 255], [12, 0, 255],
               [0, 255, 178],
               [108, 255, 0], [184, 0, 255], [255, 0, 76], [146, 255, 0], [51, 0, 255], [0, 197, 255], [255, 248, 0],
               [255, 0, 19],
               [255, 0, 38], [89, 255, 0], [127, 255, 0], [255, 153, 0], [0, 255, 255], [0, 255, 216], [0, 255, 121],
               [255, 0, 248],
               [70, 0, 255], [0, 255, 159], [0, 216, 255], [0, 6, 255], [0, 63, 255], [31, 255, 0], [255, 57, 0],
               [255, 0, 210],
               [0, 255, 102], [242, 255, 0], [255, 191, 0], [0, 255, 63], [255, 0, 95], [146, 0, 255], [184, 255, 0],
               [255, 114, 0],
               [0, 255, 235], [255, 229, 0], [0, 178, 255], [255, 0, 114], [255, 0, 57], [0, 140, 255], [0, 121, 255],
               [12, 255, 0],
               [255, 210, 0], [0, 255, 44], [165, 255, 0], [0, 25, 255], [0, 255, 140], [0, 101, 255], [0, 255, 82],
               [223, 255, 0],
               [242, 0, 255], [89, 0, 255], [165, 0, 255], [70, 255, 0], [255, 0, 172], [255, 76, 0], [203, 255, 0],
               [204, 0, 255],
               [255, 0, 229], [255, 133, 0], [127, 0, 255], [0, 235, 255], [0, 255, 197], [255, 0, 191], [0, 44, 255],
               [50, 255, 0],
               [31, 0, 255], [0, 159, 255], [255, 95, 0], [255, 19, 0], [255, 0, 0], [255, 38, 0], [0, 255, 25],
               [255, 0, 133],
               [255, 172, 0], [108, 0, 255], [0, 82, 255], [0, 255, 6], [255, 0, 152], [223, 0, 255], [12, 0, 255],
               [0, 255, 178],
               [108, 255, 0], [184, 0, 255], [255, 0, 76], [146, 255, 0], [51, 0, 255], [0, 197, 255], [255, 248, 0],
               [255, 0, 19]]
