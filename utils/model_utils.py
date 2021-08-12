# -*- coding: utf-8 -*-
# @Time    : 2021/6/21 21:17
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import torch
from torch.nn.utils import clip_grad


def clip_grads(model, grad_clip=dict(max_norm=35, norm_type=2)):
    params = model.parameters()
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        grad_normal = clip_grad.clip_grad_norm_(params, **grad_clip)
        return grad_normal
    else:
        print("cannot find parameters in model.parameters(), no clipping grad")


def load_model(model, model_path, optimizer=None, scaler=None, resume=False):
    assert os.path.isfile(model_path), "model {} not find".format(model_path)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('==>> loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    start_epoch = 0
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('--> Skip loading parameter {}, required shape {}, loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('--> Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    # for index, (k, v) in enumerate(state_dict.items()):
    #    print("Load pretrained weights: {}, {}, {}".format(index, k, v.size()))

    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        start_epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==>> Resumed optimizer")
            if scaler is not None:
                if 'scaler' in checkpoint:
                    if checkpoint['scaler'] != {}:
                        scaler.load_state_dict(checkpoint['scaler'])
                        print("==>> Resumed scaler")
                    else:
                        print("Skip load scaler: '{}'")
                else:
                    print('==>> No scaler in checkpoint.')
        else:
            print('==>> No optimizer in checkpoint.')

    if optimizer is not None:
        print("==>> Set start_epoch {}".format(start_epoch))
        return model, optimizer, start_epoch, scaler
    else:
        return model


def save_model(path, epoch, model, optimizer=None, scaler=None, logger=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    data = {'epoch': epoch, 'state_dict': state_dict}

    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    if not (scaler is None):
        data["scaler"] = scaler.state_dict()
    print("epoch {}, save weight to {}".format(epoch, path))
    if logger:
        logger.write("epoch {}, save weight to {}\n".format(epoch, path))
    torch.save(data, path)


class EMA(object):
    '''
        apply expontential moving average to a model. This should have same function as the `tf.train.ExponentialMovingAverage` of tensorflow.
        usage:
            model = resnet()
            model.train()
            ema = EMA(model, 0.9999)
            ....
            for img, lb in dataloader:
                loss = ...
                loss.backward()
                optim.step()
                ema.update_params() # apply ema
            evaluate(model)  # evaluate with original model as usual
            ema.apply_shadow() # copy ema status to the model
            evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters
        args:
            - model: the model that ema is applied
            - alpha: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
            - buffer_ema: whether the model buffers should be computed with ema method or just get kept
        methods:
            - update_params(): apply ema to the model, usually call after the optimizer.step() is called
            - apply_shadow(): copy the ema processed parameters to the model
            - restore(): restore the original model parameters, this would cancel the operation of apply_shadow()
    '''

    def __init__(self, model, alpha=0.9998, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()

        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])

        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}


def ensure_same(model1, model2):
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    state_dict1 = {}
    state_dict2 = {}

    # convert data_parallal to model
    for k in model2_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict2[k[7:]] = model2_state_dict[k]
        else:
            state_dict2[k] = model2_state_dict[k]

    for k in model1_state_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict1[k[7:]] = model1_state_dict[k]
        else:
            state_dict1[k] = model1_state_dict[k]

    # check loaded parameters and created model parameters
    for k in state_dict1:
        assert k in state_dict2, "key '{}' not in model2".format(k)
        assert state_dict1[k].shape == state_dict2[k].shape, "shape not same"
        assert state_dict1[k].equal(state_dict2[k]), "value not same"
    print("same params in model1 and model2")


def de_sigmoid(x, eps=1e-4):
    x = x.clip(eps, 1. / eps)
    x = 1. / x - 1.
    x = x.clip(eps, 1. / eps)
    x = -torch.log(x)
    return x
