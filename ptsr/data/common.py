from typing import List
import random

import numpy as np
import skimage.color as sc

import torch
from torch.distributions import Beta


def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

############
# Augments #
############

def augment(*args, hflip=True, rot=True, invert=False, c_shuffle=False):
    # apply augmentation to images of shape (H, W, C)
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    invert = invert and random.random() < 0.5
    c_shuffle = c_shuffle and random.random() < 0.6
    if c_shuffle:  # pre-calculate channel order
        c_order = list(range(3))
        random.shuffle(c_order)  # in-place

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        if c_shuffle:
            img = channel_shuffle(img, c_order)
        if invert:
            img = 255 - img # uint8 image
        return img

    # apply the same transformation for all images
    return [_augment(a) for a in args]


def channel_shuffle(img: np.array, c_order: List[int]):
    # change the order of (color) channels
    return img[..., c_order]

class Mixup:
    def __init__(self, bs=16, beta=0.15, choice_thresh=0.3):
        self.beta = Beta(torch.zeros(bs)+beta, torch.zeros(bs)+beta)
        self.choice_thresh = choice_thresh

    @torch.no_grad()
    def __call__(self, lr, hr):
        betas = self.beta.sample().to(lr.device)
        perm = torch.randperm(lr.shape[0])
        lr_perm, hr_perm = lr[perm], hr[perm]
        choices = torch.rand(lr.shape[0])
        idx = torch.where(choices > self.choice_thresh)
        betas[idx] = 1. # only choice_thresh% of samples in batch will be mixed
        lr = lr * betas.view(-1, 1, 1, 1) + lr_perm * (1-betas.view(-1, 1, 1, 1))
        hr = hr * betas.view(-1, 1, 1, 1) + hr_perm * (1-betas.view(-1, 1, 1, 1))
        return lr, hr
