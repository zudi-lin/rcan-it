from typing import List, Callable, Union, List, Tuple

import os
import math
import itertools
import numpy as np
import torch.nn as nn
from .common import *
from ._utils import conv3x3, get_num_params

BLOCK_DICT = {
    'basicblock': PreActBasicBlock,
    'bottleneck': PreActBottleneck,
    'mbconv': MBConvBlock,
    'basicblock_dw': PreActBasicBlockDW,
    'edsr_block': EDSRBlock,
    'rcan_block': RCANBlock,
    'rcan_block_dw': RCANBlockDW,
    'rcan_block_all_dw': RCANBlockAllDW,
}

AFFINE_LIST = ['basicblock', 'bottleneck',
               'mbconv', 'basicblock_dw']


class ResidualGroup(nn.Module):
    def __init__(self, block_type: str, n_resblocks: int, planes: int,
                 short_skip: bool = False, out_conv: bool = False, 
                 df_conv: bool = False, **kwargs):
        super().__init__()
        self.short_skip = short_skip

        assert block_type in BLOCK_DICT
        blocks = [BLOCK_DICT[block_type](planes, **kwargs)
                  for _ in range(n_resblocks)]
        if out_conv:
            # the final convolution for each residual block can be deformable
            blocks.append(conv3x3(planes, planes, bias=True, df_conv=df_conv))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.body(x)
        if self.short_skip:
            res += x
        return res


class ISRNet(nn.Module):
    def __init__(self, n_resgroups: int, n_resblocks: int, planes: int, scale: int,
                 prob: List[float], block_type: str, channels: int = 3, rgb_range: int = 255,
                 act_mode: str = 'relu', **kwargs):
        super().__init__()
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        modules_head = [conv3x3(channels, planes, bias=True)]

        modules_body = [ResidualGroup(
            block_type, n_resblocks, planes, act_mode=act_mode,
            prob=prob[i], **kwargs) for i in range(n_resgroups)]
        modules_body.append(conv3x3(planes, planes, bias=True))

        modules_tail = [
            Upsampler(scale, planes, act_mode, use_affine=(
                block_type in AFFINE_LIST)),
            conv3x3(planes, channels, bias=True)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x  # long skip-connection

        x = self.tail(res)
        x = self.add_mean(x)

        return x


class Model(nn.Module):
    def __init__(self, cfg, ckp=None):
        super().__init__()
        self.scale = cfg.DATASET.DATA_SCALE[0]
        self.ensemble = cfg.MODEL.ENSEMBLE.ENABLED
        self.ensemble_mode = cfg.MODEL.ENSEMBLE.MODE
        self.chop = cfg.DATASET.CHOP
        self.chop_options = (cfg.DATASET.CHOP_PAD, cfg.DATASET.CHOP_THRES)
        self.overlap = cfg.DATASET.OVERLAP.ENABLED
        if self.overlap: # overlap inference overwrite chop
            self.chop = False
            self.overlap_options = (cfg.DATASET.OVERLAP.STRIDE, 
                                    cfg.DATASET.OVERLAP.SIZE)

        self.model = self.make_model(cfg)
        if ckp is not None:
            print("Params: " + str(get_num_params(self.model)), file=ckp.log_file)
            print(self.model, file=ckp.log_file)

    def make_model(self, cfg):
        n = cfg.MODEL.N_RESGROUPS
        options = {
            'n_resgroups': n,
            'n_resblocks': cfg.MODEL.N_RESBLOCKS,
            'out_conv': cfg.MODEL.OUT_CONV,
            'planes': cfg.MODEL.PLANES,
            'scale': cfg.DATASET.DATA_SCALE[0],
            'block_type': cfg.MODEL.BLOCK_TYPE,
            'short_skip': cfg.MODEL.SHORT_SKIP,
            'channels': cfg.DATASET.CHANNELS,
            'rgb_range': cfg.DATASET.RGB_RANGE,
            'act_mode': cfg.MODEL.ACT_MODE,
            'stochastic_depth': cfg.MODEL.STOCHASTIC_DEPTH,
            'multFlag': cfg.MODEL.MULT_FLAG,
            'reduction': cfg.MODEL.SE_REDUCTION,  # SE block
            'affine_init_w': cfg.MODEL.AFFINE_INIT_W,
            'df_conv': cfg.MODEL.DEFORM_CONV, # Deformable convolution
            'zero_inti_residual': cfg.MODEL.ZERO_INIT_RESIDUAL,
            'res_scale': cfg.MODEL.RES_SCALE, # Scale of residual connection
            'res_scale_learnable': cfg.MODEL.RES_SCALE_LEARNABLE,
            'normal_init_std': cfg.MODEL.NORMAL_INIT_STD,
        }

        # build a probability list for stochastic depth
        prob = cfg.MODEL.STOCHASTIC_DEPTH_PROB
        if prob is None:
            options['prob'] = [0.5] * n
        elif isinstance(prob, float):
            options['prob'] = [prob] * n
        elif isinstance(prob, list):
            assert len(prob) == 2
            n = cfg.MODEL.N_RESGROUPS
            temp = np.arange(n) / float(n-1)
            prob_list = prob[0] + temp * (prob[1] - prob[0])
            options['prob'] = list(prob_list)

        return ISRNet(**options)

    def forward(self, x):
        if self.training:
            return self.model(x)

        # inference mode
        if self.chop:
            forward_func = self.forward_patch
        elif self.overlap:
            forward_func = self.forward_overlap
        else: # whole-image inference
            forward_func = self.model.forward

        if self.ensemble:
            return self.forward_ensemble(x, forward_func=forward_func)

        return forward_func(x)

    def forward_patch(self, x):
        padding, threshold = self.chop_options
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + padding, w_half + padding
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        n_samples = 2
        if (w_size * h_size) < threshold:  # smaller than the threshold
            sr_list = []
            for i in range(0, 4, n_samples):
                lr_batch = torch.cat(lr_list[i:(i + n_samples)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_samples, dim=0))
        else:
            sr_list = [
                self.forward_patch(patch) for patch in lr_list]

        h, w = self.scale * h, self.scale * w
        h_half, w_half = self.scale * h_half, self.scale * w_half
        h_size, w_size = self.scale * h_size, self.scale * w_size

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_ensemble(self, x, forward_func: Callable):
        def _transform(data, xflip, yflip, transpose, reverse=False):
            if not reverse: # forward transform
                if xflip:
                    data = torch.flip(data, [3])
                if yflip:
                    data = torch.flip(data, [2])
                if transpose:
                    data = torch.transpose(data, 2, 3)
            else: # reverse transform
                if transpose:
                    data = torch.transpose(data, 2, 3)
                if yflip:
                    data = torch.flip(data, [2])
                if xflip:
                    data = torch.flip(data, [3])
            return data

        outputs = []
        opts = itertools.product((False, True), (False, True), (False, True))
        for xflip, yflip, transpose in opts:
            data = x.clone()
            data = _transform(data, xflip, yflip, transpose)
            data = forward_func(data)
            outputs.append(
                _transform(data, xflip, yflip, transpose, reverse=True))

        if self.ensemble_mode == 'mean':
            return torch.stack(outputs, 0).mean(0)
        elif self.ensemble_mode == 'median':
            # https://pytorch.org/docs/stable/generated/torch.median.html
            return torch.stack(outputs, 0).median(0)[0]
        else:
            raise ValueError("Unknown ensemble mode %s." % self.ensemble_mode)

    def forward_overlap(self, x):
        b, c, h, w = x.size()
        image_size = [h, w]
        stride, patch_size = self.overlap_options
        stride, patch_size = np.array([stride] * 2), np.array([patch_size] * 2)
        sz = count_image(np.array((h, w)), patch_size, stride)
        num_sample = np.prod(sz)

        h, w = self.scale * h, self.scale * w
        output = x.new(b, c, h, w).cpu()
        weight = x.new_zeros(b, 1, h, w).cpu()
        out_sz = self.scale * patch_size
        ww = blend_gaussian(tuple(out_sz))

        for i in range(num_sample):
            pos = get_pos_test(i, sz, stride, patch_size, image_size)
            x_in = x[:, :, pos[0]:pos[0]+patch_size[0], pos[1]:pos[1]+patch_size[1]]
            x_out = self.model(x_in).cpu()
            pos = [x * self.scale for x in pos]
            output[:, :, pos[0]:pos[0]+out_sz[0], pos[1]:pos[1]+out_sz[1]] = x_out * ww
            weight[:, :, pos[0]:pos[0]+out_sz[0], pos[1]:pos[1]+out_sz[1]] = ww

        return output / weight

# utils for forward_overlap

def index_to_location(index, sz):
    pos = [0, 0]
    pos[0] = np.floor(index/sz[1])
    pos[1] = index % sz[1]
    return pos

def get_pos_test(index, sz, stride, patch_size, image_size):
    pos = index_to_location(index, sz)
    for i in range(2): 
        if pos[i] != sz[i]-1:
            pos[i] = int(pos[i] * stride[i])
        else:
            pos[i] = int(image_size[i] - patch_size[i])
    return pos

def count_image(data_sz, sz, stride):
    return 1 + np.ceil((data_sz - sz) / stride.astype(float)).astype(int)

def blend_gaussian(sz: Union[Tuple[int], List[int]], 
                   sigma: float=0.2, 
                   mu: float=0.0) -> np.ndarray:  
    """
    Gaussian blending matrix for sliding-window inference.
    Args:
        sz: size of the blending matrix
        sigma (float): standard deviation of the Gaussian distribution. Default: 0.2
        mu (float): mean of the Gaussian distribution. Default: 0.0
    """
    xx, yy = np.meshgrid(np.linspace(-1,1,sz[0], dtype=np.float32), 
                         np.linspace(-1,1,sz[1], dtype=np.float32), 
                         indexing='ij')

    dd = np.sqrt(xx*xx + yy*yy)
    ww = 1e-4 + np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 )))
    return torch.from_numpy(ww.astype(np.float32)).unsqueeze(0).unsqueeze(1)
