import unittest

from numpy.lib.function_base import percentile

import numpy as np
import torch
import skimage
from skimage.transform import resize
from ptsr.utils.utility import calc_psnr_torch, calc_psnr_numpy


class TestUtils(unittest.TestCase):
    def test_psnr_calculation(self):
        img_caller = getattr(skimage.data, 'astronaut')
        hr = img_caller()

        scale = 4
        lr_shape = (hr.shape[0] // scale,
                    hr.shape[1] // scale,
                    hr.shape[2])
        lr = resize(hr, lr_shape, order=3, preserve_range=True)
        sr = resize(lr, hr.shape, order=3, preserve_range=True,
                    anti_aliasing=False)

        sr, hr = sr.astype(np.float32), hr.astype(np.float32)
        numpy_psnr = calc_psnr_numpy(sr, hr, scale)
        torch_psnr = calc_psnr_torch(
            torch.from_numpy(sr).permute(2, 0, 1).unsqueeze(0),
            torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0),
            scale, use_gray_coeffs=False)
        self.assertAlmostEqual(numpy_psnr, torch_psnr, delta=1e-3)
