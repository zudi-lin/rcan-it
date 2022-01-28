import unittest

import torch
from ptsr.config import get_cfg_defaults
from ptsr.model import Model
from ptsr.model.common import *
from ptsr.model._utils import conv3x3

class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def test_layers(self):
        """Test regular and deformable convolution layers.
        """
        c = 64
        input_size = (4, c, 32, 32)
        x = torch.rand(input_size).to(self.device)
        for df_conv in [False, True]:
            m = conv3x3(c, c, bias=True, df_conv=df_conv).to(self.device)
            self.assertTupleEqual(input_size, m(x).size())

    def test_building_blocks(self):
        c, act_mode = 64, 'elu'
        input_size = (4, c, 32, 32)
        x = torch.rand(input_size).to(self.device)
        for block in [
                PreActBasicBlock, PreActBottleneck, MBConvBlock, PreActBasicBlockDW,
                EDSRBlock, RCANBlock, RCANBlockDW, RCANBlockAllDW]:
            m = block(planes=c, act_mode=act_mode).to(self.device)
            self.assertTupleEqual(input_size, m(x).size())

    def test_model(self):
        def _get_input(size, bs=4):
            lr_size = (bs, 3, size, size)
            size *= cfg.DATASET.DATA_SCALE[0]
            sr_size = (bs, 3, size, size)
            x = torch.rand(lr_size).to(self.device)
            return x, sr_size

        cfg = get_cfg_defaults()
        for scale in [3, 4]:
            cfg.DATASET.DATA_SCALE = [scale]
            x, sr_size = _get_input(64)
            model = Model(cfg).eval().to(self.device)
            with torch.no_grad():
                output = model(x)
            self.assertTupleEqual(sr_size, output.size())

        # test self-ensemble
        cfg.DATASET.DATA_SCALE = [4]
        cfg.MODEL.SELF_ENSEMBLE = True
        x, sr_size = _get_input(64)
        model = Model(cfg).eval().to(self.device)
        with torch.no_grad():
            output = model(x)
        self.assertTupleEqual(sr_size, output.size())

        # test_forward_patch
        cfg.MODEL.SELF_ENSEMBLE = False
        cfg.DATASET.DATA_SCALE = [2]
        cfg.DATASET.CHOP = True
        x, sr_size = _get_input(1000, bs=1)
        model = Model(cfg).eval().to(self.device)
        with torch.no_grad():
            output = model(x)
        self.assertTupleEqual(sr_size, output.size())
