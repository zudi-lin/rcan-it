import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ._utils import conv3x3, conv1x1, get_activation


class ResidualBase(nn.Module):
    def __init__(self, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__()
        self.sd = stochastic_depth
        if stochastic_depth:
            self.prob = prob
            self.multFlag = multFlag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        return self._forward_train(x, identity) if self.training \
            else self._forward_test(x, identity)

    def _forward_train(self, x, identity) -> torch.Tensor:
        if not self.sd: # no stochastic depth
            res = self._forward_res(x)
            return identity + res
        
        if torch.rand(1) < self.prob: # no skip
            for param in self.parameters():
                param.requires_grad = True
            res = self._forward_res(x)
            return identity + res

        # This block is skipped during training
        for param in self.parameters():
            param.requires_grad = False
        return identity

    def _forward_test(self, x, identity) -> torch.Tensor:
        res = self._forward_res(x)
        if self.sd and self.multFlag:
            res *= self.prob

        return identity + res

    def _forward_res(self, _) -> torch.Tensor:
        # Residual forward function should be
        # defined in child classes.
        raise NotImplementedError


class PreActBasicBlock(ResidualBase):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 act_mode: str = 'relu', prob: float = 1.0, multFlag: bool = True,
                 zero_inti_residual: bool = False, affine_init_w: float = 0.1,
                 **_) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes, affine_init_w)
        self.conv1 = conv3x3(planes, planes)

        self.aff2 = Affine2d(planes, affine_init_w)
        self.conv2 = conv3x3(planes, planes)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff2.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)

        return x


class PreActBasicBlockDW(ResidualBase):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 act_mode: str = 'relu', prob: float = 1.0, multFlag: bool = True,
                 zero_inti_residual: bool = False, affine_init_w: float = 0.1,
                 reduction: int = 8) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes, affine_init_w)
        self.conv1 = conv3x3(planes, planes, groups=planes)
        self.se1 = SEBlock(planes, reduction, act_mode)

        self.aff2 = Affine2d(planes, affine_init_w)
        self.conv2 = conv3x3(planes, planes, groups=planes)
        self.se2 = SEBlock(planes, reduction, act_mode)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff2.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.se1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.se2(x)

        return x


class PreActBottleneck(ResidualBase):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 act_mode: str = 'relu', prob: float = 1.0, multFlag: bool = True,
                 zero_inti_residual: bool = False, affine_init_w: float = 0.1,
                 **_) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes, affine_init_w)
        self.conv1 = conv1x1(planes, planes)

        self.aff2 = Affine2d(planes, affine_init_w)
        self.conv2 = conv3x3(planes, planes)

        self.aff3 = Affine2d(planes, affine_init_w)
        self.conv3 = conv1x1(planes, planes)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff3.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.aff3(x)
        x = self.act(x)
        x = self.conv3(x)

        return x


class MBConvBlock(ResidualBase):
    def __init__(self, planes: int, stochastic_depth: bool = False, act_mode: str = 'relu',
                 prob: float = 1.0, multFlag: bool = True, reduction: int = 8,
                 zero_inti_residual: bool = False, affine_init_w: float = 0.1) -> None:
        super().__init__(stochastic_depth, prob, multFlag)

        self.conv1 = conv1x1(planes, planes)
        self.aff1 = Affine2d(planes, affine_init_w)

        self.conv2 = conv3x3(planes, planes, groups=planes)  # depth-wise
        self.aff2 = Affine2d(planes, affine_init_w)
        self.se = SEBlock(planes, reduction, act_mode)

        self.conv3 = conv1x1(planes, planes)
        self.aff3 = Affine2d(planes, affine_init_w)
        self.act = get_activation(act_mode)

        if zero_inti_residual:
            nn.init.constant_(self.aff3.weight, 0.0)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.aff1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.aff2(x)
        x = self.act(x)

        x = self.se(x)

        x = self.conv3(x)
        x = self.aff3(x)  # no activation

        return x


class EDSRBlock(ResidualBase):
    def __init__(self, planes: int, bias: bool = True,  act_mode: str = 'relu',
                 res_scale: float = 0.1, res_scale_learnable: bool = False, 
                 stochastic_depth: bool = False, prob: float = 1.0, multFlag: bool = True, **_):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias))

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class RCANBlock(ResidualBase):
    def __init__(self, planes: int, bias: bool = True, act_mode: str = 'relu',
                 res_scale: float = 0.1, reduction: int = 16, res_scale_learnable: bool = False, 
                 stochastic_depth: bool = False, prob: float = 1.0, multFlag: bool = True, 
                 normal_init_std: Optional[float] = None, **_):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias),
            SEBlock(planes, reduction, act_mode))

        # normal initialization
        if normal_init_std is not None:
            for idx in [0, 2]:
                nn.init.normal_(self.body[idx].weight, 0.0, normal_init_std)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class RCANBlockDW(ResidualBase):
    """RCAN building block with depth-wise convolution for the second conv layer.
    """
    def __init__(self, planes: int, bias: bool = True, act_mode: str = 'relu',
                 res_scale: float = 0.1, reduction: int = 16, res_scale_learnable: bool = False, 
                 stochastic_depth: bool = False, prob: float = 1.0, multFlag: bool = True, **_):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias, groups=planes),
            SEBlock(planes, reduction, act_mode))

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class RCANBlockAllDW(ResidualBase):
    """RCAN building block with depth-wise convolution for all conv layers. An
    additional squeeze-and-excitation (SE) block is used for the cross-channel
    communication.  
    """
    def __init__(self, planes: int, bias: bool = True, act_mode: str = 'relu',
                 res_scale: float = 0.1, reduction: int = 16, res_scale_learnable: bool = False, 
                 stochastic_depth: bool = False, prob: float = 1.0, multFlag: bool = True, **_):
        super().__init__(stochastic_depth, prob, multFlag)
        if res_scale_learnable:
            self.res_scale = Parameter(torch.ones(1))
            nn.init.constant_(self.res_scale, res_scale)
        else:
            self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(planes, planes, bias=bias, groups=planes),
            SEBlock(planes, reduction, act_mode),
            get_activation(act_mode),
            conv3x3(planes, planes, bias=bias, groups=planes),
            SEBlock(planes, reduction, act_mode))

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x).mul(self.res_scale)
        return x


class SEBlock(nn.Module):
    def __init__(self, planes: int, reduction: int = 8, act_mode: str = 'relu'):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(planes, planes // reduction, kernel_size=1),
            get_activation(act_mode),
            nn.Conv2d(planes // reduction, planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Affine2d(nn.Module):
    def __init__(self, planes: int, init_w: float = 0.1) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(1, planes, 1, 1))
        self.bias = Parameter(torch.zeros(1, planes, 1, 1))
        nn.init.constant_(self.weight, init_w)

    def forward(self, x):
        return x * self.weight + self.bias


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, planes: int, act_mode: str = 'relu',
                 use_affine: bool = True):
        m = []
        if (scale & (scale - 1)) == 0:  # is power of 2
            if use_affine:
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv3x3(planes, 4 * planes))
                    m.append(nn.PixelShuffle(2))
                    m.append(Affine2d(planes))
                    m.append(get_activation(act_mode))
            else:
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv3x3(planes, 4 * planes, bias=True))
                    m.append(nn.PixelShuffle(2))
                    m.append(get_activation(act_mode))

        elif scale == 3:
            if use_affine:
                m.append(conv3x3(planes, 9 * planes))
                m.append(nn.PixelShuffle(3))
                m.append(Affine2d(planes))
                m.append(get_activation(act_mode))
            else:
                m.append(conv3x3(planes, 9 * planes, bias=True))
                m.append(nn.PixelShuffle(3))
                m.append(get_activation(act_mode))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
