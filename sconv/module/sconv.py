import torch as th
import torch.nn as nn
from sconv.functional import spherical_conv
from numpy import pi
import math
from torch.nn import Parameter


class SphericalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_rad=pi/12, kernel_sr=(15/(pi/12), 30/(2*pi)), kernel_size=None,
                 stride=(1, 1), padding='valid', padding_mode='sphere', bias=False):
        """
        The Spherical Convolution module.
        :param in_channels: Specify number of input channels
        :param out_channels: Specify number of output channels
        :param kernel_rad: The radiation of the spherical kernel, i.e. the great-circle distance from center of the
        crown kernel to its border on an unit spherical.
        :param kernel_sr: Specify the sampling rate $(sr_\theta, sr_\phi)$ of the spherical kernel on equirectangular
        panorama, so that kernel size of sampled spherical kernel can be infered. Note that this parameter is exclude
        with kernel_size.
        :param kernel_size: Directly specify the size $(s_\theta, s_phi)$ of sampled spherical kernel. Note that this
        parameter is exclude with kernel_sr.
        :param stride: Convolution stride on panorama feature map. Currently stride size greater than 1 has not been
        implemented yet.
        :param padding: Must be 'valid', which makes sure that padded feature map is valid for given kernel size
        and stride size.
        :param padding_mode: Can be 'sphere', 'reflect', 'replicate' or a positive integer.
        'sphere': Spherical padding.
        'reflect', 'replicate' and 'replicate': Have the similar behavior as in planner convolution.
        positive integer: Constant padding with specified value.
        :param bias: Whether use bias.
        """
        super(SphericalConv, self).__init__()
        assert not (kernel_sr and kernel_size) and (kernel_sr or kernel_size)
        if stride != (1, 1):
            raise NotImplementedError("stride size other than (1, 1) has not been implemented yet.")
        if kernel_sr:
            self.kernel_sr = kernel_sr
            self.kernel_size = int(kernel_sr[0] * kernel_rad + 0.5), int(kernel_sr[1] * 2 * pi + 0.5)
        if kernel_size:
            self.kernel_sr = kernel_size[0] / kernel_rad, kernel_size[1] / (2 * pi)
            self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_rad = kernel_rad
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.weight = Parameter(th.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(th.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, kernel_rad={kernel_rad}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, image):
        _, _, h, w = image.size()
        assert pi / h < 2 * self.kernel_rad and 2 * pi / w < 2 * self.kernel_rad, 'kernel rad is too small'
        return spherical_conv(image, self.weight, bias=self.bias, stride=self.stride, kernel_rad=self.kernel_rad,
                              padding=self.padding, padding_mode=self.padding_mode)
