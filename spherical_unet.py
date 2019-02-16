# image size: 150 * 300
# image rad delta: 180 deg / 128 = 1.4 deg
# pool1 rad delta: 180 deg / 64 = 2.8 deg
# pool2 rad delta: 180 deg / 32 = 5.6 deg
# ---------------------- arch ------------------------ #
#       128 * 256
# conv(3, 64, pi/32, (8, 16)) -> bn -> relu -> pool(2) -> conv(64, 128, pi/16, (4, 8)) -> bn -> relu -> pool(2) ->
#
# conv(128, 256, pi/4, (4, 8)) -> bn -> relu -> pool(2) -> conv(256, 256, pi, (4, 8)) -> bn -> relu ->
#
# conv(256 + 256, 128, pi/4, (4, 8)) -> bn -> relu -> conv(128 + 128, 64, pi/16, (4, 8)) -> bn -> relu ->
#
# conv(64 + 64, 1, pi/32, (8, 16)) -> bn -> relu -> conv(1, 1, (5, 5))

from sconv.module import SphericalConv, SphereMSE
from torch import nn
import numpy as np
import torch as th
from torch.autograd import Variable

class Final1(nn.Module):
    def __init__(self):
        super(Final1, self).__init__()
        self.conv1 = SphericalConv(4, 64, np.pi/32, kernel_size=(8, 16), kernel_sr=None)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = SphericalConv(64, 128, np.pi/16, kernel_size=(4, 8), kernel_sr=None)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = SphericalConv(128, 256, np.pi / 4, kernel_size=(4, 8), kernel_sr=None)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = SphericalConv(256, 256, np.pi, kernel_size=(8, 16), kernel_sr=None)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv5 = SphericalConv(256 + 256, 128, np.pi / 4, kernel_size=(4, 8), kernel_sr=None)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv6 = SphericalConv(128 + 128, 64, np.pi / 16, kernel_size=(4, 8), kernel_sr=None)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(inplace=True)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv7 = SphericalConv(64 + 64, 1, np.pi / 32, kernel_size=(4, 8), kernel_sr=None)

    def forward(self, image, last):
        c1 = self.conv1(th.cat([image, last], dim=1))
        b1 = self.bn1(c1)
        r1 = self.relu1(b1)
        p1 = self.pool1(r1)

        c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        r2 = self.relu2(b2)
        p2 = self.pool2(r2)

        c3 = self.conv3(p2)
        b3 = self.bn3(c3)
        r3 = self.relu3(b3)
        p3 = self.pool3(r3)

        c4 = self.conv4(p3)
        b4 = self.bn4(c4)
        r4 = self.relu4(b4)

        r4u = self.up1(r4)
        c5 = self.conv5(th.cat([r4u, r3], dim=1))
        b5 = self.bn5(c5)
        r5 = self.relu5(b5)

        r5u = self.up2(r5)
        c6 = self.conv6(th.cat([r5u, r2], dim=1))
        b6 = self.bn6(c6)
        r6 = self.relu6(b6)

        r6u = self.up3(r6)
        c7 = self.conv7(th.cat([r6u, r1], dim=1))

        return c7
