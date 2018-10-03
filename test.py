from sconv.module.sconv import SphericalConv
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch as th
import time


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = SphericalConv(3, 64, np.pi/150 * 7, kernel_size=(4, 8), kernel_sr=None)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = SphericalConv(64, 128, np.pi/75 * 5, kernel_size=(4, 8), kernel_sr=None)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = SphericalConv(128, 128, np.pi / 37 * 5, kernel_size=(4, 8), kernel_sr=None)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SphericalConv(128, 128, np.pi / 37 * 5, kernel_size=(4, 8), kernel_sr=None)

    def forward(self, image):
        out = self.conv1(image)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)

        return out


if __name__ == '__main__':
    from distutils.version import LooseVersion
    import sys
    assert sys.version_info >= (3, 6), \
        f"Python version greater than 3.6 is required, current version is {sys.version}."
    assert LooseVersion(th.__version__) < LooseVersion('0.4.0'), \
        f"Torch version greater than 0.3.1 is not supported, current version is {th.__version__}."
    test = TestModel().cuda()
    t = 0
    for i in range(10):
        img = Variable(th.rand(1, 3, 128, 256).cuda(), requires_grad=True)
        tic = time.time()
        out = test(img)
        out.sum().backward()
        t += time.time() - tic

    print('fp+bp gpu time: {}'.format(t/10))
