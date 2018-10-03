from torch import nn
from sconv.functional import spherical_pad


class SphericalPad(nn.Module):
    def __init__(self, pad=(0, 0, 0, 0)):
        super(SphericalPad, self).__init__()
        self.pad = pad

    def forward(self, data):
        spherical_pad(data, self.pad)
