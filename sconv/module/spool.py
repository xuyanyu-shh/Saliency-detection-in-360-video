import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from sconv.functional.common import get_kernel_area
from sconv.functional.spad import spherical_pad


class SphericalPooling(nn.Module):
    def __init__(self):
        super(SphericalPooling, self).__init__()

    def forward(self, image):
        theta_in_sr = image.size(2) / np.pi
        phi_in_sr = image.size(3) / np.pi / 2
        ker_rad = 1 / theta_in_sr
        out_lines = []
        for theta_in_idx in range(0, image.size(2), 2):
            theta_in = theta_in_idx / theta_in_sr + 1 / theta_in_sr / 2
            img_in = image[:, :, theta_in_idx:theta_in_idx+2, :]
            assert img_in.size(2) == 2
            phi_range = get_kernel_area((theta_in, np.pi), ker_rad, (theta_in_idx + 0.5) / theta_in_sr)
            assert len(phi_range) == 1
            phi_range = phi_range[0][1] - phi_range[0][0]
            assert phi_range > 0
            phi_idx_range = int(min([max([2, np.round(phi_range * phi_in_sr)]), image.size(3)]))
            img_padded = spherical_pad(img_in, (int((phi_idx_range - 2) / 2 + 0.5), int((phi_idx_range - 2) / 2), 0, 0))
            out_lines.append(F.max_pool2d(img_padded, (2, phi_idx_range), (2, 2)))
            assert out_lines[-1].size(2) == 1 and out_lines[-1].size(3) == image.size(3) // 2

        return th.cat(out_lines, dim=2)
