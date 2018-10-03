from functools import lru_cache
import numpy as np
import torch as th


@lru_cache(maxsize=None)
def get_kernel_area(ker_loc=(0, 0), theta_c=np.pi / 12, theta=0):
    """
    Helper function to identify kernel area after rotate on the sphere.
    Assume circle center: $\mathbf{c}=(x_c, y_c, z_c)$, radius degree: $\theta_c$. Then, point $(\theta, \phi)$
    in equirectangular projection is on the circle i.f.f.
    $$x_c\sin\theta\cos\phi+y_c\sin\theta\sin\phi+z_c\cos\theta=\cos{\theta_c}$$
    We can rewrite above equation as
    $$\sin(\phi+\psi)=C$$
    where $a=x_c\sin\theta$, $b=y_c\sin\theta$, $c=z_c\cos\theta$, $d=\cos{\theta_c}$,
    $\sin\psi=\frac{a}{\sqrt{a^2+b^2}}$, $\cos\psi=\frac{b}{\sqrt{a^2+b^2}}$ and $C=\frac{d-c}{\sqrt{a^2+b^2}}$
    Given kernel location and theta, output a list of tuple representing valid phi ranges
    :param ker_loc: tuple of ($\theta$ and $\phi$)
    :param theta_c: kernel angle radius in rad
    :param theta: current $\theta$ location on sphere to indentify $\phi$ range
    :return: list of tuples ($\phi_{st}$, $\phi_{ed}$)
    """
    # if np.abs(theta - ker_loc[0]) <= theta_c and np.isclose(theta, 0) or np.isclose(theta, np.pi):
    #     return [(0, 2 * np.pi)]

    if np.abs(theta - ker_loc[0]) >= theta_c:
        return []

    def ang_sum(x, y):
        return (x + y) % (2 * np.pi)

    x_c, y_c, z_c = [np.sin(ker_loc[0]) * np.cos(ker_loc[1]), np.sin(ker_loc[0]) * np.sin(ker_loc[1]),
                     np.cos(ker_loc[0])]
    a, b, c, d = x_c * np.sin(theta), y_c * np.sin(theta), z_c * np.cos(theta), np.cos(theta_c)
    if np.isclose(ker_loc[0], 0) or np.isclose(ker_loc[0], np.pi):
        return [(0, 2 * np.pi)]
    elif np.isclose(theta, 0) or np.isclose(theta, np.pi):
        return []
    C = (d - c) / (np.sqrt(a ** 2 + b ** 2))
    if not (-1 <= C <= 1):
        return [(0, 2 * np.pi)]
    omega = np.arcsin(C)
    psi = np.arctan2(a, b)
    st, ed = ang_sum(omega, -psi), ang_sum(np.pi - omega, -psi)
    st, ed = min(st, ed), max(st, ed)
    if ker_loc[1] - st > 0 and ed - ker_loc[1] > 0:
        return [(st, ed)]
    elif ang_sum(ker_loc[1], np.pi) - st > 0 and ed - ang_sum(ker_loc[1], np.pi) > 0:
        print('warning!')
        return [(0, st), (ed, 2 * np.pi)]
    else:
        return []


@lru_cache(maxsize=None)
def gen_kernel_grid(ker_shape, ker_loc_theta, ker_radius, target_theta_sr, target_phi_sr, ker_loc_phi=None):
    """
    Given crown kernel centered on north pole, current kernel location and sampling rate of theta and phi,
    generate sampling grid w.r.t original kernel area.
    :param ker_shape: tuple of ($\theta$, $\phi$)
    :param ker_loc_theta: current kernel center location $\theta$
    :param ker_radius:
    :param target_theta_sr:
    :param target_phi_sr:
    :return: resampling grid
    """
    ker_theta_sr, ker_phi_sr = ker_shape[0] / ker_radius, ker_shape[1] / (2 * np.pi)
    target_theta_range = max(ker_loc_theta - ker_radius, 0), min(ker_loc_theta + ker_radius, np.pi)
    target_theta_grid = np.linspace(*target_theta_range, int(-np.subtract(*target_theta_range) * target_theta_sr))
    target_phi_ranges = []
    invalid_theta_idx = []
    for i, theta in enumerate(target_theta_grid):
        phi_range = get_kernel_area((ker_loc_theta, np.pi), ker_radius, theta)
        if not phi_range:
            if not (np.isclose(theta+ker_radius, ker_loc_theta) or np.isclose(theta-ker_radius, ker_loc_theta)):
                # print('target_theta: {}, kernel_rad: {}, ker_loc_theta: {}'.format(theta, ker_radius, ker_loc_theta))
                pass
            target_phi_ranges += [(0, 0)]
            invalid_theta_idx.append(i)
        elif len(phi_range) == 2:
            print(phi_range)
            raise NotImplementedError()
        else:
            target_phi_ranges += phi_range
    target_phi_ranges_arr = np.array(target_phi_ranges)
    target_phi_range = target_phi_ranges_arr[np.argmax(target_phi_ranges_arr[:, 1] - target_phi_ranges_arr[:, 0])]
    ker_area = np.sum((target_phi_ranges_arr[:, 1] - target_phi_ranges_arr[:, 0]) * target_phi_sr)
    target_phi_grid = np.linspace(*target_phi_range, int(-np.subtract(*target_phi_range) * target_phi_sr))
    target_theta_grid, target_phi_grid = np.meshgrid(target_theta_grid, target_phi_grid, indexing='ij')
    target_grid_xyz = np.stack([np.sin(target_theta_grid) * np.cos(target_phi_grid),
                                np.sin(target_theta_grid) * np.sin(target_phi_grid),
                                np.cos(target_theta_grid)], axis=2)
    Y = np.array([[np.cos(ker_loc_theta), 0, -np.sin(ker_loc_theta)],
                  [0, 1, 0],
                  [np.sin(ker_loc_theta), 0, np.cos(ker_loc_theta)]])
    if not ker_loc_phi:
        ker_loc_phi = np.pi
    Z = np.array([[np.cos(ker_loc_phi), np.sin(ker_loc_phi), 0],
                  [-np.sin(ker_loc_phi), np.cos(ker_loc_phi), 0],
                  [0, 0, 1]])
    ker_grid_xyz = target_grid_xyz.reshape(-1, 3).dot(Z.T).dot(Y.T).reshape(target_grid_xyz.shape)
    ker_grid_xyz[ker_grid_xyz > 1] = 1
    ker_grid_xyz[ker_grid_xyz < -1] = -1
    ker_grid = np.stack([np.arctan2(ker_grid_xyz[..., 1], ker_grid_xyz[..., 0]) % (2 * np.pi) * ker_phi_sr /
                         ker_shape[1] * 2 - 1, np.arccos(ker_grid_xyz[..., 2]) * ker_theta_sr / ker_shape[0] * 2 - 1],
                         axis=2)
    for i in invalid_theta_idx:
        ker_grid[i, :, :] = -3
    return th.from_numpy(ker_grid[:, :, :]), ker_area  # , target_theta_range, target_phi_ranges
