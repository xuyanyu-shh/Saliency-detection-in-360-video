import numbers
import numpy as np
import torch as th
import torch._thnn as thnn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
from .common import gen_kernel_grid


def kernel_resampling_forward(kernel, grid):
    grid_sz = grid.size()
    backend = thnn.type2backend[kernel.type()]
    assert kernel.dim() == 4 and grid.dim() == 3, 'invalid kernel/grid dimension'
    grid = grid.unsqueeze(0).expand(kernel.size(0), *grid.size()).contiguous()
    output = kernel.new(grid_sz[0], kernel.size(1), grid_sz[1], grid_sz[2])
    backend.SpatialGridSamplerBilinear_updateOutput(backend.library_state, kernel, grid, output, 0)

    return output


def kernel_resampling_backward(kernel, grid, grad_output):
    backend = thnn.type2backend[kernel.type()]
    grad_kernel = kernel.new(kernel.size())
    grad_grid = grid.new(grid.size())
    grid = grid.unsqueeze(0).expand(kernel.size(0), *grid.size()).contiguous()
    backend.SpatialGridSamplerBilinear_updateGradInput(
        backend.library_state, kernel, grad_kernel, grid, grad_grid, grad_output, 0)

    return grad_kernel, None


def conv2d_updateOutput(image, kernel, stride, padding, output, bias=None):
    """
    Expose PyTorch conv2d forward function
    """
    _backend = thnn.type2backend[type(image)]
    finput = image.new()
    fgradInput = th.zeros(1).type_as(image)
    och, ich, kh, kw = kernel.shape
    kernel = kernel.view(och, -1).contiguous()
    image = image.contiguous()
    _backend.SpatialConvolutionMM_updateOutput(
        _backend.library_state,
        image,
        output,
        kernel,
        bias,
        finput,
        fgradInput,
        kw, kh,
        stride[0], stride[1],
        padding[0], padding[1]
    )
    kernel.view((och, ich, kh, kw))

    return finput


def conv2d_updateGradInput(grad_output, image, kernel, grad_image, grad_kernel, stride, padding, finput, och, ich, kh, kw, grad_bias=None):
    """
    Expose PyTorch conv2d backward function
    """
    _backend = thnn.type2backend[type(image)]
    fgradInput = kernel.new()
    _backend.SpatialConvolutionMM_updateGradInput(
        _backend.library_state,
        image,
        grad_output,
        grad_image,
        kernel,
        finput,
        fgradInput,
        kw, kh,
        stride[0], stride[1],
        padding[0], padding[1]
    )
    _backend.SpatialConvolutionMM_accGradParameters(
        _backend.library_state,
        image,
        grad_output,
        grad_kernel,
        grad_bias,
        finput,
        fgradInput,
        kw, kh,
        stride[0], stride[1],
        padding[0], padding[1],
        1
    )


def conv2d_updateGradParam(grad_output, image, grad_kernel, stride, padding, grad_bias=None, finput=None):
    """
    Expose PyTorch conv2d backward function
    """
    _backend = thnn.type2backend[type(image)]
    image = image.contiguous()
    och, ich, kh, kw = grad_kernel.size()
    grad_kernel = grad_kernel.view(och, -1).contiguous()
    grad_output = grad_output.contiguous()
    _backend.SpatialConvolutionMM_accGradParameters(
        _backend.library_state,
        image,
        grad_output,
        grad_kernel,
        grad_bias,
        finput,
        image.new(),
        kw, kh,
        stride[0], stride[1],
        padding[0], padding[1],
        1
    )


def sphere_pad(image, pad=(0, 0, 0, 0)):
    """
    Pad equirectangular projected spherical image for 2D convolution
    """
    assert 0 <= pad[0] <= image.size(3) and 0 <= pad[1] <= image.size(3) and 0 <= pad[2] <= image.size(2) and 0 <= pad[3] <= image.size(2)
    size = list(image.size())
    size[3] = size[3] + pad[0] + pad[1]
    output = image.new(*size).zero_()
    output[:, :, :, pad[0]:-pad[1]].copy_(image)
    if pad[0]:
        output[:, :, :, :pad[0]].copy_(output[:, :, :, -pad[0]-pad[1]:output.size(3)-pad[1]])
    if pad[1]:
        output[:, :, :, -pad[1]:].copy_(output[:, :, :, pad[0]:pad[1]+pad[0]])

    return output


def sphere_grad_unpad(grad_image_padded, pad=(0, 0, 0, 0)):
    grad_image_padded = grad_image_padded.clone()
    if pad[1]:
        grad_image_padded[:, :, :, pad[0]:pad[0]+pad[1]] += grad_image_padded[:, :, :, -pad[1]:]
    if pad[0]:
        grad_image_padded[:, :, :, -pad[1]-pad[0]:-pad[1]] += grad_image_padded[:, :, :, :pad[0]]

    grad_image = grad_image_padded[:, :, :, pad[0]:grad_image_padded.size(3)-pad[1]]

    return grad_image


class SphericalConv(Function):
    @staticmethod
    def _conv_line_forward(
            image, kernel, kernel_grid, bias=None, stride=(1, 1), padding='valid', padding_mode='sphere', area=None
    ):
        # resampling kernel
        kernel_resampled = kernel_resampling_forward(kernel, kernel_grid)
        # print(kernel_resampled.shape[2:])
        # print(area)
        # cv2.imshow('line', image[0, 0].cpu().numpy())
        # cv2.imshow('ker', kernel_resampled[0, 0].cpu().numpy())
        # cv2.waitKey(-1)
        out_line = th.zeros(image.size(0), kernel_resampled.size(0), 1, int((image.size(3) - kernel_resampled.size(3)) / stride[0] + 1)).type_as(image)
        assert image.size(2) == kernel_resampled.size(2)
        finput = conv2d_updateOutput(image, kernel_resampled, stride, (0, 0), out_line, bias=bias)
        assert out_line.size(2) == 1
        och, ich, kh, kw = kernel_resampled.size()
        return out_line/area, finput

    @staticmethod
    def _conv_line_backward(
            grad_out, image, kernel, kernel_grid, finput, bias=None, stride=(1, 1), padding='valid', padding_mode='sphere', area=None,
    ):
        # resampling kernel
        kernel_resampled = kernel_resampling_forward(kernel, kernel_grid)

        grad_out = grad_out.contiguous()
        grad_image = image.new(image.size()).zero_().contiguous()
        grad_kernel_resampled = kernel_resampled.new(kernel_resampled.size()).zero_().contiguous()
        grad_bias = bias.new(bias.size()).zero_() if bias is not None else None

        # then perform conv2d backward
        och, ich, kh, kw = kernel_resampled.size()
        kernel_resampled = kernel_resampled.view(och, -1)
        conv2d_updateGradInput(grad_out, image, kernel_resampled, grad_image, grad_kernel_resampled, stride, (0, 0), finput,
                               och, ich, kh, kw, grad_bias)
        # and resampling backward
        grad_kernel, _ = kernel_resampling_backward(kernel, kernel_grid, grad_kernel_resampled)

        return grad_image/area, grad_kernel/area, grad_bias

    @staticmethod
    # bias is an optional argument
    def forward(ctx, image, kernel, bias=None, stride=(1, 1), padding='valid', padding_mode='sphere', kernel_rad=np.pi/3):
        ctx.save_for_backward(image, kernel, bias)
        finputs = []
        kernel_grids = []
        ker_areas = []
        output = th.zeros(image.size(0), kernel.size(0), int(image.size(2) / stride[1]),
                          int(image.size(3) / stride[0])).type_as(image)
        # first pad if necessary
        pad = [0, 0, 0, 0]
        if padding == 'valid':
            for theta_idx_ct_in in range(0, image.size(2), stride[1]):
                target_theta_sr = image.size(2) / np.pi
                target_phi_sr = image.size(3) / (2 * np.pi)
                ker_loc_theta = theta_idx_ct_in / target_theta_sr
                kernel_grid, ker_area = gen_kernel_grid(kernel.shape, ker_loc_theta, kernel_rad, target_theta_sr,
                                              target_phi_sr)
                kernel_grid = kernel_grid.type_as(kernel)
                kernel_grids.append(kernel_grid)
                ker_areas.append(ker_area)
                padh, padw = int(kernel_grid.size(0) - 1), int(kernel_grid.size(1) - 1)
                assert padh >= 0 and padw >= 0
                padl, padr, padt, padb = int(padw / 2 + 0.5), int(padw / 2), int(padh / 2 + 0.5), int(padh / 2)
                pad[0], pad[1], pad[2], pad[3] = max(pad[0], padl), max(pad[1], padr), max(pad[2], padt), max(
                    pad[3], padb)
        else:
            raise NotImplementedError('padding')
        pad = tuple(pad)
        if padding_mode == 'sphere':
            image_padded = sphere_pad(image, pad)
        elif padding_mode == 'reflect':
            new_pad = list(pad)
            new_pad[2], new_pad[3] = 0, 0
            new_pad = tuple(new_pad)
            image_padded = F.pad(image, new_pad, 'reflect').data
        elif padding_mode == 'replicate':
            new_pad = list(pad)
            new_pad[2], new_pad[3] = 0, 0
            new_pad = tuple(new_pad)
            image_padded = F.pad(image, new_pad, 'replicate').data
        elif isinstance(padding_mode, numbers.Real):
            new_pad = list(pad)
            new_pad[2], new_pad[3] = 0, 0
            new_pad = tuple(new_pad)
            image_padded = F.pad(image, new_pad, 'constant', padding_mode).data
        else:
            raise ValueError('unknown padding mode {}, expect one of "sphere", "reflect", '
                             '"replicate" or real number for constant padding')
        theta_st_idx = 0
        for (theta_idx_out, theta_idx_ct_in), kernel_grid, ker_area in zip(enumerate(range(0, image.size(2), stride[1])), kernel_grids, ker_areas):
            target_theta_sr = image.size(2) / np.pi
            ker_loc_theta = theta_idx_ct_in / target_theta_sr
            padw = int(kernel_grid.size(1) - 1)
            padl, padr = int(padw / 2 + 0.5), int(padw / 2)
            phi_st, phi_ed = pad[0] - padl, image_padded.size(3) - (pad[1] - padr)
            if ker_loc_theta - kernel_rad < 0:
                out_line, finput = SphericalConv._conv_line_forward(
                    image_padded[:, :, :kernel_grid.size(0), phi_st:phi_ed], kernel,
                    kernel_grid, bias=bias, stride=stride, area=ker_area
                )
            elif ker_loc_theta - kernel_rad > np.pi:
                out_line, finput = SphericalConv._conv_line_forward(
                    image_padded[:, :, -kernel_grid.size(0):, phi_st:phi_ed], kernel,
                    kernel_grid, bias=bias, stride=stride, area=ker_area
                )
            else:
                out_line, finput = SphericalConv._conv_line_forward(
                    image_padded[:, :, theta_st_idx:theta_st_idx+kernel_grid.size(0), phi_st:phi_ed], kernel,
                    kernel_grid, bias=bias, stride=stride, area=ker_area
                )
                theta_st_idx += 1
            output[:, :, theta_idx_out, :].copy_(out_line.squeeze(2))
            finputs.append(finput)

        ctx.finputs = finputs
        ctx.kernel_grids = kernel_grids
        ctx.phi_ranges = ker_areas
        ctx.param = stride, padding, padding_mode, kernel_rad

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        image, kernel, bias = ctx.saved_tensors
        finputs = ctx.finputs
        stride, padding, padding_mode, kernel_rad = ctx.param
        kernel_grids = ctx.kernel_grids
        ker_areas = ctx.phi_ranges

        # first pad if necessary
        pad = [0, 0, 0, 0]
        if padding == 'valid':
            for kernel_grid in kernel_grids:
                padh, padw = int(kernel_grid.size(0) - 1), int(kernel_grid.size(1) - 1)
                assert padh >= 0 and padw >= 0
                padl, padr, padt, padb = int(padw / 2 + 0.5), int(padw / 2), int(padh / 2 + 0.5), int(padh / 2)
                pad[0], pad[1], pad[2], pad[3] = max(pad[0], padl), max(pad[1], padr), max(pad[2], padt), max(
                    pad[3], padb)
        else:
            raise NotImplementedError('padding')
        pad = tuple(pad)
        if padding_mode == 'sphere':
            image_padded = sphere_pad(image, pad)
        elif padding_mode == 'reflect':
            new_pad = list(pad)
            new_pad[2], new_pad[3] = 0, 0
            new_pad = tuple(new_pad)
            image_padded = F.pad(image, new_pad, 'reflect').data
        elif padding_mode == 'replicate':
            new_pad = list(pad)
            new_pad[2], new_pad[3] = 0, 0
            new_pad = tuple(new_pad)
            image_padded = F.pad(image, new_pad, 'replicate').data
        elif isinstance(padding_mode, numbers.Real):
            new_pad = list(pad)
            new_pad[2], new_pad[3] = 0, 0
            new_pad = tuple(new_pad)
            image_padded = F.pad(image, new_pad, 'constant', padding_mode).data
        else:
            raise ValueError('unknown padding mode {}, expect one of "sphere", "reflect", '
                             '"replicate" or real number for constant padding')

        grad_kernel = kernel.new(kernel.size()).zero_()
        grad_image_padded = image_padded.new(image_padded.size()).zero_()
        grad_bias = bias.new(bias.size()).zero_() if bias is not None else None

        theta_st_idx = 0
        for (theta_idx_out, theta_idx_ct_in), finput, kernel_grid, ker_area in zip(enumerate(range(0, image.size(2), stride[1])), finputs, kernel_grids, ker_areas):
            target_theta_sr = image.size(2) / np.pi
            ker_loc_theta = theta_idx_ct_in / target_theta_sr
            padw = int(kernel_grid.size(1) - 1)
            padl, padr = int(padw / 2 + 0.5), int(padw / 2)
            phi_st, phi_ed = pad[0] - padl, image_padded.size(3) - (pad[1] - padr)
            if ker_loc_theta - kernel_rad < 0:
                grad_in_line, grad_kernel_line, grad_bias_line = SphericalConv._conv_line_backward(
                    grad_output[:, :, theta_idx_out, :].unsqueeze(2),
                    image_padded[:, :, :kernel_grid.size(0), phi_st:phi_ed],
                    kernel, kernel_grid, finput, bias=bias, stride=stride,
                    padding=padding, padding_mode=padding_mode, area=ker_area)
                grad_image_padded[:, :, :kernel_grid.size(0), phi_st:phi_ed] += grad_in_line
                grad_kernel += grad_kernel_line
                if grad_bias is not None:
                    grad_bias += grad_bias_line
            elif ker_loc_theta + kernel_rad > np.pi:
                grad_in_line, grad_kernel_line, grad_bias_line = SphericalConv._conv_line_backward(
                    grad_output[:, :, theta_idx_out, :].unsqueeze(2),
                    image_padded[:, :, -kernel_grid.size(0):, phi_st:phi_ed],
                    kernel, kernel_grid, finput, bias=bias, stride=stride,
                    padding=padding, padding_mode=padding_mode, area=ker_area)
                grad_image_padded[:, :, -kernel_grid.size(0):, phi_st:phi_ed] += grad_in_line
                grad_kernel += grad_kernel_line
                if grad_bias is not None:
                    grad_bias += grad_bias_line
            else:
                grad_in_line, grad_kernel_line, grad_bias_line = SphericalConv._conv_line_backward(
                    grad_output[:, :, theta_idx_out, :].unsqueeze(2),
                    image_padded[:, :, theta_st_idx:theta_st_idx+kernel_grid.size(0), phi_st:phi_ed],
                    kernel, kernel_grid, finput, bias=bias, stride=stride,
                    padding=padding, padding_mode=padding_mode, area=ker_area)
                grad_image_padded[:, :, theta_st_idx:theta_st_idx+kernel_grid.size(0), phi_st:phi_ed] += grad_in_line
                grad_kernel += grad_kernel_line
                if grad_bias is not None:
                    grad_bias += grad_bias_line
                theta_st_idx += 1

        if padding_mode == 'sphere':
            grad_image = sphere_grad_unpad(grad_image_padded, pad)
        elif padding_mode == 'reflect':
            grad_image = grad_image_padded[:, :, pad[2]:-pad[3], pad[0]:-pad[1]]
        elif padding_mode == 'replicate':
            grad_image = grad_image_padded[:, :, pad[2]:-pad[3], pad[0]:-pad[1]]
        elif isinstance(padding_mode, numbers.Real):
            grad_image = grad_image_padded[:, :, pad[2]:-pad[3], pad[0]:-pad[1]]
        else:
            raise ValueError('unknown padding mode {}, expect one of "sphere", "reflect", '
                         '"replicate" or real number for constant padding')

        return grad_image, grad_kernel, grad_bias, None, None, None, None


def spherical_conv(image, kernel, bias=None, stride=(1, 1), padding='valid', padding_mode='sphere', kernel_rad=np.pi/3):

    return SphericalConv.apply(image, kernel, bias, stride, padding, padding_mode, kernel_rad)
