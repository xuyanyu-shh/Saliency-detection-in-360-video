from torch.autograd.function import once_differentiable
from torch.autograd import Function


def sphere_pad(image, pad=(0, 0, 0, 0)):
    """
    Pad equirectangular projected spherical image for 2D convolution
    """
    assert 0 <= pad[0] <= image.size(3) and 0 <= pad[1] <= image.size(3) and 0 <= pad[2] <= image.size(2) and 0 <= pad[3] <= image.size(2)
    size = list(image.size())
    size[3] = size[3] + pad[0] + pad[1]
    output = image.new(*size).zero_()
    if pad[1]:
        output[:, :, :, pad[0]:-pad[1]].copy_(image)
    else:
        output[:, :, :, pad[0]:].copy_(image)
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
        if pad[1]:
            grad_image_padded[:, :, :, -pad[1]-pad[0]:-pad[1]] += grad_image_padded[:, :, :, :pad[0]]
        else:
            grad_image_padded[:, :, :, -pad[1] - pad[0]:] += grad_image_padded[:, :, :, :pad[0]]

    grad_image = grad_image_padded[:, :, :, pad[0]:grad_image_padded.size(3)-pad[1]]

    return grad_image


class SphericalPad(Function):
    @staticmethod
    def forward(ctx, image, pad):
        ctx.pad = pad
        return sphere_pad(image, pad)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        pad = ctx.pad
        return sphere_grad_unpad(grad_output, pad), None


spherical_pad = SphericalPad.apply
