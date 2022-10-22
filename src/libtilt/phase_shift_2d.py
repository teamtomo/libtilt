from typing import Tuple

import einops
import torch


def get_phase_shifts_2d(shifts: torch.Tensor, image_shape: Tuple[int, int]):
    """Generate a complex-valued array of phase shifts for 2D images.

    Parameters
    ----------
    shifts: torch.Tensor
        (n, 2) array of xy shifts where xy coordinates index into
        dims 1 and 0 of 2D images respectively.
    image_shape: Tuple[int, int]
        shape of 2D image(s) onto which phase shifts will be applied.
    """
    x = torch.fft.fftshift(torch.fft.fftfreq(image_shape[1]))
    y = torch.fft.fftshift(torch.fft.fftfreq(image_shape[0]))
    yy = einops.repeat(y, 'h -> h w', w=image_shape[1])
    xx = einops.repeat(x, 'w -> h w', h=image_shape[0])
    x_shifts = einops.rearrange(shifts[:, 0], 'b -> b 1 1')
    y_shifts = einops.rearrange(shifts[:, 1], 'b -> b 1 1')
    factors = -2 * torch.pi * (x_shifts * xx + y_shifts * yy)
    return torch.cos(factors) + 1j * torch.sin(factors)


def fourier_shift_dfts_2d(dfts: torch.Tensor, shifts: torch.Tensor):
    phase_shifts = get_phase_shifts_2d(shifts=shifts, image_shape=dfts.shape[-2:])
    return dfts * phase_shifts


def phase_shift_images_2d(images: torch.Tensor, shifts: torch.Tensor):
    images = torch.fft.fftn(images, dim=(-2, -1))
    images = torch.fft.fftshift(images, dim=(-2, -1))
    images = fourier_shift_dfts_2d(images, shifts)
    images = torch.fft.ifftshift(images, dim=(-2, -1))
    images = torch.fft.ifftn(images, dim=(-2, -1))
    return torch.real(images)
