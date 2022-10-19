from typing import Tuple

import einops
import torch


def get_phase_shifts_2d(shifts: torch.Tensor, image_shape: Tuple[int, int]):
    """

    Parameters
    ----------
    shifts: torch.Tensor
        (n, 2) array of yx shifts.
    image_shape: Tuple[int, int]
        shape of images onto which phase shifts will be applied.
    """
    y, x = (
        torch.arange(image_shape[0]) - image_shape[0] // 2,
        torch.arange(image_shape[1]) - image_shape[1] // 2,
    )
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    y_shifts = einops.rearrange(shifts[:, 0], 'b -> b 1 1')
    x_shifts = einops.rearrange(shifts[:, 1], 'b -> b 1 1')
    factors = -2 * torch.pi * (y_shifts * yy / image_shape[0] +
                               x_shifts * xx / image_shape[1])
    return torch.cos(factors) + 1j * torch.sin(factors)


def phase_shift_2d(dft: torch.Tensor, shifts: torch.Tensor):
    image_shape = dft.shape[-2:]
    phase_shifts = get_phase_shifts_2d(shifts=shifts, image_shape=image_shape)
    return dft * phase_shifts