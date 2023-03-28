from typing import Tuple, Sequence

import einops
import torch

from ..utils.fft import construct_fftfreq_grid_2d


def get_phase_shifts_2d(
    shifts: torch.Tensor, image_shape: Tuple[int, int], rfft: bool = False
):
    """Generate a complex-valued array of phase shifts for 2D images.

    Parameters
    ----------
    shifts: torch.Tensor
        `(b, 2)` array of 2D shifts in the last two image dimensions (h, w).
    image_shape: Tuple[int, int]
        Height and width of 2D image(s) on which phase shifts will be applied.
    rfft: bool
        If `True` the phase shifts generated will be compatible with
        the non-redundant half DFT outputs of the FFT for real inputs from `rfft`.

    Returns
    -------
    phase_shifts: torch.Tensor
        `(b, h, w)` complex valued array of phase shifts for the fft or rfft
        of images with `image_shape`. Outputs are compatible with the DFT without
        fftshift applied if `rfft=False`.
    """
    fftfreq_grid = construct_fftfreq_grid_2d(
        image_shape=image_shape, rfft=rfft, device=shifts.device
    )  # (h, w, 2)
    shifts = einops.rearrange(shifts, '... shift -> ... 1 1 shift')
    angles = einops.reduce(
        -2 * torch.pi * fftfreq_grid * shifts, '... h w 2 -> ... h w', reduction='sum'
    )  # radians/cycle, cycles/sample, samples
    return torch.complex(real=torch.cos(angles), imag=torch.sin(angles))


def phase_shift_dfts_2d(
    dfts: torch.Tensor,
    image_shape: Tuple[int, int],
    shifts: torch.Tensor,
    rfft: bool = False,
    fftshifted: bool = False,
):
    if rfft is True and fftshifted is True:
        raise ValueError('Bad arguments: rfft cannot be fftshifted.')
    phase_shifts = get_phase_shifts_2d(shifts=shifts, image_shape=image_shape,
                                       rfft=rfft)
    if fftshifted:
        phase_shifts = torch.fft.fftshift(phase_shifts, dim=(-2, -1))
    return dfts * phase_shifts


def phase_shift_images_2d(images: torch.Tensor, shifts: torch.Tensor):
    image_shape = images.shape[-2:]
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = phase_shift_dfts_2d(
        images,
        image_shape=image_shape,
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    images = torch.fft.irfftn(images, dim=(-2, -1))
    return torch.real(images)
