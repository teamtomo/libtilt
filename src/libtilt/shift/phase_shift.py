from typing import Tuple, Sequence

import einops
import torch


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
        `(b, h, w)` array of phase shifts for the fft or rfft of images with `image_shape`.
        Outputs are compatible with the DFT without fftshift applied if `rfft=False`.
    """
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    dft_shape = _rfft_shape_from_input_shape(image_shape) if rfft is True else image_shape
    x = last_axis_frequency_func(image_shape[-1])
    y = torch.fft.fftfreq(image_shape[-2])
    xx = einops.repeat(x, 'w -> h w', h=dft_shape[-2])
    yy = einops.repeat(y, 'h -> h w', w=dft_shape[-1])
    x_shifts = einops.rearrange(shifts[:, 1], 'b -> b 1 1')
    y_shifts = einops.rearrange(shifts[:, 0], 'b -> b 1 1')
    factors = -2 * torch.pi * (x_shifts * xx + y_shifts * yy)
    return torch.cos(factors) + 1j * torch.sin(factors)


def fourier_shift_dfts_2d(
        dfts: torch.Tensor,
        image_shape: Tuple[int, int],
        shifts: torch.Tensor,
        rfft: bool = False,
        spectrum_is_fftshifted: bool = False,
):
    if rfft is True and spectrum_is_fftshifted is True:
        raise ValueError('rfft cannot be fftshifted.')
    phase_shifts = get_phase_shifts_2d(shifts=shifts, image_shape=image_shape, rfft=rfft)
    if spectrum_is_fftshifted:
        phase_shifts = torch.fft.fftshift(phase_shifts, dim=(-2, -1))
    return dfts * phase_shifts


def phase_shift_images_2d(images: torch.Tensor, shifts: torch.Tensor):
    image_shape = images.shape[-2:]
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = fourier_shift_dfts_2d(
        images,
        image_shape=image_shape,
        shifts=shifts,
        rfft=True,
        spectrum_is_fftshifted=False
    )
    images = torch.fft.irfftn(images, dim=(-2, -1))
    return torch.real(images)


def _rfft_shape_from_input_shape(input_shape: Sequence[int]) -> Tuple[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return rfft_shape
