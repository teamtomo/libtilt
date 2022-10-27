from typing import Sequence, List

import einops
import torch


def _rfft_shape_from_input_shape(input_shape: Sequence[int]) -> List[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return rfft_shape


def construct_fftfreq_grid_2d(image_shape: Sequence[int], rfft: bool) -> torch.Tensor:
    """Construct a grid of DFT sample frequencies for a 2D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 2D shape `(h, w)` of the input image for which DFT frequencies should be calculated.
    rfft: bool
        Controls Whether the frequency grid is for a real fft (rfft).

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 2)` array of DFT sample frequencies.
        Order of frequencies in the last dimension corresponds to the order of dimensions of the grid.
    """
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    dft_shape = _rfft_shape_from_input_shape(image_shape) if rfft is True else image_shape
    freq_y = torch.fft.fftfreq(image_shape[-2])
    freq_x = last_axis_frequency_func(image_shape[-1])
    freq_yy = einops.repeat(freq_y, 'h -> h w', w=dft_shape[-1])
    freq_xx = einops.repeat(freq_x, 'w -> h w', h=dft_shape[-2])
    return einops.rearrange([freq_yy, freq_xx], 'freq h w -> h w freq')

