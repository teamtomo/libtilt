import functools
from typing import Sequence, Tuple, Union

import einops
import torch

from libtilt.utils.fft import rfft_shape, fftshift_2d, fftshift_3d


@functools.lru_cache(maxsize=1)
def fftfreq_grid(
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    spacing: float | tuple[float, float] | tuple[float, float, float] = 1,
    norm: bool = False,
    device: torch.device | None = None,
):
    """Construct a 2D or 3D grid of DFT sample frequencies.

    For a 2D image with shape `(h, w)` and `rfft=False` this function will produce
    a `(h, w, 2)` array of DFT sample frequencies in the `h` and `w` dimensions.
    If `norm` is True the Euclidean norm will be calculated over the last dimension
    leaving a `(h, w)` grid.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        Shape of the 2D or 3D image before computing the DFT.
    rfft: bool
        Whether the output should contain frequencies for a real-valued DFT.
    fftshift: bool
        Whether to fftshift the output grid.
    spacing: float | tuple[float, float] | tuple[float, float, float]
        Spacing between samples in each dimension. Sampling is considered to be
        isotropic if a single value is passed.
    norm: bool
        Whether to compute the Euclidean norm over the last dimension.
    device: torch.device | None
        PyTorch device on which the returned grid will be stored.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(*image_shape, ndim)` array of DFT sample frequencies in each
        image dimension if `norm` is `False` else `(*image_shape, )`.
    """
    if len(image_shape) == 2:
        frequency_grid = _construct_fftfreq_grid_2d(
            image_shape=image_shape,
            rfft=rfft,
            spacing=spacing,
            device=device,
        )
        if fftshift is True:
            frequency_grid = einops.rearrange(frequency_grid, '... freq -> freq ...')
            frequency_grid = fftshift_2d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, 'freq ... -> ... freq')
    elif len(image_shape) == 3:
        frequency_grid = _construct_fftfreq_grid_3d(
            image_shape=image_shape,
            rfft=rfft,
            spacing=spacing,
            device=device,
        )
        if fftshift is True:
            frequency_grid = einops.rearrange(frequency_grid, '... freq -> freq ...')
            frequency_grid = fftshift_3d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, 'freq ... -> ... freq')
    else:
        raise NotImplementedError(
            "Construction of fftfreq grids is currently only supported for "
            "2D and 3D images."
        )
    if norm is True:
        frequency_grid = einops.reduce(
            frequency_grid ** 2, '... d -> ...', reduction='sum'
        ) ** 0.5
    return frequency_grid


def _construct_fftfreq_grid_2d(
    image_shape: Tuple[int, int],
    rfft: bool,
    spacing: float | tuple[float, float] = 1,
    device: torch.device = None
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 2D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 2D shape `(h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Whether the frequency grid is for a real fft (rfft).
    spacing: float | Tuple[float, float]
        Sample spacing in `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 2)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of
        the two dimensions of the grid.
    """
    dh, dw = spacing if isinstance(spacing, Sequence) else (spacing, spacing)
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    h, w = image_shape
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_yy = einops.repeat(freq_y, 'h -> h w', w=w)
    freq_xx = einops.repeat(freq_x, 'w -> h w', h=h)
    return einops.rearrange([freq_yy, freq_xx], 'freq h w -> h w freq')


def _construct_fftfreq_grid_3d(
    image_shape: Sequence[int],
    rfft: bool,
    spacing: float | Tuple[float, float, float] = 1,
    device: torch.device = None
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 3D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 3D shape `(d, h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Controls Whether the frequency grid is for a real fft (rfft).
    spacing: float | Tuple[float, float, float]
        Sample spacing in `d`, `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 3)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of dimensions
        of the grid.
    """
    dd, dh, dw = spacing if isinstance(spacing, Sequence) else (spacing, spacing, spacing)
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    d, h, w = image_shape
    freq_z = torch.fft.fftfreq(d, d=dd, device=device)
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    d, h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_zz = einops.repeat(freq_z, 'd -> d h w', h=h, w=w)
    freq_yy = einops.repeat(freq_y, 'h -> d h w', d=d, w=w)
    freq_xx = einops.repeat(freq_x, 'w -> d h w', d=d, h=h)
    return einops.rearrange([freq_zz, freq_yy, freq_xx], 'freq ... -> ... freq')
