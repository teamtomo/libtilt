from typing import Sequence, Optional

import einops
import numpy as np
import torch

from libtilt.utils.fft import dft_center, rfft_shape_from_signal_shape


def coordinate_grid(
    image_shape: Sequence[int],
    center_grid: bool = False,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Get a dense grid of array coordinates from grid dimensions.

    For input `image_shape` of `(d, h, w)`, this function produces a
    `(d, h, w, 3)` grid of coordinates. Coordinate order matches the order of
    dimensions in `image_shape`.

    Parameters
    ----------
    image_shape: Sequence[int]
        Shape of the image for which coordinates should be returned.
    device: torch.device
        PyTorch device on which to put the coordinate grid.
    center_grid: bool
        Whether the coordinates should be centered on the rotation center of the grid.

    Returns
    -------
    grid: torch.LongTensor
        `(*image_shape, grid_ndim)` array of coordinates
    """
    grid = torch.tensor(
        np.indices(image_shape),
        device=device,
        dtype=torch.int64
    )  # (coordinates, *image_shape)
    grid = einops.rearrange(grid, 'coordinates ... -> ... coordinates')
    if center_grid is True:
        grid_center = dft_center(image_shape, rfft=False, fftshifted=True, device=device)
        grid -= grid_center
    return grid


def coordinate_grid_dft(
    image_shape: Sequence[int],
    rfft: bool,
    fftshift: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Construct a dense grid of coordinates relative to the DFT center with wraparound.

    For input `image_shape` of `(d, h, w)`, this function produces a
    `(d, h, w, 3)` grid of 3D coordinates.
    Coordinates match the order of dimensions in `image_shape`.

    Parameters
    ----------
    image_shape: Sequence[int]
        Shape of the image prior to DFT for which coordinates should be returned.
    rfft: bool
        Whether the coordinates generated are compatible with the result of `rfft`.
    fftshift: bool
        Whether the coordinates should be compatible with fftshifted DFTs.
    device: torch.device
        PyTorch device on which to put the coordinate grid.


    Returns
    -------
    grid: torch.LongTensor
        `(*image_shape, grid_ndim)` array of coordinates
    """
    original_image_shape = tuple(image_shape)
    if rfft is True:
        image_shape = rfft_shape_from_signal_shape(image_shape)
    grid = torch.tensor(
        data=np.indices(image_shape),
        device=device,
        dtype=torch.int64
    )  # (coordinates, *image_shape)
    grid = einops.rearrange(grid, 'coordinates ... -> ... coordinates')

    # center on fftshifted DC component of DFT
    grid_center = dft_center(original_image_shape, rfft=rfft, fftshifted=True, device=device)
    grid -= grid_center

    # grid is correct for fftshifted case, undo fftshift if not required.
    if fftshift is False:
        # ifftshift all except the coordinate dimension
        print(grid.ndim, ' dims ', tuple(torch.arange(grid.ndim - 1)))
        print(grid.shape)
        grid = torch.fft.ifftshift(grid, dim=tuple(torch.arange(grid.ndim - 1)))
        if rfft is True:
            # undo the ifftshift of the last image dimension
            grid = torch.fft.fftshift(grid, dim=-2)
    return grid
