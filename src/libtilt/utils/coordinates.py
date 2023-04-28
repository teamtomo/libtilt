from typing import Sequence

import einops
import torch
from torch.nn import functional as F


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


def grid_sample_to_array(
    grid_sample_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate array coordinates from `torch.nn.functional.grid_sample` grids.

    Parameters
    ----------
    grid_sample_coordinates: torch.Tensor
        `(..., d)` array of coordinates to be used with `torch.nn.functional.grid_sample`.
    array_shape: Sequence[int]
        Shape of the array `grid_sample_coordinates` are used to sample.
    """
    dtype, device = grid_sample_coordinates.dtype, grid_sample_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    array_shape = torch.flip(array_shape, dims=(-1,))
    array_coordinates = (grid_sample_coordinates + 1) * (0.5 * array_shape - 0.5)
    array_coordinates = torch.flip(array_coordinates, dims=(-1,))
    return array_coordinates


def fftfreq_to_array(
    frequency_grid: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate array coordinates from DFT sample frequencies.

    Parameters
    ----------
    frequency_grid: torch.Tensor
        `(..., d)` grid of DFT sample frequencies.
    array_shape: Sequence[int]


    Returns
    -------

    """
    pass


def promote_2d_shifts_to_3d(shifts: torch.Tensor) -> torch.Tensor:
    """Promote arrays of 2D shifts to 3D with zeros in the last column.

    Last dimension of array should be of length 2.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 2)` array of 2D shifts

    Returns
    -------
    output: torch.Tensor
        `(..., 3)` array of 3D shifts with 0 in the last column.
    """
    shifts = torch.as_tensor(shifts)
    if shifts.ndim == 1:
        shifts = einops.rearrange(shifts, 's -> 1 s')
    if shifts.shape[-1] != 2:
        raise ValueError('last dimension must have length 2.')
    shifts = F.pad(shifts, pad=(0, 1), mode='constant', value=0)
    return torch.squeeze(shifts)


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """3D coordinates to 4D homogenous coordinates with ones in the last column.

    Parameters
    ----------
    coords: torch.Tensor
        `(..., 3)` array of 3D coordinates

    Returns
    -------
    output: torch.Tensor
        `(..., 4)` array of homogenous coordinates
    """
    return F.pad(torch.as_tensor(coords), pad=(0, 1), mode='constant', value=1)


def add_positional_coordinate(
    coordinates: torch.Tensor, dim: int, prepend: bool = False
) -> torch.Tensor:
    """Make an implicit coordinate in a multidimensional stack of coordinates explicit.

    For an array of coordinates with shape `(n, t, 3)`, this function produces an array of
    shape `(n, t, 4)`.
    The values in the new column reflect the position of the coordinate in `dim`.
    `prepend_new_coordinate` controls whether the new coordinate is prepended
    (`prepend_new_coordinate=True`) or appended (`prepend_new_coordinate=False`) to the existing
    coordinates.

    Parameters
    ----------
    coordinates: torch.Tensor
        `(..., d)` array of coordinates where d is the dimensionality of coordinates.
    dim: int
        dimension from which the value of the new coordinate will be inferred.
    prepend: bool
        Whether to prepend (`False`) or append (`True`) to existing dimension `d`.

    Returns
    -------
    coordinates: torch.Tensor
        `(..., d+1)`
    """
    if prepend is True:
        pad, new_coordinate_index = (1, 0), 0
    else:  # append
        pad, new_coordinate_index = (0, 1), -1
    output = F.pad(coordinates, pad=pad, mode='constant', value=0)
    output[..., new_coordinate_index] = torch.arange(coordinates.shape[dim])
    return output
