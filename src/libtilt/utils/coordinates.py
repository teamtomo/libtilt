from typing import Sequence, Tuple, Literal

import einops
import numpy as np
import torch
from torch.nn import functional as F


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate coordinates for use with `torch.nn.functional.grid_sample` from array coordinates.

    - array coordinates are from [0, N-1] for N elements in each dimension.
        - 0 is at the center of the first element
        - N is the length of the dimension
    - grid sample coordinates are from [-1, 1]
        - if align_corners=True, -1 and 1 are at the edges of array elements 0 and N-1
        - if align_corners=False, -1 and 1 are at the centers of array elements 0 and N-1


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    coords = [
        _array_coordinates_to_grid_sample_coordinates_1d(
            array_coordinates[..., idx], dim_length
        )
        for idx, dim_length
        in enumerate(array_shape)
    ]
    return einops.rearrange(coords[::-1], 'xyz ... -> ... xyz')


def grid_sample_to_array(
    grid_sample_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate array coordinates from coordinates used for `torch.nn.functional.grid_sample`.

    Parameters
    ----------
    grid_sample_coordinates: torch.Tensor
        `(..., d)` array of coordinates to be used with `torch.nn.functional.grid_sample`.
    array_shape: Sequence[int]
        shape of the array `grid_sample_coordinates` sample.
    """
    indices = [
        _grid_sample_coordinates_to_array_coordinates_1d(
            grid_sample_coordinates[..., idx], dim_length
        )
        for idx, dim_length
        in enumerate(array_shape[::-1])
    ]
    return einops.rearrange(indices[::-1], 'zyx b h w -> b h w zyx')


def get_array_coordinates(grid_dimensions: Sequence[int]) -> torch.Tensor:
    """Get a dense grid of array coordinates from grid dimensions.

    For input `grid_dimensions` of `(d, h, w)`, produce a `(d, h, w, 3)`
    array of indices into a `(d, h, w)` array. Ordering of the coordinates
    matches the order of dimensions in `grid_dimensions`.

    Parameters
    ----------
    grid_dimensions: Sequence[int]
        the dimensions of the grid for which coordinates should be returned.
    """
    indices = torch.tensor(np.indices(grid_dimensions)).float()
    # (coordinates, *grid_dimensions)
    return einops.rearrange(indices, 'coordinates ... -> ... coordinates')


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


def generate_rotated_slice_coordinates(
    rotations: torch.Tensor, sidelength: int
) -> torch.Tensor:
    """Generate an array of rotated central slice coordinates for sampling a 3D image.

    Rotation matrices left multiply `xyz` coordinates in column vectors.
    Coordinates returned are ordered `zyx` to match volumetric array indices.

    Parameters
    ----------
    rotations: torch.Tensor
        `(batch, 3, 3)` array of rotation matrices which rotate xyz coordinates.
    sidelength: int
        sidelength of cubic volume for which coordinates are generated.

    Returns
    -------
    coordinates: torch.Tensor
        `(batch, n, n, zyx)` array of coordinates where `n == sidelength`.
    """
    if rotations.ndim == 2:
        rotations = einops.rearrange(rotations, 'i j -> 1 i j')
    # generate [x, y, z] coordinates for a central slice
    # the slice spans the XY plane with origin on DFT center
    x = y = torch.arange(sidelength) - (sidelength // 2)
    xx = einops.repeat(x, 'w -> h w', h=sidelength)
    yy = einops.repeat(y, 'h -> h w', w=sidelength)
    zz = torch.zeros(size=(sidelength, sidelength))
    xyz = einops.rearrange([xx, yy, zz], 'xyz h w -> 1 h w xyz 1')

    # rotate coordinates
    rotations = einops.rearrange(rotations, 'b i j -> b 1 1 i j')
    xyz = einops.rearrange(rotations @ xyz, 'b h w xyz 1 -> b h w xyz')

    # recenter slice on DFT center and flip to zyx
    xyz += sidelength // 2
    zyx = torch.flip(xyz, dims=(-1,))
    return zyx


def add_positional_coordinate_from_dimension(
    coordinates: torch.Tensor, dim: int, prepend_new_coordinate: bool = False
) -> torch.Tensor:
    """Make an implicit coordinate in a multidimensional arrays of coordinates explicit.

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
    prepend_new_coordinate: bool
        controls whether the new coordinate is prepended or appended to existing coordinates.

    Returns
    -------
    coordinates: torch.Tensor
        `(..., d+1)`
    """
    if prepend_new_coordinate is True:
        pad, new_coordinate_index = (1, 0), 0
    else:  # append
        pad, new_coordinate_index = (0, 1), -1
    output = F.pad(coordinates, pad=pad, mode='constant', value=0)
    # swapped = False
    # if len(output.shape) > 3 and dim != output.shape[-2]:
    #     output = torch.swapaxes(output, dim, -2)
    #     swapped = True
    output[..., new_coordinate_index] = torch.arange(coordinates.shape[dim])
    # if swapped: # unswap
    #     output = torch.swapaxes(output, dim, -2)
    return output


def _array_coordinates_to_grid_sample_coordinates_1d(
    coordinates: torch.Tensor, dim_length: int
) -> torch.Tensor:
    return (coordinates / (0.5 * dim_length - 0.5)) - 1


def _grid_sample_coordinates_to_array_coordinates_1d(
    coordinates: torch.Tensor, dim_length: int
) -> torch.Tensor:
    return (coordinates + 1) * (0.5 * dim_length - 0.5)
