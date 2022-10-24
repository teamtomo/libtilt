from typing import Sequence, Tuple, Literal

import einops
import numpy as np
import torch
from torch.nn import functional as F


def get_array_coordinates(grid_dimensions: Sequence[int]) -> torch.Tensor:
    """Get a dense grid of array coordinates from grid dimensions.

    e.g. for an input array shape of (d, h, w), produce a (d, h, w, 3)
    array of indices into a (d, h, w) array. Ordering of the coordinates matches the order of
    dimensions in the input array shape.
    """
    indices = torch.tensor(np.indices(grid_dimensions)).float()  # (coordinates, *grid_dimensions)
    return einops.rearrange(indices, 'coordinates ... -> ... coordinates')


def promote_2d_to_3d(vectors: torch.Tensor) -> torch.Tensor:
    """Promote arrays of 2D vectors to 3D with zeros in the last column.

    Last dimension of array should be of length 2.

    Parameters
    ----------
    vectors: torch.Tensor
        (..., 2) array of 2D vectors

    Returns
    -------
    output: torch.Tensor
        (..., 3) array of 3D vectors with 0 in the last column.
    """
    vectors = F.pad(torch.tensor(vectors), pad=(0, 1), mode='constant', value=0)
    return torch.squeeze(vectors)


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """3D coordinates to 4D homogenous coordinates with ones in the last column.

    Parameters
    ----------
    coords: torch.Tensor
        (..., 3) array of 3D coordinates

    Returns
    -------
    output: torch.Tensor
        (..., 4) array of homogenous coordinates
    """
    return F.pad(torch.Tensor(coords), pad=(0, 1), mode='constant', value=1)


def generate_rotated_slice_coordinates(rotations: torch.Tensor, n: int) -> torch.Tensor:
    """Generate an array of rotated central slice coordinates for sampling a 3D image.

    Notes
    -----
    - rotation matrices rotate coordinates ordered xyz
    - coordinates returned are ordered zyx to match volumetric array indices

    Parameters
    ----------
    rotations: torch.Tensor
        (batch, 3, 3) array of rotation matrices which rotate xyz coordinates.
    n: int
        sidelength of cubic grid for which coordinates are generated.

    Returns
    -------
    coordinates: torch.Tensor
        (batch, n, n, zyx) array of coordinates.
    """
    # generate [x, y, z] coordinates for a central slice
    # the slice spans the XY plane with origin on DFT center
    x = y = torch.arange(n) - (n // 2)
    xx = einops.repeat(x, 'w -> h w', h=n)
    yy = einops.repeat(y, 'h -> h w', w=n)
    zz = torch.zeros(size=(n, n))
    xyz = einops.rearrange([xx, yy, zz], 'xyz h w -> 1 h w xyz 1')

    # rotate coordinates
    rotations = einops.rearrange(rotations, 'b i j -> b 1 1 i j')
    xyz = einops.rearrange(rotations @ xyz, 'b h w xyz 1 -> b h w xyz')

    # recenter slice on DFT center and flip to zyx
    xyz += n // 2
    zyx = torch.flip(xyz, dims=(-1,))
    return zyx


def add_implied_coordinate_from_dimension(
        coordinates: torch.Tensor, dim: int, prepend_new_coordinate: bool = False
) -> torch.Tensor:
    """Make an implicit coordinate in a multidimensional arrays of coordinates explicit.

    For an array of coordinates with shape (n, t, 3), this function produces an array of
    shape (n, t, 4). The values in the new column reflect the position of the coordinate in `dim`.
    `prepend_new_coordinate` controls whether the new coordinate is prepended
    (`prepend_new_coordinate=True`) or appended (`prepend_new_coordinate=False`) to the existing
    coordinates.

    Parameters
    ----------
    coordinates: torch.Tensor
        (..., d) array of coordinates where d is the dimensionality of coordinates.
    dim: int
        dimension from which the value of the new coordinate will be inferred.
    prepend_new_coordinate: bool
        controls whether the new coordinate is prepended or appended to existing coordinates.

    Returns
    -------
    coordinates: torch.Tensor
        (..., d+1)
    """
    if prepend_new_coordinate is True:
        pad, new_coordinate_index = (1, 0), 0
    else:  # append
        pad, new_coordinate_index = (0, 1), -1
    output = F.pad(coordinates, pad=pad, mode='constant', value=0)
    output[..., new_coordinate_index] = torch.arange(coordinates.shape[dim])
    return output


def _array_coordinates_to_grid_sample_coordinates_1d(
        coordinates: torch.Tensor, dim_length: int
) -> torch.Tensor:
    return (coordinates / (0.5 * dim_length - 0.5)) - 1


def _grid_sample_coordinates_to_array_coordinates_1d(
        coordinates: torch.Tensor, dim_length: int
) -> torch.Tensor:
    return (coordinates + 1) * (0.5 * dim_length - 0.5)


def array_coordinates_to_grid_sample_coordinates(
        array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate coordinates for use with torch.nn.functional.grid_sample from array coordinates.

    Notes
    -----
    - array coordinates are from [0, N-1] for N elements in each dimension.
        - 0 is at the center of the first element
        - N is the length of the dimension
    - grid sample coordinates are from [-1, 1]
        - if align_corners=True, -1 and 1 are at the edges of array elements 0 and N-1
        - if align_corners=False, -1 and 1 are at the centers of array elements 0 and N-1
    - generated coordinates are
    """
    coords = [
        _array_coordinates_to_grid_sample_coordinates_1d(array_coordinates[..., idx], dim_length)
        for idx, dim_length
        in enumerate(array_shape)
    ]
    return einops.rearrange(coords[::-1], 'xyz ... -> ... xyz')


def grid_sample_coordinates_to_array_coordinates(coordinates: torch.Tensor,
                                                 array_shape: Sequence[int]) -> torch.Tensor:
    indices = [
        _grid_sample_coordinates_to_array_coordinates_1d(coordinates[..., idx], dim_length)
        for idx, dim_length
        in enumerate(array_shape[::-1])
    ]
    return einops.rearrange(indices[::-1], 'zyx b h w -> b h w zyx')
