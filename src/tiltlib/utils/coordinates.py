from typing import Sequence

import einops
import torch
from torch.nn import functional as F


def promote_2d_to_3d(shifts: torch.Tensor) -> torch.Tensor:
    """Promote 2D coordinates to 3D with zeros in the last dimension.

    Last dimension of array should be of length 2.
    """
    shifts = F.pad(torch.tensor(shifts), pad=(0, 1), mode='constant', value=0)
    return torch.squeeze(shifts)


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """3D coordinates to 4D homogenous coordinates with ones in the last dimension.

    Last dimension of array should be of length 3.
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
    # generate [x, y, z] coordinates centered on image_sidelength // 2 (center of DFT)
    x = y = torch.arange(n) - (n // 2)
    xx = einops.repeat(x, 'w -> h w', h=n)
    yy = einops.repeat(y, 'h -> h w', w=n)
    zz = torch.zeros(size=(n, n))
    xyz = einops.rearrange([xx, yy, zz], 'xyz h w -> 1 h w xyz 1')

    # rotate coordinates
    rotations = einops.rearrange(rotations, 'b i j -> b 1 1 i j')
    xyz = einops.rearrange(rotations @ xyz, 'b h w xyz 1 -> b h w xyz')

    # recenter
    xyz += n // 2
    zyx = torch.flip(xyz, dims=(-1,))
    return zyx


def stacked_2d_coordinates_to_3d_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """Turn stacks of 2D coordinates for n images into 3D array coordinates into image stack.

    b n yx -> (b n) zyx
    - new z coord is implied by position in dimension n
    - can be used to sample from tilt-series
    """
    b, n = coordinates.shape[:2]
    output = torch.empty([b, n, 3])
    output[:, :, 1:] = coordinates
    output[:, :, 0] = torch.arange(n)
    return einops.rearrange(output, 'b n zyx -> (b n) zyx')


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
    return einops.rearrange(coords[::-1], 'xyz b h w -> b h w xyz')


def grid_sample_coordinates_to_array_coordinates(coordinates: torch.Tensor,
                                                 array_shape: Sequence[int]) -> torch.Tensor:
    indices = [
        _grid_sample_coordinates_to_array_coordinates_1d(coordinates[..., idx], dim_length)
        for idx, dim_length
        in enumerate(array_shape[::-1])
    ]
    return einops.rearrange(indices[::-1], 'zyx b h w -> b h w zyx')
