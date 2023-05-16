import functools
from typing import Tuple

import einops
import torch
import torch.nn.functional as F

from libtilt.coordinate_utils import (
    homogenise_coordinates,
    array_to_grid_sample,
)
from libtilt.grids.coordinate_grid import coordinate_grid


def backproject_in_real_space(
        projection_images: torch.Tensor,
        projection_matrices: torch.Tensor,
        output_dimensions: Tuple[int, int, int]
) -> torch.Tensor:
    """3D reconstruction from 2D images by real space backprojection.

    - Coordinates for voxels in the output volume are projected down into 2D
    by left-multiplication with `projection matrix`.
    - For each 3D coordinate each image in `projection_images` is sampled
    with bicubic interpolation at the 2D coordinates `yx`.
    - The final value of a voxel is the sum of contributions from each projection image.

    Parameters
    ----------
    projection_images: torch.Tensor
        `(batch, h, w)` array of 2D projection images.
    projection_matrices: torch.Tensor
        `(batch, 4, 4)` array of projection matrices which relate homogenous coordinates (xyzw)
        in the output volume to coordinates in the projection images.
    output_dimensions: Tuple[int, int, int]
        dimensions of the output volume.

    Returns
    -------
    reconstruction: torch.Tensor
        `(d, h, w)` array containing the reconstructed 3D volume.
    """
    grid_coordinates = coordinate_grid(output_dimensions)  # (d, h, w, zyx)
    grid_coordinates = torch.flip(grid_coordinates, dims=(-1,))  # (d, h, w, xyz)
    grid_coordinates = homogenise_coordinates(grid_coordinates)  # (d, h, w, xyzw)
    grid_coordinates = einops.rearrange(grid_coordinates, 'd h w xyzw -> d h w xyzw 1')

    def _backproject_single_image(image, projection_matrix) -> torch.Tensor:
        image = einops.rearrange(image, 'h w -> 1 1 h w')
        coords_2d = projection_matrix[:2, :] @ grid_coordinates  # (d, h, w, xy, 1)
        coords_2d = einops.rearrange(coords_2d, 'd h w xy 1 -> d h w xy')
        coords_2d = torch.flip(coords_2d, dims=(-1,))  # xy -> yx
        coords_2d = array_to_grid_sample(coords_2d, array_shape=image.shape[-2:])
        samples = F.grid_sample(
            input=image,
            grid=einops.rearrange(coords_2d, 'd h w xy -> 1 (d h) w xy'),
            mode='bicubic',
            padding_mode='zeros',
            align_corners=True,
        )
        d, h = coords_2d.shape[:2]
        return einops.rearrange(samples, '1 1 (d h) w -> d h w', d=d, h=h)

    backprojection_volume_generator = (
        _backproject_single_image(image, projection_matrix)
        for image, projection_matrix
        in zip(projection_images, projection_matrices)
    )
    return functools.reduce(lambda x, y: x + y, backprojection_volume_generator)
