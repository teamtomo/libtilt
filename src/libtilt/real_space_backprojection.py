import functools
from typing import Tuple

import einops
import torch
import torch.nn.functional as F

from .coordinate_utils import (
    get_array_coordinates,
    homogenise_coordinates,
    array_coordinates_to_grid_sample_coordinates,
)


def backproject(
        projection_images: torch.Tensor,  # (batch, h, w)
        projection_matrices: torch.Tensor,  # (batch, 4, 4)
        output_dimensions: Tuple[int, int, int]
) -> torch.Tensor:
    grid_coordinates = get_array_coordinates(output_dimensions)  # (d, h, w, zyx)
    grid_coordinates = torch.flip(grid_coordinates, dims=(-1,))  # (d, h, w, xyz)
    grid_coordinates = homogenise_coordinates(grid_coordinates)  # (d, h, w, xyzw)
    grid_coordinates = einops.rearrange(grid_coordinates, 'd h w xyzw -> d h w xyzw 1')

    def _backproject_one_image(image, projection_matrix) -> torch.Tensor:
        coords_2d = projection_matrix[:2, :] @ grid_coordinates  # (d, h, w, xy, 1)
        coords_2d = einops.rearrange(coords_2d, 'd h w xy 1 -> d h w xy')
        coords_2d = torch.flip(coords_2d, dims=(-1,))  # xy -> yx
        coords_2d = array_coordinates_to_grid_sample_coordinates(
            coords_2d, array_shape=image.shape
        )
        image = einops.rearrange(image, 'h w -> 1 1 h w')
        d, h = coords_2d.shape[:2]
        # (n h w 2) coordinates required for grid sample on 2D images
        coords_2d = einops.rearrange(coords_2d, 'd h w xy -> 1 (d h) w xy')
        samples = F.grid_sample(
            input=image,
            grid=coords_2d,
            mode='bicubic',
            padding_mode='zeros',
            align_corners=False,
        )
        return einops.rearrange(samples, '1 1 (d h) w -> d h w', d=d, h=h)

    backprojection_generator = (
        _backproject_one_image(image, projection_matrix)
        for image, projection_matrix
        in zip(projection_images, projection_matrices)
    )
    return functools.reduce(lambda x, y: x + y, backprojection_generator)
