from typing import Tuple

import einops
import torch
import torch.nn.functional as F

from .coordinate_utils import (
    get_array_coordinates,
    homogenise_coordinates,
    add_implied_coordinate_from_dimension,
    array_coordinates_to_grid_sample_coordinates,
)


def backproject(
        image_stack: torch.Tensor,  # (b, h, w)
        projection_matrices: torch.Tensor,  # (b, 4, 4)
        output_dimensions: Tuple[int, int, int]
) -> torch.Tensor:
    grid_coordinates = get_array_coordinates(output_dimensions)  # (d, h, w, zyx)
    grid_coordinates = torch.flip(grid_coordinates, dims=(-1,))  # (d, h, w, xyz)
    grid_coordinates = homogenise_coordinates(grid_coordinates)  # (d, h, w, xyzw)
    grid_coordinates = einops.rearrange(grid_coordinates, 'd h w xyzw -> d h w 1 xyzw 1')
    projected_coordinates = projection_matrices[..., :2,
                            :] @ grid_coordinates  # only xy coords needed
    projected_coordinates = einops.rearrange(
        projected_coordinates, 'd h w img xy 1 -> d h w img xy'
    )
    image_stack_coordinates = add_implied_coordinate_from_dimension(projected_coordinates, dim=3)
    image_stack_coordinates = einops.rearrange(
        image_stack_coordinates, 'd h w img xyz -> img d h w xyz'
    )
    image_stack_coordinates = torch.flip(image_stack_coordinates, dims=(-1,))  # xyz -> zyx
    image_stack_coordinates = array_coordinates_to_grid_sample_coordinates(
        image_stack_coordinates, array_shape=image_stack.shape
    )
    n_images = image_stack_coordinates.shape[0]
    image_stack = einops.repeat(image_stack, 'b h w -> img 1 b h w',
                                img=n_images)  # (b, c, d, h, w) for sampling
    samples = F.grid_sample(
        input=image_stack,
        grid=image_stack_coordinates,
        mode='bilinear',  # this is trilinear when input is volumetric
        padding_mode='zeros',
        align_corners=False,
    )
    reconstruction = einops.reduce(samples, 'img 1 d h w -> d h w', reduction='sum')
    return reconstruction
