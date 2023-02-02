from typing import Tuple

import einops
import torch
import torch.nn.functional as F

from libtilt.utils.coordinates import get_array_indices, array_to_grid_sample

images = torch.rand((41, 224, 224))
coordinates = torch.randint(low=0, high=224, size=(1000, 41, 2)) + torch.rand(size=(1000, 41, 2))


def extract_at_integer_coordinates(
    image: torch.Tensor,  # (h, w)
    positions: torch.Tensor,  # (b, 2) yx
    output_image_sidelength: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = image.shape
    b, _ = positions.shape
    # find integer particle_extraction positions
    extraction_positions = torch.round(positions)
    shifts = positions - extraction_positions

    # generate sampling grids
    output_image_shape = (output_image_sidelength, output_image_sidelength)
    coordinate_grid = get_array_indices(output_image_shape)  # (h, w, 2)
    grid_center = torch.div(torch.as_tensor(output_image_shape), 2, rounding_mode='floor')
    centered_grid = coordinate_grid - grid_center
    broadcastable_coordinates = einops.rearrange(extraction_positions, 'b yx -> b 1 1 yx')
    grid = centered_grid + broadcastable_coordinates  # (b, h, w, 2)
    grid = array_to_grid_sample(grid, array_shape=(h, w))

    # sample subregions
    sampled_grids = F.grid_sample(
        input=einops.repeat(image, 'h w -> b 1 h w', b=b),
        grid=grid,
        mode='nearest',
        padding_mode='reflection',
        align_corners=True
    )
    return einops.rearrange(sampled_grids, 'b 1 h w -> b h w'), shifts
