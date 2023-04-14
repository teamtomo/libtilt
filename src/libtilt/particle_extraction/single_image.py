from typing import Tuple

import einops
import torch
import torch.nn.functional as F

from libtilt.utils.coordinates import array_to_grid_sample
from libtilt.grids.coordinate import coordinate_grid
from libtilt.utils.fft import dft_center


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
    ph, pw = (output_image_sidelength, output_image_sidelength)
    coordinates = coordinate_grid((ph, pw), device=image.device)  # (h, w, 2)
    grid_center = dft_center((ph, pw), rfft=False, fftshifted=True, device=image.device)
    centered_grid = coordinates - grid_center
    broadcastable_coordinates = einops.rearrange(extraction_positions, 'b yx -> b 1 1 yx')
    grid = centered_grid + broadcastable_coordinates  # (b, h, w, 2)

    # sample subregions, grid sample handles boundaries
    sampled_grids = F.grid_sample(
        input=einops.repeat(image, 'h w -> b 1 h w', b=b),
        grid=array_to_grid_sample(grid, array_shape=(h, w)),
        mode='nearest',
        padding_mode='reflection',
        align_corners=True
    )
    return einops.rearrange(sampled_grids, 'b 1 h w -> b h w'), shifts
