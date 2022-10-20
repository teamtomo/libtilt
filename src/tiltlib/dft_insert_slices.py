from typing import Tuple

import einops
import numpy as np
import torch


def insert_slices_dft(
        dft: torch.Tensor,  # (d, d, d)
        weights: torch.Tensor,  # (d, d, d)
        slices: torch.Tensor,  # (batch, h, w)
        slice_coordinates: torch.Tensor,  # (batch, h, w, 3) ordered zyx
) -> Tuple[torch.Tensor, torch.Tensor]:
    slices = einops.rearrange(slices, 'b h w -> (b h w)')
    slice_coordinates = einops.rearrange(slice_coordinates, 'b h w zyx -> (b h w) zyx').float()
    in_volume_idx = (slice_coordinates >= 0) & (slice_coordinates <= torch.tensor(dft.shape) - 1)
    in_volume_idx = torch.all(in_volume_idx, dim=-1)
    slices, slice_coordinates = slices[in_volume_idx], slice_coordinates[in_volume_idx]
    cz, cy, cx = einops.rearrange(torch.ceil(slice_coordinates), 'n c -> c n').long()
    fz, fy, fx = einops.rearrange(torch.floor(slice_coordinates), 'n c -> c n').long()

    def add_for_corner(z, y, x):
        # calculate weighting
        difference = einops.rearrange([z, y, x], 'zyx n -> n zyx') - slice_coordinates
        distance = einops.reduce(difference ** 2, 'n zyx -> n', reduction='sum') ** 0.5
        corner_weights = 1 - distance
        corner_weights[corner_weights < 0] = 0

        # insert data
        dft.index_put_(indices=(z, y, x), values=corner_weights * slices, accumulate=True)
        weights.index_put_(indices=(z, y, x), values=corner_weights, accumulate=True)

    add_for_corner(fz, fy, fx)
    add_for_corner(fz, fy, cx)
    add_for_corner(fz, cy, fx)
    add_for_corner(cz, fy, fx)
    add_for_corner(fz, cy, cx)
    add_for_corner(cz, fy, cx)
    add_for_corner(cz, cy, fx)
    add_for_corner(cz, cy, cx)

    return dft, weights


def grid_sinc2(shape: Tuple[int, int, int]):
    d = torch.tensor(np.stack(np.indices(tuple(shape)), axis=-1)).float()
    d -= torch.tensor(tuple(shape)) // 2
    d = torch.linalg.norm(d, dim=-1)
    d /= shape[-1]
    sinc2 = torch.sinc(d) ** 2
    return sinc2


def reconstruct_from_images(
        data: torch.Tensor,  # (b, h, w)
        coordinates: torch.Tensor,  # (b, h, w, zyx)
        do_gridding_correction: bool = True,
):
    """"""
    b, h, w = data.shape
    assert h == w
    volume_shape = (w, w, w)

    output = torch.zeros(size=volume_shape, dtype=torch.complex64)
    weights = torch.zeros_like(output, dtype=torch.float32)

    data = torch.fft.fftshift(data, dim=(-2, -1))
    data = torch.fft.fftn(data, dim=(-2, -1))
    data = torch.fft.fftshift(data, dim=(-2, -1))
    output, weights = insert_slices_dft(output, weights, data, coordinates)
    valid_weights = weights > 1e-3
    output[valid_weights] /= weights[valid_weights]
    output = torch.fft.ifftshift(output, dim=(-3, -2, -1))
    output = torch.fft.ifftn(output, dim=(-3, -2, -1))
    output = torch.fft.ifftshift(output, dim=(-3, -2, -1))
    if do_gridding_correction is True:
        output /= grid_sinc2(volume_shape)
    return torch.real(output)




