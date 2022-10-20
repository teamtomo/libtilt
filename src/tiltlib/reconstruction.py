from typing import Tuple

import einops
import numpy as np
import torch


def insert_data_dft(
        dft: torch.Tensor,  # (d, d, d)
        multiplicity: torch.Tensor,  # (d, d, d)
        data: torch.Tensor,  # (n, )
        coordinates: torch.Tensor,  # (n, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    cz, cy, cx = einops.rearrange(torch.ceil(coordinates), 'n c -> c n').long()
    fz, fy, fx = einops.rearrange(torch.floor(coordinates), 'n c -> c n').long()

    def add_for_corner(z, y, x):
        difference = einops.rearrange([z, y, x], 'zyx n -> n zyx') - coordinates
        distance = einops.reduce(difference ** 2, 'n zyx -> n', reduction='sum') ** 0.5
        weights = 1 - distance
        weights[weights < 0] = 0
        dft.index_put_(indices=(z, y, x), values=weights * data, accumulate=True)
        multiplicity.index_put_(indices=(z, y, x), values=weights, accumulate=True)

    add_for_corner(fz, fy, fx)
    add_for_corner(fz, fy, cx)
    add_for_corner(fz, cy, fx)
    add_for_corner(cz, fy, fx)
    add_for_corner(fz, cy, cx)
    add_for_corner(cz, fy, cx)
    add_for_corner(cz, cy, fx)
    add_for_corner(cz, cy, cx)

    return dft, multiplicity


def grid_sinc2(shape: Tuple[int, int, int]):
    d = torch.Tensor(np.stack(np.indices(tuple(shape)), axis=-1))
    d -= torch.Tensor(tuple(shape)) // 2
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
    data = einops.rearrange(data, 'b h w -> (b h w)')
    coordinates = einops.rearrange(coordinates, 'b h w zyx -> (b h w) zyx').float()
    valid_coordinates = torch.logical_and(
        coordinates >= 0, coordinates < torch.Tensor(volume_shape) - 1
    )
    valid_coordinates = torch.all(valid_coordinates, dim=-1)
    data, coordinates = data[valid_coordinates], coordinates[valid_coordinates]
    output, weights = insert_data_dft(output, weights, data, coordinates)
    valid_weights = weights > 1e-3
    output[valid_weights] /= weights[valid_weights]
    output = torch.fft.ifftshift(output, dim=(-3, -2, -1))
    output = torch.fft.ifftn(output, dim=(-3, -2, -1))
    output = torch.fft.ifftshift(output, dim=(-3, -2, -1))
    if do_gridding_correction is True:
        output /= grid_sinc2(volume_shape)
    return output




