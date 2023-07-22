from typing import Literal

import einops
import torch
import torch.nn.functional as F

from libtilt.coordinate_utils import array_to_grid_sample


def sample_image_2d(
    image: torch.Tensor,
    coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample a 2D image with bicubic interpolation.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` image.
    coordinates: torch.Tensor
        `(..., yx)` array of coordinates at which `image` should be sampled.
        Coordinates should be ordered zyx, aligned with image dimensions `(d, h, w)`.
        Coordinates should be array coordinates, spanning `[0, N-1]` for a
        dimension of length N.
    Returns
    -------
    samples: torch.Tensor
        `(..., )` array of complex valued samples from `dft`.
    """
    coordinates, ps = einops.pack([coordinates], pattern='* yx')
    n_samples = coordinates.shape[0]

    image = einops.repeat(image, 'h w -> b 1 h w', b=n_samples)  # b c h w
    coordinates = einops.rearrange(coordinates, 'b yx -> b 1 1 yx')  # b h w zyx

    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-2:]),
        mode='bilinear',
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )
    samples = einops.rearrange(samples, 'b 1 1 1 -> b')

    # zero out samples from outside of image
    coordinates = einops.rearrange(coordinates, 'b 1 1 yx -> b yx')
    image_shape = torch.as_tensor(image.shape)
    inside = torch.logical_and(coordinates >= 0, coordinates <= image_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
    samples[~inside] *= 0

    # pack data back up and return
    [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
    return samples  # (...)


def insert_into_image_2d(
    data: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 2D image with trilinear interpolation (rasterisation).

    Parameters
    ----------
    data: torch.Tensor
        `(...)` array of values to be inserted into the DFT.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
    image: torch.Tensor
        `(d, d, d)` volume containing the discrete Fourier transform into which data will be inserted.
    weights: torch.Tensor
        `(d, d, d)` volume containing the weights associated with each voxel of `dft`

    Returns
    -------
    dft, weights: tuple[torch.Tensor, torch.Tensor]
        The dft and weights after updating with data from `slices` at `slice_coordinates`.
    """
    if data.shape != coordinates.shape[:-1]:
        raise ValueError('One coordinate triplet is required for each value in data.')

    # linearise data and coordinates
    data, _ = einops.pack([data], pattern='*')
    coordinates, _ = einops.pack([coordinates], pattern='* zyx')
    coordinates = coordinates.float()

    # only keep data and coordinates inside the volume
    in_volume_idx = (coordinates >= 0) & (coordinates <= torch.tensor(image.shape) - 1)
    in_volume_idx = torch.all(in_volume_idx, dim=-1)
    data, coordinates = data[in_volume_idx], coordinates[in_volume_idx]

    # calculate and cache floor and ceil of coordinates for each piece of slice data
    corner_coordinates = torch.empty(size=(data.shape[0], 2, 2), dtype=torch.long)
    corner_coordinates[:, 0] = torch.floor(coordinates)  # for lower corners
    corner_coordinates[:, 1] = torch.ceil(coordinates)  # for upper corners

    # cache linear interpolation weights for each data point being inserted
    _weights = torch.empty(size=(data.shape[0], 2, 2))  # (b, 2, yx)
    _weights[:, 1] = coordinates - corner_coordinates[:, 0]  # upper corner weights
    _weights[:, 0] = 1 - _weights[:, 1]  # lower corner weights

    def add_data_at_corner(y: Literal[0, 1], x: Literal[0, 1]):
        w = einops.reduce(_weights[:, [y, x], [0, 1]], 'b yx -> b', reduction='prod')
        yc, xc = einops.rearrange(corner_coordinates[:, [y, x], [0, 1]], 'b yx -> yx b')
        image.index_put_(indices=(yc, xc), values=w * data, accumulate=True)
        weights.index_put_(indices=(yc, xc), values=w, accumulate=True)

    add_data_at_corner(0, 0)
    add_data_at_corner(0, 1)
    add_data_at_corner(1, 0)
    add_data_at_corner(1, 1)
    return image, weights
