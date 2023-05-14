from typing import Tuple, Literal

import einops
import torch

from libtilt.grids.fftfreq import _grid_sinc2


def insert_into_dft_3d(
        data: torch.Tensor,
        coordinates: torch.Tensor,
        dft: torch.Tensor,
        weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 3D DFT with trilinear interpolation (rasterisation).

    Parameters
    ----------
    data: torch.Tensor
        `(...)` array of values to be inserted into the DFT.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
    dft: torch.Tensor
        `(d, d, d)` volume containing the discrete Fourier transform into which data will be inserted.
    weights: torch.Tensor
        `(d, d, d)` volume containing the weights associated with each voxel of `dft`

    Returns
    -------
    dft, weights: Tuple[torch.Tensor, torch.Tensor]
        The dft and weights after updating with data from `slices` at `slice_coordinates`.
    """
    if data.shape != coordinates.shape[:-1]:
        raise ValueError('One coordinate triplet is required for each value in data.')

    # linearise data and coordinates
    data, _ = einops.pack([data], pattern='*')
    coordinates, _ = einops.pack([coordinates], pattern='* zyx').float()

    # only keep data and coordinates inside the volume
    in_volume_idx = (coordinates >= 0) & (coordinates <= torch.tensor(dft.shape) - 1)
    in_volume_idx = torch.all(in_volume_idx, dim=-1)
    data, coordinates = data[in_volume_idx], coordinates[in_volume_idx]

    # calculate and cache floor and ceil of coordinates for each piece of slice data
    corner_coordinates = torch.empty(size=(data.shape[0], 2, 3), dtype=torch.long)
    corner_coordinates[:, 0] = torch.floor(coordinates)  # for lower corners
    corner_coordinates[:, 1] = torch.ceil(coordinates)  # for upper corners

    # cache linear interpolation weights for each data point being inserted
    _weights = torch.empty(size=(data.shape[0], 2, 3))  # (b, 2, zyx)
    _weights[:, 1] = coordinates - corner_coordinates[:, 0]  # upper corner weights
    _weights[:, 0] = 1 - _weights[:, 1]  # lower corner weights

    def add_data_at_corner(z: Literal[0, 1], y: Literal[0, 1], x: Literal[0, 1]):
        w = einops.reduce(_weights[:, [z, y, x], [0, 1, 2]], 'b zyx -> b', reduction='prod')
        zc, yc, xc = einops.rearrange(corner_coordinates[:, [z, y, x], [0, 1, 2]], 'b zyx -> zyx b')
        dft.index_put_(indices=(zc, yc, xc), values=w * data, accumulate=True)
        weights.index_put_(indices=(zc, yc, xc), values=w, accumulate=True)

    add_data_at_corner(0, 0, 0)
    add_data_at_corner(0, 0, 1)
    add_data_at_corner(0, 1, 0)
    add_data_at_corner(0, 1, 1)
    add_data_at_corner(1, 0, 0)
    add_data_at_corner(1, 0, 1)
    add_data_at_corner(1, 1, 0)
    add_data_at_corner(1, 1, 1)

    return dft, weights


def reconstruct_from_images(
        images: torch.Tensor,  # (b, h, w)
        slice_coordinates: torch.Tensor,  # (b, h, w, zyx)
        do_gridding_correction: bool = True,
):
    """Perform a 3D reconstruction from a set of 2D projection images.

    Parameters
    ----------
    images: torch.Tensor
        `(batch, h, w)` array of 2D projection images.
    slice_coordinates: torch.Tensor
        `(batch, h, w, zyx)` array of coordinates for pixels in `images`.
        Coordinates are array coordinates.
    do_gridding_correction: bool
        Each 2D image pixel contributes to the nearest eight voxels in 3D and weights are set
        according to a linear interpolation kernel. The effects of this trilinear interpolation in
        Fourier space can be 'undone' through division by a sinc^2 function in real space.

    Returns
    -------
    reconstruction: torch.Tensor
        `(d, h, w)` cubic volume containing the 3D reconstruction from `images`.
    """
    b, h, w = images.shape
    assert h == w
    volume_shape = (w, w, w)

    output = torch.zeros(size=volume_shape, dtype=torch.complex64)
    weights = torch.zeros_like(output, dtype=torch.float32)

    images = torch.fft.fftshift(images, dim=(-2, -1))
    images = torch.fft.fftn(images, dim=(-2, -1))
    images = torch.fft.fftshift(images, dim=(-2, -1))
    output, weights = insert_into_dft_3d(
        data=images,
        coordinates=slice_coordinates,
        dft=output,
        weights=weights
    )
    valid_weights = weights > 1e-3
    output[valid_weights] /= weights[valid_weights]
    output = torch.fft.ifftshift(output, dim=(-3, -2, -1))
    output = torch.fft.ifftn(output, dim=(-3, -2, -1))
    output = torch.fft.ifftshift(output, dim=(-3, -2, -1))
    if do_gridding_correction is True:
        output /= _grid_sinc2(volume_shape)
    return torch.real(output)




