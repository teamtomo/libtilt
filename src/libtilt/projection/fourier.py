import torch
import torch.nn.functional as F
import einops

from libtilt.grids import rotated_fftfreq_central_slice
from libtilt.grids.fftfreq import _grid_sinc2
from libtilt.utils.coordinates import array_to_grid_sample
from libtilt.utils.fft import fftshift_3d, fftfreq_to_dft_coordinates


def interpolate_dft_3d(
    dft: torch.Tensor,
    coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample from a complex volume at specified coordinates.


    Parameters
    ----------
    dft: torch.Tensor
        (d, h, w) complex valued cubic volume (d == h == w) containing
        the discrete Fourier transform of a cubic volume.
    coordinates: torch.Tensor
        (batch, h, w, zyx) array of coordinates at which `dft` should be sampled.
        Coordinates should be ordered zyx, aligned with image dimensions `(d, h, w)`.
        Coordinates should be array coordinates, spanning `[0, N-1]` for a
        dimension of length N.
    Returns
    -------
    samples: torch.Tensor
        (batch, h, w) array of complex valued images sampled from the `dft`
    """
    # cannot sample complex tensors directly with grid_sample
    # c.f. https://github.com/pytorch/pytorch/issues/67634
    # workaround: treat real and imaginary parts as separate channels
    dft = einops.rearrange(torch.view_as_real(dft), 'd h w complex -> complex d h w')
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]
    dft = einops.repeat(dft, 'complex d h w -> b complex d h w', b=n_samples)
    coordinates = array_to_grid_sample(coordinates, array_shape=dft.shape[-3:])
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx

    # sample with border values at edges to increase sampling fidelity at nyquist
    samples = F.grid_sample(
        input=dft,
        grid=coordinates,
        mode='bilinear',  # this is trilinear when input is volumetric
        padding_mode='border',
        align_corners=True,
    )
    samples = einops.rearrange(samples, 'b complex 1 1 1 -> b complex')

    # zero out samples from outside of cube
    samples = torch.view_as_complex(samples.contiguous())
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    inside = torch.logical_or(coordinates > 0, coordinates < 1)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
    samples[~inside] *= 0
    [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
    return samples  # (b, h, w)


def project(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    rotation_matrix_zyx: bool = False,
    pad: bool = True,
    do_gridding_correction: bool = True,
) -> torch.Tensor:
    """Project a cubic volume by sampling a central slice through its DFT.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of matrices which rotate coordinates of the
        central slice to be sampled.
    rotation_matrix_zyx: bool
        Whether rotation matrices apply to zyx (`True`) or xyz (`False`)
        coordinates.
    pad: bool
        Whether to pad the volume with zeros to increase sampling in the DFT.

    Returns
    -------
    projections: torch.Tensor
        `(b, d, d)` array of projection images.
    """
    # padding
    if pad is True:
        pad_length = volume.shape[-1] // 2
        volume = F.pad(volume, pad=[pad_length] * 6, mode='constant', value=0)

    # premultiply by sinc2
    if do_gridding_correction is True:
        sinc2 = _grid_sinc2(volume.shape)
        volume = volume * sinc2

    # calculate DFT
    dft = fftshift_3d(volume, rfft=False)
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = fftshift_3d(dft, rfft=True)

    # generate grid of DFT sample frequencies for a central slice in the xy-plane
    # these are a coordinate grid for the DFT
    grid = rotated_fftfreq_central_slice(
        image_shape=volume.shape,
        rotation_matrices=rotation_matrices,
        rotation_matrix_zyx=rotation_matrix_zyx,
        rfft=True,
        fftshift=True,
        spacing=1,
        device=dft.device,
    )  # (..., h, w, 3)

    # flip coordinates in redundant half transform
    conjugate_mask = grid[..., 2] < 0
    conjugate_mask = einops.repeat(conjugate_mask, '... -> ... 3')
    grid[conjugate_mask] *= -1
    conjugate_mask = conjugate_mask[..., 0]  # un-repeat

    # sample slices from DFT
    grid = fftfreq_to_dft_coordinates(
        frequencies=grid,
        image_shape=volume.shape,
        rfft=True
    )
    projections = interpolate_dft_3d(dft, grid)  # (b, h, w)

    # take complex conjugate of values from redundant half transform
    projections[conjugate_mask] = torch.conj(projections[conjugate_mask])

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2, ))
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))

    # unpadding
    if pad is True:
        projections = projections[:, pad_length:-pad_length, pad_length:-pad_length]
    return torch.real(projections)
