import torch
import torch.nn.functional as F
import einops

from libtilt.utils.coordinates import array_to_grid_sample
from libtilt.grids.central_slice import central_slice_grid
from libtilt.utils.fft import dft_center, fftshift_3d, _rfft_to_symmetrised_dft_3d, \
    _symmetrised_dft_to_dft_3d


def extract_slices(
    dft: torch.Tensor,
    slice_coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample batches of 2D images from a complex cubic volume at specified coordinates.

    The `dft` should be the result of

            volume -> fftshift(volume) -> fft3(volume) -> fftshift(volume)

    Coordinates should be ordered zyx, aligned with image dimensions.
    Coordinates should be array coordinates, spanning `[0, N-1]` for a dimension of length N.


    Parameters
    ----------
    dft: torch.Tensor
        (d, h, w) complex valued cubic volume (d == h == w) containing
        the discrete Fourier transform of a cubic volume.
    slice_coordinates: torch.Tensor
        (batch, h, w, zyx) array of coordinates at which `dft` should be sampled.

    Returns
    -------
    samples: torch.Tensor
        (batch, h, w) array of complex valued images sampled from the `dft`
    """
    # cannot sample complex tensors directly with grid_sample
    # c.f. https://github.com/pytorch/pytorch/issues/67634
    # workaround: treat real and imaginary parts as separate channels
    dft = einops.rearrange(torch.view_as_real(dft), 'd h w complex -> complex d h w')
    n_slices = slice_coordinates.shape[0]
    dft = einops.repeat(dft, 'complex d h w -> b complex d h w', b=n_slices)
    slice_coordinates = array_to_grid_sample(slice_coordinates, array_shape=dft.shape[-3:])
    slice_coordinates = einops.rearrange(slice_coordinates, 'b h w zyx -> b 1 h w zyx')

    # sample with border values at edges to increase sampling fidelity at nyquist
    samples = F.grid_sample(
        input=dft,
        grid=slice_coordinates,
        mode='bilinear',  # this is trilinear when input is volumetric
        padding_mode='border',
        align_corners=True,
    )

    # zero out samples from outside of cube
    inside = torch.logical_or(slice_coordinates > 0, slice_coordinates < 1)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
    inside = einops.repeat(inside, 'b d h w -> b 2 d h w')  # add channel dim
    samples[~inside] *= 0

    samples = einops.rearrange(samples, 'b complex 1 h w -> b h w complex')
    samples = torch.view_as_complex(samples.contiguous())
    return samples  # (b, h, w)


def project(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    zyx: bool = False,
    pad: bool = True,
) -> torch.Tensor:
    """Project a cubic volume by sampling a central slice through its DFT.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(b, 3, 3)` which rotate coordinates of central slice to be sampled.
    zyx: bool
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

    # calculate DFT
    dft = fftshift_3d(volume, rfft=False)
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = fftshift_3d(dft, rfft=True)
    dft = _rfft_to_symmetrised_dft_3d(dft)
    dft = _symmetrised_dft_to_dft_3d(dft)

    # generate grid of coordinates for central XY slice
    grid = central_slice_grid(
        sidelength=dft.shape[-1], zyx=zyx, device=volume.device
    )  # (h, w, 3)

    # rotate coordinate grid and recenter
    rotation_matrices = einops.rearrange(rotation_matrices, 'b i j -> b 1 1 i j')
    grid = einops.rearrange(grid, 'h w coords -> h w coords 1')
    grid = rotation_matrices @ grid
    grid = einops.rearrange(grid, 'b h w coords 1 -> b h w coords')
    if zyx is False:
        grid = torch.flip(grid, dims=(-1, ))
    grid = grid + dft_center(volume.shape, rfft=False, fftshifted=True)

    # sample slices from DFT
    projections = extract_slices(dft, grid)  # (b, h, w)
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))
    projections = torch.fft.ifftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))

    # unpadding
    if pad is True:
        projections = projections[:, pad_length:-pad_length, pad_length:-pad_length]
    return torch.real(projections)
