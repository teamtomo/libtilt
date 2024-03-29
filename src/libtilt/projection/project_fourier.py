import torch
import torch.nn.functional as F
import einops

from libtilt.grids import fftfreq_grid, rotated_central_slice_grid
from libtilt.fft_utils import fftfreq_to_dft_coordinates
from libtilt.interpolation import sample_dft_3d


def project_fourier(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    rotation_matrix_zyx: bool = False,
    pad: bool = True,
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
        `(..., d, d)` array of projection images.
    """
    # padding
    if pad is True:
        pad_length = volume.shape[-1] // 2
        volume = F.pad(volume, pad=[pad_length] * 6, mode='constant', value=0)

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=volume.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device
    )
    volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of rfft

    # make projections by taking central slices
    projections = extract_central_slices_rfft(
        dft=dft,
        image_shape=volume.shape,
        rotation_matrices=rotation_matrices,
        rotation_matrix_zyx=rotation_matrix_zyx
    )  # (..., h, w) rfft

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))  # recenter real space

    # unpad
    if pad is True:
        projections = projections[..., pad_length:-pad_length, pad_length:-pad_length]
    return torch.real(projections)


def extract_central_slices_rfft(
    dft: torch.Tensor,
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,
    rotation_matrix_zyx: bool,
):
    """Extract central slice from an fftshifted rfft."""
    # generate grid of DFT sample frequencies for a central slice in the xy-plane
    # these are a coordinate grid for the DFT
    grid = rotated_central_slice_grid(
        image_shape=image_shape,
        rotation_matrices=rotation_matrices,
        rotation_matrix_zyx=rotation_matrix_zyx,
        rfft=True,
        fftshift=True,
        device=dft.device,
    )  # (..., h, w, 3)

    # flip coordinates in redundant half transform
    conjugate_mask = grid[..., 2] < 0
    conjugate_mask = einops.repeat(conjugate_mask, '... -> ... 3')
    grid[conjugate_mask] *= -1
    conjugate_mask = conjugate_mask[..., 0]  # un-repeat

    # convert frequencies to array coordinates and sample from DFT
    grid = fftfreq_to_dft_coordinates(
        frequencies=grid,
        image_shape=image_shape,
        rfft=True
    )
    projections = sample_dft_3d(dft=dft, coordinates=grid)  # (..., h, w) rfft

    # take complex conjugate of values from redundant half transform
    projections[conjugate_mask] = torch.conj(projections[conjugate_mask])
    return projections
