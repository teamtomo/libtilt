import torch
import torch.nn.functional as F

from libtilt.grids import fftfreq_grid, rotated_central_slice_grid
from libtilt.fft_utils import rfft_shape, fftfreq_to_dft_coordinates
from libtilt.interpolation.interpolate_dft_3d import insert_into_dft_3d


def backproject_in_fourier_space(
    images: torch.Tensor,  # (b, h, w)
    rotation_matrices: torch.Tensor,  # (b, 3, 3)
    rotation_matrix_zyx: bool = False,
    pad: bool = True,
    do_gridding_correction: bool = True,
):
    """Perform a 3D reconstruction from a set of 2D projection images.

    Parameters
    ----------
    images: torch.Tensor
        `(batch, h, w)` array of 2D projection images.
    rotation_matrices: torch.Tensor
        `(batch, 3, 3)` array of rotation matrices for insert of `images`.
        Rotation matrices left-multiply column vectors containing coordinates.
    pad: bool
        Whether to pad the input images 2x (`True`) or not (`False`).
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
    if h != w:
        raise ValueError('images must be square.')
    if pad is True:
        p = images.shape[-1] // 4
        images = F.pad(images, pad=[p] * 4)

    # construct shapes
    b, h, w = images.shape
    volume_shape = (w, w, w)

    # initialise output volume and volume for keeping track of weights
    dft = torch.zeros(
        size=rfft_shape(volume_shape), dtype=torch.complex64, device=images.device
    )
    weights = torch.zeros_like(dft, dtype=torch.float32)

    # calculate DFTs of images
    images = torch.fft.fftshift(images, dim=(-2, -1))
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = torch.fft.fftshift(images, dim=(-2,))

    # generate grid of rotated slice coordinates for each element in image DFTs
    grid = rotated_central_slice_grid(
        image_shape=volume_shape,
        rotation_matrices=rotation_matrices,
        rotation_matrix_zyx=rotation_matrix_zyx,
        rfft=True,
        fftshift=True,
        device=images.device
    )  # centered on DC of DFT

    # flip coordinates in redundant half transform and take conjugate value
    conjugate_mask = grid[..., 2] < 0
    grid[conjugate_mask] *= -1
    images[conjugate_mask] = torch.conj(images[conjugate_mask])

    # calculate actual coordinates into DFT from rotated fftfreq grid
    grid = fftfreq_to_dft_coordinates(grid, image_shape=volume_shape, rfft=True)

    # insert data into DFT
    dft, weights = insert_into_dft_3d(
        data=images,
        coordinates=grid,
        dft=dft,
        weights=weights
    )
    valid_weights = weights > 1e-3
    dft[valid_weights] /= weights[valid_weights]
    dft = torch.fft.ifftshift(dft, dim=(-3, -2,))
    dft = torch.fft.irfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.ifftshift(dft, dim=(-3, -2, -1))
    if do_gridding_correction is True:
        grid = fftfreq_grid(
            image_shape=dft.shape,
            rfft=False,
            fftshift=True,
            norm=True,
            device=dft.device
        )
        dft = dft * torch.sinc(grid) ** 2
    if pad is True:  # un-pad
        dft = F.pad(dft, pad=[-p] * 6)
    return torch.real(dft)
