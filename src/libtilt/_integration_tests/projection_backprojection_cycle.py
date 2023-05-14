import einops
import torch
import torch.nn.functional as F
from scipy.stats import special_ortho_group

from libtilt.grids.central_slice import central_slice_grid

from libtilt.projection.fourier import extract_from_dft_3d
from libtilt.shift.fourier_shift import fourier_shift_2d, phase_shift_dft_2d
from libtilt.backprojection.fourier import insert_into_dft_3d
from libtilt.grids.fftfreq import _grid_sinc2
from libtilt.utils.fft import _symmetrised_dft_to_dft_3d, dft_center
from libtilt.fsc import fsc


def test_projection_backprojection_cycle():
    N_IMAGES = 5000
    BATCH_SIZE = 100
    RECONSTRUCT_SYMMETRISED_DFT = True
    DO_2X_ZERO_PADDING = True
    torch.manual_seed(42)

    # create some volumetric data and normalise
    volume = torch.zeros((32, 32, 32))
    volume[8:24, 8:24, 8:24] = torch.rand(size=(16, 16, 16))
    volume -= torch.mean(volume)
    volume /= torch.std(volume)

    # zero pad
    if DO_2X_ZERO_PADDING:
        p = volume.shape[0] // 4
        volume = F.pad(volume, pad=[p] * 6)

    # rotation matrices for projection (operate on xyz column vectors)
    rotations = torch.tensor(
        special_ortho_group.rvs(dim=3, size=N_IMAGES, random_state=42)).float()

    # generate grid of coordinates for central XY slice
    grid = central_slice_grid(sidelength=volume.shape[-1], zyx=False)  # (h, w, 3)

    # rotate coordinate grid and recenter
    rotation_matrices = einops.rearrange(rotations, 'b i j -> b 1 1 i j')
    grid = einops.rearrange(grid, 'h w coords -> h w coords 1')
    grid = rotation_matrices @ grid
    grid = einops.rearrange(grid, 'b h w coords 1 -> b h w coords')
    grid = torch.flip(grid, dims=(-1,))
    grid = grid + dft_center(volume.shape, rfft=False, fftshifted=True)

    # pre-correct for effects of subsequent Fourier space interpolation
    sinc2 = _grid_sinc2(volume.shape)
    volume *= sinc2

    # calculate DFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.fftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2, -1))

    # take slices
    dft_slices = extract_from_dft_3d(dft, grid)  # (b, h, w)

    # shift the projections (dft slices) by phase shifting
    std = 3
    shifts = torch.normal(mean=0, std=std, size=(N_IMAGES, 2))
    shifted_slices = phase_shift_dft_2d(
        dft=dft_slices,
        shifts=shifts,
        image_shape=dft_slices.shape[-2:],
        rfft=False,
        fftshifted=True
    )

    # generate images from shifted projections
    shifted_projections = torch.fft.ifftshift(shifted_slices, dim=(-2, -1))
    shifted_projections = torch.fft.ifftn(shifted_projections, dim=(-2, -1))
    shifted_projections = torch.fft.ifftshift(shifted_projections, dim=(-2, -1))
    shifted_projections = torch.real(shifted_projections)

    # phase shift images back again
    centered_projections = fourier_shift_2d(shifted_projections, shifts=-shifts)
    dft_slices = torch.fft.fftshift(centered_projections, dim=(-2, -1))
    dft_slices = torch.fft.fftn(dft_slices, dim=(-2, -1))
    dft_slices = torch.fft.fftshift(dft_slices, dim=(-2, -1))

    # reconstruct from recentered dft slices
    d = volume.shape[0]
    if RECONSTRUCT_SYMMETRISED_DFT is True:
        d += 1
    reconstruction = torch.zeros(size=(d, d, d), dtype=torch.complex64)
    weights = torch.zeros(size=(d, d, d), dtype=torch.float32)

    splits = torch.arange(start=0, end=N_IMAGES + BATCH_SIZE, step=BATCH_SIZE)
    for start, end in zip(splits, splits[1:]):
        end = min(end, N_IMAGES - 1)
        reconstruction, weights = insert_into_dft_3d(
            data=dft_slices[start:end],
            coordinates=grid[start:end],
            dft=reconstruction,
            weights=weights,
        )

    # reweight data in Fourier space
    valid_weights = weights > 1e-3
    reconstruction[valid_weights] /= weights[valid_weights]

    # desymmetrise dft
    if RECONSTRUCT_SYMMETRISED_DFT is True:
        reconstruction = _symmetrised_dft_to_dft_3d(reconstruction, inplace=True)

    # back to real space
    reconstruction = torch.fft.ifftshift(reconstruction, dim=(-3, -2, -1))
    reconstruction = torch.fft.ifftn(reconstruction, dim=(-3, -2, -1))
    reconstruction = torch.fft.ifftshift(reconstruction, dim=(-3, -2, -1))
    reconstruction = torch.real(reconstruction)

    # gridding correction
    reconstruction /= sinc2

    # fsc
    _fsc = fsc(reconstruction, volume)
    assert torch.all(_fsc[-16:] > 0.99)

