import torch
from scipy.spatial.transform import Rotation as R

from libtilt.projection.fourier import extract_from_dft_3d
from libtilt.backprojection.fourier import insert_into_dft_3d
from libtilt.grids.central_slice import rotated_central_slice_grid
from libtilt.utils.fft import fftfreq_to_dft_coordinates


def test_fourier_slice_extraction_insertion_cycle_no_rotation():
    sidelength = 64
    volume = torch.zeros((sidelength, sidelength, sidelength), dtype=torch.complex64)
    weights = torch.zeros_like(volume, dtype=torch.float)
    input_slice = torch.complex(torch.arange(64*64).reshape(64, 64).float(), imag=torch.zeros((64, 64)))
    rotation = torch.eye(3).reshape(1, 3, 3)
    slice_coordinates_fftfreq = rotated_central_slice_grid(
        image_shape=volume.shape,
        rotation_matrices=rotation,
        rotation_matrix_zyx=False,
        rfft=False,
        fftshift=True,
    )
    slice_coordinates_dft = fftfreq_to_dft_coordinates(
        frequencies=slice_coordinates_fftfreq,
        image_shape=volume.shape,
        rfft=False,
    )
    volume_with_slice, weights = insert_into_dft_3d(
        data=input_slice.reshape(1, 64, 64),
        coordinates=slice_coordinates_dft,
        dft=volume,
        weights=weights
    )
    volume_with_slice[weights > 1e-3] /= weights[weights > 1e-3]
    output_slice = extract_from_dft_3d(volume_with_slice, coordinates=slice_coordinates_dft)
    assert torch.allclose(input_slice, output_slice)


def test_fourier_slice_extraction_insertion_cycle_with_rotation():
    sidelength = 64

    # initialise output volumes
    volume = torch.zeros((sidelength, sidelength, sidelength), dtype=torch.complex64)
    weights = torch.zeros_like(volume, dtype=torch.float)

    # generate slice data
    input_slice = torch.complex(torch.rand(64*64).reshape(64, 64).float(), imag=torch.zeros((64, 64)))

    # generate coordinates for rotated slice
    rotation = torch.tensor(R.random().as_matrix()).float()
    grid_fftfreq = rotated_central_slice_grid(
        image_shape=volume.shape,
        rotation_matrices=rotation,
        rotation_matrix_zyx=False,
        rfft=False,
        fftshift=True
    )
    grid_dft = fftfreq_to_dft_coordinates(grid_fftfreq, image_shape=volume.shape, rfft=False)

    # insert slice
    volume_with_slice, weights = insert_into_dft_3d(
        data=input_slice,
        coordinates=grid_dft,
        dft=volume,
        weights=weights
    )
    volume_with_slice[weights > 1e-3] /= weights[weights > 1e-3]

    # extract slice
    output_slice = extract_from_dft_3d(volume_with_slice, coordinates=grid_dft)

    # calculate error on all pixels which were inside the volume
    input_slice, output_slice = torch.real(input_slice), torch.real(output_slice)
    in_volume_idx = (grid_dft >= 0) & (grid_dft <= torch.tensor(volume.shape) - 1)
    in_volume_idx = torch.all(in_volume_idx, dim=-1)
    error = torch.mean(torch.abs((input_slice[in_volume_idx] - output_slice[in_volume_idx])))
    assert error < 0.15


