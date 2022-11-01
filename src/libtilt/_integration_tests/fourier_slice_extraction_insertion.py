import torch
from scipy.spatial.transform import Rotation as R

from libtilt.projection.fourier import extract_slices
from libtilt.backprojection.fourier import insert_slices
from libtilt.utils.coordinates import generate_rotated_slice_coordinates


def test_fourier_slice_extraction_insertion_cycle_no_rotation():
    sidelength = 64
    volume = torch.zeros((sidelength, sidelength, sidelength), dtype=torch.complex64)
    weights = torch.zeros_like(volume, dtype=torch.float)
    input_slice = torch.complex(torch.arange(64*64).reshape(64, 64).float(), imag=torch.zeros((64, 64)))
    rotation = torch.eye(3).reshape(1, 3, 3)
    slice_coordinates = generate_rotated_slice_coordinates(rotation, sidelength=sidelength)
    volume_with_slice, weights = insert_slices(
        slices=input_slice.reshape(1, 64, 64),
        slice_coordinates=slice_coordinates,
        dft=volume,
        weights=weights
    )
    volume_with_slice[weights > 1e-3] /= weights[weights > 1e-3]
    output_slice = extract_slices(volume_with_slice, slice_coordinates=slice_coordinates)
    assert torch.allclose(input_slice, output_slice)


def test_fourier_slice_extraction_insertion_cycle_with_rotation():
    sidelength = 64
    volume = torch.zeros((sidelength, sidelength, sidelength), dtype=torch.complex64)
    weights = torch.zeros_like(volume, dtype=torch.float)
    input_slice = torch.complex(torch.rand(64*64).reshape(64, 64).float(), imag=torch.zeros((64, 64)))
    rotation = torch.tensor(R.random().as_matrix()).float()
    slice_coordinates = generate_rotated_slice_coordinates(rotation, sidelength=sidelength)
    in_volume_idx = (slice_coordinates >= 0) & (slice_coordinates <= torch.tensor(volume.shape) - 1)
    in_volume_idx = torch.all(in_volume_idx, dim=-1)
    volume_with_slice, weights = insert_slices(
        slices=input_slice.reshape(1, 64, 64),
        slice_coordinates=slice_coordinates,
        dft=volume,
        weights=weights
    )
    volume_with_slice[weights > 1e-3] /= weights[weights > 1e-3]

    output_slice = extract_slices(volume_with_slice, slice_coordinates=slice_coordinates)
    input_slice = torch.real(input_slice).unsqueeze(0)
    output_slice = torch.real(output_slice)
    error = torch.mean(torch.abs((input_slice[in_volume_idx] - output_slice[in_volume_idx])))
    assert error < 0.15


