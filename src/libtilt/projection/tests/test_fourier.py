import einops
import pytest
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from libtilt.projection.fourier import extract_slices, project


def test_extract_slices():
    volume = torch.zeros(4, 4, 4, dtype=torch.complex64)

    # insert unrotated slice
    slice = torch.arange(4**2).reshape(4, 4)
    volume[2, :, :] = slice

    # construct slice coordinates
    x = y = torch.arange(4)
    xx = einops.repeat(x, 'w -> h w', h=4)
    yy = einops.repeat(y, 'h -> h w', w=4)
    zz = torch.ones_like(xx) * 2
    slice_coordinates = einops.rearrange([zz, yy, xx], 'zyx h w -> 1 h w zyx')  # add batch dim

    extracted_slice = extract_slices(dft=volume, slice_coordinates=slice_coordinates)
    error = F.mse_loss(slice, torch.real(extracted_slice.squeeze()))
    assert error < 1e-10


def test_project_no_rotation():
    volume = torch.zeros((10, 10, 10))
    volume[5, 5, 5] = 1

    # no rotation
    rotation_matrix = torch.eye(3).reshape(1, 3, 3)
    projection = project(volume, rotation_matrix)
    expected = torch.sum(volume, dim=0)
    assert torch.allclose(projection, expected)


@pytest.mark.xfail
def test_project_with_rotation():
    volume = torch.zeros((10, 10, 10))
    volume[5, 5, 5] = 1

    # with rotation
    # (5, 5, 5) is rotation center so projection shouldn't change...
    rotation_matrix = torch.tensor(R.random(num=1).as_matrix()).reshape(1, 3, 3).float()
    projection = project(volume, rotation_matrix)
    expected = torch.sum(volume, dim=0)
    assert torch.allclose(projection, expected, atol=1e-5)  # this should be closer... fft size?