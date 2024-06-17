import torch
from scipy.spatial.transform import Rotation as R

from libtilt.projection.project_fourier import project_fourier
from libtilt.pytest_utils import device_test


@device_test
def test_project_no_rotation():
    volume = torch.zeros((10, 10, 10))
    volume[5, 5, 5] = 1

    # no rotation
    rotation_matrix = torch.eye(3).reshape(1, 3, 3)
    projection = project_fourier(volume, rotation_matrix)
    expected = torch.sum(volume, dim=0)
    assert torch.allclose(projection, expected)


@device_test
def test_project_with_rotation():
    volume = torch.zeros((10, 10, 10))
    volume[5, 5, 5] = 1

    # with rotation
    # (5, 5, 5) is rotation center so projection shouldn't change...
    rotation_matrix = torch.tensor(R.random(num=1).as_matrix()).reshape(1, 3, 3).float()
    projection = project_fourier(volume, rotation_matrix)
    expected = torch.sum(volume, dim=0)
    assert torch.allclose(projection, expected, atol=1e-3)
