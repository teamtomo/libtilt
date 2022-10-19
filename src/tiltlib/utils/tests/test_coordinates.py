import einops
import torch
import numpy as np


from tiltlib.utils.coordinates import array_coordinates_to_grid_sample_coordinates, \
    _array_coordinates_to_grid_sample_coordinates_1d, grid_sample_coordinates_to_array_coordinates, \
    _grid_sample_coordinates_to_array_coordinates_1d


def test_array_coordinates_to_grid_sample_coordinates_1d():
    n = 5
    array_coordinates = torch.arange(n)
    grid_sample_coordinates = _array_coordinates_to_grid_sample_coordinates_1d(array_coordinates, dim_length=n)
    expected = torch.linspace(-1, 1, n)
    assert torch.allclose(grid_sample_coordinates, expected)


def test_array_coordinates_to_grid_sample_coordinates_nd():
    array_shape = z, y, x = (4, 8, 12)
    array_coordinates = einops.rearrange(torch.tensor(np.indices(array_shape)), 'zyx d h w -> d h w zyx')
    grid_sample_coordinates = array_coordinates_to_grid_sample_coordinates(array_coordinates, array_shape=array_shape)

    expected_x = torch.linspace(-1, 1, x)
    expected_y = torch.linspace(-1, 1, y)
    expected_z = torch.linspace(-1, 1, z)

    assert torch.allclose(grid_sample_coordinates[0, 0, :, 0], expected_x)
    assert torch.allclose(grid_sample_coordinates[0, :, 0, 1], expected_y)
    assert torch.allclose(grid_sample_coordinates[:, 0, 0, 2], expected_z)


def test_grid_sample_coordinates_to_array_coordinates_1d():
    n = 5
    grid_sample_coordinates = torch.linspace(-1, 1, n)
    array_coordinates = _grid_sample_coordinates_to_array_coordinates_1d(grid_sample_coordinates, dim_length=n)
    expected = torch.arange(n).float()
    assert torch.allclose(array_coordinates, expected)


def test_grid_sample_coordinates_to_array_coordinates_nd():
    array_shape = (4, 8, 12)
    expected_array_coordinates = einops.rearrange(
        torch.tensor(np.indices(array_shape)), 'zyx d h w -> d h w zyx'
    ).float()
    grid_sample_coordinates = array_coordinates_to_grid_sample_coordinates(
        expected_array_coordinates, array_shape=array_shape
    )
    array_coordinates = grid_sample_coordinates_to_array_coordinates(
        grid_sample_coordinates, array_shape=array_shape
    )
    assert torch.allclose(array_coordinates, expected_array_coordinates)