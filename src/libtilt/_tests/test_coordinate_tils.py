import einops
import torch
import numpy as np

from libtilt.coordinate_utils import (
    array_to_grid_sample,
    grid_sample_to_array,
    add_positional_coordinate,
    homogenise_coordinates,
)
from libtilt.grids.coordinate_grid import coordinate_grid
from libtilt.pytest_utils import device_test


@device_test
def test_array_coordinates_to_grid_sample_coordinates_nd():
    array_shape = z, y, x = (4, 8, 12)
    array_coordinates = einops.rearrange(torch.tensor(np.indices(array_shape)),
                                         'zyx d h w -> d h w zyx')
    grid_sample_coordinates = array_to_grid_sample(array_coordinates,
                                                   array_shape=array_shape)

    expected_x = torch.linspace(-1, 1, x)
    expected_y = torch.linspace(-1, 1, y)
    expected_z = torch.linspace(-1, 1, z)

    assert torch.allclose(grid_sample_coordinates[0, 0, :, 0], expected_x)
    assert torch.allclose(grid_sample_coordinates[0, :, 0, 1], expected_y)
    assert torch.allclose(grid_sample_coordinates[:, 0, 0, 2], expected_z)


@device_test
def test_grid_sample_coordinates_to_array_coordinates_nd():
    array_shape = (4, 8, 12)
    expected_array_coordinates = einops.rearrange(
        torch.tensor(np.indices(array_shape)), 'zyx d h w -> d h w zyx'
    ).float()
    grid_sample_coordinates = array_to_grid_sample(
        expected_array_coordinates, array_shape=array_shape
    )
    array_coordinates = grid_sample_to_array(
        grid_sample_coordinates, array_shape=array_shape
    )
    assert torch.allclose(array_coordinates, expected_array_coordinates)


@device_test
def test_add_implied_coordinate_from_dimension():
    batch_of_stacked_2d_coords = torch.zeros(size=(1, 5, 2))  # (b, stack, 2)
    result = add_positional_coordinate(batch_of_stacked_2d_coords, dim=1)
    expected = torch.zeros(size=(1, 5, 3))
    expected[0, :, 2] = torch.arange(5)
    assert torch.allclose(result, expected)


@device_test
def test_add_implied_coordinate_from_dimension_prepend():
    batch_of_stacked_2d_coords = torch.zeros(size=(1, 5, 2))  # (b, stack, 2)
    result = add_positional_coordinate(batch_of_stacked_2d_coords, dim=1,
                                       prepend=True)
    expected = torch.zeros(size=(1, 5, 3))
    expected[0, :, 0] = torch.arange(5)
    assert torch.allclose(result, expected)


@device_test
def test_get_grid_coordinates():
    coords = coordinate_grid(image_shape=(3, 2))
    assert coords.shape == (3, 2, 2)
    expected = torch.tensor(
        [[[0., 0.],
          [0., 1.]],

         [[1., 0.],
          [1., 1.]],

         [[2., 0.],
          [2., 1.]]]
    )
    assert torch.allclose(coords, expected)


@device_test
def test_homogenise_coordinates():
    coords = torch.rand(size=(2, 3))
    homogenised = homogenise_coordinates(coords)
    assert torch.all(homogenised[..., :3] == coords)
    assert torch.all(homogenised[..., 3] == 1)



