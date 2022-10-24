import einops
import torch
import numpy as np

from libtilt.coordinate_utils import (
    array_coordinates_to_grid_sample_coordinates,
    _array_coordinates_to_grid_sample_coordinates_1d,
    grid_sample_coordinates_to_array_coordinates,
    _grid_sample_coordinates_to_array_coordinates_1d,
    add_implied_coordinate_from_dimension,
    get_array_coordinates,
    generate_rotated_slice_coordinates,
    homogenise_coordinates,
    promote_2d_shifts_to_3d,
)


def test_array_coordinates_to_grid_sample_coordinates_1d():
    n = 5
    array_coordinates = torch.arange(n)
    grid_sample_coordinates = _array_coordinates_to_grid_sample_coordinates_1d(array_coordinates,
                                                                               dim_length=n)
    expected = torch.linspace(-1, 1, n)
    assert torch.allclose(grid_sample_coordinates, expected)


def test_array_coordinates_to_grid_sample_coordinates_nd():
    array_shape = z, y, x = (4, 8, 12)
    array_coordinates = einops.rearrange(torch.tensor(np.indices(array_shape)),
                                         'zyx d h w -> d h w zyx')
    grid_sample_coordinates = array_coordinates_to_grid_sample_coordinates(array_coordinates,
                                                                           array_shape=array_shape)

    expected_x = torch.linspace(-1, 1, x)
    expected_y = torch.linspace(-1, 1, y)
    expected_z = torch.linspace(-1, 1, z)

    assert torch.allclose(grid_sample_coordinates[0, 0, :, 0], expected_x)
    assert torch.allclose(grid_sample_coordinates[0, :, 0, 1], expected_y)
    assert torch.allclose(grid_sample_coordinates[:, 0, 0, 2], expected_z)


def test_grid_sample_coordinates_to_array_coordinates_1d():
    n = 5
    grid_sample_coordinates = torch.linspace(-1, 1, n)
    array_coordinates = _grid_sample_coordinates_to_array_coordinates_1d(grid_sample_coordinates,
                                                                         dim_length=n)
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


def test_add_implied_coordinate_from_dimension():
    batch_of_stacked_2d_coords = torch.zeros(size=(1, 5, 2))  # (b, stack, 2)
    result = add_implied_coordinate_from_dimension(batch_of_stacked_2d_coords, dim=1)
    expected = torch.zeros(size=(1, 5, 3))
    expected[0, :, 2] = torch.arange(5)
    assert torch.allclose(result, expected)


def test_add_implied_coordinate_from_dimension_prepend():
    batch_of_stacked_2d_coords = torch.zeros(size=(1, 5, 2))  # (b, stack, 2)
    result = add_implied_coordinate_from_dimension(batch_of_stacked_2d_coords, dim=1,
                                                   prepend_new_coordinate=True)
    expected = torch.zeros(size=(1, 5, 3))
    expected[0, :, 0] = torch.arange(5)
    assert torch.allclose(result, expected)


def test_get_grid_coordinates():
    coords = get_array_coordinates(grid_dimensions=(3, 2))
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


def test_homogenise_coordinates():
    coords = torch.rand(size=(2, 3))
    homogenised = homogenise_coordinates(coords)
    assert torch.all(homogenised[..., :3] == coords)
    assert torch.all(homogenised[..., 3] == 1)


def test_generate_rotated_slice_coordinates():
    # generate an unrotated slice for a 4x4x4 volume.
    rotation = torch.eye(3)
    slice_coordinates = generate_rotated_slice_coordinates(rotation, sidelength=4)

    assert slice_coordinates.shape == (1, 4, 4, 3)
    slice_coordinates = einops.rearrange(slice_coordinates, '1 i j zyx -> i j zyx')

    # all z coordinates should be in the middle of the volume, slice is unrotated
    assert torch.all(slice_coordinates[..., 0] == 2)

    # y coordinates should be 0-3 repeated across columns
    assert torch.all(slice_coordinates[..., 1] == torch.tensor([[0],
                                                                [1],
                                                                [2],
                                                                [3]]))

    # x coordinates should be 0-3 repeated across rows
    assert torch.all(slice_coordinates[..., 2] == torch.tensor([[0, 1, 2, 3]]))


def test_promote_2d_shifts_to_3d():
    shifts_2d = torch.tensor([1, 1])
    shifts_3d = promote_2d_shifts_to_3d(shifts_2d)
    assert torch.all(shifts_3d[..., :2] == shifts_2d)
    assert torch.all(shifts_3d[..., 2] == 0)
