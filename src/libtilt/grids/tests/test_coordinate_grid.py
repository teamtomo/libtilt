import einops
import torch

from libtilt.grids import coordinate_grid


def test_coordinate_grid_simple():
    image_shape = (5, 3, 2)
    result = coordinate_grid(
        image_shape=image_shape,
        center=None,
    )
    assert result.shape == (5, 3, 2, 3)
    assert torch.allclose(result[0, 0, 0], torch.tensor([0, 0, 0], dtype=torch.float))
    assert torch.allclose(result[4, 2, 1], torch.tensor([4, 2, 1], dtype=torch.float))


def test_coordinate_grid_centered():
    image_shape = (28, 28)
    result = coordinate_grid(
        image_shape=image_shape,
        center=(14, 14)
    )
    assert result.shape == (28, 28, 2)
    assert torch.allclose(result[0, 0], torch.tensor([-14, -14], dtype=torch.float))


def test_coordinate_grid_centered_batched():
    image_shape = (28, 28)
    centers = [[0, 0], [14, 14]]
    result = coordinate_grid(
        image_shape=image_shape,
        center=centers,
    )
    assert result.shape == (2, 28, 28, 2)
    assert torch.allclose(result[0, 0, 0], torch.as_tensor([0, 0], dtype=torch.float))
    assert torch.allclose(result[1, 0, 0],
                          torch.as_tensor([-14, -14], dtype=torch.float))


def test_coordinate_grid_centered_stacked():
    image_shape = (28, 28)
    centers = [[0, 0], [14, 14]]
    centers = einops.rearrange(torch.as_tensor(centers), 'b i -> b 1 1 i')
    result = coordinate_grid(
        image_shape=image_shape,
        center=centers,
    )
    assert result.shape == (2, 1, 1, 28, 28, 2)
    assert torch.allclose(result[0, 0, 0, 0, 0], torch.as_tensor([0, 0]).float())
    assert torch.allclose(result[1, 0, 0, 0, 0], torch.as_tensor([-14, -14]).float())


def test_coordinate_with_norm():
    image_shape = (5, 5)
    result = coordinate_grid(
        image_shape=image_shape,
        norm=True,
    )
    assert result.shape == (5, 5)
    assert torch.allclose(result[0, 0], torch.tensor([0], dtype=torch.float))
    assert torch.allclose(result[1, 1], torch.tensor([2 ** 0.5], dtype=torch.float))
