import torch

from libtilt.atomic_models.coordinates_to_image import (
    coordinates_to_image_2d, coordinates_to_image_3d
)
from libtilt.pytest_utils import device_test


@device_test
def test_coordinates_to_image_2d():
    coordinates = torch.as_tensor([14, 14])
    image = coordinates_to_image_2d(coordinates=coordinates, image_shape=(28, 28))
    expected = torch.zeros((28, 28))
    expected[14, 14] = 1
    assert image.shape == (28, 28)
    assert torch.allclose(image, expected)


@device_test
def test_coordinates_to_image_3d():
    coordinates = torch.as_tensor([14, 14, 14])
    image = coordinates_to_image_3d(coordinates=coordinates, image_shape=(28, 28, 28))
    expected = torch.zeros((28, 28, 28))
    expected[14, 14, 14] = 1
    assert torch.allclose(image, expected)
