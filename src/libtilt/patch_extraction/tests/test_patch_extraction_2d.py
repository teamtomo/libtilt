import torch

from libtilt.patch_extraction.subpixel_square_patch_extraction import extract_squares, \
    _extract_square_patches_from_single_2d_image
from libtilt.pytest_utils import device_test


@device_test
def test_single_square_patch_from_single_image():
    """Test square patch extraction from single image."""
    img = torch.zeros((28, 28))
    img[::2, ::2] = 1
    positions = torch.tensor([14., 14.]).reshape((1, 2))
    patches = _extract_square_patches_from_single_2d_image(
        image=img, positions=positions, output_image_sidelength=4
    )
    assert patches.shape == (1, 4, 4)
    expected_image = img[12:16, 12:16]
    assert torch.allclose(patches, expected_image, atol=1e-6)


@device_test
def test_extract_square_patches_single():
    """Test extracting patches from a stack of images."""
    img = torch.zeros((2, 28, 28))
    img[:, ::2, ::2] = 1
    positions = torch.tensor([[14., 14.], [15., 15.]]).reshape((1, 2, 2))
    patches = extract_squares(
        image=img,  # (b2, h, w)
        positions=positions,  # (b1, b2, 2)
        sidelength=4
    )  # -> (b1, b2, 4, 4)
    assert patches.shape == (1, 2, 4, 4)
    expected_image_0 = img[0, 12:16, 12:16]
    expected_image_1 = img[1, 13:17, 13:17]
    assert torch.allclose(patches[0, 0], expected_image_0, atol=1e-6)
    assert torch.allclose(patches[0, 1], expected_image_1, atol=1e-6)


@device_test
def test_extract_square_patches_batched():
    """Test batched particle extraction from single image."""
    img = torch.zeros((28, 28))
    img[::2, ::2] = 1
    positions = torch.tensor([[14., 14.], [15., 15.]])
    patches = extract_squares(
        image=img,  # (h, w)
        positions=positions,  # (b, 2)
        sidelength=4
    )  # -> (b, 4, 4)
    assert patches.shape == (2, 4, 4)
    expected_image_0 = img[12:16, 12:16]
    expected_image_1 = img[13:17, 13:17]
    assert torch.allclose(patches[0], expected_image_0, atol=1e-6)
    assert torch.allclose(patches[1], expected_image_1, atol=1e-6)
