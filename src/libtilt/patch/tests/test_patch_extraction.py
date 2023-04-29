import torch

from libtilt.patch.patch_extraction import extract_patches, \
    _extract_patches_from_single_image


def test_extract_patches_from_single_image():
    """Test particle_extraction from single image."""
    img = torch.zeros((28, 28))
    img[::2, ::2] = 1
    positions = torch.tensor([14., 14.]).reshape((1, 2))
    patches = _extract_patches_from_single_image(
        image=img, positions=positions, output_image_sidelength=4
    )
    assert patches.shape == (1, 4, 4)
    expected_image = img[12:16, 12:16]
    assert torch.allclose(patches, expected_image, atol=1e-6)


def test_extract_patches():
    """Test extracting patches from a stack of images."""
    img = torch.zeros((2, 28, 28))
    img[:, ::2, ::2] = 1
    positions = torch.tensor([[14., 14.], [15., 15.]]).reshape((1, 2, 2))
    patches = extract_patches(
        images=img,  # (t, h, w)
        positions=positions,  # (b, t, 2)
        sidelength=4
    )  # -> (b, t, 4, 4)
    assert patches.shape == (1, 2, 4, 4)
    expected_image_0 = img[0, 12:16, 12:16]
    expected_image_1 = img[1, 13:17, 13:17]
    assert torch.allclose(patches[0, 0], expected_image_0, atol=1e-6)
    assert torch.allclose(patches[0, 1], expected_image_1, atol=1e-6)


def test_extract_patches_single_image():
    """Test particle_extraction from image stack."""
    img = torch.zeros((28, 28))
    img[::2, ::2] = 1
    positions = torch.tensor([[14., 14.], [15., 15.]])
    patches = extract_patches(
        images=img,  # (h, w)
        positions=positions,  # (b, 2)
        sidelength=4
    )  # -> (b, 1, 4, 4)
    assert patches.shape == (2, 1, 4, 4)
    expected_image_0 = img[12:16, 12:16]
    expected_image_1 = img[13:17, 13:17]
    assert torch.allclose(patches[0, 0], expected_image_0, atol=1e-6)
    assert torch.allclose(patches[1, 0], expected_image_1, atol=1e-6)