import torch

from libtilt.patch_extraction.patch_extraction_3d_subpixel import extract_cubic_patches, \
    _extract_cubic_patches_from_single_3d_image


def test_single_cubic_patch_from_single_image():
    """Test cubic patch extraction from single 3D image."""
    img = torch.zeros((28, 28, 28))
    img[::2, ::2, ::2] = 1
    positions = torch.tensor([14., 14., 14.]).reshape((1, 3))
    patches = _extract_cubic_patches_from_single_3d_image(
        image=img, positions=positions, sidelength=4
    )
    assert patches.shape == (1, 4, 4, 4)
    expected_image = img[12:16, 12:16, 12:16]
    assert torch.allclose(patches, expected_image, atol=1e-6)


def test_extract_cubic_patches():
    """Test extracting cubic patches from a 3D image."""
    img = torch.zeros((28, 28, 28))
    img[::2, ::2, ::2] = 1
    positions = torch.tensor([[14., 14., 14.], [15., 15., 15.]]).reshape((2, 3))
    patches = extract_cubic_patches(
        image=img,  # (d, h, w)
        positions=positions,  # (b, 3)
        sidelength=4
    )  # -> (b, 4, 4, 4)
    assert patches.shape == (2, 4, 4, 4)
    expected_image_0 = img[12:16, 12:16, 12:16]
    expected_image_1 = img[13:17, 13:17, 13:17]
    assert torch.allclose(patches[0], expected_image_0, atol=1e-6)
    assert torch.allclose(patches[1], expected_image_1, atol=1e-6)
