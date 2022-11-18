import torch

from libtilt.particle_extraction.single_image import extract_at_integer_coordinates


def test_extract_at_integer_coordinates():
    """Test particle_extraction from single image."""
    img = torch.zeros((28, 28))
    img[::2, ::2] = 1
    positions = torch.tensor([14.2, 14.4]).reshape((1, 2))
    extracted_image, shifts = extract_at_integer_coordinates(
        image=img, positions=positions, output_image_sidelength=4
    )
    assert extracted_image.shape == (1, 4, 4)
    expected_image = img[12:16, 12:16]
    assert torch.allclose(extracted_image, expected_image)
    expected_shifts = torch.tensor([0.2, 0.4])
    assert torch.allclose(shifts, expected_shifts)