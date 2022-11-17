import torch

from libtilt.extraction.image_stack import extract_at_integer_coordinates


def test_extract_at_integer_coordinates():
    """Test extraction from image stack."""
    img = torch.zeros((2, 28, 28))
    img[:, ::2, ::2] = 1
    positions = torch.tensor([[14.2, 14.4], [15.1, 15.3]]).reshape((1, 2, 2))
    # see docstring for expected semantics
    extracted_image, shifts = extract_at_integer_coordinates(
        images=img,  # (b1, h, w)
        positions=positions,  # (b2, b1, 2)
        output_image_sidelength=4
    )  # -> (b2, b1, 4, 4)
    assert extracted_image.shape == (1, 2, 4, 4)
    expected_image_0 = img[0, 12:16, 12:16]
    expected_image_1 = img[1, 13:17, 13:17]
    assert torch.allclose(extracted_image[0, 0], expected_image_0)
    assert torch.allclose(extracted_image[0, 1], expected_image_1)
    expected_shifts = torch.tensor([[0.2, 0.4], [0.1, 0.3]]).reshape((1, 2, 2))
    assert torch.allclose(shifts, expected_shifts)