import einops
import torch

from .single_image import extract_at_integer_coordinates as _extract_from_single_image


def extract_at_integer_coordinates(
    images,  # (b1, h, w)
    positions,  # (b2, b1, 2)
    output_image_sidelength: int
):
    b1, h, w = images.shape
    b2, b1_2, c = positions.shape
    s = output_image_sidelength
    if b1 != b1_2:
        raise ValueError('batching dimensions do not match along expected axes.')
    elif c != 2:
        raise ValueError('positions must be 2D.')
    positions = einops.rearrange(positions, 'b2 b1 yx -> b1 b2 yx')
    output_iterator = (
        _extract_from_single_image(
            image=_image,
            positions=_positions,
            output_image_sidelength=output_image_sidelength
        )
        for _image, _positions in zip(images, positions)
    )
    output_images = torch.empty(size=(b1, b2, s, s))
    output_shifts = torch.empty(size=(b1, b2, 2), dtype=torch.float32)
    for idx, (_image, _shifts) in enumerate(output_iterator):
        output_images[idx] = _image
        output_shifts[idx] = _shifts
    output_images = einops.rearrange(output_images, 'b1 b2 h w -> b2 b1 h w')
    output_shifts = einops.rearrange(output_shifts, 'b1 b2 yx -> b2 b1 yx')
    return output_images, output_shifts
