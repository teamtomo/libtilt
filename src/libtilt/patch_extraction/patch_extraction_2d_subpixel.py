import einops
import torch
from torch.nn import functional as F

from libtilt.grids import coordinate_grid
from libtilt.shift.shift_image import shift_2d
from libtilt.coordinate_utils import array_to_grid_sample
from libtilt.fft_utils import dft_center


def extract_patches_2d(
    images: torch.Tensor, positions: torch.Tensor, sidelength: int,
):
    """Extract patches from 2D images at positions with subpixel precision.

    Patches are extracted at the nearest integer coordinates then phase shifted
    such that the requested position is at the center of the patch.

    Parameters
    ----------
    images: torch.Tensor
        `(t, h, w)` or `(h, w)` array of 2D images.
    positions: torch.Tensor
        `(..., t, 2)` or `(..., 2)` array of coordinates for patch centers.
    sidelength: int
        Sidelength of square 2D patches extracted from `images`.


    Returns
    -------
    patches: torch.Tensor
        `(..., t, sidelength, sidelength)` array of patches from `images` with their
        centers at `positions`.
    """
    if images.ndim == 2:
        images = einops.rearrange(images, 'h w -> 1 h w')
        positions = einops.rearrange(positions, '... yx -> ... 1 yx')
    positions, ps = einops.pack([positions], pattern='* t yx')
    positions = einops.rearrange(positions, 'b t yx -> t b yx')
    patches = einops.rearrange(
        [
            _extract_patches_from_single_image(
                image=_image,
                positions=_positions,
                output_image_sidelength=sidelength
            )
            for _image, _positions
            in zip(images, positions)
        ],
        pattern='t b h w -> b t h w'
    )
    [patches] = einops.unpack(patches, pattern='* t h w', packed_shapes=ps)
    return patches


def _extract_patches_from_single_image(
    image: torch.Tensor,  # (h, w)
    positions: torch.Tensor,  # (b, 2) yx
    output_image_sidelength: int,
) -> torch.Tensor:
    h, w = image.shape
    b, _ = positions.shape

    # find integer positions and shifts to be applied
    integer_positions = torch.round(positions)
    shifts = integer_positions - positions

    # generate coordinate grids for sampling around each integer position
    # add 1px border to leave space for subpixel phase shifting
    ph, pw = (output_image_sidelength + 2, output_image_sidelength + 2)
    coordinates = coordinate_grid(
        image_shape=(ph, pw),
        center=dft_center((ph, pw), rfft=False, fftshifted=True, device=image.device),
        device=image.device
    )  # (h, w, 2)
    broadcastable_positions = einops.rearrange(integer_positions, 'b yx -> b 1 1 yx')
    grid = coordinates + broadcastable_positions  # (b, h, w, 2)

    # extract patches, grid sample handles boundaries
    patches = F.grid_sample(
        input=einops.repeat(image, 'h w -> b 1 h w', b=b),
        grid=array_to_grid_sample(grid, array_shape=(h, w)),
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )
    patches = einops.rearrange(patches, 'b 1 h w -> b h w')

    # phase shift to center images then remove border
    patches = shift_2d(images=patches, shifts=shifts)
    patches = F.pad(patches, pad=(-1, -1, -1, -1))
    return patches
