import einops
import torch
from torch.nn import functional as F

from libtilt.grids import coordinate_grid
from libtilt.shift.shift_image import shift_3d
from libtilt.coordinate_utils import array_to_grid_sample
from libtilt.fft_utils import dft_center


def extract_cubes(
    image: torch.Tensor, positions: torch.Tensor, sidelength: int,
):
    """Extract cubic patches from a 3D image at positions with subpixel precision.

    Patches are extracted at the nearest integer coordinates then phase shifted
    such that the requested position is at the center of the patch.

    Parameters
    ----------
    image: torch.Tensor
        `(d, h, w)` array containing a 3D image.
    positions: torch.Tensor
        `(..., 3)`
    sidelength: int
        Sidelength of cubic patches extracted from `image`.


    Returns
    -------
    patches: torch.Tensor
        `(..., sidelength, sidelength, sidelength)` array of cubic patches from `image`
        with their centers at `positions`.
    """
    # pack arbitrary dimensions up into one new batch dim 'b'
    positions, ps = einops.pack([positions], pattern='* zyx')

    # extract cubic patches from 3D image
    patches = _extract_cubic_patches_from_single_3d_image(
        image=image, positions=positions, sidelength=sidelength
    )

    # reassemble patches into arbitrary dimensional stacks
    [patches] = einops.unpack(patches, pattern='* d h w', packed_shapes=ps)
    return patches


def _extract_cubic_patches_from_single_3d_image(
    image: torch.Tensor,  # (h, w)
    positions: torch.Tensor,  # (b, 3) zyx
    sidelength: int,
) -> torch.Tensor:
    d, h, w = image.shape
    b, _ = positions.shape

    # find integer positions and shifts to be applied
    integer_positions = torch.round(positions)
    shifts = integer_positions - positions

    # generate coordinate grids for sampling around each integer position
    # add 1px border to leave space for subpixel phase shifting
    pd, ph, pw = (sidelength + 2, sidelength + 2, sidelength + 2)
    coordinates = coordinate_grid(
        image_shape=(pd, ph, pw),
        center=dft_center(
            image_shape=(pd, ph, pw),
            rfft=False, fftshifted=True,
            device=image.device
        ),
        device=image.device
    )  # (d, h, w, 3)
    broadcastable_positions = einops.rearrange(integer_positions, 'b zyx -> b 1 1 1 zyx')
    grid = coordinates + broadcastable_positions  # (b, d, h, w, 3)

    # extract patches, grid sample handles boundaries
    patches = F.grid_sample(
        input=einops.repeat(image, 'd h w -> b 1 d h w', b=b),
        grid=array_to_grid_sample(grid, array_shape=(d, h, w)),
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )
    patches = einops.rearrange(patches, 'b 1 d h w -> b d h w')

    # phase shift to center images then remove border
    patches = shift_3d(images=patches, shifts=shifts)
    patches = F.pad(patches, pad=(-1, -1, -1, -1, -1, -1))
    return patches
