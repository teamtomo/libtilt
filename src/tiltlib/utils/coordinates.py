import einops
import torch
from torch.nn import functional as F


def promote_2d_to_3d(shifts: torch.Tensor) -> torch.Tensor:
    """Promote 2D coordinates to 3D with zeros in the last dimension.

    Last dimension of array should be of length 2.
    """
    shifts = F.pad(torch.tensor(shifts), pad=(0, 1), mode='constant', value=0)
    return torch.squeeze(shifts)


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """3D coordinates to 4D homogenous coordinates with ones in the last dimension.

    Last dimension of array should be of length 3.
    """
    return F.pad(torch.Tensor(coords), pad=(0, 1), mode='constant', value=1)


def generate_rotated_slice_coordinates(rotations: torch.Tensor, n: int) -> torch.Tensor:
    """Generate a (batch, n, n, 3) array of rotated central slice coordinates (ordered zyx).

    Parameters
    ----------
    rotations: torch.Tensor
        (batch, 3, 3) array of rotation matrices.
    n: int
        sidelength of square grid on which coordinates are generated.

    Returns
    -------
    coordinates: torch.Tensor
        (batch, n, n, zyx) array of coordinates.
    """
    # generate [x, y, z] coordinates centered on image_sidelength // 2
    x = y = torch.arange(n) - (n // 2)
    xx = einops.repeat(x, 'w -> h w', h=n)
    yy = einops.repeat(y, 'h -> h w', w=n)
    zz = torch.zeros(size=(n, n))
    xyz = einops.rearrange([xx, yy, zz], 'xyz h w -> 1 h w xyz 1')

    # rotate coordinates
    rotations = einops.rearrange(rotations, 'b i j -> b 1 1 i j')
    xyz = einops.rearrange(rotations @ xyz, 'b h w xyz 1 -> b h w xyz')

    # recenter
    xyz += n // 2
    zyx = torch.flip(xyz, dims=(-1,))
    return zyx
