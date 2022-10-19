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
