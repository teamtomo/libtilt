import torch
import einops

from libtilt.coordinate_utils import promote_2d_to_3d


def Rx(angles_degrees: torch.Tensor) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates (xyzw) around the X-axis."""
    angles_degrees = torch.tensor(angles_degrees).reshape(-1)
    angles_radians = torch.deg2rad(angles_degrees)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=len(angles_degrees)).clone()
    matrices[:, 1, 1] = c
    matrices[:, 1, 2] = -s
    matrices[:, 2, 1] = s
    matrices[:, 2, 2] = c
    return torch.squeeze(matrices)


def Ry(angles_degrees: torch.Tensor) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates (xyzw) around the Y-axis."""
    angles_degrees = torch.tensor(angles_degrees).reshape(-1)
    angles_radians = torch.deg2rad(angles_degrees)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=len(angles_degrees)).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 2] = s
    matrices[:, 2, 0] = -s
    matrices[:, 2, 2] = c
    return torch.squeeze(matrices)


def Rz(angles_degrees: torch.Tensor) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates (xyzw) around the Z-axis."""
    angles_degrees = torch.tensor(angles_degrees).reshape(-1)
    angles_radians = torch.deg2rad(angles_degrees)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=len(angles_degrees)).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    return torch.squeeze(matrices)


def S(shifts: torch.Tensor) -> torch.Tensor:
    """4x4 matrices for shifts.
    Shifts supplied can be 2D or 3D.
    """
    shifts = torch.tensor(shifts)
    if shifts.shape[-1] == 2:
        shifts = promote_2d_to_3d(shifts)
    shifts = shifts.reshape((-1, 3))
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=shifts.shape[0]).clone()
    matrices[:, 0:3, 3] = shifts
    return torch.squeeze(matrices)
