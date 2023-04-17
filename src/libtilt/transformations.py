"""4x4 matrices for rotations and translations.

Functions in this module generate matrices which left-multiply column vectors containing
`xyzw` or `zyxw` homogenous coordinates.
"""

import torch
import einops


def Rx(angles_degrees: torch.Tensor, zyx: bool = False) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates around the X-axis.

    Parameters
    ----------
    angles_degrees: torch.Tensor
        `(..., )` array of angles
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogenous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    angles_degrees = torch.atleast_1d(torch.as_tensor(angles_degrees))
    angles_packed, ps = einops.pack([angles_degrees], pattern='*')  # to 1d
    n = angles_packed.shape[0]
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, 1, 1] = c
    matrices[:, 1, 2] = -s
    matrices[:, 2, 1] = s
    matrices[:, 2, 2] = c
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def Ry(angles_degrees: torch.Tensor, zyx: bool = False) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates around the Y-axis.

    Parameters
    ----------
    angles_degrees: torch.Tensor
        `(..., )` array of angles
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogenous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    angles_degrees = torch.atleast_1d(torch.as_tensor(angles_degrees))
    angles_packed, ps = einops.pack([angles_degrees], pattern='*')  # to 1d
    n = angles_packed.shape[0]
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 2] = s
    matrices[:, 2, 0] = -s
    matrices[:, 2, 2] = c
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def Rz(angles_degrees: torch.Tensor, zyx: bool = False) -> torch.Tensor:
    """4x4 matrices for a rotation of homogenous coordinates around the Z-axis.

    Parameters
    ----------
    angles_degrees: torch.Tensor
        `(..., )` array of angles
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogenous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    angles_degrees = torch.atleast_1d(torch.as_tensor(angles_degrees))
    angles_packed, ps = einops.pack([angles_degrees], pattern='*')  # to 1d
    n = angles_packed.shape[0]
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def T(shifts: torch.Tensor) -> torch.Tensor:
    """4x4 matrices for translations.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 3)` array of shifts.
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogenous coordinates.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 4, 4)` array of 4x4 shift matrices.
    """
    shifts = torch.atleast_1d(torch.as_tensor(shifts))
    shifts, ps = einops.pack([shifts], pattern='* coords')  # to 2d
    n = shifts.shape[0]
    matrices = einops.repeat(torch.eye(4), 'i j -> n i j', n=n).clone()
    matrices[:, :3, 3] = shifts
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices
