import torch

from .conical import fsc_conical


def xyz_fsc(
    a: torch.Tensor, b: torch.Tensor, cone_aperture: float = 30
) -> torch.Tensor:
    fsc_x = fsc_conical(a, b, cone_direction=(0, 0, 1), cone_aperture=cone_aperture)
    fsc_y = fsc_conical(a, b, cone_direction=(0, 1, 0), cone_aperture=cone_aperture)
    fsc_z = fsc_conical(a, b, cone_direction=(1, 0, 0), cone_aperture=cone_aperture)
    return torch.stack([fsc_x, fsc_y, fsc_z], dim=-1)  # (shells, zyx)
