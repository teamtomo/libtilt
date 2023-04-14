from typing import Tuple, Optional

import torch

from .shapes_nd import circle_nd as _nd_circle


def sphere(
    radius: float,
    sidelength: int,
    center: Optional[Tuple[float, float, float]],
    smoothing_radius: float,
) -> torch.Tensor:
    mask = _nd_circle(
        radius=radius,
        sidelength=sidelength,
        center=center,
        smoothing_radius=smoothing_radius,
        ndim=3
    )
    return mask
