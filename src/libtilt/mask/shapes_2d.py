from typing import Tuple, Optional

import torch

from .shapes_nd import circle as _nd_circle


def circle(
    radius: float,
    sidelength: int,
    center: Optional[Tuple[float, float]] = None,
    smoothing_radius: float = 0,
) -> torch.Tensor:
    mask = _nd_circle(
        radius=radius,
        sidelength=sidelength,
        center=center,
        smoothing_radius=smoothing_radius,
        ndim=2
    )
    return mask
