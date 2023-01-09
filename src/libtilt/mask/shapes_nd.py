from typing import Literal, Optional, Sequence

import einops
import numpy as np
import torch

from .smoothing import _smooth_binary_image


def circle(
        sidelength: int,
        ndim: Literal[2, 3],
        radius: float,
        center: Optional[Sequence[float]] = None,
        smoothing_radius: float = 0
) -> torch.Tensor:
    if center is None:
        center = torch.Tensor([sidelength] * ndim) / 2
    indices = torch.Tensor(np.indices([sidelength] * ndim))
    indices = einops.rearrange(indices, 'coords ... -> ... coords')
    indices -= torch.Tensor(center)
    distances = torch.linalg.norm(indices, dim=-1)
    circle = torch.zeros(size=[sidelength] * ndim)
    circle[distances <= radius] = 1
    return _smooth_binary_image(circle, smoothing_radius=smoothing_radius)
