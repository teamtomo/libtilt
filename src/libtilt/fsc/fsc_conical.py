from typing import Tuple

import einops
import torch

from .fsc import fsc as _fsc
from ..grids import fftfreq_grid
from ..shapes.geometry_utils import _angle_between_vectors


def fsc_conical(
    a: torch.Tensor,
    b: torch.Tensor,
    cone_direction: Tuple[float, float, float],
    cone_aperture: float,
):
    vectors = fftfreq_grid(
        image_shape=a.shape,
        rfft=True,
        fftshift=False,
    )  # (..., 3)
    vectors /= einops.reduce(vectors ** 2, '... vec -> ... 1', reduction='sum') ** 0.5
    cone_direction = torch.as_tensor(cone_direction, dtype=torch.float)
    cone_direction /= torch.linalg.norm(cone_direction)
    angles = _angle_between_vectors(vectors, cone_direction)
    acute_bound = cone_aperture / 2
    obtuse_bound = 180 - acute_bound
    in_cone_idx = torch.logical_or(angles <= acute_bound, angles >= obtuse_bound)
    in_cone_idx[0, 0, 0] = 1  # include DC
    return _fsc(a, b, rfft_mask=in_cone_idx)
