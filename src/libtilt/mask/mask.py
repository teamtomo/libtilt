from typing import Sequence, Literal, Optional

import numpy as np
import torch
import einops
import scipy.ndimage as ndi


def _nd_circle(
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


def _smooth_binary_image(mask: torch.Tensor, smoothing_radius: float) -> torch.Tensor:
    distances = torch.Tensor(ndi.distance_transform_edt(torch.logical_not(mask)))
    smoothing_idx = torch.logical_and(distances > 0, distances <= smoothing_radius)
    output = torch.clone(mask)
    output[smoothing_idx] = torch.cos((torch.pi / 2) * (distances[smoothing_idx] / smoothing_radius))
    return output
