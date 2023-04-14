from typing import Sequence, Tuple

import einops
import torch

from libtilt.patch.patch_centers import get_patch_centers_2d as _get_patch_centers_2d
from libtilt.patch.patch_centers import get_patch_centers_1d as _get_patch_centers_1d


def _patch_centers_to_indices_1d(
    patch_centers: torch.Tensor, patch_length: int, device: torch.device = None
) -> torch.Tensor:
    displacements = torch.arange(patch_length, device=device) - patch_length // 2
    patch_centers = einops.rearrange(patch_centers, '... -> ... 1')
    return patch_centers + displacements  # (..., patch_length)


def get_patch_indices_2d(
    image_shape: Sequence[int],
    patch_shape: Tuple[int, int],
    patch_step: Tuple[int, int],
    distribute_patches: bool = True,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    centers = [
        _get_patch_centers_1d(
            dim_length=_dim_length,
            patch_length=_patch_length,
            patch_step=_patch_step,
            distribute_patches=distribute_patches,
            device=device
        )
        for _dim_length, _patch_length, _patch_step
        in zip(image_shape[-2:], patch_shape, patch_step)
    ]
    idx_h, idx_w = [
        _patch_centers_to_indices_1d(
            patch_centers=per_dim_centers,
            patch_length=window_length,
            device=device,
        )
        for per_dim_centers, window_length
        in zip(centers, patch_shape)
    ]
    idx_h = einops.rearrange(idx_h, 'ph h -> ph 1 h 1')
    idx_w = einops.rearrange(idx_w, 'pw w -> 1 pw 1 w')
    return idx_h, idx_w
