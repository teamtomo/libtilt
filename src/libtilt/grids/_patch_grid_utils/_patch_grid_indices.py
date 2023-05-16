from typing import Sequence, Tuple

import einops
import torch

from libtilt.grids._patch_grid_utils._patch_grid_centers import _patch_centers_1d


def patch_grid_indices(
    image_shape: tuple[int, int] | tuple[int, int, int],
    patch_shape: tuple[int, int] | tuple[int, int, int],
    patch_step: tuple[int, int] | tuple[int, int, int],
    distribute_patches: bool = True,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters_are_valid = (
        len(image_shape) == len(patch_shape) and len(image_shape) == len(patch_step)
    )
    if parameters_are_valid is False:
        raise ValueError(
            "image shape, patch_extraction length and patch_extraction step are not the same length."
        )
    ndim = len(image_shape)
    if ndim == 2:
        return _patch_indices_2d(
            image_shape=image_shape,
            patch_shape=patch_shape,
            patch_step=patch_step,
            distribute_patches=distribute_patches,
            device=device,
        )
    elif ndim == 3:
        return _patch_indices_3d(
            image_shape=image_shape,
            patch_shape=patch_shape,
            patch_step=patch_step,
            distribute_patches=distribute_patches,
            device=device,
        )
    else:
        raise NotImplementedError("only 2D and 3D patches currently supported")


def _patch_centers_to_indices_1d(
    patch_centers: torch.Tensor, patch_length: int, device: torch.device = None
) -> torch.Tensor:
    displacements = torch.arange(patch_length, device=device) - patch_length // 2
    patch_centers = einops.rearrange(patch_centers, '... -> ... 1')
    return patch_centers + displacements  # (..., patch_shape)


def _patch_indices_2d(
    image_shape: Sequence[int],
    patch_shape: Tuple[int, int],
    patch_step: Tuple[int, int],
    distribute_patches: bool = True,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    centers = [
        _patch_centers_1d(
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


def _patch_indices_3d(
    image_shape: Sequence[int],
    patch_shape: Tuple[int, int, int],
    patch_step: Tuple[int, int, int],
    distribute_patches: bool = True,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = [
        _patch_centers_1d(
            dim_length=_dim_length,
            patch_length=_patch_length,
            patch_step=_patch_step,
            distribute_patches=distribute_patches,
            device=device
        )
        for _dim_length, _patch_length, _patch_step
        in zip(image_shape[-3:], patch_shape, patch_step)
    ]
    idx_d, idx_h, idx_w = [
        _patch_centers_to_indices_1d(
            patch_centers=per_dim_centers,
            patch_length=window_length,
            device=device,
        )
        for per_dim_centers, window_length
        in zip(centers, patch_shape)
    ]
    idx_d = einops.rearrange(idx_d, 'pd d -> pd 1 1 d 1 1')
    idx_h = einops.rearrange(idx_h, 'ph h -> 1 ph 1 1 h 1')
    idx_w = einops.rearrange(idx_w, 'pw w -> 1 1 pw 1 1 w')
    return idx_d, idx_h, idx_w
