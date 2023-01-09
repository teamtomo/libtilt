from typing import List, Sequence

import einops
import numpy as np
import torch
from torch import fft as fft

from libtilt.utils.fft import rfft_shape_from_signal_shape


def fsc(
    a: torch.Tensor,
    b: torch.Tensor,
    valid_rfft_indices: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring/shell correlation between two square/cubic images."""
    # input handling
    rfft_shape = rfft_shape_from_signal_shape(a.shape)
    if a.ndim not in (2, 3):
        raise ValueError('images must be 2D or 3D.')
    elif a.shape != b.shape:
        raise ValueError('images must be the same shape.')
    elif valid_rfft_indices is not None and valid_rfft_indices.shape != rfft_shape:
        raise ValueError('valid rfft indices must have same shape as rfft.')

    # fsc calculation
    a, b = fft.rfftn(a), fft.rfftn(b)
    shift_dims = (-3, -2) if a.ndim == 3 else (-2, )
    a, b = fft.fftshift(a, dim=shift_dims), fft.fftshift(b, dim=shift_dims)
    distance_from_dc = _distance_from_dc_for_shifted_rfft(a.shape)
    n_shells = (a.shape[0] // 2) + 1
    if valid_rfft_indices is not None:
        a, b, distance_from_dc = (arr[valid_rfft_indices] for arr in [a, b, distance_from_dc])
    a, b, distance_from_dc = (torch.flatten(arg) for arg in [a, b, distance_from_dc])
    shell_idx = _find_shell_indices_1d(distance_from_dc, n_shells=n_shells)
    fsc = [
        _normalised_cc_complex_1d(a[idx], b[idx])
        for idx in
        shell_idx
    ]
    return torch.real(torch.tensor(fsc))


def _normalised_cc_complex_1d(a: torch.Tensor, b: torch.Tensor):
    correlation = torch.dot(a, torch.conj(b))
    return correlation / (torch.linalg.norm(a) * torch.linalg.norm(b))


def _find_shell_indices_1d(
    distances: torch.Tensor, n_shells: int
) -> List[torch.Tensor]:
    """Find indices into a vector of distances for shells 1 unit apart."""
    sorted, sort_idx = torch.sort(distances, descending=False)
    split_points = torch.linspace(start=0.5, end=n_shells - 0.5, steps=n_shells)
    split_idx = torch.searchsorted(sorted, split_points)
    return torch.tensor_split(sort_idx, split_idx)[:-1]


def _indices_centered_on_dc_for_shifted_rfft(
    rfft_shape: Sequence[int]
) -> torch.Tensor:
    rfftn_dc_idx = torch.div(torch.tensor(rfft_shape), 2, rounding_mode='floor')
    rfftn_dc_idx[-1] = 0
    rfft_indices = torch.tensor(np.indices(rfft_shape)) # (c, (d), h, w)
    rfft_indices = einops.rearrange(rfft_indices, 'c ... -> ... c')
    return rfft_indices - rfftn_dc_idx


def _distance_from_dc_for_shifted_rfft(rfft_shape: Sequence[int]) -> torch.Tensor:
    centered_indices = _indices_centered_on_dc_for_shifted_rfft(rfft_shape)
    return torch.linalg.norm(centered_indices.float(), dim=-1)