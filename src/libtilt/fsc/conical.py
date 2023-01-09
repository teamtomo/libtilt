from typing import Tuple

import torch

from libtilt.utils.fft import rfft_shape_from_signal_shape
from ._general import _indices_centered_on_dc_for_shifted_rfft
from ._general import fsc as _fsc


def fsc_conical(
    a: torch.Tensor,
    b: torch.Tensor,
    cone_direction: Tuple[float, float] | Tuple[float, float, float],
    cone_aperture: float,
):
    rfft_shape = rfft_shape_from_signal_shape(a.shape)
    vectors = _indices_centered_on_dc_for_shifted_rfft(rfft_shape)  # (..., 3)
    vectors /= torch.linalg.norm(vectors, dim=-1)
    cone_direction = torch.tensor(cone_direction)
    cone_direction /= torch.linalg.norm(cone_direction)
    angles = _angle_between_vectors(vectors, cone_direction)
    acute_bound = cone_aperture / 2
    obtuse_bound = 180 - acute_bound
    in_cone_idx = torch.logical_or(angles <= acute_bound, angles >= obtuse_bound)
    return _fsc(a, b, valid_rfft_indices=in_cone_idx)


def _angle_between_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a can optionally be an nD stack, return in degrees"""
    a, b = torch.tensor(a), torch.tensor(b)
    n_stack_dims = a.ndim - 1
    stack_dims = _unique_characters(n_stack_dims)
    angles = torch.arccos(torch.einsum(f'{stack_dims}v,v->{stack_dims}', a, b))
    return torch.rad2deg(angles)


def _unique_characters(n: int) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz"
    return chars[:n]

