import torch

from ._general import fsc as _fsc


def fsc_isotropic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate the Fourier ring/shell correlation for square/cubic images."""
    return _fsc(a, b)




