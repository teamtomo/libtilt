import einops
import torch

from libtilt.correlation.correlate_dft import correlate_dft_2d


def correlate_2d(
    a: torch.Tensor, b: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    """Calculate the 2D cross correlation between images of the same size.

    The position of the maximum relative to the center of the image gives a shift.
    This is the shift that when applied to `b` best aligns it to `a`.
    """
    if normalize is True:
        h, w = a.shape[-2:]
        a_norm = einops.reduce(a ** 2, '... h w -> ... 1 1', reduction='mean') ** 0.5
        b_norm = einops.reduce(b ** 2, '... h w -> ... 1 1', reduction='mean') ** 0.5
        a = a / a_norm
        b = b / b_norm
    a = torch.fft.rfftn(a, dim=(-2, -1))
    b = torch.fft.rfftn(b, dim=(-2, -1))
    result = correlate_dft_2d(a, b, rfft=True, fftshifted=False)
    if normalize is True:
        result = result / (h * w)
    return result
