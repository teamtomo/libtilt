import torch

from libtilt.fft_utils import ifftshift_2d


def correlate_dft_2d(
    a: torch.Tensor,
    b: torch.Tensor,
    rfft: bool,
    fftshifted: bool
) -> torch.Tensor:
    """Correlate discrete Fourier transforms of images."""
    result = a * torch.conj(b)
    if fftshifted is True:
        result = ifftshift_2d(result, rfft=rfft)
    if rfft is True:
        result = torch.fft.irfftn(result, dim=(-2, -1))
    else:
        result = torch.fft.ifftn(result, dim=(-2, -1))
    return torch.real(torch.fft.ifftshift(result, dim=(-2, -1)))
