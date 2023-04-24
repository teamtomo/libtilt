import torch

from libtilt.filters.filters import bandpass_filter


def bandpass_2d(
    image: torch.Tensor,
    low: float,
    high: float,
    falloff: float = 0,
):
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    bandpassed_dft = bandpass_dft(
        dft=dft,
        image_shape=image.shape[-2:],
        low=low,
        high=high,
        falloff=falloff,
        rfft=True,
        fftshifted=False,
    )
    return torch.real(torch.fft.irfftn(bandpassed_dft, dim=(-2, -1)))


def bandpass_3d(
    image: torch.Tensor,
    low: float,
    high: float,
    falloff: float = 0,
):
    dft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    bandpassed_dft = bandpass_dft(
        dft=dft,
        image_shape=image.shape[-3:],
        low=low,
        high=high,
        falloff=falloff,
        rfft=True,
        fftshifted=False,
    )
    return torch.real(torch.fft.irfftn(bandpassed_dft, dim=(-3, -2, -1)))


def bandpass_dft(
    dft: torch.Tensor,
    image_shape: tuple[int, int] | tuple[int, int, int],
    low: float,
    high: float,
    falloff: float,
    rfft: bool,
    fftshifted: bool = False,
):
    filter = bandpass_filter(
        low=low,
        high=high,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted
    )
    return dft * filter
