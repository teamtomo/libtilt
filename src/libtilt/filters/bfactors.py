import torch

from libtilt.filters.filters import b_envelope


def bfactor_2d(
        image: torch.Tensor,
        B: float,
        pixel_size: float,
):
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    bfactored_dft = bfactor_dft(
        dft=dft,
        B=B,
        image_shape=image.shape[-2:],
        pixel_size=pixel_size,
        rfft=True,
        fftshifted=False,
    )
    return torch.real(torch.fft.irfftn(bfactored_dft, dim=(-2, -1)))


def bfactor_3d(
    image: torch.Tensor,
    B: float,
    pixel_size: float,
):
    dft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    bfactored_dft = bfactor_dft(
        dft=dft,
        B=B,
        image_shape=image.shape[-3:],
        pixel_size=pixel_size,
        rfft=True,
        fftshifted=False,
    )
    return torch.real(torch.fft.irfftn(bfactored_dft, dim=(-3, -2, -1)))


def bfactor_dft(
    dft: torch.Tensor,
    B: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,
    rfft: bool,
    fftshifted: bool = False,
):
    b_env = b_envelope(
        B=B,
        image_shape=image_shape,
        pixel_size=pixel_size,
        rfft=rfft,
        fftshift=fftshifted,
        device=dft.device
    )
    return dft * b_env
