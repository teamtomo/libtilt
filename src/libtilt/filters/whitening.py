import torch

from libtilt.filters.filters import whitening_filter


def get_whitening_2d(
    image: torch.Tensor,
):
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    whiten_dft = whitening_filter(
        image_dft=dft,
        image_shape=image.shape,
        rfft=True,
        fftshift=False,
        return_2d_average=True,
    )
    return whiten_dft


def whiten_image_2d(
    image: torch.Tensor,
    white_filter: torch.Tensor,
):
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    dft *= white_filter
    whitened_image = torch.real(torch.fft.irfftn(dft, dim=(-2, -1)))
    return whitened_image
