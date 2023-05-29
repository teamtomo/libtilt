import torch

from .phase_shift_dft import phase_shift_dft_2d, phase_shift_dft_3d


def shift_2d(images: torch.Tensor, shifts: torch.Tensor):
    h, w = images.shape[-2:]
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = phase_shift_dft_2d(
        images,
        image_shape=(h, w),
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    images = torch.fft.irfftn(images, dim=(-2, -1))
    return torch.real(images)


def shift_3d(images: torch.Tensor, shifts: torch.Tensor):
    d, h, w = images.shape[-3:]
    images = torch.fft.rfftn(images, dim=(-3, -2, -1))
    images = phase_shift_dft_3d(
        images,
        image_shape=(d, h, w),
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    images = torch.fft.irfftn(images, dim=(-3, -2, -1))
    return torch.real(images)
