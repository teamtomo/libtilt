from math import ceil, floor

import torch
import torch.nn.functional as F

from libtilt.fft_utils import (
    _target_fftfreq_from_spacing, _pad_to_best_fft_shape_3d, dft_center,
)
from libtilt.shift import phase_shift_dft_3d


def rescale_3d(
    image: torch.Tensor,
    source_spacing: float | tuple[float, float, float],
    target_spacing: float | tuple[float, float, float],
    maintain_center: bool = False
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    """Rescale 3D image(s) from `source_spacing` to `target_spacing`.

    Rescaling is performed in Fourier space by either cropping or padding the
    discrete Fourier transform (DFT).

    Parameters
    ----------
    image: torch.Tensor
        `(..., h, w)` array of image data
    source_spacing: float | tuple[float, float, float]
        Pixel spacing in the input image.
    target_spacing: float | tuple[float, float, float]
        Pixel spacing in the output image.
    maintain_center: bool
        Whether to maintain the center (position of DC component of DFT) of the
        image (`True`) or the array origin `[0, 0, 0]` (`False`).

    Returns
    -------
    rescaled_image, (new_spacing_d, new_spacing_h, new_spacing_w)
    """
    if isinstance(source_spacing, int | float):
        source_spacing = (source_spacing, source_spacing, source_spacing)
    if isinstance(target_spacing, int | float):
        target_spacing = (target_spacing, target_spacing, source_spacing)
    if source_spacing == target_spacing:
        return image, source_spacing

    # pad input to a good fft size in each dimension
    d, h, w = image.shape[-3:]
    target_fftfreq_d, target_fftfreq_h, target_fftfreq_w = _target_fftfreq_from_spacing(
        source_spacing=source_spacing, target_spacing=target_spacing
    )
    image = _pad_to_best_fft_shape_3d(
        image, target_fftfreq=(target_fftfreq_d, target_fftfreq_h, target_fftfreq_w)
    )
    padded_d, padded_h, padded_w = image.shape[-3:]

    # compute DFT
    dft = torch.fft.rfftn(image, dim=(-3, -2, -1))

    # Fourier pad/crop
    dft, (new_nyquist_d, new_nyquist_h, new_nyquist_w) = _rescale_rfft_3d(
        dft=dft,
        image_shape=(padded_d, padded_h, padded_w),
        target_fftfreq=(target_fftfreq_d, target_fftfreq_h, target_fftfreq_w)
    )

    # Calculate new spacing after rescaling
    source_spacing_d, source_spacing_h, source_spacing_w = source_spacing
    new_spacing_d = 1 / (2 * new_nyquist_d * (1 / source_spacing_d))
    new_spacing_h = 1 / (2 * new_nyquist_h * (1 / source_spacing_h))
    new_spacing_w = 1 / (2 * new_nyquist_w * (1 / source_spacing_w))

    # maintain rotation center if requested
    if maintain_center is True:
        dft = _align_to_original_image_center_3d(
            dft=dft,
            original_image_shape=(d, h, w),
            original_image_spacing=(source_spacing_d, source_spacing_h, source_spacing_w),
            rescaled_image_spacing=(new_spacing_d, new_spacing_h, new_spacing_w)
        )

    # transform back to real space
    rescaled_image = torch.fft.irfftn(dft, dim=(-2, -1))

    # calculate new spacings and unpad from rescaled optimal fft size
    rescaled_image = _unpad(
        rescaled_image=rescaled_image,
        rescaled_image_spacing=(new_spacing_d, new_spacing_h, new_spacing_w),
        original_image_shape=(d, h, w),
        original_image_spacing=(source_spacing_d, source_spacing_h, source_spacing_w)
    )
    return rescaled_image, (float(new_spacing_h), float(new_spacing_w))


def _get_final_shape(
    original_image_shape: tuple[int, int, int],
    original_image_spacing: tuple[float, float, float],
    rescaled_image_spacing: tuple[float, float, float]
):
    rescaled_spacing_d, rescaled_spacing_h, rescaled_spacing_w = rescaled_image_spacing
    original_d, original_h, original_w = original_image_shape
    original_spacing_d, original_spacing_h, original_spacing_w = original_image_spacing
    length_d = (original_d - 1) * (original_spacing_d / rescaled_spacing_d)
    length_h = (original_h - 1) * (original_spacing_h / rescaled_spacing_h)
    length_w = (original_w - 1) * (original_spacing_w / rescaled_spacing_w)
    length_d = ceil(length_d) if ceil(length_d) % 2 == 1 else floor(length_d)
    length_h = ceil(length_h) if ceil(length_h) % 2 == 1 else floor(length_h)
    length_w = ceil(length_w) if ceil(length_w) % 2 == 1 else floor(length_w)
    new_d, new_h, new_w = length_d + 1, length_h + 1, length_w + 1
    return new_d, new_h, new_w


def _unpad(
    rescaled_image: torch.Tensor,
    rescaled_image_spacing: tuple[float, float, float],
    original_image_shape: tuple[int, int, int],
    original_image_spacing: tuple[float, float, float],
):
    new_d, new_h, new_w = _get_final_shape(
        original_image_shape=original_image_shape,
        original_image_spacing=original_image_spacing,
        rescaled_image_spacing=rescaled_image_spacing,
    )
    rescaled_image = rescaled_image[..., :new_d, :, :]
    rescaled_image = rescaled_image[..., :, :new_h, :]
    rescaled_image = rescaled_image[..., :, :, :new_w]
    return rescaled_image


def _fourier_crop_d(dft: torch.Tensor, image_depth: int, target_fftfreq: float):
    frequencies = torch.fft.fftfreq(image_depth)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_d = (frequencies >= -new_nyquist) & (frequencies < new_nyquist)
    return dft[..., idx_d, :, :], new_nyquist


def _fourier_crop_h(dft: torch.Tensor, image_height: int, target_fftfreq: float):
    frequencies = torch.fft.fftfreq(image_height)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_h = (frequencies >= -new_nyquist) & (frequencies < new_nyquist)
    return dft[..., :, idx_h, :], new_nyquist


def _fourier_crop_w(dft: torch.Tensor, image_width: int, target_fftfreq: float):
    frequencies = torch.fft.rfftfreq(image_width)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_w = frequencies <= new_nyquist
    return dft[..., :, , idx_w], new_nyquist


def _fourier_pad_d(dft: torch.Tensor, image_depth: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_depth
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(idx_nyquist)
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = (dft.shape[-3] // 2) + 1
    pad_d = idx_nyquist - (n_frequencies - 1)
    dft = torch.fft.fftshift(dft, dim=(-3))
    dft = F.pad(dft, pad=(0, 0, 0, 0, pad_d, pad_d), mode='constant', value=0)
    dft = torch.fft.ifftshift(dft, dim=(-3))
    return dft, new_nyquist


def _fourier_pad_h(dft: torch.Tensor, image_height: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_height
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(
        idx_nyquist)
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = (dft.shape[-2] // 2) + 1
    pad_h = idx_nyquist - (n_frequencies - 1)
    dft = torch.fft.fftshift(dft, dim=(-2))
    dft = F.pad(dft, pad=(0, 0, pad_h, pad_h), mode='constant', value=0)
    dft = torch.fft.ifftshift(dft, dim=(-2))
    return dft, new_nyquist


def _fourier_pad_w(dft: torch.Tensor, image_width: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_width
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(
        idx_nyquist)
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = dft.shape[-1]
    pad_w = idx_nyquist - (n_frequencies - 1)
    dft = F.pad(dft, pad=(0, pad_w), mode='constant', value=0)
    return dft, new_nyquist


def _rescale_rfft_3d(
    dft: torch.Tensor,
    image_shape: tuple[int, int, int],
    target_fftfreq: tuple[float, float, float]
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    d, h, w = image_shape
    freq_d, freq_h, freq_w = target_fftfreq
    if freq_d > 0.5:
        dft, nyquist_d = _fourier_pad_d(dft, image_depth=d, target_fftfreq=freq_d)
    else:
        dft, nyquist_d = _fourier_crop_d(dft, image_depth=d, target_fftfreq=freq_d)
    if freq_h > 0.5:
        dft, nyquist_h = _fourier_pad_h(dft, image_height=h, target_fftfreq=freq_h)
    else:
        dft, nyquist_h = _fourier_crop_h(dft, image_height=h, target_fftfreq=freq_h)
    if freq_w > 0.5:
        dft, nyquist_w = _fourier_pad_w(dft, image_width=w, target_fftfreq=freq_w)
    else:
        dft, nyquist_w = _fourier_crop_w(dft, image_width=w, target_fftfreq=freq_w)
    return dft, (nyquist_d, nyquist_h, nyquist_w)


def _align_to_original_image_center_3d(
    dft: torch.Tensor,
    original_image_shape: tuple[int, int, int],
    original_image_spacing: tuple[float, float, float],
    rescaled_image_spacing: tuple[float, float, float],
):
    """Align the new image center to the original image center."""
    d, h, w = original_image_shape
    original_spacing_d, original_spacing_h, original_spacing_w = original_image_spacing
    rescaled_spacing_d, rescaled_spacing_h, rescaled_spacing_w = rescaled_image_spacing
    previous_center_d, previous_center_h, previous_center_w = dft_center(
        image_shape=(d, h, w), rfft=False, fftshifted=True
    )
    final_d, final_h, final_w = _get_final_shape(
        original_image_shape=(d, h, w),
        original_image_spacing=(original_spacing_d, original_spacing_h, original_spacing_w),
        rescaled_image_spacing=(rescaled_spacing_d, rescaled_spacing_h, rescaled_spacing_w),
    )
    target_center_d, target_center_h, target_center_w = dft_center(
        image_shape=(final_d, final_h, final_w), rfft=False, fftshifted=True
    )
    current_center_d = previous_center_d * (original_spacing_d / rescaled_spacing_d)
    current_center_h = previous_center_h * (original_spacing_h / rescaled_spacing_h)
    current_center_w = previous_center_w * (original_spacing_w / rescaled_spacing_w)
    dd, dh, dw = (
        target_center_d - current_center_d,
        target_center_h - current_center_h,
        target_center_w - current_center_w
    )
    rescaled_image_d, rescaled_image_h, rescaled_image_w = (
        dft.shape[-3], dft.shape[-2], (dft.shape[-1] - 1) * 2
    )
    dft = phase_shift_dft_3d(
        dft=dft,
        image_shape=(rescaled_image_d, rescaled_image_h, rescaled_image_w),
        shifts=torch.as_tensor([dd, dh, dw], dtype=torch.float32, device=dft.device),
        rfft=True,
        fftshifted=False,
    )
    return dft
