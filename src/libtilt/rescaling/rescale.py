from math import ceil, floor

import torch
import torch.nn.functional as F

from libtilt.utils.fft import (
    _target_fftfreq_from_spacing, _pad_to_best_fft_shape_2d, dft_center,
)
from libtilt.shift.fourier_shift import phase_shift_dft_2d


def rescale_2d(
    image: torch.Tensor,
    source_spacing: float | tuple[float, float],
    target_spacing: float | tuple[float, float],
    maintain_dft_center: bool = False
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Rescale 2D image(s) from `source_spacing` to `target_spacing`.

    - rescaling is performed in Fourier space by either cropping or padding the
    discrete Fourier transform (DFT).
    - the output image(s) will have even sidelengths in both spatial dimensions.
    - the origin [..., 0, 0] is maintained.

    Parameters
    ----------
    image: torch.Tensor
        `(..., h, w)` array of image data
    source_spacing: float | tuple[float, float]
        Pixel spacing in the input image.
    target_spacing: float | tuple[float, float]
        Pixel spacing in the output image.
    maintain_dft_center: bool
        Whether to maintain the DFT center of the image (`True`) or the array
        origin `[0, 0]` (`False`).

    Returns
    -------
    rescaled_image, (new_spacing_h, new_spacing_w)
    """
    if isinstance(source_spacing, int | float):
        source_spacing = (source_spacing, source_spacing)
    if isinstance(target_spacing, int | float):
        target_spacing = (target_spacing, target_spacing)
    if source_spacing == target_spacing:
        return image, source_spacing

    # pad input to a good fft size in each dimension
    h, w = image.shape[-2:]
    target_fftfreq_h, target_fftfreq_w = _target_fftfreq_from_spacing(
        source_spacing=source_spacing, target_spacing=target_spacing
    )
    image = _pad_to_best_fft_shape_2d(
        image, target_fftfreq=(target_fftfreq_h, target_fftfreq_w)
    )
    padded_h, padded_w = image.shape[-2:]

    # compute DFT
    dft = torch.fft.rfftn(image, dim=(-2, -1))

    # Fourier pad/crop
    dft, (new_nyquist_h, new_nyquist_w) = _rescale_rfft_2d(
        dft=dft,
        image_shape=(padded_h, padded_w),
        target_fftfreq=(target_fftfreq_h, target_fftfreq_w)
    )

    # Calculate new spacing after rescaling
    source_spacing_h, source_spacing_w = source_spacing
    new_spacing_h = 1 / (2 * new_nyquist_h * (1 / source_spacing_h))
    new_spacing_w = 1 / (2 * new_nyquist_w * (1 / source_spacing_w))

    # maintain origin at rotation center if requested
    if maintain_dft_center is True:
        dft = _align_to_original_dft_center(
            dft=dft,
            original_image_shape=(h, w),
            original_image_spacing=(source_spacing_h, source_spacing_w),
            rescaled_image_spacing=(new_spacing_h, new_spacing_w)
        )

    # transform back to real space
    rescaled_image = torch.fft.irfftn(dft, dim=(-2, -1))

    # calculate new spacings and unpad from rescaled optimal fft size
    rescaled_image = _unpad(
        rescaled_image=rescaled_image,
        rescaled_image_spacing=(new_spacing_h, new_spacing_w),
        original_image_shape=(h, w),
        original_image_spacing=(source_spacing_h, source_spacing_w)
    )
    return rescaled_image, (float(new_spacing_h), float(new_spacing_w))


def _get_final_shape(
    original_image_shape: tuple[int, int],
    original_image_spacing: tuple[float, float],
    rescaled_image_spacing: tuple[float, float],
):
    rescaled_spacing_h, rescaled_spacing_w = rescaled_image_spacing
    original_h, original_w = original_image_shape
    original_spacing_h, original_spacing_w = original_image_spacing
    length_h = (original_h - 1) * (original_spacing_h / rescaled_spacing_h)
    length_w = (original_w - 1) * (original_spacing_w / rescaled_spacing_w)
    length_h = ceil(length_h) if ceil(length_h) % 2 == 1 else floor(length_h)
    length_w = ceil(length_w) if ceil(length_w) % 2 == 1 else floor(length_w)
    new_h, new_w = length_h + 1, length_w + 1
    return new_h, new_w


def _unpad(
    rescaled_image: torch.Tensor,
    rescaled_image_spacing: tuple[float, float],
    original_image_shape: tuple[int, int],
    original_image_spacing: tuple[float, float],
):
    new_h, new_w = _get_final_shape(
        original_image_shape=original_image_shape,
        original_image_spacing=original_image_spacing,
        rescaled_image_spacing=rescaled_image_spacing,
    )
    rescaled_image = rescaled_image[..., :new_h, :]
    rescaled_image = rescaled_image[..., :, :new_w]
    return rescaled_image


def _fourier_crop_h(dft: torch.Tensor, image_height: int, target_fftfreq: float):
    frequencies = torch.fft.fftfreq(image_height)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_h = (frequencies >= -new_nyquist) & (frequencies < new_nyquist)
    return dft[..., idx_h, :], new_nyquist


def _fourier_crop_w(dft: torch.Tensor, image_width: int, target_fftfreq: float):
    frequencies = torch.fft.rfftfreq(image_width)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_w = frequencies <= new_nyquist
    return dft[..., :, idx_w], new_nyquist


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


def _rescale_rfft_2d(
    dft: torch.Tensor,
    image_shape: tuple[int, int],
    target_fftfreq: tuple[float, float]
) -> tuple[torch.Tensor, tuple[float, float]]:
    h, w = image_shape
    freq_h, freq_w = target_fftfreq
    if freq_h > 0.5:
        dft, nyquist_h = _fourier_pad_h(dft, image_height=h, target_fftfreq=freq_h)
    else:
        dft, nyquist_h = _fourier_crop_h(dft, image_height=h, target_fftfreq=freq_h)
    if freq_w > 0.5:
        dft, nyquist_w = _fourier_pad_w(dft, image_width=w, target_fftfreq=freq_w)
    else:
        dft, nyquist_w = _fourier_crop_w(dft, image_width=w, target_fftfreq=freq_w)
    return dft, (nyquist_h, nyquist_w)


def _align_to_original_dft_center(
    dft: torch.Tensor,
    original_image_shape: tuple[int, int],
    original_image_spacing: tuple[float, float],
    rescaled_image_spacing: tuple[float, float],
):
    h, w = original_image_shape
    original_spacing_h, original_spacing_w = original_image_spacing
    rescaled_spacing_h, rescaled_spacing_w = rescaled_image_spacing
    old_dft_center = dft_center(
        grid_shape=(h, w), rfft=False, fftshifted=True
    )
    final_h, final_w = _get_final_shape(
        original_image_shape=(h, w),
        original_image_spacing=(original_spacing_h, original_spacing_w),
        rescaled_image_spacing=(rescaled_spacing_h, rescaled_spacing_w),
    )
    target_center_h, target_center_w = dft_center(
        grid_shape=(final_h, final_w), rfft=False, fftshifted=True
    )
    center_h = old_dft_center[0] * (original_spacing_h / rescaled_spacing_h)
    center_w = old_dft_center[1] * (original_spacing_w / rescaled_spacing_w)
    dh, dw = target_center_h - center_h, target_center_w - center_w
    rescaled_image_h, rescaled_image_w = dft.shape[-2], (dft.shape[-1] - 1) * 2
    dft = phase_shift_dft_2d(
        dft=dft,
        image_shape=(rescaled_image_h, rescaled_image_w),
        shifts=torch.as_tensor([dh, dw], dtype=torch.float32, device=dft.device),
        rfft=True,
        fftshifted=False,
    )
    return dft
