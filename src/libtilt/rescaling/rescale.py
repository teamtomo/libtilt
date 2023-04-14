from math import ceil, floor
from typing import Sequence

import einops
import torch
import torch.nn.functional as F


def rescale_fourier_2d(
    image: torch.Tensor,
    source_spacing: float,
    target_spacing: float,
) -> tuple[torch.Tensor, float]:
    """Rescale 2D image(s) from `source_spacing` to `target_spacing`.

    Rescaling is performed in Fourier space by either cropping or padding the
    discrete Fourier transform (DFT). The input image(s) must have even
    sidelengths in both spatial dimensions. The output image(s) will have even
    sidelengths in both spatial dimensions. The origin [..., 0, 0] is maintained.
    """
    h, w = image.shape[-2:]
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError('2D image sidelengths must be divisible by two.')
    if source_spacing == target_spacing:
        return image, source_spacing

    # pad to square to ensure output has isotropic spacing
    dim_to_pad = 'hw'[torch.argmin(torch.tensor([h, w]))]
    if dim_to_pad == 'h':
        pad = w - h
        image = F.pad(image, pad=(0, 0, 0, pad), mode='reflect')
    else:  # pad w
        pad = h - w
        image = F.pad(image, pad=(0, pad), mode='reflect')

    dft = torch.fft.rfftn(image, dim=(-2, -1))
    h_sq, w_sq = image.shape[-2:]
    freqs_h, freqs_w = torch.fft.fftfreq(h_sq), torch.fft.rfftfreq(w_sq)
    fraction_of_nyquist = source_spacing / target_spacing
    target_fftfreq = fraction_of_nyquist * 0.5
    if target_spacing > source_spacing:  # Fourier crop
        new_nyquist_idx = torch.argmin(torch.abs(freqs_w - target_fftfreq))
        new_nyquist = freqs_w[new_nyquist_idx]
        idx_h = (freqs_h >= -new_nyquist) & (freqs_h < new_nyquist)
        idx_w = freqs_w <= new_nyquist
        dft = dft[..., idx_h, :]
        dft = dft[..., :, idx_w]
        rescaled_image = torch.fft.irfftn(dft, dim=(-2, -1))
    elif target_spacing < source_spacing:  # zero pad in Fourier space
        delta_fftfreq = 1 / w_sq
        idx = target_fftfreq / delta_fftfreq
        idx = ceil(idx) if ceil(idx) % 2 == 0 else floor(idx)
        new_nyquist = idx * delta_fftfreq
        wf = dft.shape[-1]
        pad_w = idx - (wf - 1)
        dft = torch.fft.fftshift(dft, dim=(-2))
        dft = F.pad(dft, pad=(0, pad_w, pad_w, pad_w), mode='constant', value=0)
        dft = torch.fft.ifftshift(dft, dim=(-2))
        rescaled_image = torch.fft.irfftn(dft, dim=(-2, -1))

    # unpad from square
    rescaled_h, rescaled_w = rescaled_image.shape[-2:]
    idx_h, idx_w = torch.arange(rescaled_h), torch.arange(rescaled_w)
    if dim_to_pad == 'h':  # crop h
        h_max = (h - 1) * fraction_of_nyquist
        if fraction_of_nyquist >= 1:
            h_max += 1 * (fraction_of_nyquist - 1)
        h_max = ceil(h_max) if ceil(h_max) % 2 == 1 else floor(h_max)
        rescaled_image = rescaled_image[..., idx_h <= h_max, :]
    else:  # crop w
        w_max = (w - 1) * fraction_of_nyquist
        if fraction_of_nyquist >= 1:
            w_max += 1 * (fraction_of_nyquist - 1)
        w_max = ceil(w_max) if ceil(w_max) % 2 == 1 else floor(w_max)
        rescaled_image = rescaled_image[..., :, idx_w <= w_max]

    final_spacing = 1 / (2 * new_nyquist * (1 / source_spacing))
    return rescaled_image, final_spacing


