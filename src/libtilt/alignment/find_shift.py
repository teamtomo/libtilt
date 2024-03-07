import torch
import numpy as np
import torch.nn.functional as F
import einops

from libtilt.correlation import correlate_2d
from libtilt.fft_utils import dft_center


def find_image_shift(
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        upsampling_factor: float = 10,
) -> torch.Tensor:
    """Find the shift between two images. The shift specifies how far image_b is
    shifted relative to image_a. Applying this shift to image_a will align it with
    image_b. Applying the inverse shift to b, will align it with image_a. The
    region around the maximum in the correlation image is by default upsampled with
    bicubic interpolation to find a more precise shift.

    Parameters
    ----------
    image_a: torch.Tensor
        `(y, x)` image.
    image_b: torch.Tensor
        `(y, x)` image with the same shape as image_a
    upsampling_factor: float
        How many times the correlation image is upsampled with bicubic
        interpolation to find an interpolated shift. The value needs to be larger or
        equal to 1.

    Returns
    -------
    shift: torch.Tensor
        `(2, )` shift in y and x.
    """
    if upsampling_factor < 1:
        raise ValueError('Upsampling factor for finding a shift between two images '
                         'cannot be smaller than 1.')
    image_shape = torch.tensor(image_a.shape, device=image_a.device)
    center = dft_center(
        image_a.shape, rfft=False, fftshifted=True, device=image_a.device
    )
    correlation = correlate_2d(
        image_a,
        image_b,
        normalize=True
    )
    maximum_idx = torch.tensor(  # explicitly put tensor on CPU in case input is on GPU
        np.unravel_index(correlation.argmax().cpu(), shape=image_a.shape),
        device=image_a.device
    )
    shift = center - maximum_idx
    if upsampling_factor == 1:
        return shift
    elif torch.any(maximum_idx < 2) or torch.any(image_shape - maximum_idx < 3):
        # if the maximum is too close to the border, it cannot be upsampled
        return shift
    else:
        # find interpolated shift by upsampling the correlation image
        peak_region_y = slice(maximum_idx[0] - 2, maximum_idx[0] + 3)
        peak_region_x = slice(maximum_idx[1] - 2, maximum_idx[1] + 3)
        upsampled = F.interpolate(
            einops.rearrange(
                correlation[peak_region_y, peak_region_x],
                'h w -> 1 1 h w'
            ),
            scale_factor=upsampling_factor,
            mode='bicubic',
            align_corners=True
        )
        upsampled = einops.rearrange(upsampled, '1 1 h w -> h w')
        upsampled_center = dft_center(
            upsampled.shape, rfft=False, fftshifted=True, device=image_a.device
        )
        upsampled_shift = upsampled_center - torch.tensor(
            np.unravel_index(upsampled.argmax().cpu(), shape=upsampled.shape),
            device=image_a.device
        )
        full_shift = shift + upsampled_shift / upsampling_factor
        return full_shift
