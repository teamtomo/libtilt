from typing import Tuple

import numpy as np
import torch
from einops import einops

from .utils import estimate_background_local_mean, estimate_background_std


def inpaint(
        image: torch.Tensor,
        mask: torch.Tensor,
        background_model_resolution: Tuple[int, int] = (5, 5),
        n_background_samples: int = 20000,
) -> torch.Tensor:
    """Inpaint image(s) with gaussian noise.


    Parameters
    ----------
    image: np.ndarray
        `(b, h, w)` or `(h, w)` array containing image data for inpainting.
    mask: np.ndarray
        `(b, h, w)` or `(h, w)` binary mask separating foreground from background pixels.
        Foreground pixels will be inpainted.
    background_model_resolution: Tuple[int, int]
        Number of points in each image dimension for the background mean model.
        Minimum of two points in each dimension.
    n_background_samples: int
        Number of sample points used to determine the model of the background mean.

    Returns
    -------
    inpainted_image: torch.Tensor
        `(h, w)` array containing image data inpainted in the foreground pixels of the mask
        with gaussian noise matching the local mean and global standard deviation of the image
        for background pixels.
    """
    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if image.shape != mask.shape:
        raise ValueError('image shape must match mask shape.')
    input_is_batched = True
    if image.ndim == 2:
        input_is_batched = False
        image = einops.rearrange(image, 'h w -> 1 h w')
        mask = einops.rearrange(mask, 'h w -> 1 h w')
    inpainted = np.empty_like(image)
    for idx, _image in enumerate(image):
        inpainted[idx] = _inpaint_single_image(
            image=_image.numpy(),
            mask=mask[idx].numpy(),
            background_model_resolution=background_model_resolution,
            n_background_samples=n_background_samples,
        )
    if input_is_batched is False:  # drop batch dim
        inpainted = einops.rearrange(inpainted, '1 h w -> h w')
    return torch.as_tensor(inpainted, dtype=torch.float32)


def _inpaint_single_image(
        image: np.ndarray,
        mask: np.ndarray,
        background_model_resolution: Tuple[int, int] = (5, 5),
        n_background_samples: int = 20000
) -> np.ndarray:
    """Inpaint a single image with gaussian noise.


    Parameters
    ----------
    image: np.ndarray
        `(h, w)` array containing image data for inpainting.
    mask: np.ndarray
        `(h, w)` binary mask separating foreground from background pixels.
        Foreground pixels (value == 1) will be inpainted.
    background_model_resolution: Tuple[int, int]
        Number of points in each image dimension for the background mean model.
        Minimum of two points in each dimension.
    n_background_samples: int
        Number of sampling points for background mean estimation.

    Returns
    -------
    inpainted_image: np.ndarray
        `(h, w)` array containing image data inpainted in the foreground pixels of the mask
        with gaussian noise matching the local mean and global standard deviation of the image
        for background pixels.
    """
    inpainted_image = np.copy(image)
    background = estimate_background_local_mean(
        image=image,
        mask=mask,
        background_model_resolution=background_model_resolution,
        n_samples_for_fit=n_background_samples
    )

    # fill foreground pixels with local mean
    idx_foreground = np.argwhere(mask == 1)
    idx_foreground = (idx_foreground[:, 0], idx_foreground[:, 1])
    inpainted_image[idx_foreground] = background[idx_foreground]

    # add noise with mean=0 std=background std estimate
    background_std = estimate_background_std(image, mask)
    n_pixels_to_inpaint = idx_foreground[0].shape[0]
    gaussian_noise = np.random.normal(loc=0, scale=background_std, size=n_pixels_to_inpaint)
    inpainted_image[idx_foreground] += gaussian_noise
    return inpainted_image
