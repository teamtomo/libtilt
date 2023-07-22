import numpy as np
import torch

from libtilt.fft_utils import dft_center as _dft_center

def central_crop_2d(image: np.ndarray, percentage: float = 25) -> np.ndarray:
    """Get a central crop of (a batch of) 2D image(s).

    Parameters
    ----------
    image: np.ndarray
        `(b, h, w)` or `(h, w)` array of 2D images.
    percentage: float
        percentage of image height and width for cropped region.
    Returns
    -------
    cropped_image: np.ndarray
        `(b, h, w)` or `(h, w)` array of cropped 2D images.
    """
    h, w = image.shape[-2], image.shape[-1]
    mh, mw = h // 2, w // 2
    dh, dw = int(h * (percentage / 100 / 2)), int(w * (percentage / 100 / 2))
    hf, wf = mh - dh, mw - dw
    hc, wc = mh + dh, mw + dw
    return image[..., hf:hc, wf:wc]


def estimate_background_std(image: torch.Tensor, mask: torch.Tensor):
    """Estimate the standard deviation of the background from a central crop.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing data for which background standard deviation will be estimated.
    mask: torch.Tensor of 0 or 1
        Binary shapes separating foreground and background.
    Returns
    -------
    standard_deviation: float
        estimated standard deviation for the background.
    """
    image = central_crop_2d(image, percentage=25).float()
    mask = central_crop_2d(mask, percentage=25)
    return torch.std(image[mask == 0])


def rotation_center(
    image_shape: tuple[int, int] | tuple[int, int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Get the rotation center of an image.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        The 2D or 3D shape of the image for which the rotation center should be returned.

    Returns
    -------
    rotation_center: torch.Tensor
        `(2, )` or `(3, )` array containing the rotation center.
    """
    return _dft_center(
        image_shape=image_shape, rfft=False, fftshifted=True, device=device
    )
