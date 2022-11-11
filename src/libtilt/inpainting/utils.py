from typing import Tuple

import numpy as np
from scipy.interpolate import LSQBivariateSpline


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


def estimate_background_local_mean(
        image: np.ndarray,
        mask: np.ndarray,
        background_model_resolution: Tuple[int, int] = (5, 5),
        n_samples_for_fit: int = 20000
):
    """Estimate image background mean with a bivariate cubic spline.

    Parameters
    ----------
    image: np.ndarray
        `(h, w)` array containing image data.
    mask: np.ndarray
        `(h, w)` array containing a binary mask specifying foreground
        and background pixels.
    background_model_resolution: Tuple[int, int]
        Resolution of the mo
    n_samples_for_fit: int
        Number of samples taken from foreground pixels for background mean estimation.
        The number of background pixels will be used if this number is greater than the
        number of background pixels.

    Returns
    -------
    background_local_mean: np.ndarray
        `(h, w)` array containing a local estimate of the background mean.
    """
    # get a random set of background pixels for the background fit
    background_sample_idx = np.argwhere(mask == 0)
    n_samples_for_fit = max(n_samples_for_fit, len(background_sample_idx))
    selection = np.random.choice(background_sample_idx.shape[0], size=n_samples_for_fit,
                                 replace=False)
    background_sample_idx = background_sample_idx[selection]
    y, x = background_sample_idx[:, 0], background_sample_idx[:, 1]
    z = image[(y, x)]

    # fit a bivariate spline to the data with the specified background model resolution
    ty = np.linspace(0, image.shape[0], num=background_model_resolution[0])
    tx = np.linspace(0, image.shape[1], num=background_model_resolution[1])
    background_model = LSQBivariateSpline(x, y, z, tx, ty)

    # evaluate the model over a grid covering the whole image
    y = np.arange(image.shape[0])
    x = np.arange(image.shape[1])
    return background_model(y, x, grid=True)


def estimate_background_std(image: np.ndarray, mask: np.ndarray):
    """Estimate the standard deviation of the background from a central crop.

    Parameters
    ----------
    image: np.ndarray
        `(h, w)` array containing data for which background standard deviation will be estimated.
    mask: np.ndarray of 0 or 1
        Binary mask separating foreground and background.
    Returns
    -------
    standard_deviation: float
        estimated standard deviation for the background.
    """
    image = central_crop_2d(image, percentage=25)
    mask = central_crop_2d(mask, percentage=25)
    return np.std(central_crop_2d(image)[central_crop_2d(mask) == 0])
