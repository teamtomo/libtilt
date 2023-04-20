import functools

import einops
import torch

from .coordinate import coordinate_grid
from ..utils.fft import dft_center


@functools.lru_cache(maxsize=1)
def distance_grid(
    image_shape: tuple[int, int] | tuple[int, int, int],
    center: tuple[float, float] | tuple[float, float, float] | None,
    device: torch.device | None = None,
):
    """Construct a 2D or 3D grid of distances from the a point a grid.

    The center_grid is defined here as the position of the DC component of the DFT.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        Shape of the 2D or 3D image before computing the DFT.
    center: tuple[float, float] | tuple[float, float, float] | None
        Point from which distances are calculated. If `None`, distances will
        be relative to the center of the grid.
    device: torch.device | None
        PyTorch device on which the grid will be stored.

    Returns
    -------
    distance_grid: torch.Tensor
        `(*image_shape)` array of distances from `center`.
    """
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    coordinates = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    ).float()
    distances = einops.reduce(
        coordinates.float() ** 2, '... coords -> ...', reduction='sum'
    ) ** 0.5
    return distances


@functools.lru_cache(maxsize=1)
def distance_grid_dft(
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device | None = None,
):
    """Construct a 2D or 3D grid of distances from the a point a grid.

    The center_grid is defined here as the position of the DC component of the DFT.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        Shape of the 2D or 3D image before computing the DFT.
    center: tuple[float, float] | tuple[float, float, float] | None
        Point from which distances are calculated. If `None`, distances will
        be relative to the center of the grid.
    device: torch.device | None
        PyTorch device on which the returned grid will be stored.

    Returns
    -------
    frequency_grid: torch.Tensor
        A `(*image_shape, ndim)` array of DFT sample frequencies in each
        image dimension.
    """
    coordinates = coordinate_grid(
        image_shape=image_shape,
        center_grid=True if center is None else False,
        device=device,
    ).float()
    if center is not None:
        coordinates -= torch.as_tensor(center, dtype=torch.float32, device=device)
    distances = einops.reduce(
        coordinates.float() ** 2, '... coords -> ...', reduction='sum'
    ) ** 0.5
    return distances



