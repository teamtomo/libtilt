import einops
import numpy as np
import torch

from libtilt.interpolation import insert_into_image_2d, insert_into_image_3d


def coordinates_to_image_2d(
    coordinates: torch.Tensor,
    image_shape: tuple[int, int],
    yx: bool = True,
) -> torch.Tensor:
    """Render an image from 2D coordinates by inserting ones with linear interpolation.

    Parameters
    ----------
    coordinates: torch.Tensor
        `(..., 2)` array of 3D coordinates at which ones will be inserted into
         the output image.
    image_shape: tuple[int, int]
        `(h, w)` shape of the output image.
    yx: bool
        Whether the coordinates are ordered yx (`True`) or xy (`False`).

    Returns
    -------
    image: torch.Tensor
        `(h, w)` output volume.
    """
    image = torch.zeros(
        size=image_shape, dtype=torch.float32, device=coordinates.device
    )
    if yx is False:
        coordinates = torch.flip(coordinates, dims=(-1,))
    coordinates, ps = einops.pack([coordinates], pattern='* coords')
    b, _ = coordinates.shape
    ones = torch.ones(size=(b,), dtype=torch.float32, device=coordinates.device)
    weights = torch.zeros_like(image)
    image, weights = insert_into_image_2d(
        data=ones, coordinates=coordinates, image=image, weights=weights
    )
    return image


def coordinates_to_image_3d(
    coordinates: torch.Tensor,
    image_shape: tuple[int, int, int],
    zyx: bool = True,
) -> torch.Tensor:
    """Render an image from 3D coordinates by inserting ones with linear interpolation.

    Parameters
    ----------
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates at which ones will be inserted into
         the output image.
    image_shape: tuple[int, int, int]
        `(d, h, w)` shape of the output image.
    zyx: bool
        Whether the coordinates are ordered zyx (`True`) or xyz (`False`).

    Returns
    -------
    image: torch.Tensor
        `(d, h, w)` output volume.
    """
    image = torch.zeros(
        size=image_shape, dtype=torch.float32, device=coordinates.device
    )
    if zyx is False:
        coordinates = torch.flip(coordinates, dims=(-1,))
    coordinates, ps = einops.pack([coordinates], pattern='* coords')
    b, _ = coordinates.shape
    ones = torch.ones(size=(b,), dtype=torch.float32, device=coordinates.device)
    weights = torch.zeros_like(image)
    image, weights = insert_into_image_3d(
        data=ones, coordinates=coordinates, image=image, weights=weights
    )
    return image
