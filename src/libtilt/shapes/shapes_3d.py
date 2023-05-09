import torch

from libtilt.grids import coordinate_grid
from libtilt.shapes.soft_edge import add_soft_edge_3d
from libtilt.utils.fft import dft_center


def sphere(
    radius: float,
    image_shape: tuple[int, int, int] | int,
    center: tuple[int, int, int] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    distances = coordinate_grid(
        image_shape=image_shape,
        center=center,
        norm=True,
        device=device,
    )
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask[distances < radius] = 1
    return add_soft_edge_3d(mask, smoothing_radius=smoothing_radius)
