import einops
import torch

from libtilt.grids import distance_grid, coordinate_grid

from .soft_edge import _add_soft_edge_single_binary_image as _smooth_binary_image, \
    add_soft_edge_2d
from .geometry_utils import _angle_between_vectors
from ..utils.fft import dft_center


def circle(
    radius: float,
    image_shape: tuple[int, int] | int,
    center: tuple[float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape)
    distances = distance_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    )
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask[distances < radius] = 1
    return add_soft_edge_2d(mask, smoothing_radius=smoothing_radius)


def box(
    dimensions: tuple[float, float] | float,
    image_shape: tuple[int, int] | int,
    center: tuple[float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    coordinates = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    )
    dh, dw = dimensions[0] / 2, dimensions[1] / 2
    height_mask = torch.logical_and(coordinates[..., 0] > -dh, coordinates[..., 0] < dh)
    width_mask = torch.logical_and(coordinates[..., 1] > -dw, coordinates[..., 1] < dw)
    mask = torch.logical_and(height_mask, width_mask)
    return add_soft_edge_2d(mask, smoothing_radius=smoothing_radius)


def square(
    sidelength: tuple[float, float] | float,
    image_shape: tuple[int, int] | int,
    center: tuple[float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    square = box(
        dimensions=(sidelength, sidelength),
        image_shape=image_shape,
        center=center,
        smoothing_radius=smoothing_radius,
        device=device,
    )
    return square


def wedge(
    aperture: float,
    image_shape: tuple[int, int] | int,
    principal_axis: tuple[float, float] = (1, 0),
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape)
    center = dft_center(
        image_shape, rfft=False, fftshifted=True
    )
    vectors = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    ).float()
    vectors_norm = einops.reduce(vectors ** 2, '... c -> ... 1', reduction='sum') ** 0.5
    vectors /= vectors_norm
    principal_axis = torch.as_tensor(principal_axis, dtype=vectors.dtype, device=device)
    principal_axis_norm = einops.reduce(
        principal_axis ** 2, '... c -> ... 1', reduction='sum'
    ) ** 0.5
    principal_axis /= principal_axis_norm
    angles = _angle_between_vectors(vectors, principal_axis)
    acute_bound = aperture / 2
    obtuse_bound = 180 - acute_bound
    in_wedge = torch.logical_or(angles <= acute_bound, angles >= obtuse_bound)
    dc_h, dc_w = dft_center(image_shape, rfft=False, fftshifted=True)
    in_wedge[dc_h, dc_w] = True
    return add_soft_edge_2d(in_wedge, smoothing_radius=smoothing_radius)
