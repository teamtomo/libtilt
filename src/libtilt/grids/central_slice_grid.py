import einops
import torch

from libtilt.grids.fftfreq_grid import _construct_fftfreq_grid_2d
from libtilt.fft_utils import rfft_shape, fftshift_2d


def central_slice_grid(
    image_shape: tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    h, w = image_shape[-2:]
    slice_hw = _construct_fftfreq_grid_2d(
        image_shape=(h, w),
        rfft=rfft,
        device=device
    )  # (h, w, 2)
    if rfft is True:
        h, w = rfft_shape((h, w))
    slice_d = torch.zeros(size=(h, w), dtype=slice_hw.dtype, device=device)
    central_slice, _ = einops.pack([slice_d, slice_hw], pattern='h w *')  # (h, w, 3)
    if fftshift is True:
        central_slice = einops.rearrange(central_slice, 'h w freq -> freq h w')
        central_slice = fftshift_2d(central_slice, rfft=rfft)
        central_slice = einops.rearrange(central_slice, 'freq h w -> h w freq')
    return central_slice


def rotated_central_slice_grid(
    image_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,
    rotation_matrix_zyx: bool,
    rfft: bool,
    fftshift: bool = False,
    device: torch.device | None = None,
):
    grid = central_slice_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
    )  # (h, w, 3)
    if rotation_matrix_zyx is False:
        grid = torch.flip(grid, dims=(-1,))
    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 1 i j')
    grid = einops.rearrange(grid, 'h w coords -> h w coords 1')
    grid = rotation_matrices @ grid
    grid = einops.rearrange(grid, '... h w coords 1 -> ... h w coords')
    if rotation_matrix_zyx is False:  # back to zyx if currently xyz
        grid = torch.flip(grid, dims=(-1,))
    return grid
