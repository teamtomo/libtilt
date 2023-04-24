from typing import List, Tuple, Optional

import einops
import torch

from libtilt.grids.coordinate import coordinate_grid
from libtilt.utils.fft import distance_from_dc_for_dft


def rotational_average_2d(
    image: torch.Tensor,
    rfft: bool = False,
    fftshifted: Optional[bool] = None,
    return_1d_average: bool = True
) -> torch.Tensor:
    n_shells = image.shape[-2] // 2
    shell_data = _split_into_shells_2d(
        image, n_shells=n_shells, rfft=rfft, fftshifted=fftshifted
    )
    shell_means = [
        einops.reduce(shell, '... shell -> ...', reduction='mean')
        for shell in shell_data
    ]
    average_1d = einops.rearrange(shell_means, 'shells ... -> ... shells')
    if return_1d_average is True:
        return average_1d
    average_2d = _1d_to_rotational_average_2d(
        data_1d=average_1d,
        image_shape=image.shape,
        rfft=rfft,
        fftshifted=fftshifted,
    )
    return average_2d


def rotational_average_3d(
    image: torch.Tensor, rfft: bool = False, fftshifted: bool = True
) -> torch.Tensor:
    n_shells = image.shape[-3] // 2
    shells = _split_into_shells_3d(
        image, n_shells=n_shells, rfft=rfft, fftshifted=fftshifted
    )
    means = [
        einops.reduce(shell, '... shell -> ...', reduction='mean')
        for shell in shells
    ]
    return einops.rearrange(means, 'shells ... -> ... shells')


def _find_shell_indices_1d(
    distances: torch.Tensor, n_shells: int
) -> List[torch.Tensor]:
    """Find indices into a vector of distances for shells 1 unit apart."""
    sorted, sort_idx = torch.sort(distances, descending=False)
    split_points = torch.linspace(start=0.5, end=n_shells - 0.5, steps=n_shells)
    split_idx = torch.searchsorted(sorted, split_points)
    return torch.tensor_split(sort_idx, split_idx)[:-1]


def _find_shell_indices_2d(
    distances: torch.Tensor, n_shells: int
) -> List[torch.Tensor]:
    """Find indices into a matrix of distances for shells 1 unit apart."""
    h, w = distances.shape[-2:]
    idx_2d = coordinate_grid([h, w]).long()
    distances = einops.rearrange(distances, 'h w -> (h w)')
    idx_2d = einops.rearrange(idx_2d, 'h w idx -> (h w) idx')
    sorted, sort_idx = torch.sort(distances, descending=False)
    split_points = torch.linspace(start=0.5, end=n_shells - 0.5, steps=n_shells)
    split_idx = torch.searchsorted(sorted, split_points)
    return torch.tensor_split(idx_2d[sort_idx], split_idx)[:-1]


def _split_into_shells_2d(
    image: torch.Tensor, n_shells: int, rfft: bool = False,
    fftshifted: Optional[bool] = None
) -> List[torch.Tensor]:
    h, w = image.shape[-2:]
    if fftshifted is None and rfft is False:
        fftshifted = True
    elif fftshifted is None and rfft is True:
        fftshifted = False
    distances = distance_from_dc_for_dft(
        dft_shape=(h, w), rfft=rfft, fftshifted=fftshifted
    )
    distances = einops.rearrange(distances, 'h w -> (h w)')
    per_shell_indices = _find_shell_indices_1d(distances, n_shells=n_shells)
    image = einops.rearrange(image, '... h w -> ... (h w)')
    shells = [
        image[..., shell_idx]
        for shell_idx in per_shell_indices
    ]
    return shells


def _1d_to_rotational_average_2d(
    data_1d: torch.Tensor,
    image_shape: Tuple[int],
    rfft: bool = False,
    fftshifted: bool = True,
) -> torch.Tensor:
    average_2d = torch.zeros(size=image_shape)
    distances = distance_from_dc_for_dft(
        dft_shape=image_shape[-2:], rfft=rfft, fftshifted=fftshifted
    )
    n_shells = data_1d.shape[-1]
    shell_idx = _find_shell_indices_2d(distances=distances, n_shells=n_shells)
    for idx, shell in enumerate(shell_idx):
        idx_h, idx_w = einops.rearrange(shell, 'b idx -> idx b')
        average_2d[..., idx_h, idx_w] = data_1d[..., [idx]]
    average_2d[..., distances > n_shells - 0.5] = data_1d[..., [-1]]
    return average_2d


def _split_into_shells_3d(
    image: torch.Tensor, n_shells: int, rfft: bool = False, fftshifted: bool = True
) -> List[torch.Tensor]:
    d, h, w = image.shape[-3:]
    distances = distance_from_dc_for_dft(
        dft_shape=(d, h, w), rfft=rfft, fftshifted=fftshifted
    )
    distances = einops.rearrange(distances, 'd h w -> (d h w)')
    image = einops.rearrange(image, '... d h w -> ... (d h w)')
    per_shell_indices = _find_shell_indices_1d(distances, n_shells=n_shells)
    shells = [
        image[..., shell_idx]
        for shell_idx in per_shell_indices
    ]
    return shells
