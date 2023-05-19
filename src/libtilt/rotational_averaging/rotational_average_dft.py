from typing import List

import einops
import torch

from libtilt.fft_utils import rfft_shape
from libtilt.grids import fftfreq_grid, coordinate_grid


def rotational_average_dft_2d(
    dft: torch.Tensor,
    image_shape: tuple[int, ...],
    rfft: bool = False,
    fftshifted: bool = False,
    return_2d_average: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:  # rotational_average, frequency_bins
    # calculate the number of bins
    h, w = image_shape[-2:]
    n_bins = min((d // 2) + 1 for d in (h, w))

    # split data into frequency bins
    frequency_bins = _frequency_bin_centers(n_bins, device=dft.device)
    shell_data = _split_into_frequency_bins_2d(
        dft, n_bins=n_bins, image_shape=(h, w), rfft=rfft, fftshifted=fftshifted
    )

    # calculate mean over each shell
    mean_per_shell = [
        einops.reduce(shell, '... shell -> ...', reduction='mean')
        for shell in shell_data
    ]
    rotational_average = einops.rearrange(mean_per_shell, 'shells ... -> ... shells')
    if return_2d_average is True:
        if len(dft.shape) > len(image_shape):
            image_shape = (*dft.shape[:-2], *image_shape[-2:])
        rotational_average = _1d_to_rotational_average_2d_dft(
            values=rotational_average,
            image_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshifted,
        )
        frequency_bins = _1d_to_rotational_average_2d_dft(
            values=frequency_bins,
            image_shape=image_shape,
            rfft=rfft,
            fftshifted=fftshifted,
        )
    return rotational_average, frequency_bins


def rotational_average_dft_3d(
    image: torch.Tensor,
    rfft: bool = False,
    fftshifted: bool = True,
    return_3d_average: bool = True,
) -> torch.Tensor:
    d, h, w = image.shape[-3:]
    n_bins = min(d // 2, h // 2, w) if rfft is True else min((d // 2) + 1 for d in (h, w))
    # define frequency bin centers,
    # extra bin on the end to facilitate bin edge calculation
    df = 0.5 / (n_bins - 1)
    frequency_bins = torch.linspace(0, 0.5 + df, steps=n_bins + 1, device=image.device)

    # define split points in data as midpoint between bin centers
    adjacent_bins = frequency_bins.unfold(dimension=0, size=2, step=1)  # (b, 2)
    split_points = einops.reduce(adjacent_bins, 'b high_low -> b', reduction='mean')
    shells = _split_into_shells_3d(
        image, n_shells=n_shells, rfft=rfft, fftshifted=fftshifted
    )
    means = [
        einops.reduce(shell, '... shell -> ...', reduction='mean')
        for shell in shells
    ]
    return einops.rearrange(means, 'shells ... -> ... shells')


def _find_shell_indices_1d(
    values: torch.Tensor, split_values: torch.Tensor
) -> list[torch.Tensor]:
    """Find indices which index to give values either side of split points."""
    sorted, sort_idx = torch.sort(values, descending=False)
    split_idx = torch.searchsorted(sorted, split_values)
    return torch.tensor_split(sort_idx, split_idx)


def _find_shell_indices_2d(
    values: torch.Tensor, split_values: torch.Tensor
) -> list[torch.Tensor]:
    """Find 2D indices which index to give values either side of split values."""
    idx_2d = coordinate_grid(values.shape[-2:]).long()
    values = einops.rearrange(values, 'h w -> (h w)')
    idx_2d = einops.rearrange(idx_2d, 'h w idx -> (h w) idx')
    sorted, sort_idx = torch.sort(values, descending=False)
    split_idx = torch.searchsorted(sorted, split_values)
    return torch.tensor_split(idx_2d[sort_idx], split_idx)


def _find_shell_indices_3d(
    values: torch.Tensor, split_values: torch.Tensor
) -> list[torch.Tensor]:
    """Find 3D indices which index to give values either side of split values."""
    idx_3d = coordinate_grid(values.shape[-3:]).long()
    values = einops.rearrange(values, 'd h w -> (d h w)')
    idx_3d = einops.rearrange(idx_3d, 'd h w idx -> (d h w) idx')
    sorted, sort_idx = torch.sort(values, descending=False)
    split_idx = torch.searchsorted(sorted, split_values)
    return torch.tensor_split(idx_3d[sort_idx], split_idx)


def _split_into_frequency_bins_2d(
    dft: torch.Tensor,
    n_bins: int,
    image_shape: tuple[int, int],
    rfft: bool = False,
    fftshifted: bool = False
) -> list[torch.Tensor]:
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted,
        norm=True,
        device=dft.device,
    )
    frequency_grid = einops.rearrange(frequency_grid, 'h w -> (h w)')
    shell_borders = _frequency_bin_borders(n_bins)
    shell_indices = _find_shell_indices_1d(frequency_grid, split_values=shell_borders)
    dft = einops.rearrange(dft, '... h w -> ... (h w)')
    shells = [
        dft[..., shell_idx]
        for shell_idx in shell_indices
    ]
    return shells[:-1]


def _split_into_shells_3d(
    dft: torch.Tensor,
    split_values: torch.Tensor,
    rfft: bool = False,
    fftshifted: bool = True
) -> List[torch.Tensor]:
    frequencies = fftfreq_grid(
        image_shape=dft.shape[-3:],
        rfft=rfft,
        fftshift=fftshifted,
        norm=True,
        device=dft.device,
    )
    frequencies = einops.rearrange(frequencies, 'd h w -> (d h w)')
    shell_indices = _find_shell_indices_1d(frequencies, split_values=split_values)
    dft = einops.rearrange(dft, '... d h w -> ... (d h w)')
    shells = [
        dft[..., shell_idx]
        for shell_idx in shell_indices
    ]
    return shells


def _1d_to_rotational_average_2d_dft(
    values: torch.Tensor,
    image_shape: tuple[int, ...],
    rfft: bool = False,
    fftshifted: bool = True,
) -> torch.Tensor:
    h, w = image_shape[-2:]
    h, w = rfft_shape((h, w)) if rfft is True else (h, w)
    result_shape = (*image_shape[:-2], h, w)
    average_2d = torch.zeros(size=result_shape, dtype=values.dtype, device=values.device)
    frequencies = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted,
        norm=True,
        device=values.device
    )

    # define frequency bin centers
    centers = torch.linspace(0, 0.5, steps=values.shape[-1])
    df = centers[1]

    # define split points in data as midpoint between bin centers
    centers = torch.cat([centers, torch.as_tensor([0.5 + df])])
    centers = centers.unfold(dimension=0, size=2, step=1)  # (n_shells, 2)
    split_points = einops.reduce(centers, 'shells high_low -> shells', reduction='mean')

    # find 2d indices for elements in each distance shell
    shell_idx = _find_shell_indices_2d(values=frequencies, split_values=split_points)[:-1]

    # insert data into each shell
    for idx, shell in enumerate(shell_idx):
        idx_h, idx_w = einops.rearrange(shell, 'b idx -> idx b')
        average_2d[..., idx_h, idx_w] = values[..., [idx]]
    average_2d[..., frequencies > 0.5] = values[..., [-1]]
    return average_2d


def _frequency_bin_centers(n: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.linspace(0, 0.5, steps=n, device=device)


def _frequency_bin_borders(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Values at the borders of DFT sample frequency bins."""
    bin_centers = _frequency_bin_centers(n, device=device)
    df = torch.atleast_1d(bin_centers[1])
    bin_centers = torch.concatenate([bin_centers, 0.5 + df], dim=0)  # (b+1, )
    adjacent_bins = bin_centers.unfold(dimension=0, size=2, step=1)  # (b, 2)
    return einops.reduce(adjacent_bins, 'b high_low -> b', reduction='mean')