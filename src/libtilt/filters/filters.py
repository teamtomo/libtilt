import einops
import torch

from libtilt.grids import fftfreq_grid


def band_pass_filter(
    low: float,
    high: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None
) -> torch.Tensor:
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    band = torch.logical_and(frequency_grid > low, frequency_grid <= high)
    band_float = band.float()

    # add cosine falloff
    band_with_falloff = torch.logical_and(
        frequency_grid > low - falloff, frequency_grid <= high + falloff
    )
    falloff_mask = torch.logical_and(band_with_falloff, ~band)
    cutoffs = torch.tensor([low, high], dtype=torch.float32, device=device)
    cutoffs = einops.rearrange(cutoffs, pattern='cutoffs -> cutoffs 1')
    distance = torch.abs(frequency_grid[falloff_mask] - cutoffs)
    distance = einops.reduce(distance, 'cutoffs b -> b', reduction='min')
    softened_values = torch.cos((distance / falloff) * (torch.pi / 2))
    band_float[falloff_mask] = softened_values
    return band_float


def low_pass_filter(
    cutoff: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
) -> torch.Tensor:
    filter = band_pass_filter(
        low=0,
        high=cutoff,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device
    )
    return filter


def high_pass_filter(
    cutoff: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
) -> torch.Tensor:
    filter = band_pass_filter(
        low=cutoff,
        high=1,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
    )
    return filter
