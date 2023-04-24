import torch
import einops

from libtilt.grids import fftfreq_grid


def bandpass_2d(
    image: torch.Tensor,
    low: float,
    high: float,
    falloff: float = 0,
):
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    bandpassed_dft = bandpass_dft(
        dft=dft,
        image_shape=image.shape[-2:],
        low=low,
        high=high,
        falloff=falloff,
        rfft=True,
        fftshifted=False,
    )
    return torch.real(torch.fft.irfftn(bandpassed_dft, dim=(-2, -1)))


def bandpass_3d(
    image: torch.Tensor,
    low: float,
    high: float,
    falloff: float = 0,
):
    dft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    bandpassed_dft = bandpass_dft(
        dft=dft,
        image_shape=image.shape[-3:],
        low=low,
        high=high,
        falloff=falloff,
        rfft=True,
        fftshifted=False,
    )
    return torch.real(torch.fft.irfftn(bandpassed_dft, dim=(-3, -2, -1)))


def bandpass_filter(
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


def bandpass_dft(
    dft: torch.Tensor,
    image_shape: tuple[int, int] | tuple[int, int, int],
    low: float,
    high: float,
    falloff: float,
    rfft: bool,
    fftshifted: bool = False,
):
    filter = bandpass_filter(
        low=low,
        high=high,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted
    )
    return dft * filter
