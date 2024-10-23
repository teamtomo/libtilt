import einops
import torch

from libtilt.grids import fftfreq_grid

# from libtilt import fft_utils
from libtilt.rotational_averaging.rotational_average_dft import (
    rotational_average_dft_2d,
)


def bandpass_filter(
    low: float,
    high: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
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
    cutoffs = einops.rearrange(cutoffs, pattern="cutoffs -> cutoffs 1")
    distance = torch.abs(frequency_grid[falloff_mask] - cutoffs)
    distance = einops.reduce(distance, "cutoffs b -> b", reduction="min")
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
    filter = bandpass_filter(
        low=0,
        high=cutoff,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
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
    filter = bandpass_filter(
        low=cutoff,
        high=1,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
    )
    return filter


def b_envelope(
    B: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
) -> torch.Tensor:

    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    frequency_grid_px = frequency_grid / pixel_size
    divisor = 4  # this is 4 for amplitude, 2 for intensity
    b_tensor = torch.exp(-(B * frequency_grid_px**2) / divisor)
    return b_tensor


def whitening_filter(
    image_dft: tuple[int, int] | tuple[int, int, int],
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool = True,
    fftshift: bool = False,
    return_2d_average: bool = True,
    device: torch.device = None,
) -> torch.tensor:

    radial_average = rotational_average_dft_2d(
        dft=torch.absolute(image_dft) ** 2,
        image_shape=image_shape,
        rfft=rfft,
        fftshifted=fftshift,
        return_2d_average=return_2d_average,
    )
    print(radial_average[0].shape)
    whiten_filter = 1 / (radial_average[0]) ** 0.5
    whiten_filter /= whiten_filter.max()

    return whiten_filter
