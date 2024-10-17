import einops
import torch

from libtilt.grids import fftfreq_grid


def critical_exposure(fft_freq: torch.Tensor) -> torch.Tensor:
    # From Grant and Grigorieff 2015
    a = 0.245
    b = -1.665
    c = 2.81
    Ne = a * torch.pow(fft_freq, b) + c
    return Ne


def dose_weight_2d(
    movie: torch.Tensor,
    pixel_size: float = 1,
    flux: float = 1,
) -> torch.Tensor:

    # FFT the movie frames
    dft = torch.fft.rfftn(movie, dim=(-2, -1))

    # Get the frequency grid for 1 frame
    fft_freq = fftfreq_grid(
        image_shape=movie[0].shape,
        rfft=True,
        fftshift=False,
        spacing=pixel_size,
        norm=True,
    )

    # Get the critical exposure for each frequency
    Ne = torch.zeros_like(dft.real)
    Ne[0] = critical_exposure(fft_freq=fft_freq)
    # Copy over each frame for ease of computation
    Ne[:] = Ne[0]

    # Set up an exposure array of the same dimensions
    frame_exposure = torch.ones_like(Ne)
    for i in range(1, Ne.shape[0] + 1):
        frame_exposure[i - 1] *= i * flux

    # Calculate the relative amplitudes
    amplitudes = torch.exp(-frame_exposure / (2 * Ne))

    # Sum in Fourier space
    fspace_sum = einops.reduce(amplitudes * dft, "z y x -> y x", "sum") / (
        (einops.reduce(amplitudes**2, "z y x -> y x", "sum")) ** 0.5
    )

    # Convert back to real space
    # Because I summed in Fourier space I am multiplying
    # by the sqrt of number of frames so the mean and std will match the original
    rspace_sum = torch.fft.irfftn(fspace_sum, dim=(-2, -1)) * (movie.shape[0]) ** 0.5
    return rspace_sum
