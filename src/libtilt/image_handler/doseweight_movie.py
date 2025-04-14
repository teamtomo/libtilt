import einops
import torch

from libtilt.grids import fftfreq_grid


def critical_exposure(fft_freq: torch.Tensor) -> torch.Tensor:
    # From Grant and Grigorieff 2015
    a = 0.245
    b = -1.665
    c = 2.81
    # Add small epsilon to prevent division by zero
    eps = 1e-10
    Ne = a * torch.pow(fft_freq.clamp(min=eps), b) + c
    return Ne


def critical_exposure_Bfac(fft_freq: torch.Tensor, Bfac: float) -> torch.Tensor:
    eps = 1e-10
    Ne = 2 / (Bfac * fft_freq.clamp(min=eps) ** 2)
    return Ne


def dose_weight_2d(
    movie: torch.Tensor, pixel_size: float = 1, flux: float = 1, Bfac: float = -1
) -> torch.Tensor:
    """
    Dose weight a 2D movie using the method described in Grant and Grigorieff 2015.

    Parameters
    ----------
    movie : torch.Tensor
        The movie to dose weight.
    pixel_size : float
        The pixel size of the movie.
    flux : float
        The fluence per frame.
    Bfac : float
        The B factor for dose weighting, -1 is use Grant and Grigorieff values.

    Returns
    -------
    torch.Tensor
        The dose weighted summed micrograph.
    """
    # FFT the movie frames
    dft = torch.fft.rfftn(movie, dim=(-2, -1))

    # Get the frequency grid for 1 frame
    fft_freq_px = (
        fftfreq_grid(
            image_shape=movie[0].shape,
            rfft=True,
            fftshift=False,
            norm=True,
        )
        / pixel_size
    )

    # Get the critical exposure for each frequency
    Ne = torch.zeros_like(dft.real)
    if Bfac < 0:
        Ne[0] = critical_exposure(fft_freq=fft_freq_px)
    else:
        Ne[0] = critical_exposure_Bfac(fft_freq=fft_freq_px, Bfac=Bfac)
    # Copy over each frame for ease of computation
    Ne[:] = Ne[0]

    # Set up an exposure array with frame indices multiplied by flux
    frame_exposure = torch.arange(1, Ne.shape[0] + 1, device=Ne.device) * flux
    frame_exposure = einops.rearrange(frame_exposure, "z -> z 1 1") * torch.ones_like(
        Ne
    )

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


def dose_weight_3d_volume(
    volume: torch.Tensor,
    num_frames=float,
    pixel_size: float = 1,
    flux: float = 1,
    Bfac: float = -1,
) -> torch.Tensor:
    """
    Dose weight a 3D volume using the method described in Grant and Grigorieff 2015.

    Parameters
    ----------
    volume : torch.Tensor
        The volume to dose weight.
    num_frames : int
        The number of frames for dose weighting.
    pixel_size : float
        The pixel size of the volume.
    flux : float
        The fluence per frame.
    Bfac : float
        The B factor for dose weighting, -1 is use Grant and Grigorieff values.

    Returns
    -------
    torch.Tensor
        The dose weighted volume.
    """
    device = volume.device
    # FFT the volume
    dft = torch.fft.rfftn(volume.contiguous(), dim=(-3, -2, -1))

    # Get the frequency grid for 1 frame
    fft_freq_px = (
        fftfreq_grid(
            image_shape=volume.shape,
            rfft=True,
            fftshift=False,
            norm=True,
            device=device,
        )
        / pixel_size
    )

    # Get the critical exposure for each frequency
    Ne = (
        critical_exposure_Bfac(fft_freq=fft_freq_px, Bfac=Bfac)
        if Bfac >= 0
        else critical_exposure(fft_freq=fft_freq_px)
    )

    # Set up an exposure array with frame indices multiplied by flux
    frame_exposure = (
        torch.arange(1, num_frames + 1, device=device).view(-1, 1, 1, 1) * flux
    )

    # Calculate the relative amplitudes
    amplitudes = torch.exp(-frame_exposure / (2 * Ne[None, ...]))

    # Sum in Fourier space
    amp_dft = amplitudes * dft
    fspace_sum = amp_dft.sum(dim=0) / torch.sqrt((amplitudes**2).sum(dim=0))

    # Convert back to real space
    rspace_sum = torch.fft.irfftn(fspace_sum, dim=(-3, -2, -1)) * (num_frames**0.5)

    return rspace_sum


def cumulative_dose_filter_3d(
    volume: torch.Tensor,
    num_frames=float,
    start_exposure=0.0,
    pixel_size: float = 1,
    flux: float = 1,
    Bfac: float = -1,
) -> torch.Tensor:
    """
    Dose weight a 3D volume using the method described in Grant and Grigorieff 2015.

    Use integration to speed up.

    Parameters
    ----------
    volume : torch.Tensor
        The volume to dose weight.
    num_frames : int
        The number of frames for dose weighting.
    start_exposure : float
        The start exposure for dose weighting.
    pixel_size : float
        The pixel size of the volume.
    flux : float
        The fluence per frame.
    Bfac : float
        The B factor for dose weighting, -1 is use Grant and Grigorieff values.

    Returns
    -------
    torch.Tensor
        The dose weighting filter.
    """
    device = volume.device

    end_exposure = start_exposure + num_frames * flux
    print(f"End exposure: {end_exposure}")
    # Get the frequency grid for 1 frame
    fft_freq_px = (
        fftfreq_grid(
            image_shape=volume.shape,
            rfft=True,
            fftshift=False,
            norm=True,
            device=device,
        )
        / pixel_size
    )
    print(f"fft_freq_px min: {fft_freq_px.min()}, max: {fft_freq_px.max()}")
    # Get the critical exposure for each frequency
    Ne = (
        critical_exposure_Bfac(fft_freq=fft_freq_px, Bfac=Bfac)
        if Bfac >= 0
        else critical_exposure(fft_freq=fft_freq_px)
    )

    # Add small epsilon to prevent division by zero
    eps = 1e-10
    Ne = Ne.clamp(min=eps)
    print(f"Ne min: {Ne.min()}, max: {Ne.max()}")

    return (
        2
        * Ne
        * (
            torch.exp((-0.5 * start_exposure) / Ne)
            - torch.exp((-0.5 * end_exposure) / Ne)
        )
        / end_exposure
    )
