import einops
import torch

from libtilt.grids.fftfreq import _construct_fftfreq_grid_2d, _construct_fftfreq_grid_3d
from libtilt.utils.fft import fftshift_2d, fftshift_3d


def get_phase_shifts_2d(
    shifts: torch.Tensor, image_shape: tuple[int, int], rfft: bool = False
):
    """Generate a complex-valued array of phase shifts for 2D images.
    todo: add fftshift support and make sure that it doesn't break proj/bproj cycle

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 2)` array of shifts in the last two image dimensions `(h, w)`.
    image_shape: tuple[int, int]
        `(h, w)` of 2D image(s) on which phase shifts will be applied.
    rfft: bool
        If `True` the phase shifts generated will be compatible with
        the non-redundant half DFT outputs of the FFT for real inputs from `rfft`.

    Returns
    -------
    phase_shifts: torch.Tensor
        `(..., h, w)` complex valued array of phase shifts for the fft or rfft
        of images with `image_shape`.
    """
    fftfreq_grid = _construct_fftfreq_grid_2d(
        image_shape=image_shape, rfft=rfft, device=shifts.device
    )  # (h, w, 2)
    shifts = einops.rearrange(shifts, '... shift -> ... 1 1 shift')
    angles = einops.reduce(
        -2 * torch.pi * fftfreq_grid * shifts, '... h w 2 -> ... h w', reduction='sum'
    )  # radians/cycle, cycles/sample, samples
    return torch.complex(real=torch.cos(angles), imag=torch.sin(angles))


def get_phase_shifts_3d(
    shifts: torch.Tensor,
    image_shape: tuple[int, int, int],
    rfft: bool = False,
    fftshift: bool = False
):
    """Generate a complex-valued array of phase shifts for 3D images.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 3)` array of shifts in the last three image dimensions `(d, h, w)`.
    image_shape: tuple[int, int, int]
        `(d, h, w)` of 3D image(s) onto which phase shifts will be applied.
    rfft: bool
        If `True` the phase shifts generated will be compatible with
        the non-redundant half DFT outputs of the FFT for real inputs from `rfft`.
    fftshift: bool
        If `True`, fftshift the output.

    Returns
    -------
    phase_shifts: torch.Tensor
        `(..., d, h, w)` complex valued array of phase shifts for the fft or rfft
        of images with `image_shape`.
    """
    fftfreq_grid = _construct_fftfreq_grid_3d(
        image_shape=image_shape, rfft=rfft, device=shifts.device
    )  # (d, h, w, 3)
    shifts = einops.rearrange(shifts, '... shift -> ... 1 1 1 shift')
    angles = einops.reduce(
        -2 * torch.pi * fftfreq_grid * shifts, '... d h w 3 -> ... d h w', reduction='sum'
    )  # radians/cycle, cycles/sample, samples
    phase_shifts = torch.complex(real=torch.cos(angles), imag=torch.sin(angles))
    if fftshift is True:
        phase_shifts = fftshift_3d(phase_shifts, rfft=rfft)
    return phase_shifts


def phase_shift_dft_2d(
    dft: torch.Tensor,
    image_shape: tuple[int, int],
    shifts: torch.Tensor,
    rfft: bool = False,
    fftshifted: bool = False,
):
    """Apply phase shifts to 2D discrete Fourier transforms.

    Parameters
    ----------
    dft: torch.Tensor
        `(..., h, w)` array containing DFTs.
    image_shape: tuple[int, int]
        `(h, w)` of images prior to DFT computation.
    shifts: torch.Tensor
        `(..., 2)` array of 2D shifts in `h` and `w`.
    rfft: bool
        Whether the input was computed using `rfft`.
    fftshifted: bool
        Whether the DFTs have been fftshifted to center_grid the DC component.

    Returns
    -------
    shifted_dfts: torch.Tensor
        `(..., h, w)` array of DFTs with phase shifts applied.
    """
    phase_shifts = get_phase_shifts_2d(
        shifts=shifts,
        image_shape=image_shape,
        rfft=rfft,
        # fftshift=fftshifted,
    )
    if fftshifted is True:
        phase_shifts = fftshift_2d(phase_shifts, rfft=rfft)
    return dft * phase_shifts


def phase_shift_dft_3d(
    dft: torch.Tensor,
    image_shape: tuple[int, int, int],
    shifts: torch.Tensor,
    rfft: bool = False,
    fftshifted: bool = False,
):
    """Apply phase shifts to 3D discrete Fourier transforms.

    Parameters
    ----------
    dft: torch.Tensor
        `(..., h, w)` array containing DFTs.
    image_shape: tuple[int, int, int]
        `(h, w)` of images prior to DFT computation.
    shifts: torch.Tensor
        `(..., 3)` array of 3D shifts in `d`, `h` and `w`.
    rfft: bool
        Whether the input was computed using `rfft`.
    fftshifted: bool
        Whether the DFTs have been fftshifted to center_grid the DC component.

    Returns
    -------
    shifted_dfts: torch.Tensor
        `(..., h, w)` array of DFTs with phase shifts applied.
    """
    phase_shifts = get_phase_shifts_3d(
        shifts=shifts,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted,
    )
    if fftshifted is True:
        phase_shifts = fftshift_3d(phase_shifts, rfft=rfft)
    return dft * phase_shifts


def fourier_shift_2d(images: torch.Tensor, shifts: torch.Tensor):
    h, w = images.shape[-2:]
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = phase_shift_dft_2d(
        images,
        image_shape=(h, w),
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    images = torch.fft.irfftn(images, dim=(-2, -1))
    return torch.real(images)


def fourier_shift_3d(images: torch.Tensor, shifts: torch.Tensor):
    d, h, w = images.shape[-3:]
    images = torch.fft.rfftn(images, dim=(-3, -2, -1))
    images = phase_shift_dft_3d(
        images,
        image_shape=(d, h, w),
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    images = torch.fft.irfftn(images, dim=(-3, -2, -1))
    return torch.real(images)
