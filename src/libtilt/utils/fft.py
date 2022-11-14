from typing import Sequence, List, Tuple

import einops
import torch


def rfft_shape_from_signal_shape(input_shape: Sequence[int]) -> Tuple[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def construct_fftfreq_grid_2d(image_shape: Sequence[int], rfft: bool) -> torch.Tensor:
    """Construct a grid of DFT sample frequencies for a 2D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 2D shape `(h, w)` of the input image for which a grid of DFT sample frequencies
        should be calculated.
    rfft: bool
        Controls Whether the frequency grid is for a real fft (rfft).

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 2)` array of DFT sample frequencies.
        Order of frequencies in the last dimension corresponds to the order of dimensions
        of the grid.
    """
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    dft_shape = rfft_shape_from_signal_shape(image_shape) if rfft is True else image_shape
    freq_y = torch.fft.fftfreq(image_shape[-2])
    freq_x = last_axis_frequency_func(image_shape[-1])
    freq_yy = einops.repeat(freq_y, 'h -> h w', w=dft_shape[-1])
    freq_xx = einops.repeat(freq_x, 'w -> h w', h=dft_shape[-2])
    return einops.rearrange([freq_yy, freq_xx], 'freq h w -> h w freq')


def rfft_to_symmetrised_dft_2d(rfft: torch.Tensor) -> torch.Tensor:
    """Construct a symmetrised discrete Fourier transform from an rfft.

    The symmetrised discrete Fourier transform contains a full FFT with components at
    the Nyquist frequency repeated on both sides. This yields a spectrum which is
    symmetric around the DC component of the FFT, useful for some applications.

    This is only valid for rffts of cubic inputs with even sidelength.
    Input should be the result of calling rfft on input data.

    1D example:
    - rfftfreq: `[0.0000, 0.1667, 0.3333, 0.5000]`
    - fftshifted fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`
    - symmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333,  0.5000]`
    """
    r = rfft.shape[-2]  # original dim length
    dc = r // 2
    if rfft.ndim == 2:
        output = torch.zeros((r + 1, r + 1), dtype=torch.complex64)
    elif rfft.ndim == 3:
        b = rfft.shape[0]
        output = torch.zeros((b, r + 1, r + 1), dtype=torch.complex64)
    # fftshift full length dims to center DC component
    rfft = torch.fft.fftshift(rfft, dim=(-2,))
    output[..., :-1, dc:] = rfft  # place rfft in output
    output[..., -1, dc:] = rfft[..., 0, :]  # replicate components at Nyquist
    # fill redundant half
    output[..., :, :dc] = torch.flip(torch.conj(output[..., :, dc + 1:]), dims=(-2, -1))
    return output


def rfft_to_symmetrised_dft_3d(rfft: torch.Tensor) -> torch.Tensor:
    """Construct a symmetrised discrete Fourier transform from an rfft.

    The symmetrised discrete Fourier transform contains a full FFT with components at
    the Nyquist frequency repeated on both sides. This yields a spectrum which is
    symmetric around the DC component of the FFT, useful for some applications.

    This is only valid for rffts of cubic inputs with even sidelength.
    Input should be the result of calling rfft on input data.

    1D example:
    - rfftfreq: `[0.0000, 0.1667, 0.3333, 0.5000]`
    - fftshifted fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`
    - symmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333,  0.5000]`
    """
    r = rfft.shape[-3]  # input dim length
    dc = r // 2  # index for DC component
    output = torch.zeros((r + 1, r + 1, r + 1), dtype=torch.complex64)
    # fftshift full length dims (i.e. not -1) to center DC component
    rfft = torch.fft.fftshift(rfft, dim=(-3, -2))
    # place rfft in output
    output[:-1, :-1, dc:] = rfft
    # replicate components at nyquist (symmetrise)
    output[:-1, -1, dc:] = rfft[:, 0, :]
    output[-1, :-1, dc:] = rfft[0, :, :]
    output[-1, -1, dc:] = rfft[0, 0, :]
    # fill redundant half-spectrum
    output[:, :, :dc] = torch.flip(torch.conj(output[:, :, dc + 1:]), dims=(-3, -2, -1))
    return output
