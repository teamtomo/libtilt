import einops
import torch

from libtilt.grids import fftfreq_grid
from libtilt.utils.fft import rfft_shape


def fsc(
    a: torch.Tensor,
    b: torch.Tensor,
    rfft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring/shell correlation between two square/cubic images."""
    # input handling
    image_shape = a.shape
    dft_shape = rfft_shape(image_shape)
    if a.ndim not in (2, 3):
        raise ValueError('images must be 2D or 3D.')
    elif a.shape != b.shape:
        raise ValueError('images must be the same shape.')
    elif rfft_mask is not None and rfft_mask.shape != dft_shape:
        raise ValueError('valid rfft indices must have same shape as rfft.')

    # fsc calculation
    a, b = torch.fft.rfftn(a), torch.fft.rfftn(b)
    shift_dims = (-3, -2) if a.ndim == 3 else (-2,)
    a, b = torch.fft.fftshift(a, dim=shift_dims), torch.fft.fftshift(b, dim=shift_dims)
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
        norm=True,
        device=a.device,
    )
    if rfft_mask is not None:
        a, b, frequencies = (arr[rfft_mask] for arr in [a, b, frequency_grid])
    else:
        a, b, frequencies = (torch.flatten(arg) for arg in [a, b, frequency_grid])

    # split frequencies into frequency bins
    bin_centers = torch.fft.rfftfreq(image_shape[0])
    df = 1 / image_shape[0]
    bin_centers = torch.cat([bin_centers, torch.as_tensor([0.5 + df])])
    bin_centers = bin_centers.unfold(dimension=0, size=2, step=1)  # (n_shells, 2)
    split_points = einops.reduce(bin_centers, 'shells high_low -> shells', reduction='mean')
    sorted, sort_idx = torch.sort(frequencies, descending=False)
    split_idx = torch.searchsorted(sorted, split_points)
    shell_idx = torch.tensor_split(sort_idx, split_idx)[:-1]

    # calculate normalised cross correlation in each shell
    fsc = [
        _normalised_cc_complex_1d(a[idx], b[idx])
        for idx in
        shell_idx
    ]
    return torch.real(torch.tensor(fsc))


def _normalised_cc_complex_1d(a: torch.Tensor, b: torch.Tensor):
    correlation = torch.dot(a, torch.conj(b))
    return correlation / (torch.linalg.norm(a) * torch.linalg.norm(b))
