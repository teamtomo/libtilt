import torch
import numpy as np

from libtilt.correlation import correlate_2d, correlate_dft_2d
from libtilt.fft_utils import fftshift_2d


def test_correlate_2d():
    a = torch.zeros((10, 10))
    a[5, 5] = 1
    b = torch.zeros((10, 10))
    b[6, 6] = 1
    cross_correlation = correlate_2d(a, b, normalize=True)
    peak_position = np.unravel_index(
        indices=torch.argmax(cross_correlation), shape=cross_correlation.shape
    )
    shift = torch.as_tensor(peak_position) - torch.tensor([5, 5])
    assert peak_position == (4, 4)
    assert torch.allclose(shift, torch.tensor([-1, -1]))
    assert torch.allclose(cross_correlation[peak_position], torch.tensor([1.]))


def test_correlate_dft_2d():
    a = torch.zeros((10, 10))
    a[5, 5] = 1
    b = torch.zeros((10, 10))
    b[6, 6] = 1

    expected = torch.zeros((10, 10))
    expected[4, 4] = 1

    a_fft2 = torch.fft.fft2(a)
    b_fft2 = torch.fft.fft2(b)
    assert torch.allclose(
        correlate_dft_2d(a_fft2, b_fft2, rfft=False, fftshifted=False),
        expected,
        atol=1e-6
    )

    a_rfft2 = torch.fft.rfft2(a)
    b_rfft2 = torch.fft.rfft2(b)
    assert torch.allclose(
        correlate_dft_2d(a_rfft2, b_rfft2, rfft=True, fftshifted=False),
        expected,
        atol=1e-6,
    )

    a_fft2_shifted = fftshift_2d(a_fft2, rfft=False)
    b_fft2_shifted = fftshift_2d(b_fft2, rfft=False)
    assert torch.allclose(
        correlate_dft_2d(a_fft2_shifted, b_fft2_shifted, rfft=False, fftshifted=True),
        expected,
        atol=1e-6,
    )

    a_rfft2_shifted = fftshift_2d(a_rfft2, rfft=True)
    b_rfft2_shifted = fftshift_2d(b_rfft2, rfft=True)
    assert torch.allclose(
        correlate_dft_2d(a_rfft2_shifted, b_rfft2_shifted, rfft=True, fftshifted=True),
        expected,
        atol=1e-6,
    )



