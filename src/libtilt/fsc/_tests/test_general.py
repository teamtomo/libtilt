import torch

from libtilt.fsc._general import fsc


def test_fsc_identical_images():
    a = torch.rand(size=(10, 10))
    result = fsc(a, a)
    assert torch.allclose(result, torch.ones(6))


def test_fsc_identical_volumes():
    a = torch.rand(size=(10, 10, 10))
    result = fsc(a, a)
    assert torch.allclose(result, torch.ones(6))


def test_fsc_identical_images_with_index_subset():
    a = torch.rand(size=(10, 10))
    rfft_idx = torch.zeros(size=(10, 6), dtype=torch.bool)
    rfft_idx[::2, ::2] = 1  # subset of fourier coefficients
    rfft_idx[5, 0] = 1  # +dc
    result = fsc(a, a, rfft_mask=rfft_idx)
    assert torch.allclose(result, torch.ones(6))


def test_fsc_identical_volumes_with_index_subset():
    a = torch.rand(size=(10, 10, 10))
    rfft_idx = torch.zeros(size=(10, 10, 6), dtype=torch.bool)
    rfft_idx[::2, ::2, ::2] = 1  # subset of fourier coefficients
    rfft_idx[5, 5, 0] = 1  # +dc
    result = fsc(a, a, rfft_mask=rfft_idx)
    assert torch.allclose(result, torch.ones(6))