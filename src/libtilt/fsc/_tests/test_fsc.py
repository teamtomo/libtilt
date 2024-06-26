import torch

from libtilt.fsc.fsc import fsc
from libtilt.pytest_utils import device_test


@device_test
def test_fsc_identical_images():
    a = torch.rand(size=(10, 10))
    result = fsc(a, a)
    assert torch.allclose(result, torch.ones(6))


@device_test
def test_fsc_identical_volumes():
    a = torch.rand(size=(10, 10, 10))
    result = fsc(a, a)
    assert torch.allclose(result, torch.ones(6))


@device_test
def test_fsc_identical_images_with_index_subset():
    a = torch.rand(size=(10, 10))
    rfft_mask = torch.zeros(size=(10, 6), dtype=torch.bool)
    rfft_mask[::2, 1::2] = 1  # subset of fourier coefficients
    rfft_mask[0, 0] = 1  # include dc
    result = fsc(a, a, rfft_mask=rfft_mask)
    assert torch.allclose(result, torch.ones(6))


@device_test
def test_fsc_identical_volumes_with_index_subset():
    a = torch.rand(size=(10, 10, 10))
    rfft_mask = torch.zeros(size=(10, 10, 6), dtype=torch.bool)
    rfft_mask[::2, 1::2, ::2] = 1  # subset of fourier coefficients
    rfft_mask[0, 0, 0] = 1  # include dc
    result = fsc(a, a, rfft_mask=rfft_mask)
    assert torch.allclose(result, torch.ones(6))