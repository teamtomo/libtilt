import torch

from libtilt.fsc.fsc_conical import fsc_conical
from libtilt.pytest_utils import device_test


@device_test
def test_fsc_conical():
    a = torch.rand((10, 10, 10))
    result = fsc_conical(a, a, cone_direction=(1, 0, 0), cone_aperture=30)
    expected = torch.ones(6)
    assert torch.allclose(result, expected)