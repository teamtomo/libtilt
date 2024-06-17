import torch

from libtilt.filters import bfactor_2d
from libtilt.pytest_utils import device_test


@device_test
def test_bfactor_2d():
    # Generate an image
    image = torch.zeros((4,4))
    image[1:, :] = 1
    # apply bfactor
    result = bfactor_2d(
        image=image,
        B=10,
        pixel_size=1,
    )
    expected = torch.tensor(
        [[0.18851198,0.18851198,0.18851198,0.18851198,],
         [0.88381535,0.88381535,0.88381535,0.88381535],
         [1.0438573,1.0438573,1.0438573,1.0438573],
         [0.88381535,0.88381535,0.88381535,0.88381535]
         ]
    )
    assert torch.allclose(result, expected, atol=1e-3)
