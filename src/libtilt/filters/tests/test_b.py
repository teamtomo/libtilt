import torch

from libtilt.filters import bfactor_2d

def test_bfactor_2d():
    # Generate an image
    img1 = torch.zeros((1,4))
    img2 = torch.ones((3,4))
    img = torch.vstack((img1,img2))
    # apply bfactor
    result = bfactor_2d(
        image=img,
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
