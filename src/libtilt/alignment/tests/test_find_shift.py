import torch
import pytest

from libtilt.alignment import find_image_shift


def test_find_image_shift():
    a = torch.zeros((4, 4))
    a[1, 1] = 1
    b = torch.zeros((4, 4))
    b[2, 2] = .7
    b[2, 3] = .3

    with pytest.raises(ValueError, match=r'Upsampling factor .*'):
        find_image_shift(a, b, upsampling_factor=0.5)

    shift = find_image_shift(a, b, upsampling_factor=5)
    assert torch.all(shift == -1), ("Interpolating a shift too close to a border is "
                                   "not possible, so an integer shift should be "
                                   "returned.")

    a = torch.zeros((8, 8))
    a[3, 3] = 1
    b = torch.zeros((8, 8))
    b[4, 4] = .7
    b[4, 5] = .3
    shift = find_image_shift(a, b, upsampling_factor=1)
    assert torch.all(shift == -1), ("Finding shift with upsampling_factor of 1 should "
                                   "return an integer shift (i.e. no interpolation.")

    shift = find_image_shift(a, b)
    assert shift[0] == -1.1, "y shift should be interpolated to specific value."
    assert shift[1] == -1.2, "x shift should be interpolated to specific value."

