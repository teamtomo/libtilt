import einops
import torch
import torch.nn.functional as F

from libtilt.interpolation.interpolate_dft_3d import sample_dft_3d


def test_extract_slices():
    volume = torch.zeros(4, 4, 4, dtype=torch.complex64)

    # insert unrotated slice
    slice = torch.arange(4 ** 2).reshape(4, 4)
    volume[2, :, :] = slice

    # construct slice coordinates
    x = y = torch.arange(4)
    xx = einops.repeat(x, 'w -> h w', h=4)
    yy = einops.repeat(y, 'h -> h w', w=4)
    zz = torch.ones_like(xx) * 2
    slice_coordinates = einops.rearrange([zz, yy, xx], 'zyx h w -> h w zyx')

    # extract slice
    extracted_slice = sample_dft_3d(dft=volume, coordinates=slice_coordinates)
    error = F.mse_loss(slice, torch.real(extracted_slice.squeeze()))
    assert error < 1e-10