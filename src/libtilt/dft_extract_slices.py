import torch
import torch.nn.functional as F
import einops

from libtilt.coordinates import array_coordinates_to_grid_sample_coordinates


def slice_dft(
        dft: torch.Tensor,
        slice_coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample batches of 2D images from a complex cubic volume at specified coordinates.

    Notes
    -----
    - dft should pre-fftshifted to place the origin in Fourier space at the center of the DFT
        i.e. dft should be the result of
            volume -> fftshift(volume) -> fft3(volume) -> fftshift(volume)
    - coordinates should be zyx ordered, match image array dimensions
    - coordinates should be array coordinates, [0, N-1] for a dimension of length N.


    Parameters
    ----------
    dft: torch.Tensor
        (d, h, w) complex valued cubic volume (d == h == w) containing the discrete Fourier transform
        of a cubic volume.
    slice_coordinates: torch.Tensor
        (batch, h, w, zyx) array of coordinates at which `dft` should be sampled.

    Returns
    -------
    samples: torch.Tensor
        (batch, h, w) array of complex valued images sampled from the `dft`
    """

    # cannot sample complex tensors directly with grid_sample
    # c.f. https://github.com/pytorch/pytorch/issues/67634
    # workaround: treat real and imaginary parts as separate channels
    dft = einops.rearrange(torch.view_as_real(dft), 'd h w complex -> complex d h w')
    n_slices = slice_coordinates.shape[0]
    dft = einops.repeat(dft, 'complex d h w -> b complex d h w', b=n_slices)
    slice_coordinates = array_coordinates_to_grid_sample_coordinates(slice_coordinates, array_shape=dft.shape[-3:])
    slice_coordinates = einops.rearrange(slice_coordinates, 'b h w xyz -> b 1 h w xyz')  # add depth dim
    samples = F.grid_sample(
        input=dft,
        grid=slice_coordinates,
        mode='bilinear',  # this is trilinear when input is volumetric
        padding_mode='zeros',
        align_corners=False,
    )
    samples = einops.rearrange(samples, 'b complex 1 h w -> b h w complex')
    samples = torch.view_as_complex(samples.contiguous())
    return samples  # (b, h, w)
