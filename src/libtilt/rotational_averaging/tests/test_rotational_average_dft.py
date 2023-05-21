import einops
import torch

from libtilt.grids import fftfreq_grid
from libtilt.rotational_averaging.rotational_average_dft import (
    _split_into_frequency_bins_2d,
    _split_into_frequency_bins_3d,
    rotational_average_dft_2d,
    rotational_average_dft_3d,
)


def test_split_into_frequency_bins_2d():
    # no rfft, fftshifted
    frequencies = fftfreq_grid(
        image_shape=(28, 28),
        rfft=False,
        fftshift=True,
        norm=True
    )  # (h, w)
    frequencies = einops.repeat(frequencies, 'h w -> 2 h w')
    shells = _split_into_frequency_bins_2d(
        frequencies, image_shape=(28, 28), n_bins=15, rfft=False, fftshifted=True
    )
    assert isinstance(shells, list)
    assert len(shells) == 15
    assert isinstance(shells[0], torch.Tensor)
    assert torch.allclose(shells[0], frequencies[:, 14, 14].reshape((2, 1)))

    # rfft, not fftshifted
    frequencies = fftfreq_grid(
        image_shape=(28, 28),
        rfft=True,
        fftshift=False,
        norm=True
    )  # (h, w)
    frequencies = einops.repeat(frequencies, 'h w -> 2 h w')
    shells = _split_into_frequency_bins_2d(
        frequencies, image_shape=(28, 28), n_bins=15, rfft=True, fftshifted=False
    )
    assert isinstance(shells[0], torch.Tensor)
    assert torch.allclose(shells[0], frequencies[:, 0, 0].reshape((2, 1)))

    # rfft, fftshifted
    frequencies = fftfreq_grid(
        image_shape=(28, 28),
        rfft=True,
        fftshift=True,
        norm=True
    )  # (h, w)
    frequencies = einops.repeat(frequencies, 'h w -> 2 h w')
    shells = _split_into_frequency_bins_2d(
        frequencies, image_shape=(28, 28), n_bins=15, rfft=True, fftshifted=True
    )
    assert torch.allclose(shells[0], frequencies[:, 14, 0].reshape((2, 1)))


def test_split_into_shells_3d():
    # no rfft, fftshifted
    frequencies = fftfreq_grid(
        image_shape=(28, 28, 28),
        rfft=False,
        fftshift=True,
        norm=True
    )  # (d, h, w)
    frequencies = einops.repeat(frequencies, 'd h w -> 2 d h w')
    shells = _split_into_frequency_bins_3d(
        frequencies, image_shape=(28, 28, 28), n_bins=15, rfft=False, fftshifted=True
    )
    assert isinstance(shells, list)
    assert len(shells) == 15
    assert isinstance(shells[0], torch.Tensor)
    assert torch.allclose(shells[0], frequencies[:, 14, 14, 14].reshape((2, 1)))

    # rfft, not fftshifted
    frequencies = fftfreq_grid(
        image_shape=(28, 28, 28),
        rfft=True,
        fftshift=False,
        norm=True
    )  # (d, h, w)
    frequencies = einops.repeat(frequencies, 'd h w -> 2 d h w')
    shells = _split_into_frequency_bins_3d(
        frequencies, image_shape=(28, 28, 28), n_bins=15, rfft=True, fftshifted=False
    )
    assert isinstance(shells[0], torch.Tensor)
    assert torch.allclose(shells[0], frequencies[:, 0, 0, 0].reshape((2, 1)))

    # rfft, fftshifted
    frequencies = fftfreq_grid(
        image_shape=(28, 28, 28),
        rfft=True,
        fftshift=True,
        norm=True
    )  # (d, h, w)
    frequencies = einops.repeat(frequencies, 'd h w -> 2 d h w')
    shells = _split_into_frequency_bins_3d(
        frequencies, image_shape=(28, 28, 28), n_bins=15, rfft=True, fftshifted=True
    )
    assert torch.allclose(shells[0], frequencies[:, 14, 14, 0].reshape((2, 1)))


def test_rotational_average_dft_2d():
    # single image
    dft = fftfreq_grid(image_shape=(28, 28), rfft=False, fftshift=True, norm=True)
    rotational_average, bins = rotational_average_dft_2d(
        dft,
        image_shape=(28, 28),
        rfft=False,
        fftshifted=True,
    )
    expected_shape = (len(torch.fft.rfftfreq(28)),)
    assert rotational_average.shape == expected_shape

    # with arbitrary stacking
    dft = fftfreq_grid(image_shape=(28, 28), rfft=False, fftshift=True, norm=True)
    dft = einops.repeat(dft, 'h w -> 2 2 h w')
    rotational_average, bins = rotational_average_dft_2d(
        dft,
        image_shape=(28, 28),
        rfft=False,
        fftshifted=True
    )
    expected_shape = (2, 2, len(torch.fft.rfftfreq(28)))
    assert rotational_average.shape == expected_shape

    # with rfft
    dft = fftfreq_grid(image_shape=(28, 28), rfft=True, fftshift=True, norm=True)
    rotational_average, bins = rotational_average_dft_2d(
        dft,
        image_shape=(28, 28),
        rfft=True,
        fftshifted=True
    )
    expected_shape = (len(torch.fft.rfftfreq(28)),)
    assert rotational_average.shape == expected_shape


def test_rotational_average_return_2d():
    # no batching
    image = fftfreq_grid(image_shape=(28, 28), rfft=False, fftshift=True, norm=True)
    rotational_average, frequency_bins = rotational_average_dft_2d(
        image, image_shape=(28, 28), rfft=False, fftshifted=True, return_2d_average=True
    )
    assert rotational_average.shape == (28, 28)
    assert rotational_average[14, 14] == 0
    assert torch.allclose(rotational_average[0, 0], torch.tensor([0.5]), atol=1e-2)

    # arbitrary stacking
    # no batching
    image = fftfreq_grid(image_shape=(28, 28), rfft=False, fftshift=True, norm=True)
    image = einops.repeat(image, 'h w -> 2 2 h w')
    rotational_average, frequency_bins = rotational_average_dft_2d(
        image, image_shape=(28, 28), rfft=False, fftshifted=True, return_2d_average=True
    )
    assert rotational_average.shape == (2, 2, 28, 28)
    assert torch.all(rotational_average[:, :, 14, 14] == 0)
    assert torch.allclose(rotational_average[:, :, 0, 0], torch.tensor([0.5]),
                          atol=1e-2)


def test_rotational_average_dft_3d():
    # single image
    dft = fftfreq_grid(image_shape=(28, 28, 28), rfft=False, fftshift=True, norm=True)
    rotational_average, bins = rotational_average_dft_3d(
        dft,
        image_shape=(28, 28, 28),
        rfft=False,
        fftshifted=True,
    )
    expected_shape = (len(torch.fft.rfftfreq(28)),)
    assert rotational_average.shape == expected_shape

    # with arbitrary stacking
    dft = fftfreq_grid(image_shape=(28, 28, 28), rfft=False, fftshift=True, norm=True)
    dft = einops.repeat(dft, 'd h w  -> 2 2 d h w')
    rotational_average, bins = rotational_average_dft_3d(
        dft,
        image_shape=(28, 28, 28),
        rfft=False,
        fftshifted=True
    )
    expected_shape = (2, 2, len(torch.fft.rfftfreq(28)))
    assert rotational_average.shape == expected_shape

    # with rfft
    dft = fftfreq_grid(image_shape=(28, 28, 28), rfft=True, fftshift=True, norm=True)
    rotational_average, bins = rotational_average_dft_3d(
        dft,
        image_shape=(28, 28, 28),
        rfft=True,
        fftshifted=True
    )
    expected_shape = (len(torch.fft.rfftfreq(28)),)
    assert rotational_average.shape == expected_shape
