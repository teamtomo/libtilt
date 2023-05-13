import numpy as np
import pytest
import torch

from libtilt.utils.fft import (
    rfft_shape,
    rfft_to_symmetrised_dft_2d,
    rfft_to_symmetrised_dft_3d,
    symmetrised_dft_to_dft_2d,
    symmetrised_dft_to_rfft_2d,
    symmetrised_dft_to_dft_3d,
    dft_center,
    fftfreq_to_spatial_frequency,
    spatial_frequency_to_fftfreq,
    fftfreq_to_rfft_coordinates,
)
from libtilt.grids.fftfreq import _construct_fftfreq_grid_2d


def test_rfft_shape_from_signal_shape():
    # even
    signal_shape = (2, 4, 8, 16)
    signal = np.random.random(signal_shape)
    rfft = np.fft.rfftn(signal)
    assert rfft.shape == rfft_shape(signal_shape)

    # odd
    signal_shape = (3, 9, 27, 81)
    signal = np.random.random(signal_shape)
    rfft = np.fft.rfftn(signal)
    assert rfft.shape == rfft_shape(signal_shape)


def test_construct_fftfreq_grid_2d():
    image_shape = (10, 30)
    # no rfft
    grid = _construct_fftfreq_grid_2d(image_shape=image_shape, rfft=False)
    assert grid.shape == (10, 30, 2)
    assert torch.allclose(grid[:, 0, 0], torch.fft.fftfreq(10))
    assert torch.allclose(grid[0, :, 1], torch.fft.fftfreq(30))

    # with rfft
    grid = _construct_fftfreq_grid_2d(image_shape=image_shape, rfft=True)
    assert grid.shape == (*rfft_shape(image_shape), 2)
    assert torch.allclose(grid[:, 0, 0], torch.fft.fftfreq(10))
    assert torch.allclose(grid[0, :, 1], torch.fft.rfftfreq(30))


def test_rfft_to_symmetrised_dft_2d():
    image = torch.rand((10, 10))
    fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))
    rfft = torch.fft.rfftn(image, dim=(-2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_2d(rfft)
    assert torch.allclose(fft, symmetrised_dft[:-1, :-1], atol=1e-7)


def test_rfft_to_symmetrised_dft_2d_batched():
    image = torch.rand((2, 10, 10))  # (b, h, w)
    fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))
    rfft = torch.fft.rfftn(image, dim=(-2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_2d(rfft)
    assert torch.allclose(fft, symmetrised_dft[..., :-1, :-1], atol=1e-7)


def test_rfft_to_symmetrised_dft_3d():
    image = torch.rand((10, 10, 10))
    fft_dims = (-3, -2, -1)
    fft = torch.fft.fftshift(torch.fft.fftn(image, dim=fft_dims), dim=fft_dims)
    rfft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_3d(rfft)
    np.array(fft - symmetrised_dft[:-1, :-1, :-1])
    assert torch.allclose(fft, symmetrised_dft[:-1, :-1, :-1], atol=1e-5)


@pytest.mark.parametrize(
    "inplace",
    [(True,), (False,)]
)
def test_symmetrised_dft_to_dft_2d(inplace: bool):
    image = torch.rand((10, 10))
    rfft = torch.fft.rfftn(image, dim=(-2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_2d(rfft)
    fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))
    desymmetrised_dft = symmetrised_dft_to_dft_2d(symmetrised_dft,
                                                  inplace=inplace)
    assert torch.allclose(desymmetrised_dft, fft, atol=1e-6)


@pytest.mark.parametrize(
    "inplace",
    [(True,), (False,)]
)
def test_symmetrised_dft_to_dft_2d_batched(inplace: bool):
    image = torch.rand((2, 10, 10))
    rfft = torch.fft.rfftn(image, dim=(-2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_2d(rfft)
    fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))
    desymmetrised_dft = symmetrised_dft_to_dft_2d(symmetrised_dft,
                                                  inplace=inplace)
    assert torch.allclose(desymmetrised_dft, fft, atol=1e-6)


@pytest.mark.parametrize(
    "inplace",
    [(True,), (False,)]
)
def test_symmetrised_dft_to_rfft_2d(inplace: bool):
    image = torch.rand((10, 10))
    rfft = torch.fft.rfftn(image, dim=(-2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_2d(rfft)
    desymmetrised_rfft = symmetrised_dft_to_rfft_2d(symmetrised_dft,
                                                    inplace=inplace)
    assert torch.allclose(desymmetrised_rfft, rfft, atol=1e-6)


@pytest.mark.parametrize(
    "inplace",
    [(True,), (False,)]
)
def test_symmetrised_dft_to_dft_2d_batched(inplace: bool):
    image = torch.rand((2, 10, 10))
    rfft = torch.fft.rfftn(image, dim=(-2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_2d(rfft)
    fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))
    desymmetrised_dft = symmetrised_dft_to_dft_2d(symmetrised_dft,
                                                  inplace=inplace)
    assert torch.allclose(desymmetrised_dft, fft, atol=1e-6)


@pytest.mark.parametrize(
    "inplace",
    [(True,), (False,)]
)
def test_symmetrised_dft_to_dft_3d(inplace: bool):
    image = torch.rand((10, 10, 10))
    rfft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_3d(rfft)
    fft = torch.fft.fftshift(
        torch.fft.fftn(image, dim=(-3, -2, -1)), dim=(-3, -2, -1)
    )
    desymmetrised_dft = symmetrised_dft_to_dft_3d(symmetrised_dft,
                                                  inplace=inplace)
    assert torch.allclose(desymmetrised_dft, fft, atol=1e-5)


@pytest.mark.parametrize(
    "inplace",
    [(True,), (False,)]
)
def test_symmetrised_dft_to_dft_3d_batched(inplace: bool):
    image = torch.rand((2, 10, 10, 10))
    rfft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    symmetrised_dft = rfft_to_symmetrised_dft_3d(rfft)
    fft = torch.fft.fftshift(torch.fft.fftn(
        image, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    desymmetrised_dft = symmetrised_dft_to_dft_3d(symmetrised_dft,
                                                  inplace=inplace)
    assert torch.allclose(desymmetrised_dft, fft, atol=1e-5)


@pytest.mark.parametrize(
    "fftshifted, rfft, input, expected",
    [
        (False, False, (5, 5, 5), torch.tensor([0., 0., 0.])),
        (False, True, (5, 5, 5), torch.tensor([0., 0., 0.])),
        (True, False, (5, 5, 5), torch.tensor([2., 2., 2.])),
        (True, True, (5, 5, 5), torch.tensor([2., 2., 0.])),
        (False, False, (4, 4, 4), torch.tensor([0., 0., 0.])),
        (False, True, (4, 4, 4), torch.tensor([0., 0., 0.])),
        (True, False, (4, 4, 4), torch.tensor([2., 2., 2.])),
        (True, True, (4, 4, 4), torch.tensor([2., 2., 0.])),
    ],
)
def test_fft_center(fftshifted, rfft, input, expected):
    result = dft_center(input, fftshifted=fftshifted, rfft=rfft)
    assert torch.allclose(result, expected.long())


def test_fftfreq_to_spatial_frequency():
    fftfreq = torch.fft.fftfreq(10)
    k = fftfreq_to_spatial_frequency(fftfreq, spacing=2)
    expected = torch.fft.fftfreq(10, d=2)
    assert torch.allclose(k, expected)

    k = fftfreq_to_spatial_frequency(fftfreq, spacing=0.1)
    expected = torch.fft.fftfreq(10, d=0.1)
    assert torch.allclose(k, expected)


def test_spatial_frequency_to_fftfreq():
    k = torch.fft.fftfreq(10, d=2)
    fftfreq = spatial_frequency_to_fftfreq(k, spacing=2)
    expected = torch.fft.fftfreq(10)
    assert torch.allclose(fftfreq, expected)

    k = torch.fft.fftfreq(10, d=0.1)
    fftfreq = spatial_frequency_to_fftfreq(k, spacing=0.1)
    expected = torch.fft.fftfreq(10)
    assert torch.allclose(fftfreq, expected)


def test_fftfreq_to_rfft_coords():
    from libtilt.grids import fftfreq_grid, coordinate_grid
    k = fftfreq_grid(image_shape=(10, 10), rfft=True, fftshift=True)
    result = fftfreq_to_rfft_coordinates(frequencies=k, image_shape=(10, 10))
    expected = coordinate_grid(image_shape=rfft_shape((10, 10)))
    assert torch.allclose(result, expected)