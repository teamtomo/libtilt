import numpy as np
import pytest
import torch

from libtilt.utils.fft import construct_fftfreq_grid_2d, \
    rfft_shape_from_signal_shape, \
    rfft_to_symmetrised_dft_2d, rfft_to_symmetrised_dft_3d, \
    symmetrised_dft_to_dft_2d, \
    symmetrised_dft_to_rfft_2d, symmetrised_dft_to_dft_3d


def test_rfft_shape_from_signal_shape():
    # even
    signal_shape = (2, 4, 8, 16)
    signal = np.random.random(signal_shape)
    rfft = np.fft.rfftn(signal)
    assert rfft.shape == rfft_shape_from_signal_shape(signal_shape)

    # odd
    signal_shape = (3, 9, 27, 81)
    signal = np.random.random(signal_shape)
    rfft = np.fft.rfftn(signal)
    assert rfft.shape == rfft_shape_from_signal_shape(signal_shape)


def test_construct_fftfreq_grid_2d():
    image_shape = (10, 30)
    # no rfft
    grid = construct_fftfreq_grid_2d(image_shape=image_shape, rfft=False)
    assert grid.shape == (10, 30, 2)
    assert torch.allclose(grid[:, 0, 0], torch.fft.fftfreq(10))
    assert torch.allclose(grid[0, :, 1], torch.fft.fftfreq(30))

    # with rfft
    grid = construct_fftfreq_grid_2d(image_shape=image_shape, rfft=True)
    assert grid.shape == (*rfft_shape_from_signal_shape(image_shape), 2)
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
    assert torch.allclose(desymmetrised_dft, fft, atol=1e-6)


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
    assert torch.allclose(desymmetrised_dft, fft, atol=1e-6)
