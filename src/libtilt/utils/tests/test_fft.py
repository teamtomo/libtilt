import numpy as np
import torch

from libtilt.utils.fft import construct_fftfreq_grid_2d, rfft_shape_from_signal_shape


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
