import torch

from libtilt.grids import fftfreq_grid, fftfreq_central_slice
from libtilt.grids.fftfreq import _grid_sinc2


def test_fftfreq_grid_2d():
    input_shape = (6, 6)
    grid = fftfreq_grid(
        image_shape=input_shape,
        rfft=False,
    )
    assert grid.shape == (6, 6, 2)
    h_freq, w_freq = torch.fft.fftfreq(6), torch.fft.fftfreq(6)
    assert torch.allclose(grid[..., 0], h_freq.view((6, 1)))
    assert torch.allclose(grid[..., 1], w_freq)

    grid = fftfreq_grid(
        image_shape=input_shape,
        rfft=True,
    )
    h_freq, w_freq = torch.fft.fftfreq(6), torch.fft.rfftfreq(6)
    assert torch.allclose(grid[..., 0], h_freq.view((6, 1)))
    assert torch.allclose(grid[..., 1], w_freq)

    grid = fftfreq_grid(
        image_shape=input_shape,
        rfft=False,
        fftshift=True,
    )
    h_freq, w_freq = torch.fft.fftfreq(6), torch.fft.fftfreq(6)
    h_freq, w_freq = torch.fft.fftshift(h_freq), torch.fft.fftshift(w_freq)
    assert torch.allclose(grid[..., 0], h_freq.view((6, 1)))
    assert torch.allclose(grid[..., 1], w_freq)

    grid = fftfreq_grid(
        image_shape=input_shape,
        rfft=True,
        fftshift=True,
    )
    h_freq, w_freq = torch.fft.fftfreq(6), torch.fft.rfftfreq(6)
    h_freq = torch.fft.fftshift(h_freq)
    assert torch.allclose(grid[..., 0], h_freq.view((6, 1)))
    assert torch.allclose(grid[..., 1], w_freq)

    grid = fftfreq_grid(
        image_shape=input_shape,
        rfft=False,
        fftshift=True,
        norm=True
    )
    assert torch.allclose(grid[3, 3], torch.tensor([0], dtype=torch.float32))
    assert torch.allclose(grid[3, 0], torch.tensor([0.5]))
    assert torch.allclose(grid[0, 3], torch.tensor([0.5]))
    assert torch.allclose(grid[0, 0], torch.tensor([0.5 ** 0.5]))


def test_fftfreq_central_slice():
    input_shape = (6, 6, 6)

    # no rfft, no fftshift
    central_slice = fftfreq_central_slice(
        image_shape=input_shape,
        rfft=False,
        fftshift=False,
    )
    # check h dim
    assert torch.allclose(central_slice[:, 0, 0], torch.zeros(6))
    assert torch.allclose(central_slice[:, 0, 1], torch.fft.fftfreq(6))
    assert torch.allclose(central_slice[:, 0, 2], torch.zeros(6))

    # check w dim
    assert torch.allclose(central_slice[0, :, 0], torch.zeros(6))
    assert torch.allclose(central_slice[0, :, 1], torch.zeros(6))
    assert torch.allclose(central_slice[0, :, 2], torch.fft.fftfreq(6))

    # no rfft, with fftshift
    central_slice = fftfreq_central_slice(
        image_shape=input_shape,
        rfft=False,
        fftshift=True,
    )
    # check h dim
    assert torch.allclose(central_slice[:, 3, 0], torch.zeros(6))
    assert torch.allclose(central_slice[:, 3, 1],
                          torch.fft.fftshift(torch.fft.fftfreq(6)))
    assert torch.allclose(central_slice[:, 3, 2], torch.zeros(6))

    # check w dim
    assert torch.allclose(central_slice[3, :, 0], torch.zeros(6))
    assert torch.allclose(central_slice[3, :, 1], torch.zeros(6))
    assert torch.allclose(central_slice[3, :, 2],
                          torch.fft.fftshift(torch.fft.fftfreq(6)))

    # with rfft, no fftshift
    central_slice = fftfreq_central_slice(
        image_shape=input_shape,
        rfft=True,
        fftshift=False,
    )
    # check h dim
    assert torch.allclose(central_slice[:, 0, 0], torch.zeros(6))
    assert torch.allclose(central_slice[:, 0, 1], torch.fft.fftfreq(6))
    assert torch.allclose(central_slice[:, 0, 2], torch.zeros(6))

    # check w dim
    assert torch.allclose(central_slice[0, :, 0], torch.zeros(4))
    assert torch.allclose(central_slice[0, :, 1], torch.zeros(4))
    assert torch.allclose(central_slice[0, :, 2], torch.fft.rfftfreq(6))

    # with rfft, with fftshift
    central_slice = fftfreq_central_slice(
        image_shape=input_shape,
        rfft=True,
        fftshift=True,
    )
    # check h dim
    assert torch.allclose(central_slice[:, 0, 0], torch.zeros(6))
    assert torch.allclose(central_slice[:, 0, 1],
                          torch.fft.fftshift(torch.fft.fftfreq(6)))
    assert torch.allclose(central_slice[:, 0, 2], torch.zeros(6))

    # check w dim
    assert torch.allclose(central_slice[3, :, 0], torch.zeros(4))
    assert torch.allclose(central_slice[3, :, 1], torch.zeros(4))
    assert torch.allclose(central_slice[3, :, 2], torch.fft.rfftfreq(6))
