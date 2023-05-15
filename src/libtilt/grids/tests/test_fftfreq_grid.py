import torch

from libtilt.grids import fftfreq_grid, central_slice_grid


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
