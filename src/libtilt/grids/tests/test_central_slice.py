import torch

from libtilt.grids import central_slice_grid


def test_central_slice_grid():
    input_shape = (6, 6, 6)

    # no rfft, no fftshift
    central_slice = central_slice_grid(
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
    central_slice = central_slice_grid(
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
    central_slice = central_slice_grid(
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
    central_slice = central_slice_grid(
        image_shape=input_shape,
        rfft=True,
        fftshift=True,
    )
    # check h dim
    assert torch.allclose(central_slice[:, 0, 0], torch.zeros(6))
    assert torch.allclose(central_slice[:, 0, 1], torch.fft.fftshift(torch.fft.fftfreq(6)))
    assert torch.allclose(central_slice[:, 0, 2], torch.zeros(6))

    # check w dim
    assert torch.allclose(central_slice[3, :, 0], torch.zeros(4))
    assert torch.allclose(central_slice[3, :, 1], torch.zeros(4))
    assert torch.allclose(central_slice[3, :, 2], torch.fft.rfftfreq(6))
