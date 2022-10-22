import torch

from libtilt.phase_shift_2d import get_phase_shifts_2d, fourier_shift_images_2d


def test_get_phase_shifts_2d():
    shifts = torch.zeros(size=(1, 2))
    phase_shifts = get_phase_shifts_2d(shifts, image_shape=(2, 2))
    assert torch.allclose(phase_shifts, torch.ones(size=(2, 2), dtype=torch.complex64))

    shifts = torch.ones(size=(1, 2))
    phase_shifts = get_phase_shifts_2d(shifts, image_shape=(2, 2))
    expected = torch.tensor([[[1. + 1.7485e-07j, -1. - 8.7423e-08j],
                              [-1. - 8.7423e-08j, 1. + 0.0000e+00j]]])
    assert torch.allclose(phase_shifts, expected)
