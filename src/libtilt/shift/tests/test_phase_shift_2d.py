import torch

from libtilt.pytest_utils import device_test
from libtilt.shift.phase_shift_dft import get_phase_shifts_2d
from libtilt.shift.shift_image import shift_2d


@device_test
def test_get_phase_shifts_2d_full_fft():
    shifts = torch.zeros(size=(1, 2))
    phase_shifts = get_phase_shifts_2d(shifts, image_shape=(2, 2), rfft=False)
    assert torch.allclose(phase_shifts, torch.ones(size=(2, 2), dtype=torch.complex64))

    shifts = torch.tensor([[1, 2]])
    phase_shifts = get_phase_shifts_2d(shifts, image_shape=(2, 2), rfft=False)
    expected = torch.tensor(
        [[[1 + 0.0000e00j, 1 + 1.7485e-07j], [-1 - 8.7423e-08j, -1 - 2.3850e-08j]]]
    )
    assert torch.allclose(phase_shifts, expected)


@device_test
def test_get_phase_shifts_2d_rfft():
    shifts = torch.zeros(size=(1, 2))
    phase_shifts = get_phase_shifts_2d(shifts, image_shape=(2, 2), rfft=True)
    assert phase_shifts.shape == (1, 2, 2)
    expected = torch.ones(size=(2, 2), dtype=torch.complex64)
    assert torch.allclose(phase_shifts, expected)

    shifts = torch.tensor([[1, 2]])
    phase_shifts = get_phase_shifts_2d(shifts, image_shape=(2, 2), rfft=False)
    expected = torch.tensor(
        [[[1 + 0.0000e00j, 1 + 1.7485e-07j], [-1 - 8.7423e-08j, -1 - 2.3850e-08j]]]
    )
    assert torch.allclose(phase_shifts, expected)


@device_test
def test_phase_shift_images_2d():
    image = torch.zeros((4, 4))
    image[2, 2] = 1

    # +1px in each dimension
    shifts = torch.ones((1, 2))
    shifted = shift_2d(image, shifts)
    expected = torch.zeros((4, 4))
    expected[3, 3] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # +1px in y
    shifts = torch.zeros((1, 2))
    shifts[0, 0] = 1
    shifted = shift_2d(image, shifts)
    expected = torch.zeros((4, 4))
    expected[3, 2] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # +1px in x
    shifts = torch.zeros((1, 2))
    shifts[0, 1] = 1
    shifted = shift_2d(image, shifts)
    expected = torch.zeros((4, 4))
    expected[2, 3] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)
