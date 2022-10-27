import torch

from libtilt.utils.transformations import R0, R1, R2


def test_rotation_around_x():
    R = R0(90)
    v = torch.tensor(
        [[0],
         [1],
         [0],
         [1]]
    ).float()
    expected = torch.tensor(
        [[0],
         [0],
         [1],
         [1]]
    ).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_rotation_around_y():
    R = R1(90)
    v = torch.tensor(
        [[0],
         [0],
         [1],
         [1]]
    ).float()
    expected = torch.tensor(
        [[1],
         [0],
         [0],
         [1]]
    ).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_rotation_around_z():
    R = R2(90)
    v = torch.tensor(
        [[1],
         [0],
         [0],
         [1]]
    ).float()
    expected = torch.tensor(
        [[0],
         [1],
         [0],
         [1]]
    ).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)
