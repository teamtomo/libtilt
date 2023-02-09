import torch

from libtilt.transformations import Rx, Ry, Rz


def test_rotation_around_x():
    R = Rx(90)
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
    R = Ry(90)
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
    R = Rz(90)
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
