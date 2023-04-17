import torch

from libtilt.transformations import Rx, Ry, Rz, T


def test_rotation_around_x():
    """Rotation of y around x should become z."""
    R = Rx(90)
    v = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)

    R = Rx(90, zyx=True)
    v = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_rotation_around_y():
    """Rotation of z around y should be x"""
    R = Ry(90)
    v = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    expected = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)

    R = Ry(90, zyx=True)
    v = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_rotation_around_z():
    """Rotation of x around z should give y."""
    R = Rz(90)
    v = torch.tensor([1, 0, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)

    R = Rz(90, zyx=True)
    v = torch.tensor([0, 0, 1, 1]).view((4, 1)).float()
    expected = torch.tensor([0, 1, 0, 1]).view((4, 1)).float()
    assert torch.allclose(R @ v, expected, atol=1e-6)


def test_translation():
    """Translations"""
    M = T([1, 1, 1])
    v = torch.tensor([0, 0, 0, 1]).view((4, 1)).float()
    expected = torch.tensor([1, 1, 1, 1]).view((4, 1)).float()
    assert torch.allclose(M @ v, expected, atol=1e-6)
