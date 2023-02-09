import torch

from libtilt.rotational_average import _split_into_shells_2d, \
    rotational_average_2d, rotational_average_3d


def test_split_into_shells_2d():
    # real space
    image = torch.rand(size=(4, 28, 28))  # (b, h, w)
    shells = _split_into_shells_2d(image, n_shells=14)
    assert isinstance(shells, list)
    assert len(shells) == 14
    assert isinstance(shells[0], torch.Tensor)
    assert torch.allclose(shells[0], image[:, 14, 14].reshape((4, 1)))

    # rfft, not fftshifted
    image = torch.rand(size=(4, 28, 15))
    shells = _split_into_shells_2d(image, n_shells=14, rfft=True, fftshifted=False)
    assert isinstance(shells[0], torch.Tensor)
    assert torch.allclose(shells[0], image[:, 0, 0].reshape((4, 1)))

    # rfft, fftshifted
    image = torch.rand(size=(4, 28, 15))
    shells = _split_into_shells_2d(image, n_shells=14, rfft=True, fftshifted=True)
    assert torch.allclose(shells[0], image[:, 14, 0].reshape((4, 1)))


def test_rotational_average_2d():
    # real space, no batching
    image = torch.rand(size=(28, 28))
    rotational_average = rotational_average_2d(image)
    assert rotational_average.shape == (14, )

    # real space, with batching
    image = torch.rand(size=(4, 28, 28))
    rotational_average = rotational_average_2d(image)
    assert rotational_average.shape == (4, 14)

    # unshifted rfft
    image = torch.rand(size=(4, 28, 15))
    rotational_average = rotational_average_2d(image, rfft=True, fftshifted=False)
    assert rotational_average.shape == (4, 14)

    # fftshifted rfft
    image = torch.rand(size=(4, 28, 15))
    rotational_average = rotational_average_2d(image, rfft=True, fftshifted=True)
    assert rotational_average.shape == (4, 14)


def test_rotational_average_3d():
    # real space, no batching
    image = torch.rand(size=(10, 10, 10))
    rotational_average = rotational_average_3d(image)
    assert rotational_average.shape == (5, )

    # real space, with batching
    image = torch.rand(size=(2, 10, 10, 10))
    rotational_average = rotational_average_3d(image)
    assert rotational_average.shape == (2, 5)

    # unshifted rfft
    image = torch.rand(size=(2, 10, 10, 10))
    rotational_average = rotational_average_3d(image, rfft=True, fftshifted=False)
    assert rotational_average.shape == (2, 5)

    # fftshifted rfft
    image = torch.rand(size=(2, 10, 10, 10))
    rotational_average = rotational_average_3d(image, rfft=True, fftshifted=True)
    assert rotational_average.shape == (2, 5)