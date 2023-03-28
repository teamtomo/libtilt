import torch

from libtilt.ctf.ctf_1d import calculate_ctf as calculate_ctf_1d


def test_1d_ctf_single():
    result = calculate_ctf_1d(
        defocus=1.5,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        b_factor=0,
        phase_shift=0,
        n_samples=10,
        oversampling_factor=3
    )
    expected = torch.tensor(
        [
            0.1000,
            0.1444,
            0.2755,
            0.4819,
            0.7283,
            0.9385,
            0.9903,
            0.7519,
            0.1801,
            -0.5461
        ]
    )
    assert torch.allclose(result[0], expected, atol=1e-4)


def test_1d_ctf_batch_defocus():
    result = calculate_ctf_1d(
        defocus=[1.5, 2.5],
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        b_factor=0,
        phase_shift=0,
        n_samples=10,
        oversampling_factor=1
    )
    print(result)
    expected = torch.tensor(
        [[0.1000, 0.1444, 0.2755, 0.4819, 0.7283, 0.9385, 0.9903, 0.7519,
          0.1801, -0.5461],
         [0.1000, 0.1738, 0.3880, 0.6970, 0.9617, 0.9237, 0.3503, -0.5734,
          -0.9877, -0.1474]]
    )
    assert torch.allclose(result, expected, atol=1e-4)
