from typing import Tuple

import einops
import numpy as np
import scipy.constants as C
import torch

from libtilt.ctf.relativistic_wavelength import \
    calculate_relativistic_electron_wavelength


def calculate_ctf(
    defocus: float,
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    b_factor: float,
    phase_shift: float,
    pixel_size: float,
    n_samples: int,
    oversampling_factor: int,
):
    """

    Parameters
    ----------
    defocus: float
        Defocus in micrometers, positive is underfocused.
    pixel_size: float
        Pixel size in Angströms per pixel (Å px⁻¹).
    voltage: float
        Acceleration voltage in kilovolts (kV).
    spherical_aberration: float
        Spherical aberration in millimeters (mm).
    amplitude_contrast: float
        Fraction of amplitude contrast (value in range [0, 1]).
    b_factor: float
        B-factor in square angstroms.
    phase_shift: float
        Angle of phase shift applied to CTF in degrees.
    n_samples: int
        Number of samples in CTF.
    """
    # to torch.Tensor and unit conversions
    defocus = torch.atleast_1d(torch.as_tensor(defocus, dtype=torch.float))
    defocus *= 1e4  # micrometers -> angstroms
    defocus = einops.rearrange(defocus, '... -> ... 1')
    pixel_size = torch.atleast_1d(torch.as_tensor(pixel_size))
    voltage = torch.atleast_1d(torch.as_tensor(voltage, dtype=torch.float))
    voltage *= 1e3  # kV -> V
    spherical_aberration = torch.atleast_1d(
        torch.as_tensor(spherical_aberration, dtype=torch.float)
    )
    spherical_aberration *= 1e7  # mm -> angstroms

    # derived quantities used in CTF calculation
    _lambda = calculate_relativistic_electron_wavelength(voltage) * 1e10  # meters -> angstroms
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda ** 3
    k3 = torch.tensor(np.deg2rad(phase_shift))
    k4 = -b_factor / 4
    k5 = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2))

    # construct frequency vector and rescale cycles / px -> cycles / Å
    fftfreq_grid = torch.linspace(0, 0.5, steps=n_samples)  # (n_samples, )

    # oversampling...
    if oversampling_factor > 1:
        frequency_delta = 0.5 / (n_samples - 1)
        oversampled_frequency_delta = frequency_delta / oversampling_factor
        oversampled_interval_length = oversampled_frequency_delta * (oversampling_factor - 1)
        per_frequency_deltas = torch.linspace(0, oversampled_interval_length, steps=oversampling_factor)
        per_frequency_deltas -= oversampled_interval_length / 2
        per_frequency_deltas = einops.rearrange(per_frequency_deltas, 'os -> os 1')
        fftfreq_grid = fftfreq_grid + per_frequency_deltas
        defocus = einops.rearrange(defocus, '... -> ... 1')  # oversampling dim

    fftfreq_grid = fftfreq_grid / pixel_size
    fftfreq_grid_squared = fftfreq_grid ** 2
    n4 = fftfreq_grid_squared ** 2

    # calculate ctf
    ctf = -torch.sin(k1 * fftfreq_grid_squared * defocus + k2 * n4 - k3 - k5)
    if k4 > 0:
        ctf *= torch.exp(k4 * n4)

    if oversampling_factor > 1:
        ctf = einops.reduce(ctf, '... os k -> ... k', reduction='mean')
    return ctf
