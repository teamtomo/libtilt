import math
from typing import Tuple

import einops
import numpy as np
import scipy.constants as C
import torch
from pydantic import BaseModel

from libtilt.utils.fft import construct_fftfreq_grid_2d


class MicroscopeParametersSI(BaseModel):
    voltage: float
    spherical_aberration: float
    amplitude_contrast: float

    @classmethod
    def from_standard_units(
            cls,
            voltage: float,
            spherical_aberration: float,
            amplitude_contrast: float
    ):
        voltage *= 1e3
        spherical_aberration *= 1e7
        return cls(
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast
        )


def calculate_relativistic_electron_wavelength(energy: float):
    """Calculate the relativistic electron wavelength in SI units.

    For derivation see:
    1.  Kirkland, E. J. Advanced Computing in Electron Microscopy.
        (Springer International Publishing, 2020). doi:10.1007/978-3-030-33260-0.

    2.  https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory

    Parameters
    ----------
    energy: float
        acceleration potential in volts.

    Returns
    -------
    wavelength: float
        relativistic wavelength of the electron in meters.
    """
    h = C.Planck
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = energy
    eV = e * V

    numerator = h * c
    denominator = math.sqrt(eV * (2 * m0 * c ** 2 + eV))
    return numerator / denominator


def ctf2d(
        defocus: float,
        astigmatism: float,
        astigmatism_angle: float,
        voltage: float,
        spherical_aberration: float,
        amplitude_contrast: float,
        b_factor: float,
        phase_shift: float,
        image_shape: Tuple[int, int],
        rfft: bool,
        fftshift: bool,
):
    """

    Parameters
    ----------
    defocus: float
        Defocus in micrometers, positive is underfocused.
        `(defocus_u + defocus_v) / 2`
    astigmatism: float
        Amount of astigmatism in micrometers.
        `(defocus_u - defocus_v) / 2`
    astigmatism_angle: float
        Angle of astigmatism in degrees. 0 places `defocus_u` along the y-axis.
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
    image_shape: Tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft: bool
        Generate the CTF generated containing only the non-redundant half transform from a rfft.
        Only one of `rfft` and `fftshift` may be `True`.
    fftshift: bool
        Whether to apply fftshift on the resulting CTF images.

    Returns
    -------

    """
    # to torch.Tensor and unit conversions
    defocus = torch.atleast_1d(torch.as_tensor(defocus, dtype=torch.float))
    defocus *= 1e5  # micrometers -> angstroms
    astigmatism = torch.atleast_1d(torch.as_tensor(astigmatism, dtype=torch.float))
    astigmatism *= 1e5  # micrometers -> angstroms
    astigmatism_angle = torch.atleast_1d(torch.as_tensor(astigmatism_angle, dtype=torch.float))
    astigmatism_angle *= (C.pi / 180)  # degrees -> radians
    voltage = torch.atleast_1d(torch.as_tensor(voltage, dtype=torch.float))
    voltage *= 1e3  # kV -> V
    spherical_aberration = torch.atleast_1d(torch.as_tensor(spherical_aberration, dtype=torch.float))
    spherical_aberration *= 1e7  # mm -> angstroms

    # derived quantities used in CTF calculation
    defocus_u = defocus + astigmatism
    defocus_v = defocus - astigmatism
    _lambda = calculate_relativistic_electron_wavelength(voltage) * 1e10  # meters -> angstroms
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda ** 3
    k3 = torch.tensor(np.deg2rad(phase_shift))
    k4 = -b_factor / 4
    k5 = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2))

    # construct 2D frequency grids
    fftfreq_grid = construct_fftfreq_grid_2d(image_shape=image_shape, rfft=rfft)
    yy, xx = einops.rearrange(fftfreq_grid ** 2, 'h w freq -> freq h w')
    xy = einops.reduce(fftfreq_grid, 'h w freq -> h w', reduction='prod')
    n4 = einops.reduce(fftfreq_grid**2, 'h w freq -> h w', reduction='sum') ** 2

    c = torch.cos(astigmatism_angle).view(1)
    s = torch.sin(astigmatism_angle).view(1)

    xx_factor = torch.tensor(c ** 2 * defocus_u + s ** 2 * defocus_v)
    xx_ = einops.rearrange(xx_factor, 'f -> f 1 1') * xx
    yy_factor = torch.tensor(s ** 2 * defocus_u + c ** 2 * defocus_v)
    yy_ = einops.rearrange(yy_factor, 'f -> f 1 1') * yy
    xy_factor = torch.tensor(c * s * (defocus_u - defocus_v))
    xy_ = einops.rearrange(xy_factor, 'f -> f 1 1') * xy

    ctf = -torch.sin(k1 * (xx_ + (2 * xy_) + yy_) + k2 * n4 - k3 - k5)
    if k4 > 0:
        ctf *= torch.exp(k4 * n4)
    if fftshift is True:
        ctf = torch.fft.fftshift(ctf, dim=(-2, -1))
    return ctf


if __name__ == '__main__':
    ctf = ctf2d(
        defocus=2,
        astigmatism=0,
        astigmatism_angle=0,
        image_shape=(256, 256),
        rfft=False,
        fftshift=True,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        b_factor=-10,
        phase_shift=0
    )
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(ctf.squeeze().numpy())
    plt.show()
