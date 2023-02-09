from typing import Tuple

import einops
import numpy as np
import scipy.constants as C
import torch

from libtilt.ctf.relativistic_wavelength import calculate_relativistic_electron_wavelength
from ..utils.fft import construct_fftfreq_grid_2d


def calculate_ctf(
        defocus: float,
        astigmatism: float,
        astigmatism_angle: float,
        voltage: float,
        spherical_aberration: float,
        amplitude_contrast: float,
        b_factor: float,
        phase_shift: float,
        pixel_size: float,
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
    image_shape: Tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft: bool
        Generate the CTF generated containing only the non-redundant half transform from a rfft.
        Only one of `rfft` and `fftshift` may be `True`.
    fftshift: bool
        Whether to apply fftshift on the resulting CTF images.
    """
    # to torch.Tensor and unit conversions
    defocus = torch.atleast_1d(torch.as_tensor(defocus, dtype=torch.float))
    defocus *= 1e4  # micrometers -> angstroms
    astigmatism = torch.atleast_1d(torch.as_tensor(astigmatism, dtype=torch.float))
    astigmatism *= 1e4  # micrometers -> angstroms
    astigmatism_angle = torch.atleast_1d(torch.as_tensor(astigmatism_angle, dtype=torch.float))
    astigmatism_angle *= (C.pi / 180)  # degrees -> radians
    pixel_size = torch.atleast_1d(torch.as_tensor(pixel_size))
    voltage = torch.atleast_1d(torch.as_tensor(voltage, dtype=torch.float))
    voltage *= 1e3  # kV -> V
    spherical_aberration = torch.atleast_1d(
        torch.as_tensor(spherical_aberration, dtype=torch.float)
    )
    spherical_aberration *= 1e7  # mm -> angstroms
    image_shape = torch.as_tensor(image_shape)

    # derived quantities used in CTF calculation
    defocus_u = defocus + astigmatism
    defocus_v = defocus - astigmatism
    _lambda = calculate_relativistic_electron_wavelength(voltage) * 1e10  # meters -> angstroms
    k1 = -C.pi * _lambda
    k2 = C.pi / 2 * spherical_aberration * _lambda ** 3
    k3 = torch.tensor(np.deg2rad(phase_shift))
    k4 = -b_factor / 4
    k5 = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2))

    # construct 2D frequency grids and rescale cycles / px -> cycles / Å
    fftfreq_grid = construct_fftfreq_grid_2d(image_shape=image_shape, rfft=rfft)  # (h, w, 2)
    fftfreq_grid = fftfreq_grid / einops.rearrange(pixel_size, 'b -> b 1 1 1')
    fftfreq_grid_squared = fftfreq_grid ** 2

    # Astigmatism
    #         Q = [[ sin, cos]
    #              [-sin, cos]]
    #         D = [[   u,   0]
    #              [   0,   v]]
    #         A = Q^T.D.Q = [[ Axx, Axy]
    #                        [ Ayx, Ayy]]
    #         Axx = cos^2 * u + sin^2 * v
    #         Ayy = sin^2 * u + cos^2 * v
    #         Axy = Ayx = cos * sin * (u - v)
    #         defocus = A.k.k^2 = Axx*x^2 + 2*Axy*x*y + Ayy*y^2

    c = torch.cos(astigmatism_angle)
    c2 = c ** 2
    s = torch.sin(astigmatism_angle)
    s2 = s ** 2

    yy2, xx2 = einops.rearrange(fftfreq_grid_squared, 'b h w freq -> freq b h w')
    xy = einops.reduce(fftfreq_grid, 'b h w freq -> b h w', reduction='prod')
    n4 = einops.reduce(fftfreq_grid_squared, 'b h w freq -> b h w', reduction='sum') ** 2

    Axx = c2 * defocus_u + s2 * defocus_v
    Axx_x2 = einops.rearrange(Axx, 'b -> b 1 1') * xx2
    Axy = c * s * (defocus_u - defocus_v)
    Axy_xy = einops.rearrange(Axy, 'b -> b 1 1') * xy
    Ayy = s2 * defocus_u + c2 * defocus_v
    Ayy_y2 = einops.rearrange(Ayy, 'b -> b 1 1') * yy2

    # calculate ctf
    ctf = -torch.sin(k1 * (Axx_x2 + (2 * Axy_xy) + Ayy_y2) + k2 * n4 - k3 - k5)
    if k4 > 0:
        ctf *= torch.exp(k4 * n4)
    if fftshift is True:
        ctf = torch.fft.fftshift(ctf, dim=(-2, -1))
    return ctf
