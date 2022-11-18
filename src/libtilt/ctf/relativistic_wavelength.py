import math

from scipy import constants as C


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
