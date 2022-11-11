"""inpainting for cryo electron microscopy images"""

from importlib.metadata import PackageNotFoundError, version
from .inpainting import inpaint

try:
    __version__ = version("teamtomo-inpainting")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"
