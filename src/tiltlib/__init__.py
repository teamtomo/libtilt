"""cryo-electron tomography image processing in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tiltlib")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"
