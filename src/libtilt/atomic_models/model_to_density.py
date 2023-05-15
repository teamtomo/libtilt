import einops
import numpy as np


def rasterise_xyz(
    xyz: np.ndarray, sidelength: float | None = None
) -> np.ndarray:
    zyx = xyz[:, ::-1]
    _min, _max = np.min(zyx, axis=0), np.max(zyx, axis=0)
    _mean = np.mean(zyx, axis=0)
    midpoint = _mean.astype(int)
    delta = _max - _min
    minimum_sidelength = np.max(delta)
    sidelength = minimum_sidelength * sidelength

    low = midpoint - (sidelength // 2)
    high = midpoint + (sidelength // 2)
    bins = np.linspace(low, high, num=int(sidelength), endpoint=True)
    bz, by, bx = einops.rearrange(bins, 'b zyx -> zyx b')
    image, _ = np.histogramdd(zyx, bins=[bz, by, bx])
    return image
