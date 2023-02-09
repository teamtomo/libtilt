import os
from pathlib import Path

import einops
import mmdf
import mrcfile
import numpy as np
import typer

cli = typer.Typer(add_completion=False)


def model_to_xyz(model_file: os.PathLike) -> np.ndarray:
    df = mmdf.read(model_file)
    return df[['x', 'y', 'z']].to_numpy()


def rasterise_xyz_on_cube(
    xyz: np.ndarray, sidelength_factor: float = 2
) -> np.ndarray:
    zyx = xyz[:, ::-1]
    _min, _max = np.min(zyx, axis=0), np.max(zyx, axis=0)
    _mean = np.mean(zyx, axis=0)
    midpoint = _mean.astype(int)
    delta = _max - _min
    minimum_sidelength = np.max(delta)
    sidelength = minimum_sidelength * sidelength_factor

    low = midpoint - (sidelength // 2)
    high = midpoint + (sidelength // 2)
    bins = np.linspace(low, high, num=int(sidelength), endpoint=True)
    bz, by, bx = einops.rearrange(bins, 'b zyx -> zyx b')
    image, _ = np.histogramdd(zyx, bins=[bz, by, bx])
    return image


def pdb2mrc(model_file: Path, output_mrc_file: Path, voxel_spacing: float = 10):
    xyz = model_to_xyz(str(model_file))
    xyz /= voxel_spacing
    volume = rasterise_xyz_on_cube(xyz, sidelength_factor=2).astype(np.float32)
    mrcfile.write(
        output_mrc_file, volume, voxel_size=voxel_spacing, overwrite=True
    )
