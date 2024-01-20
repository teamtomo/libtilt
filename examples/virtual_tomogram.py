from functools import lru_cache
from typing import Any

import mrcfile
import napari
import starfile
import torch
import torch.nn.functional as F
import einops
from pydantic import BaseModel, validator

from libtilt.transformations import Rx, Ry, Rz, T, S
from libtilt.coordinate_utils import homogenise_coordinates
from libtilt.patch_extraction.patch_extraction_2d_subpixel import extract_patches_2d
from libtilt.rescaling import rescale_2d
from libtilt.backprojection import backproject_fourier
from libtilt.fft_utils import dft_center

TILT_SERIES_FILE = 'data/TS_01.mrc'
METADATA_FILE = 'data/TS_01.star'
TOMOGRAM_DIMENSIONS = (2000, 4000, 4000)
TILT_IMAGE_DIMENSIONS = (3838, 3710)
PIXEL_SIZE = 1.35
TARGET_PIXEL_SIZE = 20
EXTRACTION_SIDELENGTH = 256


class VirtualTomogram(BaseModel):
    tilt_series: torch.Tensor  # (tilt, h, w)
    tilt_series_pixel_size: float
    tomogram_dimensions: tuple[int, int, int]  # (d, h, w) at tilt-series pixel size
    euler_angles: torch.Tensor  # (tilt, 3) extrinsic XYZ rotation of tomogram
    shifts: torch.Tensor  # (tilt, dh, dw) in pixels
    target_pixel_size: float

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)

    def __hash__(self):  # weird, seems necessary for use of lru cache on method
        return id(self)

    @property
    def tomogram_center(self) -> torch.Tensor:
        return torch.as_tensor(TOMOGRAM_DIMENSIONS, dtype=torch.float32) // 2

    @property
    def tilt_image_center(self) -> torch.Tensor:  # (3, ) 0, center_h, center_w
        center = dft_center(
            image_shape=self.tilt_series.shape[-2:],
            rfft=False,
            fftshifted=True
        )
        return F.pad(center, (1, 0), value=0)

    @property
    def scale_factor(self) -> float:
        return self.tilt_series_pixel_size / self.target_pixel_size

    @property
    def rotation_matrices(self) -> torch.Tensor:
        r0 = Rx(eulers_xyz[:, 0], zyx=True)
        r1 = Ry(eulers_xyz[:, 1], zyx=True)
        r2 = Rz(eulers_xyz[:, 2], zyx=True)
        return (r2 @ r1 @ r0)[:, :3, :3]

    @property
    def transformation_matrices(self) -> torch.Tensor:
        t0 = T(-self.tomogram_center)
        s0 = S([self.scale_factor, self.scale_factor, self.scale_factor])
        r0 = Rx(eulers_xyz[:, 0], zyx=True)
        r1 = Ry(eulers_xyz[:, 1], zyx=True)
        r2 = Rz(eulers_xyz[:, 2], zyx=True)
        t1 = T(self.shifts_3d * self.scale_factor)
        t2 = T(self.tilt_image_center * self.scale_factor)
        return t2 @ t1 @ r2 @ r1 @ r0 @ s0 @ t0  # (tilt, 4, 4)

    @property
    def projection_matrices(self) -> torch.Tensor:
        return self.transformation_matrices[:, 1:3, :]  # (tilt, 2, 4)

    @property
    def shifts_3d(self) -> torch.Tensor:
        return F.pad(self.shifts, (1, 0), value=0)

    @lru_cache(maxsize=1)
    def rescale_tilt_series(self, target_pixel_size: float) -> torch.Tensor:
        tilt_series, spacing_rescaled = rescale_2d(
            self.tilt_series,
            source_spacing=self.tilt_series_pixel_size,
            target_spacing=target_pixel_size,
            maintain_center=True,
        )
        return tilt_series

    def calculate_projected_positions(
        self, particle_position: torch.Tensor
    ) -> torch.Tensor:
        particle_position = homogenise_coordinates(particle_position)
        particle_position = einops.rearrange(particle_position, 'zyxw -> zyxw 1')
        positions_2d = self.projection_matrices @ particle_position  # (tilt, yx, 1)
        positions_2d = einops.rearrange(positions_2d, 'tilt yx 1 -> tilt yx')
        return positions_2d

    def extract_local_tilt_series(
        self, position_in_tomogram: torch.Tensor, sidelength: int
    ) -> torch.Tensor:  # (tilt, sidelength, sidelength)
        rescaled_tilt_series = self.rescale_tilt_series(
            self, target_pixel_size=self.target_pixel_size
        )
        projected_positions = self.calculate_projected_positions(position_in_tomogram)
        particle_tilt_series = extract_patches_2d(
            images=rescaled_tilt_series,
            positions=projected_positions,
            sidelength=sidelength,
        )
        return particle_tilt_series

    def reconstruct_local_volume(
        self, position_in_tomogram: torch.Tensor, sidelength: int
    ) -> torch.Tensor:  # (sidelength, sidelength, sidelength)
        local_tilt_series = self.extract_local_tilt_series(
            position_in_tomogram=position_in_tomogram, sidelength=2 * sidelength
        )
        local_reconstruction = backproject_fourier(
            images=local_tilt_series,
            rotation_matrices=torch.linalg.inv(self.rotation_matrices),
            rotation_matrix_zyx=True,
        )
        low, high = sidelength // 2, (sidelength // 2) + sidelength
        return local_reconstruction[low:high, low:high, low:high]

    @validator('tilt_series', 'euler_angles', 'shifts', pre=True)
    def to_float32_tensor(cls, value):
        return torch.as_tensor(value).float()


if __name__ == '__main__':
    # load image and read tilt-series alignment metadata
    tilt_series = torch.as_tensor(mrcfile.read(TILT_SERIES_FILE))
    df = starfile.read(METADATA_FILE)
    shifts_px = torch.as_tensor(
        df[['rlnTomoYShiftAngst', 'rlnTomoXShiftAngst']].to_numpy(),
        dtype=torch.float32
    ) / PIXEL_SIZE
    eulers_xyz = torch.as_tensor(
        df[['rlnTomoXTilt', 'rlnTomoYTilt', 'rlnTomoZRot']].to_numpy(),
        dtype=torch.float32
    )

    # create the virtual tomogram and define a position of interest
    virtual_tomogram = VirtualTomogram(
        tilt_series=tilt_series,
        tilt_series_pixel_size=PIXEL_SIZE,
        euler_angles=eulers_xyz,
        shifts=shifts_px,
        target_pixel_size=TARGET_PIXEL_SIZE,
        tomogram_dimensions=TOMOGRAM_DIMENSIONS
    )
    particle_position = virtual_tomogram.tomogram_center

    # extract a local tilt-series
    local_tilt_series = virtual_tomogram.extract_local_tilt_series(
        position_in_tomogram=particle_position,
        sidelength=EXTRACTION_SIDELENGTH
    )

    # make a local reconstruction
    local_reconstruction = virtual_tomogram.reconstruct_local_volume(
        position_in_tomogram=particle_position,
        sidelength=EXTRACTION_SIDELENGTH,
    )

    # visualise results
    viewer = napari.Viewer()
    viewer.add_image(local_tilt_series.numpy())
    viewer.add_image(local_reconstruction.numpy(), rendering='minip')
    viewer.dims.set_point(axis=0, value=len(local_reconstruction) // 2)
    napari.run()
