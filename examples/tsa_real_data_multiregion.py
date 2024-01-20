import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
import einops

from libtilt.backprojection import backproject_fourier
from libtilt.coordinate_utils import homogenise_coordinates
from libtilt.fft_utils import dft_center
from libtilt.patch_extraction import extract_patches_2d
from libtilt.projection import project_fourier
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d
from libtilt.transformations import Ry, Rz, T

IMAGE_FILE = 'data/TS_01.mrc'
IMAGE_PIXEL_SIZE = 1.35
STAGE_TILT_ANGLE_PRIORS = torch.arange(-57, 63, 3)
TILT_AXIS_ANGLE_PRIOR = 85
ALIGNMENT_PIXEL_SIZE = 40

tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

tilt_series, _ = rescale_2d(
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
    maintain_center=True,
)

tilt_series -= einops.reduce(tilt_series, 'tilt h w -> tilt 1 1', reduction='mean')
tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
n_tilts, h, w = tilt_series.shape
center = dft_center((h, w), rfft=False, fftshifted=True)
center = einops.repeat(center, 'yx -> b yx', b=len(tilt_series))
tilt_series = extract_patches_2d(
    image=tilt_series,
    positions=center,
    sidelength=min(h, w),
)

s = 64
mask = circle(
    radius=s // 3,
    smoothing_radius=s // 6,
    image_shape=(s, s),
)

predicted_shifts = torch.zeros(
    size=(len(tilt_series), 2), dtype=torch.float32, requires_grad=True
)

projection_model_optimiser = torch.optim.Adam(
    params=[predicted_shifts, ],
    lr=0.1,
)

# optimise
for i in range(250):
    # Make multiple intermediate reconstructions from 50% of the dat
    with torch.no_grad():
        tilt_mask = torch.rand((len(tilt_series))) < 0.50
        tomogram_sidelength = min(h, w)
        positions_2d = np.linspace(start=(s // 2, s // 2),
                                   stop=(h - s // 2, w - s // 2), num=8)
        positions_3d = F.pad(torch.tensor(positions_2d), pad=(1, 0),
                             value=min(h, w) // 2)
        positions_homogenous = homogenise_coordinates(positions_3d).float()

        tomogram_dimensions = (
        tomogram_sidelength, tomogram_sidelength, tomogram_sidelength)
        tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
        tilt_image_center = dft_center((h, w), rfft=False, fftshifted=True)

        s0 = T(-tomogram_center)
        r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
        r1 = Rz(TILT_AXIS_ANGLE_PRIOR, zyx=True)
        # s1 = T(F.pad(predicted_shifts, pad=(1, 0), value=0))
        s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
        M = s2 @ r1 @ r0 @ s0
        Mproj = M[:, 1:3, :]

        positions_homogenous = einops.rearrange(positions_homogenous,
                                                'b zyxw -> b 1 zyxw')
        projected_yx = Mproj @ positions_homogenous.view((-1, 1, 4, 1))
        projected_yx = projected_yx.view((8, -1, 2))

        local_ts = extract_patches_2d(
            image=tilt_series,
            positions=projected_yx,
            sidelength=s
        )

        local_reconstructions = []
        for ts in local_ts:
            local_reconstruction = backproject_fourier(
                images=ts[tilt_mask],
                rotation_matrices=torch.linalg.inv(M[:, :3, :3][tilt_mask]),
                rotation_matrix_zyx=True,
            )
            local_reconstructions.append(local_reconstruction)

    for j in range(8):
        projections = project_fourier(
            volume=local_reconstructions[j],
            rotation_matrices=torch.linalg.inv(M[:, :3, :3][~tilt_mask]),
            rotation_matrix_zyx=True
        )
        projections = projections - torch.mean(projections, dim=(-2, -1), keepdim=True)
        projections = projections / torch.std(projections, dim=(-2, -1), keepdim=True)
        projections = shift_2d(projections, shifts=predicted_shifts[~tilt_mask])
        projections = projections * mask
        projection_model_optimiser.zero_grad()
        experimental = local_ts[j][~tilt_mask] * mask
        loss = torch.mean((experimental - projections) ** 2).sqrt()
        loss.backward()
        print(i, loss.item())
        print(predicted_shifts)

# final reconstruction
centered_tilt_series = shift_2d(tilt_series, shifts=-predicted_shifts)
final_reconstruction = backproject_fourier(
    images=centered_tilt_series,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
)

import napari

viewer = napari.Viewer()
viewer.add_image(tilt_series.detach().numpy(), name='experimental')
viewer.add_image(centered_tilt_series.detach().numpy(), name='aligned')
viewer.add_image(final_reconstruction.detach().numpy(), name='final reconstruction')
napari.run()
