import mrcfile
import pandas as pd
import torch
import einops

from libtilt.backprojection import backproject_fourier
from libtilt.fft_utils import dft_center
from libtilt.patch_extraction import extract_patches_2d
from libtilt.projection import project_fourier
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d
from libtilt.transformations import Ry, Rz

IMAGE_FILE = 'data/TS_01.mrc'
FID_CENTERS_FILE = 'data/TS_01_fid_position.csv'
IMAGE_PIXEL_SIZE = 1.35
STAGE_TILT_ANGLE_PRIORS = torch.arange(-57, 63, 3)
TILT_AXIS_ANGLE_PRIOR = 85
ALIGNMENT_PIXEL_SIZE = 10

tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))
df = pd.read_csv(FID_CENTERS_FILE)
fiducial_tilt_series = extract_patches_2d(
    image=tilt_series,
    positions=torch.tensor(df[['axis-1', 'axis-2']].to_numpy()).float(),
    sidelength=256
)



fiducial_tilt_series, _ = rescale_2d(
    image=fiducial_tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
    maintain_center=True,
)

fiducial_tilt_series -= einops.reduce(fiducial_tilt_series, 'tilt h w -> tilt 1 1', reduction='mean')
fiducial_tilt_series /= torch.std(fiducial_tilt_series, dim=(-2, -1), keepdim=True)


n_tilts, h, w = fiducial_tilt_series.shape
center = dft_center((h, w), rfft=False, fftshifted=True)
center = einops.repeat(center, 'yx -> b yx', b=len(tilt_series))

mask = circle(
    radius=min(h, w) / 3,
    smoothing_radius=min(h, w) / 6,
    image_shape=(min(h, w), min(h, w)),
)

r0 = Ry(torch.linspace(-57, 60, steps=40), zyx=True)[:, :3, :3].float()
r1 = Rz(85, zyx=True)[:, :3, :3].float()
rotation_matrices = torch.linalg.inv(r1 @ r0)


predicted_shifts = torch.zeros(
    size=(len(tilt_series), 2), dtype=torch.float32, requires_grad=True
)

projection_model_optimiser = torch.optim.Adam(
    params=[predicted_shifts, ],
    lr=0.1,
)

# optimise
for i in range(250):
    # Make an intermediate reconstruction from 90% of the data
    with torch.no_grad():
        tilt_mask = torch.rand((len(tilt_series))) < 0.50
        _tilt_series = shift_2d(
            images=fiducial_tilt_series[tilt_mask],
            shifts=-predicted_shifts[tilt_mask]
        )
        intermediate_reconstruction = backproject_fourier(
            images=_tilt_series,
            rotation_matrices=rotation_matrices[tilt_mask],
            rotation_matrix_zyx=True,
        )

    # make projections in remaining 10% of orientations
    projections = project_fourier(
        volume=intermediate_reconstruction,
        rotation_matrices=rotation_matrices[~tilt_mask],
        rotation_matrix_zyx=True,
    )
    projections = projections - torch.mean(projections, dim=(-2, -1), keepdim=True)
    projections = projections / torch.std(projections, dim=(-2, -1), keepdim=True)

    # shift projections so they match 'experimental' data
    projections = shift_2d(projections, shifts=predicted_shifts[~tilt_mask])
    projections = projections * mask

    # zero gradients, calculate loss and backpropagate
    projection_model_optimiser.zero_grad()
    experimental = fiducial_tilt_series[~tilt_mask] * mask
    loss = torch.mean((experimental - projections) ** 2).sqrt()
    loss.backward()
    projection_model_optimiser.step()
    if i % 20 == 0:
        print(i, loss.item())
        # print(predicted_shifts)
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(projections.detach().numpy())
        # viewer.add_image(experimental.detach().numpy())
        # napari.run()

# final reconstruction
centered_tilt_series = shift_2d(fiducial_tilt_series, shifts=-predicted_shifts)
final_reconstruction = backproject_fourier(
    images=centered_tilt_series,
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True,
)

import napari

viewer = napari.Viewer()
viewer.add_image(fiducial_tilt_series.detach().numpy(), name='experimental')
viewer.add_image(centered_tilt_series.detach().numpy(), name='aligned')
viewer.add_image(final_reconstruction.detach().numpy(), name='final reconstruction')
napari.run()
