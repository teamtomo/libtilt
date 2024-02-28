import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
import einops
from itertools import combinations
from torch_cubic_spline_grids import CubicBSplineGrid1d

from libtilt.backprojection import backproject_fourier
from libtilt.fft_utils import dft_center
from libtilt.patch_extraction import extract_squares
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d
from libtilt.transformations import Ry, Rz, T
from libtilt.correlation import correlate_2d
from libtilt.projection import project_real_2d


def rotation_matrix(angles_degrees):
    """Calculate rotation matrices for images."""
    angles_radians = torch.deg2rad(angles_degrees)
    n = angles_radians.shape[0]
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(2), 'i j -> n i j', n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    return matrices


IMAGE_FILE = 'data/tomo200528_100.st'
IMAGE_PIXEL_SIZE = 1.724
STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 51, 3)
TILT_AXIS_ANGLE_PRIOR = -30  # -88.7 according to mdoc, but I set it faulty to see if the optimization works
ALIGNMENT_PIXEL_SIZE = 13.79 * 2
# set 0 degree tilt as reference
REFERENCE_TILT = STAGE_TILT_ANGLE_PRIORS.abs().argmin()

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
tilt_series = extract_squares(
    image=tilt_series,
    positions=center,
    sidelength=min(h, w),
)

# set tomogram and tilt-series shape
size = min(h, w)
tomogram_dimensions = (size, ) * 3
tilt_dimensions = (size, ) * 2

# mask for coarse alignment
coarse_alignment_mask = circle(
    radius=size // 3,
    smoothing_radius=size // 6,
    image_shape=tilt_dimensions,
)

# do an IMOD style coarse tilt-series alignment
reference_shift = torch.tensor([.0, .0])
center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)
coarse_shifts = torch.zeros((len(tilt_series), 2), dtype=torch.float32)

# find coarse alignment for negative tilts
current_shift = reference_shift.clone()
for i in range(REFERENCE_TILT, 0, -1):
    correlation = correlate_2d(
        tilt_series[i] * coarse_alignment_mask,
        tilt_series[i - 1] * coarse_alignment_mask,
        normalize=True
    )
    shift = center - torch.tensor(
        np.unravel_index(correlation.argmax(), shape=tilt_dimensions)
    )
    current_shift += shift
    coarse_shifts[i - 1] = current_shift

# find coarse alignment positive tilts
current_shift = reference_shift.clone()
for i in range(REFERENCE_TILT, tilt_series.shape[0] - 1, 1):
    correlation = correlate_2d(
        tilt_series[i] * coarse_alignment_mask,
        tilt_series[i + 1] * coarse_alignment_mask,
        normalize=True
    )
    shift = center - torch.tensor(
        np.unravel_index(correlation.argmax(), shape=tilt_dimensions)
    )
    current_shift += shift
    coarse_shifts[i + 1] = current_shift

# apply the shifts for coarse aligned series
coarse_aligned_tilt_series = shift_2d(tilt_series, shifts=-coarse_shifts)

tilt_axis_grid = CubicBSplineGrid1d(resolution=1, n_channels=1)
tilt_axis_grid.data = torch.tensor([TILT_AXIS_ANGLE_PRIOR, ], dtype=torch.float32)
interpolation_points = torch.linspace(0, 1, len(tilt_series))

common_lines_optimiser = torch.optim.Adam(
    tilt_axis_grid.parameters(),
    lr=1,
)

print('initial tilt axis:', tilt_axis_grid.data)
for epoch in range(200):
    # interpolate the grid
    tilt_axis_angles = tilt_axis_grid(interpolation_points)

    # for common lines each 2d image is projected perpendicular to the tilt axis, thus add 90 degrees
    R = rotation_matrix(tilt_axis_angles + 90)
    projections = []
    for i in range(len(tilt_series)):
        p = project_real_2d(tilt_series[i] * coarse_alignment_mask, R[i:i+1])
        projections.append((p - p.mean()) / p.std())

    common_lines_optimiser.zero_grad()
    loss = 0
    for x, y in combinations(projections, 2):
        loss = loss - (x * y).sum() / y.numel()
    loss.backward()
    common_lines_optimiser.step()
    if not (epoch % 10):
        print(tilt_axis_grid.data)
        print(epoch, loss.item())
print('final tilt axis angle:', tilt_axis_grid.data)

# TODO After this should calculate stretching to determine ideal tilt-angle offset

tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
tilt_image_center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(TILT_AXIS_ANGLE_PRIOR, zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

# coarse reconstruction
shifts_only_reconstruction = backproject_fourier(
    images=coarse_aligned_tilt_series,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
)

s0 = T(-tomogram_center)
r0 = Ry(STAGE_TILT_ANGLE_PRIORS, zyx=True)
r1 = Rz(tilt_axis_grid(interpolation_points), zyx=True)
s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
M = s2 @ r1 @ r0 @ s0

# coarse reconstruction
coarse_reconstruction = backproject_fourier(
    images=coarse_aligned_tilt_series,
    rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
    rotation_matrix_zyx=True,
)

import napari

viewer = napari.Viewer()
viewer.add_image(tilt_series.detach().numpy(), name='experimental')
viewer.add_image(coarse_aligned_tilt_series.detach().numpy(), name='coarse aligned')
viewer.add_image(shifts_only_reconstruction.detach().numpy(), name='shifts only reconstruction')
viewer.add_image(coarse_reconstruction.detach().numpy(), name='coarse reconstruction')
napari.run()
