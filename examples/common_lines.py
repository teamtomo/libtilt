import mrcfile
import torch
import einops
from libtilt.projection import project_real_2d
from libtilt.rescaling.rescale_fourier import rescale_2d
from itertools import combinations
from libtilt.shapes import circle


def rotation_matrix(angles_radians):
    """Calculate rotation matrices for images."""
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
TILT_AXIS_ANGLE_PRIOR = -88.7
ALIGNMENT_PIXEL_SIZE = 13.79 * 2

tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))
tilt_series, _ = rescale_2d(
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
    maintain_center=True,
)
tilt_series -= einops.reduce(tilt_series, 'tilt h w -> tilt 1 1', reduction='mean')
tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)

# add a mask for the common area between tilts
mask = circle(
    radius=tilt_series.shape[-1] // 3,
    smoothing_radius=tilt_series.shape[-1] // 6,
    image_shape=tilt_series.shape[1:],
)

# need to project perpendicular to the tilt axis, so add 90 degrees
projection_angle = torch.deg2rad(torch.tensor([TILT_AXIS_ANGLE_PRIOR + 90])).requires_grad_(True)
R = torch.ones((1, 2, 2), dtype=torch.float32)

common_lines_optimiser = torch.optim.Adam(
    params=[projection_angle, ],
    lr=0.01,
)

print('initial tilt axis: ', torch.rad2deg(projection_angle) - 90)
for epoch in range(50):
    R = rotation_matrix(projection_angle)
    projections = []
    for i in range(len(tilt_series)):
        p = project_real_2d(tilt_series[i] * mask, R).squeeze()
        projections.append((p - p.mean()) / p.std())

    common_lines_optimiser.zero_grad()
    loss = 0
    for x, y in combinations(projections, 2):
        loss = loss - (x * y).sum() / y.numel()
    loss.backward()
    common_lines_optimiser.step()
    print(torch.rad2deg(projection_angle) - 90)
    print(epoch, loss.item())

print('final tilt axis angle: ', torch.rad2deg(projection_angle) - 90)
