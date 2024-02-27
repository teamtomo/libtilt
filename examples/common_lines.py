import mrcfile
import torch
import einops
from libtilt.projection import project_real_2d
from libtilt.rescaling.rescale_fourier import rescale_2d
from itertools import combinations

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

# need to project perpendicular to the tilt axis, so add 90 degrees
projection_angle = torch.deg2rad(torch.tensor([TILT_AXIS_ANGLE_PRIOR + 90])).requires_grad_(True)

common_lines_optimiser = torch.optim.SGD(
    params=[projection_angle, ],
    lr=0.01,
)

# add a mask for the common area between tilts
projection_size = max(tilt_series.shape)
grid = torch.abs(torch.arange(projection_size) - projection_size // 2)
mask = (grid < projection_size / 4) * 1

for _ in range(10):
    common_lines_optimiser.zero_grad()  # reset gradients

    # calculate the projection matrix for the current tilt angle
    cosa, sina = torch.cos(projection_angle), torch.sin(projection_angle)
    R = torch.tensor([[cosa, -sina], [sina, cosa]]).reshape(1, 2, 2)

    projections = []
    for i in range(len(tilt_series)):
        p = project_real_2d(tilt_series[i], R).squeeze()
        projections.append((p - p.mean()) / p.std())

    loss = torch.tensor([.0], requires_grad=True)
    for x, y in combinations(projections, 2):
        loss = loss - (x * y * mask).sum() / mask.sum()
    loss.backward()
    common_lines_optimiser.step()
    print(projection_angle)
    print(loss.item())


