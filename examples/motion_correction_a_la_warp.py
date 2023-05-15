from datetime import datetime

import torch
import numpy as np
import mrcfile
import einops

import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d

from libtilt.utils.coordinates import array_to_grid_sample
from libtilt.shift.fourier_shift import phase_shift_dft_2d
from libtilt.shapes.shapes_2d import circle
from libtilt.grids.patch_grid import _patch_indices_2d
from libtilt.grids.patch_grid import _patch_centers_3d

IMAGE_FILE = 'data/TS_01_000_0.0.mrc'
GT_DEFORMATION_FIELD_RESOLUTION = (3, 3, 3)  # (t, h, w)
LEARNED_DEFORMATION_FIELD_RESOLUTION = (5, 5, 5)  # (t, h, w)
PATCH_SIDELENGTH = 64
N_ITERATIONS = 200
N_PATCHES_PER_BATCH = 15
LEARNING_RATE = 0.05

image = torch.tensor(mrcfile.read(IMAGE_FILE)).float()
image = F.avg_pool2d(image, kernel_size=8)
image = einops.reduce(image, 't h w -> h w', reduction='mean')
image = (image - torch.mean(image)) / torch.std(image)

### simulate continuous beam induced motion on the target
multi_frame_micrograph = einops.repeat(image, 'h w -> 10 h w').clone()
t, h, w = multi_frame_micrograph.shape[-3:]

gt_deformation_field_data = np.random.uniform(
    low=-5, high=5, size=(2, *GT_DEFORMATION_FIELD_RESOLUTION)
)
gt_deformation_field = CubicBSplineGrid3d.from_grid_data(
    torch.tensor(gt_deformation_field_data).float()
)
_t, _y, _x = torch.linspace(0, 1, t), torch.linspace(0, 1, h), torch.linspace(0, 1, w)
tt, yy, xx = torch.meshgrid(_t, _y, _x)
tyx = einops.rearrange([tt, yy, xx], 'tyx t h w -> t h w tyx')

predicted_shifts = gt_deformation_field(tyx)
array_coordinates = torch.tensor(np.indices(image.shape[-2:]))
array_coordinates = einops.rearrange(array_coordinates, 'yx ... -> ... yx')

deformed_coordinates = array_coordinates + predicted_shifts
grid_sample_coordinates = array_to_grid_sample(deformed_coordinates, array_shape=(h, w))

multi_frame_micrograph = F.grid_sample(
    input=einops.rearrange(multi_frame_micrograph, 't h w -> t 1 h w'),
    grid=grid_sample_coordinates,
    mode='bicubic',
    padding_mode='zeros',
    align_corners=True,
)

multi_frame_micrograph = einops.rearrange(multi_frame_micrograph, 't 1 h w -> t h w')

### learn the motion
reference = image.clone()

# extract patches...
patch_idx_h, patch_idx_w = _patch_indices_2d(
    image_shape=image.shape,
    patch_shape=(PATCH_SIDELENGTH, PATCH_SIDELENGTH),
    patch_step=(PATCH_SIDELENGTH // 2, PATCH_SIDELENGTH // 2),
    distribute_patches=True,
)
patch_centers = _patch_centers_3d(
    image_shape=multi_frame_micrograph.shape,
    patch_shape=(1, PATCH_SIDELENGTH, PATCH_SIDELENGTH),
    patch_step=(1, PATCH_SIDELENGTH // 2, PATCH_SIDELENGTH // 2),
    distribute_patches=True
)  # (t, h, w, thw)
patch_centers = patch_centers / torch.tensor([t - 1, h - 1, w - 1])

data_patches = multi_frame_micrograph[:, patch_idx_h, patch_idx_w].detach()
reference_patches = reference[patch_idx_h, patch_idx_w].detach()

# shapes the reference and the data
mask = circle(
    radius=PATCH_SIDELENGTH / 4,
    image_shape=(PATCH_SIDELENGTH, PATCH_SIDELENGTH),
    smoothing_radius=PATCH_SIDELENGTH / 8,
)
data_patches *= mask
reference_patches *= mask

# fft the data and the reference
data_patches = torch.fft.rfftn(data_patches, dim=(-2, -1))
reference_patches = torch.fft.rfftn(reference_patches, dim=(-2, -1))

deformation_field = CubicBSplineGrid3d(
    resolution=LEARNED_DEFORMATION_FIELD_RESOLUTION,
    n_channels=2
)
motion_optimiser = torch.optim.Adam(
    params=deformation_field.parameters(),
    lr=LEARNING_RATE,
)
ph, pw = patch_centers.shape[1:3]
patch_shape = (PATCH_SIDELENGTH, PATCH_SIDELENGTH)

start = datetime.now()
for i in range(N_ITERATIONS):
    # take a random subset of the 2D patch grid and the reference patches
    patch_idx = np.random.randint(
        low=(0, 0), high=(ph, pw), size=(N_PATCHES_PER_BATCH, 2)
    )
    patch_idx_h, patch_idx_w = einops.rearrange(patch_idx, 'b idx -> idx b')
    patch_subset = data_patches[:, patch_idx_h, patch_idx_w]
    patch_subset_centers = patch_centers[:, patch_idx_h, patch_idx_w]
    reference_patch_subset = reference_patches[patch_idx_h, patch_idx_w]

    # predict the shifts at patch centers
    predicted_shifts = deformation_field(patch_subset_centers)

    # shift the data patches by the current shifts
    shifted_patches = phase_shift_dft_2d(
        patch_subset,
        shifts=predicted_shifts,
        rfft=True,
        image_shape=patch_shape,
    )  # (b, ph, pw, h, w)

    # calculate the loss, MSE between data patches and reference patches
    loss = torch.sqrt(torch.mean((reference_patch_subset - shifted_patches).abs() ** 2))

    # zero gradients, backpropagate and step optimiser
    motion_optimiser.zero_grad()
    loss.backward()
    motion_optimiser.step()

    if i % 20 == 0:
        print(loss.item())

end = datetime.now()
delta = (end - start).total_seconds()
print(f'time taken: {delta}s')


# quantify how well we're doing
gt = gt_deformation_field(patch_centers)
learned = deformation_field(patch_centers)
print(torch.mean(torch.abs(gt - learned)))

# invert the motion
predicted_shifts = -1 * deformation_field(tyx)
sample_coords = array_coordinates + predicted_shifts
reconstruction = F.grid_sample(
    input=einops.rearrange(multi_frame_micrograph, 't h w -> t 1 h w'),
    grid=array_to_grid_sample(sample_coords, array_shape=(h, w)),
    mode='bicubic',
    padding_mode='zeros',
    align_corners=True,
)
reconstruction = einops.rearrange(reconstruction, 't 1 h w -> t h w')

# visualise
import napari
viewer = napari.Viewer()
viewer.add_image(reference.detach().numpy(), name='reference')
viewer.add_image(multi_frame_micrograph.detach().numpy(), name='wavy data')
viewer.add_image(reconstruction.detach().numpy(), name='motion corrected')
napari.run()





