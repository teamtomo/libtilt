"""Toy implementation of a motion correction algorithm similar to the
one in Warp (https://doi.org/10.1038/s41592-019-0580-y)
"""
from datetime import datetime

import torch
import numpy as np
import mrcfile
import einops

import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d

from libtilt.coordinate_utils import array_to_grid_sample
from libtilt.shift.fourier_shift import phase_shift_dft_2d
from libtilt.shapes.shapes_2d import circle
from libtilt.grids import patch_grid

IMAGE_FILE = 'data/TS_01_000_0.0.mrc'
GT_DEFORMATION_FIELD_RESOLUTION = (3, 3, 3)  # (t, h, w)
LEARNED_DEFORMATION_FIELD_RESOLUTION = (5, 5, 5)  # (t, h, w)
PATCH_SIDELENGTH = 64
N_ITERATIONS = 200
N_PATCHES_PER_BATCH = 15
LEARNING_RATE = 0.05

# load data, bin 8x, reduce to 2D and normalise
image = torch.tensor(mrcfile.read(IMAGE_FILE)).float()
image = F.avg_pool2d(image, kernel_size=8)
image = einops.reduce(image, 't h w -> h w', reduction='mean')
image = (image - torch.mean(image)) / torch.std(image)

# simulate continuous beam induced motion on the image
# make image multi-frame, initialise a deformation field with random shifts
multi_frame_micrograph = einops.repeat(image, 'h w -> 10 h w').clone()
gt_deformation_field_data = np.random.uniform(
    low=-5, high=5, size=(2, *GT_DEFORMATION_FIELD_RESOLUTION)
)
gt_deformation_field = CubicBSplineGrid3d.from_grid_data(
    torch.tensor(gt_deformation_field_data).float()
)

# evaluate deformation field at every pixel
t, h, w = multi_frame_micrograph.shape[-3:]
_t, _y, _x = torch.linspace(0, 1, t), torch.linspace(0, 1, h), torch.linspace(0, 1, w)
tt, yy, xx = torch.meshgrid(_t, _y, _x, indexing='ij')
tyx = einops.rearrange([tt, yy, xx], 'tyx t h w -> t h w tyx')
shifts = gt_deformation_field(tyx)
array_coordinates = torch.tensor(np.indices(image.shape[-2:]))
array_coordinates = einops.rearrange(array_coordinates, 'yx ... -> ... yx')

# find coordinates to sample (after deforming by shifts)
deformed_coordinates = array_coordinates + shifts

# sample image at deformed positions
multi_frame_micrograph = F.grid_sample(
    input=einops.rearrange(multi_frame_micrograph, 't h w -> t 1 h w'),
    grid=array_to_grid_sample(deformed_coordinates, array_shape=(h, w)),
    mode='bicubic',
    padding_mode='zeros',
    align_corners=True,
)
multi_frame_micrograph = einops.rearrange(multi_frame_micrograph, 't 1 h w -> t h w')

# prepare data for learning the motion...
reference = image.clone()

# extract patches from data and reference
ph, pw = (PATCH_SIDELENGTH, PATCH_SIDELENGTH)
data_patches, data_patch_centers = patch_grid(
    images=multi_frame_micrograph,
    patch_shape=(1, ph, pw),
    patch_step=(1, ph // 2, pw // 2),
    distribute_patches=True,
)
data_patches = einops.rearrange(data_patches, 't gh gw 1 ph pw -> t gh gw ph pw')
reference_patches, _ = patch_grid(
    images=reference,
    patch_shape=(PATCH_SIDELENGTH, PATCH_SIDELENGTH),
    patch_step=(PATCH_SIDELENGTH // 2, PATCH_SIDELENGTH // 2),
    distribute_patches=True,
)  # (grid_h, grid_w, ph, pw)
gh, gw = data_patch_centers.shape[1:3]

# apply a soft circular mask on both reference and data patches
mask = circle(
    radius=PATCH_SIDELENGTH / 4,
    image_shape=(PATCH_SIDELENGTH, PATCH_SIDELENGTH),
    smoothing_radius=PATCH_SIDELENGTH / 8,
)
data_patches *= mask
reference_patches *= mask

# rfft the data and the reference
data_patches = torch.fft.rfftn(data_patches, dim=(-2, -1))
reference_patches = torch.fft.rfftn(reference_patches, dim=(-2, -1))

# initialise the deformation field with learnable parameters and normalise
# patch_extraction centers to [0, 1] for evaluation of shifts.
deformation_field = CubicBSplineGrid3d(
    resolution=LEARNED_DEFORMATION_FIELD_RESOLUTION,
    n_channels=2
)
data_patch_centers = data_patch_centers / torch.tensor([t - 1, h - 1, w - 1])

# initialise optimiser and detach data
motion_optimiser = torch.optim.Adam(
    params=deformation_field.parameters(),
    lr=LEARNING_RATE,
)
data_patches = data_patches.detach()
reference_patches=reference_patches.detach()


# optimise shifts at grid points on deformation field
start = datetime.now()
for i in range(N_ITERATIONS):
    # take a random subset of the patch_extraction grid over spatial dimensions
    subset_idx = np.random.randint(
        low=(0, 0), high=(gh, gw), size=(N_PATCHES_PER_BATCH, 2)
    )
    patch_idx_h, patch_idx_w = einops.rearrange(subset_idx, 'b idx -> idx b')
    patch_subset = data_patches[:, patch_idx_h, patch_idx_w]
    patch_subset_centers = data_patch_centers[:, patch_idx_h, patch_idx_w]
    reference_patch_subset = reference_patches[patch_idx_h, patch_idx_w]

    # predict the shifts at patch_extraction centers
    predicted_shifts = deformation_field(patch_subset_centers)

    # shift the patches by the predicted shifts
    shifted_patches = phase_shift_dft_2d(
        dft=patch_subset,
        image_shape=(ph, pw),
        shifts=predicted_shifts,
        rfft=True,
    )  # (b, ph, pw, h, w)

    # calculate the loss, MSE between data patches and reference patches
    loss = torch.mean((reference_patch_subset - shifted_patches).abs() ** 2)

    # zero gradients, backpropagate loss and step optimiser
    motion_optimiser.zero_grad()
    loss.backward()
    motion_optimiser.step()

    if i % 20 == 0:
        print(loss.item())
end = datetime.now()
delta = (end - start).total_seconds()
print(f'time taken: {delta}s')

# quantify how well we're doing over whole field
gt = gt_deformation_field(tyx)
learned = deformation_field(tyx)
print(torch.mean(torch.abs(gt - learned)))

# invert the motion to reconstruct a 'motion corrected' image
predicted_shifts = -1 * deformation_field(tyx)
sample_coords = array_coordinates + predicted_shifts
corrected_image_thw = F.grid_sample(
    input=einops.rearrange(multi_frame_micrograph, 't h w -> t 1 h w'),
    grid=array_to_grid_sample(sample_coords, array_shape=(h, w)),
    mode='bicubic',
    padding_mode='zeros',
    align_corners=True,
)
corrected_image_thw = einops.rearrange(corrected_image_thw, 't 1 h w -> t h w')
corrected_image = einops.reduce(corrected_image_thw, 't h w -> h w', reduction='mean')

# visualise results
import napari
viewer = napari.Viewer()
viewer.add_image(reference.detach().numpy(), name='reference')
viewer.add_image(multi_frame_micrograph.detach().numpy(), name='beam induced motion (simulated)')
viewer.add_image(corrected_image_thw.detach().numpy(), name='motion corrected (2d+t)')
viewer.add_image(corrected_image.detach().numpy(), name='motion corrected (2d)')
napari.run()
