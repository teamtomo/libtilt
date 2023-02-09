from typing import Tuple, Sequence

import mrcfile
import numpy as np
import torch
import einops
import napari
from tiler import Tiler
from torch_cubic_b_spline_grid import CubicBSplineGrid2d, CubicBSplineGrid1d
from libtilt.utils.rotational_average import rotational_average_2d
from libtilt.ctf.ctf_1d import calculate_ctf as calculate_ctf_1d

image_file = 'data/TS_01_015_21.0.mrc'

image = torch.tensor(mrcfile.read(image_file)).float()
image = einops.repeat(image, 'h w -> 2 h w')  # pretend the image is multi-frame

grid_resolution = (1, 2, 2)
grid_t, grid_h, grid_w = grid_resolution
power_spectrum = torch.log(torch.abs(torch.fft.fftn(image, dim=(-2, -1))))
power_spectrum = torch.fft.fftshift(power_spectrum, dim=(-2, -1))

# vis image
# viewer = napari.Viewer()
# viewer.add_image(image.numpy())
# viewer.add_image(power_spectrum.numpy())
# napari.run()


def get_2d_patch_indices(
    image_shape: Sequence[int],
    patch_shape: Tuple[int, int],
    patch_step: Tuple[int, int],
) -> torch.Tensor:
    image_h, image_w = image_shape[-2:]
    patch_h, patch_w = patch_shape
    step_h, step_w = patch_step
    idx_center_h = torch.arange(patch_h // 2, image_h - patch_h // 2 + 1, step=step_h)
    idx_center_w = torch.arange(patch_w // 2, image_w - patch_w // 2 + 1, step=step_w)
    idx_h = einops.rearrange(idx_center_h, 'h -> h 1 1 1')
    idx_h = idx_h + einops.rearrange(np.arange(patch_h) - patch_h // 2, 'h -> h 1')
    idx_w = einops.rearrange(idx_center_w, 'w -> 1 w 1 1')
    idx_w = idx_w + einops.rearrange(np.arange(patch_w) - patch_w // 2, 'w -> 1 w')
    return idx_h, idx_w

idx_h, idx_w = get_2d_patch_indices(
    image_shape=image.shape,
    patch_shape=(512, 512),
    patch_step=(512, 512),
)

patches = image[..., idx_h, idx_w]
ps = torch.abs(torch.fft.rfftn(patches, dim=(-2, -1))) ** 2
# temporarily do mean over everything, might spatially resolve means later
_t = 'b' if grid_t > 1 else ''
_h = 'ph' if grid_h > 1 else ''
_w = 'pw' if grid_w > 1 else ''
reduced_dims = f'{_t} {_h} {_w}'
ps = einops.reduce(ps, f'b ph pw h w -> {reduced_dims} h w', reduction='mean')



# tiler - calculate average 2D PS of all patches
# tiler = Tiler(
#     data_shape=image.shape,
#     channel_dimension=0,
#     tile_shape=(2, 512, 512),
#     overlap=256,
#     mode='drop'
# )
# b = 8
# ps = 0
# for batch_id, tile in tiler(image.numpy(), batch_size=b, copy_data=False):
#     bbox = np.asarray(
#         [
#             tiler.get_tile_bbox(tile_id=_tile_id)
#             for _tile_id in [batch_id * b + i for i in range(b) if batch_id * b + i <
#                              len(tiler)]
#         ], dtype=np.float32)
#     centers = einops.reduce(bbox, 'b bbox hw -> b hw', reduction='mean')
#     centers_grid_coordinates = centers / (np.array(image.shape[-2:]) - 1)
#     print(centers_grid_coordinates)
#     _ps = torch.abs(torch.fft.fftn(torch.tensor(tile), dim=(-2, -1))) ** 2
#     _ps = torch.fft.fftshift(_ps, dim=(-2, -1))
#
#     # if no temporal resolution desired, average over t
#     if grid_resolution[0] == 1:
#         _ps = einops.reduce(_ps, 'b ... h w -> ... h w', reduction='sum'
#     )
#     ps = _ps + ps
#
# ps /= len(tiler)

# rotationally average
rotational_average = rotational_average_2d(
    ps, rfft=True, fftshifted=False
)
rotational_average = einops.reduce(
    rotational_average, '... shell -> shell', reduction='mean'
)

# estimate background
background_model = CubicBSplineGrid1d(resolution=3)
background_optimiser = torch.optim.Adam(params=background_model.parameters(), lr=1)

y = rotational_average[15:80]
x = torch.linspace(0, 1, steps=len(y))
for i in range(250):
    prediction = background_model(x).squeeze()
    loss = torch.mean((torch.log(y) - prediction)**2)
    loss.backward()
    background_optimiser.step()
    background_optimiser.zero_grad()
    print(loss.item())

# # vis avg ps from patches
# viewer = napari.Viewer()
# viewer.add_image(image.numpy())
# viewer.add_image(torch.log(ps).numpy())
# napari.run()

# vis RAPS
# from matplotlib import pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(np.log(rotational_average[15:80].numpy()))
# ax.plot(background_model(x).squeeze().detach().numpy())
# ax.set(xlim=(0, 80), ylim=(17, 18))
# plt.show()

# subtract background
y -= torch.exp(background_model(x).squeeze())

# vis background subtracted RAPS
# from matplotlib import pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(y.detach().numpy())
# plt.show()

# simulate 1d ctfs
ctfs = calculate_ctf_1d(
    defocus=np.linspace(
        start=0.5,
        stop=12,
        num=1000,
    ),
    voltage=300,
    spherical_aberration=2.7,
    amplitude_contrast=0.10,
    b_factor=0,
    phase_shift=0,
    pixel_size=1.1,
    n_samples=256,
    oversampling_factor=3,
)
# vis simulated ctfs
# viewer = napari.Viewer()
# viewer.add_image(torch.atleast_2d(torch.log(rotational_average)).detach().numpy())
# viewer.add_image(ctfs.detach().numpy()[::-1])
# napari.run()

ctf_fit_range = ctfs[:, 15:80] ** 2
ctf_fit_range_normed = ctf_fit_range / torch.linalg.norm(ctf_fit_range, dim=-1,
                                                         keepdim=True)
y_fit = y / torch.linalg.norm(y)
zncc = einops.einsum(ctf_fit_range_normed, y_fit, 'b i, i -> b')

# vis cc vs. defocus
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.scatter(x=np.linspace(
        start=0.5,
        stop=12,
        num=1000,
    ), y=zncc.detach().numpy())
plt.show()

fig, ax = plt.subplots()
ax.plot(ctf_fit_range[torch.argmax(zncc).long()].detach().numpy())
ax.plot(y_fit.detach().numpy())
plt.show()

