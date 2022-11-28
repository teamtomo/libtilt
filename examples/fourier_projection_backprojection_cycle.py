import mrcfile
import napari
import torch
import numpy as np
from scipy.stats import special_ortho_group

from libtilt.utils.coordinates import generate_rotated_slice_coordinates
from libtilt.projection.fourier import extract_slices
from libtilt.shift.phase_shift_2d import phase_shift_dfts_2d, phase_shift_images_2d
from libtilt.backprojection.fourier import reconstruct_from_images, insert_slices


N_IMAGES = 1000
USE_SMALL_VOLUME = False

big_volume_file = '4v6x.mrc'
small_volume_file = 'ribo-16Apx.mrc'

volume_file = small_volume_file if USE_SMALL_VOLUME is True else big_volume_file

volume = torch.tensor(mrcfile.read(volume_file))
volume -= torch.mean(volume)
volume /= torch.std(volume)

# volume = torch.nn.functional.pad(volume, (48, 48, 48, 48, 48, 48))

# forward model, make n projections
rotations = torch.tensor(special_ortho_group.rvs(dim=3, size=N_IMAGES)).float()
slice_coordinates = generate_rotated_slice_coordinates(rotations, sidelength=volume.shape[0])
dft = torch.fft.fftshift(volume, dim=(0, 1, 2))
dft = torch.fft.fftn(dft, dim=(0, 1, 2))
dft = torch.fft.fftshift(dft, dim=(0, 1, 2))
dft_slices = extract_slices(dft, slice_coordinates)  # (b, h, w)
# shifts = torch.normal(mean=0, std=5, size=(N_IMAGES, 2))
# shifted_slices = fourier_shift_dfts_2d(dft_slices, shifts=shifts, image_shape=dft_slices.shape[-2:], rfft=False, spectrum_is_fftshifted=True)
reconstruction = torch.zeros_like(volume, dtype=torch.complex64)
weights = torch.zeros_like(volume, dtype=torch.float)
reconstruction, weights = insert_slices(
    slice_data=dft_slices,
    slice_coordinates=slice_coordinates,
    dft=reconstruction,
    weights=weights,
)
valid_weights = weights > 1e-3
reconstruction[valid_weights] /= weights[valid_weights]
reconstruction = torch.fft.ifftshift(reconstruction, dim=(-3, -2, -1))
reconstruction = torch.fft.ifftn(reconstruction, dim=(-3, -2, -1))
reconstruction = torch.fft.ifftshift(reconstruction, dim=(-3, -2, -1))
reconstruction = torch.real(reconstruction)
# projections = torch.fft.ifftshift(dft_slices, dim=(1, 2))
# projections = torch.fft.ifftn(projections, dim=(1, 2))
# projections = torch.fft.ifftshift(projections, dim=(1, 2))
# projections = torch.real(projections)

# shifts can also be applied on images directly
# let's do this to recenter our projections for 3D reconstruction
# recentered_projections = phase_shift_images_2d(projections, shifts=-shifts)

# 3D reconstruction from projection data
# for i in torch.arange(100, N_IMAGES, step=100):
#     reconstruction = reconstruct_from_images(images=recentered_projections[:i], slice_coordinates=slice_coordinates[:i], do_gridding_correction=True)
#     mask = volume > 0.05
#     loss = torch.mean(torch.abs((reconstruction[mask] - volume[mask])) / torch.abs(reconstruction[mask]))
#     print(loss, f' for {i} images')
# reconstruction = reconstruct_from_images(images=recentered_projections, slice_coordinates=slice_coordinates, do_gridding_correction=True)
# mask = volume > 0.05
# loss = torch.mean(torch.abs((reconstruction[mask] - volume[mask])) / torch.abs(reconstruction[mask]))
# print(loss)
viewer = napari.Viewer()
viewer.add_image(np.array(volume), name='input volume')
# viewer.add_image(np.array(projections), name='projections')
# viewer.add_image(np.array(recentered_projections), name='recentered projections')
viewer.add_image(np.array(reconstruction), name='3D reconstruction from projections')
viewer.add_image(np.array(reconstruction - volume), name='difference map')


def _on_ndisplay_change():
    viewer.camera.center = np.array(volume.shape) / 2


viewer.dims.events.ndisplay.connect(_on_ndisplay_change)
for layer_name in ('3D reconstruction from projections', 'difference map'):
    viewer.layers[layer_name].contrast_limits = \
        viewer.layers['input volume'].contrast_limits
napari.run()
