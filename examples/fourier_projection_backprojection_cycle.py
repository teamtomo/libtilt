import mrcfile
import napari
import torch
import numpy as np
from scipy.stats import special_ortho_group

from libtilt.utils.coordinates import generate_rotated_slice_coordinates
from libtilt.projection.fourier import extract_slices
from libtilt.shift.phase_shift_2d import fourier_shift_dfts_2d, phase_shift_images_2d
from libtilt.backprojection.fourier import reconstruct_from_images

VOLUME_FILE = 'ribo-16Apx.mrc'
N_IMAGES = 2500

volume = torch.tensor(mrcfile.read(VOLUME_FILE))
# volume = torch.nn.functional.pad(volume, (48, 48, 48, 48, 48, 48))

# forward model, make n projections
rotations = torch.tensor(special_ortho_group.rvs(dim=3, size=N_IMAGES)).float()
slice_coordinates = generate_rotated_slice_coordinates(rotations, sidelength=volume.shape[0])
dft = torch.fft.fftshift(volume, dim=(0, 1, 2))
dft = torch.fft.fftn(dft, dim=(0, 1, 2))
dft = torch.fft.fftshift(dft, dim=(0, 1, 2))
dft_slices = extract_slices(dft, slice_coordinates)  # (b, h, w)
shifts = torch.normal(mean=0, std=5, size=(N_IMAGES, 2))
shifted_slices = fourier_shift_dfts_2d(dft_slices, shifts=shifts, image_shape=dft_slices.shape[-2:], rfft=False, spectrum_is_fftshifted=True)
projections = torch.fft.ifftshift(shifted_slices, dim=(1, 2))
projections = torch.fft.ifftn(projections, dim=(1, 2))
projections = torch.fft.ifftshift(projections, dim=(1, 2))
projections = torch.real(projections)

# shifts can also be applied on images directly
# let's do this to recenter our projections for 3D reconstruction
recentered_projections = phase_shift_images_2d(projections, shifts=-shifts)

# 3D reconstruction from projection data
for i in torch.arange(100, N_IMAGES, step=100):
    reconstruction = reconstruct_from_images(images=recentered_projections[:i], slice_coordinates=slice_coordinates[:i])
    mask = volume > 0.05
    loss = torch.mean(torch.abs((reconstruction[mask] - volume[mask])) / torch.abs(reconstruction[mask]))
    print(loss, f' for {i} images')

viewer = napari.Viewer()
viewer.add_image(np.array(volume), name='input volume')
viewer.add_image(np.array(projections), name='projections')
viewer.add_image(np.array(recentered_projections), name='recentered projections')
viewer.add_image(np.array(reconstruction), name='3D reconstruction from projections')
viewer.add_image(np.array(reconstruction - volume), name='difference map')


def _on_ndisplay_change():
    viewer.camera.center = np.array(volume.shape) / 2


viewer.dims.events.ndisplay.connect(_on_ndisplay_change)
for layer_name in ('3D reconstruction from projections', 'difference map'):
    viewer.layers[layer_name].contrast_limits = \
        viewer.layers['input volume'].contrast_limits
napari.run()
