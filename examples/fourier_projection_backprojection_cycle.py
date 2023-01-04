import sys

import mrcfile
import napari
import torch
import numpy as np
from scipy.stats import special_ortho_group

from libtilt.mask.mask import _nd_circle
from libtilt.utils.coordinates import generate_rotated_slice_coordinates
from libtilt.projection.fourier import extract_slices
from libtilt.shift.phase_shift_2d import phase_shift_dfts_2d, \
    phase_shift_images_2d
from libtilt.backprojection.fourier import reconstruct_from_images, \
    insert_slices, _grid_sinc2
from libtilt.utils.fft import symmetrised_dft_to_dft_3d
from libtilt.fsc import fsc
from libtilt.utils.transformations import Ry

N_IMAGES = 5000
USE_SMALL_VOLUME = False
RECONSTRUCT_SYMMETRISED_DFT = True
DO_VIS = True

big_volume_file = '4v6x.mrc'
small_volume_file = 'ribo-16Apx.mrc'

volume_file = small_volume_file if USE_SMALL_VOLUME is True else big_volume_file

# read and normalise volume
volume = torch.tensor(mrcfile.read(volume_file))
volume -= torch.mean(volume)
volume /= torch.std(volume)

# forward model, gridding correction then make n projections
rotations = torch.tensor(special_ortho_group.rvs(dim=3, size=N_IMAGES)).float()
# rotations = Ry(torch.linspace(-60, 60, 41))[:, :3, :3]
slice_coordinates = generate_rotated_slice_coordinates(rotations,
                                                       sidelength=volume.shape[
                                                           0])
sinc2 = _grid_sinc2(volume.shape)
volume *= sinc2
dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
dft = torch.fft.fftn(dft, dim=(-3, -2, -1))
dft = torch.fft.fftshift(dft, dim=(-3, -2, -1))
dft_slices = extract_slices(dft, slice_coordinates)  # (b, h, w)

# shift the projections (dft slices) by phase shifting
std = 5 if USE_SMALL_VOLUME is True else 20
shifts = torch.normal(mean=0, std=std, size=(N_IMAGES, 2))
shifted_slices = phase_shift_dfts_2d(
    dfts=dft_slices,
    shifts=shifts,
    image_shape=dft_slices.shape[-2:],
    rfft=False,
    spectrum_is_fftshifted=True
)

# generate images from shifted projections
shifted_projections = torch.fft.ifftshift(shifted_slices, dim=(-2, -1))
shifted_projections = torch.fft.ifftn(shifted_projections, dim=(-2, -1))
shifted_projections = torch.fft.ifftshift(shifted_projections, dim=(-2, -1))
shifted_projections = torch.real(shifted_projections)

# phase shift images back again and mask
centered_projections = phase_shift_images_2d(shifted_projections,
                                             shifts=-shifts)
mrcfile.write('projection_for_thomas.mrc', centered_projections[0].numpy(

).astype(np.float32), overwrite=True)
# mask = _nd_circle(
#     sidelength=centered_projections.shape[-1],
#     ndim=2,
#     radius=centered_projections.shape[-1] * 0.2,
#     smoothing_radius=5
# )
# centered_projections *= mask
dft_slices = torch.fft.fftshift(centered_projections, dim=(-2, -1))
dft_slices = torch.fft.fftn(dft_slices, dim=(-2, -1))
dft_slices = torch.fft.fftshift(dft_slices, dim=(-2, -1))

# reconstruct from recentered dft slices
d = volume.shape[0]
if RECONSTRUCT_SYMMETRISED_DFT is True:
    d += 1
reconstruction = torch.zeros(size=(d, d, d), dtype=torch.complex64)
weights = torch.zeros(size=(d, d, d), dtype=torch.float)
reconstruction, weights = insert_slices(
    slice_data=dft_slices,
    slice_coordinates=slice_coordinates,
    dft=reconstruction,
    weights=weights,
)

# reweight data in Fourier space
valid_weights = weights > 1e-3
reconstruction[valid_weights] /= weights[valid_weights]

# desymmetrise dft
if RECONSTRUCT_SYMMETRISED_DFT is True:
    reconstruction = symmetrised_dft_to_dft_3d(reconstruction, inplace=True)

# back to real space
reconstruction = torch.fft.ifftshift(reconstruction, dim=(-3, -2, -1))
reconstruction = torch.fft.ifftn(reconstruction, dim=(-3, -2, -1))
reconstruction = torch.fft.ifftshift(reconstruction, dim=(-3, -2, -1))
reconstruction = torch.real(reconstruction)

# gridding correction
reconstruction /= sinc2

fsc = fsc(reconstruction, volume)
print(fsc)
mrcfile.write('output.mrc', data=reconstruction.numpy().astype(np.float32),
              overwrite=True)

if DO_VIS is False:
    sys.exit(0)

# view in napari
viewer = napari.Viewer()
viewer.add_image(np.array(volume), name='input volume')
viewer.add_image(np.array(shifted_projections), name='projections')
viewer.add_image(np.array(centered_projections), name='centered projections')
viewer.add_image(np.array(reconstruction),
                 name='3D reconstruction from projections')
viewer.add_image(np.array(reconstruction - volume), name='difference map')
# viewer.add_image(
#     torch.log(torch.abs(torch.real(
#         torch.fft.fftshift(
#             torch.fft.fftn(reconstruction, dim=(-3, -2, -1)),
#             dim=(-3, -2, -1))
#     )
#     )).cpu().numpy()
# )



def _on_ndisplay_change():
    viewer.camera.center = np.array(volume.shape) / 2


viewer.dims.events.ndisplay.connect(_on_ndisplay_change)
for layer_name in ('3D reconstruction from projections', 'difference map'):
    viewer.layers[layer_name].contrast_limits = \
        viewer.layers['input volume'].contrast_limits
napari.run()
