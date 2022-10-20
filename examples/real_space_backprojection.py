import einops
import mrcfile
import napari
import numpy as np
import scipy.stats
import torch
from scipy.stats import special_ortho_group

from libtilt.dft_extract_slices import slice_dft
from libtilt.backprojection import backproject
from libtilt.transformations import Rx, Ry, Rz, S
from libtilt.utils.coordinates import generate_rotated_slice_coordinates, get_grid_coordinates

VOLUME_FILE = 'ribo-16Apx.mrc'

volume = torch.tensor(mrcfile.read(VOLUME_FILE))
volume_shape = torch.tensor(volume.shape)

volume_center = volume_shape // 2
tilt_image_center = volume_center[:2]

s0 = S(-volume_center)
r0 = Rx(0)
r1 = Ry(torch.linspace(-90, 90, steps=41))
r2 = Rz(0)
s1 = S([0, 0])
s2 = S(tilt_image_center)

projection_matrices = s2 @ s1 @ r2 @ r1 @ r0 @ s0
# take out rotation matrices and invert as rotating coordinates of slice in DFT rather than tomogram itself
rotation_matrices = einops.rearrange(projection_matrices[:, :3, :3], 'b i j -> b j i')
print(projection_matrices.shape, rotation_matrices.shape)

slice_coordinates = generate_rotated_slice_coordinates(rotation_matrices, n=volume_shape[0])
dft = torch.fft.fftshift(volume, dim=(0, 1, 2))
dft = torch.fft.fftn(dft, dim=(0, 1, 2))
dft = torch.fft.fftshift(dft, dim=(0, 1, 2))
slices = slice_dft(dft, slice_coordinates)
image_shape = slices.shape[-2:]
image_center = torch.tensor(image_shape) // 2
r_max = volume_shape[0] // 2
ramp = torch.linalg.norm(get_grid_coordinates(image_shape) - image_center, dim=-1) / r_max
slices *= ramp
projections = torch.fft.ifftshift(slices, dim=(1, 2))
projections = torch.fft.ifftn(projections, dim=(1, 2))
projections = torch.fft.ifftshift(projections, dim=(1, 2))
projections = torch.real(projections)

reconstruction = backproject(
    image_stack=projections,
    projection_matrices=projection_matrices,
    output_dimensions=volume_shape,
)

viewer = napari.Viewer()
viewer.add_image(np.array(volume))
viewer.add_image(np.array(projections))
viewer.add_image(np.array(reconstruction))
napari.run()
