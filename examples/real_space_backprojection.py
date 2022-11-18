import einops
import mrcfile
import napari
import numpy as np
import torch

from libtilt.projection.fourier import extract_slices
from libtilt.backprojection.real import backproject, backproject_reduce
from libtilt.utils.transformations import Rx, Ry, Rz, S
from libtilt.utils.coordinates import generate_rotated_slice_coordinates, get_array_coordinates

VOLUME_FILE = 'ribo-16Apx.mrc'

volume = torch.tensor(mrcfile.read(VOLUME_FILE))
volume_shape = torch.tensor(volume.shape)

volume_center = volume_shape // 2
tilt_image_center = volume_center[:2]

s0 = S(-volume_center)
r0 = Rx(0)
r1 = Ry(torch.linspace(-90, 90, steps=100))
r2 = Rz(0)
s1 = S([0, 0])
s2 = S(tilt_image_center)

projection_matrices = s2 @ s1 @ r2 @ r1 @ r0 @ s0

# do random orientations instead for experiments
# n_projections = 100
# random_rotations = einops.repeat(torch.eye(4), 'i j -> b i j', b=n_projections).clone()
# random_rotations[:, :3, :3] = torch.tensor(special_ortho_group.rvs(dim=3, size=n_projections))
# projection_matrices = s2 @ random_rotations @ s0

# take out rotation matrices and invert as rotating coordinates of slice in DFT rather than tomogram itself
rotation_matrices = einops.rearrange(projection_matrices[:, :3, :3], 'b i j -> b j i')
print(projection_matrices.shape, rotation_matrices.shape)

slice_coordinates = generate_rotated_slice_coordinates(rotation_matrices,
                                                       sidelength=volume_shape[0])
dft = torch.fft.fftshift(volume, dim=(0, 1, 2))
dft = torch.fft.fftn(dft, dim=(0, 1, 2))
dft = torch.fft.fftshift(dft, dim=(0, 1, 2))
slices = extract_slices(dft, slice_coordinates)
image_shape = slices.shape[-2:]
image_center = torch.tensor(image_shape) // 2
r_max = volume_shape[0] // 2
ramp_filter = torch.linalg.norm(get_array_coordinates(image_shape) - image_center, dim=-1) / r_max
slices *= ramp_filter
projections = torch.fft.ifftshift(slices, dim=(1, 2))
projections = torch.fft.ifftn(projections, dim=(1, 2))
projections = torch.fft.ifftshift(projections, dim=(1, 2))
projections = torch.real(projections)

# xyzw = torch.tensor([24, 24, 12, 1]).reshape(4, 1).float()
# xyzw_proj = projection_matrices @ xyzw
# xy_proj = torch.squeeze(xyzw_proj)[..., :2] # xy
# xyz = add_implied_coordinate_from_dimension(xy_proj, dim=0)
# zyx = torch.flip(xyz, dims=(-1, ))


reconstruction = backproject(
    projection_images=projections,
    projection_matrices=projection_matrices,
    output_dimensions=volume_shape,
)
reconstruction2 = backproject_reduce(
    images=projections,
    projection_matrices=projection_matrices,
    output_dimensions=volume_shape
)

viewer = napari.Viewer()
volume_layer = viewer.add_image(np.array(volume), name='original 3D volume')
projection_layer = viewer.add_image(np.array(projections), name='projection images')
reconstruction_layer = viewer.add_image(np.array(reconstruction), name='3D reconstruction (WBP)')
reconstruction_layer2 = viewer.add_image(np.array(reconstruction2))
napari.run()
