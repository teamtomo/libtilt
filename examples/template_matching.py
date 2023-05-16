"""A toy implementation of a template matching routine."""

import einops
import mmdf
import numpy as np
import torch
import napari

from libtilt.interpolation import insert_into_image_2d

input_model_file = 'data/4v6x-ribo.cif'
n_particles = 30
output_pixel_spacing = 1
output_image_shape = (4096, 4096)
add_noise = False

# load molecular model, center and rescale
print(f"loading model from {input_model_file}")
df = mmdf.read(input_model_file)
atom_zyx = torch.tensor(df[['z', 'y', 'x']].to_numpy()).float()  # (n_atoms, 3)
atom_zyx -= torch.mean(atom_zyx, dim=-1, keepdim=True)
atom_zyx /= output_pixel_spacing

# randomly place in volume
print("placing particles in 3D volume")
volume_shape = (output_image_shape[0] // 3, *output_image_shape)
pz, py, px = [
    np.random.uniform(low=0, high=dim_length, size=n_particles)
    for dim_length in volume_shape
]
particle_positions = einops.rearrange([pz, py, px], 'zyx b -> b 1 zyx')
per_particle_atom_positions = atom_zyx + particle_positions

# simulate image
print("simulating image")
atom_yx = per_particle_atom_positions[..., 1:]
atom_yx = einops.rearrange(atom_yx, 'particles atoms yx -> (particles atoms) yx')
n_atoms = atom_yx.shape[0]
values = torch.ones(n_atoms)
image = torch.zeros(output_image_shape)
weights = torch.zeros_like(image)
image, weights = insert_into_image_2d(
    data=values,
    coordinates=atom_yx,
    image=image,
    weights=weights
)

if add_noise is True:
    image = image + np.random.normal(loc=0, scale=50, size=output_image_shape)
else:
    image = image

# simulate a reference image for template matching
print("simulating reference")
reference_zyx = atom_zyx + np.array([0, *output_image_shape]) // 2
reference_yx = reference_zyx[..., 1:]
n_atoms = reference_yx.shape[0]

values = torch.ones(n_atoms)
reference = torch.zeros(output_image_shape)
weights = torch.zeros_like(image)
reference, weights = insert_into_image_2d(
    data=values,
    coordinates=reference_yx,
    image=reference,
    weights=weights,
)
reference = torch.fft.fftshift(reference, dim=(-2, -1))

print("convolution theorem-ing it up")
image_dft = torch.fft.fftn(image, dim=(-2, -1))
reference_dft = torch.fft.fftn(reference, dim=(-2, -1))
product = image_dft * reference_dft
result = torch.real(torch.fft.ifftn(product, dim=(-2, -1)))

# visualise results
viewer = napari.Viewer()
viewer.add_image(
    image.numpy(),
    name='image',
    contrast_limits=(0, torch.max(image))
)
viewer.add_image(
    reference.numpy(),
    name='reference',
    visible=False,
    contrast_limits=(0, torch.max(reference))
)
viewer.add_image(
    result.numpy(),
    name='template matching result',
    contrast_limits=(0, torch.max(result)),
    colormap='inferno',
    blending='additive',
    opacity=0.3,
)
napari.run()