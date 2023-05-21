"""A toy implementation of a template matching routine."""

import einops
import mmdf
import numpy as np
import torch
import napari


from libtilt.atomic_models import coordinates_to_image_2d
from libtilt.filters import bfactor_2d

INPUT_MODEL_FILE = 'data/4v6x-ribo.cif'
N_PARTICLES = 30
SIMULATION_PIXEL_SPACING = 1
SIMULATION_IMAGE_SHAPE = (4096, 4096)
ADD_NOISE = False
BFACTOR = 85

# load molecular model, center and rescale
print(f"loading model from {INPUT_MODEL_FILE}")
df = mmdf.read(INPUT_MODEL_FILE)
atom_zyx = torch.tensor(df[['z', 'y', 'x']].to_numpy()).float()  # (n_atoms, 3)
atom_zyx -= torch.mean(atom_zyx, dim=-1, keepdim=True)  # center
atom_zyx /= SIMULATION_PIXEL_SPACING  # rescale

# randomly place in volume
print("placing particles in 3D volume")
volume_shape = (SIMULATION_IMAGE_SHAPE[0] // 3, *SIMULATION_IMAGE_SHAPE)
pz, py, px = [
    np.random.uniform(low=0, high=dim_length, size=N_PARTICLES)
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
image = torch.zeros(SIMULATION_IMAGE_SHAPE)
weights = torch.zeros_like(image)
image = coordinates_to_image_2d(
    coordinates=atom_yx,
    image_shape=SIMULATION_IMAGE_SHAPE,
)

if ADD_NOISE is True:
    image = image + np.random.normal(loc=0, scale=50, size=SIMULATION_IMAGE_SHAPE)
else:
    image = image


# simulate a reference image for template matching
print("simulating reference")
reference_zyx = atom_zyx + np.array([0, *SIMULATION_IMAGE_SHAPE]) // 2
reference_yx = reference_zyx[..., 1:]
n_atoms = reference_yx.shape[0]

values = torch.ones(n_atoms)
reference = torch.zeros(SIMULATION_IMAGE_SHAPE)
weights = torch.zeros_like(image)
reference = coordinates_to_image_2d(
    coordinates=reference_yx,
    image_shape=SIMULATION_IMAGE_SHAPE,
)
reference = torch.fft.fftshift(reference, dim=(-2, -1))

# Here the B factor is being applied to each image but
# it will be more efficient to apply it to the 3D reference
# (I was just testing it)
if BFACTOR > 0:
    reference = bfactor_2d(
        image=reference,
        B=BFACTOR,
        pixel_size=SIMULATION_PIXEL_SPACING,
    )

print("convolution theorem-ing it up")
image_dft = torch.fft.rfftn(image, dim=(-2, -1))
reference_dft = torch.fft.rfftn(reference, dim=(-2, -1))
product = image_dft * reference_dft
result = torch.real(torch.fft.irfftn(product, dim=(-2, -1)))

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