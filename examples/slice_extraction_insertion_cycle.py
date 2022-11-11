import torch
from scipy.spatial.transform import Rotation as R

from libtilt.backprojection.fourier import insert_slices
from libtilt.projection.fourier import extract_slices
from libtilt.utils.coordinates import generate_rotated_slice_coordinates

sidelength = 64
volume = torch.zeros((sidelength, sidelength, sidelength), dtype=torch.complex64)
weights = torch.zeros_like(volume, dtype=torch.float)
input_slice = torch.complex(torch.arange(64 * 64).reshape(64, 64).float(),
                            imag=torch.zeros((64, 64)))
rotation = torch.tensor(R.random().as_matrix()).float()
slice_coordinates = generate_rotated_slice_coordinates(rotation, sidelength=sidelength)
volume_with_slice, weights = insert_slices(
    slice_data=input_slice.reshape(1, 64, 64),
    slice_coordinates=slice_coordinates,
    dft=volume,
    weights=weights
)
volume_with_slice[weights > 1e-3] /= weights[weights > 1e-3]
output_slice = extract_slices(volume_with_slice, slice_coordinates=slice_coordinates)
print(torch.nn.functional.mse_loss(torch.real(input_slice), torch.real(output_slice)))
import napari
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(torch.real(input_slice).numpy())
viewer.add_image(torch.real(volume_with_slice).numpy())
viewer.add_image(torch.real(output_slice).numpy())
napari.run()

s0 = torch.real(input_slice).numpy()
s1 = torch.real(output_slice).numpy()
import numpy as np
s2 = np.abs(s0 - s1)
s3 = s2 ** 2
