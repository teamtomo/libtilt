import dask.array as da
import numpy as np
import torch
from scipy.stats import special_ortho_group

from libtilt.utils.coordinates import generate_rotated_slice_coordinates
from libtilt.projection.fourier import extract_slices
from libtilt.shift.phase_shift_2d import phase_shift_dfts_2d, \
    phase_shift_images_2d
from libtilt.backprojection.fourier import reconstruct_from_images, \
    insert_slices, _grid_sinc2
from libtilt.utils.fft import symmetrised_dft_to_dft_3d
from libtilt.fsc import fsc
from libtilt.utils.transformations import Ry

N_IMAGES = 1000
SIDELENGTH = 128

dft_slices = da.ones(
    (N_IMAGES, SIDELENGTH, SIDELENGTH),
    chunks=(100, -1, -1),
    dtype=np.complex64
)
random_rotations = torch.tensor(special_ortho_group.rvs(dim=3, size=N_IMAGES)).float()
slice_coordinates = generate_rotated_slice_coordinates(
    rotations=random_rotations, sidelength=SIDELENGTH
)

d = SIDELENGTH
reconstruction = torch.zeros(size=(d, d, d), dtype=torch.complex64)
weights = torch.zeros(size=(d, d, d), dtype=torch.float)
reconstruction, weights = insert_slices(
    slice_data=dft_slices,
    slice_coordinates=slice_coordinates,
    dft=reconstruction,
    weights=weights,
)


