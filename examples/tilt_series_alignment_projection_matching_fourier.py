"""A toy implementation of a tilt-series alignment routine using projection matching.

This implementation is a little more complicated but avoids many unnecessary Fourier
transforms during the optimisation loop.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from libtilt.backprojection.backproject_fourier import insert_central_slices_rfft
from libtilt.grids import fftfreq_grid
from libtilt.projection.project_fourier import extract_central_slices_rfft
from libtilt.shapes import sphere
from libtilt.shift import phase_shift_dft_2d
from libtilt.transformations import Ry

# simulate a volume with two spheres
ground_truth_volume = sphere(
    radius=4, image_shape=(32, 32, 32), center=(16, 16, 8), smoothing_radius=2
)
ground_truth_volume = ground_truth_volume + sphere(
    radius=4, image_shape=(32, 32, 32), center=(16, 16, 24), smoothing_radius=2
)
# pad_length = ground_truth_volume.shape[-1] // 2
pad_length = 32 // 2
ground_truth_volume = F.pad(ground_truth_volume, pad=[pad_length] * 6, mode='constant', value=0)


# simulate a tilt-series from the volume
# setup shifts and rotations
rotation_matrices = Ry(torch.linspace(-60, 60, steps=41), zyx=True)[:, :3, :3].float()
ground_truth_shifts = torch.as_tensor(
    np.random.normal(loc=0, scale=1.5, size=(41, 2))
).float()
ground_truth_shifts[20] = 0

# premultiply by sinc2
grid = fftfreq_grid(
    image_shape=ground_truth_volume.shape,
    rfft=False,
    fftshift=True,
    norm=True,
)
ground_truth_volume = ground_truth_volume * torch.sinc(grid) ** 2

# calculate 3D DFT
dft = torch.fft.fftshift(ground_truth_volume, dim=(-3, -2, -1))  # volume center to array origin
dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
dft = torch.fft.fftshift(dft, dim=(-3, -2,))  # actual fftshift of rfft

# make projections by taking central slices and phase shift
tilt_series = extract_central_slices_rfft(
    dft=dft,
    image_shape=ground_truth_volume.shape,
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True
)  # (..., h, w) rfft
tilt_series = phase_shift_dft_2d(
    dft=tilt_series,
    shifts=ground_truth_shifts,
    image_shape=ground_truth_volume.shape[-2:],
    rfft=True,
    fftshifted=True,
)

# setup optimisation
# assume rotations are fixed, only optimise shifts
predicted_shifts = torch.zeros(
    size=(len(tilt_series), 2), dtype=torch.float32, requires_grad=True
)

projection_model_optimiser = torch.optim.Adam(
    params=[predicted_shifts, ],
    lr=1e-1,
)

# initial reconstruction (for comparison)
initial_dft, weights = insert_central_slices_rfft(
    slices=tilt_series,
    image_shape=ground_truth_volume.shape[-3:],
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True,
)
valid_weights = weights > 1e-3
initial_dft[valid_weights] /= weights[valid_weights]

# optimise
for i in range(1000):
    # Make an intermediate reconstruction
    with torch.no_grad():
        tilt_mask = torch.rand((len(tilt_series))) < 0.10
        _tilt_series = phase_shift_dft_2d(
            dft=tilt_series[tilt_mask],
            shifts=-predicted_shifts[tilt_mask],
            image_shape=ground_truth_volume.shape[-2:],
            rfft=True,
            fftshifted=True,
        )
        intermediate_reconstruction, weights = insert_central_slices_rfft(
            slices=tilt_series,
            image_shape=ground_truth_volume.shape[-3:],
            rotation_matrices=rotation_matrices,
            rotation_matrix_zyx=True,
        )
        valid_weights = weights > 1e-3
        intermediate_reconstruction[valid_weights] /= weights[valid_weights]

    # make projections in remaining 10% of orientations
    projections = extract_central_slices_rfft(
        dft=intermediate_reconstruction,
        image_shape=ground_truth_volume.shape,
        rotation_matrices=rotation_matrices[~tilt_mask],
        rotation_matrix_zyx=True,
    )

    # shift projections so they match 'experimental' data
    projections = phase_shift_dft_2d(
        dft=projections,
        shifts=predicted_shifts[~tilt_mask],
        image_shape=ground_truth_volume.shape[-2:],
        rfft=True,
        fftshifted=True,
    )

    # zero gradients, calculate loss and backpropagate
    projection_model_optimiser.zero_grad()
    loss = torch.mean((tilt_series[~tilt_mask] - projections).abs() ** 2)
    loss.backward()
    projection_model_optimiser.step()
    if i % 20 == 0:
        print(i, loss.item())

# final reconstruction
centered_tilt_series = phase_shift_dft_2d(
    dft=tilt_series,
    shifts=-predicted_shifts,
    image_shape=ground_truth_volume.shape[-2:],
    rfft=True,
    fftshifted=True,
)
dft, weights = insert_central_slices_rfft(
    slices=centered_tilt_series,
    image_shape=ground_truth_volume.shape[-3:],
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True,
)
valid_weights = weights > 1e-3
dft[valid_weights] /= weights[valid_weights]


# visualise results
import napari
viewer = napari.Viewer()


def view_image_stack(data: torch.Tensor, name: str):
    data = torch.fft.ifftshift(data, dim=(-2,))  # ifftshift of rfft
    data = torch.fft.irfftn(data, dim=(-2, -1))
    data = torch.fft.ifftshift(data, dim=(-2, -1))  # recenter real space
    viewer.add_image(data.detach().numpy(), name=name)


def view_volume(data: torch.Tensor, name: str):
    data = torch.fft.ifftshift(data, dim=(-3, -2,)) # actual ifftshift
    data = torch.fft.irfftn(data, dim=(-3, -2, -1))  # to real space
    data = torch.fft.ifftshift(data, dim=(-3, -2, -1))  # center in real space

    # correct for convolution with linear interpolation kernel
    grid = fftfreq_grid(
        image_shape=data.shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=dft.device
    )
    data = data / torch.sinc(grid) ** 2
    viewer.add_image(data.detach().numpy(), name=name)



# calculate best case reconstruction
ideal_ts = phase_shift_dft_2d(
    dft=tilt_series,
    shifts=-ground_truth_shifts,
    image_shape=ground_truth_volume.shape[-2:],
    rfft=True,
    fftshifted=True,
)
ideal_reconstruction, weights = insert_central_slices_rfft(
    slices=ideal_ts,
    image_shape=ground_truth_volume.shape[-3:],
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True,
)
valid_weights = weights > 1e-3
ideal_reconstruction[valid_weights] /= weights[valid_weights]


view_image_stack(tilt_series, name='experimental')
view_image_stack(centered_tilt_series, name='aligned')
viewer.add_image(ground_truth_volume.detach().numpy(), name='ground truth volume')
view_volume(ideal_reconstruction, name='best case reconstruction')
view_volume(initial_dft, name='initial reconstruction')
view_volume(dft, name='final reconstruction')
napari.run()
