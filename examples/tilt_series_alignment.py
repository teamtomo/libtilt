"""A toy implementation of a tilt-series alignment routine using projection matching."""
import numpy as np
import torch

from libtilt.backprojection import backproject_fourier
from libtilt.projection import project_fourier
from libtilt.shapes import sphere
from libtilt.shift.fourier_shift import fourier_shift_2d
from libtilt.transformations import Ry

# simulate a volume with two spheres
ground_truth_volume = sphere(
    radius=4, image_shape=(32, 32, 32), center=(16, 16, 8), smoothing_radius=2
)
ground_truth_volume = ground_truth_volume + sphere(
    radius=4, image_shape=(32, 32, 32), center=(16, 16, 24), smoothing_radius=2
)

# simulate a tilt-series from the volume
rotation_matrices = Ry(torch.linspace(-60, 60, steps=41), zyx=True)[:, :3, :3].float()
ground_truth_shifts = torch.as_tensor(np.random.normal(loc=0, scale=1.5, size=(41, 2))).float()
ground_truth_shifts[20] = 0
tilt_series = project_fourier(
    ground_truth_volume,
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True
)
experimental_tilt_series = fourier_shift_2d(tilt_series, shifts=ground_truth_shifts)
experimental_tilt_series = experimental_tilt_series - torch.mean(
    experimental_tilt_series, dim=(-2, -1), keepdim=True)
experimental_tilt_series = experimental_tilt_series / torch.std(
    experimental_tilt_series, dim=(-2, -1), keepdim=True)

# calculate best case reconstruction
ideal_ts = fourier_shift_2d(
    images=experimental_tilt_series,
    shifts=-ground_truth_shifts,
)
ideal_reconstruction = backproject_fourier(
    images=ideal_ts,
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True,
)

# setup optimisation
# assume rotations are fixed, only optimise shifts
predicted_shifts = torch.zeros(size=(len(tilt_series), 2), dtype=torch.float32,
                               requires_grad=True)

projection_model_optimiser = torch.optim.Adam(
    params=[predicted_shifts, ],
    lr=0.1,
)

for i in range(250):
    # Make an intermediate reconstruction from 90% of the data
    with torch.no_grad():
        tilt_mask = torch.rand((len(tilt_series))) < 0.90
        _tilt_series = fourier_shift_2d(
            images=experimental_tilt_series[tilt_mask],
            shifts=-predicted_shifts[tilt_mask]
        )
        intermediate_reconstruction = backproject_fourier(
            images=_tilt_series,
            rotation_matrices=rotation_matrices[tilt_mask],
            rotation_matrix_zyx=True,
        )

    # make projections in remaining 10% of orientations
    projections = project_fourier(
        volume=intermediate_reconstruction,
        rotation_matrices=rotation_matrices[~tilt_mask],
        rotation_matrix_zyx=True,
    )
    projections = projections - torch.mean(projections, dim=(-2, -1), keepdim=True)
    projections = projections / torch.std(projections, dim=(-2, -1), keepdim=True)

    # shift projections so they match 'experimental' data
    projections = fourier_shift_2d(projections, shifts=predicted_shifts[~tilt_mask])

    # zero gradients, calculate loss and backpropagate
    projection_model_optimiser.zero_grad()
    projection_loss = torch.mean(
        (experimental_tilt_series[~tilt_mask] - projections) ** 2).sqrt()
    centering_loss = 100 * torch.mean(predicted_shifts[20].abs())
    loss = projection_loss + centering_loss
    loss.backward()
    projection_model_optimiser.step()
    if i % 20 == 0:
        print(i, loss.item())

# final reconstruction
centered_tilt_series = fourier_shift_2d(experimental_tilt_series, shifts=-predicted_shifts)
final_reconstruction = backproject_fourier(
    images=centered_tilt_series,
    rotation_matrices=rotation_matrices,
    rotation_matrix_zyx=True,
)

import napari

viewer = napari.Viewer()
viewer.add_image(tilt_series.detach().numpy(), name='ideal - zero shifts')
viewer.add_image(experimental_tilt_series.detach().numpy(), name='experimental')
viewer.add_image(centered_tilt_series.detach().numpy(), name='aligned')
viewer.add_image(ground_truth_volume.detach().numpy(), name='ground truth volume')
viewer.add_image(final_reconstruction.detach().numpy(), name='reconstruction')
viewer.add_image(ideal_reconstruction.detach().numpy(), name='ideal reconstruction')
napari.run()
