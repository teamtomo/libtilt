import einops
import mrcfile
import napari
import numpy as np
import torch
from magicgui import magicgui
from napari.types import ImageData

from libtilt.dft_extract_slices import slice_dft
from libtilt.real_space_backprojection import backproject
from libtilt.transformations import Ry, S
from libtilt.coordinate_utils import generate_rotated_slice_coordinates, get_array_coordinates

VOLUME_FILE = 'ribo-16Apx.mrc'

volume = torch.tensor(mrcfile.read(VOLUME_FILE))
volume_shape = torch.tensor(volume.shape)

volume_center = volume_shape // 2
tilt_image_center = volume_center[:2]


def simulate_single_axis_tilt_series(start_angle: float, end_angle: float,
                                     num_images: int) -> ImageData:
    s0 = S(-volume_center)
    r1 = Ry(torch.linspace(start_angle, end_angle, steps=num_images))
    s2 = S(tilt_image_center)
    projection_matrices = s2 @ r1 @ s0
    rotation_matrices = einops.rearrange(projection_matrices[:, :3, :3], 'b i j -> b j i')
    slice_coordinates = generate_rotated_slice_coordinates(rotation_matrices, n=volume_shape[0])
    dft = torch.fft.fftshift(volume, dim=(0, 1, 2))
    dft = torch.fft.fftn(dft, dim=(0, 1, 2))
    dft = torch.fft.fftshift(dft, dim=(0, 1, 2))
    slices = slice_dft(dft, slice_coordinates)
    image_shape = slices.shape[-2:]
    image_center = torch.tensor(image_shape) // 2
    r_max = volume_shape[0] // 2
    ramp_filter = torch.linalg.norm(
        get_array_coordinates(image_shape) - image_center, dim=-1
    ) / r_max
    slices *= ramp_filter
    projections = torch.fft.ifftshift(slices, dim=(1, 2))
    projections = torch.fft.ifftn(projections, dim=(1, 2))
    projections = torch.fft.ifftshift(projections, dim=(1, 2))
    projections = torch.real(projections)
    return np.array(projections)


@magicgui(
    auto_call=True,
    max_angle={'widget_type': 'Slider', 'min': 0, 'max': 90},
    num_images={'widget_type': 'Slider', 'min': 1, 'max': 100}
)
def simulate_tomogram(max_angle: float, num_images: int) -> ImageData:
    s0 = S(-volume_center)
    r1 = Ry(torch.linspace(-max_angle, max_angle, steps=num_images))
    s2 = S(tilt_image_center)
    projection_matrices = s2 @ r1 @ s0
    tilt_series = simulate_single_axis_tilt_series(-max_angle, max_angle, num_images)
    reconstruction = backproject(
        image_stack=torch.tensor(tilt_series),
        projection_matrices=projection_matrices,
        output_dimensions=volume_shape,
    )
    reconstruction -= torch.mean(reconstruction)
    reconstruction /= torch.std(reconstruction)
    return np.array(reconstruction)


viewer = napari.Viewer(ndisplay=3)
volume_layer = viewer.add_image(np.array(volume), name='original 3D volume')
viewer.window.add_dock_widget(simulate_tomogram, name='WBP simulator')
napari.run()
