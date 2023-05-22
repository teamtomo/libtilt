"""A simple projection/backprojection cycle implementation."""
import mrcfile
import torch
from scipy.stats import special_ortho_group

from libtilt.fsc import fsc
from libtilt.projection import project_fourier
from libtilt.backprojection import backproject_fourier

N_IMAGES = 1000
torch.manual_seed(42)

# load a volume and normalise
volume = torch.tensor(mrcfile.read('data/4v6x.mrc'))
volume -= torch.mean(volume)
volume /= torch.std(volume)

# rotation matrices for projection (operate on xyz column vectors)
rotations = torch.tensor(
    special_ortho_group.rvs(dim=3, size=N_IMAGES, random_state=42)
).float()

# make projections
projections = project_fourier(
    volume,
    rotation_matrices=rotations,
    rotation_matrix_zyx=False
)  # (b, h, w)

# reconstruct volume from projections
reconstruction = backproject_fourier(
    images=projections,
    rotation_matrices=rotations,
    rotation_matrix_zyx=False,
    pad=True,
)
reconstruction -= torch.mean(reconstruction)
reconstruction = reconstruction / torch.std(reconstruction)

# fsc
_fsc = fsc(reconstruction, volume)
print(_fsc)

# visualise
import napari

viewer = napari.Viewer()
viewer.add_image(projections.numpy(), name='projections')

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(volume.numpy(), name='ground truth')
viewer.add_image(reconstruction.numpy(), name='reconstruction')
napari.run()
