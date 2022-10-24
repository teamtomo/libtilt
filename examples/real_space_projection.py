import mrcfile
import napari
import numpy as np
import torch

from libtilt.real_space_projection import project
from libtilt.transformations import Ry
from scipy.spatial.transform import Rotation as R

VOLUME_FILE = 'ribo-16Apx.mrc'

volume = torch.tensor(mrcfile.read(VOLUME_FILE))

rotations = torch.tensor(R.random(num=1000).as_matrix()).float()
# rotations = Ry(torch.linspace(-90, 90, steps=100))[:, :3, :3]
projections = project(volume, rotation_matrices=rotations)

viewer = napari.Viewer()
viewer.add_image(np.array(projections))
napari.run()