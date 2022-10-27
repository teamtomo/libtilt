import mrcfile
import napari
import numpy as np
import torch

from libtilt.projection.real import project
from scipy.spatial.transform import Rotation as R

VOLUME_FILE = 'ribo-16Apx.mrc'

volume = torch.tensor(mrcfile.read(VOLUME_FILE))

rotations = torch.tensor(R.random(num=1000).as_matrix(), dtype=torch.float)
projections = project(volume, rotation_matrices=rotations)

viewer = napari.Viewer()
viewer.add_image(np.array(projections))
napari.run()