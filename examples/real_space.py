from pathlib import Path
import requests

URL = "https://www.ebi.ac.uk/emdb/structures/EMD-13234/map/emd_13234.map.gz"
VOLUME_FILE = "emd_13234.map.gz"
DOWNSAMPLE_VOLUME = True

if not Path(VOLUME_FILE).exists():
    open(VOLUME_FILE, "wb").write(requests.get(URL).content)

###

import torch
import mrcfile

volume = mrcfile.read(VOLUME_FILE)
volume = torch.tensor(volume)
volume.to(device=torch.device('mps'))

###

if DOWNSAMPLE_VOLUME is True:
    import torch.nn.functional as F
    volume = F.avg_pool3d(volume[None, None, ...], 4).squeeze()


###

import napari
import numpy as np
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(np.array(volume))
napari.run()

###
import numpy as np
from scipy.spatial.transform import Rotation as R
from libtilt.real_space import project

angles = np.linspace(-60, 60, num=41)
rotation_matrices = R.from_euler('y', angles=angles, degrees=True).as_matrix()
projections = project(volume, rotation_matrices=rotation_matrices)

###

import napari
import numpy as np
viewer = napari.Viewer()
viewer.add_image(np.array(projections))
napari.run()

###

from libtilt.real_space import backproject
from libtilt.utils.transformations import S

r4x4 = torch.zeros(size=(len(rotation_matrices), 4, 4))
r4x4[:, :3, :3] = torch.tensor(np.linalg.inv(rotation_matrices))
r4x4[:, 3, 3] = 1
specimen_center = torch.tensor(volume.shape) // 2
projection_matrices = S(specimen_center) @ r4x4 @ S(-specimen_center)

reconstruction = backproject(
    projection_images=projections,
    projection_matrices=projection_matrices,
    output_dimensions=volume.shape,
)

###

import napari
import numpy as np
viewer = napari.Viewer()
viewer.add_image(np.array(reconstruction))
napari.run()