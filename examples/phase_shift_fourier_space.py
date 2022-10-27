from pathlib import Path

import mrcfile
import numpy as np
import torch
import napari

from libtilt.shift.phase_shift import phase_shift_images_2d

volume_file = Path(__file__).parent / 'ribo-16Apx.mrc'

volume = torch.tensor(mrcfile.read(volume_file))
image = torch.mean(volume, axis=0)

shifts = torch.normal(mean=0, std=5, size=(10, 2))
shifted = phase_shift_images_2d(image, shifts)

viewer = napari.Viewer()
viewer.add_image(np.array(image), name='original')
viewer.add_image(np.array(shifted), name='shifted')
napari.run()
