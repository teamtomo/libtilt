# real space

`libtilt.real_space` contains functions for making 2D projections from 3D volumes
and performing 3D reconstruction from 2D projection images by simple backprojection.


## outline
In this example we will use *libtilt* to

- **make a set of projection images every 3° over a range of ±60°**
- **reconstruct a 3D volume from these projection images**




## download example data
First, let's download some example data.

<div style="text-align: center;">
<img src="https://user-images.githubusercontent.com/7307488/198159141-b38f6386-64bd-4923-a49b-cb417215553d.gif" alt="volume visualisation" width="200">
</div>

We have chosen [EMD-13234](https://www.ebi.ac.uk/emdb/EMD-13234), a 3D reconstruction of a ribosome 
inside a *Mycoplasma pneumoniae* cell from cryo-electron tomography data.

```python
from pathlib import Path
import requests

URL = "https://www.ebi.ac.uk/emdb/structures/EMD-13234/map/emd_13234.map.gz"
VOLUME_FILE = "emd_13234.map.gz"

if not Path(VOLUME_FILE).exists():
    open(VOLUME_FILE, "wb").write(requests.get(URL).content)
```

## read the data
The data is in `.mrc` format and can be read into memory using 
[*mrcfile*](https://mrcfile.readthedocs.io/).

```python
import mrcfile

volume = mrcfile.read(VOLUME_FILE)
```

This yields a 3D *NumPy* array of shape `(d, h, w)` where `d`=`h`=`w`=`336`.

## make a series of projections around one axis
Making a projection using `libtilt.real_space.project` involves 

1. rotating a 3D grid of coordinates (shape `(d, h, w, 3)`)
2. sampling the volume on the rotated grid 
3. summing along the depth dimension of the `(d, h, w)` grid.

The grid of `xyz` coordinates is rotated by left multiplication with a rotation matrix, which must
be supplied to the `project` function. We can generate a set of rotation matrices for rotations
around the y-axis using the
[`scipy.spatial.transform.Rotation`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
class.

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from libtilt.real_space import project

angles = np.linspace(-60, 60, num=41)
rotation_matrices = R.from_euler('y', angles=angles, degrees=True).as_matrix()
projections = project(volume, rotation_matrices=rotation_matrices)
```

<div style="text-align: center;">
<img src="https://user-images.githubusercontent.com/7307488/198159150-5420ca02-6447-4e67-b749-727f3346672f.gif" alt="tilt-series" width="200">
</div>

## perform a reconstruction by backprojection

<div style="text-align: center;">
<img src="https://user-images.githubusercontent.com/7307488/198159154-8eb6cd47-936f-4373-9e37-f3c69bdf9e8b.gif" alt="tomogram" width="200">
</div>