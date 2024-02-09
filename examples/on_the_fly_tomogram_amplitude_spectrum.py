import mrcfile
import numpy as np
import torch
import time

from libtilt.rotational_averaging import rotational_average_dft_3d
from libtilt.patch_extraction import extract_cubes

# https://zenodo.org/records/6504891
TOMOGRAM_FILE = '/Users/burta2/Downloads/01_10.00Apx.mrc'
N_CUBES = 20
SIDELENGTH = 128

tomogram = torch.tensor(mrcfile.read(TOMOGRAM_FILE), dtype=torch.float32)

# sample some points in the volume (could be smarter and sample from masked regions)
d, h, w = tomogram.shape
lower_bound = SIDELENGTH // 2
z = np.random.uniform(low=lower_bound, high=d - lower_bound, size=N_CUBES)
y = np.random.uniform(low=lower_bound, high=h - lower_bound, size=N_CUBES)
x = np.random.uniform(low=lower_bound, high=w - lower_bound, size=N_CUBES)

zyx = torch.tensor(np.stack([z, y, x], axis=-1), dtype=torch.float32)

# start timing here
t0 = time.time()

# extract cubes at those points
cubes = extract_cubes(image=tomogram, positions=zyx, sidelength=SIDELENGTH)

# calculate amplitude spectra and rotational average
cubes_amplitudes = torch.fft.rfftn(cubes, dim=(-3, -2, -1)).abs().pow(2)
raps, bins = rotational_average_dft_3d(cubes_amplitudes, rfft=True, fftshifted=False,
                                       image_shape=(SIDELENGTH, SIDELENGTH, SIDELENGTH))
raps = torch.mean(raps, dim=0)  # average over each of 10 cubes

# end timing here
t1 = time.time()

print(f"Elapsed time: {t1 - t0:.2f} seconds")

# plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(np.log(raps.numpy()))
plt.show()
