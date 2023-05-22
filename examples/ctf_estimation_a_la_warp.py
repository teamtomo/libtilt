"""A contrast transfer function estimation algorithm similar to the
one in Warp (https://doi.org/10.1038/s41592-019-0580-y).
"""
import mrcfile
import napari
import numpy as np
import torch
import torchvision.transforms.functional as TF
import einops

from torch_cubic_spline_grids import CubicBSplineGrid1d, CubicBSplineGrid3d
from libtilt.rotational_averaging import rotational_average_dft_2d
from libtilt.grids import patch_grid
from libtilt.filters import bandpass_filter
from libtilt.ctf.ctf_1d import calculate_ctf as calculate_ctf_1d
from libtilt.ctf.ctf_2d import calculate_ctf as calculate_ctf_2d
from libtilt.fft_utils import spatial_frequency_to_fftfreq

# image parameters
IMAGE_FILE = 'data/TS_01_015_21.0.mrc'
PIXEL_SIZE = 1.1
VOLTAGE = 300  # kV
SPHERICAL_ABERRATION = 2.7  # mm
AMPLITUDE_CONTRAST = 0.10  # fraction

# model parameters
GRID_RESOLUTION = (1, 2, 2)  # (t, h, w)

# fitting parameters
N_PATCHES_PER_BATCH = 20
PATCH_SIDELENGTH = 512
DEFOCUS_RANGE = (1, 12)  # microns
FITTING_RANGE = (40, 10)  # angstroms

# read image
image = torch.tensor(mrcfile.read(IMAGE_FILE)).float()
image = (image - torch.mean(image)) / torch.std(image)
image = einops.repeat(image, 'h w -> 2 h w')  # pretend the target is multi-frame
grid_t, grid_h, grid_w = GRID_RESOLUTION
t, h, w = image.shape
ph, pw = PATCH_SIDELENGTH, PATCH_SIDELENGTH

# extract patches and calculate patch power spectra
patches, patch_centers = patch_grid(
    images=image,
    patch_shape=(1, ph, pw),
    patch_step=(1, ph // 2, pw // 2)
)
patch_ps = torch.abs(torch.fft.rfftn(patches, dim=(-2, -1))) ** 2

# normalise patch centers to [0, 1]
patch_centers = patch_centers / torch.tensor([t - 1, h - 1, w - 1])

# average over dims which are not resolved in desired spatiotemporal grid model
_t = 't' if grid_t > 1 else '1'
_h = 'gh' if grid_h > 1 else '1'
_w = 'gw' if grid_w > 1 else '1'
_reduced_dims = f'{_t} {_h} {_w}'
patch_ps = einops.reduce(
    patch_ps,
    pattern=f't gh gw 1 ph pw -> {_reduced_dims} ph pw',
    reduction='mean',
)

# estimate background in 1D from rotational average of mean power spectrum
mean_power_spectrum = einops.reduce(patch_ps, '... ph pw -> ph pw', reduction='mean')
raps_1d, _ = rotational_average_dft_2d(
    mean_power_spectrum,
    image_shape=(ph, pw),
    rfft=True,
    fftshifted=False,
)
raps_1d = einops.reduce(raps_1d, '... shell -> shell', reduction='mean')

# determine fit range
fftfreq = torch.fft.rfftfreq(PATCH_SIDELENGTH)
lower_limit_fftfreq = spatial_frequency_to_fftfreq(
    1 / FITTING_RANGE[0], spacing=PIXEL_SIZE
)
upper_limit_fftfreq = spatial_frequency_to_fftfreq(
    1 / FITTING_RANGE[1], spacing=PIXEL_SIZE
)
fit_range = torch.logical_and(
    fftfreq > lower_limit_fftfreq, fftfreq < upper_limit_fftfreq
)

# estimate 1D background by fitting a cubic B-spline with 3 control points
background_model = CubicBSplineGrid1d(resolution=3)
background_optimiser = torch.optim.Adam(
    params=background_model.parameters(),
    lr=1
)
raps_1d_in_fit_range = raps_1d[fit_range]
x = torch.linspace(0, 1, steps=len(raps_1d_in_fit_range))
for i in range(200):
    prediction = background_model(x).squeeze()
    loss = torch.mean((torch.log(raps_1d_in_fit_range) - prediction) ** 2)
    loss.backward()
    background_optimiser.step()
    background_optimiser.zero_grad()
    print(loss.item())

# subtract background
background = torch.exp(background_model(x).squeeze())
raps_1d_in_fit_range -= background

# simulate a set of 1D ctf^2 at different defoci to find best match
defocus_step = 0.01
test_defoci = torch.arange(
    start=DEFOCUS_RANGE[0],
    end=DEFOCUS_RANGE[1] + defocus_step,
    step=defocus_step,
)
ctf2 = calculate_ctf_1d(
    defocus=test_defoci,
    voltage=VOLTAGE,
    spherical_aberration=SPHERICAL_ABERRATION,
    amplitude_contrast=AMPLITUDE_CONTRAST,
    b_factor=0,
    phase_shift=0,
    pixel_size=PIXEL_SIZE,
    n_samples=PATCH_SIDELENGTH // 2 + 1,
    oversampling_factor=3,
) ** 2

# find best 1D fit by zero normalised cross correlation (ZNCC)
ctf_fit_range = ctf2[:, fit_range]
ctf_fit_range_norm = torch.linalg.norm(ctf_fit_range, dim=-1, keepdim=True)
ctf_fit_range_normed = ctf_fit_range / ctf_fit_range_norm
raps_1d_fit_range_norm = torch.linalg.norm(raps_1d_in_fit_range)
raps_1d_in_fit_range_normed = raps_1d_in_fit_range / raps_1d_fit_range_norm
zncc = einops.einsum(ctf_fit_range_normed, raps_1d_in_fit_range_normed, 'b i, i -> b')

max_correlation_idx = torch.argmax(zncc)
best_defocus = test_defoci[max_correlation_idx]
print(f'best defocus from 1D fit: {best_defocus}')

# estimate 2D background and subtract from power spectra
raps_2d, _ = rotational_average_dft_2d(
    dft=mean_power_spectrum,
    image_shape=(ph, pw),
    rfft=True,
    fftshifted=False,
    return_2d_average=True,
)
raps_2d[0, 0] = 0
raps_2d = einops.rearrange(raps_2d, 'h w -> 1 1 h w')
bg_estimate_2d = TF.gaussian_blur(raps_2d, kernel_size=25, sigma=10)
bg_estimate_2d = einops.rearrange(bg_estimate_2d, '1 1 h w -> h w')
patch_ps = patch_ps - bg_estimate_2d

# define spatiotemporal defocus field for 2D+t defocus fitting
defocus_grid_data = torch.ones(size=GRID_RESOLUTION) * best_defocus
defocus_field = CubicBSplineGrid3d.from_grid_data(defocus_grid_data)
defocus_optimiser = torch.optim.Adam(
    params=defocus_field.parameters(),
    lr=0.005,
)

# bandpass filter power spectra to fitting range
filter = bandpass_filter(
    low=lower_limit_fftfreq,
    high=upper_limit_fftfreq,
    falloff=0,
    image_shape=(ph, pw),
    rfft=True,
    fftshift=False,
    device=patch_ps.device
)
patch_ps *= filter

# optimise 2d+t defocus model at grid points
for i in range(100):
    _, ph, pw = patch_centers.shape[:3]
    patch_idx = np.random.randint(
        low=(0, 0), high=(ph, pw), size=(N_PATCHES_PER_BATCH, 2)
    )
    patch_idx_h, patch_idx_w = einops.rearrange(patch_idx, 'b idx -> idx b')
    subset_patch_ps = patch_ps[:, patch_idx_h, patch_idx_w]
    subset_patch_centers = patch_centers[:, patch_idx_h, patch_idx_w]

    predicted_patch_defoci = defocus_field(subset_patch_centers)
    predicted_patch_defoci = einops.rearrange(predicted_patch_defoci, '... 1 -> ...')
    simulated_ctfs = calculate_ctf_2d(
        defocus=predicted_patch_defoci,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.10,
        b_factor=0,
        phase_shift=0,
        pixel_size=PIXEL_SIZE,
        image_shape=(PATCH_SIDELENGTH, PATCH_SIDELENGTH),
        astigmatism=0,
        astigmatism_angle=0,
        rfft=True,
        fftshift=False,
    ) ** 2  # (t ph pw h w)
    simulated_ctfs *= filter

    # zero gradients, calculate loss and backpropagate
    defocus_optimiser.zero_grad()
    loss = torch.mean((subset_patch_ps - simulated_ctfs) ** 2).sqrt()
    loss.backward()
    defocus_optimiser.step()

    if i % 5 == 0:
        print(loss.item())
        print(defocus_field.data)

# evaluate defocus over grid
_t, _y, _x = (
    torch.linspace(0, 1, steps=10),
    torch.linspace(0, 1, steps=h // 10),
    torch.linspace(0, 1, steps=w // 10),
)
tt, yy, xx = torch.meshgrid(_t, _y, _x, indexing='ij')
tyx = einops.rearrange([tt, yy, xx], 'tyx t h w -> t h w tyx')
defocus = defocus_field(tyx)

# visualise resulting defocus field
viewer = napari.Viewer()
viewer.add_image(image.mean(dim=0).detach().numpy())
viewer.add_image(defocus.detach().numpy()[..., 0], colormap='PiYG', scale=(10, 10))
napari.run()
