import einops
import mrcfile
import torch
import torch.nn.functional as F
from scipy.stats import special_ortho_group

from libtilt.fsc import fsc
from libtilt.projection.fourier import project
from libtilt.backprojection.fourier import reconstruct_from_images


def test_projection_backprojection_cycle_rfft():
    N_IMAGES = 5000
    DO_2X_ZERO_PADDING = True
    torch.manual_seed(42)

    # create some volumetric data and normalise
    volume = torch.zeros((32, 32, 32))
    volume[8:24, 8:24, 8:24] = torch.rand(size=(16, 16, 16))
    volume -= torch.mean(volume)
    volume /= torch.std(volume)

    # zero pad
    if DO_2X_ZERO_PADDING:
        p = volume.shape[0] // 4
        volume = F.pad(volume, pad=[p] * 6)

    # rotation matrices for projection (operate on xyz column vectors)
    rotations = torch.tensor(
        special_ortho_group.rvs(dim=3, size=N_IMAGES, random_state=42)
    ).float()

    # take slices
    projections = project(
        volume,
        rotation_matrices=rotations,
        rotation_matrix_zyx=False
    )  # (b, h, w)

    if DO_2X_ZERO_PADDING:
        p = volume.shape[0] // 4
        volume = F.pad(volume, pad=[p] * 6)
        p = projections.shape[-1] // 4
        projections = F.pad(projections, pad=[p] * 4)

    reconstruction = reconstruct_from_images(
        images=projections, rotation_matrices=rotations, rotation_matrix_zyx=False
    )

    # fsc
    _fsc = fsc(reconstruction, volume)
    print(_fsc)
    assert torch.all(_fsc[-16:] > 0.99)
