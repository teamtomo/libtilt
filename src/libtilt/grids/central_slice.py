import einops
import torch


def rotated_central_slice(
    rotations: torch.Tensor, sidelength: int
) -> torch.Tensor:
    """Generate an array of rotated central slice coordinates for sampling a 3D image.

    Rotation matrices left multiply `zyx` coordinates in column vectors.
    Coordinates returned are ordered `zyx` to match volumetric array indices.

    Parameters
    ----------
    rotations: torch.Tensor
        `(batch, 3, 3)` array of rotation matrices which rotate zyx coordinates.
    sidelength: int
        Sidelength of cubic volume for which coordinates are generated.

    Returns
    -------
    coordinates: torch.Tensor
        `(batch, n, n, zyx)` array of coordinates where `n == sidelength`.
    """
    if rotations.ndim == 2:
        rotations = einops.rearrange(rotations, 'i j -> 1 i j')
    # generate [x, y, z] coordinates for a central slice
    # the slice spans the XY plane with origin on DFT center_grid
    x = y = torch.arange(sidelength) - (sidelength // 2)
    xx = einops.repeat(x, 'w -> h w', h=sidelength)
    yy = einops.repeat(y, 'h -> h w', w=sidelength)
    zz = torch.zeros(size=(sidelength, sidelength))
    grid = einops.rearrange([xx, yy, zz], 'xyz h w -> 1 h w xyz 1')

    # rotate coordinates
    rotations = einops.rearrange(rotations, 'b i j -> b 1 1 i j')
    grid = einops.rearrange(rotations @ grid, 'b h w coords 1 -> b h w coords')

    # recenter slice on DFT center_grid and flip to zyx
    grid += sidelength // 2
    grid = torch.flip(grid, dims=(-1,))
    return grid
