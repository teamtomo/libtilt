import einops
import torch


def central_slice_grid(
    sidelength: int, zyx: bool = False, device: torch.device | None = None
) -> torch.Tensor:
    """Generate a central slice for sampling a 3D DFT.

    The origin of the slice will be at the DFT center.

    Parameters
    ----------
    sidelength: int
        Sidelength of cubic volume for which coordinates are generated.
    zyx: bool
        Whether coordinates should be ordered zyx (`True`) or xyz (`False`).

    Returns
    -------
    coordinates: torch.Tensor
        `(n, n, 3)` array of coordinates where `n == sidelength`.
    """
    x = y = torch.arange(sidelength, device=device) - (sidelength // 2)
    xx = einops.repeat(x, 'w -> h w', h=sidelength)
    yy = einops.repeat(y, 'h -> h w', w=sidelength)
    zz = torch.zeros(size=(sidelength, sidelength), device=device)
    grid = einops.rearrange([xx, yy, zz], 'xyz h w -> h w xyz')
    if zyx is True:
        grid = torch.flip(grid, dims=(-1,))
    return grid
