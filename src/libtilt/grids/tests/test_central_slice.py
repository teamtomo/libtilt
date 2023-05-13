import torch

from libtilt.grids.central_slice import central_slice_grid


def test_central_slice_grid():
    # generate xyz coordinates for a (4, 4, 4) volume
    grid_xyz = central_slice_grid(sidelength=4, zyx=False)

    assert grid_xyz.shape == (4, 4, 3)

    # x coordinates should be 0-3 repeated across rows
    assert torch.all(grid_xyz[..., 0] == torch.tensor([[-2, -1, 0, 1]]))

    # y coordinates should be -2 to 1 repeated across columns
    assert torch.all(grid_xyz[..., 1] == torch.tensor([[-2],
                                                       [-1],
                                                       [0],
                                                       [1]]))

    # all z coordinates should be 0
    assert torch.all(grid_xyz[..., 2] == 0)

    # zyx should be xyz flipped
    grid_zyx = central_slice_grid(sidelength=4, zyx=True)
    assert torch.allclose(torch.flip(grid_xyz, dims=(-1,)), grid_zyx)
