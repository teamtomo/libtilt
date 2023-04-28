import torch
import einops

from libtilt.grids.central_slice import rotated_central_slice


def test_generate_rotated_slice_coordinates():
    # generate an unrotated slice for a 4x4x4 volume.
    rotation = torch.eye(3)
    slice_coordinates = rotated_central_slice(rotation, sidelength=4)

    assert slice_coordinates.shape == (1, 4, 4, 3)
    slice_coordinates = einops.rearrange(slice_coordinates, '1 i j zyx -> i j zyx')

    # all z coordinates should be in the middle of the volume, slice is unrotated
    assert torch.all(slice_coordinates[..., 2] == 2)

    # y coordinates should be 0-3 repeated across columns
    assert torch.all(slice_coordinates[..., 1] == torch.tensor([[0],
                                                                [1],
                                                                [2],
                                                                [3]]))

    # x coordinates should be 0-3 repeated across rows
    assert torch.all(slice_coordinates[..., 0] == torch.tensor([[0, 1, 2, 3]]))

