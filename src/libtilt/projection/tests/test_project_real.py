import torch

from libtilt.projection.project_real import project_in_real_space


def test_real_space_projection():
    volume_shape = (2, 10, 10)
    volume = torch.arange(2*10*10).reshape(volume_shape).float()
    rotation_matrix = torch.eye(3).reshape(1, 3, 3)
    projection = project_in_real_space(volume, rotation_matrices=rotation_matrix)
    assert torch.allclose(projection.squeeze(), torch.sum(volume, dim=0))

