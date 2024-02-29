import torch

from libtilt.projection.project_real import project_image_real, project_volume_real


def test_real_space_projection_3d():
    volume_shape = (2, 10, 10)
    volume = torch.arange(2*10*10).reshape(volume_shape).float()
    rotation_matrix = torch.eye(3).reshape(1, 3, 3)
    projection = project_volume_real(volume, rotation_matrices=rotation_matrix)
    assert torch.allclose(projection.squeeze(), torch.sum(volume, dim=0))


def test_real_space_projection_2d():
    image_shape = (8, 12)
    image = torch.arange(8 * 12).reshape(image_shape).float()
    rotation_matrix = torch.eye(2).reshape(1, 2, 2)
    projection = project_image_real(image, rotation_matrices=rotation_matrix)
    assert torch.allclose(projection.squeeze(), torch.sum(image, dim=0))
