import einops
import torch
import torch.nn.functional as F

from .coordinate_utils import (
    get_array_coordinates,
    array_coordinates_to_grid_sample_coordinates,
)


def project(volume: torch.Tensor, rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Make 2D projections of a 3D volume in specific orientations."""
    volume_shape = torch.tensor(volume.shape)
    ps = padded_sidelength = int(3 ** 0.5 * torch.max(volume_shape))
    shape_difference = torch.abs(padded_sidelength - volume_shape)
    padding = torch.empty(size=(3, 2), dtype=torch.int16)
    padding[:, 0] = shape_difference // 2
    padding[:, 1] = shape_difference - (shape_difference // 2)
    torch_padding = torch.flip(padding, dims=(0,))  # dhw -> whd for torch.nn.functional.pad
    torch_padding = einops.rearrange(torch_padding, 'whd pad -> (whd pad)')
    volume = F.pad(volume, pad=tuple(torch_padding), mode='constant', value=0)
    padded_volume_shape = (ps, ps, ps)
    volume_coordinates = get_array_coordinates(grid_dimensions=padded_volume_shape)
    volume_coordinates -= padded_sidelength // 2  # (d, h, w, zyx)
    volume_coordinates = torch.flip(volume_coordinates, dims=(-1,))  # (d, h, w, xyz)
    volume_coordinates = einops.rearrange(volume_coordinates, 'd h w xyz -> d h w xyz 1')

    def _project_volume(rotation_matrix) -> torch.Tensor:
        rotated_coordinates = rotation_matrix @ volume_coordinates
        rotated_coordinates += padded_sidelength // 2
        rotated_coordinates = einops.rearrange(rotated_coordinates, 'd h w xyz 1 -> 1 d h w xyz')
        rotated_coordinates = torch.flip(rotated_coordinates, dims=(-1,))  # xyz -> zyx
        rotated_coordinates = array_coordinates_to_grid_sample_coordinates(
            rotated_coordinates, array_shape=padded_volume_shape
        )
        samples = F.grid_sample(
            input=einops.rearrange(volume, 'd h w -> 1 1 d h w'),  # add batch and channel dims
            grid=rotated_coordinates,
            mode='bilinear',  # trilinear for volume
            padding_mode='zeros',
            align_corners=False,
        )
        return einops.reduce(samples, '1 1 d h w -> h w', reduction='sum')

    yl, yh = padding[1, 0], -padding[1, 1]
    xl, xh = padding[2, 0], -padding[2, 1]
    images = [_project_volume(matrix)[yl:yh, xl:xh] for matrix in rotation_matrices]
    return torch.stack(images, dim=0)
