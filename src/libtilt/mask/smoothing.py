import torch
from scipy import ndimage as ndi


def _smooth_binary_image(mask: torch.Tensor, smoothing_radius: float) -> torch.Tensor:
    distances = torch.Tensor(ndi.distance_transform_edt(torch.logical_not(mask)))
    smoothing_idx = torch.logical_and(distances > 0, distances <= smoothing_radius)
    output = torch.clone(mask)
    output[smoothing_idx] = torch.cos((torch.pi / 2) * (distances[smoothing_idx] / smoothing_radius))
    return output
