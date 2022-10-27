import torch


def apply_ramp_filter(images: torch.Tensor):
    r_max = images.shape[-1] // 2
    ramp_filter = torch.linalg.norm(get_array_coordinates(images.shape[-2:]) - image_center,
                                    dim=-1) / r_max