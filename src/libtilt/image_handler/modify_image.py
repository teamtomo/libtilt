import einops
import torch


def std_reduction(image: torch.Tensor, reduced_axes: torch.Tensor):
    return torch.std(image, dim=reduced_axes, unbiased=False)


def mean_zero(image: torch.Tensor):
    mean_img = einops.reduce(image, "... h w -> ...", "mean")
    mean_img = einops.rearrange(mean_img, "... -> ... 1 1")
    return image - mean_img


def std_one(image: torch.Tensor):
    std_img = einops.reduce(image, "... h w -> ...", reduction=std_reduction)
    std_img = einops.rearrange(std_img, "... -> ... 1 1")
    return image / std_img
