import einops
import numpy as np
import torch
import torch.fft as fft


def fsc(a: torch.Tensor, b: torch.Tensor):
    """Calculate the Fourier shell correlation between two cubic volume."""
    # specify position of the DC component of the rfft
    rfftn_dc_idx = torch.div(torch.Tensor(tuple(a.shape)), 2, rounding_mode='floor')
    rfftn_dc_idx[-1] = 0

    # calculate DFTs of volume a and b and store length of half-transform
    a, b = fft.rfftn(a), fft.rfftn(b)
    a, b = fft.fftshift(a, dim=(-3, -2)), fft.fftshift(b, dim=(-3, -2))
    half_transform_length = a.shape[0] // 2

    # calculate distance from DC component for each fourier feature
    array_indices = torch.Tensor(np.indices(a.shape)) # (c, (z), y, x)
    array_indices = einops.rearrange(array_indices, 'c ... -> ... c')
    distances = torch.linalg.norm(array_indices - rfftn_dc_idx, dim=-1)

    # linearise array and sort to enable one pass split into separate shells
    a, b = torch.flatten(a), torch.flatten(b)
    distances = torch.flatten(distances)
    distances, distances_sorted_idx = torch.sort(distances, descending=False)

    # find indices for fourier features in each shell
    split_points = torch.linspace(
        start=0.5, end=half_transform_length + 0.5, steps=half_transform_length + 1
    )
    split_indices = torch.searchsorted(distances, split_points)
    shell_vector_idx = torch.tensor_split(distances_sorted_idx, split_indices)[:-1]

    # extract shells as separate arrays
    shells_a = [a[idx] for idx in shell_vector_idx]
    shells_b = [b[idx] for idx in shell_vector_idx]

    # calculate the correlation in each shell
    fsc = [
        torch.dot(ai, torch.conj(bi)) / (torch.linalg.norm(ai) * torch.linalg.norm(bi))
        for ai, bi
        in zip(shells_a, shells_b)
    ]

    return torch.real(torch.tensor(fsc))
