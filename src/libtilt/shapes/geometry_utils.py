import torch


def _angle_between_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a can optionally be an nD stack, return in degrees"""
    a = torch.as_tensor(a)
    b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
    n_stack_dims = a.ndim - 1
    stack_dims = _unique_characters(n_stack_dims)
    angles = torch.arccos(torch.einsum(f'{stack_dims}v,v->{stack_dims}', a, b))
    return torch.rad2deg(angles)


def _unique_characters(n: int) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz"
    return chars[:n]
