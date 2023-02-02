from pathlib import Path

import einops
import mrcfile
import numpy as np
import torch
import torch.fft as fft

import typer

cli = typer.Typer(add_completion=False)


def calculate_fsc(a: torch.Tensor, b: torch.Tensor):
    """Calculate the Fourier shell correlation between two cubic volume."""
    # specify position of the DC component of the rfft
    rfftn_dc_idx = torch.div(torch.Tensor(tuple(a.shape)), 2,
                             rounding_mode='floor')
    rfftn_dc_idx[-1] = 0

    # calculate DFTs of volume a and b
    a, b = fft.rfftn(a), fft.rfftn(b)
    a, b = fft.fftshift(a, dim=(-3, -2)), fft.fftshift(b, dim=(-3, -2))
    n_shells = a.shape[-1]

    # calculate distance from DC component for each fourier coefficient
    array_indices = torch.Tensor(np.indices(a.shape))  # (c, (z), y, x)
    array_indices = einops.rearrange(array_indices, 'c ... -> ... c')
    distances = torch.linalg.norm(array_indices - rfftn_dc_idx, dim=-1)

    # linearise array and distances then sort on distance to enable splitting
    # into shells in one pass
    a, b = torch.flatten(a), torch.flatten(b)
    distances = torch.flatten(distances)
    distances, distances_sorted_idx = torch.sort(distances, descending=False)

    # find indices for fourier features in each shell
    split_at_idx = torch.searchsorted(distances, torch.arange(n_shells + 1)[1:])
    shell_vector_idx = torch.tensor_split(distances_sorted_idx, split_at_idx)

    # extract shells as separate arrays
    shells_a = [a[idx] for idx in shell_vector_idx]
    shells_b = [b[idx] for idx in shell_vector_idx]

    # calculate the correlation in each shell
    fsc = [
        torch.dot(ai, torch.conj(bi)) / (
            torch.linalg.norm(ai) * torch.linalg.norm(bi))
        for ai, bi
        in zip(shells_a, shells_b)
    ]

    return torch.real(torch.tensor(fsc))


@cli.command(no_args_is_help=True)
def fsc_between_volume_files(volume_a: Path, volume_b: Path):
    a, b = mrcfile.read(volume_a), mrcfile.read(volume_b)
    # a /= np.linalg.norm(a)
    # b /= np.linalg.norm(b)
    # b += np.random.normal(scale=0.001, size=b.shape)
    a, b = torch.tensor(a), torch.tensor(b)
    # from libtilt.fsc.conical import conical_fsc as calculate_fsc
    # fsc = calculate_fsc(a, b, vectors=[1,0,0], aperture_angle=30)
    from libtilt.fsc import fsc as calculate_fsc
    fsc = calculate_fsc(a, b)
    typer.echo(fsc)


if __name__ == '__main__':
    fsc_between_volume_files('input.mrc', 'output.mrc')
    # cli()
