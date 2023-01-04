import os
from pathlib import Path
from typing import Tuple, Optional

import einops
import fast_histogram
import mmdf
import mrcfile
import numpy as np
from scipy.stats import special_ortho_group
import typer

cli = typer.Typer(add_completion=False)


def model_to_xyz(model_file: os.PathLike) -> np.ndarray:
    df = mmdf.read(str(model_file))
    return df[['x', 'y', 'z']].to_numpy()


def center_xyz(xyz: np.ndarray) -> np.ndarray:
    return xyz - xyz.mean(axis=0)


def random_rotations(n: int) -> np.ndarray:
    return special_ortho_group.rvs(dim=3, size=n)


def random_positions(n: int, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    z, y, x = [
        np.random.uniform(low=0, high=dim_length, size=n)
        for dim_length in grid_shape
    ]
    return einops.rearrange([z, y, x], 'zyx b -> b zyx')


def random_rotate_xyz(xyz: np.ndarray, n: int) -> np.ndarray:  # (b, n, 3)
    rotations = random_rotations(n)  # (n, 3, 3)
    xyz = einops.rearrange(xyz, 'b xyz -> b 1 xyz 1')
    rotated_xyz = rotations @ xyz  # (b, n, 3, 1)
    return einops.rearrange(rotated_xyz, 'b n xyz 1 -> b n xyz')


def rasterise_3d(
    xyz: np.ndarray, grid_shape: Tuple[int, int, int]
) -> np.ndarray:
    bd, bh, bw = [np.arange(i + 1) - 0.5 for i in grid_shape]
    zyx = xyz[:, ::-1]
    image, _ = np.histogramdd(zyx, bins=[bw, bh, bd])
    return image


def rasterise_2d(
    xy: np.ndarray, grid_shape: Tuple[int, int]
) -> np.ndarray:
    yx = xy[:, ::-1]
    h, w = grid_shape
    image = fast_histogram.histogramdd(
        sample=yx, bins=(h, w), range=[[-0.5, h - 0.5], [-0.5, w - 0.5]]
    )
    return image


def promote_2d_shifts_to_3d(shifts: np.ndarray) -> np.ndarray:
    """Promote arrays of 2D shifts to 3D with zeros in the last column.

    Last dimension of array should be of length 2.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 2)` array of 2D shifts

    Returns
    -------
    output: torch.Tensor
        `(..., 3)` array of 3D shifts with 0 in the last column.
    """
    shifts = np.asarray(shifts)
    if shifts.ndim == 1:
        shifts = einops.rearrange(shifts, 's -> 1 s')
    if shifts.shape[-1] != 2:
        raise ValueError('last dimension must have length 2.')
    shifts = np.pad(shifts, pad_width=((0, 0), (0, 1)))
    return np.squeeze(shifts)


def homogenise_coordinates(coords: np.ndarray) -> np.ndarray:
    """3D coordinates to 4D homogenous coordinates with ones in the last column.

    Parameters
    ----------
    coords: torch.Tensor
        `(..., 3)` array of 3D coordinates

    Returns
    -------
    output: torch.Tensor
        `(..., 4)` array of homogenous coordinates
    """
    return np.c_[coords, np.ones(len(coords))]


def S(shifts: np.ndarray) -> np.ndarray:
    """4x4 matrices for shifts.
    Shifts supplied can be 2D or 3D.
    """
    shifts = np.asarray(shifts)
    if shifts.shape[-1] == 2:
        shifts = promote_2d_shifts_to_3d(shifts)
    shifts = shifts.reshape((-1, 3))
    matrices = einops.repeat(np.eye(4), 'i j -> n i j', n=shifts.shape[0])
    matrices = np.ascontiguousarray(matrices)
    matrices[:, 0:3, 3] = shifts
    return np.squeeze(matrices)


def Rx(angles_degrees: np.ndarray) -> np.ndarray:
    """4x4 matrices for a rotation of homogenous coordinates (xyzw) around the X-axis."""
    angles_degrees = np.asarray(angles_degrees).reshape(-1)
    angles_radians = np.deg2rad(angles_degrees)
    c = np.cos(angles_radians)
    s = np.sin(angles_radians)
    matrices = einops.repeat(np.eye(4), 'i j -> n i j', n=len(angles_degrees))
    matrices = np.ascontiguousarray(matrices)
    matrices[:, 1, 1] = c
    matrices[:, 1, 2] = -s
    matrices[:, 2, 1] = s
    matrices[:, 2, 2] = c
    return np.squeeze(matrices)


def Ry(angles_degrees: np.ndarray) -> np.ndarray:
    """4x4 matrices for a rotation of homogenous coordinates (xyzw) around the Y-axis."""
    angles_degrees = np.asarray(angles_degrees).reshape(-1)
    angles_radians = np.deg2rad(angles_degrees)
    c = np.cos(angles_radians)
    s = np.sin(angles_radians)
    matrices = einops.repeat(np.eye(4), 'i j -> n i j', n=len(angles_degrees))
    matrices = np.ascontiguousarray(matrices)
    matrices[:, 0, 0] = c
    matrices[:, 0, 2] = s
    matrices[:, 2, 0] = -s
    matrices[:, 2, 2] = c
    return np.squeeze(matrices)


def Rz(angles_degrees: np.ndarray) -> np.ndarray:
    """4x4 matrices for a rotation of homogenous coordinates (xyzw) around the Z-axis."""
    angles_degrees = np.asarray(angles_degrees).reshape(-1)
    angles_radians = np.deg2rad(angles_degrees)
    c = np.cos(angles_radians)
    s = np.sin(angles_radians)
    matrices = einops.repeat(np.eye(4), 'i j -> n i j', n=len(angles_degrees))
    matrices = np.ascontiguousarray(matrices)
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    return np.squeeze(matrices)


@cli.command(no_args_is_help=True)
def simulate_tilt_series(
    input_model_file: Path = typer.Option(...),
    output_tilt_series_file: Path = typer.Option(...),
    output_volume_file: Optional[Path] = None,
    output_pixel_spacing: float = 10,
    output_image_shape: Tuple[int, int] = (512, 512),
    n_tilt_images: int = 41,
    n_particles: int = 100,
    minimum_angle: float = -60,
    maximum_angle: float = 60,
    apply_random_2d_shifts: bool = False
):
    # load molecular model, center and rescale
    print(f"loading model from {input_model_file}")
    particle_xyz = model_to_xyz(input_model_file)
    particle_xyz = center_xyz(particle_xyz)
    particle_xyz /= output_pixel_spacing

    # randomly orient particles and place in volume
    print("randomly orienting particles")
    oriented_particles_xyz = random_rotate_xyz(particle_xyz, n=n_particles)
    volume_shape = (*output_image_shape, output_image_shape[0] // 3)
    particle_positions = random_positions(n_particles, grid_shape=volume_shape)

    print("placing oriented particles in 3D volume")
    positioned_particle_xyz = oriented_particles_xyz + particle_positions

    # set up projection geometry
    print("setting up projection geometry of tilt-series")
    tilt = np.linspace(minimum_angle, maximum_angle, num=n_tilt_images)
    shift_std = output_image_shape[0] / 50
    if apply_random_2d_shifts is True:
        shifts = np.random.normal(scale=shift_std, size=(n_tilt_images, 2))
    else:
        shifts = np.array([[0, 0]])

    volume_center = np.array(volume_shape) // 2
    tilt_image_center = np.array(output_image_shape) // 2

    s0 = S(-volume_center)
    r1 = Ry(tilt)
    s1 = S(shifts)
    s2 = S(tilt_image_center)
    T = s2 @ s1 @ r1 @ s0
    Txy = T[..., :2, :]  # only need to keep xy component of output

    # setup particle coordinates for projection
    print('setting up particles for projection')
    positioned_particle_xyz = einops.rearrange(
        positioned_particle_xyz, 'b particle xyz -> (b particle) xyz'
    )
    particle_xyzw = homogenise_coordinates(positioned_particle_xyz)
    particle_xyzw = einops.rearrange(particle_xyzw, 'b xyzw -> b xyzw 1')

    # project into xy plane and render
    print('acquiring images...')
    output_image = np.zeros(shape=(n_tilt_images, *output_image_shape))
    for tilt_idx, projection_matrix in enumerate(Txy):
        projected_xy = projection_matrix @ particle_xyzw
        projected_xy = einops.rearrange(projected_xy, 'b xy 1 -> b xy')
        output_image[tilt_idx] = rasterise_2d(
            projected_xy, grid_shape=output_image_shape
        )
        end = '\r' if tilt_idx < len(Txy) - 1 else None
        print(f'rendered image {tilt_idx + 1}/{n_tilt_images}', end=end)

    # write outputs
    print(f'writing tilt-series to {output_tilt_series_file}')
    mrcfile.write(
        name=output_tilt_series_file,
        data=output_image.astype(np.float32),
        voxel_size=output_pixel_spacing,
        overwrite=True
    )

    if apply_random_2d_shifts is True:
        output_directory = output_tilt_series_file.parent
        shifts_file_name = f'{output_tilt_series_file.stem}_xy_shifts.txt'
        shifts_file = output_directory / shifts_file_name
        print(f'writing shifts to {shifts_file}')
        np.savetxt(
            fname=shifts_file,
            X=shifts[:, ::-1],
            fmt='%04f',
        )

    if output_volume_file is not None:
        print('rendering volume')
        volume = rasterise_3d(positioned_particle_xyz, volume_shape)
        print(f'writing volume to {output_volume_file}')
        mrcfile.write(
            name=output_volume_file,
            data=volume.astype(np.float32),
            voxel_size=output_pixel_spacing,
            overwrite=True,
        )

        volume_directory = output_volume_file.parent
        positions_filename = f'{output_volume_file.stem}_particle_positions_xyz.txt'
        positions_file = volume_directory / positions_filename
        print(f'writing particle positions to {positions_file}')
        np.savetxt(
            fname=positions_file,
            X=positioned_particle_xyz,
            fmt='%04f',
        )


if __name__ == '__main__':
    cli()
