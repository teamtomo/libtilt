import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import einops
import fast_histogram
import mmdf
import mrcfile
import napari
import numpy as np
from einops import rearrange
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


def random_xyz_positions(n: int, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    z, y, x = [
        np.random.uniform(low=0, high=dim_length, size=n)
        for dim_length in grid_shape
    ]
    return einops.rearrange([z, y, x], 'zyx b -> b zyx')[:, ::-1]


def rotate_xyz(
    xyz: np.ndarray, n: int, rotations: np.ndarray = None
) -> np.ndarray:  # (b, n, 3)
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

@dataclass
class ParticleSystem:
    atoms: np.ndarray  # (n, 3)
    particle_positions: np.ndarray  # (n, 3)
    particle_orientations: np.ndarray  # (n, 3, 3)


def generate_particle_system(
    particle_xyz: np.ndarray,
    volume_shape: Tuple[int, int, int],
    n_particles: int,  # (b )
) -> ParticleSystem:  # (b * n_atoms, 3) particle system
    particle_orientations = random_rotations(n=n_particles)
    oriented_particle_xyz = rotate_xyz(
        particle_xyz, n=n_particles, rotations=particle_orientations
    )
    particle_positions = random_xyz_positions(n=n_particles, grid_shape=volume_shape)
    positioned_atoms = oriented_particle_xyz + particle_positions
    all_atoms = rearrange(positioned_atoms, 'b n xyz -> (b n) xyz')
    return ParticleSystem(
        atoms=all_atoms,
        particle_positions=particle_positions,
        particle_orientations=particle_orientations,
    )


def reshuffle_stacked_points_for_napari(points: np.ndarray):
    """b n d -> (b n) d+1"""
    new_shape = np.array(points.shape)
    new_shape[-1] += 1
    new_points = np.zeros(new_shape)
    for idx, _points in enumerate(points):
        new_points[idx, :, 0] = idx
        new_points[idx, :, 1:] = _points
    new_points = einops.rearrange(new_points, 'b n d -> (b n) d')
    new_points[:, 1:] = new_points[:, :0:-1]
    return new_points


@cli.command(no_args_is_help=True)
def example_local_fourier_reconstruction(
    input_model_file: Path = typer.Option(...),
    volume_shape: Tuple[int, int, int] = (200, 512, 512),
    volume_pixel_spacing: float = 10,
    output_volume_file: Optional[Path] = None,
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
    particle_xyz /= volume_pixel_spacing

    # generate particle system (particles in a box)
    print("generating particle system...")
    particle_system = generate_particle_system(
        particle_xyz=particle_xyz, volume_shape=volume_shape, n_particles=n_particles
    )

    # set up projection geometry
    print("setting up projection geometry of tilt-series")
    tilt = np.linspace(minimum_angle, maximum_angle, num=n_tilt_images)
    shift_std = volume_shape[0] / 50
    if apply_random_2d_shifts is True:
        shifts = np.random.normal(scale=shift_std, size=(n_tilt_images, 2))
    else:
        shifts = np.array([[0, 0]])

    volume_shape_zyx = np.array(volume_shape)
    volume_shape_xyz = volume_shape_zyx[::-1]
    volume_center = volume_shape_xyz // 2
    tilt_image_center = volume_shape_xyz[:2] // 2

    # these matrices are operating on the tomogram in real space
    # tilt geometry is extrinsic XYZ
    s0 = S(-volume_center)
    r1 = Rx(0)  # non-perpendicularity of tilt-series
    r2 = Ry(tilt)  # stage tilt
    r3 = Rz(0)  # in plane rotation
    s4 = S(shifts)  # shifts in camera plane (tracking)
    s5 = S(tilt_image_center)
    T = s5 @ s4 @ r3 @ r2 @ r1 @ s0  # (b, 4, 4)
    Txy = T[..., :2, :]  # (b, 2, 4) - only need xy coordinate of output

    # setup particle coordinates for projection
    print('setting up particles for projection')
    all_atoms_xyzw = homogenise_coordinates(particle_system.atoms)
    all_atoms_xyzw = einops.rearrange(all_atoms_xyzw, 'b xyzw -> b xyzw 1')

    # project into xy plane and render
    print('acquiring images...')
    tilt_series = np.zeros(shape=(n_tilt_images, *volume_shape[-2:]))
    for tilt_idx, projection_matrix in enumerate(Txy):
        projected_xy = projection_matrix @ all_atoms_xyzw
        projected_xy = einops.rearrange(projected_xy, 'b xy 1 -> b xy')
        tilt_series[tilt_idx] = rasterise_2d(
            projected_xy, grid_shape=volume_shape[-2:]
        )
        end = '\r' if tilt_idx < len(Txy) - 1 else None
        print(f'rendered image {tilt_idx + 1}/{n_tilt_images}', end=end)

    print('projecting particle positions into 2d...')
    particle_positions_xyzw = homogenise_coordinates(
        particle_system.particle_positions
    )
    particle_positions_xyzw = einops.rearrange(
        particle_positions_xyzw, 'b xyzw -> b xyzw 1'
    )
    Txy_broadcastable = einops.rearrange(Txy, 'b i j -> b 1 i j')
    projected_particle_positions = Txy_broadcastable @ particle_positions_xyzw
    projected_particle_positions = einops.rearrange(
        projected_particle_positions, 'b n xy 1 -> b n xy'
    )

    # visualise projected positions on tilt-series in napari
    viewer = napari.Viewer()
    viewer.add_image(tilt_series)
    # gotta reshuffle those points for visualisation
    reshuffled_points = reshuffle_stacked_points_for_napari(projected_particle_positions)
    viewer.add_points(reshuffled_points, face_color='cornflowerblue')
    napari.run()

    # extract local image windows at projected particle positions
    local_image_windows = None

    # do reconstruction in fourier space...
    # the rotation to apply to the slice is the inverse of the
    # extrinsic rotation of our tilt geometry
    rotation_fourier = np.linalg.pinv(T[:, :3, :3])

    def do_reconstruction(*args):
        pass

    reconstructed_cubes = do_reconstruction(local_image_windows, rotation_fourier)



if __name__ == '__main__':
    # cli()
    example_local_fourier_reconstruction(
        input_model_file='4v6x-ribo.cif',
        n_particles=10,
        n_tilt_images=41,
        minimum_angle=-60,
        maximum_angle=60,
    )
