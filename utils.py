import torch
import numpy as np
import trimesh
import multiprocessing
from skimage.measure import marching_cubes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TIME = 100
SIZE = 64
random = np.random.RandomState(42)


def corrupt(voxel_grid: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """
    Adds noise to a voxel grid in a reproducible but varied way.

    Args:
        voxel_grid: Input voxel grid (shape: [B, C, D, H, W] or [D, H, W])
        amount: Strength of corruption (0=almost no noise, 1=a lot of noise)

    Returns:
        Noisy voxel grid (same shape as input)
    """

    # Generate correlated noise patterns
    base_noise = random.randn(*voxel_grid.shape)

    # Apply noise with structure preservation
    corrupted = voxel_grid + amount * base_noise * (0.5 + 0.5*voxel_grid)

    # Clamp and convert back to original dtype
    return np.clip(corrupted, 0, 1)


def obj_to_voxel(filename: str, grid_size: int = 8) -> np.ndarray:
    """
    Convert an OBJ file to a voxel grid.

    Args:
        filename: Path to the OBJ file
        grid_size: Size of the voxel grid (grid_size x grid_size x grid_size)

    Returns:
        A 3D numpy array representing the voxel grid (1 = occupied, 0 = empty)
    """
    # Load the mesh
    mesh = trimesh.load(filename)

    # Get the bounding box and scale the mesh to fit in the unit cube
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1 / np.max(mesh.extents))

    # Voxelize the mesh
    voxels = mesh.voxelized(pitch=2.0 / grid_size).fill()
    # Convert to binary array and return
    return voxels.matrix.astype(np.uint8)


def voxel_to_obj(voxel_grid: np.ndarray, filename: str) -> None:
    """
    Convert a voxel grid to an OBJ file using marching cubes.

    Args:
        voxel_grid: 3D numpy array representing the voxel grid
        filename: Path to save the OBJ file
        threshold: Threshold for the marching cubes algorithm
    """
    # Pad the voxel grid to ensure watertight mesh
    padded_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)

    # Use marching cubes to convert voxels to a mesh
    vertices, faces, _, _ = marching_cubes(padded_grid)

    # Scale vertices to original size (remove padding and normalize)
    vertices = (vertices - 1) / (voxel_grid.shape[0] - 1)

    # Create and export the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)


def process_index(i, voxel_grid, file):
    print(f'Processing {file} {i}')
    row = {}
    row[f'noisy_data_{i}'] = corrupt(voxel_grid, torch.tensor(i / MAX_TIME))

    if i % 10 == 0:
        print(f'\t\tSaving {file} {i}')
        voxel_to_obj(voxel_grid.numpy(), f'obj/{file[:-4]}_{i}.obj')

    return i, row


def parallel_process(file, voxel_grid, columns):
    row = {col: None for col in columns}
    row['filename'] = file
    row['original_data'] = voxel_grid

    # Create a pool of workers to process each index i in parallel
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            process_index, [(i, voxel_grid, file) for i in range(MAX_TIME)])

    # Sort results by index to ensure the correct order of noisy data
    results.sort(key=lambda x: x[0])

    # Fill the row with results
    for _, result in results:
        row.update(result)

    return row
