import torch
import numpy as np
import trimesh
import multiprocessing
from skimage.measure import marching_cubes
from transformers import CLIPTokenizer, CLIPTextModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TIME = 100
OBJ_SIZE = 256
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


def obj_to_voxel(filename: str, grid_size: int = OBJ_SIZE) -> np.ndarray:
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


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained(
    "openai/clip-vit-base-patch32").eval().to(DEVICE)


def text_encoder(text_list):
    inputs = tokenizer(text_list, padding=True,
                       truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        embeddings = text_model(
            **inputs).last_hidden_state[:, 0, :]  # CLS token
    return embeddings
