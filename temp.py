import os
import trimesh
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from diffusers import UNet3DConditionModel
from CustomDataset import CustomDataset
from utils import DEVICE, MAX_TIME, OBJ_SIZE, corrupt, obj_to_voxel, voxel_to_obj
import pandas as pd
import multiprocessing as mp
from functools import partial
import shutil


def process_time_step(i, voxel_grid, MAX_TIME, file):
    print(f'Processing {file} {i}...')
    result = corrupt(voxel_grid, torch.tensor(i / MAX_TIME))
    print(f'{file} {i} processed')
    if i % 10 == 0:
        print(f'\tSaving {file} {i}...')
        voxel_to_obj(result.numpy(), f'generated/{file[:-4]}_{i}.obj')
        print(f'\t{file} {i} saved')

    return i, result


if __name__ == '__main__':
    if os.path.exists('generated'):
        shutil.rmtree('generated')
    os.makedirs('generated')
    device = DEVICE
    columns = ['filename', 'original_data'] + \
        [f'noisy_data_{i}' for i in range(MAX_TIME)]
    df = pd.DataFrame(columns=columns)

    rows = []

    for file in os.listdir('obj'):
        if file.lower().endswith('.obj'):
            row = {col: None for col in columns}
            row['filename'] = file

            voxel_grid = torch.from_numpy(obj_to_voxel(f'obj/{file}'))

            row['original_data'] = voxel_grid

            # Create a partial function with the constant arguments
            process_func = partial(process_time_step,
                                   voxel_grid=voxel_grid,
                                   MAX_TIME=MAX_TIME,
                                   file=file)

            # Parallelize the time steps
            with mp.Pool(processes=8) as pool:
                results = pool.map(process_func, range(MAX_TIME))

            # Update the row with the results
            for i, result in results:
                row[f'noisy_data_{i}'] = result

            rows.append(row)

    object_names = list(
        set(filename.split('_')[0] for filename in df['filename']))
    name_idx = {name: idx for idx, name in enumerate(object_names)}
    object_emb = torch.nn.Embedding(len(object_names), 1024).to(device)

    df = pd.DataFrame(rows, columns=columns)

    dataset = CustomDataset(df, name_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = UNet3DConditionModel(
        sample_size=OBJ_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock3D",
            "DownBlock3D",
            "DownBlock3D",
            "DownBlock3D",
        ),
        up_block_types=(
            "UpBlock3D",
            "UpBlock3D",
            "UpBlock3D",
            "UpBlock3D",
        ),
        cross_attention_dim=1024,     # important! enables conditioning on the object embedding
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(10):
        for batch in dataloader:
            # (batch_size, 100, 1, 256, 256, 256)
            noisy_list = batch['noisy_data_list']
            # (batch_size, 1, 256, 256, 256)
            original = batch['original_data']
            name_idx = batch['name_idx']            # (batch_size,)

            batch_size, n_noisy, c, d, h, w = noisy_list.shape

            # (B, 100, 1, 256, 256, 256)
            noisy_list = noisy_list.to(device)
            original = original.to(device)          # (B, 1, 256, 256, 256)
            name_idx = name_idx.to(device)

            # Expand labels to match noisy_list
            name_idx = name_idx.unsqueeze(
                1).expand(-1, n_noisy).reshape(-1)  # (B × 100)

            # Expand original to match noisy_list
            original = original.unsqueeze(
                1).expand(-1, n_noisy, -1, -1, -1, -1)

            # Reshape to (B × 100, C, D, H, W)
            noisy_list = noisy_list.reshape(-1, c, d, h, w)
            original = original.reshape(-1, c, d, h, w)

            # Get embeddings
            encoder_hidden_states = object_emb(name_idx)  # (B × 100, 1024)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(
                1)  # (B × 100, 1, 1024)

            # Forward pass
            output = model(
                noisy_list, encoder_hidden_states=encoder_hidden_states)

            loss = loss_fn(output, original)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss:.6f}')
