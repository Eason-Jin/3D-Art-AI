import os
import trimesh
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from diffusers import UNet3DConditionModel
from CustomDataset import CustomDataset
from utils import DEVICE, MAX_TIME, OBJ_SIZE, corrupt, obj_to_voxel, voxel_to_obj, text_encoder
import pandas as pd
import multiprocessing as mp
from functools import partial
import shutil
from tqdm import tqdm
import sys
import ast


def process_time_step(i, voxel_grid, MAX_TIME, filename):
    # print(f'Processing {filename} {i}...')
    result = corrupt(voxel_grid, torch.tensor(i / MAX_TIME))
    # print(f'{filename} {i} processed')
    if i % 10 == 0:
        # print(f'\tSaving {filename} {i}...')
        voxel_to_obj(result.numpy(), f'generated/{filename}_{i}.obj')
        # print(f'\t{filename} {i} saved')

    return i, result


if __name__ == '__main__':
    if os.path.exists('generated'):
        shutil.rmtree('generated')
    os.makedirs('generated')
    device = DEVICE
    columns = ['filename', 'original_data', 'noise_level', 'noisy_data']
    df = pd.DataFrame(columns=columns)

    rows = []

    for file in os.listdir('obj'):
        if file.lower().endswith('.obj'):
            voxel_grid = torch.from_numpy(obj_to_voxel(f'obj/{file}'))
            filename = file[:-4]
            # Create a partial function with the constant arguments
            process_func = partial(process_time_step,
                                   voxel_grid=voxel_grid,
                                   MAX_TIME=MAX_TIME,
                                   filename=filename)

            # Parallelize the time steps
            with mp.Pool(processes=8) as pool:
                results = pool.map(process_func, range(MAX_TIME))

            # Update the row with the results
            for i, result in results:
                new_row = {col: None for col in columns}
                new_row.update({
                    'filename': filename,
                    'original_data': voxel_grid,
                    'noise_level': i,
                    'noisy_data': result,
                })
                rows.append(new_row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('data.csv', index=False)

    dataset = CustomDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = UNet3DConditionModel(
        sample_size=OBJ_SIZE+1,
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
        cross_attention_dim=512,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    model.train()

    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            noisy_voxel = batch['noisy_voxel'].to(
                device)            # [B, 1, D, H, W]
            clean_voxel = batch['clean_voxel'].to(
                device)            # [B, 1, D, H, W]
            timestep = batch['noise_level'].to(device)  # [B]
            descriptions = batch['description']

            text_emb = text_encoder(descriptions).to(device)    # [B, seq_len, D]

            # Predict noise from noisy voxel
            pred = model(sample=noisy_voxel, timestep=timestep,
                         encoder_hidden_states=text_emb).sample

            loss = loss_fn(pred, clean_voxel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.6f}")
