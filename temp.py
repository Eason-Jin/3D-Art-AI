import os
import trimesh
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from diffusers import UNet3DConditionModel
from CustomDataset import CustomDataset
from utils import DEVICE, MAX_TIME, SIZE, corrupt, obj_to_voxel, voxel_to_obj, process_index, parallel_process
import pandas as pd
import multiprocessing as mp
from functools import partial


def process_time_step(i, voxel_grid, MAX_TIME, file):
    print(f'Processing {file} {i}')
    result = corrupt(voxel_grid, torch.tensor(i / MAX_TIME))

    if i % 10 == 0:
        print(f'\tSaving {file} {i}')
        voxel_to_obj(voxel_grid.numpy(), f'obj/{file[:-4]}_{i}.obj')

    return i, result


if __name__ == '__main__':
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

            for i in range(MAX_TIME):
                print(f'Processing {file} {i}')
                result = corrupt(
                    voxel_grid, torch.tensor(i / MAX_TIME))
                row[f'noisy_data_{i}'] = result
                if i % 10 == 0:
                    print(f'\tSaving {file} {i}')
                    voxel_to_obj(result.numpy(),
                                 f'obj/{file[:-4]}_{i}.obj')

            rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

'''
# Create dataset and dataloader
dataset = CustomDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Initialize UNet3DConditionModel
model = UNet3DConditionModel().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    for batch in dataloader:
        # Get original (clean) data
        original_data = batch['original_data'].to(device)
        # Get the list of noisy samples at different timesteps
        noisy_data_list = batch['noisy_data_list'].to(device)

        # Shape: [batch_size, max_time, *spatial_dims]
        batch_size = original_data.shape[0]

        # For each batch, randomly select a timestep to train on
        # This is a common approach in diffusion models
        timesteps = torch.randint(
            0, MAX_TIME, (batch_size,), device=device).long()

        # Get the noisy samples at the selected timesteps for each item in the batch
        batch_noisy_samples = []
        for i in range(batch_size):
            t = timesteps[i]
            batch_noisy_samples.append(noisy_data_list[i, t])

        # Stack the selected noisy samples
        noisy_samples = torch.stack(batch_noisy_samples).unsqueeze(
            1)  # Add channel dimension

        # Forward pass - model predicts the noise to be removed
        # The 'encoder_hidden_states' would be used for conditional generation if needed
        noise_pred = model(
            noisy_samples,
            timesteps,
            encoder_hidden_states=None
        ).sample

        # In diffusion models, we typically train the model to predict
        # either the noise or the clean sample. Here, we're predicting the noise.
        # Calculate the noise (difference between noisy and original)
        noise_target = noisy_samples - original_data.unsqueeze(1)

        # Calculate loss
        loss = loss_fn(noise_pred, noise_target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print epoch results
    print(f'Epoch {epoch}, Average Loss: {loss:.6f}')
'''
