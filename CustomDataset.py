from torch.utils.data import Dataset
import torch
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        clean_voxel = torch.tensor(row['original_data'], dtype=torch.float32).unsqueeze(0)
        noisy_voxel = torch.tensor(row['noisy_data'], dtype=torch.float32).unsqueeze(0)
        noise_level = row['noise_level']
        description = row['filename']

        return {
            'clean_voxel': clean_voxel,
            'noisy_voxel': noisy_voxel,
            'noise_level': noise_level,
            'description': description
        }
