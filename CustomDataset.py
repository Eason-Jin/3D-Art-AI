from torch.utils.data import Dataset
from utils import MAX_TIME
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_time: int = MAX_TIME):
        self.filenames = dataframe['filename'.tolist()]
        self.data = dataframe['original_data'].tolist()
        self.noisy_data = [
            [row[f'noisy_data_{i}'] for i in range(max_time)]
            for _, row in dataframe.iterrows()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return {
            'filename': self.filenames[idx],
            'original_data': self.data[idx],
            'noisy_data_list': self.noisy_data[idx],
        }
