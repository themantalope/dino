import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CheXpertDINODataset(Dataset):

    def __init__(self, file_table, transforms) -> None:
        super().__init__()
        assert 'csv' in file_table
        self.df = pd.read_csv(file_table)
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        file_path = self.df.at[idx, 'Path']
        im = Image.open(file_path)
        tim = torch.from_numpy(np.array(im))
        x = self.transforms(tim)
        return x , 0 # the y value doesnt matter here

    