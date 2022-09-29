from sympy import root
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class CheXpertDINODataset(Dataset):

    def __init__(self, file_table, root_dir, transforms) -> None:
        super().__init__()
        assert 'csv' in file_table
        self.df = pd.read_csv(file_table)
        self.transforms = transforms
        self.root_dir = root_dir
        assert os.path.isdir(self.root_dir)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.df.at[idx, 'Path'])
        im = Image.open(file_path)
        tim = torch.from_numpy(np.array(im))
        x = self.transforms(tim)
        return x , 0 # the y value doesnt matter here

    