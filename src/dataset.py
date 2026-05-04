import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CheXpertDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, sample_size=None):
        self.df = pd.read_csv(csv_file)

        if sample_size:
            self.df = self.df.sample(sample_size, random_state=42)

        self.df["Cardiomegaly"] = self.df["Cardiomegaly"].fillna(0).replace(-1, 0)

        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = os.path.join(
            self.root,
            row["Path"].replace("CheXpert-v1.0-small/", "")
        )

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row["Cardiomegaly"], dtype=torch.float32)
        sex = row["Sex"]

        return img, label, sex