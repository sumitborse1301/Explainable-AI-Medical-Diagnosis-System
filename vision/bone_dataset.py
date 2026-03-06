import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random

class MuraBoneDataset(Dataset):
    def __init__(self, csv_path, split="train", bone_filter=None):
        self.df = pd.read_csv(csv_path)

        if split == "train":
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        if bone_filter:
            self.df = self.df[self.df["bone"] == bone_filter.upper()]

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((224,224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): 
        while True:
            row = self.df.iloc[idx]

            try:
                img = Image.open(row["image_path"]).convert("L")
                img = self.transform(img)

                label = torch.tensor(row["label"], dtype=torch.long)
                bone = row["bone"]

                return img, label, bone

            except Exception as e:
                # Skip corrupted file
                idx = random.randint(0, len(self.df) - 1)
