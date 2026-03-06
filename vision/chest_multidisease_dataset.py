import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


CHEST_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia", "Lung_Opacity", "Tuberculosis", "Normal"
]


class ChestMultiDiseaseDataset(Dataset):
    def __init__(self, csv_path, mode="train"):
        """
        csv_path: results/chest_multidisease_index.csv
        mode: train / val
        """

        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.num_classes = len(CHEST_CLASSES)

        print(f"[INFO] Chest samples loaded: {len(self.df)}")

        # ----------------------------
        # MEDICAL TRANSFORMS
        # ----------------------------

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        labels_str = row["labels"]

        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        # ----------------------------
        # MULTI-LABEL TARGET
        # ----------------------------
        target = [0] * self.num_classes

        if labels_str.lower() == "normal":
            target[CHEST_CLASSES.index("Normal")] = 1
        else:
            diseases = labels_str.split("|")
            for d in diseases:
                d = d.strip()
                if d in CHEST_CLASSES:
                    target[CHEST_CLASSES.index(d)] = 1

        return img, target
