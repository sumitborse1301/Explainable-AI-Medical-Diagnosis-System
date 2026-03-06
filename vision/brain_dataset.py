import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        """
        root_dir: data/brain
        mode: train or test
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.samples = []

        self.class_map = {
            "glioma": 0,
            "meningioma": 1,
            "pituitary": 2,
            "notumor": 3
        }

        for cls in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_path):
                continue

            label = self.class_map[cls]

            for file in os.listdir(class_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(class_path, file), label, cls)
                    )

        print(f"[INFO] Brain {mode} samples loaded:", len(self.samples))

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cls = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return img, label, cls
