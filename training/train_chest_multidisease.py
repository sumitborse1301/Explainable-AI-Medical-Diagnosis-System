import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import numpy as np

from vision.chest_multidisease_dataset import ChestMultiDiseaseDataset, CHEST_CLASSES


# =====================
# CONFIG
# =====================
CSV_PATH = "results/chest_multidisease_index.csv"
BATCH_SIZE = 8
EPOCHS = 6
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "vision/weights/chest_multidisease_tb.pt"

os.makedirs("vision/weights", exist_ok=True)


# =====================
# CLASS WEIGHTS
# =====================
def compute_class_weights(df, classes):
    counts = np.zeros(len(classes))
    for labels in df["labels"]:
        if labels.lower() == "normal":
            counts[classes.index("Normal")] += 1
        else:
            for d in labels.split("|"):
                if d.strip() in classes:
                    counts[classes.index(d.strip())] += 1

    weights = 1.0 / (counts + 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# =====================
# TRAINING
# =====================
def main():
    print("[INFO] Loading chest dataset...")
    full_dataset = ChestMultiDiseaseDataset(CSV_PATH, mode="train")

    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("[INFO] Loading EfficientNet-B0...")
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CHEST_CLASSES))
    model = model.to(DEVICE)

    class_weights = compute_class_weights(full_dataset.df, CHEST_CLASSES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # =====================
    # TRAIN LOOP
    # =====================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            targets = torch.stack(targets).T.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[TRAIN] Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE)
                targets = torch.stack(targets).T.float().to(DEVICE)
                
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f"[VAL] Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print("\n✅ Chest multi-disease training complete.")
    print("Model saved to:", SAVE_PATH)


if __name__ == "__main__":
    main()
