import sys, os
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from vision.bone_dataset import MuraBoneDataset
from tqdm import tqdm

CSV_PATH = "results/mura_index.csv"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "vision/weights/efficientnet_bone_fracture.pt"

print("[INFO] Loading dataset...")
dataset = MuraBoneDataset(CSV_PATH)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


print("[INFO] Loading EfficientNet-B0...")
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------
# Training loop
# --------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[TRAIN] Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"[VAL] Accuracy: {acc:.4f}")

os.makedirs("vision/weights", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)

print("✅ Training complete.")
print("Model saved to:", SAVE_PATH)
