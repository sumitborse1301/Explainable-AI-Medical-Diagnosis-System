import os, sys
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "results/chest_multidisease_index.csv"
WEIGHT_PATH = "vision/weights/chest_multidisease_tb.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_CSV = "results/chest_eval_results.csv"

CHEST_CLASSES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","Lung_Opacity","Tuberculosis","Normal"
]

# -----------------------
# LOAD MODEL
# -----------------------
print("[INFO] Loading chest model...")

model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CHEST_CLASSES))
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -----------------------
# LOAD CSV
# -----------------------
df = pd.read_csv(CSV_PATH)
print("[INFO] Total samples:", len(df))

# -----------------------
# EVALUATION
# -----------------------
results = []
tb_total = 0
tb_correct = 0

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):

        img = Image.open(row["image_path"]).convert("L")
        x = transform(img).unsqueeze(0).to(DEVICE)

        out = model(x)
        probs = torch.sigmoid(out)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        pred_label = CHEST_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        gt = row["labels"]

        # ---- TB accuracy check ----
        if "Tuberculosis" in str(gt):
            tb_total += 1
            if pred_label == "Tuberculosis":
                tb_correct += 1

        results.append({
            "image": os.path.basename(row["image_path"]),
            "ground_truth": gt,
            "prediction": pred_label,
            "confidence": confidence
        })

# -----------------------
# SAVE RESULTS
# -----------------------
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_CSV, index=False)

print("\n✅ Evaluation finished")
print("Saved to:", OUT_CSV)

# -----------------------
# ANALYSIS
# -----------------------
print("\nTop predictions:")
print(out_df["prediction"].value_counts().head(10))

print("\nMean confidence per disease:")
print(out_df.groupby("prediction")["confidence"].mean().sort_values(ascending=False).head(10))

# -----------------------
# TB RESULT
# -----------------------
if tb_total > 0:
    print("\n🦠 TB detection accuracy:", round((tb_correct/tb_total)*100, 2), "%")
    print("TB samples:", tb_total)
else:
    print("\n⚠ No TB samples found in CSV")
