import os
import pandas as pd

NIH_ROOT = "data/chest/nih"
TB_ROOT = "data/chest/tb"
OUT_CSV = "results/chest_multidisease_index.csv"

os.makedirs("results", exist_ok=True)

print("[INFO] Loading NIH metadata...")

nih_csv = os.path.join(NIH_ROOT, "Data_Entry_2017.csv")
df = pd.read_csv(nih_csv)

nih_map = dict(zip(df["Image Index"], df["Finding Labels"]))

records = []

# -------- AUTO FIND NIH IMAGE FOLDERS --------
nih_image_dirs = []

for root, dirs, files in os.walk(NIH_ROOT):
    if any(f.lower().endswith(".png") for f in files):
        nih_image_dirs.append(root)

print("[INFO] Found NIH image folders:", nih_image_dirs)

# ---------------- NIH IMAGES ----------------
for folder_path in nih_image_dirs:
    for img in os.listdir(folder_path):
        if img.endswith(".png"):
            labels = nih_map.get(img, "No Finding")

            records.append({
                "image_path": os.path.join(folder_path, img),
                "source": "NIH",
                "labels": labels,
                "tb": 0
            })

# ---------------- TB IMAGES ----------------
print("[INFO] Adding TB dataset...")

for cls in ["Normal", "Tuberculosis"]:
    cls_path = os.path.join(TB_ROOT, cls)

    for img in os.listdir(cls_path):
        if img.lower().endswith((".png", ".jpg", ".jpeg")):
            records.append({
                "image_path": os.path.join(cls_path, img),
                "source": "TB",
                "labels": "Tuberculosis" if cls == "Tuberculosis" else "No Finding",
                "tb": 1 if cls == "Tuberculosis" else 0
            })

# ---------------- SAVE ----------------
final_df = pd.DataFrame(records)
final_df.to_csv(OUT_CSV, index=False)

print("✅ Unified chest index built")
print("Total images:", len(final_df))
print("Saved to:", OUT_CSV)
