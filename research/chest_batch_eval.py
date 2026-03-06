import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import csv
from PIL import Image
from datetime import datetime
from tqdm import tqdm

from vision.chest_xray_engine import ChestXrayEngine

# =========================
# PATH CONFIG
# =========================
DATASET_DIR = "datasets/ChestXray14/eval_images"
RESULTS_DIR = "results"
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")
CSV_PATH = os.path.join(RESULTS_DIR, "chest_eval_results.csv")
WEIGHT_PATH = "vision/weights/densenet121_chestxray.pt"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print("[INFO] Loading chest X-ray model...")
engine = ChestXrayEngine(weight_path=WEIGHT_PATH)
print("[INFO] Model loaded.")

# =========================
# IMAGE LIST
# =========================
images = [f for f in os.listdir(DATASET_DIR)
          if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"[INFO] Found {len(images)} images for evaluation.")

# =========================
# CSV HEADER
# =========================
header = [
    "image_name",
    "prediction",
    "confidence",
    "top_3_probs",
    "gradcam_path",
    "timestamp"
]

# =========================
# RUN BATCH EVALUATION
# =========================
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for img_name in tqdm(images):
        try:
            img_path = os.path.join(DATASET_DIR, img_name)
            image = Image.open(img_path).convert("RGB")

            result = engine.predict(image)

            # top 3 probabilities
            probs_sorted = sorted(
                result["all_probs"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            top3_str = "; ".join([f"{k}:{round(v,4)}" for k, v in probs_sorted])

            # save gradcam
            cam_save_path = os.path.join(GRADCAM_DIR, img_name)
            
            from PIL import Image
            import numpy as np

            cam = result["cam"]

            # Convert numpy CAM to proper image
            if isinstance(cam, np.ndarray):
                cam = cam.astype("float32")

                # normalize to 0–255
                cam = cam - cam.min()
                if cam.max() != 0:
                    cam = cam / cam.max()
                cam = (cam * 255).astype("uint8")

                cam = Image.fromarray(cam)

            # ensure RGB
            if cam.mode != "RGB":
                cam = cam.convert("RGB")

            cam.save(cam_save_path)



            # write row
            writer.writerow([
                img_name,
                result["prediction"],
                round(result["confidence"], 4),
                top3_str,
                cam_save_path,
                datetime.now().isoformat()
            ])

        except Exception as e:
            print(f"[ERROR] Failed on {img_name} -> {e}")

print("\n[FINISHED] Batch evaluation completed.")
print(f"[RESULT] CSV saved at: {CSV_PATH}")
print(f"[RESULT] Grad-CAM images saved in: {GRADCAM_DIR}")
