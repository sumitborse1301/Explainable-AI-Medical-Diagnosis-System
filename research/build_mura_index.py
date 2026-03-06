import os
import csv

ROOT = "data/mura"   # change if your folder name is different
OUT_CSV = "results/mura_index.csv"

BONES = ["XR_WRIST", "XR_HAND", "XR_FOREARM", "XR_FINGER", "XR_SHOULDER"]

rows = [["image_path", "bone", "patient", "study", "label"]]

def get_label(study_folder):
    return 1 if "positive" in study_folder.lower() else 0

for split in ["train", "valid"]:
    split_path = os.path.join(ROOT, split)

    for bone in BONES:
        bone_path = os.path.join(split_path, bone)
        if not os.path.isdir(bone_path):
            continue

        for patient in os.listdir(bone_path):
            patient_path = os.path.join(bone_path, patient)
            if not os.path.isdir(patient_path):
                continue

            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue

                label = get_label(study)

                for file in os.listdir(study_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(study_path, file).replace("\\","/")

                        rows.append([
                            img_path,
                            bone.replace("XR_", ""),
                            patient,
                            study,
                            label
                        ])

os.makedirs("results", exist_ok=True)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("✅ MURA index built.")
print("Total images:", len(rows)-1)
print("Saved to:", OUT_CSV)
