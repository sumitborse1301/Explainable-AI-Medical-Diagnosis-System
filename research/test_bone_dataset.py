import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.bone_dataset import MuraBoneDataset

ds = MuraBoneDataset("results/mura_index.csv")

print("Total images:", len(ds))

img, label, bone = ds[0]

print("Image shape:", img.shape)
print("Label:", label)
print("Bone:", bone)
