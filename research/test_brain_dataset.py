import sys, os
sys.path.append(os.path.abspath("."))

from vision.brain_dataset import BrainTumorDataset

ds = BrainTumorDataset("data/brain", mode="train")

print("Total samples:", len(ds))
img, label, cls = ds[0]
print("Image shape:", img.shape)
print("Label:", label)
print("Class:", cls)
