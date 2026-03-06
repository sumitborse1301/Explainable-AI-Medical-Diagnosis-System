import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
from vision.brain_dataset import BrainTumorDataset


ds = BrainTumorDataset("data/brain", mode="train")
labels = [x[1] for x in ds.samples]

count = Counter(labels)
print("\nClass distribution:")
print(count)
