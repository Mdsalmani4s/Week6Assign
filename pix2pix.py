"""
Pix2Pix GAN for Satellite-to-Map Image Translation
Week 6 Assignment
"""

# ============================================================
# STEP 1: Project Setup
# Commit: "Initial commit - project structure created"
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 2: Install Required Libraries (run in terminal)
# pip install torch torchvision numpy matplotlib pillow opencv-python
# Commit: "Installed required libraries"
# ============================================================

# ============================================================
# STEP 3: Load and Preprocess the Dataset
# Commit: "Loaded and preprocessed image datasets for Pix2Pix GAN"
# ============================================================
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Transforms: resize, normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class Pix2PixDataset(Dataset):
    """
    Loads paired satellite/map images.
    Each image in the dataset is side-by-side (satellite | map).
    Falls back to synthetic data if real data is missing.
    """
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = []

        if os.path.exists(data_dir):
            self.image_paths = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(('.jpg', '.png'))
            ]

        # If no real data found, generate synthetic pairs
        if len(self.image_paths) == 0:
            print("No data found. Using synthetic dataset for demo.")
            self.synthetic = True
            self.length = 100
        else:
            self.synthetic = False

    def __len__(self):
        return self.length if hasattr(self, 'length') else len(self.image_paths)

    def __getitem__(self, idx):
        if self.synthetic:
            # Synthetic: satellite = random noise image, map = slightly different noise
            sat = torch.rand(3, 256, 256) * 2 - 1
            mp = torch.clamp(sat + torch.randn_like(sat) * 0.3, -1, 1)
            return sat, mp
        else:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            w, h = img.size
            # Images are side-by-side: left=satellite, right=map
            sat_img = img.crop((0, 0, w // 2, h))
            map_img = img.crop((w // 2, 0, w, h))
            if self.transform:
                sat_img = self.transform(sat_img)
                map_img = self.transform(map_img)
            return sat_img, map_img

# Load dataset
dataset = Pix2PixDataset("./data/maps", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f"Dataset size: {len(dataset)} image pairs")



