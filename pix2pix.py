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

# ============================================================
# STEP 4: Implement Pix2Pix Model (Generator + Discriminator)
# Commit: "Implemented Pix2Pix Generator and Discriminator models"
# ============================================================
import torch.nn as nn

class UNetBlock(nn.Module):
    """Single U-Net encoder-decoder block with skip connection support."""
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
        self.act_down = nn.LeakyReLU(0.2)
        self.act_up = nn.ReLU()
        self.down = down

    def forward(self, x):
        act = self.act_down if self.down else self.act_up
        return act(self.block(x))


class Generator(nn.Module):
    """Simplified U-Net Generator for Pix2Pix."""
    def __init__(self):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)

        # Decoder (upsampling with skip connections)
        self.dec1 = UNetBlock(512, 256, down=False, dropout=True)
        self.dec2 = UNetBlock(512, 128, down=False)   # 512 = 256 + skip 256
        self.dec3 = UNetBlock(256, 64, down=False)    # 256 = 128 + skip 128
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),      # 128 = 64 + skip 64
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        return self.final(torch.cat([d3, e1], dim=1))


class Discriminator(nn.Module):
    """PatchGAN Discriminator: classifies 70x70 image patches."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),                         # input: sat+map concatenated
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1),                        # patch output
            nn.Sigmoid()
        )

    def forward(self, sat, target):
        x = torch.cat([sat, target], dim=1)  # Concatenate along channel dim
        return self.model(x)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)
print(f"Models initialized on: {device}")

# ============================================================
# STEP 5: Train Pix2Pix GAN
# Commit: "Trained Pix2Pix GAN for satellite-to-map translation"
# ============================================================
import torch.optim as optim

adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Track losses for visualization
g_losses, d_losses = [], []

EPOCHS = 10
LAMBDA_L1 = 10  # Weight for L1 reconstruction loss

print("\nStarting training...")
for epoch in range(EPOCHS):
    epoch_g_loss, epoch_d_loss, batches = 0, 0, 0

    for sat_imgs, map_imgs in dataloader:
        sat_imgs = sat_imgs.to(device)
        map_imgs = map_imgs.to(device)
        batch_size = sat_imgs.size(0)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        fake_maps = G(sat_imgs)

        # Dynamically match label size to actual discriminator output
        d_out_shape = D(sat_imgs, map_imgs).shape
        real_labels = torch.ones(d_out_shape).to(device)
        fake_labels = torch.zeros(d_out_shape).to(device)

        real_loss = adversarial_loss(D(sat_imgs, map_imgs), real_labels)
        fake_loss = adversarial_loss(D(sat_imgs, fake_maps.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        optimizer_G.zero_grad()
        fake_maps = G(sat_imgs)
        g_adv = adversarial_loss(D(sat_imgs, fake_maps), real_labels)
        g_l1 = l1_loss(fake_maps, map_imgs) * LAMBDA_L1
        g_loss = g_adv + g_l1
        g_loss.backward()
        optimizer_G.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        batches += 1

    avg_g = epoch_g_loss / batches
    avg_d = epoch_d_loss / batches
    g_losses.append(avg_g)
    d_losses.append(avg_d)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")

print("Training complete!")

# ============================================================
# STEP 6: Evaluate and Visualize Translated Images
# Commit: "Evaluated Pix2Pix and visualized image translation"
# ============================================================

os.makedirs("outputs", exist_ok=True)
G.eval()

with torch.no_grad():
    # Get a test sample
    sat_sample, map_sample = dataset[0]
    sat_tensor = sat_sample.unsqueeze(0).to(device)
    translated = G(sat_tensor).squeeze().cpu()

    # Denormalize from [-1,1] to [0,1] for display
    def denorm(t):
        return (t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

    # ------- OUTPUT 1: Sample Translation Comparison -------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(denorm(sat_sample))
    axes[0].set_title("Input (Satellite)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(denorm(translated))
    axes[1].set_title("Generated (Map)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(denorm(map_sample))
    axes[2].set_title("Ground Truth (Map)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("Pix2Pix GAN: Satellite → Map Translation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/sample_output.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/sample_output.jpg")

    # ------- OUTPUT 2: Training Loss Curves -------
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_range = range(1, EPOCHS + 1)
    ax.plot(epochs_range, g_losses, label="Generator Loss", color="#e74c3c", linewidth=2, marker='o')
    ax.plot(epochs_range, d_losses, label="Discriminator Loss", color="#3498db", linewidth=2, marker='s')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Pix2Pix GAN Training Loss Curves", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(epochs_range)
    plt.tight_layout()
    plt.savefig("outputs/training_losses.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/training_losses.png")

    # ------- OUTPUT 3: Pixel Intensity Distribution -------
    # Compare pixel distributions: real map vs generated map (across 10 samples)
    real_pixels, fake_pixels = [], []
    for i in range(min(10, len(dataset))):
        sat_i, map_i = dataset[i]
        sat_tensor_i = sat_i.unsqueeze(0).to(device)
        fake_i = G(sat_tensor_i).squeeze().cpu()
        real_pixels.append(map_i.numpy().flatten())
        fake_pixels.append(fake_i.numpy().flatten())

    real_pixels = np.concatenate(real_pixels)
    fake_pixels = np.concatenate(fake_pixels)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_pixels, bins=60, alpha=0.6, color="#2ecc71", label="Real Maps", density=True)
    ax.hist(fake_pixels, bins=60, alpha=0.6, color="#e74c3c", label="Generated Maps", density=True)
    ax.set_xlabel("Pixel Intensity (normalized)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pixel Intensity Distribution:\nReal vs Generated Map Images", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/pixel_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/pixel_distribution.png")

print("\nAll outputs saved to ./outputs/")
print("Done! Push to GitHub with commit: 'Evaluated Pix2Pix and visualized image translation'")