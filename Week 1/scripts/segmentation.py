from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

IMG_DIR = Path("FRUS")
MASK_DIR = Path("Masks")


# -------------------------
# 1. Configuración
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



print(f"Using: {DEVICE}", flush=True)

SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-3

print("DEVICE:", DEVICE)


# -------------------------
# 2. Dataset
# -------------------------
class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, mask_dir):
        self.image_paths = image_paths
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        # Gray-scale images
        image = Image.open(img_path).convert("L")
        image = image.resize(
            SIZE, Image.Resampling.BILINEAR
        )
        image = np.array(image, dtype=np.float32) / 255.0

        # Binary masks
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            SIZE, Image.Resampling.NEAREST
        )
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 127).astype(np.float32)

        # [H, W] -> [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask



# -------------------------
# 3. U-Net Architecture
# -------------------------
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = conv_block(1, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)

        self.bottleneck = conv_block(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = conv_block(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(
            torch.cat([self.up4(b), e4], dim=1)
        )
        d3 = self.dec3(
            torch.cat([self.up3(d4), e3], dim=1)
        )
        d2 = self.dec2(
            torch.cat([self.up2(d3), e2], dim=1)
        )
        d1 = self.dec1(
            torch.cat([self.up1(d2), e1], dim=1)
        )

        # return logits, not probabilities
        return self.out(d1)




# -------------------------
# 4. Loss function
# -------------------------
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = (
        probs.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets
    )
    return bce + dice_loss(logits, targets)


def dice_score(logits, targets, eps=1e-6):
    preds = torch.sigmoid(logits) > 0.5
    targets = targets > 0.5

    intersection = (preds & targets).sum(dim=(1, 2, 3))
    denominator = (
        preds.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2 * intersection + eps) / (denominator + eps)
    return dice.mean().item()




# -------------------------
# 5. Load and prepare the dataset
# -------------------------
image_paths = sorted(IMG_DIR.glob("*.png"))

assert len(image_paths) >= 2, (
    "Necesitas al menos dos imágenes PNG."
)

for p in image_paths:
    assert (MASK_DIR / p.name).exists(), (
        f"No se encontró la máscara de {p.name}"
    )


random.shuffle(image_paths)

n_val = max(1, int(0.2 * len(image_paths)))
val_paths = image_paths[:n_val]
train_paths = image_paths[n_val:]

print(n_val)
print(val_paths)
print(train_paths)

train_ds = UltrasoundDataset(train_paths, MASK_DIR)
val_ds = UltrasoundDataset(val_paths, MASK_DIR)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False
)

print("Training:", len(train_ds))
print("Validation:", len(val_ds))


# -------------------------
# 6. Training
# -------------------------
model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_dice = -1.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = combined_loss(logits, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            val_dice += dice_score(logits, masks)

    train_loss /= len(train_loader)
    val_dice /= len(val_loader)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "unet_best.pth")

print("Best validation dice:", best_val_dice)
print("Model saved in unet_best.pth")


########################################################

model.load_state_dict(
    torch.load("unet_best.pth", map_location=DEVICE)
)
model.eval()

image, mask = val_ds[0]

with torch.no_grad():
    logits = model(image.unsqueeze(0).to(DEVICE))
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()

image = image.squeeze().cpu().numpy()
mask = mask.squeeze().cpu().numpy()
pred = pred.squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Imagen original")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Máscara real")

axes[2].imshow(pred, cmap="gray")
axes[2].set_title("Predicción")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()