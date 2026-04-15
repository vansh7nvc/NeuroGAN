"""
train_local.py — Full NeuroGAN pipeline: GAN training + CNN training + weight export.

Reads data from the local parquet file, runs GAN augmentation on the minority class,
trains the SimpleCNN (exact architecture from neurogan.py), and saves the weights to:
    model_weights/cnn_alzheimer.pt

Run with:
    python train_local.py

No internet or Colab needed.
"""

import os
import io
import random
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR  = r"MRI Dataset-20251203T142309Z-1-001\MRI Dataset"
TRAIN_PARQUET = os.path.join(DATASET_DIR, "train.parquet")
SYNTH_FOLDER  = "synthetic_data/MinorityClass_Synth"
WEIGHTS_DIR   = "model_weights"
WEIGHTS_PATH  = os.path.join(WEIGHTS_DIR, "cnn_alzheimer.pt")
PLOT_PATH     = "training_results.png"

os.makedirs(SYNTH_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_DIR,  exist_ok=True)

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")


# ==============================================================================
# 1. LOAD & PREPROCESS DATA
# ==============================================================================
def extract_bytes(blob):
    if isinstance(blob, dict):
        for key in ("bytes", "data", "image"):
            if key in blob and isinstance(blob[key], (bytes, bytearray)):
                return blob[key]
        for v in blob.values():
            if isinstance(v, (bytes, bytearray)):
                return v
    return blob


def bytes_to_pixels(b: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("L")   # force grayscale
    return np.array(img)


def preprocess_image(arr: np.ndarray, size: int = 64) -> np.ndarray:
    """Resize to size×size and normalise to [-1, 1]."""
    img = Image.fromarray(arr.astype("uint8"))
    img = img.resize((size, size))
    return (np.array(img, dtype=np.float32) / 127.5) - 1.0


print("[INFO] Loading train.parquet …")
train_df = pd.read_parquet(TRAIN_PARQUET)

print("[INFO] Decoding images …")
train_df["image"] = train_df["image"].apply(lambda b: bytes_to_pixels(extract_bytes(b)))

label_counts = train_df["label"].value_counts()
minority_label = label_counts.idxmin()
print(f"[INFO] Label distribution:\n{label_counts}")
print(f"[INFO] Minority class -> label {minority_label} ({label_counts[minority_label]} samples)")

# Preprocess minority class for GAN
minority_df = train_df[train_df["label"] == minority_label].copy()
minority_df["image"] = minority_df["image"].apply(preprocess_image)

# Preprocess full dataset for CNN
print("[INFO] Preprocessing all images to 64×64 …")
train_df["image"] = train_df["image"].apply(preprocess_image)


# ==============================================================================
# 2. DCGAN — minority class augmentation
# ==============================================================================
nz = 100; ngf = 64; ndf = 64; nc = 1; lr_gan = 0.0002; beta1 = 0.5


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf,     4, 2, 1, bias=False), nn.BatchNorm2d(ngf),     nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc,           4, 2, 1, bias=False), nn.Tanh(),
        )
    def forward(self, x): return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf,     4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2,  4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4,4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8,4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1,    4, 1, 0, bias=False), nn.Sigmoid(),
        )
    def forward(self, x): return self.main(x).view(-1, 1).squeeze(1)


class MinorityDataset(Dataset):
    def __init__(self, df):
        self.images = df["image"].tolist()
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).float().unsqueeze(0), 0


netG = Generator().to(device)
netD = Discriminator().to(device)
criterion_bce = nn.BCELoss()
optD = optim.Adam(netD.parameters(), lr=lr_gan, betas=(beta1, 0.999))
optG = optim.Adam(netG.parameters(), lr=lr_gan, betas=(beta1, 0.999))

gan_loader = DataLoader(MinorityDataset(minority_df), batch_size=16, shuffle=True)
GAN_EPOCHS  = 100      # fast on CPU — takes ~2 min

print(f"\n[INFO] Training DCGAN for {GAN_EPOCHS} epochs on {len(minority_df)} minority samples …")
for epoch in range(GAN_EPOCHS):
    for imgs, _ in gan_loader:
        imgs = imgs.to(device)
        b = imgs.size(0)

        # ── Discriminator ──
        netD.zero_grad()
        real_out = netD(imgs)
        errD_real = criterion_bce(real_out, torch.ones(b, device=device))
        errD_real.backward()

        noise = torch.randn(b, nz, 1, 1, device=device)
        fake  = netG(noise)
        fake_out = netD(fake.detach())
        errD_fake = criterion_bce(fake_out, torch.zeros(b, device=device))
        errD_fake.backward()
        optD.step()

        # ── Generator ──
        netG.zero_grad()
        gen_out = netD(fake)
        errG = criterion_bce(gen_out, torch.ones(b, device=device))
        errG.backward()
        optG.step()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  Epoch [{epoch+1}/{GAN_EPOCHS}]  D_loss: {(errD_real+errD_fake).item():.4f}  G_loss: {errG.item():.4f}")

# Generate 1000 synthetic images
NUM_SYNTH = 1000
print(f"\n[INFO] Generating {NUM_SYNTH} synthetic images …")
netG.eval()
count = 0
with torch.no_grad():
    while count < NUM_SYNTH:
        batch = min(64, NUM_SYNTH - count)
        noise = torch.randn(batch, nz, 1, 1, device=device)
        fakes = netG(noise)
        for i in range(batch):
            vutils.save_image(fakes[i], f"{SYNTH_FOLDER}/synth_{count+i:04d}.png",
                              normalize=True, value_range=(-1, 1))
        count += batch
print(f"[INFO] Saved to {SYNTH_FOLDER}/")


# ==============================================================================
# 3. BUILD COMBINED DATASET
# ==============================================================================
print("\n[INFO] Loading synthetic images …")
synth_records = []
for fname in os.listdir(SYNTH_FOLDER):
    if fname.endswith(".png"):
        arr = np.array(Image.open(os.path.join(SYNTH_FOLDER, fname)).convert("L").resize((64, 64)), dtype=np.float32)
        synth_records.append({"image": (arr / 127.5) - 1.0, "label": minority_label})

synth_df   = pd.DataFrame(synth_records)
combined_df = pd.concat([train_df[["image", "label"]], synth_df], ignore_index=True)
print(f"[INFO] Combined dataset: {len(combined_df)} samples")
print(combined_df["label"].value_counts())


# ==============================================================================
# 4. SimpleCNN — exact architecture from neurogan.py
# ==============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32,  3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128,3, 1, 1)   # ← Grad-CAM target
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 8 * 8, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ImageDataset(Dataset):
    def __init__(self, df):
        self.images = df["image"].tolist()
        self.labels = df["label"].tolist()
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).float().unsqueeze(0), self.labels[idx]


train_split, val_split = train_test_split(combined_df, test_size=0.2, random_state=SEED)
train_loader = DataLoader(ImageDataset(train_split), batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(ImageDataset(val_split),   batch_size=32, shuffle=False, num_workers=0)

model     = SimpleCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 10
print(f"\n[INFO] Training SimpleCNN for {NUM_EPOCHS} epochs …")
train_losses, val_accs = [], []

for epoch in range(NUM_EPOCHS):
    # ── Train ──
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── Validate ──
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = 100 * correct / total
    val_accs.append(acc)
    print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss: {avg_train_loss:.4f}  Val Acc: {acc:.2f}%")


# ==============================================================================
# 5. EVALUATE
# ==============================================================================
model.eval()
all_true, all_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        preds  = model(inputs).argmax(1).cpu()
        all_true.extend(labels.numpy())
        all_pred.extend(preds.numpy())

overall_acc = accuracy_score(all_true, all_pred)
overall_f1  = f1_score(all_true, all_pred, average="weighted")
target_names = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

print(f"\n{'='*55}")
print(f"  FINAL RESULTS")
print(f"{'='*55}")
print(f"  Accuracy : {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"  F1 Score : {overall_f1:.4f}")
print(f"{'='*55}")
print(classification_report(all_true, all_pred, target_names=target_names, zero_division=0))

# ── Confusion matrix ──
cm   = confusion_matrix(all_true, all_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
disp.plot(cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].tick_params(axis="x", rotation=30)

axes[1].plot(range(1, NUM_EPOCHS+1), train_losses, label="Train Loss")
axes[1].plot(range(1, NUM_EPOCHS+1), [a/100 for a in val_accs], label="Val Acc (scaled)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Value")
axes[1].set_title("Training Curves"); axes[1].legend()
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=120)
print(f"\n[INFO] Plot saved -> {PLOT_PATH}")


# ==============================================================================
# 6. SAVE WEIGHTS
# ==============================================================================
torch.save({
    "model_state_dict" : model.state_dict(),
    "class_names"      : target_names,
    "class_mapping"    : {i: n for i, n in enumerate(target_names)},
    "accuracy"         : overall_acc,
    "f1_score"         : overall_f1,
    "architecture"     : "SimpleCNN-3conv-64x64",
    "input_size"       : (1, 64, 64),
}, WEIGHTS_PATH)

print(f"[INFO] Weights saved -> {WEIGHTS_PATH}")
print("\n✅  Done!  Restart the backend and upload any MRI scan to test the dashboard.")
