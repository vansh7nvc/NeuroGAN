"""
model.py — SimpleCNN architecture + weight loader

This is the EXACT architecture used in neurogan.py (the training script):
  - 3 Conv layers (32 → 64 → 128) each followed by ReLU + MaxPool2d
  - Flatten → FC(128*8*8, 512) → ReLU → FC(512, 4)
  - Input:  (B, 1, 64, 64) grayscale MRI
  - Output: (B, 4) logits for [NonDemented, VeryMildDemented, MildDemented, ModerateDemented]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

CLASS_NAMES = [
    "NonDemented",        # Class 0
    "VeryMildDemented",   # Class 1  ← minority class augmented by NeuroGAN
    "MildDemented",       # Class 2
    "ModerateDemented",   # Class 3
]


class AlzheimerCNN(nn.Module):
    """
    Exact replica of SimpleCNN from neurogan.py.
    3 conv blocks (no BatchNorm) + 2 FC layers.
    Grad-CAM target: conv3 (last conv layer).
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()

        # ── Convolutional layers ─────────────────────────────────────────
        # Block 1: (1, 64, 64) → (32, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: (32, 32, 32) → (64, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: (64, 16, 16) → (128, 8, 8)   ← Grad-CAM target
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Fully connected layers ───────────────────────────────────────
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ── Grad-CAM target: last conv layer ──────────────────────────────────────
GRADCAM_LAYER = "conv3"


def load_model(weights_path: str | Path, device: str = "cpu") -> AlzheimerCNN:
    """
    Load a saved AlzheimerCNN checkpoint.

    Accepts two save formats from neurogan.py:
      1. Full dict:  {'model_state_dict': state_dict, ...}
      2. Raw state_dict from torch.save(model.state_dict(), path)

    Args:
        weights_path: Path to the .pt checkpoint file.
        device:       'cpu', 'cuda', or 'mps'.

    Returns:
        AlzheimerCNN in eval mode on the requested device.
    """
    weights_path = Path(weights_path)
    model = AlzheimerCNN(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume raw state_dict saved by: torch.save(model.state_dict(), path)
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def get_dummy_model(device: str = "cpu") -> AlzheimerCNN:
    """Return an untrained model (random weights) — used for testing the API without a .pt file."""
    model = AlzheimerCNN()
    model.to(device)
    model.eval()
    return model
