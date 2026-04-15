"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)

Produces a heatmap overlay PNG showing *where* in the MRI the CNN focused
its attention when making a prediction.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from
           Deep Networks via Gradient-based Localization"
"""

import base64
import io
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import AlzheimerCNN, GRADCAM_LAYER


class GradCAM:
    """
    Hooks into a target layer of a CNN to compute Grad-CAM visualisations.

    Usage:
        gc = GradCAM(model, target_layer_name="features.14")
        heatmap_b64, raw_cam = gc.generate(input_tensor, class_idx=None)
    """

    def __init__(self, model: AlzheimerCNN, target_layer_name: str = GRADCAM_LAYER):
        self.model = model
        self.target_layer_name = target_layer_name
        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    # ── Hook registration ──────────────────────────────────────────────────

    def _find_layer(self) -> torch.nn.Module:
        """Traverse the model by dot-separated name to find the target layer.
        Works for both direct attributes (e.g. 'conv3') and nested modules
        (e.g. 'features.14').
        """
        parts = self.target_layer_name.split(".")
        module = self.model
        for p in parts:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p)
        return module

    def _register_hooks(self):
        layer = self._find_layer()

        def forward_hook(module, input, output):
            self._feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Core Grad-CAM algorithm ────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        original_image: Optional[np.ndarray] = None,
    ) -> tuple[str, np.ndarray]:
        """
        Compute Grad-CAM for the given input.

        Args:
            input_tensor:   Normalised image tensor of shape (1, 1, 64, 64).
            class_idx:      Target class index. If None, uses the predicted class.
            original_image: Optional (H, W) uint8 numpy array for overlay.
                            If None, the input tensor is decoded for overlay.

        Returns:
            (heatmap_b64, cam_array)
            heatmap_b64 — base64-encoded PNG: original MRI + coloured heatmap overlay
            cam_array   — raw (H, W) float32 CAM values in [0, 1]
        """
        self.model.zero_grad()

        # Forward pass (requires grad)
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward for the target class score
        score = logits[0, class_idx]
        score.backward()

        # ── Compute CAM ────────────────────────────────────────────────────
        # Pool gradients across spatial dims to get channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._feature_maps).sum(dim=1).squeeze(0)  # (H', W')
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Resize to input resolution (64×64)
        cam_resized = cv2.resize(cam, (64, 64))

        # ── Build overlay ──────────────────────────────────────────────────
        if original_image is None:
            # Decode from tensor: un-normalise from [-1, 1] → [0, 255]
            img_np = input_tensor.detach().squeeze().cpu().numpy()
            img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            original_image = img_np

        # Scale original to display size (256×256 for clarity)
        display_size = 256
        orig_display = cv2.resize(original_image, (display_size, display_size))
        orig_rgb = cv2.cvtColor(orig_display, cv2.COLOR_GRAY2RGB)

        cam_display = cv2.resize(cam_resized, (display_size, display_size))
        heatmap = cv2.applyColorMap(
            (cam_display * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend: 55% original + 45% heatmap
        overlay = cv2.addWeighted(orig_rgb, 0.55, heatmap_rgb, 0.45, 0)

        # Encode to PNG → base64
        pil_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return b64, cam_resized


def run_gradcam(
    model: AlzheimerCNN,
    image_array: np.ndarray,
    class_idx: Optional[int] = None,
) -> tuple[str, np.ndarray, int, list[float]]:
    """
    Convenience wrapper: takes a raw (H, W) uint8 numpy MRI image,
    preprocesses it, runs inference + Grad-CAM, and returns results.

    Args:
        model:       Loaded AlzheimerCNN in eval mode.
        image_array: Grayscale MRI numpy array (any size).
        class_idx:   Override predicted class (for visualising alternative classes).

    Returns:
        (heatmap_b64, cam_array, predicted_class, probabilities_list)
    """
    # Preprocess: resize → normalise → tensor
    img_resized = cv2.resize(image_array, (64, 64)).astype(np.float32)
    # Ensure single channel
    if img_resized.ndim == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Keep a uint8 copy for overlay
    img_uint8 = img_resized.astype(np.uint8)

    # Normalise to [-1, 1]
    img_norm = (img_resized / 127.5) - 1.0
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float()  # (1,1,64,64)

    # Get probabilities
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        predicted = int(logits.argmax(dim=1).item())

    # Grad-CAM requires gradients
    gc = GradCAM(model)
    heatmap_b64, cam_array = gc.generate(tensor, class_idx=class_idx or predicted, original_image=img_uint8)
    gc.remove_hooks()

    return heatmap_b64, cam_array, predicted, probs
