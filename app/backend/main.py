"""
main.py — NeuroGAN FastAPI Server

Endpoints:
  POST /predict  — Upload MRI image → get prediction + Grad-CAM heatmap
  POST /report   — Generate downloadable PDF clinical report
  GET  /health   — Health check

Run with:
  uvicorn app.backend.main:app --reload --port 8000
"""

import base64
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from .gradcam import run_gradcam
from .model import CLASS_NAMES, AlzheimerCNN, get_dummy_model, load_model
from .report import generate_report

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroGAN API",
    description="Alzheimer's MRI Classification with Grad-CAM Explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ─────────────────────────────────────────────────────

WEIGHTS_PATH = Path(__file__).parent.parent.parent / "model_weights" / "cnn_alzheimer.pt"
_model: Optional[AlzheimerCNN] = None


def get_model() -> AlzheimerCNN:
    global _model
    if _model is None:
        if WEIGHTS_PATH.exists():
            print(f"[NeuroGAN] Loading weights from {WEIGHTS_PATH}")
            _model = load_model(WEIGHTS_PATH)
        else:
            print("[NeuroGAN] ⚠️  No weights file found — using untrained dummy model for testing.")
            _model = get_dummy_model()
    return _model


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: list[float]
    gradcam_png: str          # base64-encoded PNG
    original_png: str         # base64-encoded PNG (resized to 256×256 for display)
    model_loaded: bool


class ReportRequest(BaseModel):
    filename: str
    class_id: int
    probabilities: list[float]
    original_png: str         # base64-encoded PNG
    gradcam_png: str          # base64-encoded PNG
    patient_id: Optional[str] = None
    scan_date: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


async def _read_image(upload: UploadFile) -> np.ndarray:
    """Read an uploaded image file and return a grayscale uint8 numpy array."""
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {upload.content_type}. "
                   f"Accepted: PNG, JPEG, BMP, TIFF.",
        )
    data = await upload.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    try:
        img = Image.open(io.BytesIO(data)).convert("L")  # grayscale
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")


def _array_to_b64(arr: np.ndarray, size: int = 256) -> str:
    """Resize a grayscale array and encode as base64 PNG."""
    img = Image.fromarray(arr).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — also reports whether model weights are loaded."""
    model_loaded = WEIGHTS_PATH.exists()
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "weights_path": str(WEIGHTS_PATH),
        "classes": CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a grayscale MRI scan (PNG/JPEG) and receive:
      - Predicted Alzheimer's stage
      - Confidence score
      - All class probabilities
      - Grad-CAM heatmap (base64 PNG)
      - Original MRI resized for display (base64 PNG)
    """
    model = get_model()
    img_array = await _read_image(file)

    try:
        heatmap_b64, _, pred_class, probs = run_gradcam(model, img_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    original_b64 = _array_to_b64(img_array)

    return PredictionResponse(
        class_id=pred_class,
        class_name=CLASS_NAMES[pred_class],
        confidence=round(probs[pred_class], 4),
        probabilities=[round(p, 4) for p in probs],
        gradcam_png=heatmap_b64,
        original_png=original_b64,
        model_loaded=WEIGHTS_PATH.exists(),
    )


@app.post("/report")
async def report(req: ReportRequest):
    """
    Generate and return a downloadable PDF clinical report.
    Accepts the full prediction result payload from /predict.
    """
    try:
        pdf_bytes = generate_report(
            filename=req.filename,
            predicted_class=req.class_id,
            probabilities=req.probabilities,
            original_mri_b64=req.original_png,
            gradcam_b64=req.gradcam_png,
            patient_id=req.patient_id,
            scan_date=req.scan_date,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="NeuroGAN_Report_{req.filename}.pdf"'
        },
    )
