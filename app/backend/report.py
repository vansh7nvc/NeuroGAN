"""
report.py — Automated Clinical PDF Report Generator

Produces a professional A4 PDF report containing:
  - NeuroGAN header + report metadata
  - Original MRI scan thumbnail
  - Grad-CAM heatmap overlay
  - Predicted Alzheimer's stage + confidence
  - Class probability table
  - Research disclaimer

Requires: fpdf2  (pip install fpdf2)
"""

import base64
import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF, XPos, YPos
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]

CLASS_COLORS = {
    "NonDemented": (34, 197, 94),        # green
    "VeryMildDemented": (234, 179, 8),   # yellow
    "MildDemented": (249, 115, 22),      # orange
    "ModerateDemented": (239, 68, 68),   # red
}

ACCENT_COLOR = (0, 212, 255)    # electric cyan
DARK_BG = (10, 15, 30)         # deep navy
LIGHT_TEXT = (220, 230, 245)


# ── PDF Builder ───────────────────────────────────────────────────────────────

class NeuroGANReport(FPDF):
    """Custom FPDF subclass with a branded header and footer."""

    def header(self):
        # Dark banner
        self.set_fill_color(*DARK_BG)
        self.rect(0, 0, 210, 22, style="F")

        # Logo text
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*ACCENT_COLOR)
        self.set_xy(10, 6)
        self.cell(60, 10, "NeuroGAN", new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Subtitle
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*LIGHT_TEXT)
        self.set_xy(10, 14)
        self.cell(120, 6, "AI-Powered Alzheimer's MRI Analysis System", new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Page number (top right)
        self.set_font("Helvetica", "", 8)
        self.set_xy(160, 8)
        self.cell(40, 6, f"Page {self.page_no()}", align="R")

        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_fill_color(*DARK_BG)
        self.rect(0, 282, 210, 15, style="F")
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*LIGHT_TEXT)
        self.cell(
            0, 10,
            "RESEARCH USE ONLY. This report is not a substitute for professional medical diagnosis.",
            align="C",
        )


def _b64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base64 PNG string to a PIL Image."""
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def _pil_to_tmp_path(img: Image.Image, tmp_dir: Path, name: str) -> Path:
    """Save a PIL image to a temp directory and return its path."""
    p = tmp_dir / name
    img.save(p, format="PNG")
    return p


def generate_report(
    filename: str,
    predicted_class: int,
    probabilities: list[float],
    original_mri_b64: str,
    gradcam_b64: str,
    patient_id: Optional[str] = None,
    scan_date: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF clinical report as bytes.

    Args:
        filename:         Original uploaded filename.
        predicted_class:  Integer class index (0–3).
        probabilities:    List of 4 float probabilities.
        original_mri_b64: Base64-encoded PNG of the original MRI.
        gradcam_b64:      Base64-encoded PNG of the Grad-CAM overlay.
        patient_id:       Optional anonymous patient identifier.
        scan_date:        Optional scan date string.

    Returns:
        PDF file as bytes.
    """
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())

    # ── Save images to temp files ─────────────────────────────────────────
    mri_img = _b64_to_pil(original_mri_b64).convert("RGB")
    cam_img = _b64_to_pil(gradcam_b64).convert("RGB")

    mri_path = _pil_to_tmp_path(mri_img, tmp_dir, "mri.png")
    cam_path = _pil_to_tmp_path(cam_img, tmp_dir, "gradcam.png")

    # ── Build PDF ─────────────────────────────────────────────────────────
    pdf = NeuroGANReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    class_name = CLASS_NAMES[predicted_class]
    confidence = probabilities[predicted_class] * 100
    color = CLASS_COLORS[class_name]

    # ── Section: Report Metadata ──────────────────────────────────────────
    _section_header(pdf, "Report Information")

    meta = [
        ("Report ID",   report_id),
        ("Generated",   now),
        ("Patient ID",  patient_id or "Anonymous"),
        ("Scan File",   filename),
        ("Scan Date",   scan_date or "Not Specified"),
    ]
    for label, value in meta:
        _info_row(pdf, label, value)

    pdf.ln(6)

    # ── Section: Diagnosis ────────────────────────────────────────────────
    _section_header(pdf, "AI Diagnosis")

    # Coloured badge
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, f"  {class_name}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Confidence:  {confidence:.1f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Confidence bar
    bar_w = 160
    bar_h = 5
    fill_w = bar_w * (confidence / 100)
    pdf.set_y(pdf.get_y() + 1)
    pdf.set_x(10)
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(10, pdf.get_y(), bar_w, bar_h, style="F")
    pdf.set_fill_color(*color)
    pdf.rect(10, pdf.get_y(), fill_w, bar_h, style="F")
    pdf.ln(10)

    # ── Section: Class Probabilities Table ────────────────────────────────
    _section_header(pdf, "Class Probability Breakdown")

    col_widths = [80, 50, 50]
    headers = ["Class", "Probability", "Confidence Bar"]
    _table_header(pdf, headers, col_widths)

    for i, (cname, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        c = CLASS_COLORS[cname]
        pct = prob * 100

        # Row background
        if i % 2 == 0:
            pdf.set_fill_color(245, 247, 250)
        else:
            pdf.set_fill_color(255, 255, 255)

        y_before = pdf.get_y()
        pdf.set_font("Helvetica", "B" if i == predicted_class else "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(col_widths[0], 8, f"  {cname}", fill=True)
        pdf.cell(col_widths[1], 8, f"{pct:.2f}%", fill=True, align="C")

        # Mini bar in last column
        cell_x = pdf.get_x()
        pdf.cell(col_widths[2], 8, "", fill=True)
        bar_max = col_widths[2] - 10
        pdf.set_fill_color(*c)
        bar_fill = max(1, bar_max * (pct / 100))
        pdf.rect(cell_x + 2, y_before + 2, bar_fill, 4, style="F")

        pdf.ln()

    pdf.ln(6)

    # ── Section: MRI Images ───────────────────────────────────────────────
    _section_header(pdf, "Scan Visualisation")

    img_w = 85
    img_h = 85
    margin = 10
    x_left = margin
    x_right = x_left + img_w + 5
    y_start = pdf.get_y()

    # Labels
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(x_left)
    pdf.cell(img_w, 6, "Original MRI", align="C")
    pdf.set_x(x_right)
    pdf.cell(img_w, 6, "Grad-CAM Attention Map", align="C")
    pdf.ln(6)

    y_img = pdf.get_y()
    pdf.image(str(mri_path),  x=x_left,  y=y_img, w=img_w, h=img_h)
    pdf.image(str(cam_path),  x=x_right, y=y_img, w=img_w, h=img_h)
    pdf.set_y(y_img + img_h + 4)

    # Caption
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 5,
        "The Grad-CAM heatmap highlights regions of the MRI the model weighted most "
        "heavily in its prediction. Red/yellow areas indicate high attention; blue areas "
        "indicate low attention.",
        align="C",
    )

    pdf.ln(4)

    # ── Section: Clinical Notes ───────────────────────────────────────────
    _section_header(pdf, "Clinical Notes & Limitations")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    notes = (
        "This report was generated by NeuroGAN, an AI research system trained on the NIAGADS "
        "Alzheimer's MRI dataset. The model achieves 95.26% accuracy (weighted F1 = 0.9528) on "
        "held-out validation data.\n\n"
        "The GAN-based data augmentation pipeline (NeuroGAN) was used to balance the minority "
        "class (VeryMildDemented) from 49 to a clinically representative distribution, improving "
        "model robustness.\n\n"
        "DISCLAIMER: This system is intended for research and educational demonstration purposes "
        "only. It has not been validated for clinical use and must not be used to make medical "
        "decisions. Always consult a qualified neurologist or radiologist."
    )
    pdf.multi_cell(0, 5, notes)

    # ── Output as bytes ───────────────────────────────────────────────────
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section_header(pdf: FPDF, title: str):
    pdf.set_fill_color(*DARK_BG)
    pdf.set_text_color(*ACCENT_COLOR)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, f"  {title.upper()}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)


def _info_row(pdf: FPDF, label: str, value: str):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(45, 6, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 6, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _table_header(pdf: FPDF, headers: list[str], col_widths: list[int]):
    pdf.set_fill_color(*DARK_BG)
    pdf.set_text_color(*ACCENT_COLOR)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, f"  {h}", fill=True)
    pdf.ln()
