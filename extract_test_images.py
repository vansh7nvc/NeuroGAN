# extract_test_images.py — Extract sample MRI images from train.parquet for testing
import io, os, pandas as pd
from PIL import Image

PARQUET = r"MRI Dataset-20251203T142309Z-1-001\MRI Dataset\train.parquet"
OUT_DIR = "test_images"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_bytes(blob):
    if isinstance(blob, dict):
        for key in ("bytes","data","image"):
            if key in blob and isinstance(blob[key],(bytes,bytearray)): return blob[key]
    return blob

print("Loading parquet (fast - just first 20 rows)...")
df = pd.read_parquet(PARQUET).head(20)

label_names = {0:"NonDemented", 1:"VeryMildDemented", 2:"MildDemented", 3:"ModerateDemented"}
saved = {}

for _, row in df.iterrows():
    label = row["label"]
    if label in saved:  continue   # one per class
    raw = extract_bytes(row["image"])
    img = Image.open(io.BytesIO(raw)).convert("L")
    fname = f"{OUT_DIR}/test_{label_names[label]}.png"
    img.save(fname)
    saved[label] = fname
    print(f"  Saved: {fname}  ({img.size})")
    if len(saved) == 4: break

# If we didn't get all 4 classes from first 20 rows, take more
if len(saved) < 4:
    print("Getting remaining classes from full dataset...")
    df_full = pd.read_parquet(PARQUET)
    for _, row in df_full.iterrows():
        label = row["label"]
        if label in saved: continue
        raw = extract_bytes(row["image"])
        img = Image.open(io.BytesIO(raw)).convert("L")
        fname = f"{OUT_DIR}/test_{label_names[label]}.png"
        img.save(fname)
        saved[label] = fname
        print(f"  Saved: {fname}  ({img.size})")
        if len(saved) == 4: break

print(f"\nDone! {len(saved)} test images saved to '{OUT_DIR}/'")
for l, f in saved.items():
    print(f"  Class {l} ({label_names[l]}): {f}")
