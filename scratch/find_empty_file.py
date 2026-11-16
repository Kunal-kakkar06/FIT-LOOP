import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\Lenovo\Downloads\Fitloop")
sys.path.insert(0, str(ROOT / "wardrobe_ml"))
from config import TRAIN_ANNOTATION_DIR, TRAIN_IMAGE_DIR
from PIL import Image

train_labels_dir = ROOT / "wardrobe_ml" / "data" / "processed" / "deepfashion2_yolo" / "labels" / "train"

# Find empty files
empty_files = []
for fname in os.listdir(train_labels_dir):
    p = train_labels_dir / fname
    if os.path.getsize(p) == 0:
        empty_files.append(fname)

print(f"Empty label files in Train: {empty_files}")
for ef in empty_files:
    image_id = ef[:-4]
    ap = TRAIN_ANNOTATION_DIR / f"{image_id}.json"
    ip = TRAIN_IMAGE_DIR / f"{image_id}.jpg"
    with open(ap, "r") as f:
        d = json.load(f)
    with Image.open(ip) as im:
        w, h = im.size
    print(f"  Empty file {ef} (Image size: {w}x{h}):")
    print(json.dumps(d, indent=2))
