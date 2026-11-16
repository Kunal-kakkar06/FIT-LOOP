import os
import sys
import json
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\Lenovo\Downloads\Fitloop")
sys.path.insert(0, str(ROOT / "wardrobe_ml"))
sys.path.insert(0, str(ROOT / "wardrobe_ml" / "src"))

from config import TRAIN_ANNOTATION_DIR, TRAIN_IMAGE_DIR, DF2_CATEGORIES
from annotation_converter import DeepFashion2ToYOLOConverter

train_labels_dir = ROOT / "wardrobe_ml" / "data" / "processed" / "deepfashion2_yolo" / "labels" / "train"

# 1. Find empty label files
empty_files = []
for fname in os.listdir(train_labels_dir):
    p = train_labels_dir / fname
    if os.path.getsize(p) == 0:
        empty_files.append(fname)

print(f"Empty label files found: {len(empty_files)}")
for ef in empty_files:
    image_id = ef[:-4]
    ap = TRAIN_ANNOTATION_DIR / f"{image_id}.json"
    ip = TRAIN_IMAGE_DIR / f"{image_id}.jpg"
    with open(ap, "r", encoding="utf-8") as f:
        d = json.load(f)
    with Image.open(ip) as im:
        w, h = im.size
    print(f"  Empty file: {ef} | image size: ({w}, {h}) | anno: {d}")

# 2. Find all invalid instances across training
conv = DeepFashion2ToYOLOConverter()
print("\nScanning for invalid instances across all training annotations...")
invalid_list = []
for fname in os.listdir(TRAIN_ANNOTATION_DIR):
    if not fname.endswith(".json"):
        continue
    image_id = fname[:-5]
    ap = TRAIN_ANNOTATION_DIR / fname
    ip = TRAIN_IMAGE_DIR / f"{image_id}.jpg"
    with open(ap, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Quick check: check if any item fails
    items = [k for k in data if k.startswith("item")]
    for k in items:
        item = data[k]
        cat_id = item.get("category_id")
        bbox = item.get("bounding_box")
        ok_cat, _, _ = conv.validate_category(cat_id)
        if not ok_cat:
            invalid_list.append((image_id, k, "invalid_category", cat_id, bbox))
            continue
        # Bbox check
        if bbox is None or len(bbox) != 4:
            invalid_list.append((image_id, k, "malformed_bbox", cat_id, bbox))
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            invalid_list.append((image_id, k, "zero_or_negative_area", cat_id, bbox))
            continue
        # Check image bounds
        with Image.open(ip) as im:
            w, h = im.size
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            invalid_list.append((image_id, k, "out_of_bounds", cat_id, bbox, (w, h)))

print(f"Total invalid instances found in Train: {len(invalid_list)}")
for inv in invalid_list:
    print(" ", inv)
