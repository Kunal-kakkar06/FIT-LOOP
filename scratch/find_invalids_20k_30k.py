import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\Lenovo\Downloads\Fitloop")
sys.path.insert(0, str(ROOT / "wardrobe_ml"))
sys.path.insert(0, str(ROOT / "wardrobe_ml" / "src"))

from config import TRAIN_ANNOTATION_DIR, TRAIN_IMAGE_DIR
from annotation_converter import DeepFashion2ToYOLOConverter

conv = DeepFashion2ToYOLOConverter()
source_files = sorted(os.listdir(TRAIN_ANNOTATION_DIR))

# Search between index 20,000 and 30,000
for f in source_files[20000:30000]:
    if not f.endswith(".json"):
        continue
    image_id = f[:-5]
    ap = TRAIN_ANNOTATION_DIR / f
    ip = TRAIN_IMAGE_DIR / f"{image_id}.jpg"
    res = conv.convert_single_annotation(ap, ip)
    if res.invalid_count > 0:
        print(f"Found invalid in {image_id}:")
        for inv in res.invalid_instances:
            print(f"  {inv.item_key}: reason={inv.reason}, bbox={inv.raw_bbox}, details={inv.details}")
