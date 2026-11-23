import os
import sys
import time
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml")
sys.path.insert(0, r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\src")

from annotation_converter import DeepFashion2ToYOLOConverter
import config

annos_dir = config.TRAIN_ANNOTATION_DIR
images_dir = config.TRAIN_IMAGE_DIR

# Pick 1,000 sequential files from 1 to 1000
test_ids = [f"{i:06d}" for i in range(1, 1001)]

conv = DeepFashion2ToYOLOConverter()
out_test = Path(r"C:\Users\Lenovo\Downloads\Fitloop\scratch\bench_labels")
out_test.mkdir(parents=True, exist_ok=True)

t0 = time.time()
converted = 0
for s_id in test_ids:
    ap = annos_dir / f"{s_id}.json"
    ip = images_dir / f"{s_id}.jpg"
    lp = out_test / f"{s_id}.txt"
    res = conv.convert_single_annotation(ap, ip, lp)
    if res.valid_count > 0:
        converted += 1

elapsed = time.time() - t0
rate = len(test_ids) / elapsed
print(f"Sequential 1,000 files took: {elapsed:.2f}s ({rate:.1f} files/sec)")
print(f"Estimated time for 224,114 files at this rate: {224114 / rate / 60:.1f} minutes")

# Cleanup bench files
shutil.rmtree(out_test, ignore_errors=True)
