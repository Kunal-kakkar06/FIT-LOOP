import os
import sys
import shutil
from pathlib import Path

# Add project root and src to path
ROOT = Path(r"C:\Users\Lenovo\Downloads\Fitloop")
sys.path.insert(0, str(ROOT / "wardrobe_ml"))
sys.path.insert(0, str(ROOT / "wardrobe_ml" / "src"))

import config
from annotation_converter import DeepFashion2ToYOLOConverter
import run_conversion_test

print("=== TASK 1: PRE-CONVERSION AUDIT ===")

# 1. Inspect category mapping in config
print("\n1. Checking Category Mapping in config.py:")
assert len(config.DF2_CATEGORIES) == 13, f"Expected 13 categories, got {len(config.DF2_CATEGORIES)}"
assert len(config.DF2_TO_YOLO_CLASS_MAP) == 13, f"Expected 13 mappings, got {len(config.DF2_TO_YOLO_CLASS_MAP)}"
assert len(config.YOLO_CLASS_NAMES) == 13, f"Expected 13 class names, got {len(config.YOLO_CLASS_NAMES)}"

for df2_id, yolo_id in config.DF2_TO_YOLO_CLASS_MAP.items():
    name = config.DF2_CATEGORIES[df2_id]
    assert yolo_id == df2_id - 1, f"Mismatch: df2 {df2_id} -> yolo {yolo_id}"
    print(f"  DF2 {df2_id:2d} -> YOLO {yolo_id:2d}: {name}")
print("[OK] Category mapping verified: exact 1-to-1 match (DF2 1..13 -> YOLO 0..12)")

# 2. Re-verify validation rules in converter
print("\n2. Checking Converter Validation Rules:")
conv = DeepFashion2ToYOLOConverter()
# Category validation
assert conv.validate_category(1) == (True, 0, "")
assert conv.validate_category(13) == (True, 12, "")
assert conv.validate_category(0)[0] is False
assert conv.validate_category(14)[0] is False
assert conv.validate_category(None)[0] is False
assert conv.validate_category(True)[0] is False
assert conv.validate_category(False)[0] is False
# BBox validation
w, h = 1000, 1000
assert conv.validate_and_normalize_box([0, 0, 0, 0], w, h)[0] is False
assert conv.validate_and_normalize_box([100, 100, 100, 200], w, h)[0] is False
assert conv.validate_and_normalize_box([200, 200, 100, 100], w, h)[0] is False
assert conv.validate_and_normalize_box([-1, 0, 100, 100], w, h)[0] is False
assert conv.validate_and_normalize_box([0, 0, 1001, 100], w, h)[0] is False
assert conv.validate_and_normalize_box([0, 0, 100, float('nan')], w, h)[0] is False
assert conv.validate_and_normalize_box([0, 0, 100, float('inf')], w, h)[0] is False
assert conv.validate_and_normalize_box([0, 0, 100], w, h)[0] is False
print("[OK] All validation rules verified intact")

# 3. Check 20-sample test output preservation & visual QA preservation
print("\n3. Checking Previous Outputs Preservation:")
qa_dir = ROOT / "wardrobe_ml" / "outputs" / "step_6_4b_visual_qa"
assert qa_dir.exists(), f"Visual QA dir missing: {qa_dir}"
assert (qa_dir / "visual_qa_report.json").exists()
assert (ROOT / "wardrobe_ml" / "outputs" / "step_6_4b_visual_qa_report.json").exists()
assert (ROOT / "wardrobe_ml" / "outputs" / "step_6_4a_conversion_test_report.json").exists()
sample_test_dir = ROOT / "wardrobe_ml" / "data" / "processed" / "yolo_conversion_test"
assert sample_test_dir.exists()
print(f"[OK] Visual QA and 6.4A test outputs verified preserved (NOT to be overwritten)")

# 4. Check Dataset Directories
print("\n4. Checking DeepFashion2 Source Directories:")
train_annos = config.TRAIN_ANNOTATION_DIR
train_images = config.TRAIN_IMAGE_DIR
val_annos = config.VALIDATION_ANNOTATION_DIR
val_images = config.VALIDATION_IMAGE_DIR

print(f"  Train annos: {train_annos} (exists: {train_annos.exists()})")
print(f"  Train image: {train_images} (exists: {train_images.exists()})")
print(f"  Val annos:   {val_annos} (exists: {val_annos.exists()})")
print(f"  Val image:   {val_images} (exists: {val_images.exists()})")

# 5. Check Disk Space & Output Size Estimation
print("\n5. Checking Disk Space & Output Estimation:")
total, used, free = shutil.disk_usage("C:")
free_gb = free / (1024**3)
print(f"  Drive C: Free space = {free_gb:.2f} GB")

# Estimation:
# Labeled images: 191,961 (train) + 32,153 (val) = 224,114 TXT files.
# Each YOLO label file is ~40-150 bytes (1-4 lines of text).
# Average file size on disk with 4KB cluster size:
# 224,114 files * 4 KB = ~896 MB on disk (raw text content is ~15-20 MB).
est_mb_content = 224114 * 70 / (1024 * 1024)
est_mb_disk = 224114 * 4 / 1024
print(f"  Estimated total label files: 224,114")
print(f"  Estimated raw content size: ~{est_mb_content:.1f} MB")
print(f"  Estimated disk space required (with 4KB cluster overhead): ~{est_mb_disk:.1f} MB (~0.88 GB)")
print(f"  Available space ({free_gb:.2f} GB) is ~{free_gb / 0.88:.0f}x larger than required.")
assert free_gb > 10.0, f"Low disk space: {free_gb} GB"
print("[OK] Ample disk space confirmed.")

print("\n=== TASK 1 AUDIT: ALL PRE-CONDITIONS PASSED ===")
