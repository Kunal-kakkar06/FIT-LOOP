"""
Stage 6.5: Dataset Structure & Correspondence Verification
==========================================================
Verifies Task 1:
- 191,961 original training images
- 32,153 original validation images
- 191,961 training label files
- 32,153 validation label files
- Filename correspondence between images and labels
- Verification of training exclusion manifest (exactly 3 excluded images)
- Verification that no validation images are excluded
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

TRAIN_IMG_DIR = Path("C:/Users/Lenovo/Downloads/DeepFashion2/train/image")
VAL_IMG_DIR = Path("C:/Users/Lenovo/Downloads/DeepFashion2/validation/image")
TRAIN_LBL_DIR = Path("C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/data/processed/deepfashion2_yolo/labels/train")
VAL_LBL_DIR = Path("C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/data/processed/deepfashion2_yolo/labels/val")
EXCLUSION_CSV = Path("C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_4c2_training_exclusion_manifest.csv")


def main():
    print("=" * 70)
    print("TASK 1: INSPECT CURRENT DIRECTORY STRUCTURE & CORRESPONDENCE")
    print("=" * 70)

    # 1. Image file counts
    train_images = set(os.listdir(TRAIN_IMG_DIR))
    val_images = set(os.listdir(VAL_IMG_DIR))
    print(f"Train images found: {len(train_images):,} (expected 191,961)")
    print(f"Val images found:   {len(val_images):,} (expected 32,153)")
    assert len(train_images) == 191961, f"Expected 191,961 train images, got {len(train_images)}"
    assert len(val_images) == 32153, f"Expected 32,153 val images, got {len(val_images)}"

    # 2. Label file counts
    train_labels = set(os.listdir(TRAIN_LBL_DIR))
    val_labels = set(os.listdir(VAL_LBL_DIR))
    print(f"Train labels found: {len(train_labels):,} (expected 191,961)")
    print(f"Val labels found:   {len(val_labels):,} (expected 32,153)")
    assert len(train_labels) == 191961, f"Expected 191,961 train labels, got {len(train_labels)}"
    assert len(val_labels) == 32153, f"Expected 32,153 val labels, got {len(val_labels)}"

    # 3. Exact 1-to-1 Filename Correspondence
    train_img_stems = {f[:-4] for f in train_images if f.endswith(".jpg")}
    train_lbl_stems = {f[:-4] for f in train_labels if f.endswith(".txt")}
    val_img_stems = {f[:-4] for f in val_images if f.endswith(".jpg")}
    val_lbl_stems = {f[:-4] for f in val_labels if f.endswith(".txt")}

    train_missing_labels = train_img_stems - train_lbl_stems
    train_extra_labels = train_lbl_stems - train_img_stems
    val_missing_labels = val_img_stems - val_lbl_stems
    val_extra_labels = val_lbl_stems - val_img_stems

    print(f"Train stems match: {train_img_stems == train_lbl_stems} (Missing: {len(train_missing_labels)}, Extra: {len(train_extra_labels)})")
    print(f"Val stems match:   {val_img_stems == val_lbl_stems} (Missing: {len(val_missing_labels)}, Extra: {len(val_extra_labels)})")
    assert len(train_missing_labels) == 0 and len(train_extra_labels) == 0, "Train correspondence mismatch!"
    assert len(val_missing_labels) == 0 and len(val_extra_labels) == 0, "Val correspondence mismatch!"

    # 4. Check Exclusion Manifest
    with open(EXCLUSION_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        excluded_rows = list(reader)

    excluded_filenames = [r["image_filename"] for r in excluded_rows]
    excluded_splits = [r["original_split"] for r in excluded_rows]

    print(f"\nExclusion Manifest ({EXCLUSION_CSV}):")
    print(f"  Total excluded images listed: {len(excluded_rows)}")
    for r in excluded_rows:
        print(f"  - {r['image_filename']} (Split: {r['original_split']}) -> {r['reason_for_exclusion']}")

    expected_exclusions = {"028225.jpg", "115063.jpg", "101338.jpg"}
    assert set(excluded_filenames) == expected_exclusions, f"Expected {expected_exclusions}, got {set(excluded_filenames)}"
    assert all(s == "train" for s in excluded_splits), "Exclusion manifest contains non-train splits!"
    print("  Exclusion manifest verification: EXACT MATCH (all 3 train images, 0 val images)")

    # 5. Baseline Training Eligible Count
    eligible_train_count = len(train_images) - len(expected_exclusions)
    print(f"\nEligible Baseline Training Images: {eligible_train_count:,} ({len(train_images):,} - {len(expected_exclusions)})")
    print(f"Eligible Validation Images:        {len(val_images):,} (100.0%)")
    print("\n[OK] Task 1 verification PASSED!")


if __name__ == "__main__":
    main()
