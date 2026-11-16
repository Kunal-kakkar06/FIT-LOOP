"""
Stage 6.5: Directory Junction Creator
=====================================
Creates zero-copy Windows NTFS directory junctions:
  deepfashion2_yolo/images/train -> C:\\Users\\Lenovo\\Downloads\\DeepFashion2\\train\\image
  deepfashion2_yolo/images/val   -> C:\\Users\\Lenovo\\Downloads\\DeepFashion2\\validation\\image

Checks:
- Targets exist and contain expected image counts
- Links do not already exist
- Verifies junction creation using os.path and PIL
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/data/processed/deepfashion2_yolo")
IMAGES_DIR = BASE_DIR / "images"
TRAIN_LINK = IMAGES_DIR / "train"
VAL_LINK = IMAGES_DIR / "val"

TRAIN_TARGET = Path("C:/Users/Lenovo/Downloads/DeepFashion2/train/image")
VAL_TARGET = Path("C:/Users/Lenovo/Downloads/DeepFashion2/validation/image")


def main():
    print("=" * 70)
    print("CREATING ZERO-COPY NTFS DIRECTORY JUNCTIONS FOR YOLO IMAGES")
    print("=" * 70)

    # 1. Pre-condition checks on targets
    assert TRAIN_TARGET.is_dir(), f"Train target {TRAIN_TARGET} is not a directory!"
    assert VAL_TARGET.is_dir(), f"Val target {VAL_TARGET} is not a directory!"

    train_cnt = len(os.listdir(TRAIN_TARGET))
    val_cnt = len(os.listdir(VAL_TARGET))
    print(f"Verified train target {TRAIN_TARGET}: {train_cnt:,} images")
    print(f"Verified val target   {VAL_TARGET}: {val_cnt:,} images")
    assert train_cnt == 191961, f"Expected 191,961 train images, found {train_cnt}"
    assert val_cnt == 32153, f"Expected 32,153 val images, found {val_cnt}"

    # 2. Ensure parent images directory exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Parent directory ready: {IMAGES_DIR}")

    # 3. Create train junction
    if TRAIN_LINK.exists():
        print(f"Train link already exists: {TRAIN_LINK}")
    else:
        cmd = f'cmd /c mklink /J "{TRAIN_LINK}" "{TRAIN_TARGET}"'
        print(f"Executing: {cmd}")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("Output:", res.stdout.strip())
        if res.returncode != 0:
            print("Error:", res.stderr.strip())
            sys.exit(res.returncode)

    # 4. Create val junction
    if VAL_LINK.exists():
        print(f"Val link already exists: {VAL_LINK}")
    else:
        cmd = f'cmd /c mklink /J "{VAL_LINK}" "{VAL_TARGET}"'
        print(f"Executing: {cmd}")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("Output:", res.stdout.strip())
        if res.returncode != 0:
            print("Error:", res.stderr.strip())
            sys.exit(res.returncode)

    # 5. Verification
    print("\n--- Verifying Junctions ---")
    train_link_cnt = len(os.listdir(TRAIN_LINK))
    val_link_cnt = len(os.listdir(VAL_LINK))
    print(f"Accessible via junction {TRAIN_LINK}: {train_link_cnt:,} images")
    print(f"Accessible via junction {VAL_LINK}:   {val_link_cnt:,} images")
    assert train_link_cnt == 191961, f"Junction verification failed for train: {train_link_cnt}"
    assert val_link_cnt == 32153, f"Junction verification failed for val: {val_link_cnt}"

    # Verify sample file access
    sample_train = TRAIN_LINK / "000001.jpg"
    assert sample_train.is_file(), f"Sample train image {sample_train} not found!"
    print(f"Sample train image accessible: {sample_train} ({sample_train.stat().st_size:,} bytes)")

    sample_val = VAL_LINK / "000001.jpg"
    assert sample_val.is_file(), f"Sample val image {sample_val} not found!"
    print(f"Sample val image accessible:   {sample_val} ({sample_val.stat().st_size:,} bytes)")

    print("\n[OK] Zero-copy directory junctions created and verified successfully!")


if __name__ == "__main__":
    main()
