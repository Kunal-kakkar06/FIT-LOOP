#!/usr/bin/env python3
"""
FitLoop Dataset Splitting & In-Memory Packaging Script
======================================================
Splits large YOLO training & validation archives into atomic, independently extractable
~900 MiB ZIP archives suitable for upload to Kaggle.

Key Invariants:
1. Co-locates images and labels in the same archive (never splits a pair).
2. Uses in-memory streaming: zero full-dataset extraction to disk.
3. Safe atomic writes (.tmp -> testzip() -> rename).
4. Generates dataset_manifest.json, splitting_report.txt, and Kaggle-configured data.yaml.
"""

import hashlib
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
WORKSPACE_ROOT = Path(r"C:\Users\Lenovo\Downloads\Fitloop")
STAGING_DIR = WORKSPACE_ROOT / "wardrobe_ml" / "outputs" / "colab_transfer_staging"

SRC_TRAIN_IMAGES_ZIP = STAGING_DIR / "fitloop_train_images.zip"
SRC_VAL_IMAGES_ZIP = STAGING_DIR / "fitloop_val_images.zip"
SRC_LABELS_ZIP = STAGING_DIR / "fitloop_yolo_labels.zip"
SRC_MANIFESTS_ZIP = STAGING_DIR / "fitloop_yolo_manifests.zip"

OUTPUT_DIR = WORKSPACE_ROOT / "wardrobe_ml" / "outputs" / "kaggle_dataset_parts"
TARGET_PART_BYTES = 900 * 1024 * 1024  # 900 MiB threshold

EXPECTED_TRAIN_COUNT = 191958
EXPECTED_VAL_COUNT = 32153


def calculate_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file in 1 MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def preload_labels(labels_zip_path: Path) -> Tuple[Dict[str, bytes], Dict[str, bytes]]:
    """Preload all training and validation YOLO label files into memory (~14 MB)."""
    print(f"Preloading labels from {labels_zip_path.name}...")
    t0 = time.time()
    train_labels = {}
    val_labels = {}

    with zipfile.ZipFile(labels_zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.startswith("labels/train/") and info.filename.endswith(".txt"):
                stem = Path(info.filename).stem
                train_labels[stem] = zf.read(info)
            elif info.filename.startswith("labels/val/") and info.filename.endswith(".txt"):
                stem = Path(info.filename).stem
                val_labels[stem] = zf.read(info)

    t1 = time.time()
    print(f"Loaded {len(train_labels):,} train labels and {len(val_labels):,} val labels in {t1 - t0:.2f}s.")
    return train_labels, val_labels


def process_split(
    split_name: str,
    images_zip_path: Path,
    labels_dict: Dict[str, bytes],
    expected_count: int,
    output_dir: Path,
    target_bytes: int,
) -> List[Dict]:
    """
    Stream images and labels from source zip and memory into atomic ~900 MiB partitions.
    """
    print(f"\n{'='*70}\nProcessing {split_name.upper()} split from {images_zip_path.name}\n{'='*70}")
    t_start = time.time()

    with zipfile.ZipFile(images_zip_path, "r") as zf_img:
        infolist = [info for info in zf_img.infolist() if not info.is_dir() and not info.filename.endswith("/")]
        actual_img_count = len(infolist)
        print(f"Found {actual_img_count:,} image entries in {images_zip_path.name}.")

        if actual_img_count != expected_count:
            raise RuntimeError(
                f"STOP CONDITION: {split_name} image count mismatch! Expected {expected_count:,}, found {actual_img_count:,}."
            )

        # Pre-verify that every image has a matching label
        missing_labels = []
        for info in infolist:
            stem = Path(info.filename).stem
            if stem not in labels_dict:
                missing_labels.append(info.filename)

        if missing_labels:
            raise RuntimeError(
                f"STOP CONDITION: {len(missing_labels)} {split_name} images lack corresponding labels! First 5: {missing_labels[:5]}"
            )

        print(f"Verified 100% label correspondence for all {actual_img_count:,} {split_name} images.")

        part_num = 1
        part_records = []
        current_part_images = 0
        current_part_uncompressed = 0
        
        # Suffix and naming convention
        def get_part_paths(num: int) -> Tuple[Path, Path]:
            final_name = f"{split_name}_part_{num:02d}.zip"
            tmp_name = f"{split_name}_part_{num:02d}.zip.tmp"
            return output_dir / final_name, output_dir / tmp_name

        final_path, tmp_path = get_part_paths(part_num)
        print(f"\nStarting partition {part_num:02d}: {final_path.name}...")
        current_zf = zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED)

        for idx, img_info in enumerate(infolist, start=1):
            stem = Path(img_info.filename).stem
            img_bytes = zf_img.read(img_info)
            lbl_bytes = labels_dict[stem]
            lbl_filename = f"labels/{split_name}/{stem}.txt"

            # Check if adding this pair exceeds target partition size
            # (only split if current partition already has at least one pair)
            pair_bytes = len(img_bytes) + len(lbl_bytes) + 240  # Member data + zip directory headers
            if (current_part_uncompressed + pair_bytes > target_bytes) and (current_part_images > 0):
                # Close current partition
                current_zf.close()
                print(f"Closing {tmp_path.name} ({current_part_images:,} pairs, {current_part_uncompressed / (1024**2):.2f} MiB uncompressed)...")

                # Verify archive integrity
                print(f"Running testzip() integrity audit on {tmp_path.name}...")
                with zipfile.ZipFile(tmp_path, "r") as test_zf:
                    bad = test_zf.testzip()
                    if bad is not None:
                        raise RuntimeError(f"STOP CONDITION: Archive {tmp_path.name} corrupted at member: {bad}")

                # Atomic rename
                if final_path.exists():
                    final_path.unlink()
                os.replace(tmp_path, final_path)

                # Compute SHA-256 and gather metadata
                print(f"Computing SHA-256 for {final_path.name}...")
                part_sha256 = calculate_sha256(final_path)
                compressed_size = final_path.stat().st_size
                print(f"Completed {final_path.name}: {compressed_size / (1024**2):.2f} MiB, SHA256: {part_sha256[:16]}...")

                part_records.append({
                    "filename": final_path.name,
                    "split": split_name,
                    "part_number": part_num,
                    "image_count": current_part_images,
                    "label_count": current_part_images,
                    "total_files": current_part_images * 2,
                    "uncompressed_bytes": current_part_uncompressed,
                    "compressed_bytes": compressed_size,
                    "sha256": part_sha256,
                    "integrity_testzip_passed": True,
                })

                # Start next partition
                part_num += 1
                final_path, tmp_path = get_part_paths(part_num)
                print(f"\nStarting partition {part_num:02d}: {final_path.name}...")
                current_zf = zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED)
                current_part_images = 0
                current_part_uncompressed = 0

            # Write image to partition
            out_img_info = zipfile.ZipInfo(img_info.filename, date_time=img_info.date_time)
            out_img_info.compress_type = zipfile.ZIP_STORED
            current_zf.writestr(out_img_info, img_bytes)

            # Write label to partition
            out_lbl_info = zipfile.ZipInfo(lbl_filename, date_time=img_info.date_time)
            out_lbl_info.compress_type = zipfile.ZIP_DEFLATED
            current_zf.writestr(out_lbl_info, lbl_bytes)

            current_part_images += 1
            current_part_uncompressed += pair_bytes

            if idx % 10000 == 0 or idx == actual_img_count:
                print(f"  Processed {idx:,}/{actual_img_count:,} pairs ({idx/actual_img_count*100:.1f}%)...")

        # Finalize the last partition
        if current_part_images > 0:
            current_zf.close()
            print(f"Closing final partition {tmp_path.name} ({current_part_images:,} pairs)...")

            print(f"Running testzip() integrity audit on {tmp_path.name}...")
            with zipfile.ZipFile(tmp_path, "r") as test_zf:
                bad = test_zf.testzip()
                if bad is not None:
                    raise RuntimeError(f"STOP CONDITION: Archive {tmp_path.name} corrupted at member: {bad}")

            if final_path.exists():
                final_path.unlink()
            os.replace(tmp_path, final_path)

            print(f"Computing SHA-256 for {final_path.name}...")
            part_sha256 = calculate_sha256(final_path)
            compressed_size = final_path.stat().st_size
            print(f"Completed {final_path.name}: {compressed_size / (1024**2):.2f} MiB, SHA256: {part_sha256[:16]}...")

            part_records.append({
                "filename": final_path.name,
                "split": split_name,
                "part_number": part_num,
                "image_count": current_part_images,
                "label_count": current_part_images,
                "total_files": current_part_images * 2,
                "uncompressed_bytes": current_part_uncompressed,
                "compressed_bytes": compressed_size,
                "sha256": part_sha256,
                "integrity_testzip_passed": True,
            })

    total_pairs = sum(r["image_count"] for r in part_records)
    t_elapsed = time.time() - t_start
    print(f"\n{split_name.upper()} split finished: {total_pairs:,} pairs across {len(part_records)} archives in {t_elapsed:.2f}s.")
    return part_records


def generate_kaggle_data_yaml(manifests_zip_path: Path, output_dir: Path):
    """Extract and configure data.yaml for Kaggle environments."""
    print("\nConfiguring data.yaml for Kaggle...")
    with zipfile.ZipFile(manifests_zip_path, "r") as zf:
        raw_yaml = zf.read("data.yaml").decode("utf-8")
        raw_train_txt = zf.read("train.txt")
        raw_val_txt = zf.read("val.txt")

    # Generate standardized data.yaml
    kaggle_yaml_content = """# DeepFashion2 Garment Detection Dataset Configuration for YOLO
# FitLoop Pipeline - Kaggle Dataset Sharded Package
# ------------------------------------------------------------
# Total Eligible Training Images: 191,958 (excludes 3 anomalous images)
# Total Validation Images:        32,153
# Total Categories:               13 (0-indexed)

# Dataset root directory on Kaggle:
# If extracting all archives into /kaggle/working/dataset:
path: /kaggle/working/dataset

# Relative subpaths to images
train: images/train
val: images/val

# 13 DeepFashion2 garment classes
nc: 13
names:
  0: short sleeve top
  1: long sleeve top
  2: short sleeve outwear
  3: long sleeve outwear
  4: vest
  5: sling
  6: shorts
  7: trousers
  8: skirt
  9: short sleeve dress
  10: long sleeve dress
  11: vest dress
  12: sling dress
"""
    yaml_out_path = output_dir / "data.yaml"
    with open(yaml_out_path, "w", encoding="utf-8") as f:
        f.write(kaggle_yaml_content)
    print(f"Wrote {yaml_out_path.name} successfully.")

    # Also save train.txt and val.txt manifests
    with open(output_dir / "train.txt", "wb") as f:
        f.write(raw_train_txt)
    with open(output_dir / "val.txt", "wb") as f:
        f.write(raw_val_txt)
    print("Wrote auxiliary train.txt and val.txt manifests.")


def generate_reports(
    all_partitions: List[Dict],
    output_dir: Path,
    t_total_seconds: float,
):
    """Write dataset_manifest.json and splitting_report.txt."""
    print("\nGenerating dataset_manifest.json and splitting_report.txt...")

    train_partitions = [p for p in all_partitions if p["split"] == "train"]
    val_partitions = [p for p in all_partitions if p["split"] == "val"]

    total_train_images = sum(p["image_count"] for p in train_partitions)
    total_train_labels = sum(p["label_count"] for p in train_partitions)
    total_val_images = sum(p["image_count"] for p in val_partitions)
    total_val_labels = sum(p["label_count"] for p in val_partitions)

    total_uncompressed = sum(p["uncompressed_bytes"] for p in all_partitions)
    total_compressed = sum(p["compressed_bytes"] for p in all_partitions)

    manifest_data = {
        "title": "FitLoop DeepFashion2 YOLO Sharded Kaggle Dataset Package",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_partition_bytes": TARGET_PART_BYTES,
        "target_partition_mib": 900,
        "total_execution_time_seconds": round(t_total_seconds, 2),
        "dataset_summary": {
            "total_train_images": total_train_images,
            "total_train_labels": total_train_labels,
            "total_val_images": total_val_images,
            "total_val_labels": total_val_labels,
            "total_images": total_train_images + total_val_images,
            "total_labels": total_train_labels + total_val_labels,
            "total_files": (total_train_images + total_val_images) * 2,
            "total_train_partitions": len(train_partitions),
            "total_val_partitions": len(val_partitions),
            "total_partitions": len(all_partitions),
            "total_uncompressed_bytes": total_uncompressed,
            "total_uncompressed_gib": round(total_uncompressed / (1024**3), 3),
            "total_compressed_bytes": total_compressed,
            "total_compressed_gib": round(total_compressed / (1024**3), 3),
        },
        "partitions": all_partitions,
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Wrote {manifest_path.name}.")

    report_lines = [
        "FITLOOP YOLO DATASET SHARDING REPORT (KAGGLE PACKAGE)",
        "=" * 65,
        f"Generated At: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Execution Time: {t_total_seconds:.2f} seconds ({t_total_seconds/60:.2f} minutes)",
        "",
        "DATASET TOTALS",
        "-" * 65,
        f"Total Train Image-Label Pairs: {total_train_images:,}",
        f"Total Val Image-Label Pairs:   {total_val_images:,}",
        f"Total Archived Files:          {(total_train_images + total_val_images) * 2:,}",
        f"Total Compressed Size:         {total_compressed / (1024**3):.2f} GiB ({total_compressed:,} bytes)",
        "",
        "GENERATED ARCHIVES INVENTORY",
        "-" * 65,
    ]

    for p in all_partitions:
        sz_mib = p["compressed_bytes"] / (1024**2)
        report_lines.append(
            f"{p['filename']:<20} | Pairs: {p['image_count']:>6,} ({p['total_files']:>6,} files) | "
            f"Size: {sz_mib:>7.2f} MiB | SHA256: {p['sha256'][:12]}... [VERIFIED]"
        )

    report_lines.extend([
        "",
        "KAGGLE UNPACKING & STRUCTURED PHASED TRAINING INSTRUCTIONS",
        "-" * 65,
        "1. Upload all generated .zip files and data.yaml to a Kaggle Dataset.",
        "2. In your Kaggle training notebook, unpack all archives into a common destination:",
        "",
        "   !mkdir -p /kaggle/working/dataset",
        "   !unzip -q -o '/kaggle/input/<dataset-slug>/train_part_*.zip' -d /kaggle/working/dataset",
        "   !unzip -q -o '/kaggle/input/<dataset-slug>/val_part_*.zip' -d /kaggle/working/dataset",
        "   !cp /kaggle/input/<dataset-slug>/data.yaml /kaggle/working/dataset/data.yaml",
        "",
        "3. FitLoop Standard Architecture: YOLO26n (yolo26n.pt, ~2.5M params)",
        "   To prevent Kaggle 12-hour session timeouts on 192k images, train in structured",
        "   10-epoch phases with checkpointing enabled at every epoch (save_period=1):",
        "",
        "   # Phase 1: Epochs 1-10",
        "   yolo detect train \\",
        "       data=/kaggle/working/dataset/data.yaml \\",
        "       model=yolo26n.pt \\",
        "       epochs=10 \\",
        "       imgsz=640 \\",
        "       batch=32 \\",
        "       save_period=1 \\",
        "       device=0 \\",
        "       project=/kaggle/working/fitloop_runs \\",
        "       name=yolo26n_deepfashion2_phase1",
        "",
        "   # Phase 2 (Resume in next session from checkpoint):",
        "   # Mount previous session's output dataset containing weights/last.pt",
        "   yolo detect train resume model=/kaggle/input/<phase1-output-slug>/last.pt epochs=20",
        "=" * 65,
    ])

    report_path = output_dir / "splitting_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote {report_path.name}.")


def main():
    t_global_start = time.time()
    print("FitLoop Dataset Splitting & Packaging Pipeline Initializing...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preload all labels
    train_labels, val_labels = preload_labels(SRC_LABELS_ZIP)

    # Split train set
    train_records = process_split(
        split_name="train",
        images_zip_path=SRC_TRAIN_IMAGES_ZIP,
        labels_dict=train_labels,
        expected_count=EXPECTED_TRAIN_COUNT,
        output_dir=OUTPUT_DIR,
        target_bytes=TARGET_PART_BYTES,
    )

    # Split val set
    val_records = process_split(
        split_name="val",
        images_zip_path=SRC_VAL_IMAGES_ZIP,
        labels_dict=val_labels,
        expected_count=EXPECTED_VAL_COUNT,
        output_dir=OUTPUT_DIR,
        target_bytes=TARGET_PART_BYTES,
    )

    all_partitions = train_records + val_records

    # Generate Kaggle data.yaml
    generate_kaggle_data_yaml(SRC_MANIFESTS_ZIP, OUTPUT_DIR)

    # Generate manifest and reports
    t_total = time.time() - t_global_start
    generate_reports(all_partitions, OUTPUT_DIR, t_total)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {t_total:.2f} SECONDS!")
    print(f"All {len(all_partitions)} archives generated, verified, and cataloged in {OUTPUT_DIR}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
