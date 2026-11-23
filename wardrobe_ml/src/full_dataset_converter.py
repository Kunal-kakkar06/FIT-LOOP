"""
Stage 6.4C: Full DeepFashion2 Annotation Conversion
===================================================
Converts all 191,961 training and 32,153 validation DeepFashion2 JSON
annotations into normalized YOLO detection TXT label files.

Architecture:
-------------
- Output directory: wardrobe_ml/data/processed/deepfashion2_yolo/
  - labels/train/
  - labels/val/
- Streaming batch multiprocessing via ProcessPoolExecutor (chunked processing)
- Memory-safe: Never loads entire dataset into memory at once
- Strict validation: Rejects zero/negative area, out-of-bounds, non-numeric,
  invalid category IDs. Never silently clips. Preserves valid instances.
- Reconciles valid train instances with Phase 1 ground truth (312,186 instances).
- Full audit report written to:
  wardrobe_ml/outputs/step_6_4c_full_conversion_report.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import psutil

# Ensure unbuffered output and utf-8 encoding so task logs show live progress cleanly
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

# Add project root and src to path
_CURRENT_DIR = Path(__file__).resolve().parent
_ML_ROOT = _CURRENT_DIR.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from config import (
    TRAIN_ANNOTATION_DIR,
    TRAIN_IMAGE_DIR,
    VALIDATION_ANNOTATION_DIR,
    VALIDATION_IMAGE_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    DF2_CATEGORIES,
    DF2_TO_YOLO_CLASS_MAP,
    YOLO_CLASS_NAMES,
)
from annotation_converter import DeepFashion2ToYOLOConverter


def _convert_chunk_worker(
    chunk_ids: List[str],
    annos_dir_str: str,
    images_dir_str: str,
    labels_dir_str: str,
) -> Dict[str, Any]:
    """
    Worker function to process a batch of image IDs in an isolated process.
    Reads each JSON annotation and corresponding JPG image dimensions,
    validates and normalizes boxes, and writes the YOLO TXT label file.
    """
    annos_dir = Path(annos_dir_str)
    images_dir = Path(images_dir_str)
    labels_dir = Path(labels_dir_str)

    converter = DeepFashion2ToYOLOConverter()

    processed_count = 0
    valid_instances_count = 0
    invalid_instances_count = 0
    empty_label_files_count = 0
    invalid_records: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {name: 0 for name in YOLO_CLASS_NAMES}
    class_id_counts: Dict[int, int] = {i: 0 for i in range(13)}
    multi_garment_counts: Dict[str, int] = {
        "0_instances": 0,
        "1_instance": 0,
        "2_instances": 0,
        "3_or_more_instances": 0,
    }
    missing_images: List[str] = []
    missing_annotations: List[str] = []

    for image_id in chunk_ids:
        anno_path = annos_dir / f"{image_id}.json"
        image_path = images_dir / f"{image_id}.jpg"
        label_path = labels_dir / f"{image_id}.txt"

        if not anno_path.exists():
            missing_annotations.append(image_id)
            continue
        if not image_path.exists():
            missing_images.append(image_id)
            continue

        res = converter.convert_single_annotation(
            anno_path=anno_path,
            image_path=image_path,
            output_label_path=label_path,
        )

        processed_count += 1
        n_valid = res.valid_count
        valid_instances_count += n_valid

        if n_valid == 0:
            multi_garment_counts["0_instances"] += 1
            empty_label_files_count += 1
        elif n_valid == 1:
            multi_garment_counts["1_instance"] += 1
        elif n_valid == 2:
            multi_garment_counts["2_instances"] += 1
        else:
            multi_garment_counts["3_or_more_instances"] += 1

        for inst in res.valid_instances:
            category_counts[inst.category_name] += 1
            class_id_counts[inst.class_id] += 1

        for inv in res.invalid_instances:
            invalid_instances_count += 1
            invalid_records.append({
                "image_id": image_id,
                "item_key": inv.item_key,
                "reason": inv.reason,
                "raw_bbox": inv.raw_bbox,
                "category_id": inv.category_id,
                "details": inv.details,
            })

    return {
        "processed_count": processed_count,
        "valid_instances_count": valid_instances_count,
        "invalid_instances_count": invalid_instances_count,
        "empty_label_files_count": empty_label_files_count,
        "invalid_records": invalid_records,
        "category_counts": category_counts,
        "class_id_counts": class_id_counts,
        "multi_garment_counts": multi_garment_counts,
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
    }


def run_split_conversion(
    split_name: str,
    annos_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    expected_count: int,
    n_workers: int = 6,
    chunk_size: int = 400,
) -> Dict[str, Any]:
    """
    Executes full parallel conversion for a single partition (train or val).
    """
    print(f"\n{'=' * 65}")
    print(f"STARTING {split_name.upper()} CONVERSION ({expected_count:,} images)")
    print(f"{'=' * 65}")
    print(f"  Source annotations: {annos_dir}")
    print(f"  Source images:      {images_dir}")
    print(f"  Target labels:      {labels_dir}")
    print(f"  Workers:            {n_workers} processes (chunk size: {chunk_size})")

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate all annotation IDs from source directory
    t_scan_start = time.time()
    source_files = sorted(os.listdir(annos_dir))
    image_ids = [f[:-5] for f in source_files if f.endswith(".json")]
    t_scan = time.time() - t_scan_start
    print(f"  Scanned {len(image_ids):,} JSON files in {t_scan:.2f}s")
    assert len(image_ids) == expected_count, (
        f"File count mismatch in {annos_dir}: expected {expected_count}, found {len(image_ids)}"
    )

    # Chunk image IDs
    chunks = [image_ids[i:i + chunk_size] for i in range(0, len(image_ids), chunk_size)]
    print(f"  Partitioned into {len(chunks):,} batch chunks")

    # Tracking metrics
    t0 = time.time()
    total_processed = 0
    total_valid = 0
    total_invalid = 0
    total_empty = 0
    all_invalid_records = []
    agg_category_counts = {name: 0 for name in YOLO_CLASS_NAMES}
    agg_class_id_counts = {i: 0 for i in range(13)}
    agg_multi_garment = {
        "0_instances": 0,
        "1_instance": 0,
        "2_instances": 0,
        "3_or_more_instances": 0,
    }
    missing_images_all = []
    missing_annos_all = []

    # Process in parallel
    print(f"  Dispatching chunks across {n_workers} worker processes...")
    last_reported = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _convert_chunk_worker,
                chunk,
                str(annos_dir),
                str(images_dir),
                str(labels_dir),
            )
            for chunk in chunks
        ]

        for idx, f in enumerate(futures):
            res = f.result()
            total_processed += res["processed_count"]
            total_valid += res["valid_instances_count"]
            total_invalid += res["invalid_instances_count"]
            total_empty += res["empty_label_files_count"]
            all_invalid_records.extend(res["invalid_records"])
            missing_images_all.extend(res["missing_images"])
            missing_annos_all.extend(res["missing_annotations"])

            for cat_name, cnt in res["category_counts"].items():
                agg_category_counts[cat_name] += cnt
            for cid, cnt in res["class_id_counts"].items():
                agg_class_id_counts[cid] += cnt
            for mg_key, cnt in res["multi_garment_counts"].items():
                agg_multi_garment[mg_key] += cnt

            # Progress reporting every ~10k images or at completion
            if total_processed - last_reported >= 10000 or (idx + 1) == len(chunks):
                elapsed = time.time() - t0
                pct = (total_processed / len(image_ids)) * 100.0
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta_s = (len(image_ids) - total_processed) / rate if rate > 0 else 0
                print(
                    f"  [{split_name.upper()}] {total_processed:>7,}/{len(image_ids):,} ({pct:5.1f}%) | "
                    f"{rate:>6.1f} img/s | Valid: {total_valid:>7,} | Invalid: {total_invalid:>3} | "
                    f"ETA: {eta_s:4.0f}s",
                    flush=True
                )
                last_reported = total_processed

    elapsed_total = time.time() - t0
    overall_rate = total_processed / elapsed_total if elapsed_total > 0 else 0
    print(f"  {split_name.upper()} conversion finished in {elapsed_total:.2f}s ({overall_rate:.1f} img/s)")

    # Group invalid instances by reason
    invalid_by_reason: Dict[str, int] = {}
    for inv in all_invalid_records:
        r = inv["reason"]
        invalid_by_reason[r] = invalid_by_reason.get(r, 0) + 1

    return {
        "split": split_name,
        "expected_image_count": expected_count,
        "processed_image_count": total_processed,
        "total_instances_inspected": total_valid + total_invalid,
        "valid_instance_count": total_valid,
        "invalid_instance_count": total_invalid,
        "invalid_instances_by_reason": invalid_by_reason,
        "invalid_records": all_invalid_records,
        "empty_label_files_count": total_empty,
        "category_counts_by_name": agg_category_counts,
        "category_counts_by_class_id": agg_class_id_counts,
        "multi_garment_image_counts": agg_multi_garment,
        "missing_images_count": len(missing_images_all),
        "missing_annotations_count": len(missing_annos_all),
        "missing_images": missing_images_all,
        "missing_annotations": missing_annos_all,
        "elapsed_seconds": round(elapsed_total, 2),
        "throughput_images_per_second": round(overall_rate, 2),
    }


def verify_split_integrity(
    split_name: str,
    labels_dir: Path,
    expected_image_count: int,
    expected_valid_instances: int,
    sample_verify_size: int = 5000,
) -> Dict[str, Any]:
    """
    Performs thorough post-conversion validation across all generated label files.
    """
    print(f"\n--- Verifying Integrity for {split_name.upper()} ---")
    t0 = time.time()

    label_files = os.listdir(labels_dir)
    actual_file_count = len(label_files)
    print(f"  Total label files in {labels_dir}: {actual_file_count:,}")

    count_match = (actual_file_count == expected_image_count)
    assert count_match, (
        f"File count mismatch in {labels_dir}: expected {expected_image_count}, got {actual_file_count}"
    )

    # Detailed line-by-line verification
    total_lines = 0
    invalid_lines = 0
    all_class_ids: Set[int] = set()
    coords_in_bounds = True
    dims_positive = True
    finite_check = True

    # Check all files or sample
    print(f"  Validating coordinate formats, finiteness, and class IDs across all {actual_file_count:,} files...")
    empty_files = 0
    single_item_files = 0
    multi_item_files = 0

    for fname in label_files:
        fpath = labels_dir / fname
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        cnt = len(lines)
        total_lines += cnt
        if cnt == 0:
            empty_files += 1
        elif cnt == 1:
            single_item_files += 1
        else:
            multi_item_files += 1

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                invalid_lines += 1
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            all_class_ids.add(cls_id)
            if not (0 <= cls_id <= 12):
                invalid_lines += 1
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                coords_in_bounds = False
            if not (bw > 0.0 and bh > 0.0 and bw <= 1.0 and bh <= 1.0):
                dims_positive = False
            if any(math.isnan(v) or math.isinf(v) for v in [xc, yc, bw, bh]):
                finite_check = False

    t_verify = time.time() - t0
    print(f"  Verified {total_lines:,} label rows in {t_verify:.2f}s")
    print(f"  Empty label files: {empty_files}")
    print(f"  Single-garment files: {single_item_files:,}")
    print(f"  Multi-garment files:  {multi_item_files:,}")

    assert total_lines == expected_valid_instances, (
        f"Instance count mismatch for {split_name}: expected {expected_valid_instances:,}, got {total_lines:,}"
    )
    assert invalid_lines == 0, f"Found {invalid_lines} malformed label lines"
    assert coords_in_bounds, "Some coordinates fell outside [0, 1]"
    assert dims_positive, "Some bounding box dimensions were not strictly positive"
    assert finite_check, "Some coordinates were NaN or Inf"
    assert all(0 <= c <= 12 for c in all_class_ids), f"Unexpected class IDs: {all_class_ids}"

    print(f"  [OK] All {total_lines:,} rows passed: valid class IDs (0..12), strictly positive dims, coords in [0,1]")

    return {
        "split": split_name,
        "label_files_count": actual_file_count,
        "expected_image_count": expected_image_count,
        "file_count_matches": count_match,
        "total_valid_label_rows": total_lines,
        "expected_valid_instances": expected_valid_instances,
        "instance_count_matches": (total_lines == expected_valid_instances),
        "empty_files_count": empty_files,
        "single_garment_files_count": single_item_files,
        "multi_garment_files_count": multi_item_files,
        "distinct_class_ids": sorted(list(all_class_ids)),
        "all_class_ids_in_range_0_12": all(0 <= c <= 12 for c in all_class_ids),
        "all_coordinates_normalized_and_finite": (coords_in_bounds and finite_check),
        "all_widths_heights_positive": dims_positive,
        "verification_status": "PASS",
        "verification_time_seconds": round(t_verify, 2),
    }


def compute_directory_size(path: Path) -> Tuple[int, float]:
    """Computes total file count and size in megabytes."""
    total_bytes = 0
    count = 0
    for root, _, files in os.walk(path):
        for f in files:
            count += 1
            total_bytes += os.path.getsize(os.path.join(root, f))
    return count, round(total_bytes / (1024 * 1024), 2)


def main():
    t_global_start = time.time()
    vm_start = psutil.virtual_memory()

    output_base = PROCESSED_DATA_DIR / "deepfashion2_yolo"
    train_labels_dir = output_base / "labels" / "train"
    val_labels_dir = output_base / "labels" / "val"

    print("=" * 65)
    print("STAGE 6.4C: FULL DEEPFASHION2 ANNOTATION CONVERSION")
    print("=" * 65)
    print(f"Output Root: {output_base}")
    print(f"Train labels destination: {train_labels_dir}")
    print(f"Val labels destination:   {val_labels_dir}")
    print(f"System RAM: {vm_start.total / (1024**3):.2f} GB total | {vm_start.available / (1024**3):.2f} GB available")

    # ─────────────────────────────────────────────────────────────────────────
    # Task 3: Full Train Conversion (191,961 images)
    # ─────────────────────────────────────────────────────────────────────────
    train_summary = run_split_conversion(
        split_name="train",
        annos_dir=TRAIN_ANNOTATION_DIR,
        images_dir=TRAIN_IMAGE_DIR,
        labels_dir=train_labels_dir,
        expected_count=191961,
        n_workers=6,
        chunk_size=400,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Task 4: Full Validation Conversion (32,153 images)
    # ─────────────────────────────────────────────────────────────────────────
    val_summary = run_split_conversion(
        split_name="val",
        annos_dir=VALIDATION_ANNOTATION_DIR,
        images_dir=VALIDATION_IMAGE_DIR,
        labels_dir=val_labels_dir,
        expected_count=32153,
        n_workers=6,
        chunk_size=400,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Task 5: Integrity Checks
    # ─────────────────────────────────────────────────────────────────────────
    train_integrity = verify_split_integrity(
        split_name="train",
        labels_dir=train_labels_dir,
        expected_image_count=191961,
        expected_valid_instances=train_summary["valid_instance_count"],
    )

    val_integrity = verify_split_integrity(
        split_name="val",
        labels_dir=val_labels_dir,
        expected_image_count=32153,
        expected_valid_instances=val_summary["valid_instance_count"],
    )

    # Reconciliation with Phase 1 Train Ground Truth:
    # Phase 1 ground truth total garment instances: 312,186
    # Inspected instances matches Phase 1 exactly: 312,186
    # Rejected instances: 3
    #   1. 101338 item2 [0, 0, 0, 0] (zero area, identified in Phase 1 audit)
    #   2. 028225 item2 [1, 198, 422, 626] on 468x624 image (y2=626 > 624)
    #   3. 115063 item1 [15, 195, 466, 829] on 468x702 image (y2=829 > 702, resulting in 1 empty file)
    # Valid converted instances: 312,183 (312,186 - 3)
    phase1_train_total = 312186
    expected_rejected = 3
    expected_valid = phase1_train_total - expected_rejected
    train_reconciled = (
        train_summary["total_instances_inspected"] == phase1_train_total
        and train_summary["invalid_instance_count"] == expected_rejected
        and train_summary["valid_instance_count"] == expected_valid
    )
    print("\n" + "=" * 65)
    print("GROUND TRUTH RECONCILIATION (TRAINING)")
    print("=" * 65)
    print(f"  Phase 1 Audited Training Instances:   {phase1_train_total:,}")
    print(f"  Stage 6.4C Inspected Instances:       {train_summary['total_instances_inspected']:,}")
    print(f"  Stage 6.4C Rejected Instances:        {train_summary['invalid_instance_count']:,} (1 zero-area + 2 out-of-bounds)")
    for rec in train_summary["invalid_records"]:
        print(f"    - Image {rec['image_id']} {rec['item_key']}: {rec['reason']} ({rec['raw_bbox']})")
    print(f"  Stage 6.4C Valid Converted Instances: {train_summary['valid_instance_count']:,}")
    print(f"  Reconciliation Match:                 {'EXACT MATCH [PASS]' if train_reconciled else 'MISMATCH [FAIL]'}")
    assert train_reconciled, "Training ground truth reconciliation failed!"

    # Directory sizes
    train_file_cnt, train_size_mb = compute_directory_size(train_labels_dir)
    val_file_cnt, val_size_mb = compute_directory_size(val_labels_dir)
    total_size_mb = train_size_mb + val_size_mb

    t_global_total = time.time() - t_global_start
    vm_end = psutil.virtual_memory()

    # ─────────────────────────────────────────────────────────────────────────
    # Task 7: Comprehensive JSON Report
    # ─────────────────────────────────────────────────────────────────────────
    full_report = {
        "stage": "6.4C",
        "timestamp": "2026-11-06T22:35:00+05:30",
        "description": "Full DeepFashion2 to YOLO Annotation Conversion Report",
        "output_directory": str(output_base),
        "split_paths": {
            "train_labels": str(train_labels_dir),
            "val_labels": str(val_labels_dir),
        },
        "dataset_inventory": {
            "train_images_expected": 191961,
            "train_images_converted": train_summary["processed_image_count"],
            "val_images_expected": 32153,
            "val_images_converted": val_summary["processed_image_count"],
            "test_images_withheld": 29765,
            "total_labeled_images_converted": train_summary["processed_image_count"] + val_summary["processed_image_count"],
        },
        "ground_truth_reconciliation": {
            "phase1_audited_train_instances": phase1_train_total,
            "train_total_instances_inspected": train_summary["total_instances_inspected"],
            "train_valid_instances": train_summary["valid_instance_count"],
            "train_rejected_instances": train_summary["invalid_instance_count"],
            "reconciliation_status": "EXACT_MATCH" if train_reconciled else "MISMATCH",
            "val_total_instances_inspected": val_summary["total_instances_inspected"],
            "val_valid_instances": val_summary["valid_instance_count"],
            "val_rejected_instances": val_summary["invalid_instance_count"],
        },
        "train_conversion_summary": {
            "image_count": train_summary["processed_image_count"],
            "valid_instance_count": train_summary["valid_instance_count"],
            "invalid_instance_count": train_summary["invalid_instance_count"],
            "invalid_instances_by_reason": train_summary["invalid_instances_by_reason"],
            "invalid_records": train_summary["invalid_records"],
            "empty_label_files_count": train_summary["empty_label_files_count"],
            "category_counts_by_name": train_summary["category_counts_by_name"],
            "category_counts_by_class_id": {str(k): v for k, v in train_summary["category_counts_by_class_id"].items()},
            "multi_garment_image_counts": train_summary["multi_garment_image_counts"],
            "elapsed_seconds": train_summary["elapsed_seconds"],
            "throughput_images_per_second": train_summary["throughput_images_per_second"],
        },
        "val_conversion_summary": {
            "image_count": val_summary["processed_image_count"],
            "valid_instance_count": val_summary["valid_instance_count"],
            "invalid_instance_count": val_summary["invalid_instance_count"],
            "invalid_instances_by_reason": val_summary["invalid_instances_by_reason"],
            "invalid_records": val_summary["invalid_records"],
            "empty_label_files_count": val_summary["empty_label_files_count"],
            "category_counts_by_name": val_summary["category_counts_by_name"],
            "category_counts_by_class_id": {str(k): v for k, v in val_summary["category_counts_by_class_id"].items()},
            "multi_garment_image_counts": val_summary["multi_garment_image_counts"],
            "elapsed_seconds": val_summary["elapsed_seconds"],
            "throughput_images_per_second": val_summary["throughput_images_per_second"],
        },
        "integrity_checks": {
            "train": train_integrity,
            "val": val_integrity,
        },
        "performance_and_storage": {
            "total_runtime_seconds": round(t_global_total, 2),
            "total_runtime_minutes": round(t_global_total / 60.0, 2),
            "train_labels_file_count": train_file_cnt,
            "train_labels_size_mb": train_size_mb,
            "val_labels_file_count": val_file_cnt,
            "val_labels_size_mb": val_size_mb,
            "total_output_size_mb": round(total_size_mb, 2),
            "ram_start_gb": round(vm_start.used / (1024**3), 2),
            "ram_end_gb": round(vm_end.used / (1024**3), 2),
            "ram_percent": vm_end.percent,
        },
        "overall_status": "SUCCESS",
    }

    report_path = OUTPUT_DIR / "step_6_4c_full_conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"REPORT SAVED: {report_path}")
    print(f"TOTAL CONVERTED LABELS: {train_file_cnt + val_file_cnt:,} files ({total_size_mb:.1f} MB)")
    print(f"TOTAL RUNTIME: {t_global_total:.1f}s ({t_global_total / 60.0:.2f} min)")
    print("OVERALL STATUS: SUCCESS [OK]")
    print("=" * 65)


if __name__ == "__main__":
    main()
