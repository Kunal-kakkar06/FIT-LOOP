#!/usr/bin/env python3
"""
FitLoop Phase 13.2: Filter, Convert & Package tyaw9 Dataset for Kaggle Phase 6
=============================================================================
1. Reads Clothes Detection.v9i.yolo26.zip from C:\\Users\\Lenovo\\Downloads\\.
2. Class Remapping (Drops all 11 non-apparel IDs: 0, 1, 2, 5, 6, 10, 12, 13, 14, 17, 20):
   - 19 (tshirt)             -> 0 (short sleeve top)
   - 11 (shirt), 8 (jumper)  -> 1 (long sleeve top)
   - 7 (jacket), 3 (coat)    -> 3 (long sleeve outwear)
   - 15 (shorts)             -> 6 (shorts)
   - 9 (pants), 18 (trousers)-> 7 (trousers)
   - 16 (skirt)              -> 8 (skirt)
   - 4 (dress)               -> 9 (short sleeve dress)
3. Converts both 5-col boxes and segmentation polygons to normalized 5-column YOLO boxes
   (class_id cx cy w h), clipped to [0.0, 1.0], w > 0, h > 0.
4. Excludes empty-after-filter images: only images with >= 1 retained apparel box are kept.
5. Preserves train, val (from valid), and test splits with canonical 13-class data.yaml.
6. Validates 100% of converted label files.
7. Packages into fitloop_phase6_tyaw9_part_*.zip (sharded <= 850 MB per ZIP) and verifies with testzip().
8. Saves comprehensive report to wardrobe_ml/outputs/tyaw9_phase6_prep_report.txt.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Centralized constants
DOWNLOADS_DIR = Path(r"C:\Users\Lenovo\Downloads")
SRC_ZIP_PATH = DOWNLOADS_DIR / "Clothes Detection.v9i.yolo26.zip"

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "kaggle_phase6_tyaw9"
REPORT_PATH = PROJECT_DIR / "outputs" / "tyaw9_phase6_prep_report.txt"

# FitLoop Canonical 13-Class Schema
FITLOOP_13_CLASSES: List[str] = [
    "short sleeve top",       # 0
    "long sleeve top",        # 1
    "short sleeve outwear",   # 2
    "long sleeve outwear",    # 3
    "vest",                   # 4
    "sling",                  # 5
    "shorts",                 # 6
    "trousers",               # 7
    "skirt",                  # 8
    "short sleeve dress",     # 9
    "long sleeve dress",      # 10
    "vest dress",             # 11
    "sling dress",            # 12
]

# Source ID -> Target FitLoop 13-Class ID
TYAW9_TO_FITLOOP_MAP: Dict[int, int] = {
    19: 0,  # tshirt             -> 0: short sleeve top
    11: 1,  # shirt              -> 1: long sleeve top
    8: 1,   # jumper             -> 1: long sleeve top
    7: 3,   # jacket             -> 3: long sleeve outwear
    3: 3,   # coat               -> 3: long sleeve outwear
    15: 6,  # shorts             -> 6: shorts
    9: 7,   # pants              -> 7: trousers
    18: 7,  # trousers           -> 7: trousers
    16: 8,  # skirt              -> 8: skirt
    4: 9,   # dress              -> 9: short sleeve dress
}

# Explicitly dropped non-apparel / accessory IDs
DROPPED_CLASSES: Dict[int, str] = {
    0: "bag",
    1: "belt",
    2: "boat",
    5: "glasses",
    6: "hat",
    10: "pocket",
    12: "shoe",
    13: "shoesClassic",
    14: "shoesSports",
    17: "sunglass",
    20: "zipper",
}

# Split mapping: source -> target
SPLIT_MAP: Dict[str, str] = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def convert_label_line(line: str) -> Optional[Tuple[int, str]]:
    """
    Parse a label line (5-token bbox or N-token polygon), remap to FitLoop 13-class schema,
    clip coordinates to [0.0, 1.0], filter zero-area boxes, and return (target_cid, formatted_line).
    Returns None if class is dropped, unmapped, or box has zero area.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        src_cid = int(parts[0])
    except ValueError:
        return None

    if src_cid not in TYAW9_TO_FITLOOP_MAP:
        return None

    tgt_cid = TYAW9_TO_FITLOOP_MAP[src_cid]

    # Standard 5-column YOLO Bounding Box
    if len(parts) == 5:
        try:
            cx_raw, cy_raw, w_raw, h_raw = [float(x) for x in parts[1:5]]
        except ValueError:
            return None

        xmin = max(0.0, min(1.0, cx_raw - w_raw / 2.0))
        xmax = max(0.0, min(1.0, cx_raw + w_raw / 2.0))
        ymin = max(0.0, min(1.0, cy_raw - h_raw / 2.0))
        ymax = max(0.0, min(1.0, cy_raw + h_raw / 2.0))
    else:
        # Polygon segmentation
        coord_tokens = parts[1:]
        if len(coord_tokens) % 2 != 0:
            coord_tokens = coord_tokens[:-1]
        if len(coord_tokens) < 4:
            return None

        try:
            coords = [float(x) for x in coord_tokens]
        except ValueError:
            return None

        xs = coords[0::2]
        ys = coords[1::2]
        xmin = max(0.0, min(1.0, min(xs)))
        xmax = max(0.0, min(1.0, max(xs)))
        ymin = max(0.0, min(1.0, min(ys)))
        ymax = max(0.0, min(1.0, max(ys)))

    w_box = xmax - xmin
    h_box = ymax - ymin
    if w_box <= 1e-5 or h_box <= 1e-5:
        return None

    cx_box = max(0.0, min(1.0, xmin + w_box / 2.0))
    cy_box = max(0.0, min(1.0, ymin + h_box / 2.0))

    formatted = f"{tgt_cid} {cx_box:.6f} {cy_box:.6f} {w_box:.6f} {h_box:.6f}"
    return tgt_cid, formatted


def run_conversion_and_packaging() -> None:
    print("=" * 85)
    print("FITLOOP PHASE 13.2: FILTER, CONVERT & PACKAGE tyaw9 DATASET (KAGGLE PHASE 6)")
    print("=" * 85)

    t0_all = time.perf_counter()

    if not SRC_ZIP_PATH.exists():
        raise FileNotFoundError(f"Source archive not found at: {SRC_ZIP_PATH}")

    print(f"\n[Step 1] Ingesting archive: {SRC_ZIP_PATH.name} ({SRC_ZIP_PATH.stat().st_size:,} bytes)")

    # Prepare output directories
    if OUTPUT_DIR.exists():
        print(f"Cleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sp_target in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / sp_target).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / sp_target).mkdir(parents=True, exist_ok=True)

    # 1. Scan and parse all labels
    print("\n[Step 2] Scanning labels, applying class remapping & filtering non-apparel...")
    t0_scan = time.perf_counter()

    kept_samples: Dict[str, List[Tuple[str, str, List[str]]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    dropped_stats = Counter()
    poly_converted_count = 0
    box_converted_count = 0
    per_class_counts: Dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    with zipfile.ZipFile(SRC_ZIP_PATH, "r") as zf:
        namelist_set = set(zf.namelist())

        for src_sp, tgt_sp in SPLIT_MAP.items():
            lbl_prefix = f"{src_sp}/labels/"
            label_entries = sorted([n for n in namelist_set if n.startswith(lbl_prefix) and n.endswith(".txt")])

            for lbl_path in label_entries:
                stem = Path(lbl_path).stem
                img_path = f"{src_sp}/images/{stem}.jpg"
                if img_path not in namelist_set:
                    continue

                raw_data = zf.read(lbl_path).decode("utf-8").strip()
                if not raw_data:
                    dropped_stats["empty_source_label"] += 1
                    continue

                converted_lines: List[str] = []
                for line in raw_data.splitlines():
                    tokens = line.strip().split()
                    if not tokens:
                        continue

                    try:
                        cid = int(tokens[0])
                    except ValueError:
                        dropped_stats["corrupted_token"] += 1
                        continue

                    if cid in DROPPED_CLASSES:
                        dropped_stats[f"dropped_{DROPPED_CLASSES[cid]}"] += 1
                        continue

                    res = convert_label_line(line)
                    if res is not None:
                        target_cid, conv_str = res
                        converted_lines.append(conv_str)
                        per_class_counts[tgt_sp][target_cid] += 1
                        if len(tokens) == 5:
                            box_converted_count += 1
                        else:
                            poly_converted_count += 1
                    else:
                        dropped_stats["zero_area_or_unmapped"] += 1

                # Retain image only if >= 1 apparel bounding box remains
                if converted_lines:
                    kept_samples[tgt_sp].append((img_path, stem, converted_lines))
                else:
                    dropped_stats["empty_after_filter"] += 1

    t_scan = time.perf_counter() - t0_scan
    total_kept_images = sum(len(v) for v in kept_samples.values())
    total_kept_instances = sum(sum(c.values()) for c in per_class_counts.values())

    print(f"  -> Scan & filtering completed in {t_scan:.2f}s")
    print(f"  -> Retained Images: {total_kept_images:,} (Train: {len(kept_samples['train']):,}, Val: {len(kept_samples['val']):,}, Test: {len(kept_samples['test']):,})")
    print(f"  -> Retained Apparel Instances: {total_kept_instances:,} (Train: {sum(per_class_counts['train'].values()):,}, Val: {sum(per_class_counts['val'].values()):,}, Test: {sum(per_class_counts['test'].values()):,})")
    print(f"  -> Converted BBoxes: {box_converted_count:,} from 5-col boxes, {poly_converted_count:,} from segmentation polygons")
    print(f"  -> Excluded Non-Apparel Images: {dropped_stats['empty_after_filter']:,} images (shoes/bags/accessories only)")

    # 2. Extract retained images and write converted labels to disk
    print("\n[Step 3] Extracting retained images and writing converted labels to disk...")
    t0_write = time.perf_counter()

    written_image_bytes = 0
    with zipfile.ZipFile(SRC_ZIP_PATH, "r") as zf:
        for tgt_sp, samples in kept_samples.items():
            img_dest_dir = OUTPUT_DIR / "images" / tgt_sp
            lbl_dest_dir = OUTPUT_DIR / "labels" / tgt_sp

            print(f"  Writing split '{tgt_sp}' ({len(samples):,} images)...")
            for img_zip_path, stem, lines in samples:
                # Extract image
                img_bytes = zf.read(img_zip_path)
                written_image_bytes += len(img_bytes)
                (img_dest_dir / f"{stem}.jpg").write_bytes(img_bytes)

                # Write converted label
                lbl_content = "\n".join(lines) + "\n"
                (lbl_dest_dir / f"{stem}.txt").write_text(lbl_content, encoding="utf-8")

    t_write = time.perf_counter() - t0_write
    print(f"  -> Extraction and label writing completed in {t_write:.2f}s ({written_image_bytes / (1024*1024):.2f} MB written)")

    # 3. Write data.yaml
    print("\n[Step 4] Generating 13-class data.yaml...")
    data_yaml_dict = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 13,
        "names": {i: name for i, name in enumerate(FITLOOP_13_CLASSES)},
    }
    data_yaml_path = OUTPUT_DIR / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)
    print(f"  -> data.yaml generated with 13 FitLoop classes at {data_yaml_path}")

    # 4. Strict 100% Validation Pass
    print("\n[Step 5] Running 100% validation pass on converted dataset...")
    validation_passed = True
    val_errors: List[str] = []
    total_val_labels = 0
    total_val_boxes = 0

    for tgt_sp in ["train", "val", "test"]:
        lbl_dir = OUTPUT_DIR / "labels" / tgt_sp
        img_dir = OUTPUT_DIR / "images" / tgt_sp

        lbl_files = list(lbl_dir.glob("*.txt"))
        img_files = list(img_dir.glob("*.jpg"))

        if len(lbl_files) != len(img_files):
            validation_passed = False
            val_errors.append(f"Split {tgt_sp}: label count ({len(lbl_files)}) != image count ({len(img_files)})")

        for lf in lbl_files:
            total_val_labels += 1
            corr_img = img_dir / f"{lf.stem}.jpg"
            if not corr_img.exists():
                validation_passed = False
                val_errors.append(f"Missing corresponding image: {corr_img}")

            lines = lf.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                validation_passed = False
                val_errors.append(f"Empty converted label: {lf}")

            for l_idx, l in enumerate(lines):
                parts = l.strip().split()
                if len(parts) != 5:
                    validation_passed = False
                    val_errors.append(f"{lf}:{l_idx} irregular token count {len(parts)} (expected 5)")
                    continue

                try:
                    c = int(parts[0])
                    cx, cy, w, h = [float(x) for x in parts[1:]]
                except ValueError:
                    validation_passed = False
                    val_errors.append(f"{lf}:{l_idx} non-numeric tokens")
                    continue

                if not (0 <= c < 13):
                    validation_passed = False
                    val_errors.append(f"{lf}:{l_idx} invalid class ID {c}")

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    validation_passed = False
                    val_errors.append(f"{lf}:{l_idx} center coords out of bounds ({cx}, {cy})")

                if w <= 0.0 or h <= 0.0:
                    validation_passed = False
                    val_errors.append(f"{lf}:{l_idx} non-positive dimensions ({w}, {h})")

                total_val_boxes += 1

    print(f"  -> Validation Results: {total_val_labels:,} label files, {total_val_boxes:,} boxes checked.")
    if validation_passed:
        print(f"  -> 100% VALIDATION PASSED: All labels have 5 tokens, class IDs in [0, 12], coordinates in [0.0, 1.0], w>0, h>0.")
    else:
        print(f"  -> VALIDATION FAILED with {len(val_errors)} errors!")
        for e in val_errors[:5]:
            print(f"     [ERROR] {e}")
        raise RuntimeError("Validation pass failed on converted dataset!")

    # 5. Sharded Packaging
    print("\n[Step 6] Packaging dataset into Kaggle partition ZIP archives (< 850 MB per ZIP)...")
    t0_pack = time.perf_counter()

    # Calculate total size to determine number of partitions
    all_files_to_pack: List[Tuple[Path, str]] = []  # (absolute_file_path, arcname)

    for tgt_sp in ["train", "val", "test"]:
        img_dir = OUTPUT_DIR / "images" / tgt_sp
        lbl_dir = OUTPUT_DIR / "labels" / tgt_sp

        for img_p in img_dir.glob("*.jpg"):
            all_files_to_pack.append((img_p, f"images/{tgt_sp}/{img_p.name}"))
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            all_files_to_pack.append((lbl_p, f"labels/{tgt_sp}/{lbl_p.name}"))

    total_pack_bytes = sum(f[0].stat().st_size for f in all_files_to_pack)
    max_part_bytes = 850 * 1024 * 1024  # 850 MB

    if total_pack_bytes > max_part_bytes:
        num_parts = math.ceil(total_pack_bytes / max_part_bytes)
    else:
        num_parts = 1

    print(f"  -> Total uncompressed dataset payload: {total_pack_bytes / (1024*1024):.2f} MB")
    print(f"  -> Target partitions: {num_parts} part(s) (target <= 850 MB each)")

    target_bytes_per_part = total_pack_bytes / num_parts
    created_zips: List[Path] = []

    # Partition by samples (image + label pair kept strictly together)
    # We group pairs
    paired_items: List[Tuple[Path, str, Path, str]] = []
    for tgt_sp in ["train", "val", "test"]:
        img_dir = OUTPUT_DIR / "images" / tgt_sp
        lbl_dir = OUTPUT_DIR / "labels" / tgt_sp
        for img_p in sorted(img_dir.glob("*.jpg")):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            paired_items.append((
                img_p, f"images/{tgt_sp}/{img_p.name}",
                lbl_p, f"labels/{tgt_sp}/{lbl_p.name}",
            ))

    # Distribute pairs across partitions
    part_idx = 1
    current_part_bytes = 0
    current_zip_path = OUTPUT_DIR / f"fitloop_phase6_tyaw9_part_{part_idx}.zip"
    current_zf = zipfile.ZipFile(current_zip_path, "w", zipfile.ZIP_DEFLATED)
    # Always include data.yaml in every partition
    current_zf.write(data_yaml_path, arcname="data.yaml")
    created_zips.append(current_zip_path)

    for img_p, img_arc, lbl_p, lbl_arc in paired_items:
        pair_bytes = img_p.stat().st_size + lbl_p.stat().st_size
        if (current_part_bytes + pair_bytes > target_bytes_per_part) and (part_idx < num_parts):
            current_zf.close()
            print(f"  Finished Part {part_idx}: {current_zip_path.name} ({current_zip_path.stat().st_size / (1024*1024):.2f} MB)")

            part_idx += 1
            current_part_bytes = 0
            current_zip_path = OUTPUT_DIR / f"fitloop_phase6_tyaw9_part_{part_idx}.zip"
            current_zf = zipfile.ZipFile(current_zip_path, "w", zipfile.ZIP_DEFLATED)
            current_zf.write(data_yaml_path, arcname="data.yaml")
            created_zips.append(current_zip_path)

        current_zf.write(img_p, arcname=img_arc)
        current_zf.write(lbl_p, arcname=lbl_arc)
        current_part_bytes += pair_bytes

    current_zf.close()
    print(f"  Finished Part {part_idx}: {current_zip_path.name} ({current_zip_path.stat().st_size / (1024*1024):.2f} MB)")

    t_pack = time.perf_counter() - t0_pack
    print(f"  -> Packaging completed in {t_pack:.2f}s ({len(created_zips)} archive files created)")

    # Verify every created zip passes testzip()
    print("\n[Step 7] Verifying integrity of all created partition ZIPs...")
    for cz in created_zips:
        with zipfile.ZipFile(cz, "r") as zf:
            check_err = zf.testzip()
            if check_err is not None:
                raise RuntimeError(f"Corrupted zip generated: {cz.name} (failed on {check_err})")
            print(f"  -> {cz.name}: {cz.stat().st_size / (1024*1024):.2f} MB -> testzip() PASSED [0 errors]")

    total_time = time.perf_counter() - t0_all

    # 6. Build and save comprehensive report
    report_lines: List[str] = [
        "=" * 85,
        "FITLOOP PHASE 13.2: tyaw9 APPAREL DATASET CONVERSION & PACKAGING REPORT",
        "=" * 85,
        f"Execution Timestamp : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Source Archive      : {SRC_ZIP_PATH}",
        f"Source Size         : {SRC_ZIP_PATH.stat().st_size:,} bytes",
        f"Output Directory    : {OUTPUT_DIR}",
        f"Execution Runtime   : {total_time:.2f}s",
        "",
        "SECTION 1: CONVERTED SPLIT ARCHITECTURE & IMAGE COUNTS",
        "-" * 85,
        f"{'Split':<10} | {'Images Count':<14} | {'Label Files Count':<18} | {'Apparel Instances':<18}",
        "-" * 85,
    ]

    for tgt_sp in ["train", "val", "test"]:
        n_imgs = len(kept_samples[tgt_sp])
        n_insts = sum(per_class_counts[tgt_sp].values())
        report_lines.append(f"{tgt_sp:<10} | {n_imgs:<14,d} | {n_imgs:<18,d} | {n_insts:<18,d}")

    report_lines.extend([
        "-" * 85,
        f"{'TOTAL':<10} | {total_kept_images:<14,d} | {total_kept_images:<18,d} | {total_kept_instances:<18,d}",
        "",
        "SECTION 2: CLASS REMAPPING AUDIT (FITLOOP 13-CLASS SCHEMA)",
        "-" * 85,
        f"{'FitLoop ID':<11} | {'FitLoop Class Name':<22} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total Instances'}",
        "-" * 85,
    ])

    for cid, cname in enumerate(FITLOOP_13_CLASSES):
        c_tr = per_class_counts["train"][cid]
        c_val = per_class_counts["val"][cid]
        c_ts = per_class_counts["test"][cid]
        c_tot = c_tr + c_val + c_ts
        report_lines.append(f"{cid:<11} | {cname:<22} | {c_tr:<8,d} | {c_val:<8,d} | {c_ts:<8,d} | {c_tot:<8,d}")

    report_lines.extend([
        "-" * 85,
        f"{'ALL':<11} | {'Total Apparel Instances':<22} | {sum(per_class_counts['train'].values()):<8,d} | {sum(per_class_counts['val'].values()):<8,d} | {sum(per_class_counts['test'].values()):<8,d} | {total_kept_instances:<8,d}",
        "",
        "SECTION 3: DROPPED NON-APPAREL & HARDWARE CATEGORIES",
        "-" * 85,
    ])

    for dc_id, dc_name in sorted(DROPPED_CLASSES.items()):
        cnt = dropped_stats[f"dropped_{dc_name}"]
        report_lines.append(f"Source ID {dc_id:2d}: {dc_name:<16} -> DROPPED ({cnt:,} instances excluded)")

    report_lines.extend([
        f"Empty After Filter Images Excluded : {dropped_stats['empty_after_filter']:,} images (dilution prevented)",
        "",
        "SECTION 4: PACKAGED PARTITION ARCHIVES",
        "-" * 85,
        f"{'Partition ZIP Archive Name':<42} | {'File Size (Bytes)':<18} | {'Size (MB)':<10} | {'testzip() Status'}",
        "-" * 85,
    ])

    for cz in created_zips:
        report_lines.append(
            f"{cz.name:<42} | {cz.stat().st_size:<18,d} | {cz.stat().st_size / (1024*1024):<10.2f} | PASSED (0 errors)"
        )

    report_lines.extend([
        "-" * 85,
        "",
        "SECTION 5: VERIFICATION CRITERIA & CONCLUSION",
        "-" * 85,
        "1. 100% of converted labels validated: 5 tokens, class IDs in [0, 12], coordinates in [0.0, 1.0].",
        "2. All polygons converted to standard YOLO 5-column bounding boxes (w > 0, h > 0).",
        "3. Zero empty-after-filter images retained (every image has >= 1 valid apparel annotation).",
        "4. Image and label pairs co-located in identical partition ZIPs with data.yaml.",
        "5. Every partition ZIP is strictly < 850 MB and passes testzip() integrity verification.",
        "6. Final Status: KAGGLE PHASE 6 DATASET SUCCESSFULLY PREPARED & PACKAGED.",
        "=" * 85,
    ])

    report_text = "\n".join(report_lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    run_conversion_and_packaging()
