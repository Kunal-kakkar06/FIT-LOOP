"""
FitLoop Phase 8.2: Roboflow to FitLoop Phase 3 Dataset Converter & Packager
============================================================================
1. Ingests C:\\Users\\Lenovo\\Downloads\\Clothes detection.v6i.yolo26.zip.
2. Converts polygon coordinates and bounding boxes into normalized 5-column YOLO boxes
   (class_id cx cy w h), strictly clipped to [0.0, 1.0], skipping zero-area boxes.
3. Remaps the 8 Roboflow classes to FitLoop's 13-class DeepFashion2 schema:
     0 (Tshirt)  -> 0 (short sleeve top)
     1 (dress)   -> 9 (short sleeve dress)
     2 (jacket)  -> 3 (long sleeve outwear)
     3 (pants)   -> 7 (trousers)
     4 (shirt)   -> 1 (long sleeve top)
     5 (short)   -> 6 (shorts)
     6 (skirt)   -> 8 (skirt)
     7 (sweater) -> 1 (long sleeve top)
4. Organizes images and labels into standard YOLO structure:
     images/train, labels/train
     images/val,   labels/val
     images/test,  labels/test
     data.yaml (13 classes)
5. Validates 100% of converted label files (5 tokens, class in [0, 12], coordinates in [0, 1]).
6. Packages images/ and labels/ into fitloop_phase3_roboflow.zip and verifies with testzip().
7. Generates conversion report at wardrobe_ml/outputs/roboflow_phase3_prep_report.txt.
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
SRC_ZIP_PATH = Path(r"C:\Users\Lenovo\Downloads\Clothes detection.v6i.yolo26.zip")
OUTPUT_DIR = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\kaggle_phase3_roboflow")
PACKAGE_ZIP_PATH = OUTPUT_DIR / "fitloop_phase3_roboflow.zip"
REPORT_PATH = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\roboflow_phase3_prep_report.txt")

# Class Remapping Table: Roboflow 8 classes -> FitLoop 13 classes
ROBOFLOW_TO_FITLOOP_MAP: Dict[int, int] = {
    0: 0,  # Tshirt   -> short sleeve top
    1: 9,  # dress    -> short sleeve dress
    2: 3,  # jacket   -> long sleeve outwear
    3: 7,  # pants    -> trousers
    4: 1,  # shirt    -> long sleeve top
    5: 6,  # short    -> shorts
    6: 8,  # skirt    -> skirt
    7: 1,  # sweater  -> long sleeve top
}

ROBOFLOW_NAMES: List[str] = [
    "Tshirt",
    "dress",
    "jacket",
    "pants",
    "shirt",
    "short",
    "skirt",
    "sweater",
]

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

# Split naming alignment: Roboflow split name -> YOLO standard directory name
SPLIT_ALIGNMENT: Dict[str, str] = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def parse_and_convert_label_line(
    raw_line: str,
    line_num: int = 1,
    file_name: str = "",
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse a label line (standard 5-token bbox or N-token polygon), remap the class,
    clip coordinates to [0.0, 1.0], filter zero-area boxes, and return the formatted
    YOLO line string `class_id cx cy w h`.

    Returns:
        Tuple of (converted_line, target_class_id, rejection_reason)
    """
    line = raw_line.strip()
    if not line:
        return None, None, "empty_line"

    tokens = line.split()
    if len(tokens) < 5:
        return None, None, f"insufficient_tokens_{len(tokens)}"

    try:
        raw_cls = int(tokens[0])
    except ValueError:
        return None, None, f"invalid_class_token_{tokens[0]}"

    if raw_cls not in ROBOFLOW_TO_FITLOOP_MAP:
        return None, None, f"unknown_class_id_{raw_cls}"

    target_cls = ROBOFLOW_TO_FITLOOP_MAP[raw_cls]

    # Case 1: Standard 5-column YOLO Bounding Box
    if len(tokens) == 5:
        try:
            cx_raw = float(tokens[1])
            cy_raw = float(tokens[2])
            w_raw = float(tokens[3])
            h_raw = float(tokens[4])
        except ValueError:
            return None, None, "unparseable_bbox_floats"

        # Clip values to [0.0, 1.0]
        cx = max(0.0, min(1.0, cx_raw))
        cy = max(0.0, min(1.0, cy_raw))
        w = max(0.0, min(1.0, w_raw))
        h = max(0.0, min(1.0, h_raw))

        # Check for zero or negative area
        if w <= 1e-6 or h <= 1e-6:
            return None, None, f"zero_area_bbox (w={w_raw}, h={h_raw})"

        return f"{target_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}", target_cls, None

    # Case 2: Polygon Segmentation (11 tokens for 5-vertex closed box, or arbitrary polygon)
    coord_tokens = tokens[1:]
    if len(coord_tokens) % 2 != 0:
        coord_tokens = coord_tokens[: len(coord_tokens) - 1]

    if len(coord_tokens) < 4:
        return None, None, "insufficient_polygon_coordinates"

    try:
        coords = [float(x) for x in coord_tokens]
    except ValueError:
        return None, None, "unparseable_polygon_floats"

    xs = coords[0::2]
    ys = coords[1::2]

    # Compute bounding extents
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # Strictly clip bounding extents to [0.0, 1.0]
    c_xmin = max(0.0, min(1.0, min_x))
    c_xmax = max(0.0, min(1.0, max_x))
    c_ymin = max(0.0, min(1.0, min_y))
    c_ymax = max(0.0, min(1.0, max_y))

    w = c_xmax - c_xmin
    h = c_ymax - c_ymin

    if w <= 1e-6 or h <= 1e-6:
        return None, None, f"zero_area_polygon (w={w:.6f}, h={h:.6f})"

    cx = c_xmin + (w / 2.0)
    cy = c_ymin + (h / 2.0)

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return f"{target_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}", target_cls, None


def validate_label_file(file_path: Path) -> List[str]:
    """
    Strictly validate a converted YOLO label file.
    Returns list of error messages (empty list if 100% valid).
    """
    errors: List[str] = []
    if not file_path.exists():
        return [f"File does not exist: {file_path}"]

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        # Empty label file is valid for negative background images
        return []

    lines = content.splitlines()
    for idx, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{file_path.name}:{idx}: Expected exactly 5 tokens, got {len(parts)} -> '{line}'")
            continue

        # Class ID check
        try:
            cid = int(parts[0])
            if cid < 0 or cid > 12:
                errors.append(f"{file_path.name}:{idx}: Class ID {cid} outside valid range [0, 12]")
        except ValueError:
            errors.append(f"{file_path.name}:{idx}: Non-integer Class ID '{parts[0]}'")

        # Coordinates check
        try:
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            errors.append(f"{file_path.name}:{idx}: Unparseable float coordinates")
            continue

        # Finite check
        for val, name in [(cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")]:
            if not math.isfinite(val):
                errors.append(f"{file_path.name}:{idx}: Non-finite float value {name}={val}")

        # Range bounds
        if not (0.0 <= cx <= 1.0):
            errors.append(f"{file_path.name}:{idx}: cx={cx} out of bounds [0.0, 1.0]")
        if not (0.0 <= cy <= 1.0):
            errors.append(f"{file_path.name}:{idx}: cy={cy} out of bounds [0.0, 1.0]")
        if not (0.0 < w <= 1.0):
            errors.append(f"{file_path.name}:{idx}: w={w} out of bounds (0.0, 1.0]")
        if not (0.0 < h <= 1.0):
            errors.append(f"{file_path.name}:{idx}: h={h} out of bounds (0.0, 1.0]")

    return errors


def run_pipeline() -> int:
    t0 = time.time()
    print("=" * 78)
    print("FITLOOP PHASE 8.2: ROBOFLOW DATASET CONVERTER & PACKAGER")
    print("=" * 78)

    # 1. Verify Source Archive
    print(f"[STEP 1] Validating source archive: {SRC_ZIP_PATH}")
    if not SRC_ZIP_PATH.exists():
        print(f"[FATAL] Source archive not found: {SRC_ZIP_PATH}")
        return 1

    with zipfile.ZipFile(SRC_ZIP_PATH, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            print(f"[FATAL] Source ZIP corruption detected: {bad_file}")
            return 1
        namelist = set(zf.namelist())
        print(f"[PASS] Source archive verified cleanly ({len(namelist):,} entries).")

    # 2. Setup Clean Output Directory Structure
    print(f"\n[STEP 2] Setting up output directories under: {OUTPUT_DIR}")
    dataset_root = OUTPUT_DIR / "dataset"
    images_base = dataset_root / "images"
    labels_base = dataset_root / "labels"

    # Reset dataset directory if present
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    for target_split in ["train", "val", "test"]:
        (images_base / target_split).mkdir(parents=True, exist_ok=True)
        (labels_base / target_split).mkdir(parents=True, exist_ok=True)

    # 3. Extraction, Label Conversion & Class Remapping
    print("\n[STEP 3] Converting labels, remapping classes, and extracting images...")
    split_stats: Dict[str, Dict[str, Any]] = {
        "train": {"images": 0, "labels": 0, "instances": 0, "empty_labels": 0, "rejected_boxes": 0},
        "val": {"images": 0, "labels": 0, "instances": 0, "empty_labels": 0, "rejected_boxes": 0},
        "test": {"images": 0, "labels": 0, "instances": 0, "empty_labels": 0, "rejected_boxes": 0},
    }
    class_instance_counts: Dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    rejection_reasons: Counter[str] = Counter()

    with zipfile.ZipFile(SRC_ZIP_PATH, "r") as zf:
        for src_split, target_split in SPLIT_ALIGNMENT.items():
            print(f"  Processing split: '{src_split}' -> '{target_split}'...")
            img_entries = [
                n for n in namelist
                if n.startswith(f"{src_split}/images/") and Path(n).suffix.lower() in (".jpg", ".jpeg", ".png")
            ]

            for img_entry in img_entries:
                stem = Path(img_entry).stem
                suffix = Path(img_entry).suffix
                lbl_entry = f"{src_split}/labels/{stem}.txt"

                # 3a. Extract image
                dest_img_path = images_base / target_split / f"{stem}{suffix}"
                with zf.open(img_entry, "r") as src_f, open(dest_img_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                split_stats[target_split]["images"] += 1

                # 3b. Read and convert label
                converted_lines: List[str] = []
                if lbl_entry in namelist:
                    raw_text = zf.read(lbl_entry).decode("utf-8", errors="replace").strip()
                    for idx, line in enumerate(raw_text.splitlines(), 1):
                        conv_line, tgt_cls, rej_reason = parse_and_convert_label_line(
                            line, line_num=idx, file_name=lbl_entry
                        )
                        if conv_line and tgt_cls is not None:
                            converted_lines.append(conv_line)
                            class_instance_counts[target_split][tgt_cls] += 1
                            split_stats[target_split]["instances"] += 1
                        else:
                            if rej_reason != "empty_line":
                                rejection_reasons[rej_reason or "unknown"] += 1
                                split_stats[target_split]["rejected_boxes"] += 1

                dest_lbl_path = labels_base / target_split / f"{stem}.txt"
                if converted_lines:
                    dest_lbl_path.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
                else:
                    dest_lbl_path.write_text("", encoding="utf-8")
                    split_stats[target_split]["empty_labels"] += 1

                split_stats[target_split]["labels"] += 1

    # 4. Write data.yaml for Kaggle / YOLO Phase 3 Training
    print("\n[STEP 4] Writing data.yaml specification...")
    data_yaml_dict = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 13,
        "names": {i: name for i, name in enumerate(FITLOOP_13_CLASSES)},
    }
    data_yaml_path = dataset_root / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)
    print(f"  data.yaml written to: {data_yaml_path}")

    # 5. Strict Label Validation Suite
    print("\n[STEP 5] Validating 100% of converted label files across all splits...")
    all_label_files: List[Path] = []
    for sp in ["train", "val", "test"]:
        all_label_files.extend(list((labels_base / sp).glob("*.txt")))

    print(f"  Validating {len(all_label_files):,} label files against strict schema...")
    total_validation_errors: List[str] = []

    for lbf in all_label_files:
        errs = validate_label_file(lbf)
        if errs:
            total_validation_errors.extend(errs)

    if total_validation_errors:
        print(f"[FATAL ERROR] {len(total_validation_errors)} validation errors encountered!")
        for e in total_validation_errors[:20]:
            print(f"  - {e}")
        return 1

    print(f"[PASS] All {len(all_label_files):,} label files passed 100% schema validation.")
    print("  - Exactly 5 tokens per line.")
    print("  - All class IDs in valid range [0, 12].")
    print("  - All cx, cy, w, h strictly finite floats in [0.0, 1.0] with w > 0, h > 0.")

    # 6. Package into fitloop_phase3_roboflow.zip
    print(f"\n[STEP 6] Packaging dataset into {PACKAGE_ZIP_PATH}...")
    PACKAGE_ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PACKAGE_ZIP_PATH.exists():
        PACKAGE_ZIP_PATH.unlink()

    files_to_pack: List[Tuple[Path, str]] = []

    # Add data.yaml at root
    files_to_pack.append((data_yaml_path, "data.yaml"))

    # Add images/ and labels/
    for root_dir in [images_base, labels_base]:
        for p in root_dir.rglob("*"):
            if p.is_file():
                rel_path = p.relative_to(dataset_root).as_posix()
                files_to_pack.append((p, rel_path))

    print(f"  Compressing {len(files_to_pack):,} files with ZIP_DEFLATED...")
    with zipfile.ZipFile(PACKAGE_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for disk_path, arc_path in files_to_pack:
            zf.write(disk_path, arc_path)

    zip_size_mb = PACKAGE_ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"  Archive created: {zip_size_mb:.2f} MB ({PACKAGE_ZIP_PATH.stat().st_size:,} bytes)")

    # 7. Test Output ZIP Integrity
    print(f"\n[STEP 7] Verifying packaged archive integrity (testzip)...")
    with zipfile.ZipFile(PACKAGE_ZIP_PATH, "r") as zf:
        bad = zf.testzip()
        if bad:
            print(f"[FATAL] Packaged ZIP corruption detected: {bad}")
            return 1
        pack_entries = len(zf.namelist())
    print(f"[PASS] Packaged ZIP passed testzip() with zero errors ({pack_entries:,} entries verified).")

    # 8. Generate Conversion & Audit Report
    t_elapsed = time.time() - t0
    print(f"\n[STEP 8] Writing conversion report to: {REPORT_PATH}")

    total_imgs = sum(split_stats[s]["images"] for s in split_stats)
    total_lbls = sum(split_stats[s]["labels"] for s in split_stats)
    total_insts = sum(split_stats[s]["instances"] for s in split_stats)

    report_lines: List[str] = [
        "=" * 78,
        "FITLOOP PHASE 8.2: ROBOFLOW DATASET CONVERSION & PACKAGING REPORT",
        "=" * 78,
        f"Execution Timestamp   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Source Archive        : {SRC_ZIP_PATH}",
        f"Output Directory      : {OUTPUT_DIR}",
        f"Output Archive        : {PACKAGE_ZIP_PATH}",
        f"Output Archive Size   : {zip_size_mb:.2f} MB ({PACKAGE_ZIP_PATH.stat().st_size:,} bytes)",
        f"Total Packaged Files  : {pack_entries:,}",
        f"Execution Total Time  : {t_elapsed:.2f} s",
        "",
        "=" * 78,
        "SECTION 1: CLASS REMAPPING SCHEMA (ROBOFLOW 8 -> FITLOOP 13)",
        "=" * 78,
        f"{'Roboflow Class':<22} | {'Target DeepFashion2 Class':<24} | {'Target ID':<10}",
        "-" * 78,
    ]

    for rf_id, rf_name in enumerate(ROBOFLOW_NAMES):
        tgt_id = ROBOFLOW_TO_FITLOOP_MAP[rf_id]
        tgt_name = FITLOOP_13_CLASSES[tgt_id]
        report_lines.append(f"{rf_id}: {rf_name:<19} -> {tgt_name:<24} | ID: {tgt_id}")

    report_lines.extend([
        "",
        "=" * 78,
        "SECTION 2: SPLIT IMAGE & LABEL FILE COUNTS",
        "=" * 78,
        f"{'Split':<8} | {'Images Count':<14} | {'Label Files':<14} | {'Instances':<12} | {'Empty Labels':<14}",
        "-" * 78,
    ])

    for sp in ["train", "val", "test"]:
        st = split_stats[sp]
        report_lines.append(
            f"{sp:<8} | {st['images']:>14,} | {st['labels']:>14,} | {st['instances']:>12,} | {st['empty_labels']:>14,}"
        )

    report_lines.extend([
        "-" * 78,
        f"{'TOTAL':<8} | {total_imgs:>14,} | {total_lbls:>14,} | {total_insts:>12,} | {sum(split_stats[s]['empty_labels'] for s in split_stats):>14,}",
        "",
        "=" * 78,
        "SECTION 3: PER-CLASS CONVERTED INSTANCE COUNTS (FITLOOP 13 SCHEMA)",
        "=" * 78,
        f"{'ID':<4} | {'FitLoop Class Name':<24} | {'Train':<9} | {'Val':<8} | {'Test':<8} | {'Total':<9}",
        "-" * 78,
    ])

    for cid, cname in enumerate(FITLOOP_13_CLASSES):
        tr_c = class_instance_counts["train"].get(cid, 0)
        val_c = class_instance_counts["val"].get(cid, 0)
        test_c = class_instance_counts["test"].get(cid, 0)
        tot_c = tr_c + val_c + test_c
        report_lines.append(f"{cid:<4} | {cname:<24} | {tr_c:>9,} | {val_c:>8,} | {test_c:>8,} | {tot_c:>9,}")

    report_lines.extend([
        "-" * 78,
        f"{'ALL':<4} | {'TOTAL CONVERTED INSTANCES':<24} | {sum(class_instance_counts['train'].values()):>9,} | {sum(class_instance_counts['val'].values()):>8,} | {sum(class_instance_counts['test'].values()):>8,} | {total_insts:>9,}",
        "",
        "=" * 78,
        "SECTION 4: 100% LABEL INTEGRITY VALIDATION AUDIT",
        "=" * 78,
        f"Total Label Files Audited    : {len(all_label_files):,}",
        f"Validation Errors Encountered: {len(total_validation_errors)} (100% PASSED)",
        f"5-Token Line Format Check    : 100.0% Compliant (class cx cy w h)",
        f"Class ID Range [0, 12] Check : 100.0% Compliant",
        f"Coordinates [0.0, 1.0] Bounds: 100.0% Compliant & strictly clipped",
        f"Zero-Area Boxes Rejection    : {sum(rejection_reasons.values()):,} boxes rejected",
        f"Packaged ZIP testzip() Status: PASSED (Zero corruption)",
        "=" * 78,
    ])

    full_report_str = "\n".join(report_lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(full_report_str, encoding="utf-8")
    print(f"\nReport successfully saved to:\n  {REPORT_PATH}")
    print("\n" + full_report_str)

    print(f"\n[COMPLETE] Phase 8.2 conversion pipeline completed successfully in {t_elapsed:.2f}s.")
    return 0


if __name__ == "__main__":
    code = run_pipeline()
    sys.exit(code)
