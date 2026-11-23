"""
FitLoop Phase 10.1: Angled/Folded Dataset (v2) Converter & Packager for Kaggle Phase 5
========================================================================================
1. Ingests C:\\Users\\Lenovo\\Downloads\\fitloop-folded-trousers.v2i.yolo26.zip.
2. Dynamically reads source class names from data.yaml and maps them to FitLoop's 13-class schema
   (handling both exact FitLoop names like 'short sleeve top', 'long sleeve top', 'long sleeve outwear',
   'trousers', and aliases like 'Tshirt' -> 0, 'shirt'/'sweater' -> 1, 'jacket' -> 3, 'pants' -> 7,
   'dress' -> 9; aborts immediately if any unrecognized class name is encountered).
3. Converts any polygon coordinates to normalized 5-column YOLO boxes (class_id cx cy w h),
   clipped strictly to [0.0, 1.0], skipping zero-area boxes.
4. Outputs organized images/ and labels/ splits (train, val, test) and 13-class data.yaml
   into wardrobe_ml/outputs/kaggle_phase5_angled/.
5. Validates 100% of converted label files (5 tokens, class in [0, 12], coordinates in [0, 1]).
6. Packages images, labels, and data.yaml into fitloop_phase5_angled_wardrobe.zip and
   verifies archive integrity with testzip().
7. Generates conversion and audit report at wardrobe_ml/outputs/angled_phase5_prep_report.txt.
"""

from __future__ import annotations

import argparse
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

# Centralized default constants
DEFAULT_SRC_ZIP = Path(r"C:\Users\Lenovo\Downloads\fitloop-folded-trousers.v2i.yolo26.zip")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\kaggle_phase5_angled")
DEFAULT_REPORT_PATH = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\angled_phase5_prep_report.txt")
DEFAULT_PACKAGE_NAME = "fitloop_phase5_angled_wardrobe.zip"

# FitLoop Canonical 13-Class DeepFashion2 Schema
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

# Comprehensive fashion taxonomy synonym lookup for dynamic class mapping
SYNONYM_TO_FITLOOP_CLASS: Dict[str, int] = {
    # 0: short sleeve top
    "short sleeve top": 0,
    "short_sleeve_top": 0,
    "tshirt": 0,
    "t-shirt": 0,
    "t shirt": 0,
    "tee": 0,
    "top": 0,
    # 1: long sleeve top
    "long sleeve top": 1,
    "long_sleeve_top": 1,
    "shirt": 1,
    "sweater": 1,
    "sweatshirt": 1,
    "hoodie": 1,
    "cardigan": 1,
    "pullover": 1,
    "blouse": 1,
    "long sleeve": 1,
    # 2: short sleeve outwear
    "short sleeve outwear": 2,
    "short_sleeve_outwear": 2,
    # 3: long sleeve outwear
    "long sleeve outwear": 3,
    "long_sleeve_outwear": 3,
    "jacket": 3,
    "coat": 3,
    "outwear": 3,
    "outerwear": 3,
    "blazer": 3,
    # 4: vest
    "vest": 4,
    "waistcoat": 4,
    # 5: sling
    "sling": 5,
    "tank top": 5,
    "tank": 5,
    "camisole": 5,
    "halter": 5,
    # 6: shorts
    "short": 6,
    "shorts": 6,
    # 7: trousers
    "trousers": 7,
    "trouser": 7,
    "pants": 7,
    "pant": 7,
    "jeans": 7,
    "slacks": 7,
    "trousers/pants": 7,
    "leggings": 7,
    "sweatpants": 7,
    "chinos": 7,
    "cargo": 7,
    # 8: skirt
    "skirt": 8,
    "mini skirt": 8,
    "maxi skirt": 8,
    # 9: short sleeve dress
    "dress": 9,
    "short sleeve dress": 9,
    "short_sleeve_dress": 9,
    # 10: long sleeve dress
    "long sleeve dress": 10,
    "long_sleeve_dress": 10,
    # 11: vest dress
    "vest dress": 11,
    "vest_dress": 11,
    # 12: sling dress
    "sling dress": 12,
    "sling_dress": 12,
}

# Split naming alignment: Roboflow split name -> Standard YOLO target directory
SPLIT_ALIGNMENT: Dict[str, str] = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def resolve_class_name(raw_name: str) -> int:
    """
    Resolve a free-form fashion class name to FitLoop's 13-class ID.
    Raises ValueError if unrecognized.
    """
    clean_name = raw_name.strip().lower()
    if clean_name in SYNONYM_TO_FITLOOP_CLASS:
        return SYNONYM_TO_FITLOOP_CLASS[clean_name]

    # Try normalizing delimiters (hyphens/underscores to spaces)
    normalized = clean_name.replace("-", " ").replace("_", " ")
    if normalized in SYNONYM_TO_FITLOOP_CLASS:
        return SYNONYM_TO_FITLOOP_CLASS[normalized]

    raise ValueError(
        f"Unrecognized class name: '{raw_name}' cannot be mapped to FitLoop's 13-class schema."
    )


def build_dynamic_class_map(raw_yaml_text: str) -> Tuple[Dict[int, int], Dict[int, str], Dict[str, Any]]:
    """
    Parse source data.yaml, extract class names, and dynamically construct
    the mapping from source class ID -> FitLoop 13-class ID.
    """
    data = yaml.safe_load(raw_yaml_text)
    if not isinstance(data, dict):
        raise ValueError("Invalid data.yaml: Expected top-level YAML dictionary.")

    names_entry = data.get("names")
    if names_entry is None:
        raise ValueError("Invalid data.yaml: Missing 'names' field.")

    source_id_to_name: Dict[int, str] = {}
    if isinstance(names_entry, list):
        for idx, name in enumerate(names_entry):
            source_id_to_name[idx] = str(name)
    elif isinstance(names_entry, dict):
        for idx_k, name_v in names_entry.items():
            source_id_to_name[int(idx_k)] = str(name_v)
    else:
        raise ValueError(f"Invalid 'names' field type in data.yaml: {type(names_entry)}")

    source_to_fitloop_map: Dict[int, int] = {}
    for src_id, src_name in sorted(source_id_to_name.items()):
        fitloop_id = resolve_class_name(src_name)
        source_to_fitloop_map[src_id] = fitloop_id

    return source_to_fitloop_map, source_id_to_name, data


def parse_and_convert_label_line(
    raw_line: str,
    class_map: Dict[int, int],
    line_num: int = 1,
    file_name: str = "",
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse a label line (standard 5-token bbox or N-token polygon), remap the class,
    clip coordinates to [0.0, 1.0], filter zero-area boxes, and return the formatted
    YOLO line string `class_id cx cy w h`.
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

    if raw_cls not in class_map:
        return None, None, f"unmapped_source_class_id_{raw_cls}"

    target_cls = class_map[raw_cls]

    # Case 1: Standard 5-column YOLO Bounding Box
    if len(tokens) == 5:
        try:
            cx_raw = float(tokens[1])
            cy_raw = float(tokens[2])
            w_raw = float(tokens[3])
            h_raw = float(tokens[4])
        except ValueError:
            return None, None, "unparseable_bbox_floats"

        cx = max(0.0, min(1.0, cx_raw))
        cy = max(0.0, min(1.0, cy_raw))
        w = max(0.0, min(1.0, w_raw))
        h = max(0.0, min(1.0, h_raw))

        if w <= 1e-6 or h <= 1e-6:
            return None, None, f"zero_area_bbox (w={w_raw}, h={h_raw})"

        return f"{target_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}", target_cls, None

    # Case 2: Polygon Segmentation (N-vertex coordinates)
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

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

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

        try:
            cid = int(parts[0])
            if cid < 0 or cid > 12:
                errors.append(f"{file_path.name}:{idx}: Class ID {cid} outside valid range [0, 12]")
        except ValueError:
            errors.append(f"{file_path.name}:{idx}: Non-integer Class ID '{parts[0]}'")

        try:
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            errors.append(f"{file_path.name}:{idx}: Unparseable float coordinates")
            continue

        for val, name in [(cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")]:
            if not math.isfinite(val):
                errors.append(f"{file_path.name}:{idx}: Non-finite float value {name}={val}")

        if not (0.0 <= cx <= 1.0):
            errors.append(f"{file_path.name}:{idx}: cx={cx} out of bounds [0.0, 1.0]")
        if not (0.0 <= cy <= 1.0):
            errors.append(f"{file_path.name}:{idx}: cy={cy} out of bounds [0.0, 1.0]")
        if not (0.0 < w <= 1.0):
            errors.append(f"{file_path.name}:{idx}: w={w} out of bounds (0.0, 1.0]")
        if not (0.0 < h <= 1.0):
            errors.append(f"{file_path.name}:{idx}: h={h} out of bounds (0.0, 1.0]")

    return errors


def run_conversion(
    src_zip_path: Path = DEFAULT_SRC_ZIP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    package_name: str = DEFAULT_PACKAGE_NAME,
) -> int:
    """
    Main conversion and packaging pipeline.
    """
    t0 = time.time()
    print("=" * 78)
    print("FITLOOP PHASE 10.1: ANGLED/FOLDED DATASET CONVERTER & PACKAGER")
    print("=" * 78)

    # 1. Verify Source Archive
    print(f"\n[STEP 1] Validating source archive: {src_zip_path}")
    if not src_zip_path.exists():
        print(f"[FATAL] Source archive not found: {src_zip_path}")
        return 1

    with zipfile.ZipFile(src_zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            print(f"[FATAL] Source ZIP corruption detected: {bad_file}")
            return 1
        namelist = set(zf.namelist())
        print(f"[PASS] Source archive verified cleanly ({len(namelist):,} entries).")

        # 2. Inspect data.yaml & Build Dynamic Class Map
        print("\n[STEP 2] Inspecting data.yaml and dynamically mapping class names...")
        if "data.yaml" not in namelist:
            print("[FATAL] 'data.yaml' not found in source archive.")
            return 1

        raw_yaml_text = zf.read("data.yaml").decode("utf-8", errors="replace")
        try:
            source_to_fitloop_map, source_id_to_name, raw_yaml_dict = build_dynamic_class_map(raw_yaml_text)
        except ValueError as e:
            print(f"[FATAL] Dynamic class mapping failed: {e}")
            return 1

    print(f"  Source dataset 'data.yaml' classes count: {len(source_id_to_name)}")
    for src_id, src_name in sorted(source_id_to_name.items()):
        fit_id = source_to_fitloop_map[src_id]
        fit_name = FITLOOP_13_CLASSES[fit_id]
        print(f"  Mapping: Source [{src_id}] '{src_name}' -> FitLoop [{fit_id}] '{fit_name}'")

    # 3. Setup Clean Output Directory Structure
    print(f"\n[STEP 3] Setting up clean output directory structure at: {output_dir}")
    images_base = output_dir / "images"
    labels_base = output_dir / "labels"

    # Reset existing output directory to ensure clean conversion
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    for target_split in ["train", "val", "test"]:
        (images_base / target_split).mkdir(parents=True, exist_ok=True)
        (labels_base / target_split).mkdir(parents=True, exist_ok=True)

    # 4. Extraction, Label Conversion & Class Remapping
    print("\n[STEP 4] Converting labels, remapping classes, and extracting images...")
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

    with zipfile.ZipFile(src_zip_path, "r") as zf:
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

                # 4a. Extract image
                dest_img_path = images_base / target_split / f"{stem}{suffix}"
                with zf.open(img_entry, "r") as src_f, open(dest_img_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                split_stats[target_split]["images"] += 1

                # 4b. Read and convert label
                converted_lines: List[str] = []
                if lbl_entry in namelist:
                    raw_text = zf.read(lbl_entry).decode("utf-8", errors="replace").strip()
                    for idx, line in enumerate(raw_text.splitlines(), 1):
                        conv_line, tgt_cls, rej_reason = parse_and_convert_label_line(
                            line, source_to_fitloop_map, line_num=idx, file_name=lbl_entry
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

    # Clean up empty splits if any
    for sp in ["train", "val", "test"]:
        if split_stats[sp]["images"] == 0:
            if (images_base / sp).exists():
                (images_base / sp).rmdir()
            if (labels_base / sp).exists():
                (labels_base / sp).rmdir()

    # 5. Write 13-class data.yaml for Kaggle / YOLO Phase 5 Training
    print("\n[STEP 5] Writing 13-class data.yaml specification...")
    data_yaml_dict: Dict[str, Any] = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": 13,
        "names": {i: name for i, name in enumerate(FITLOOP_13_CLASSES)},
    }
    if split_stats["test"]["images"] > 0:
        data_yaml_dict["test"] = "images/test"

    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_dict, f, sort_keys=False)
    print(f"  data.yaml written to: {data_yaml_path}")

    # 6. Strict Label Validation Suite
    print("\n[STEP 6] Validating 100% of converted label files across all splits...")
    all_label_files: List[Path] = []
    for sp in ["train", "val", "test"]:
        sp_dir = labels_base / sp
        if sp_dir.exists():
            all_label_files.extend(list(sp_dir.glob("*.txt")))

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

    # 7. Package into fitloop_phase5_angled_wardrobe.zip
    package_zip_path = output_dir / package_name
    print(f"\n[STEP 7] Packaging dataset into {package_zip_path}...")
    if package_zip_path.exists():
        package_zip_path.unlink()

    files_to_pack: List[Tuple[Path, str]] = []
    files_to_pack.append((data_yaml_path, "data.yaml"))

    for root_dir in [images_base, labels_base]:
        if root_dir.exists():
            for p in root_dir.rglob("*"):
                if p.is_file():
                    rel_path = p.relative_to(output_dir).as_posix()
                    files_to_pack.append((p, rel_path))

    print(f"  Compressing {len(files_to_pack):,} files with ZIP_DEFLATED...")
    with zipfile.ZipFile(package_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for disk_path, arc_path in files_to_pack:
            zf.write(disk_path, arc_path)

    zip_size_mb = package_zip_path.stat().st_size / (1024 * 1024)
    print(f"  Archive created: {zip_size_mb:.2f} MB ({package_zip_path.stat().st_size:,} bytes)")

    # 8. Test Output ZIP Integrity
    print("\n[STEP 8] Verifying packaged archive integrity (testzip)...")
    with zipfile.ZipFile(package_zip_path, "r") as zf:
        bad = zf.testzip()
        if bad:
            print(f"[FATAL] Packaged ZIP corruption detected: {bad}")
            return 1
        pack_entries = len(zf.namelist())
    print(f"[PASS] Packaged ZIP passed testzip() with zero errors ({pack_entries:,} entries verified).")

    # 9. Generate Conversion & Audit Report
    t_elapsed = time.time() - t0
    print(f"\n[STEP 9] Writing conversion report to: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_imgs = sum(split_stats[s]["images"] for s in split_stats)
    total_lbls = sum(split_stats[s]["labels"] for s in split_stats)
    total_insts = sum(split_stats[s]["instances"] for s in split_stats)

    report_lines: List[str] = [
        "=" * 78,
        "FITLOOP PHASE 10.1: ANGLED/FOLDED DATASET PREPARATION REPORT",
        "=" * 78,
        f"Execution Timestamp : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Execution Duration  : {t_elapsed:.2f} s",
        f"Python Version      : {sys.version.split()[0]}",
        f"Source Archive      : {src_zip_path}",
        f"Source ZIP Size     : {src_zip_path.stat().st_size:,} bytes ({src_zip_path.stat().st_size / (1024*1024):.2f} MB)",
        f"Destination Folder  : {output_dir}",
        f"Packaged ZIP Path   : {package_zip_path}",
        f"Packaged ZIP Size   : {package_zip_path.stat().st_size:,} bytes ({zip_size_mb:.2f} MB)",
        f"ZIP Integrity Check : PASSED (testzip returned None, {pack_entries:,} entries)",
        "-" * 78,
        "SOURCE DATA.YAML SPECIFICATION",
        "-" * 78,
        f"Raw YAML Keys       : {list(raw_yaml_dict.keys())}",
        f"Source Classes (nc) : {len(source_id_to_name)}",
        "Raw Classes Content :",
    ]
    for line in raw_yaml_text.strip().splitlines():
        report_lines.append(f"  | {line}")

    report_lines.extend([
        "-" * 78,
        "DYNAMIC CLASS REMAPPING TABLE (Source -> FitLoop 13-Class Schema)",
        "-" * 78,
        f"{'Src ID':<8} {'Source Name':<18} -> {'FitLoop ID':<12} {'FitLoop Class Name':<25}",
        f"{'-'*8} {'-'*18}    {'-'*12} {'-'*25}",
    ])
    for src_id, src_name in sorted(source_id_to_name.items()):
        fit_id = source_to_fitloop_map[src_id]
        fit_name = FITLOOP_13_CLASSES[fit_id]
        report_lines.append(f"{src_id:<8} {src_name:<18} -> {fit_id:<12} {fit_name:<25}")

    report_lines.extend([
        "-" * 78,
        "PER-SPLIT CONVERSION SUMMARY",
        "-" * 78,
        f"{'Split':<8} {'Images':<8} {'Labels':<8} {'Instances':<11} {'Empty':<8} {'Rejected':<10}",
        f"{'-'*8} {'-'*8} {'-'*8} {'-'*11} {'-'*8} {'-'*10}",
    ])
    for sp in ["train", "val", "test"]:
        st = split_stats[sp]
        report_lines.append(
            f"{sp:<8} {st['images']:<8} {st['labels']:<8} {st['instances']:<11} {st['empty_labels']:<8} {st['rejected_boxes']:<10}"
        )
    report_lines.append(f"{'-'*8} {'-'*8} {'-'*8} {'-'*11} {'-'*8} {'-'*10}")
    report_lines.append(
        f"{'TOTAL':<8} {total_imgs:<8} {total_lbls:<8} {total_insts:<11} "
        f"{sum(split_stats[s]['empty_labels'] for s in split_stats):<8} "
        f"{sum(split_stats[s]['rejected_boxes'] for s in split_stats):<10}"
    )

    report_lines.extend([
        "-" * 78,
        "REMAPPED INSTANCE COUNTS PER FITLOOP CLASS",
        "-" * 78,
        f"{'Class ID':<10} {'FitLoop Class Name':<24} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}",
        f"{'-'*10} {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}",
    ])

    total_by_class: Counter[int] = Counter()
    for fit_id, fit_name in enumerate(FITLOOP_13_CLASSES):
        tr_cnt = class_instance_counts["train"][fit_id]
        val_cnt = class_instance_counts["val"][fit_id]
        tst_cnt = class_instance_counts["test"][fit_id]
        tot_cnt = tr_cnt + val_cnt + tst_cnt
        total_by_class[fit_id] = tot_cnt
        report_lines.append(
            f"{fit_id:<10} {fit_name:<24} {tr_cnt:<8} {val_cnt:<8} {tst_cnt:<8} {tot_cnt:<8}"
        )
    report_lines.append(f"{'-'*10} {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    report_lines.append(
        f"{'SUM':<10} {'All 13 Classes':<24} "
        f"{sum(class_instance_counts['train'].values()):<8} "
        f"{sum(class_instance_counts['val'].values()):<8} "
        f"{sum(class_instance_counts['test'].values()):<8} "
        f"{sum(total_by_class.values()):<8}"
    )

    report_lines.extend([
        "-" * 78,
        "LABEL VALIDATION SUITE RESULTS",
        "-" * 78,
        f"Total Label Files Inspected : {len(all_label_files):,}",
        f"Validation Errors Detected  : {len(total_validation_errors)}",
        "Schema Checks               : Exactly 5 tokens (class cx cy w h)",
        "Class ID Bounds             : Strictly integer in [0, 12]",
        "Coordinate Bounds           : Strictly finite floats, cx, cy in [0, 1], w, h in (0, 1]",
        f"Validation Result           : {'PASSED (100% Valid)' if not total_validation_errors else 'FAILED'}",
        "-" * 78,
        "OUTPUT ZIP INTEGRITY VERIFICATION",
        "-" * 78,
        f"Archive Filename            : {package_name}",
        f"Archive File Path           : {package_zip_path}",
        f"Archive Size (bytes)        : {package_zip_path.stat().st_size:,}",
        f"Archive Size (MB)           : {zip_size_mb:.2f} MB",
        f"Total Packaged Files        : {pack_entries:,}",
        f"testzip() Integrity Check  : PASSED (0 corrupted files)",
        "=" * 78,
    ])

    report_text = "\n".join(report_lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Report written successfully to: {report_path}")
    print("\n" + report_text)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FitLoop Phase 10.1: Angled/Folded Dataset Converter & Packager"
    )
    parser.add_argument(
        "--src-zip",
        type=Path,
        default=DEFAULT_SRC_ZIP,
        help="Path to source Roboflow zip file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for converted dataset and packaged zip.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for conversion diagnostic report.",
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default=DEFAULT_PACKAGE_NAME,
        help="Filename for packaged zip archive.",
    )

    args = parser.parse_args()
    sys.exit(
        run_conversion(
            src_zip_path=args.src_zip,
            output_dir=args.output_dir,
            report_path=args.report_path,
            package_name=args.package_name,
        )
    )
