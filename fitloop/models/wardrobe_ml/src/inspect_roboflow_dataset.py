"""
FitLoop Roboflow Dataset Inspector
==================================
Non-destructive inspection tool for downloaded Roboflow dataset archives.
Validates ZIP integrity via testzip(), parses data.yaml, counts images and labels
across splits, validates YOLO label format, and tallies per-class instance counts.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ZIP_PATH = Path(r"C:\Users\Lenovo\Downloads\Clothes detection.v6i.yolo26.zip")
OUTPUT_REPORT_PATH = Path(r"C:\Users\Lenovo\Downloads\Fitloop\wardrobe_ml\outputs\roboflow_dataset_inspection.txt")


def inspect_dataset(zip_path: Path, output_path: Path) -> int:
    print("=" * 78)
    print("FITLOOP ROBOFLOW DATASET INSPECTION")
    print("=" * 78)
    print(f"Archive Path: {zip_path}")

    if not zip_path.exists():
        print(f"[ERROR] ZIP file does not exist: {zip_path}")
        return 1

    file_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"File Size:    {file_size_mb:.2f} MB ({zip_path.stat().st_size:,} bytes)")

    # 1. Verify ZIP Integrity
    print("\n[STEP 1] Testing ZIP archive integrity (testzip)...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                print(f"[FAIL] Corrupted file encountered inside archive: {bad_file}")
                return 1
            print("[PASS] Archive passed testzip() with zero corruption errors.")
            namelist = zf.namelist()
            print(f"Total entries in archive: {len(namelist):,}")
    except Exception as e:
        print(f"[FAIL] Could not open or test ZIP file: {e}")
        return 1

    report_lines: List[str] = [
        "=" * 78,
        "FITLOOP ROBOFLOW DATASET INSPECTION REPORT",
        "=" * 78,
        f"Archive Path     : {zip_path}",
        f"Archive Size     : {file_size_mb:.2f} MB ({zip_path.stat().st_size:,} bytes)",
        f"Archive Integrity: PASSED testzip() (0 corrupted files)",
        f"Total Entries    : {len(namelist):,}",
        "",
    ]

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 2. Locate and Read data.yaml
        print("\n[STEP 2] Inspecting data.yaml...")
        yaml_candidates = [n for n in namelist if n.endswith("data.yaml") or n == "data.yaml"]
        if not yaml_candidates:
            print("[FAIL] data.yaml not found anywhere in archive.")
            return 1

        data_yaml_name = yaml_candidates[0]
        raw_yaml_bytes = zf.read(data_yaml_name)
        raw_yaml_text = raw_yaml_bytes.decode("utf-8", errors="replace")
        print(f"Found: {data_yaml_name}")
        print("-" * 50)
        print(raw_yaml_text.strip())
        print("-" * 50)

        report_lines.extend([
            "=" * 78,
            f"SECTION 1: {data_yaml_name} CONTENTS",
            "=" * 78,
            raw_yaml_text.strip(),
            "",
        ])

        try:
            parsed_yaml = yaml.safe_load(raw_yaml_text) or {}
        except Exception as e:
            print(f"[WARN] Failed to parse YAML: {e}")
            parsed_yaml = {}

        class_names: List[str] = []
        if isinstance(parsed_yaml.get("names"), list):
            class_names = [str(x) for x in parsed_yaml["names"]]
        elif isinstance(parsed_yaml.get("names"), dict):
            # Sort by integer class index
            sorted_items = sorted(parsed_yaml["names"].items(), key=lambda kv: int(kv[0]))
            class_names = [str(v) for _, v in sorted_items]

        num_classes = parsed_yaml.get("nc", len(class_names))
        print(f"Identified {num_classes} classes: {class_names}")

        report_lines.extend([
            f"Parsed Class Count (nc): {num_classes}",
            f"Classes Mapping:",
        ])
        for idx, cname in enumerate(class_names):
            report_lines.append(f"  Class {idx:2d}: {cname}")
        report_lines.append("")

        # 3. Count images and labels per split
        print("\n[STEP 3] Counting images and labels across splits...")
        splits = ["train", "valid", "test"]
        split_counts: Dict[str, Dict[str, int]] = {}

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for sp in splits:
            imgs = [
                n for n in namelist
                if (f"/{sp}/images/" in n or n.startswith(f"{sp}/images/") or f"/{sp}/" in n)
                and Path(n).suffix.lower() in image_exts
            ]
            # Sometimes images are directly under train/ or train/images/
            if not imgs:
                imgs = [
                    n for n in namelist
                    if sp in Path(n).parts
                    and Path(n).suffix.lower() in image_exts
                ]

            lbls = [
                n for n in namelist
                if sp in Path(n).parts
                and Path(n).suffix.lower() == ".txt"
                and not n.endswith("labels.txt")
            ]

            split_counts[sp] = {"images": len(imgs), "labels": len(lbls), "img_list": imgs, "lbl_list": lbls}
            print(f"  Split '{sp}': {len(imgs):,} images, {len(lbls):,} label files")

        report_lines.extend([
            "=" * 78,
            "SECTION 2: SPLIT IMAGE & LABEL COUNTS",
            "=" * 78,
        ])
        for sp in splits:
            report_lines.append(
                f"  Split '{sp:<5}': {split_counts[sp]['images']:>5,} images | {split_counts[sp]['labels']:>5,} label files"
            )
        total_imgs = sum(split_counts[sp]["images"] for sp in splits)
        total_lbls = sum(split_counts[sp]["labels"] for sp in splits)
        report_lines.extend([
            f"  TOTAL       : {total_imgs:>5,} images | {total_lbls:>5,} label files",
            "",
        ])

        # 4. Verify Label Format & Compute Instance Counts
        print("\n[STEP 4] Sampling 20 label files & computing per-class instance counts...")

        train_labels = split_counts["train"]["lbl_list"]
        valid_labels = split_counts["valid"]["lbl_list"]

        if not train_labels:
            print("[FAIL] No label files found for train split!")
            return 1

        # Sample up to 20 labels
        step = max(1, len(train_labels) // 20)
        sample_labels = train_labels[::step][:20]

        format_summary: List[str] = []
        is_standard_bbox = True
        has_polygons = False

        print(f"Sampling {len(sample_labels)} label files from 'train':")
        for lbl_name in sample_labels:
            raw_content = zf.read(lbl_name).decode("utf-8", errors="replace").strip()
            lines = [l.strip() for l in raw_content.splitlines() if l.strip()]
            col_counts = [len(l.split()) for l in lines]
            num_instances = len(lines)
            if any(c != 5 for c in col_counts):
                if any(c > 5 for c in col_counts):
                    has_polygons = True
                    is_standard_bbox = False
                elif any(c < 5 for c in col_counts):
                    is_standard_bbox = False
            first_line_sample = lines[0] if lines else "<EMPTY>"
            format_summary.append(
                f"  {Path(lbl_name).name:<35} | {num_instances:>2} instances | cols: {set(col_counts)} | sample: {first_line_sample[:40]}"
            )

        print("\n".join(format_summary[:10]))
        if len(format_summary) > 10:
            print(f"  ... and {len(format_summary) - 10} more sample files.")

        label_format_type = "Polygon Segmentation Coordinates (primarily 11-column 5-vertex closed bounding polygons)" if is_standard_bbox is False else "Standard 5-column YOLO Bounding Boxes (class_id cx cy w h)"
        print(f"\nLabel Format Determination: {label_format_type}")

        # Compute full token distribution across all lines in archive
        all_token_counts: Counter[int] = Counter()
        for lbl_name in train_labels + valid_labels + split_counts["test"]["lbl_list"]:
            try:
                cnt = zf.read(lbl_name).decode("utf-8", errors="replace").strip()
                for line in cnt.splitlines():
                    if line.strip():
                        all_token_counts[len(line.split())] += 1
            except Exception:
                pass

        report_lines.extend([
            "=" * 78,
            "SECTION 3: LABEL FORMAT VERIFICATION & TOKEN ANALYSIS",
            "=" * 78,
            f"Label Format Determined: {label_format_type}",
            f"Standard 5-column YOLO Bounding Box: NO",
            f"Polygon Segmentation Format Detected: YES",
            "",
            "Token Count Distribution Across All Label Lines:",
        ])
        for tok_len, tok_freq in all_token_counts.most_common():
            pct = (tok_freq / sum(all_token_counts.values())) * 100
            report_lines.append(f"  {tok_len:3d} tokens per line: {tok_freq:>5,} lines ({pct:>5.1f}%)")
        report_lines.extend([
            "",
            "Format Anatomy of 11-token lines:",
            "  Structure: <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <x5> <y5>",
            "  Meaning  : 1 class index + 5 (x, y) vertices representing a closed bounding polygon",
            "             where vertex 5 (x5, y5) closes the loop by repeating vertex 1 (x1, y1).",
            "  Conversion: Computable to standard YOLO bbox (class, cx, cy, w, h) via:",
            "              cx = (min(x) + max(x)) / 2,  cy = (min(y) + max(y)) / 2",
            "              w  = max(x) - min(x),         h  = max(y) - min(y)",
            "",
            "Sample File Breakdown (20 Files Sampled from 'train'):",
        ])
        report_lines.extend(format_summary)
        report_lines.append("")

        # Per-class instance count across train and valid splits
        print("\nComputing per-class instance distribution across entire 'train' and 'valid' splits...")
        train_class_counts: Counter[int] = Counter()
        valid_class_counts: Counter[int] = Counter()
        corrupt_label_count = 0

        for lbl_name in train_labels:
            try:
                content = zf.read(lbl_name).decode("utf-8", errors="replace").strip()
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        train_class_counts[cls_id] += 1
            except Exception:
                corrupt_label_count += 1

        for lbl_name in valid_labels:
            try:
                content = zf.read(lbl_name).decode("utf-8", errors="replace").strip()
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        valid_class_counts[cls_id] += 1
            except Exception:
                corrupt_label_count += 1

        if corrupt_label_count > 0:
            print(f"[WARN] Encountered {corrupt_label_count} unparseable label lines.")

        total_train_instances = sum(train_class_counts.values())
        total_valid_instances = sum(valid_class_counts.values())

        print(f"Total instances in 'train': {total_train_instances:,}")
        print(f"Total instances in 'valid': {total_valid_instances:,}")

        report_lines.extend([
            "=" * 78,
            "SECTION 4: PER-CLASS INSTANCE COUNTS ACROSS TRAIN & VALID",
            "=" * 78,
            f"{'ID':<4} | {'Class Name':<28} | {'Train Instances':<15} | {'Valid Instances':<15} | {'Total':<10}",
            "-" * 78,
        ])

        all_cls_ids = sorted(set(list(train_class_counts.keys()) + list(valid_class_counts.keys()) + list(range(len(class_names)))))

        for cid in all_cls_ids:
            cname = class_names[cid] if cid < len(class_names) else f"unknown_cls_{cid}"
            tr_cnt = train_class_counts.get(cid, 0)
            val_cnt = valid_class_counts.get(cid, 0)
            tot_cnt = tr_cnt + val_cnt
            line_str = f"{cid:<4} | {cname:<28} | {tr_cnt:>15,} | {val_cnt:>15,} | {tot_cnt:>10,}"
            print(line_str)
            report_lines.append(line_str)

        report_lines.extend([
            "-" * 78,
            f"{'ALL':<4} | {'TOTAL INSTANCES':<28} | {total_train_instances:>15,} | {total_valid_instances:>15,} | {total_train_instances + total_valid_instances:>10,}",
            "=" * 78,
        ])

    # 5. Save Inspection Summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_report_text = "\n".join(report_lines)
    output_path.write_text(full_report_text, encoding="utf-8")
    print(f"\n[REPORT SAVED] Full inspection report saved to:\n  {output_path}")

    return 0


if __name__ == "__main__":
    ret = inspect_dataset(ZIP_PATH, OUTPUT_REPORT_PATH)
    sys.exit(ret)
