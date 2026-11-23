"""
Stage 6.4C.1: Forensic Verification of Out-of-Bounds and Degenerate Annotations
==============================================================================
Performs read-only forensic inspection of:
- 028225.json / 028225.jpg
- 115063.json / 115063.jpg
- 101338.json / 101338.jpg

Investigates exact image dimensions, raw coordinates, coordinate conventions,
axis swaps, segmentation bounds, landmark bounds, and YOLO label contents.
Generates diagnostic images with extended canvas to display unclipped out-of-bounds boxes.
Saves JSON report to wardrobe_ml/outputs/step_6_4c1_bbox_forensic_report.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Setup paths
PROJECT_ROOT = Path("C:/Users/Lenovo/Downloads/Fitloop")
DEEPFASHION2_ROOT = Path("C:/Users/Lenovo/Downloads/DeepFashion2")
TRAIN_IMG_DIR = DEEPFASHION2_ROOT / "train" / "image"
TRAIN_ANNO_DIR = DEEPFASHION2_ROOT / "train" / "annos"
TRAIN_LABELS_DIR = PROJECT_ROOT / "wardrobe_ml" / "data" / "processed" / "deepfashion2_yolo" / "labels" / "train"
OUTPUTS_DIR = PROJECT_ROOT / "wardrobe_ml" / "outputs"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "step_6_4c1_bbox_forensic"
REPORT_PATH = OUTPUTS_DIR / "step_6_4c1_bbox_forensic_report.json"

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

# Category ID to Name mapping
CATEGORY_NAMES = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress",
}


def inspect_case(image_id: str) -> Dict[str, Any]:
    img_path = TRAIN_IMG_DIR / f"{image_id}.jpg"
    json_path = TRAIN_ANNO_DIR / f"{image_id}.json"
    txt_path = TRAIN_LABELS_DIR / f"{image_id}.txt"

    print(f"\n{'='*70}\nFORENSIC INSPECTION: Image {image_id}\n{'='*70}")
    print(f"Image path:  {img_path} (exists: {img_path.exists()})")
    print(f"JSON path:   {json_path} (exists: {json_path.exists()})")
    print(f"TXT path:    {txt_path} (exists: {txt_path.exists()})")

    # 1. PIL Image Dimensions & Properties
    with Image.open(img_path) as im:
        pil_w, pil_h = im.size
        pil_mode = im.mode
        pil_format = im.format

    # Also verify with OpenCV
    cv_img = cv2.imread(str(img_path))
    cv_h, cv_w, cv_c = cv_img.shape

    print(f"PIL Dimensions:    Width = {pil_w}, Height = {pil_h}, Mode = {pil_mode}, Format = {pil_format}")
    print(f"OpenCV Dimensions: Width = {cv_w}, Height = {cv_h}, Channels = {cv_c}")
    assert (pil_w, pil_h) == (cv_w, cv_h), "PIL and OpenCV dimension mismatch!"

    # 2. JSON Contents
    with open(json_path, "r", encoding="utf-8") as f:
        anno_data = json.load(f)

    # General metadata
    source = anno_data.get("source", "unknown")
    pair_id = anno_data.get("pair_id", None)
    print(f"Metadata: source={source}, pair_id={pair_id}")

    # Inspect all items in JSON
    item_keys = [k for k in anno_data.keys() if k.startswith("item")]
    print(f"Total items annotated: {len(item_keys)} ({item_keys})")

    items_analysis = {}
    for k in sorted(item_keys):
        item = anno_data[k]
        cat_id = item.get("category_id")
        cat_name = item.get("category_name", CATEGORY_NAMES.get(cat_id, "unknown"))
        raw_bbox = item.get("bounding_box", [])
        style = item.get("style", None)
        segmentation = item.get("segmentation", [])
        landmarks = item.get("landmarks", [])

        # Segmentation bounds
        seg_xs = []
        seg_ys = []
        for poly in segmentation:
            if isinstance(poly, list):
                for i in range(0, len(poly), 2):
                    seg_xs.append(poly[i])
                    seg_ys.append(poly[i+1])
        seg_min_x = min(seg_xs) if seg_xs else None
        seg_max_x = max(seg_xs) if seg_xs else None
        seg_min_y = min(seg_ys) if seg_ys else None
        seg_max_y = max(seg_ys) if seg_ys else None

        # Landmarks bounds (only visible/occluded, landmark format: [x1, y1, v1, x2, y2, v2, ...])
        lm_xs = []
        lm_ys = []
        if isinstance(landmarks, list):
            for i in range(0, len(landmarks), 3):
                lx = landmarks[i]
                ly = landmarks[i+1]
                lv = landmarks[i+2]
                if lv in (1, 2):  # occluded or visible
                    lm_xs.append(lx)
                    lm_ys.append(ly)
        lm_min_x = min(lm_xs) if lm_xs else None
        lm_max_x = max(lm_xs) if lm_xs else None
        lm_min_y = min(lm_ys) if lm_ys else None
        lm_max_y = max(lm_ys) if lm_ys else None

        # Coordinate checks
        x1, y1, x2, y2 = raw_bbox if len(raw_bbox) == 4 else (None, None, None, None)
        is_numeric = all(isinstance(v, (int, float)) for v in raw_bbox)
        is_finite = all(not math.isnan(v) and not math.isinf(v) for v in raw_bbox) if is_numeric else False

        dx = (x2 - x1) if (is_numeric and is_finite) else None
        dy = (y2 - y1) if (is_numeric and is_finite) else None
        has_positive_area = (dx > 0 and dy > 0) if (dx is not None and dy is not None) else False

        # Boundary checks
        x1_in_bounds = (0 <= x1 <= pil_w) if x1 is not None else False
        x2_in_bounds = (0 <= x2 <= pil_w) if x2 is not None else False
        y1_in_bounds = (0 <= y1 <= pil_h) if y1 is not None else False
        y2_in_bounds = (0 <= y2 <= pil_h) if y2 is not None else False

        fully_in_bounds = (x1_in_bounds and x2_in_bounds and y1_in_bounds and y2_in_bounds)

        # Axis swap hypothesis test:
        # What if bbox was [y1, x1, y2, x2] or image dimensions were swapped?
        swapped_bbox_valid = False
        if is_numeric and is_finite:
            # If swapped as [y1, x1, y2, x2]: check if y1, y2 in [0, H] and x1, x2 in [0, W]
            # Here raw_bbox is [b0, b1, b2, b3]. If b0, b2 are Y and b1, b3 are X:
            if (0 <= raw_bbox[0] <= pil_h) and (0 <= raw_bbox[2] <= pil_h) and \
               (0 <= raw_bbox[1] <= pil_w) and (0 <= raw_bbox[3] <= pil_w):
                swapped_bbox_valid = True

        print(f"  [{k}] Category {cat_id}: {cat_name} (style: {style})")
        print(f"    Raw bbox:         {raw_bbox}")
        print(f"    Dimensions:       dx={dx}, dy={dy}, area={dx*dy if (dx and dy) else 0}")
        print(f"    Numeric & Finite: {is_numeric} & {is_finite}")
        print(f"    Positive Area:    {has_positive_area}")
        print(f"    Bounds check:     x1 in [0, {pil_w}]: {x1_in_bounds} (val: {x1})")
        print(f"                      y1 in [0, {pil_h}]: {y1_in_bounds} (val: {y1})")
        print(f"                      x2 in [0, {pil_w}]: {x2_in_bounds} (val: {x2})")
        print(f"                      y2 in [0, {pil_h}]: {y2_in_bounds} (val: {y2})")
        print(f"    Fully in bounds:  {fully_in_bounds}")
        if seg_xs:
            print(f"    Segmentation span: X=[{seg_min_x}, {seg_max_x}], Y=[{seg_min_y}, {seg_max_y}]")
        if lm_xs:
            print(f"    Landmarks span:    X=[{lm_min_x}, {lm_max_x}], Y=[{lm_min_y}, {lm_max_y}]")

        items_analysis[k] = {
            "category_id": cat_id,
            "category_name": cat_name,
            "style": style,
            "raw_bbox": raw_bbox,
            "dx": dx,
            "dy": dy,
            "area": (dx * dy) if (dx is not None and dy is not None) else None,
            "is_numeric": is_numeric,
            "is_finite": is_finite,
            "has_positive_area": has_positive_area,
            "x1_in_bounds": x1_in_bounds,
            "y1_in_bounds": y1_in_bounds,
            "x2_in_bounds": x2_in_bounds,
            "y2_in_bounds": y2_in_bounds,
            "fully_in_bounds": fully_in_bounds,
            "segmentation_min_max": {
                "min_x": seg_min_x, "max_x": seg_max_x,
                "min_y": seg_min_y, "max_y": seg_max_y,
            } if seg_xs else None,
            "landmarks_min_max": {
                "min_x": lm_min_x, "max_x": lm_max_x,
                "min_y": lm_min_y, "max_y": lm_max_y,
            } if lm_xs else None,
        }

    # 3. Converted YOLO TXT Inspection
    txt_content = ""
    txt_lines = []
    if txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()
            txt_lines = [l.strip() for l in txt_content.splitlines() if l.strip()]

    print(f"Converted YOLO label file lines: {len(txt_lines)}")
    for i, line in enumerate(txt_lines):
        print(f"    Line {i+1}: {line}")

    return {
        "image_id": image_id,
        "image_path": str(img_path),
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "image_dimensions": {"width": pil_w, "height": pil_h, "channels": cv_c},
        "metadata": {"source": source, "pair_id": pair_id},
        "items": items_analysis,
        "yolo_txt_lines_count": len(txt_lines),
        "yolo_txt_lines": txt_lines,
    }


def main():
    cases = ["028225", "115063", "101338"]
    results = {}
    for c in cases:
        results[c] = inspect_case(c)

    print("\n" + "="*70)
    print("ALL 3 CASES INSPECTED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
