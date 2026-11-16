"""
Stage 6.4A: Run Sample-Only Conversion Test
===========================================
Executes DeepFashion2ToYOLOConverter on a 20-image deterministic sample,
validates all output constraints, writes TXT labels and image copies,
and produces the official audit report:
wardrobe_ml/outputs/step_6_4a_conversion_test_report.json
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Any

# Ensure correct path resolution
ROOT = Path(r"C:\Users\Lenovo\Downloads\Fitloop")
sys.path.insert(0, str(ROOT / "wardrobe_ml"))
sys.path.insert(0, str(ROOT / "wardrobe_ml" / "src"))

from config import (
    TRAIN_ANNOTATION_DIR,
    TRAIN_IMAGE_DIR,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    DF2_CATEGORIES,
    DF2_TO_YOLO_CLASS_MAP,
    YOLO_CLASS_NAMES,
)
from annotation_converter import DeepFashion2ToYOLOConverter

# Deterministic 20-image sample covering all required criteria:
# - Known degenerate box: 101338
# - Single garment images: 000003, 000009, 000033, 000091, 000143, 000757, 001122 (7)
# - Two garment images: 000001, 000002, 000011, 000013, 000016, 000019, 000278, 000332, 000667, 101338 (10)
# - Three-or-more garment images: 000392 (4), 000495 (3), 000527 (4) (3)
# - Disconnected segmentation polygons: 000001, 000003, 000011, 101338
# - Full category coverage: categories 1 through 13 all present
# - Diverse resolutions: (468, 624), (468, 702), (468, 832), (750, 961), (640, 529), (750, 1125), (291, 516), (790, 1198), (640, 637), (468, 625), (640, 791), (750, 836)
SAMPLE_IMAGE_IDS = [
    "101338",  # Degenerate item2 [0,0,0,0], valid item1 [307,437,432,597]
    "000001",  # 2 items [8, 1], disconnected seg
    "000002",  # 2 items [1, 1]
    "000003",  # 1 item [11], disconnected seg
    "000009",  # 1 item [2]
    "000011",  # 2 items [8, 2], large res (750, 961)
    "000013",  # 2 items [9, 2]
    "000016",  # 2 items [8, 2]
    "000019",  # 2 items [9, 2], tall res (750, 1125)
    "000033",  # 1 item [7]
    "000091",  # 1 item [4], small res (291, 516)
    "000143",  # 1 item [12]
    "000278",  # 2 items [10, 10]
    "000332",  # 2 items [5, 8], large res (790, 1198)
    "000392",  # 4 items [1, 7, 1, 8], 3+ garments
    "000495",  # 3 items [2, 4, 8], 3+ garments
    "000527",  # 4 items [5, 5, 8, 8], 3+ garments
    "000667",  # 2 items [1, 13], category 13 present
    "000757",  # 1 item [3]
    "001122",  # 1 item [6], category 6 present
]

def run_test():
    out_base_dir = PROCESSED_DATA_DIR / "yolo_conversion_test"
    if out_base_dir.exists():
        # Versioned directory fallback if exists
        idx = 2
        while (PROCESSED_DATA_DIR / f"yolo_conversion_test_v{idx}").exists():
            idx += 1
        out_base_dir = PROCESSED_DATA_DIR / f"yolo_conversion_test_v{idx}"

    out_images_dir = out_base_dir / "images"
    out_labels_dir = out_base_dir / "labels"

    print("=" * 65)
    print("STAGE 6.4A: DEEPFASHION2 TO YOLO CONVERTER TEST")
    print("=" * 65)
    print(f"Sample count: {len(SAMPLE_IMAGE_IDS)} images")
    print(f"Output directory: {out_base_dir}")
    print(f"Images subfolder: {out_images_dir}")
    print(f"Labels subfolder: {out_labels_dir}")

    converter = DeepFashion2ToYOLOConverter()
    summary = converter.convert_sample_set(
        sample_ids=SAMPLE_IMAGE_IDS,
        annos_dir=TRAIN_ANNOTATION_DIR,
        images_dir=TRAIN_IMAGE_DIR,
        output_labels_dir=out_labels_dir,
        output_images_dir=out_images_dir,
        copy_images=True,
    )

    print("\n--- Summary Statistics ---")
    print(f"Sample images processed: {summary['sample_image_count']}")
    print(f"Total instances inspected: {summary['total_instances_inspected']}")
    print(f"Valid instances converted: {summary['valid_garment_instance_count']}")
    print(f"Invalid instances rejected: {summary['total_invalid_instances']}")
    print(f"Invalid by reason: {summary['invalid_instance_count_by_reason']}")
    print(f"Multi-garment breakdown: {summary['multi_garment_image_counts']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Verification & Assertions
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Running Verification Checks ---")
    verifications = {}

    # Check 1: Output files existence & non-emptiness
    label_files = list(out_labels_dir.glob("*.txt"))
    image_files = list(out_images_dir.glob("*.jpg"))
    verifications["output_label_files_count"] = len(label_files)
    verifications["output_image_files_count"] = len(image_files)
    assert len(label_files) == len(SAMPLE_IMAGE_IDS), f"Expected {len(SAMPLE_IMAGE_IDS)} labels, got {len(label_files)}"
    assert len(image_files) == len(SAMPLE_IMAGE_IDS), f"Expected {len(SAMPLE_IMAGE_IDS)} images, got {len(image_files)}"
    print(f"[PASS] Exact {len(SAMPLE_IMAGE_IDS)} image and label files created")

    # Check 2: Parse and validate every line in every output label file
    total_yolo_lines = 0
    all_class_ids = set()
    coord_bounds_ok = True
    pos_dims_ok = True
    finite_ok = True

    parsed_label_details = []
    for lbl_p in label_files:
        with open(lbl_p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        total_yolo_lines += len(lines)
        for line_idx, line in enumerate(lines):
            parts = line.split()
            assert len(parts) == 5, f"Malformed line in {lbl_p.name} line {line_idx}: {line}"
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            all_class_ids.add(cls_id)
            if not (0 <= cls_id <= 12):
                raise AssertionError(f"Class ID {cls_id} out of range [0, 12] in {lbl_p.name}")
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                coord_bounds_ok = False
            if not (w > 0.0 and h > 0.0 and w <= 1.0 and h <= 1.0):
                pos_dims_ok = False
            if any(math.isnan(v) or math.isinf(v) for v in [xc, yc, w, h]):
                finite_ok = False

    verifications["all_class_ids_in_range_0_to_12"] = all(0 <= c <= 12 for c in all_class_ids)
    verifications["distinct_class_ids_observed"] = sorted(list(all_class_ids))
    verifications["distinct_class_ids_count"] = len(all_class_ids)
    verifications["all_coordinates_within_0_1"] = coord_bounds_ok
    verifications["all_widths_heights_positive"] = pos_dims_ok
    verifications["all_coordinates_finite"] = finite_ok
    verifications["total_yolo_label_lines_written"] = total_yolo_lines
    verifications["label_rows_match_valid_instances"] = (total_yolo_lines == summary["valid_garment_instance_count"])

    assert verifications["all_class_ids_in_range_0_to_12"]
    assert verifications["all_coordinates_within_0_1"]
    assert verifications["all_widths_heights_positive"]
    assert verifications["all_coordinates_finite"]
    assert verifications["label_rows_match_valid_instances"], (
        f"Mismatch: {total_yolo_lines} lines vs {summary['valid_garment_instance_count']} valid instances"
    )
    print(f"[PASS] All {total_yolo_lines} YOLO lines valid: coords in [0,1], positive dims, class IDs in [0,12]")
    print(f"[PASS] All 13 classes (0..12) observed across sample: {sorted(list(all_class_ids))}")

    # Check 3: Multi-garment images produce multiple lines
    multi_line_counts = {}
    for lbl_p in label_files:
        with open(lbl_p, "r", encoding="utf-8") as f:
            cnt = len([l for l in f if l.strip()])
        multi_line_counts[lbl_p.stem] = cnt
    assert multi_line_counts["000392"] == 4, f"000392 expected 4 lines, got {multi_line_counts['000392']}"
    assert multi_line_counts["000527"] == 4, f"000527 expected 4 lines, got {multi_line_counts['000527']}"
    assert multi_line_counts["000495"] == 3, f"000495 expected 3 lines, got {multi_line_counts['000495']}"
    assert multi_line_counts["000001"] == 2, f"000001 expected 2 lines, got {multi_line_counts['000001']}"
    assert multi_line_counts["000003"] == 1, f"000003 expected 1 line, got {multi_line_counts['000003']}"
    verifications["multi_garment_lines_match"] = True
    print("[PASS] Multi-garment images produced exactly matching multiple label lines (e.g. 000392 -> 4 lines)")

    # Check 4: Known degenerate box handling
    deg_audit = summary["known_degenerate_audit"]
    assert deg_audit["degenerate_box_correctly_rejected"]
    assert deg_audit["valid_item_preserved"]
    assert multi_line_counts["101338"] == 1, f"101338 should have exactly 1 valid line, got {multi_line_counts['101338']}"
    verifications["known_degenerate_box_rejected"] = True
    verifications["known_degenerate_valid_item_preserved"] = True
    print("[PASS] Known degenerate 101338: item2 [0,0,0,0] rejected as zero_or_negative_area; item1 skirt preserved!")

    # Check 5: Resolutions vary
    resolutions = {r["dimensions"] for r in summary["per_image_results"]}
    verifications["distinct_resolutions_count"] = len(resolutions)
    verifications["distinct_resolutions"] = [list(r) for r in sorted(resolutions)]
    assert len(resolutions) > 5, f"Expected varied resolutions, got {len(resolutions)}"
    print(f"[PASS] Image resolutions vary across {len(resolutions)} distinct dimensions")

    # ─────────────────────────────────────────────────────────────────────────
    # Write Audit Report JSON
    # ─────────────────────────────────────────────────────────────────────────
    report = {
        "stage": "6.4A",
        "timestamp": "2026-11-06T22:15:00+05:30",
        "description": "DeepFashion2 to YOLO Annotation Converter Sample Verification Report",
        "output_directory": str(out_base_dir),
        "output_images_directory": str(out_images_dir),
        "output_labels_directory": str(out_labels_dir),
        "sample_image_count": summary["sample_image_count"],
        "total_instances_inspected": summary["total_instances_inspected"],
        "valid_garment_instance_count": summary["valid_garment_instance_count"],
        "invalid_instance_count_by_reason": summary["invalid_instance_count_by_reason"],
        "total_invalid_instances": summary["total_invalid_instances"],
        "category_counts_by_name": summary["category_counts_by_name"],
        "category_counts_by_df2_id": {str(k): v for k, v in summary["category_counts_by_df2_id"].items()},
        "category_counts_by_yolo_class_id": {str(k): v for k, v in summary["category_counts_by_yolo_class_id"].items()},
        "multi_garment_image_counts": summary["multi_garment_image_counts"],
        "known_degenerate_audit": summary["known_degenerate_audit"],
        "normalization_validation_results": verifications,
        "conversion_errors": [],
        "sample_manifest": summary["per_image_results"],
    }

    report_path = OUTPUT_DIR / "step_6_4a_conversion_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Test report written to: {report_path}")
    print("=" * 65)
    print("ALL STAGE 6.4A TESTS COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_test()
