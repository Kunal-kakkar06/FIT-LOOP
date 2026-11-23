"""
Stage 6.4C.1: Generate Forensic Audit JSON Report
=================================================
Writes wardrobe_ml/outputs/step_6_4c1_bbox_forensic_report.json
"""

import json
from pathlib import Path

REPORT_PATH = Path("C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_4c1_bbox_forensic_report.json")

report = {
    "stage": "6.4C.1",
    "timestamp": "2026-11-07T09:25:00+05:30",
    "title": "Out-of-Bounds & Degenerate Annotation Forensic Verification Report",
    "status": "CONCLUDED_ALL_REJECTIONS_VERIFIED",
    "summary": {
        "total_rejected_annotations": 3,
        "conclusively_verified_count": 3,
        "unresolved_count": 0,
        "rejection_justification_valid": True,
        "accounting_reconciliation_exact": True,
        "phase1_audited_train_instances": 312186,
        "stage_6_4c_inspected_instances": 312186,
        "stage_6_4c_valid_instances": 312183,
        "stage_6_4c_rejected_instances": 3,
    },
    "cases": {
        "case_028225": {
            "image_id": "028225",
            "image_path": "C:/Users/Lenovo/Downloads/DeepFashion2/train/image/028225.jpg",
            "json_path": "C:/Users/Lenovo/Downloads/DeepFashion2/train/annos/028225.json",
            "label_path": "C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/data/processed/deepfashion2_yolo/labels/train/028225.txt",
            "diagnostic_image": "C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_4c1_bbox_forensic/forensic_028225_out_of_bounds.png",
            "actual_image_dimensions": {
                "width": 468,
                "height": 624,
                "channels": 3,
                "verified_by": ["PIL", "OpenCV"],
            },
            "source_metadata": {
                "source": "user",
                "pair_id": 2234,
                "total_items_in_json": 2,
            },
            "investigated_item": {
                "item_key": "item2",
                "category_id": 8,
                "category_name": "trousers",
                "style": 0,
                "raw_bounding_box": [1, 198, 422, 626],
                "is_numeric": True,
                "is_finite": True,
                "dx": 421,
                "dy": 428,
                "area": 180188,
                "has_positive_area": True,
                "bounds_check": {
                    "x1_in_bounds": True,
                    "y1_in_bounds": True,
                    "x2_in_bounds": True,
                    "y2_in_bounds": False,
                    "violation": "y2=626 > height=624 (exceeds image boundary by 2 px)",
                },
                "axis_swap_hypothesis": {
                    "tested_swapped_dims_W624_H468": "INVALID (y2=626 > 468)",
                    "tested_swapped_coords_y1_x1_y2_x2": "INVALID (x2=626 > 468)",
                    "conclusion": "Standard DeepFashion2 [x1, y1, x2, y2] convention used; annotator drew box 2px past bottom image edge",
                },
                "segmentation_evidence": {
                    "seg_x_range": [3, 412],
                    "seg_y_range": [202, 625],
                    "max_seg_y": 625,
                    "evidence_note": "Segmentation polygon also reaches y=625 (>624), confirming human annotator drew past bottom edge",
                },
                "landmarks_evidence": {
                    "lm_x_range": [21, 412],
                    "lm_y_range": [202, 403],
                },
                "forensic_verdict": "CONCLUDED_CORRECTLY_REJECTED",
                "rejection_reason": "out_of_bounds_coordinates",
            },
            "co_occurring_items": {
                "item1": {
                    "category_id": 1,
                    "category_name": "short sleeve top",
                    "style": 1,
                    "raw_bounding_box": [40, 0, 399, 381],
                    "bounds_check": "FULLY_IN_BOUNDS",
                    "preserved_in_yolo_label": True,
                    "yolo_line": "0 0.469017 0.305288 0.767094 0.610577",
                }
            },
            "label_file_verification": {
                "file_exists": True,
                "lines_count": 1,
                "content": ["0 0.469017 0.305288 0.767094 0.610577"],
                "accounting_status": "EXACT_MATCH_PRESERVED_ITEM1_REJECTED_ITEM2",
            },
        },
        "case_115063": {
            "image_id": "115063",
            "image_path": "C:/Users/Lenovo/Downloads/DeepFashion2/train/image/115063.jpg",
            "json_path": "C:/Users/Lenovo/Downloads/DeepFashion2/train/annos/115063.json",
            "label_path": "C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/data/processed/deepfashion2_yolo/labels/train/115063.txt",
            "diagnostic_image": "C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_4c1_bbox_forensic/forensic_115063_out_of_bounds.png",
            "actual_image_dimensions": {
                "width": 468,
                "height": 702,
                "channels": 3,
                "verified_by": ["PIL", "OpenCV"],
            },
            "source_metadata": {
                "source": "shop",
                "pair_id": 9101,
                "total_items_in_json": 1,
            },
            "investigated_item": {
                "item_key": "item1",
                "category_id": 1,
                "category_name": "short sleeve top",
                "style": 1,
                "raw_bounding_box": [15, 195, 466, 829],
                "is_numeric": True,
                "is_finite": True,
                "dx": 451,
                "dy": 634,
                "area": 285934,
                "has_positive_area": True,
                "bounds_check": {
                    "x1_in_bounds": True,
                    "y1_in_bounds": True,
                    "x2_in_bounds": True,
                    "y2_in_bounds": False,
                    "violation": "y2=829 > height=702 (exceeds image boundary by 127 px / 18.09%)",
                },
                "axis_swap_hypothesis": {
                    "tested_swapped_dims_W702_H468": "INVALID (y2=829 > 468)",
                    "tested_swapped_coords_y1_x1_y2_x2": "INVALID (x2=829 > 468)",
                    "conclusion": "Standard DeepFashion2 [x1, y1, x2, y2] convention used; upstream annotator entered erroneous bottom coordinate",
                },
                "segmentation_evidence": {
                    "seg_x_range": [26, 377],
                    "seg_y_range": [226, 641],
                    "max_seg_y": 641,
                    "evidence_note": "True garment hem ends at y=641 (well within H=702); reported bbox extends 188px below garment hem and 127px into exterior void",
                },
                "landmarks_evidence": {
                    "lm_x_range": [26, 377],
                    "lm_y_range": [226, 641],
                    "max_lm_y": 641,
                },
                "forensic_verdict": "CONCLUDED_CORRECTLY_REJECTED",
                "rejection_reason": "out_of_bounds_coordinates",
            },
            "co_occurring_items": {},
            "label_file_verification": {
                "file_exists": True,
                "lines_count": 0,
                "file_size_bytes": 0,
                "content": [],
                "accounting_status": "EXACT_MATCH_VALID_EMPTY_NEGATIVE_IMAGE",
                "explanation": "Because item1 was the only item in 115063 and was safely rejected, the YOLO label file is 0 bytes (empty), which is the standard representation for a background image without bounding boxes.",
            },
        },
        "case_101338": {
            "image_id": "101338",
            "image_path": "C:/Users/Lenovo/Downloads/DeepFashion2/train/image/101338.jpg",
            "json_path": "C:/Users/Lenovo/Downloads/DeepFashion2/train/annos/101338.json",
            "label_path": "C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/data/processed/deepfashion2_yolo/labels/train/101338.txt",
            "diagnostic_image": "C:/Users/Lenovo/Downloads/Fitloop/wardrobe_ml/outputs/step_6_4c1_bbox_forensic/forensic_101338_zero_area.png",
            "actual_image_dimensions": {
                "width": 750,
                "height": 836,
                "channels": 3,
                "verified_by": ["PIL", "OpenCV"],
            },
            "source_metadata": {
                "source": "shop",
                "pair_id": 8028,
                "total_items_in_json": 2,
            },
            "investigated_item": {
                "item_key": "item2",
                "category_id": 1,
                "category_name": "short sleeve top",
                "style": 0,
                "raw_bounding_box": [0, 0, 0, 0],
                "is_numeric": True,
                "is_finite": True,
                "dx": 0,
                "dy": 0,
                "area": 0,
                "has_positive_area": False,
                "bounds_check": {
                    "x1_in_bounds": True,
                    "y1_in_bounds": True,
                    "x2_in_bounds": True,
                    "y2_in_bounds": True,
                    "violation": "dx=0, dy=0, area=0 (degenerate point at origin [0,0,0,0])",
                },
                "axis_swap_hypothesis": {
                    "conclusion": "Degenerate point at origin; axis swap irrelevant",
                },
                "segmentation_evidence": {
                    "seg_x_range": [338, 446],
                    "seg_y_range": [332, 448],
                    "evidence_note": "Legitimate segmentation points and landmarks exist on the model's torso, proving annotator captured the mask but failed to populate bounding box",
                },
                "landmarks_evidence": {
                    "lm_x_range": [338, 446],
                    "lm_y_range": [327, 428],
                },
                "forensic_verdict": "CONCLUDED_CORRECTLY_REJECTED",
                "rejection_reason": "zero_or_negative_area",
            },
            "co_occurring_items": {
                "item1": {
                    "category_id": 9,
                    "category_name": "skirt",
                    "style": 2,
                    "raw_bounding_box": [307, 437, 432, 597],
                    "bounds_check": "FULLY_IN_BOUNDS",
                    "preserved_in_yolo_label": True,
                    "yolo_line": "8 0.492667 0.618421 0.166667 0.191388",
                }
            },
            "label_file_verification": {
                "file_exists": True,
                "lines_count": 1,
                "content": ["8 0.492667 0.618421 0.166667 0.191388"],
                "accounting_status": "EXACT_MATCH_PRESERVED_ITEM1_REJECTED_ITEM2",
            },
        },
    },
    "conversion_accounting_audit": {
        "phase1_audited_total_instances": 312186,
        "stage_6_4c_inspected_instances": 312186,
        "valid_converted_instances": 312183,
        "rejected_instances_count": 3,
        "rejected_instances_list": [
            {"image_id": "028225", "item_key": "item2", "reason": "out_of_bounds_coordinates", "raw_bbox": [1, 198, 422, 626]},
            {"image_id": "101338", "item_key": "item2", "reason": "zero_or_negative_area", "raw_bbox": [0, 0, 0, 0]},
            {"image_id": "115063", "item_key": "item1", "reason": "out_of_bounds_coordinates", "raw_bbox": [15, 195, 466, 829]},
        ],
        "empty_label_files_count": 1,
        "empty_label_files_list": ["115063.txt"],
        "single_garment_files_count": 76108,
        "multi_garment_files_count": 115852,
        "total_train_label_files": 191961,
        "total_val_label_files": 32153,
        "total_label_files": 224114,
        "accounting_equation": "312186 (audited) == 312183 (valid) + 3 (rejected)",
        "reconciliation_verdict": "EXACT_MATCH_AUDIT_PASSED",
    },
    "safety_and_preservation_confirmations": {
        "source_dataset_modified": False,
        "source_annotations_modified": False,
        "converter_modified": False,
        "labels_regenerated": False,
        "empty_files_deleted": False,
        "models_trained": False,
        "splits_generated": False,
    },
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Report written successfully to: {REPORT_PATH}")
