"""
DeepFashion2 to YOLO Annotation Converter
==========================================
Converts DeepFashion2 JSON bounding-box annotations into normalized YOLO
format TXT files for object detection training.

Coordinate Convention:
----------------------
DeepFashion2 bounding box: [x1, y1, x2, y2]
  - (x1, y1): Top-left corner in absolute pixels
  - (x2, y2): Bottom-right corner in absolute pixels

YOLO format line:
  <class_id> <x_center> <y_center> <width> <height>
  - class_id: 0-indexed integer (0 to 12)
  - x_center = ((x1 + x2) / 2.0) / image_width
  - y_center = ((y1 + y2) / 2.0) / image_height
  - width    = (x2 - x1) / image_width
  - height   = (y2 - y1) / image_height
  - All normalized coordinates are in [0.0, 1.0]

Validation & Rejection Rules:
-----------------------------
- Rejects missing, non-integer, or unknown category IDs.
- Rejects missing, malformed, or non-numeric/non-finite coordinates.
- Rejects zero-area and negative-area boxes (e.g. x2 <= x1 or y2 <= y1).
- Rejects coordinates outside image boundaries (x1 < 0, y1 < 0, x2 > w, y2 > h).
- Never silently clips invalid boxes.
- Preserves valid garment instances when an image contains both valid and invalid items.
- Known degenerate box: train/annos/101338.json item2 [0, 0, 0, 0] is rejected as zero-area.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Add parent directory to sys.path so config can be imported directly
_CURRENT_DIR = Path(__file__).resolve().parent
_ML_ROOT = _CURRENT_DIR.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from config import (
    DF2_CATEGORIES,
    DF2_TO_YOLO_CLASS_MAP,
    YOLO_CLASS_NAMES,
    YOLO_TO_DF2_CLASS_MAP,
)


@dataclass
class ValidGarmentInstance:
    """Represents a validated and normalized garment instance."""
    item_key: str
    category_id: int
    category_name: str
    class_id: int
    raw_bbox: List[float]
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self, precision: int = 6) -> str:
        """Format as a single YOLO label text line."""
        fmt = f"{{:.{precision}f}}"
        xc_str = fmt.format(self.x_center)
        yc_str = fmt.format(self.y_center)
        w_str = fmt.format(self.width)
        h_str = fmt.format(self.height)
        return f"{self.class_id} {xc_str} {yc_str} {w_str} {h_str}\n"


@dataclass
class InvalidGarmentInstance:
    """Represents an invalid garment instance rejected during validation."""
    item_key: str
    reason: str
    category_id: Optional[Any] = None
    raw_bbox: Optional[Any] = None
    details: str = ""


@dataclass
class ConversionResult:
    """Detailed result of converting a single image annotation."""
    image_id: str
    anno_path: Path
    image_path: Path
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    valid_instances: List[ValidGarmentInstance] = field(default_factory=list)
    invalid_instances: List[InvalidGarmentInstance] = field(default_factory=list)
    status: str = "success"  # "success", "partial_success", "failed"
    error_message: Optional[str] = None
    output_label_path: Optional[Path] = None

    @property
    def valid_count(self) -> int:
        return len(self.valid_instances)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_instances)


class DeepFashion2ToYOLOConverter:
    """
    Reusable converter transforming DeepFashion2 JSON annotations into YOLO TXT format.
    """

    def __init__(
        self,
        category_map: Optional[Dict[int, int]] = None,
        category_names: Optional[Dict[int, str]] = None,
        coord_precision: int = 6,
    ):
        self.category_map = category_map or DF2_TO_YOLO_CLASS_MAP
        self.category_names = category_names or DF2_CATEGORIES
        self.coord_precision = coord_precision

    def validate_category(self, category_id: Any) -> Tuple[bool, Optional[int], str]:
        """
        Validate category ID and map to YOLO 0-based class ID.
        Returns: (is_valid, class_id, reason_if_invalid)
        """
        if category_id is None:
            return False, None, "missing_category_id"
        if isinstance(category_id, bool):
            return False, None, "invalid_category_id_type"
        if not isinstance(category_id, int):
            try:
                category_id = int(category_id)
            except (ValueError, TypeError):
                return False, None, "invalid_category_id_type"

        if category_id not in self.category_map:
            return False, None, f"unknown_category_id_{category_id}"

        class_id = self.category_map[category_id]
        return True, class_id, ""

    def validate_and_normalize_box(
        self,
        raw_bbox: Any,
        image_width: int,
        image_height: int,
    ) -> Tuple[bool, Optional[Tuple[float, float, float, float]], str, str]:
        """
        Validate bounding box coordinates and compute normalized YOLO coordinates.
        Never silently clips out-of-bounds coordinates.

        Returns:
            (is_valid, (x_center, y_center, width, height), rejection_reason, details)
        """
        # 1. Structure check
        if raw_bbox is None or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return False, None, "malformed_bounding_box", f"Expected 4 elements, got: {raw_bbox}"

        # 2. Coordinate numeric and finiteness checks
        coords: List[float] = []
        for idx, val in enumerate(raw_bbox):
            if val is None or isinstance(val, bool):
                return False, None, "non_numeric_coordinates", f"Element {idx} is not numeric: {val}"
            try:
                f_val = float(val)
            except (ValueError, TypeError):
                return False, None, "non_numeric_coordinates", f"Element {idx} is not numeric: {val}"

            if math.isnan(f_val) or math.isinf(f_val):
                return False, None, "non_finite_coordinates", f"Element {idx} is non-finite: {f_val}"
            coords.append(f_val)

        x1, y1, x2, y2 = coords

        # 3. Area check (positive area required: x2 > x1 and y2 > y1)
        if x2 <= x1 or y2 <= y1:
            details = f"[x1={x1}, y1={y1}, x2={x2}, y2={y2}] -> dx={x2-x1}, dy={y2-y1}"
            return False, None, "zero_or_negative_area", details

        # 4. Image bounds check (coordinates must not exceed [0, width] and [0, height])
        # Note: Never silently clip invalid coordinates.
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            details = (
                f"[x1={x1}, y1={y1}, x2={x2}, y2={y2}] outside bounds for "
                f"image size ({image_width}, {image_height})"
            )
            return False, None, "out_of_bounds_coordinates", details

        # 5. Compute normalized YOLO parameters
        x_center = ((x1 + x2) / 2.0) / float(image_width)
        y_center = ((y1 + y2) / 2.0) / float(image_height)
        norm_width = (x2 - x1) / float(image_width)
        norm_height = (y2 - y1) / float(image_height)

        # 6. Sanity check normalized values in [0.0, 1.0]
        for name, val in [("x_center", x_center), ("y_center", y_center), ("width", norm_width), ("height", norm_height)]:
            if not (0.0 <= val <= 1.0):
                return False, None, "normalization_out_of_range", f"{name}={val} outside [0, 1]"

        return True, (x_center, y_center, norm_width, norm_height), "", ""

    def get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """
        Read image dimensions (width, height) using Pillow.
        Raises ValueError or FileNotFoundError if unreadable.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: width={w}, height={h} for {image_path}")
        return w, h

    def convert_single_annotation(
        self,
        anno_path: Path | str,
        image_path: Path | str,
        output_label_path: Optional[Path | str] = None,
    ) -> ConversionResult:
        """
        Convert a single DeepFashion2 JSON file and its image into YOLO format.

        Args:
            anno_path: Path to the DeepFashion2 JSON annotation file.
            image_path: Path to the corresponding JPG image file.
            output_label_path: Optional destination Path for writing the YOLO TXT file.

        Returns:
            ConversionResult dataclass with full auditing detail.
        """
        anno_path = Path(anno_path)
        image_path = Path(image_path)
        image_id = anno_path.stem

        result = ConversionResult(
            image_id=image_id,
            anno_path=anno_path,
            image_path=image_path,
        )

        # 1. Read image dimensions
        try:
            w, h = self.get_image_dimensions(image_path)
            result.image_width = w
            result.image_height = h
        except Exception as e:
            result.status = "failed"
            result.error_message = f"missing_or_unreadable_image: {e}"
            return result

        # 2. Read JSON annotation
        if not anno_path.exists():
            result.status = "failed"
            result.error_message = f"annotation_file_not_found: {anno_path}"
            return result

        try:
            with open(anno_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            result.status = "failed"
            result.error_message = f"corrupt_or_invalid_json: {e}"
            return result

        if not isinstance(data, dict):
            result.status = "failed"
            result.error_message = f"invalid_json_root_type: expected dict, got {type(data).__name__}"
            return result

        # 3. Iterate through garment items
        item_keys = [k for k in data.keys() if k.startswith("item")]

        for k in item_keys:
            item_data = data[k]
            if not isinstance(item_data, dict):
                result.invalid_instances.append(
                    InvalidGarmentInstance(
                        item_key=k,
                        reason="item_not_a_dictionary",
                        details=f"Item value is {type(item_data).__name__}"
                    )
                )
                continue

            raw_cat_id = item_data.get("category_id")
            raw_bbox = item_data.get("bounding_box")

            # Validate category
            cat_valid, class_id, cat_reason = self.validate_category(raw_cat_id)
            if not cat_valid:
                result.invalid_instances.append(
                    InvalidGarmentInstance(
                        item_key=k,
                        reason=cat_reason,
                        category_id=raw_cat_id,
                        raw_bbox=raw_bbox,
                        details=f"Failed category validation: {cat_reason}"
                    )
                )
                continue

            # Validate bounding box
            box_valid, norm_box, box_reason, box_details = self.validate_and_normalize_box(
                raw_bbox=raw_bbox,
                image_width=w,
                image_height=h,
            )
            if not box_valid:
                result.invalid_instances.append(
                    InvalidGarmentInstance(
                        item_key=k,
                        reason=box_reason,
                        category_id=raw_cat_id,
                        raw_bbox=raw_bbox,
                        details=box_details,
                    )
                )
                continue

            xc, yc, nw, nh = norm_box
            cat_name = self.category_names.get(raw_cat_id, "unknown")

            result.valid_instances.append(
                ValidGarmentInstance(
                    item_key=k,
                    category_id=raw_cat_id,
                    category_name=cat_name,
                    class_id=class_id,
                    raw_bbox=list(raw_bbox),
                    x_center=xc,
                    y_center=yc,
                    width=nw,
                    height=nh,
                )
            )

        # Determine overall status
        if result.invalid_count > 0 and result.valid_count > 0:
            result.status = "partial_success"
        elif result.invalid_count > 0 and result.valid_count == 0:
            result.status = "failed"
        else:
            result.status = "success"

        # 4. Write YOLO label file if output path is requested
        if output_label_path is not None:
            out_p = Path(output_label_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                for inst in result.valid_instances:
                    f.write(inst.to_yolo_line(precision=self.coord_precision))
            result.output_label_path = out_p

        return result

    def convert_sample_set(
        self,
        sample_ids: List[str],
        annos_dir: Path | str,
        images_dir: Path | str,
        output_labels_dir: Optional[Path | str] = None,
        output_images_dir: Optional[Path | str] = None,
        copy_images: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert a designated list of sample images and produce a full summary report.
        Optionally copies sample images to an output images/ directory for standalone inspection.
        """
        annos_dir = Path(annos_dir)
        images_dir = Path(images_dir)
        if output_labels_dir:
            output_labels_dir = Path(output_labels_dir)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
        if output_images_dir and copy_images:
            output_images_dir = Path(output_images_dir)
            output_images_dir.mkdir(parents=True, exist_ok=True)

        results: List[ConversionResult] = []
        category_counts: Dict[str, int] = {name: 0 for name in YOLO_CLASS_NAMES}
        category_id_counts: Dict[int, int] = {cid: 0 for cid in range(1, 14)}
        class_id_counts: Dict[int, int] = {cls_id: 0 for cls_id in range(13)}
        invalid_reason_counts: Dict[str, int] = {}
        multi_garment_counts: Dict[str, int] = {
            "0_instances": 0,
            "1_instance": 0,
            "2_instances": 0,
            "3_or_more_instances": 0,
        }
        known_degenerate_audit: Dict[str, Any] = {}

        for image_id in sample_ids:
            anno_file = annos_dir / f"{image_id}.json"
            image_file = images_dir / f"{image_id}.jpg"
            out_label = (output_labels_dir / f"{image_id}.txt") if output_labels_dir else None

            conv_res = self.convert_single_annotation(
                anno_path=anno_file,
                image_path=image_file,
                output_label_path=out_label,
            )
            results.append(conv_res)

            # Copy image if requested
            if output_images_dir and copy_images and image_file.exists():
                dest_img = output_images_dir / f"{image_id}.jpg"
                if not dest_img.exists():
                    shutil.copy2(image_file, dest_img)

            # Tally counts
            n_valid = conv_res.valid_count
            if n_valid == 0:
                multi_garment_counts["0_instances"] += 1
            elif n_valid == 1:
                multi_garment_counts["1_instance"] += 1
            elif n_valid == 2:
                multi_garment_counts["2_instances"] += 1
            else:
                multi_garment_counts["3_or_more_instances"] += 1

            for inst in conv_res.valid_instances:
                category_counts[inst.category_name] += 1
                category_id_counts[inst.category_id] += 1
                class_id_counts[inst.class_id] += 1

            for inv in conv_res.invalid_instances:
                invalid_reason_counts[inv.reason] = invalid_reason_counts.get(inv.reason, 0) + 1

            # Special audit for 101338
            if image_id == "101338":
                known_degenerate_audit = {
                    "image_id": "101338",
                    "status": conv_res.status,
                    "valid_instances_count": conv_res.valid_count,
                    "invalid_instances_count": conv_res.invalid_count,
                    "valid_instances": [
                        {
                            "item_key": v.item_key,
                            "category_id": v.category_id,
                            "category_name": v.category_name,
                            "class_id": v.class_id,
                            "raw_bbox": v.raw_bbox,
                            "yolo_line": v.to_yolo_line().strip(),
                        }
                        for v in conv_res.valid_instances
                    ],
                    "invalid_instances": [
                        {
                            "item_key": inv.item_key,
                            "reason": inv.reason,
                            "raw_bbox": inv.raw_bbox,
                            "category_id": inv.category_id,
                            "details": inv.details,
                        }
                        for inv in conv_res.invalid_instances
                    ],
                    "degenerate_box_correctly_rejected": any(
                        inv.reason == "zero_or_negative_area" and inv.raw_bbox == [0, 0, 0, 0]
                        for inv in conv_res.invalid_instances
                    ),
                    "valid_item_preserved": any(
                        v.category_id == 9 and v.raw_bbox == [307, 437, 432, 597]
                        for v in conv_res.valid_instances
                    ),
                }

        total_valid = sum(len(r.valid_instances) for r in results)
        total_invalid = sum(len(r.invalid_instances) for r in results)

        summary = {
            "sample_image_count": len(sample_ids),
            "total_instances_inspected": total_valid + total_invalid,
            "valid_garment_instance_count": total_valid,
            "invalid_instance_count_by_reason": invalid_reason_counts,
            "total_invalid_instances": total_invalid,
            "category_counts_by_name": category_counts,
            "category_counts_by_df2_id": category_id_counts,
            "category_counts_by_yolo_class_id": class_id_counts,
            "multi_garment_image_counts": multi_garment_counts,
            "known_degenerate_audit": known_degenerate_audit,
            "per_image_results": [
                {
                    "image_id": r.image_id,
                    "status": r.status,
                    "dimensions": (r.image_width, r.image_height),
                    "valid_count": r.valid_count,
                    "invalid_count": r.invalid_count,
                    "yolo_lines": [v.to_yolo_line().strip() for v in r.valid_instances],
                    "invalid_details": [
                        {"item": inv.item_key, "reason": inv.reason, "bbox": inv.raw_bbox}
                        for inv in r.invalid_instances
                    ],
                }
                for r in results
            ],
        }

        return summary
