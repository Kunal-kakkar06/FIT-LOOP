"""
FitLoop Real-World Wardrobe Evaluation Pipeline
================================================
Automated visual & metric diagnostic pipeline (Roadmap Section 12):
1. Ingests real-world fashion photos across 3 capture contexts:
   - closet/
   - flat_lay/
   - single_garment/
2. Executes inference with fine-tuned YOLO26n (best.pt) on CPU:
   - Configurable conf and iou thresholds (default: conf=0.25, iou=0.45, agnostic_nms=True).
   - Supports low-confidence diagnostic mode (e.g. conf=0.15, iou=0.50).
3. Center-Weighted Color Extraction:
   - Insets 20% on each bounding box edge (inner 60% core) to eliminate background closet/bedsheet bleed.
   - Extracts dominant & secondary colors using GarmentColorExtractor (CIE LAB + K-Means).
4. Generates visual annotated inspection sheets with bounding boxes, core crop boxes, labels, and color swatches.
5. Computes diagnostic metrics (detection counts, confidence averages, low-confidence <0.45 warnings, zero-detection lists).
6. Exports structured reports to wardrobe_ml/outputs/real_world_eval/ (eval_report.txt, eval_summary.json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from color_extractor import GarmentColorExtractor


# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EVAL_FOLDERS = ["closet", "flat_lay", "single_garment"]


class RealWorldEvaluator:
    """
    Coordinates model loading, image batch processing, center-weighted color extraction,
    visual annotation, and diagnostic metrics generation for real-world wardrobe images.
    """

    def __init__(
        self,
        model_path: Path,
        data_dir: Path,
        output_dir: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        agnostic_nms: bool = True,
        core_trim: float = 0.20,
        report_name: str = "eval_report.txt",
        json_name: str = "eval_summary.json",
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.agnostic_nms = agnostic_nms
        self.core_trim = core_trim
        self.report_name = report_name
        self.json_name = json_name
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"[INIT] Loading YOLO model from {self.model_path} on {self.device}...")
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        print(f"[INIT] Model loaded successfully: {len(self.model.names)} classes detected.")

        self.color_extractor = GarmentColorExtractor()

        # Ensure output directories exist
        self.annotated_dir = self.output_dir / "annotated"
        for folder in EVAL_FOLDERS:
            (self.annotated_dir / folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_inner_core_bbox(
        bbox: List[int],
        trim_ratio: float = 0.20,
        img_w: int = 10000,
        img_h: int = 10000,
    ) -> List[int]:
        """
        Compute the inner core region of a bounding box by trimming `trim_ratio`
        (default 20% from each edge), resulting in the central 60% core.
        Eliminates background closet/bedsheet bleed.
        """
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        inset_x = int(round(bw * trim_ratio))
        inset_y = int(round(bh * trim_ratio))

        # Clamp insets so at least 2 pixels remain
        max_inset_x = max(0, (bw - 2) // 2)
        max_inset_y = max(0, (bh - 2) // 2)
        actual_inset_x = min(inset_x, max_inset_x)
        actual_inset_y = min(inset_y, max_inset_y)

        cx1 = max(0, min(img_w - 1, x1 + actual_inset_x))
        cy1 = max(0, min(img_h - 1, y1 + actual_inset_y))
        cx2 = max(cx1 + 1, min(img_w, x2 - actual_inset_x))
        cy2 = max(cy1 + 1, min(img_h, y2 - actual_inset_y))

        return [cx1, cy1, cx2, cy2]

    def find_images_in_folder(self, folder_path: Path) -> List[Path]:
        """Collect all supported image files in a directory."""
        if not folder_path.exists():
            return []
        images = [
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(images)

    def draw_annotation(
        self,
        image_bgr: np.ndarray,
        bbox: List[int],
        core_bbox: Optional[List[int]],
        category: str,
        confidence: float,
        color_name: str,
        color_hex: str,
    ) -> np.ndarray:
        """
        Render bounding box, inner core box, label badge, and a visual color swatch onto the image.
        """
        img = image_bgr.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox

        # Parse color_hex to BGR for border/swatch display
        try:
            r, g, b = GarmentColorExtractor.hex_to_rgb(color_hex)
            border_bgr = (b, g, r)
        except Exception:
            border_bgr = (0, 255, 0)

        # Draw main bounding box (solid)
        cv2.rectangle(img, (x1, y1), (x2, y2), border_bgr, 2)

        # Draw inner core box (dotted/thin white box showing the 60% sample region)
        if core_bbox is not None:
            cx1, cy1, cx2, cy2 = core_bbox
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (240, 240, 240), 1)

        # Build label text
        conf_str = f"{confidence * 100:.1f}%"
        label_text = f"{category} {conf_str} | {color_name} ({color_hex})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Label badge positioning
        badge_y1 = max(0, y1 - text_h - 10)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(w, x1 + text_w + 24)

        # Draw filled badge background (dark charcoal)
        cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), (30, 30, 30), -1)
        cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), border_bgr, 1)

        # Draw small color swatch square in the badge
        swatch_x1 = badge_x1 + 4
        swatch_y1 = badge_y1 + 4
        swatch_x2 = swatch_x1 + text_h
        swatch_y2 = swatch_y1 + text_h
        cv2.rectangle(img, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2), border_bgr, -1)
        cv2.rectangle(img, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2), (255, 255, 255), 1)

        # Draw text in white
        text_origin = (swatch_x2 + 6, badge_y2 - baseline - 2)
        cv2.putText(img, label_text, text_origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return img

    def evaluate_image(self, image_path: Path) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run detection and center-weighted color extraction on a single image.
        Returns list of detection records and the annotated image.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image at {image_path}")

        h, w = img_bgr.shape[:2]

        results = self.model.predict(
            img_bgr,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        annotated_img = img_bgr.copy()

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()
                x1 = max(0, min(w - 1, int(round(coords[0]))))
                y1 = max(0, min(h - 1, int(round(coords[1]))))
                x2 = max(x1 + 1, min(w, int(round(coords[2]))))
                y2 = max(y1 + 1, min(h, int(round(coords[3]))))
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf_val = float(box.conf[0].item())

                # Compute inner 60% core region (trimming 20% from each edge)
                core_bbox = self.get_inner_core_bbox([x1, y1, x2, y2], trim_ratio=self.core_trim, img_w=w, img_h=h)

                # Center-weighted color extraction on inner core
                try:
                    color_res = self.color_extractor.extract(img_bgr, bbox=core_bbox)
                    dom_col = color_res["dominant_color"]
                    color_name = dom_col["name"]
                    color_hex = dom_col["hex"]
                    sec_col = color_res.get("secondary_color")
                    sec_name = sec_col["name"] if sec_col else None
                    sec_hex = sec_col["hex"] if sec_col else None
                except Exception:
                    color_name = "Unknown"
                    color_hex = "#808080"
                    sec_name = None
                    sec_hex = None

                det_record = {
                    "bbox": [x1, y1, x2, y2],
                    "core_bbox": core_bbox,
                    "category": cls_name,
                    "confidence": round(conf_val, 4),
                    "dominant_color": {
                        "name": color_name,
                        "hex": color_hex,
                    },
                    "secondary_color": {
                        "name": sec_name,
                        "hex": sec_hex,
                    } if sec_name else None,
                    "color_name": color_name,
                    "color_hex": color_hex,
                }
                detections.append(det_record)

                annotated_img = self.draw_annotation(
                    annotated_img,
                    bbox=[x1, y1, x2, y2],
                    core_bbox=core_bbox,
                    category=cls_name,
                    confidence=conf_val,
                    color_name=color_name,
                    color_hex=color_hex,
                )

        return detections, annotated_img

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Execute evaluation across all 3 categories (closet, flat_lay, single_garment).
        """
        start_time = time.perf_counter()
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        metrics: Dict[str, Any] = {
            "timestamp": timestamp_str,
            "model_path": str(self.model_path),
            "parameters": {
                "conf": self.conf_threshold,
                "iou": self.iou_threshold,
                "agnostic_nms": self.agnostic_nms,
                "core_trim": self.core_trim,
                "device": self.device,
            },
            "overall": {
                "total_images": 0,
                "total_detections": 0,
                "avg_confidence": 0.0,
                "low_confidence_count": 0,
                "zero_detection_images": [],
                "elapsed_seconds": 0.0,
            },
            "per_folder": {},
        }

        all_confidences: List[float] = []

        for folder_name in EVAL_FOLDERS:
            folder_path = self.data_dir / folder_name
            target_annotated_dir = self.annotated_dir / folder_name
            images = self.find_images_in_folder(folder_path)

            folder_stats: Dict[str, Any] = {
                "images_count": len(images),
                "detections_count": 0,
                "avg_confidence": 0.0,
                "low_confidence_count": 0,
                "zero_detection_images": [],
                "category_distribution": {},
                "images": [],
            }

            folder_confs: List[float] = []

            print(f"[SCAN] Evaluating folder '{folder_name}' ({len(images)} images)...")

            for img_path in images:
                try:
                    dets, annotated = self.evaluate_image(img_path)
                    out_img_path = target_annotated_dir / f"annotated_{img_path.name}"
                    cv2.imwrite(str(out_img_path), annotated)

                    img_rec: Dict[str, Any] = {
                        "filename": img_path.name,
                        "detections_count": len(dets),
                        "detections": dets,
                    }
                    folder_stats["images"].append(img_rec)

                    if len(dets) == 0:
                        folder_stats["zero_detection_images"].append(img_path.name)
                        metrics["overall"]["zero_detection_images"].append(f"{folder_name}/{img_path.name}")
                    else:
                        for d in dets:
                            c = d["confidence"]
                            cat = d["category"]
                            folder_confs.append(c)
                            all_confidences.append(c)
                            folder_stats["detections_count"] += 1
                            folder_stats["category_distribution"][cat] = (
                                folder_stats["category_distribution"].get(cat, 0) + 1
                            )
                            if c < 0.45:
                                folder_stats["low_confidence_count"] += 1
                                metrics["overall"]["low_confidence_count"] += 1

                except Exception as e:
                    print(f"[ERROR] Failed processing {img_path.name}: {e}")

            if folder_confs:
                folder_stats["avg_confidence"] = round(float(np.mean(folder_confs)), 4)

            metrics["per_folder"][folder_name] = folder_stats
            metrics["overall"]["total_images"] += len(images)
            metrics["overall"]["total_detections"] += folder_stats["detections_count"]

        if all_confidences:
            metrics["overall"]["avg_confidence"] = round(float(np.mean(all_confidences)), 4)

        elapsed = time.perf_counter() - start_time
        metrics["overall"]["elapsed_seconds"] = round(elapsed, 2)

        self._save_reports(metrics)
        return metrics

    def _save_reports(self, metrics: Dict[str, Any]) -> None:
        """Write JSON summary and human-readable text report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. JSON Report
        json_path = self.output_dir / self.json_name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[REPORT] Saved JSON metrics to {json_path}")

        # 2. Text Report
        txt_path = self.output_dir / self.report_name
        lines = [
            "=" * 78,
            "FITLOOP REAL-WORLD WARDROBE EVALUATION DIAGNOSTIC REPORT",
            "=" * 78,
            f"Execution Timestamp : {metrics['timestamp']}",
            f"Model Checkpoint    : {metrics['model_path']}",
            f"Inference Parameters: conf={metrics['parameters']['conf']}, iou={metrics['parameters']['iou']}, agnostic_nms={metrics['parameters']['agnostic_nms']}",
            f"Center-Weighted Core: {round((1.0 - 2 * metrics['parameters']['core_trim']) * 100)}% (trimmed {round(metrics['parameters']['core_trim'] * 100)}% each edge)",
            f"Total Runtime       : {metrics['overall']['elapsed_seconds']} s",
            "-" * 78,
            "OVERALL SUMMARY",
            "-" * 78,
            f"Total Images Evaluated   : {metrics['overall']['total_images']}",
            f"Total Garments Detected  : {metrics['overall']['total_detections']}",
            f"Average Confidence       : {metrics['overall']['avg_confidence'] * 100:.2f}%" if metrics['overall']['total_detections'] > 0 else "Average Confidence       : N/A (0 detections)",
            f"Low-Confidence (< 0.45)  : {metrics['overall']['low_confidence_count']}",
            f"Zero-Detection Images    : {len(metrics['overall']['zero_detection_images'])}",
        ]

        if metrics["overall"]["zero_detection_images"]:
            lines.append("  Zero-Detection List:")
            for zd in metrics["overall"]["zero_detection_images"]:
                lines.append(f"    - {zd}")

        lines.extend([
            "-" * 78,
            "PER-FOLDER BREAKDOWN",
            "-" * 78,
        ])

        for folder_name in EVAL_FOLDERS:
            f_stats = metrics["per_folder"].get(folder_name, {})
            n_imgs = f_stats.get("images_count", 0)
            n_dets = f_stats.get("detections_count", 0)
            avg_c = f_stats.get("avg_confidence", 0.0)
            low_c = f_stats.get("low_confidence_count", 0)
            zero_list = f_stats.get("zero_detection_images", [])

            lines.extend([
                f"\nContext: [{folder_name.upper()}]",
                f"  Images Found        : {n_imgs}",
                f"  Garments Detected   : {n_dets}",
                f"  Average Confidence  : {avg_c * 100:.2f}%" if n_dets > 0 else "  Average Confidence  : N/A",
                f"  Low-Confidence (<45): {low_c}",
                f"  Zero Detections     : {len(zero_list)}",
            ])

            if f_stats.get("category_distribution"):
                lines.append("  Category Distribution:")
                for cat, count in sorted(f_stats["category_distribution"].items(), key=lambda x: -x[1]):
                    lines.append(f"    - {cat:<24}: {count}")

        lines.extend([
            "\n" + "-" * 78,
            "ITEMIZED DETECTION & COLOR EXTRACTION DETAILS",
            "-" * 78,
        ])

        for folder_name in EVAL_FOLDERS:
            f_stats = metrics["per_folder"].get(folder_name, {})
            for img_rec in f_stats.get("images", []):
                fname = img_rec["filename"]
                dets = img_rec.get("detections", [])
                lines.append(f"\nImage: {folder_name}/{fname} ({len(dets)} detected):")
                if not dets:
                    lines.append("  (No garments detected above threshold)")
                for idx, d in enumerate(dets, 1):
                    c = d['confidence'] * 100
                    cat = d['category']
                    dom = d.get('dominant_color', {})
                    sec = d.get('secondary_color')
                    sec_str = f", Secondary: {sec['name']} ({sec['hex']})" if sec else ""
                    lines.append(
                        f"  [{idx}] {cat:<20} | Conf: {c:5.1f}% | BBox: {d['bbox']} | Core: {d.get('core_bbox')} | Dominant: {dom.get('name')} ({dom.get('hex')}){sec_str}"
                    )

        lines.append("\n" + "=" * 78)
        report_text = "\n".join(lines)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[REPORT] Saved text diagnostic report to {txt_path}")
        print("\n" + report_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FitLoop Real-World Wardrobe Evaluation Pipeline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "real_world_eval",
        help="Root folder containing closet, flat_lay, and single_garment subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "real_world_eval",
        help="Destination directory for reports and annotated images.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_DIR / "models" / "best.pt",
        help="Path to YOLO26n checkpoint.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for class-agnostic NMS.",
    )
    parser.add_argument(
        "--core-trim",
        type=float,
        default=0.20,
        help="Ratio to trim from each bbox edge for center-weighted color extraction (default 0.20 = inner 60%).",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="eval_report.txt",
        help="Filename for text evaluation report.",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="eval_summary.json",
        help="Filename for JSON summary metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference compute device ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Automatically set diagnostic report filenames when conf=0.15 or conf=0.35 is requested
    if abs(args.conf - 0.15) < 1e-4 and args.report_name == "eval_report.txt":
        args.report_name = "eval_report_conf015.txt"
        args.json_name = "eval_summary_conf015.json"
    elif abs(args.conf - 0.35) < 1e-4 and args.report_name == "eval_report.txt":
        args.report_name = "eval_report_conf035.txt"
        args.json_name = "eval_summary_conf035.json"

    evaluator = RealWorldEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        agnostic_nms=True,
        core_trim=args.core_trim,
        report_name=args.report_name,
        json_name=args.json_name,
        device=args.device,
    )
    metrics = evaluator.run_evaluation()
    print(f"\n[COMPLETE] Real-world evaluation finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
