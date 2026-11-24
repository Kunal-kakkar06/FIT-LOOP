"""
scripts/run_stage7e_comparison.py
---------------------------------
A/B Evaluation Harness comparing BaselineFixedFractionStrategy vs
AnatomicalCrossViewStrategy on participant image pairs.

Outputs detailed comparison tables, candidate correspondence evidence,
circumference discrepancies, execution runtime, and diagnostic warnings.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from fitloop_measurement.evaluation.dataset import load_ground_truth_csv
from fitloop_measurement.features import GeometricFeatureExtractor
from fitloop_measurement.measurements.calibration import calibrate_dual_views
from fitloop_measurement.measurements.estimator import CircumferenceEstimator
from fitloop_measurement.measurements.schemas import AnatomicalLevel
from fitloop_measurement.measurements.slice_strategy import (
    AnatomicalCrossViewStrategy,
    BaselineFixedFractionStrategy,
)
from fitloop_measurement.pose import PoseExtractor
from fitloop_measurement.segmentation import BodySegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ab_comparison(gt_csv_path: str = "pilot_template.csv") -> Dict[str, Any]:
    sessions = load_ground_truth_csv(gt_csv_path)
    logger.info(f"Loaded {len(sessions)} participant session(s) from {gt_csv_path}")

    pose_extractor = PoseExtractor()
    segmenter = BodySegmenter()
    feature_extractor = GeometricFeatureExtractor()

    base_estimator = CircumferenceEstimator(strategy=BaselineFixedFractionStrategy())
    exp_estimator = CircumferenceEstimator(strategy=AnatomicalCrossViewStrategy())

    results: List[Dict[str, Any]] = []

    try:
        for s in sessions:
            pid = s.participant_id
            sid = s.session_id
            height_cm = s.standing_height_cm
            logger.info(f"Processing {pid} ({sid}), height={height_cm} cm...")

            f_img = cv2.imread(s.front_image_path)
            s_img = cv2.imread(s.side_image_path)
            if f_img is None or s_img is None:
                logger.error(f"Could not load images for {pid}")
                continue

            # Stage 3A & 3B
            f_pose = pose_extractor.extract(f_img, view_type="front")
            s_pose = pose_extractor.extract(s_img, view_type="side")
            f_seg = segmenter.segment(f_img, view_type="front")
            s_seg = segmenter.segment(s_img, view_type="side")

            # Stage 4A
            f_geom = feature_extractor.extract_view_features(f_pose, f_seg)
            s_geom = feature_extractor.extract_view_features(s_pose, s_seg)
            paired = feature_extractor.extract_paired_features(f_geom, s_geom, expected_height_cm=height_cm)

            f_mask = f_seg.mask if f_seg.is_successful else None
            s_mask = s_seg.mask if s_seg.is_successful else None

            # Calibration
            calibration = calibrate_dual_views(height_cm, f_geom, s_geom)

            # Benchmark Baseline
            t0 = time.perf_counter()
            base_out = base_estimator.estimate_measurements(
                paired_features=paired,
                known_height_cm=height_cm,
                front_mask=f_mask,
                side_mask=s_mask,
            )
            t_base_ms = (time.perf_counter() - t0) * 1000.0

            # Benchmark Experimental
            t1 = time.perf_counter()
            exp_out = exp_estimator.estimate_measurements(
                paired_features=paired,
                known_height_cm=height_cm,
                front_mask=f_mask,
                side_mask=s_mask,
            )
            t_exp_ms = (time.perf_counter() - t1) * 1000.0

            # Extract detailed comparison per level
            level_comparisons = {}
            for lvl in [AnatomicalLevel.CHEST, AnatomicalLevel.WAIST, AnatomicalLevel.ABDOMEN, AnatomicalLevel.HIPS]:
                lvl_key = lvl.value
                base_m = base_out.measurements.get(lvl_key)
                exp_m = exp_out.measurements.get(lvl_key)

                # Query slice details from strategies directly
                base_pair = base_estimator.strategy.select_slices(
                    lvl, f_geom, s_geom, calibration, f_mask, s_mask
                )
                exp_pair = exp_estimator.strategy.select_slices(
                    lvl, f_geom, s_geom, calibration, f_mask, s_mask
                )

                ref_val = s.measurements.get(lvl_key).reference_value_cm if s.measurements.get(lvl_key) else None

                level_comparisons[lvl_key] = {
                    "reference_tape_cm": ref_val,
                    "baseline": {
                        "circumference_cm": base_m.circumference_cm if base_m else None,
                        "front_row": base_pair.front_slice.image_y if base_pair.front_slice else None,
                        "side_row": base_pair.side_slice.image_y if base_pair.side_slice else None,
                        "front_width_px": base_m.front_width_px if base_m else None,
                        "side_depth_px": base_m.side_depth_px if base_m else None,
                        "height_delta_cm": base_pair.height_delta_cm,
                        "is_aligned": base_pair.is_aligned,
                        "confidence": base_pair.confidence.value,
                        "method_front": base_pair.front_result.selection_method,
                        "method_side": base_pair.side_result.selection_method,
                        "warnings": base_pair.warnings,
                    },
                    "experimental": {
                        "circumference_cm": exp_m.circumference_cm if exp_m else None,
                        "front_row": exp_pair.front_slice.image_y if exp_pair.front_slice else None,
                        "side_row": exp_pair.side_slice.image_y if exp_pair.side_slice else None,
                        "front_width_px": exp_m.front_width_px if exp_m else None,
                        "side_depth_px": exp_m.side_depth_px if exp_m else None,
                        "height_delta_cm": exp_pair.height_delta_cm,
                        "is_aligned": exp_pair.is_aligned,
                        "confidence": exp_pair.confidence.value,
                        "method_front": exp_pair.front_result.selection_method,
                        "method_side": exp_pair.side_result.selection_method,
                        "warnings": exp_pair.warnings,
                    },
                    "delta_circumference_cm": (
                        round(exp_m.circumference_cm - base_m.circumference_cm, 2)
                        if (exp_m and exp_m.circumference_cm and base_m and base_m.circumference_cm)
                        else None
                    ),
                }

            results.append({
                "participant_id": pid,
                "session_id": sid,
                "height_cm": height_cm,
                "runtime_ms": {
                    "baseline": round(t_base_ms, 2),
                    "experimental": round(t_exp_ms, 2),
                },
                "levels": level_comparisons,
            })

    finally:
        pose_extractor.close()
        segmenter.close()

    return {"results": results}


def main():
    parser = argparse.ArgumentParser(description="Stage 7E.3 A/B Strategy Comparison Harness")
    parser.add_argument("--gt-csv", default="pilot_template.csv", help="Path to ground truth CSV")
    parser.add_argument("--out-json", default="evaluation_output/stage7e_comparison.json", help="Output JSON path")
    args = parser.parse_args()

    out_data = run_ab_comparison(args.gt_csv)
    
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print("\n" + "="*80)
    print("STAGE 7E.3 STRATEGY COMPARISON SUMMARY")
    print("="*80)
    for p in out_data["results"]:
        pid = p["participant_id"]
        print(f"\nParticipant: {pid} (Height: {p['height_cm']} cm)")
        print(f"Runtime: Baseline={p['runtime_ms']['baseline']} ms | Experimental={p['runtime_ms']['experimental']} ms")
        print(f"{'Site':8} | {'Ref (cm)':8} | {'Base Circ':10} {'Exp Circ':10} {'Diff':6} | {'Base Rows (F/S)':16} {'Exp Rows (F/S)':16} | {'Base Δh':7} {'Exp Δh':7} {'Exp Conf':8}")
        print("-" * 105)
        for lvl, data in p["levels"].items():
            ref = f"{data['reference_tape_cm']:.1f}" if data['reference_tape_cm'] else "N/A"
            bc = f"{data['baseline']['circumference_cm']:.2f}" if data['baseline']['circumference_cm'] else "N/A"
            ec = f"{data['experimental']['circumference_cm']:.2f}" if data['experimental']['circumference_cm'] else "N/A"
            diff = f"{data['delta_circumference_cm']:+.2f}" if data['delta_circumference_cm'] is not None else "N/A"
            brows = f"{data['baseline']['front_row']}/{data['baseline']['side_row']}"
            erows = f"{data['experimental']['front_row']}/{data['experimental']['side_row']}"
            bdh = f"{data['baseline']['height_delta_cm']:.1f}cm" if data['baseline']['height_delta_cm'] is not None else "N/A"
            edh = f"{data['experimental']['height_delta_cm']:.1f}cm" if data['experimental']['height_delta_cm'] is not None else "N/A"
            conf = data['experimental']['confidence']
            print(f"{lvl.upper():8} | {ref:8} | {bc:10} {ec:10} {diff:6} | {brows:16} {erows:16} | {bdh:7} {edh:7} {conf:8}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
