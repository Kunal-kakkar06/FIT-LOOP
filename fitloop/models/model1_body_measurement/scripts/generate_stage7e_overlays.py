"""
scripts/generate_stage7e_overlays.py
-----------------------------------
Generates diagnostic visual overlays for Stage 7E.3 comparing
BaselineFixedFractionStrategy scanlines vs AnatomicalCrossViewStrategy
candidate scanlines on front and side perspectives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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


def draw_slice_line(img, y_row, left_x, right_x, color, label, is_dashed=False):
    h, w = img.shape[:2]
    if y_row is None or not (0 <= y_row < h):
        return

    # Draw horizontal scanline
    if left_x is not None and right_x is not None:
        cv2.line(img, (int(left_x), int(y_row)), (int(right_x), int(y_row)), color, 3)
        # Endpoint circles
        cv2.circle(img, (int(left_x), int(y_row)), 5, color, -1)
        cv2.circle(img, (int(right_x), int(y_row)), 5, color, -1)
    else:
        cv2.line(img, (0, int(y_row)), (w - 1, int(y_row)), color, 1)

    # Label text
    text_pos_x = min(w - 280, max(15, (int(right_x) + 15) if right_x else 20))
    cv2.putText(img, label, (text_pos_x, int(y_row) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 7E.3 Strategy Overlays")
    parser.add_argument("--gt-csv", default="pilot_template.csv", help="Path to pilot CSV")
    parser.add_argument("--out-dir", default="diagnostic_overlays/stage7e", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = load_ground_truth_csv(args.gt_csv)
    pose_ext = PoseExtractor()
    seg_ext = BodySegmenter()
    feat_ext = GeometricFeatureExtractor()

    base_strat = BaselineFixedFractionStrategy()
    exp_strat = AnatomicalCrossViewStrategy()

    levels = [AnatomicalLevel.CHEST, AnatomicalLevel.WAIST, AnatomicalLevel.ABDOMEN, AnatomicalLevel.HIPS]

    try:
        for s in sessions:
            pid = s.participant_id
            f_raw = cv2.imread(s.front_image_path)
            s_raw = cv2.imread(s.side_image_path)
            if f_raw is None or s_raw is None:
                continue

            fp = pose_ext.extract(f_raw, "front")
            sp = pose_ext.extract(s_raw, "side")
            fs = seg_ext.segment(f_raw, "front")
            ss = seg_ext.segment(s_raw, "side")

            ff = feat_ext.extract_view_features(fp, fs)
            sf = feat_ext.extract_view_features(sp, ss)
            calib = calibrate_dual_views(s.standing_height_cm, ff, sf)

            f_mask = fs.mask if fs.is_successful else None
            s_mask = ss.mask if ss.is_successful else None

            # Prepare overlay images
            f_vis = f_raw.copy()
            s_vis = s_raw.copy()

            # Level colors (BGR): Baseline in Solid, Experimental in Cyan/Yellow
            level_colors = {
                AnatomicalLevel.CHEST:   (255, 180, 0),    # Blue/Gold
                AnatomicalLevel.WAIST:   (0, 140, 255),    # Orange
                AnatomicalLevel.ABDOMEN: (0, 230, 0),      # Green
                AnatomicalLevel.HIPS:    (200, 0, 220),    # Magenta
            }

            for lvl in levels:
                b_pair = base_strat.select_slices(lvl, ff, sf, calib, f_mask, s_mask)
                e_pair = exp_strat.select_slices(lvl, ff, sf, calib, f_mask, s_mask)

                col = level_colors[lvl]
                exp_col = (255, 255, 0) if lvl == AnatomicalLevel.WAIST else (0, 255, 255)

                # Front view annotations
                if b_pair.front_slice:
                    draw_slice_line(
                        f_vis, b_pair.front_slice.image_y,
                        b_pair.front_slice.left_x, b_pair.front_slice.right_x,
                        col, f"Base {lvl.value.upper()} (y={b_pair.front_slice.image_y})"
                    )
                if e_pair.front_slice and (not b_pair.front_slice or e_pair.front_slice.image_y != b_pair.front_slice.image_y):
                    draw_slice_line(
                        f_vis, e_pair.front_slice.image_y,
                        e_pair.front_slice.left_x, e_pair.front_slice.right_x,
                        exp_col, f"Exp {lvl.value.upper()} (y={e_pair.front_slice.image_y})"
                    )

                # Side view annotations
                if b_pair.side_slice:
                    draw_slice_line(
                        s_vis, b_pair.side_slice.image_y,
                        b_pair.side_slice.left_x, b_pair.side_slice.right_x,
                        col, f"Base {lvl.value.upper()} (y={b_pair.side_slice.image_y})"
                    )
                if e_pair.side_slice and (not b_pair.side_slice or e_pair.side_slice.image_y != b_pair.side_slice.image_y):
                    draw_slice_line(
                        s_vis, e_pair.side_slice.image_y,
                        e_pair.side_slice.left_x, e_pair.side_slice.right_x,
                        exp_col, f"Exp {lvl.value.upper()} (y={e_pair.side_slice.image_y})"
                    )

            # Save overlay outputs
            f_out_path = out_dir / f"{pid}_front_stage7e_overlay.jpg"
            s_out_path = out_dir / f"{pid}_side_stage7e_overlay.jpg"
            cv2.imwrite(str(f_out_path), f_vis)
            cv2.imwrite(str(s_out_path), s_vis)
            print(f"Saved overlays for {pid}: {f_out_path}, {s_out_path}")

    finally:
        pose_ext.close()
        seg_ext.close()


if __name__ == "__main__":
    main()
