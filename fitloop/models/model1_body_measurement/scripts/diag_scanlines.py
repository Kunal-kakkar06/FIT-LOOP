"""
scripts/diag_scanlines.py
--------------------------
Generates per-level diagnostic overlays for front + side views showing:

  - Original image + segmentation mask overlay
  - Pose landmark dots (key body landmarks labelled)
  - For each anatomical scanline (chest / waist / abdomen / hips):
      * All disconnected foreground intervals — each coloured distinctly
      * The SELECTED interval (widest) shown in bright green / bold
      * The OLD outermost-span (pre-fix) shown as a dashed red line
      * Text annotation:  y, N segments, selected_width, old_width, delta

Run:
    PYTHONPATH=src conda run -n ml_env python scripts/diag_scanlines.py \
        --front  /path/to/front.jpg \
        --side   /path/to/side.jpg \
        --height 175 \
        --out    diagnostic_overlays/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC = _SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(_SRC))

LEVEL_COLOURS = {
    "chest":   (255, 80,  80),
    "waist":   (80,  160, 255),
    "abdomen": (80,  255, 255),
    "hips":    (255, 80,  255),
}
SELECTED_COLOUR = (0,   255,  0)
REJECTED_COLOUR = (0,   60,  255)
OLD_SPAN_COLOUR = (0,   0,   200)
LANDMARK_COLOUR = (0,   0,   255)
LABEL_BG        = (0,   0,   0)


def _extract_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    fg = (row > 0).astype(np.int8)
    diffs = np.diff(fg, prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0] - 1
    runs = [(int(s), int(e)) for s, e in zip(starts, ends)]
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs


def _put_label(img, text, x, y, colour=(255,255,255), scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = max(0, x-2), max(0, y-th-4)
    x2, y2 = min(img.shape[1]-1, x+tw+2), min(img.shape[0]-1, y+4)
    cv2.rectangle(img, (x1, y1), (x2, y2), LABEL_BG, -1)
    cv2.putText(img, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def _draw_dashed_hline(img, y, x_start, x_end, colour, dash=12):
    for x in range(x_start, x_end, dash*2):
        cv2.line(img, (x, y), (min(x+dash, x_end), y), colour, 2)


def build_overlay(bgr_img, mask, landmarks_px, scanlines, alpha=0.35):
    h, w = bgr_img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    green = np.zeros_like(bgr_img); green[:, :] = (0, 220, 0)
    fg = (mask > 0).astype(np.uint8)[:, :, np.newaxis]
    canvas = bgr_img.copy()
    canvas = np.where(fg, (canvas*(1-alpha)+green*alpha).astype(np.uint8), canvas)
    KEY = {"LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_HIP","RIGHT_HIP","LEFT_WRIST","RIGHT_WRIST"}
    for name, px, py in landmarks_px:
        if 0<=px<w and 0<=py<h:
            cv2.circle(canvas, (px,py), 7, LANDMARK_COLOUR, -1)
            if name in KEY:
                _put_label(canvas, name, px+8, py, (200,200,200), scale=0.38)
    for level_name, y_row in scanlines:
        if not (0<=y_row<h): continue
        line_col = LEVEL_COLOURS.get(level_name, (200,200,200))
        row = mask[y_row, :]
        runs = _extract_runs(row)
        if len(runs)==0:
            _put_label(canvas, f"{level_name.upper()}: y={y_row} NO FG", 6, y_row-6, line_col)
            continue
        fg_indices = np.where(row>0)[0]
        old_left, old_right = int(fg_indices[0]), int(fg_indices[-1])
        old_width = old_right - old_left + 1
        _draw_dashed_hline(canvas, max(0,y_row-4), old_left, old_right, OLD_SPAN_COLOUR)
        for i,(rx0,rx1) in enumerate(runs):
            col = SELECTED_COLOUR if i==0 else REJECTED_COLOUR
            lw  = 4 if i==0 else 2
            cv2.line(canvas, (rx0,y_row), (rx1,y_row), col, lw)
            cv2.circle(canvas, (rx0,y_row), 5, col, -1)
            cv2.circle(canvas, (rx1,y_row), 5, col, -1)
        sel = runs[0][1]-runs[0][0]+1
        delta = old_width-sel
        label = f"{level_name.upper()}: y={y_row} segs={len(runs)} selected={sel}px old_span={old_width}px delta={delta}px"
        _put_label(canvas, label, 6, y_row-10, line_col, scale=0.48)
    return canvas


def run_pipeline(front_path, side_path, height_cm, out_dir):
    from fitloop_measurement.segmentation.segmenter import BodySegmenter
    from fitloop_measurement.pose.extractor import PoseExtractor
    from fitloop_measurement.features.extractor import GeometricFeatureExtractor
    from fitloop_measurement.measurements.calibration import calibrate_dual_views
    from fitloop_measurement.measurements.slice_selector import select_anatomical_slice
    from fitloop_measurement.measurements.schemas import AnatomicalLevel

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n[DIAG] Loading images...")
    front_bgr = cv2.imread(front_path)
    side_bgr  = cv2.imread(side_path)
    if front_bgr is None or side_bgr is None:
        print(f"[FAIL] Could not load images"); sys.exit(1)
    print(f"  front: {front_bgr.shape[1]}x{front_bgr.shape[0]}")
    print(f"  side : {side_bgr.shape[1]}x{side_bgr.shape[0]}")

    print("[DIAG] Segmentation...")
    with BodySegmenter() as seg:
        front_seg = seg.segment(front_bgr, view_type="front")
        side_seg  = seg.segment(side_bgr,  view_type="side")
    print(f"  front: {front_seg.status} fg={front_seg.foreground_ratio:.3f}")
    print(f"  side : {side_seg.status}  fg={side_seg.foreground_ratio:.3f}")

    print("[DIAG] Pose extraction...")
    with PoseExtractor() as pose:
        front_pose = pose.extract(front_bgr, view_type="front")
        side_pose  = pose.extract(side_bgr,  view_type="side")
    print(f"  front: {front_pose.status} lms={len(front_pose.landmarks)}")
    print(f"  side : {side_pose.status}  lms={len(side_pose.landmarks)}")

    print("[DIAG] Feature extraction...")
    extractor = GeometricFeatureExtractor()
    front_feat = extractor.extract_view_features(front_pose, front_seg)
    side_feat  = extractor.extract_view_features(side_pose,  side_seg)
    extractor.extract_paired_features(front_feat, side_feat, expected_height_cm=height_cm)

    cal = calibrate_dual_views(height_cm, front_feat, side_feat)
    print(f"\n[DIAG] Calibration: valid={cal.is_valid}")
    print(f"  front scale={cal.front.scale_cm_per_pixel:.6f} cm/px  sil_h={front_feat.silhouette_height}px")
    print(f"  side  scale={cal.side.scale_cm_per_pixel:.6f} cm/px   sil_h={side_feat.silhouette_height}px")

    LEVELS = [
        (AnatomicalLevel.CHEST,   "chest"),
        (AnatomicalLevel.WAIST,   "waist"),
        (AnatomicalLevel.ABDOMEN, "abdomen"),
        (AnatomicalLevel.HIPS,    "hips"),
    ]

    print("\n[DIAG] Per-level analysis:")
    front_scanlines, side_scanlines = [], []
    for level, name in LEVELS:
        f_sl, f_res = select_anatomical_slice(level, front_feat)
        s_sl, s_res = select_anatomical_slice(level, side_feat)
        f_y  = f_sl.image_y if f_sl else None
        f_w  = f_sl.width_pixels if f_sl else None
        f_n  = f_sl.segment_count if f_sl else 0
        s_y  = s_sl.image_y if s_sl else None
        s_w  = s_sl.width_pixels if s_sl else None

        # Old bug: outermost span
        f_old = None
        if f_sl and front_seg.mask is not None and f_y is not None:
            row = front_seg.mask[f_y, :]
            fi = np.where(row>0)[0]
            if len(fi): f_old = int(fi[-1])-int(fi[0])+1

        f_cm     = f_w   * cal.front.scale_cm_per_pixel if (f_w   and cal.is_valid) else None
        f_old_cm = f_old * cal.front.scale_cm_per_pixel if (f_old and cal.is_valid) else None
        s_cm     = s_w   * cal.side.scale_cm_per_pixel  if (s_w   and cal.is_valid) else None

        print(f"\n  {name.upper()}:")
        print(f"    front y={f_y} segs={f_n}")
        print(f"      SELECTED  width={f_w}px = {f_cm:.1f}cm" if f_cm else f"      SELECTED  width={f_w}px")
        print(f"      OLD SPAN  width={f_old}px = {f_old_cm:.1f}cm  (BUG)" if f_old_cm else f"      OLD SPAN  width={f_old}px")
        print(f"    side  y={s_y} width={s_w}px = {s_cm:.1f}cm" if s_cm else f"    side  y={s_y} width={s_w}px")
        print(f"    warnings: {f_res.warnings + s_res.warnings}")

        if f_y is not None: front_scanlines.append((name, f_y))
        if s_y is not None: side_scanlines.append((name, s_y))

    def _lm_pixels(feat):
        out = []
        for lm in feat.pixel_landmarks:
            if lm.is_valid and lm.pixel_x is not None and lm.pixel_y is not None:
                out.append((lm.name, int(lm.pixel_x), int(lm.pixel_y)))
        return out

    print("\n[DIAG] Generating overlay images...")
    fmask = front_seg.mask if front_seg.mask is not None else np.zeros(front_bgr.shape[:2], np.uint8)
    smask = side_seg.mask  if side_seg.mask  is not None else np.zeros(side_bgr.shape[:2],  np.uint8)

    fo = build_overlay(front_bgr, fmask, _lm_pixels(front_feat), front_scanlines)
    so = build_overlay(side_bgr,  smask, _lm_pixels(side_feat),  side_scanlines)

    fout = str(Path(out_dir)/"diag_front_scanlines.jpg")
    sout = str(Path(out_dir)/"diag_side_scanlines.jpg")
    cv2.imwrite(fout, fo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(sout, so, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  {fout}")
    print(f"  {sout}")
    print("\n[DIAG] Complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front",  default=None)
    ap.add_argument("--side",   default=None)
    ap.add_argument("--height", type=float, default=175.0)
    ap.add_argument("--out",    default="diagnostic_overlays")
    args = ap.parse_args()

    front_path = args.front
    side_path  = args.side

    if not front_path or not side_path:
        candidates = sorted(
            glob.glob("/tmp/fitloop_diag_front*.jpg") +
            glob.glob("/tmp/fitloop_diag_side*.jpg"),
            key=os.path.getmtime, reverse=True
        )
        for p in candidates:
            if "front" in p and not front_path: front_path = p
            if "side"  in p and not side_path:  side_path  = p

    if not front_path or not side_path:
        print("[DIAG] Usage: PYTHONPATH=src conda run -n ml_env python scripts/diag_scanlines.py \\")
        print("              --front /path/front.jpg --side /path/side.jpg --height 172")
        sys.exit(1)

    print(f"[DIAG] front : {front_path}")
    print(f"[DIAG] side  : {side_path}")
    print(f"[DIAG] height: {args.height} cm")
    run_pipeline(front_path, side_path, args.height, args.out)


if __name__ == "__main__":
    main()
