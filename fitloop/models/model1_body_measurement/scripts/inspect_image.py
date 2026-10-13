#!/usr/bin/env python3
"""
scripts/inspect_image.py
-------------------------
CLI tool: load an image and print quality-check results to stdout.
Useful for quickly evaluating a capture before feeding it to the pipeline.

Usage::

    conda run -n ml_env python scripts/inspect_image.py \\
        --image data/raw/front.jpg \\
        --height 175

Options
-------
--image    Path to the image file (required).
--height   Subject's known standing height in centimetres (optional but
           recommended — without it scale calibration will fail).
--show     Display the image with annotated landmarks (requires a display).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the src package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import mediapipe as mp
import numpy as np

from fitloop_measurement.config import DEFAULT_MODEL_PATH
from fitloop_measurement.image_io import ImageLoadError, load_image, resize_for_display
from fitloop_measurement.quality import QualityChecker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a body image for pipeline quality.")
    p.add_argument("--image", required=True, help="Path to the image file.")
    p.add_argument(
        "--height",
        type=float,
        default=None,
        help="Subject's known height in cm (for scale calibration).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image (requires a display).",
    )
    return p.parse_args()


def draw_landmarks_tasks(bgr: np.ndarray, landmarks_list) -> np.ndarray:
    """Overlay landmark points and connections onto bgr."""
    annotated = bgr.copy()
    h, w = annotated.shape[:2]

    for landmarks in landmarks_list:
        coords = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        # Draw standard pose connections if available
        try:
            connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
            for edge in connections:
                start_idx = edge.start
                end_idx = edge.end
                if start_idx < len(coords) and end_idx < len(coords):
                    cv2.line(
                        annotated,
                        coords[start_idx],
                        coords[end_idx],
                        (0, 128, 255),
                        2,
                    )
        except Exception:
            pass

    return annotated


def main() -> None:
    args = parse_args()

    # ---------- Load ----------
    try:
        img = load_image(args.image)
    except ImageLoadError as exc:
        print(f"ERROR loading image: {exc}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"\nImage   : {args.image}")
    print(f"Size    : {w} × {h} px")

    # ---------- Quality check ----------
    with QualityChecker() as checker:
        report = checker.run(img, subject_height_cm=args.height)
        print(f"\n{report}")

        # ---------- Optionally display ----------
        if args.show:
            from fitloop_measurement.image_io import image_to_rgb

            rgb = image_to_rgb(img)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker = checker._get_landmarker()
            results = landmarker.detect(mp_image)

            display = img
            if results.pose_landmarks:
                display = draw_landmarks_tasks(img, results.pose_landmarks)

            display = resize_for_display(display, max_side=900)
            cv2.imshow("FitLoop — Inspect Image", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
