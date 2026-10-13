"""
measurements/calibration.py
---------------------------
Metric pixel-to-centimetre scale calibration using user standing height.

Assumptions and Limitations:
- scale_cm_per_pixel = known_height_cm / silhouette_height_px
- Calculates separate scales for front and side views; never averages them.
- Assumes the subject is standing upright and the full body (head apex to heels) is unclipped.
- Does NOT correct for camera perspective divergence or vertical tilt.
"""

from __future__ import annotations

import math
from typing import Optional

from ..config import HEIGHT_MAX_CM, HEIGHT_MIN_CM
from ..features.schemas import ViewGeometricFeatures
from .schemas import CalibrationStatus, DualViewCalibration, ViewCalibration


def calibrate_view(
    known_height_cm: Optional[float],
    view_features: ViewGeometricFeatures,
) -> ViewCalibration:
    """Calculate metric scale for a single perspective view."""
    warnings: list[str] = []

    # 1. Height sanity check
    if known_height_cm is None or not math.isfinite(known_height_cm) or known_height_cm <= 0:
        return ViewCalibration(
            view_type=view_features.view_type,
            silhouette_height_px=0,
            scale_cm_per_pixel=0.0,
            is_valid=False,
            status=CalibrationStatus.INVALID_HEIGHT,
            warnings=["Subject standing height is missing, non-finite, or non-positive."],
        )

    if not (HEIGHT_MIN_CM <= known_height_cm <= HEIGHT_MAX_CM):
        warnings.append(
            f"Known height {known_height_cm} cm is outside plausible human stature range [{HEIGHT_MIN_CM}, {HEIGHT_MAX_CM}] cm."
        )

    # 2. Silhouette height check
    sil_h = view_features.silhouette_height
    if sil_h is None or not math.isfinite(sil_h) or sil_h <= 0:
        return ViewCalibration(
            view_type=view_features.view_type,
            silhouette_height_px=0,
            scale_cm_per_pixel=0.0,
            is_valid=False,
            status=CalibrationStatus.INVALID_SILHOUETTE,
            warnings=["Visible silhouette height is missing or zero."],
        )

    # 3. Check for boundary clipping (head at y=0 or feet at bottom y=H-1)
    bbox_y = view_features.silhouette_bbox_y or 0
    img_h = view_features.image_height
    if bbox_y <= 2:
        warnings.append("Silhouette appears clipped at the top of the frame (head cropped).")
    if (bbox_y + sil_h) >= (img_h - 2):
        warnings.append("Silhouette appears clipped at the bottom of the frame (feet cropped).")

    is_clipped = len([w for w in warnings if "clipped" in w]) > 0
    status = CalibrationStatus.CLIPPED_BODY if is_clipped else CalibrationStatus.VALID

    # Calculate scale: cm per pixel
    scale_cm_per_pixel = float(known_height_cm) / float(sil_h)

    return ViewCalibration(
        view_type=view_features.view_type,
        silhouette_height_px=int(sil_h),
        scale_cm_per_pixel=scale_cm_per_pixel,
        is_valid=(status == CalibrationStatus.VALID),
        status=status,
        warnings=warnings,
    )


def calibrate_dual_views(
    known_height_cm: Optional[float],
    front_features: ViewGeometricFeatures,
    side_features: ViewGeometricFeatures,
) -> DualViewCalibration:
    """Calculate independent front and side scales and verify cross-view compatibility."""
    front_calib = calibrate_view(known_height_cm, front_features)
    side_calib = calibrate_view(known_height_cm, side_features)

    warnings: list[str] = []
    warnings.extend([f"[Front] {w}" for w in front_calib.warnings])
    warnings.extend([f"[Side] {w}" for w in side_calib.warnings])

    # Cross-view scale consistency check
    is_valid = front_calib.is_valid and side_calib.is_valid
    if front_calib.scale_cm_per_pixel > 0 and side_calib.scale_cm_per_pixel > 0:
        scale_ratio = front_calib.scale_cm_per_pixel / side_calib.scale_cm_per_pixel
        if abs(scale_ratio - 1.0) > 0.15:
            warnings.append(
                f"Front and side scale factors differ by {abs(scale_ratio - 1.0):.1%}. "
                "Camera distance, zoom, or subject distance differed between views."
            )

    return DualViewCalibration(
        known_height_cm=float(known_height_cm) if (known_height_cm and math.isfinite(known_height_cm)) else 0.0,
        front=front_calib,
        side=side_calib,
        is_valid=is_valid,
        warnings=warnings,
    )
