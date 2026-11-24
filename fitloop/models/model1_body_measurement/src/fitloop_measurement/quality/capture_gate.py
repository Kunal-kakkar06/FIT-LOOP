"""
quality/capture_gate.py
-----------------------
Capture quality validation gate for paired front and side images in FitLoop Model 1 (Stage 7G / 7G.1).

Evaluates:
1. Image dimensions & aspect ratio consistency.
2. Vertical framing margins (provisional 5% warning threshold default).
3. Silhouette bounding boxes and edge clipping.
4. Silhouette pixel-height discrepancy between views (provisional 10% threshold).
5. Calibration-scale discrepancy using the symmetric metric:
       abs(front_scale - side_scale) / ((front_scale + side_scale) / 2)
   with separate configurable warning (default 5%) and rejection (default 15%) thresholds.
6. Cross-view slice alignment metadata from the estimator strategy.

All thresholds are provisional and configurable via constructor arguments.
The quality gate runs AFTER feature extraction and circumference estimation so that
it can inspect the complete pipeline diagnostics. A REJECT status means one or more
critical violations were detected; diagnostic measurements are still returned for
debugging, but are explicitly marked as non-quality-approved via `quality_status`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import HEIGHT_MAX_CM, HEIGHT_MIN_CM, MIN_IMAGE_HEIGHT, MIN_IMAGE_WIDTH
from ..features.schemas import PairedGeometricFeatures, ViewGeometricFeatures
from ..measurements.schemas import DualViewCalibration, MeasurementEngineResult
from ..pose.schemas import PoseExtractionResult
from ..segmentation.schemas import SegmentationResult
from .schemas import (
    CaptureQualityReport,
    QualityIssueCode,
    QualityIssueDetail,
    QualitySeverity,
    QualityStatus,
    ViewFramingDiagnostics,
)

logger = logging.getLogger(__name__)


def compute_symmetric_scale_discrepancy(
    front_scale: Optional[float],
    side_scale: Optional[float],
) -> Optional[float]:
    """Calculate symmetric cross-view scale discrepancy metric:
    
    abs(front_scale - side_scale) / ((front_scale + side_scale) / 2)
    
    Returns None if either scale is missing, non-finite, non-positive, or if
    the denominator is non-positive.
    """
    if front_scale is None or side_scale is None:
        return None
    if not (math.isfinite(front_scale) and math.isfinite(side_scale)):
        return None
    if front_scale <= 0.0 or side_scale <= 0.0:
        return None

    denom = (front_scale + side_scale) / 2.0
    if denom <= 0.0 or not math.isfinite(denom):
        return None

    return abs(front_scale - side_scale) / denom


class CaptureQualityGate:
    """Configurable capture quality validation gate."""

    def __init__(
        self,
        min_margin_threshold: float = 0.05,        # 5% default provisional top/bottom margin
        scale_warning_threshold: float = 0.05,     # 5% default provisional warning threshold
        scale_rejection_threshold: float = 0.15,   # 15% default provisional rejection threshold
        aspect_ratio_tolerance: float = 0.02,      # 2% aspect ratio mismatch tolerance
        silhouette_height_discrepancy_threshold: float = 0.10,  # 10% provisional silhouette height mismatch
        scale_discrepancy_threshold: Optional[float] = None,    # Backward compatibility alias for warning threshold
    ) -> None:
        self.min_margin_threshold = min_margin_threshold
        if scale_discrepancy_threshold is not None:
            self.scale_warning_threshold = scale_discrepancy_threshold
        else:
            self.scale_warning_threshold = scale_warning_threshold
        self.scale_rejection_threshold = scale_rejection_threshold
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.silhouette_height_discrepancy_threshold = silhouette_height_discrepancy_threshold

    @property
    def scale_discrepancy_threshold(self) -> float:
        """Backward compatibility property pointing to scale_warning_threshold."""
        return self.scale_warning_threshold

    def _extract_view_framing(
        self,
        features: Optional[ViewGeometricFeatures],
        calibration_scale: Optional[float],
    ) -> Optional[ViewFramingDiagnostics]:
        if features is None or features.image_width <= 0 or features.image_height <= 0:
            return None

        w = features.image_width
        h = features.image_height
        ar = float(w) / float(h)

        bbox_x = features.silhouette_bbox_x
        bbox_y = features.silhouette_bbox_y
        sil_w = features.silhouette_width
        sil_h = features.silhouette_height

        top_margin = None
        bottom_margin = None
        is_clipped = False

        if bbox_y is not None and sil_h is not None and sil_h > 0:
            top_margin = float(bbox_y) / float(h)
            bottom_margin = float(h - (bbox_y + sil_h)) / float(h)
            if bbox_y <= 2 or (bbox_y + sil_h) >= (h - 2):
                is_clipped = True

        return ViewFramingDiagnostics(
            image_width=w,
            image_height=h,
            aspect_ratio=ar,
            silhouette_bbox_x=bbox_x,
            silhouette_bbox_y=bbox_y,
            silhouette_width=sil_w,
            silhouette_height=sil_h,
            top_margin_fraction=top_margin,
            bottom_margin_fraction=bottom_margin,
            scale_cm_per_pixel=calibration_scale,
            is_clipped=is_clipped,
        )

    def evaluate_pair(
        self,
        front_features: Optional[ViewGeometricFeatures],
        side_features: Optional[ViewGeometricFeatures],
        calibration: Optional[DualViewCalibration],
        front_pose: Optional[PoseExtractionResult] = None,
        side_pose: Optional[PoseExtractionResult] = None,
        front_seg: Optional[SegmentationResult] = None,
        side_seg: Optional[SegmentationResult] = None,
        measurement_result: Optional[MeasurementEngineResult] = None,
        standing_height_cm: Optional[float] = None,
    ) -> CaptureQualityReport:
        """Run all capture-quality gate checks."""
        issues: List[QualityIssueDetail] = []

        # 1. Height sanity check
        if standing_height_cm is not None:
            if not math.isfinite(standing_height_cm) or standing_height_cm <= 0:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.INVALID_HEIGHT,
                        severity=QualitySeverity.CRITICAL,
                        message=f"Provided standing height ({standing_height_cm}) is non-positive or non-finite.",
                        actionable_fix="Provide a positive finite barefoot height measurement in centimetres.",
                    )
                )
            elif not (HEIGHT_MIN_CM <= standing_height_cm <= HEIGHT_MAX_CM):
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.INVALID_HEIGHT,
                        severity=QualitySeverity.WARNING,
                        message=f"Provided standing height ({standing_height_cm} cm) is outside standard human stature range [{HEIGHT_MIN_CM}, {HEIGHT_MAX_CM}] cm.",
                        actionable_fix=f"Verify entered stature value is between {HEIGHT_MIN_CM} and {HEIGHT_MAX_CM} cm.",
                    )
                )

        # 2. View presence & Resolution check
        f_framing = None
        s_framing = None

        f_scale = calibration.front.scale_cm_per_pixel if (calibration and calibration.front.is_valid) else None
        s_scale = calibration.side.scale_cm_per_pixel if (calibration and calibration.side.is_valid) else None

        if front_features is not None:
            f_framing = self._extract_view_framing(front_features, f_scale)
            if front_features.image_width < MIN_IMAGE_WIDTH or front_features.image_height < MIN_IMAGE_HEIGHT:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.RESOLUTION_BELOW_MINIMUM,
                        severity=QualitySeverity.CRITICAL,
                        message=f"Front image resolution ({front_features.image_width}x{front_features.image_height}) is below minimum {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}.",
                        actionable_fix="Capture and upload front view with at least 480x640 resolution.",
                    )
                )
        else:
            issues.append(
                QualityIssueDetail(
                    code=QualityIssueCode.MISSING_IMAGE,
                    severity=QualitySeverity.CRITICAL,
                    message="Front view features were not provided.",
                    actionable_fix="Ensure front view image is captured and processed.",
                )
            )

        if side_features is not None:
            s_framing = self._extract_view_framing(side_features, s_scale)
            if side_features.image_width < MIN_IMAGE_WIDTH or side_features.image_height < MIN_IMAGE_HEIGHT:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.RESOLUTION_BELOW_MINIMUM,
                        severity=QualitySeverity.CRITICAL,
                        message=f"Side image resolution ({side_features.image_width}x{side_features.image_height}) is below minimum {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}.",
                        actionable_fix="Capture and upload side view with at least 480x640 resolution.",
                    )
                )
        else:
            issues.append(
                QualityIssueDetail(
                    code=QualityIssueCode.MISSING_IMAGE,
                    severity=QualitySeverity.CRITICAL,
                    message="Side view features were not provided.",
                    actionable_fix="Ensure side view image is captured and processed.",
                )
            )

        # 3. Geometry Parity (Dimensions & Aspect Ratio)
        if f_framing is not None and s_framing is not None:
            if f_framing.image_width != s_framing.image_width or f_framing.image_height != s_framing.image_height:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.IMAGE_DIMENSION_MISMATCH,
                        severity=QualitySeverity.WARNING,
                        message=(
                            f"Image dimensions differ between views (Front: {f_framing.image_width}x{f_framing.image_height}, "
                            f"Side: {s_framing.image_width}x{s_framing.image_height}). "
                            "Indicates captures were taken at different sensor resolutions or aspect crops; "
                            "does not by itself prove camera-to-subject distance change."
                        ),
                        actionable_fix="Capture both perspectives using the same camera device, orientation, and resolution.",
                        context={"front_dims": [f_framing.image_width, f_framing.image_height], "side_dims": [s_framing.image_width, s_framing.image_height]},
                    )
                )

            ar_diff = abs(f_framing.aspect_ratio - s_framing.aspect_ratio)
            if ar_diff > self.aspect_ratio_tolerance:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.ASPECT_RATIO_MISMATCH,
                        severity=QualitySeverity.WARNING,
                        message=f"Aspect ratios differ by {ar_diff:.3f} (Front: {f_framing.aspect_ratio:.3f}, Side: {s_framing.aspect_ratio:.3f}).",
                        actionable_fix="Use the same camera aspect ratio (e.g. 3:4 or 9:16) for both captures.",
                    )
                )

            # Separate check: Silhouette-height discrepancy in pixels between views
            if (
                f_framing.silhouette_height is not None
                and s_framing.silhouette_height is not None
                and f_framing.silhouette_height > 0
                and s_framing.silhouette_height > 0
            ):
                sil_h_diff = abs(f_framing.silhouette_height - s_framing.silhouette_height)
                sil_h_mean = (f_framing.silhouette_height + s_framing.silhouette_height) / 2.0
                sil_h_disc = sil_h_diff / sil_h_mean if sil_h_mean > 0 else 0.0

                if sil_h_disc > self.silhouette_height_discrepancy_threshold:
                    issues.append(
                        QualityIssueDetail(
                            code=QualityIssueCode.SILHOUETTE_HEIGHT_DISCREPANCY,
                            severity=QualitySeverity.WARNING,
                            message=(
                                f"Silhouette pixel heights differ by {sil_h_disc:.1%} (Front: {f_framing.silhouette_height}px, "
                                f"Side: {s_framing.silhouette_height}px). May reflect camera-distance change, posture sag, or cropping differences."
                            ),
                            actionable_fix="Maintain identical distance to camera and consistent upright posture across views.",
                            context={"silhouette_height_discrepancy": sil_h_disc, "front_px": f_framing.silhouette_height, "side_px": s_framing.silhouette_height},
                        )
                    )

        # 4. Silhouette & Framing Margins
        for view_name, framing, seg, pose in [
            ("Front", f_framing, front_seg, front_pose),
            ("Side", s_framing, side_seg, side_pose),
        ]:
            if framing is None:
                continue

            if pose is not None and not pose.is_successful:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.NO_PERSON_DETECTED,
                        severity=QualitySeverity.CRITICAL,
                        message=f"{view_name} pose landmark detection failed ({pose.status.value}).",
                        actionable_fix=f"Ensure subject is in full view with unobstructed lighting in {view_name.lower()} capture.",
                    )
                )

            if seg is not None and not seg.is_successful:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.SILHOUETTE_SEGMENTATION_FAILED,
                        severity=QualitySeverity.CRITICAL,
                        message=f"{view_name} silhouette segmentation failed ({seg.status.value}).",
                        actionable_fix=f"Ensure plain background and high contrast between subject clothing and wall in {view_name.lower()} capture.",
                    )
                )

            if framing.is_clipped:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.SILHOUETTE_CLIPPED,
                        severity=QualitySeverity.CRITICAL,
                        message=f"{view_name} silhouette touches the frame edge (head or feet cropped).",
                        actionable_fix=f"Step back or tilt camera to ensure subject's entire body from top of head to heels is unclipped in {view_name.lower()} view.",
                    )
                )

            # Check vertical margins (top and bottom >= threshold)
            if framing.top_margin_fraction is not None and framing.top_margin_fraction < self.min_margin_threshold:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.INSUFFICIENT_VERTICAL_MARGIN,
                        severity=QualitySeverity.WARNING,
                        message=f"{view_name} top margin ({framing.top_margin_fraction:.1%}) is below recommended {self.min_margin_threshold:.1%}.",
                        actionable_fix=f"Leave at least {self.min_margin_threshold:.0%} headroom above subject's head.",
                    )
                )

            if framing.bottom_margin_fraction is not None and framing.bottom_margin_fraction < self.min_margin_threshold:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.INSUFFICIENT_VERTICAL_MARGIN,
                        severity=QualitySeverity.WARNING,
                        message=f"{view_name} bottom margin ({framing.bottom_margin_fraction:.1%}) is below recommended {self.min_margin_threshold:.1%}.",
                        actionable_fix=f"Leave at least {self.min_margin_threshold:.0%} space below subject's feet.",
                    )
                )

            # Plausible silhouette height check (body must occupy at least 40% and at most 96% of frame)
            if framing.silhouette_height is not None and framing.image_height > 0:
                bh_frac = framing.silhouette_height / float(framing.image_height)
                if bh_frac < 0.40 or bh_frac > 0.96:
                    issues.append(
                        QualityIssueDetail(
                            code=QualityIssueCode.IMPLAUSIBLE_SILHOUETTE_HEIGHT,
                            severity=QualitySeverity.WARNING,
                            message=f"{view_name} silhouette occupies {bh_frac:.1%} of frame height (recommended: 60%-90%).",
                            actionable_fix="Reposition camera so the subject occupies approximately 70%-80% of vertical framing.",
                        )
                    )

        # 5. Multi-View Scale Discrepancy (Calibration Scale)
        scale_disc = None
        if calibration is not None:
            if not calibration.is_valid:
                issues.append(
                    QualityIssueDetail(
                        code=QualityIssueCode.CALIBRATION_UNAVAILABLE,
                        severity=QualitySeverity.CRITICAL,
                        message="Dual-view metric scale calibration failed.",
                        actionable_fix="Ensure valid standing height and unclipped silhouettes in both views.",
                    )
                )
            else:
                scale_disc = compute_symmetric_scale_discrepancy(
                    calibration.front.scale_cm_per_pixel,
                    calibration.side.scale_cm_per_pixel,
                )
                if scale_disc is not None:
                    if scale_disc > self.scale_rejection_threshold:
                        issues.append(
                            QualityIssueDetail(
                                code=QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED,
                                severity=QualitySeverity.CRITICAL,
                                message=(
                                    f"Cross-view scale discrepancy is {scale_disc:.1%} (exceeds rejection threshold {self.scale_rejection_threshold:.1%}). "
                                    f"Front scale={calibration.front.scale_cm_per_pixel:.5f} cm/px, Side scale={calibration.side.scale_cm_per_pixel:.5f} cm/px."
                                ),
                                actionable_fix="Ensure camera distance, optical zoom, and subject floor position are strictly identical between front and side captures.",
                                context={
                                    "scale_discrepancy": scale_disc,
                                    "warning_threshold": self.scale_warning_threshold,
                                    "rejection_threshold": self.scale_rejection_threshold,
                                },
                            )
                        )
                    elif scale_disc > self.scale_warning_threshold:
                        issues.append(
                            QualityIssueDetail(
                                code=QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED,
                                severity=QualitySeverity.WARNING,
                                message=(
                                    f"Cross-view scale discrepancy is {scale_disc:.1%} (exceeds warning threshold {self.scale_warning_threshold:.1%}). "
                                    f"Front scale={calibration.front.scale_cm_per_pixel:.5f} cm/px, Side scale={calibration.side.scale_cm_per_pixel:.5f} cm/px."
                                ),
                                actionable_fix="Ensure camera distance, optical zoom, and subject floor position are strictly identical between front and side captures.",
                                context={
                                    "scale_discrepancy": scale_disc,
                                    "warning_threshold": self.scale_warning_threshold,
                                    "rejection_threshold": self.scale_rejection_threshold,
                                },
                            )
                        )

        # 6. Cross-View Slice Alignment Diagnostics
        if measurement_result is not None:
            # Check if any measurement warnings report height discrepancies or unverified alignment
            for lvl_name, m in measurement_result.measurements.items():
                for w in m.warnings:
                    if "height discrepancy" in w.lower():
                        issues.append(
                            QualityIssueDetail(
                                code=QualityIssueCode.CROSS_VIEW_ALIGNMENT_REJECTED,
                                severity=QualitySeverity.WARNING,
                                message=f"[{lvl_name.upper()}] {w}",
                                actionable_fix="Ensure consistent standing posture without sagittal pitch between views.",
                            )
                        )
                        break

        # Overall Status Determination
        has_critical = any(i.severity == QualitySeverity.CRITICAL for i in issues)
        has_warning = any(i.severity == QualitySeverity.WARNING for i in issues)

        if has_critical:
            overall_status = QualityStatus.REJECT
        elif has_warning:
            overall_status = QualityStatus.WARNING
        else:
            overall_status = QualityStatus.PASS

        return CaptureQualityReport(
            status=overall_status,
            issues=issues,
            front_framing=f_framing,
            side_framing=s_framing,
            scale_discrepancy_fraction=scale_disc,
            discrepancy_threshold_fraction=self.scale_warning_threshold,
            scale_rejection_threshold_fraction=self.scale_rejection_threshold,
            min_margin_threshold_fraction=self.min_margin_threshold,
        )
