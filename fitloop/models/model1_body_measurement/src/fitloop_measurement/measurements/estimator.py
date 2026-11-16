"""
measurements/estimator.py
-------------------------
Circumference measurement estimation engine coordinating calibration,
anatomical slice selection, and Ramanujan elliptical perimeter calculations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..features.schemas import PairedGeometricFeatures, ViewGeometricFeatures
from .calibration import calibrate_dual_views
from .circumference import estimate_circumference_at_level
from .schemas import (
    AnatomicalLevel,
    CircumferenceMeasurement,
    DualViewCalibration,
    MeasurementEngineResult,
    MeasurementStatus,
)
from .slice_selector import select_anatomical_slice

logger = logging.getLogger(__name__)


class CircumferenceEstimator:
    """Estimates body circumferences using calibrated front and side multi-view slices."""

    SUPPORTED_LEVELS: list[AnatomicalLevel] = [
        AnatomicalLevel.CHEST,
        AnatomicalLevel.WAIST,
        AnatomicalLevel.ABDOMEN,
        AnatomicalLevel.HIPS,
    ]

    def __init__(self, target_levels: Optional[List[AnatomicalLevel]] = None) -> None:
        self.target_levels = target_levels or self.SUPPORTED_LEVELS

    def estimate_measurements(
        self,
        paired_features: PairedGeometricFeatures,
        known_height_cm: Optional[float] = None,
    ) -> MeasurementEngineResult:
        """Compute circumferences across targeted anatomical levels.

        Parameters
        ----------
        paired_features:
            Stage 4A PairedGeometricFeatures container.
        known_height_cm:
            Standing height in cm (if not already recorded in paired_features).

        Returns
        -------
        MeasurementEngineResult
            Consolidated measurement result across all supported levels.
        """
        # Determine height: caller argument or paired features record
        height_cm = known_height_cm if known_height_cm is not None else paired_features.expected_height_cm

        # 1. Metric calibration
        calibration = calibrate_dual_views(
            known_height_cm=height_cm,
            front_features=paired_features.front,
            side_features=paired_features.side,
        )

        all_warnings: list[str] = list(calibration.warnings)
        all_warnings.extend(paired_features.warnings)

        measurements: Dict[str, CircumferenceMeasurement] = {}

        if not calibration.is_valid:
            # Uncalibrated baseline
            for lvl in self.target_levels:
                measurements[lvl.value] = CircumferenceMeasurement(
                    level=lvl,
                    circumference_cm=None,
                    status=MeasurementStatus.UNCALIBRATED,
                    front_width_px=None,
                    side_depth_px=None,
                    front_scale_cm_per_pixel=None,
                    side_scale_cm_per_pixel=None,
                    semi_major_cm=None,
                    semi_minor_cm=None,
                    warnings=["Metric calibration failed or known standing height is invalid."],
                )
            return MeasurementEngineResult(
                status=MeasurementStatus.UNCALIBRATED,
                calibration=calibration,
                measurements=measurements,
                warnings=all_warnings,
            )

        # 2. Iterate each anatomical level
        for lvl in self.target_levels:
            f_slice, f_result = select_anatomical_slice(lvl, paired_features.front)
            s_slice, s_result = select_anatomical_slice(lvl, paired_features.side)

            level_warnings: list[str] = []
            level_warnings.extend([f"[Front] {w}" for w in f_result.warnings])
            level_warnings.extend([f"[Side] {w}" for w in s_result.warnings])

            if f_slice is None or s_slice is None or not f_result.is_valid or not s_result.is_valid:
                measurements[lvl.value] = CircumferenceMeasurement(
                    level=lvl,
                    circumference_cm=None,
                    status=MeasurementStatus.ANATOMICAL_LEVEL_UNAVAILABLE,
                    front_width_px=f_slice.width_pixels if f_slice else None,
                    side_depth_px=s_slice.width_pixels if s_slice else None,
                    front_scale_cm_per_pixel=calibration.front.scale_cm_per_pixel,
                    side_scale_cm_per_pixel=calibration.side.scale_cm_per_pixel,
                    semi_major_cm=None,
                    semi_minor_cm=None,
                    warnings=level_warnings or ["Could not locate corresponding anatomical slice in both views."],
                )
                continue

            # Compute circumference at level
            m = estimate_circumference_at_level(
                level=lvl,
                front_width_px=f_slice.width_pixels,
                side_depth_px=s_slice.width_pixels,
                front_scale_cm_per_pixel=calibration.front.scale_cm_per_pixel,
                side_scale_cm_per_pixel=calibration.side.scale_cm_per_pixel,
                level_warnings=level_warnings,
            )
            measurements[lvl.value] = m

        # Overall status evaluation
        success_count = sum(1 for m in measurements.values() if m.status == MeasurementStatus.SUCCESS)
        if success_count == len(self.target_levels):
            overall_status = MeasurementStatus.SUCCESS
        elif any(m.circumference_cm is not None for m in measurements.values()):
            overall_status = MeasurementStatus.PARTIAL
        else:
            overall_status = MeasurementStatus.INVALID_INPUT

        return MeasurementEngineResult(
            status=overall_status,
            calibration=calibration,
            measurements=measurements,
            warnings=all_warnings,
        )
