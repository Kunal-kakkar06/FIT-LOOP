"""
measurements/schemas.py
-----------------------
Structured, JSON-compatible data schemas for Stage 4B Calibrated Circumference Estimation.

Defines:
- CalibrationStatus: Status of metric pixel-to-centimetre scaling.
- ViewCalibration: Per-view scale and bounding geometry.
- DualViewCalibration: Paired calibration container.
- AnatomicalLevel: Enum of supported body circumference locations.
- SliceSelectionResult: Per-view anatomical slice coordinates and selection method.
- CircumferenceMeasurement: Final estimated circumference with geometric metadata and warnings.
- MeasurementEngineResult: Consolidated multi-level measurement output.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional


class CalibrationStatus(str, Enum):
    """Execution status for metric scale calibration."""

    VALID = "VALID"
    INVALID_HEIGHT = "INVALID_HEIGHT"
    INVALID_SILHOUETTE = "INVALID_SILHOUETTE"
    CLIPPED_BODY = "CLIPPED_BODY"
    FAILED = "FAILED"


class MeasurementStatus(str, Enum):
    """Status of an individual circumference measurement."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    UNCALIBRATED = "UNCALIBRATED"
    ANATOMICAL_LEVEL_UNAVAILABLE = "ANATOMICAL_LEVEL_UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_LEVEL = "UNSUPPORTED_LEVEL"


class AnatomicalLevel(str, Enum):
    """Anatomical landmarks supported for circumference estimation."""

    CHEST = "chest"
    WAIST = "waist"
    ABDOMEN = "abdomen"
    HIPS = "hips"


@dataclasses.dataclass(frozen=True)
class ViewCalibration:
    """Scale calibration for a single perspective (front or side)."""

    view_type: str
    silhouette_height_px: int
    scale_cm_per_pixel: float
    is_valid: bool
    status: CalibrationStatus
    warnings: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_type": self.view_type,
            "silhouette_height_px": self.silhouette_height_px,
            "scale_cm_per_pixel": round(self.scale_cm_per_pixel, 6) if self.scale_cm_per_pixel > 0 else None,
            "is_valid": self.is_valid,
            "status": self.status.value,
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(frozen=True)
class DualViewCalibration:
    """Independent front and side metric calibrations derived from standing height."""

    known_height_cm: float
    front: ViewCalibration
    side: ViewCalibration
    is_valid: bool
    assumptions: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Assumes standing height represents visible vertical silhouette extent.",
            "Assumes pinhole camera model without non-linear lens distortion.",
            "Does not correct for out-of-plane body sway, pitch, or perspective divergence.",
            "Front and side scales are calculated independently and not averaged.",
        ]
    )
    warnings: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "known_height_cm": self.known_height_cm,
            "is_valid": self.is_valid,
            "front": self.front.to_dict(),
            "side": self.side.to_dict(),
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(frozen=True)
class SliceSelectionResult:
    """Anatomical slice location determined on a single view."""

    view_type: str
    level: AnatomicalLevel
    image_y: Optional[int]
    relative_y: Optional[float]
    selection_method: str
    reference_landmarks: List[str]
    is_valid: bool
    warnings: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_type": self.view_type,
            "level": self.level.value,
            "image_y": self.image_y,
            "relative_y": round(self.relative_y, 4) if self.relative_y is not None else None,
            "selection_method": self.selection_method,
            "reference_landmarks": list(self.reference_landmarks),
            "is_valid": self.is_valid,
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(frozen=True)
class CircumferenceMeasurement:
    """Estimated circumference at a specific anatomical level using Ramanujan ellipse approximation."""

    level: AnatomicalLevel
    circumference_cm: Optional[float]
    status: MeasurementStatus
    front_width_px: Optional[int]
    side_depth_px: Optional[int]
    front_scale_cm_per_pixel: Optional[float]
    side_scale_cm_per_pixel: Optional[float]
    semi_major_cm: Optional[float]  # a = (front_width_px * front_scale) / 2
    semi_minor_cm: Optional[float]  # b = (side_depth_px * side_scale) / 2
    approximation_method: str = "Ramanujan Second Ellipse Approximation"
    caveats: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Assumes planar elliptical cross-section; human cross-sections have non-elliptical asymmetry.",
            "Experimental geometric research baseline; not clinically or tailored validated.",
        ]
    )
    warnings: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "circumference_cm": round(self.circumference_cm, 2) if self.circumference_cm is not None else None,
            "status": self.status.value,
            "front_width_px": self.front_width_px,
            "side_depth_px": self.side_depth_px,
            "front_scale_cm_per_pixel": round(self.front_scale_cm_per_pixel, 6) if self.front_scale_cm_per_pixel else None,
            "side_scale_cm_per_pixel": round(self.side_scale_cm_per_pixel, 6) if self.side_scale_cm_per_pixel else None,
            "semi_major_cm": round(self.semi_major_cm, 2) if self.semi_major_cm is not None else None,
            "semi_minor_cm": round(self.semi_minor_cm, 2) if self.semi_minor_cm is not None else None,
            "approximation_method": self.approximation_method,
            "caveats": list(self.caveats),
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass
class MeasurementEngineResult:
    """Master output result across all evaluated anatomical levels."""

    status: MeasurementStatus
    calibration: DualViewCalibration
    measurements: Dict[str, CircumferenceMeasurement] = dataclasses.field(default_factory=dict)
    warnings: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "calibration": self.calibration.to_dict(),
            "measurements": {
                k: v.to_dict() for k, v in self.measurements.items()
            },
            "warnings": list(self.warnings),
        }
