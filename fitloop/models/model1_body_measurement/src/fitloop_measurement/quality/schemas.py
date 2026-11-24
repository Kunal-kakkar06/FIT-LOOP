"""
quality/schemas.py
------------------
Structured, machine-readable quality report schemas for FitLoop Model 1 (Stage 7G).
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional


class QualityStatus(str, Enum):
    """Overall capture-quality gate status."""

    PASS = "PASS"
    WARNING = "WARNING"
    REJECT = "REJECT"


class QualitySeverity(str, Enum):
    """Severity of a detected quality issue."""

    CRITICAL = "CRITICAL"  # Triggers REJECT
    WARNING = "WARNING"    # Triggers WARNING


class QualityIssueCode(str, Enum):
    """Machine-readable quality failure codes."""

    # Pre-inference / input errors
    MISSING_IMAGE = "MISSING_IMAGE"
    UNREADABLE_IMAGE = "UNREADABLE_IMAGE"
    INVALID_IMAGE_TYPE = "INVALID_IMAGE_TYPE"
    INVALID_HEIGHT = "INVALID_HEIGHT"
    RESOLUTION_BELOW_MINIMUM = "RESOLUTION_BELOW_MINIMUM"

    # Geometric & Framing checks
    IMAGE_DIMENSION_MISMATCH = "IMAGE_DIMENSION_MISMATCH"
    ASPECT_RATIO_MISMATCH = "ASPECT_RATIO_MISMATCH"
    INSUFFICIENT_VERTICAL_MARGIN = "INSUFFICIENT_VERTICAL_MARGIN"
    SILHOUETTE_CLIPPED = "SILHOUETTE_CLIPPED"

    # Silhouette & Detection checks
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
    SILHOUETTE_SEGMENTATION_FAILED = "SILHOUETTE_SEGMENTATION_FAILED"
    IMPLAUSIBLE_SILHOUETTE_HEIGHT = "IMPLAUSIBLE_SILHOUETTE_HEIGHT"
    SILHOUETTE_HEIGHT_DISCREPANCY = "SILHOUETTE_HEIGHT_DISCREPANCY"

    # Calibration & Multi-view coherence
    SCALE_DISCREPANCY_EXCEEDED = "SCALE_DISCREPANCY_EXCEEDED"
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"

    # Anatomical alignment
    CROSS_VIEW_ALIGNMENT_UNVERIFIED = "CROSS_VIEW_ALIGNMENT_UNVERIFIED"
    CROSS_VIEW_ALIGNMENT_REJECTED = "CROSS_VIEW_ALIGNMENT_REJECTED"


@dataclasses.dataclass(frozen=True)
class QualityIssueDetail:
    """Structured quality issue with code, severity, human explanation, and recommendation."""

    code: QualityIssueCode
    severity: QualitySeverity
    message: str
    actionable_fix: str
    context: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "actionable_fix": self.actionable_fix,
            "context": self.context,
        }


@dataclasses.dataclass(frozen=True)
class ViewFramingDiagnostics:
    """Detailed framing and boundary diagnostics for a single image view."""

    image_width: int
    image_height: int
    aspect_ratio: float
    silhouette_bbox_x: Optional[int]
    silhouette_bbox_y: Optional[int]
    silhouette_width: Optional[int]
    silhouette_height: Optional[int]
    top_margin_fraction: Optional[float]
    bottom_margin_fraction: Optional[float]
    scale_cm_per_pixel: Optional[float]
    is_clipped: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "silhouette_bbox": (
                {
                    "x": self.silhouette_bbox_x,
                    "y": self.silhouette_bbox_y,
                    "width": self.silhouette_width,
                    "height": self.silhouette_height,
                }
                if self.silhouette_width is not None
                else None
            ),
            "top_margin_fraction": (
                round(self.top_margin_fraction, 4)
                if self.top_margin_fraction is not None
                else None
            ),
            "bottom_margin_fraction": (
                round(self.bottom_margin_fraction, 4)
                if self.bottom_margin_fraction is not None
                else None
            ),
            "scale_cm_per_pixel": (
                round(self.scale_cm_per_pixel, 6)
                if self.scale_cm_per_pixel is not None
                else None
            ),
            "is_clipped": self.is_clipped,
        }


@dataclasses.dataclass
class CaptureQualityReport:
    """Comprehensive, structured quality report for paired front and side captures."""

    status: QualityStatus
    issues: List[QualityIssueDetail] = dataclasses.field(default_factory=list)
    front_framing: Optional[ViewFramingDiagnostics] = None
    side_framing: Optional[ViewFramingDiagnostics] = None
    scale_discrepancy_fraction: Optional[float] = None
    discrepancy_threshold_fraction: float = 0.05
    scale_rejection_threshold_fraction: float = 0.15
    min_margin_threshold_fraction: float = 0.05
    assumptions_and_disclaimers: List[str] = dataclasses.field(
        default_factory=lambda: [
            "A quality gate PASS verifies geometric and capture constraints, NOT anthropometric accuracy.",
            "Scale calibration assumes unclipped upright stature and pinhole camera perspective.",
            "Independent 2D calibration scales do not correct for pitch, roll, or 3D posture lean.",
        ]
    )

    @property
    def has_rejections(self) -> bool:
        return any(i.severity == QualitySeverity.CRITICAL for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == QualitySeverity.WARNING for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "has_rejections": self.has_rejections,
            "has_warnings": self.has_warnings,
            "scale_discrepancy_fraction": (
                round(self.scale_discrepancy_fraction, 4)
                if self.scale_discrepancy_fraction is not None
                else None
            ),
            "discrepancy_threshold_fraction": self.discrepancy_threshold_fraction,
            "scale_rejection_threshold_fraction": self.scale_rejection_threshold_fraction,
            "min_margin_threshold_fraction": self.min_margin_threshold_fraction,
            "issues": [i.to_dict() for i in self.issues],
            "front_framing": self.front_framing.to_dict() if self.front_framing else None,
            "side_framing": self.side_framing.to_dict() if self.side_framing else None,
            "assumptions_and_disclaimers": list(self.assumptions_and_disclaimers),
        }
