"""
features/schemas.py
-------------------
Structured, JSON-compatible data schemas for deterministic geometric features.

Defines:
- PixelPoint: Pixel coordinates with landmark metadata and validity.
- ScanlineSlice: Horizontal silhouette boundaries and pixel width at a specific y-level.
- ViewGeometricFeatures: Landmark and silhouette features extracted from a single view.
- PairedGeometricFeatures: Joint representation aligning front and side views.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional


class FeatureStatus(str, Enum):
    """Extraction status for geometric features."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclasses.dataclass(frozen=True)
class PixelPoint:
    """Anatomical landmark projected into 2D image pixel space."""

    index: int
    name: str
    pixel_x: Optional[float]
    pixel_y: Optional[float]
    relative_z: Optional[float]  # Uncalibrated relative MediaPipe z-depth
    visibility: Optional[float]
    presence: Optional[float]
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "pixel_x": round(self.pixel_x, 2) if self.pixel_x is not None else None,
            "pixel_y": round(self.pixel_y, 2) if self.pixel_y is not None else None,
            "relative_z": round(self.relative_z, 4) if self.relative_z is not None else None,
            "visibility": round(self.visibility, 4) if self.visibility is not None else None,
            "presence": round(self.presence, 4) if self.presence is not None else None,
            "is_valid": self.is_valid,
        }


@dataclasses.dataclass(frozen=True)
class ScanlineSlice:
    """Horizontal slice through silhouette mask at a specific vertical scanline."""

    image_y: int
    relative_y: Optional[float]  # [0.0, 1.0] relative to visible silhouette height
    left_x: Optional[int]
    right_x: Optional[int]
    width_pixels: Optional[int]
    is_valid: bool
    segment_count: int = 0  # Number of disconnected foreground segments along this scanline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_y": self.image_y,
            "relative_y": round(self.relative_y, 4) if self.relative_y is not None else None,
            "left_x": self.left_x,
            "right_x": self.right_x,
            "width_pixels": self.width_pixels,
            "is_valid": self.is_valid,
            "segment_count": self.segment_count,
        }


@dataclasses.dataclass
class ViewGeometricFeatures:
    """Comprehensive geometric features extracted from a single perspective."""

    view_type: str  # "front" or "side"
    image_width: int
    image_height: int
    status: FeatureStatus
    pixel_landmarks: List[PixelPoint] = dataclasses.field(default_factory=list)
    silhouette_bbox_x: Optional[int] = None
    silhouette_bbox_y: Optional[int] = None
    silhouette_width: Optional[int] = None
    silhouette_height: Optional[int] = None
    silhouette_area: Optional[float] = None
    foreground_ratio: Optional[float] = None
    # Key anatomical heights in body-relative coordinates [0.0 = top of head, 1.0 = bottom of feet]
    anatomical_levels: Dict[str, Optional[float]] = dataclasses.field(default_factory=dict)
    # Sampled horizontal silhouette width profile
    width_profile: List[ScanlineSlice] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)

    def get_landmark_by_name(self, name: str) -> Optional[PixelPoint]:
        for p in self.pixel_landmarks:
            if p.name == name:
                return p
        return None

    def get_landmark_by_index(self, index: int) -> Optional[PixelPoint]:
        if 0 <= index < len(self.pixel_landmarks):
            return self.pixel_landmarks[index]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary representation."""
        return {
            "view_type": self.view_type,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "status": self.status.value,
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
            "silhouette_area": self.silhouette_area,
            "foreground_ratio": round(self.foreground_ratio, 4) if self.foreground_ratio is not None else None,
            "anatomical_levels": {
                k: round(v, 4) if v is not None else None
                for k, v in self.anatomical_levels.items()
            },
            "num_landmarks": len(self.pixel_landmarks),
            "valid_landmarks_count": sum(1 for lm in self.pixel_landmarks if lm.is_valid),
            "width_profile_samples": len(self.width_profile),
            "width_profile": [s.to_dict() for s in self.width_profile],
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass
class PairedGeometricFeatures:
    """Joint representation aligning front and side geometric features."""

    status: FeatureStatus
    front: ViewGeometricFeatures
    side: ViewGeometricFeatures
    height_ratio_side_to_front: Optional[float] = None
    vertical_alignment_warning: Optional[str] = None
    expected_height_cm: Optional[float] = None  # User input from Stage 1/2 if provided
    warnings: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "height_ratio_side_to_front": (
                round(self.height_ratio_side_to_front, 4)
                if self.height_ratio_side_to_front is not None
                else None
            ),
            "vertical_alignment_warning": self.vertical_alignment_warning,
            "expected_height_cm": self.expected_height_cm,
            "front": self.front.to_dict(),
            "side": self.side.to_dict(),
            "warnings": list(self.warnings),
        }
