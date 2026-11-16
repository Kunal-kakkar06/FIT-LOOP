"""
segmentation/schemas.py
-----------------------
Structured, JSON-compatible data schemas for body silhouette segmentation.

The binary mask is preserved as a NumPy uint8 array for computer vision operations,
while metadata and geometry diagnostics serialize safely into JSON structures.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SegmentationStatus(str, Enum):
    """Execution status of body silhouette segmentation."""

    SUCCESS = "SUCCESS"
    NO_FOREGROUND = "NO_FOREGROUND"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ERROR = "ERROR"


@dataclasses.dataclass
class BoundingBox:
    """Bounding box coordinates in image pixel space."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x_max(self) -> int:
        return self.x + self.width

    @property
    def y_max(self) -> int:
        return self.y + self.height

    @property
    def aspect_ratio(self) -> float:
        return float(self.width) / float(self.height) if self.height > 0 else 0.0

    def to_dict(self) -> Dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclasses.dataclass
class ContourResult:
    """Geometric contour representation of the principal body silhouette."""

    contour_points: np.ndarray  # Shape (N, 1, 2) in image pixel coordinates
    area: float
    perimeter: float
    bounding_box: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe metadata without dumping massive coordinate arrays."""
        return {
            "num_points": int(len(self.contour_points)),
            "area": round(float(self.area), 2),
            "perimeter": round(float(self.perimeter), 2),
            "bounding_box": self.bounding_box.to_dict(),
        }


@dataclasses.dataclass
class SegmentationResult:
    """Standard container for single-image silhouette segmentation."""

    view_type: str  # "front" or "side"
    image_width: int
    image_height: int
    status: SegmentationStatus
    model_id: str
    mask: Optional[np.ndarray] = None          # uint8 array {0, 255} with shape (H, W)
    raw_mask: Optional[np.ndarray] = None      # uint8 array before morphological cleanup
    foreground_ratio: float = 0.0             # Proportion of foreground pixels [0.0, 1.0]
    principal_contour: Optional[ContourResult] = None
    warnings: List[str] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return self.status == SegmentationStatus.SUCCESS and self.mask is not None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization (omits raw NumPy arrays)."""
        return {
            "view_type": self.view_type,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "status": self.status.value,
            "model_id": self.model_id,
            "has_mask": self.mask is not None,
            "foreground_ratio": round(float(self.foreground_ratio), 4),
            "principal_contour": (
                self.principal_contour.to_dict() if self.principal_contour is not None else None
            ),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
