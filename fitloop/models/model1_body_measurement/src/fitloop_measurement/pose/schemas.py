"""
pose/schemas.py
---------------
Deterministic, JSON-compatible data schemas for MediaPipe Pose Landmark extraction.

Coordinate Conventions:
- x, y: Normalized image coordinates [0.0, 1.0] where (0,0) is top-left.
- z: Landmark depth relative to hips center midpoint.
- visibility: Likelihood [0.0, 1.0] that the landmark is within the image frame and not occluded.
- presence: Likelihood [0.0, 1.0] that the landmark exists in the image frame.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional


class ViewType(str, Enum):
    """Perspective of the body capture."""

    FRONT = "front"
    SIDE = "side"


class ExtractionStatus(str, Enum):
    """Overall status of pose landmark extraction."""

    SUCCESS = "SUCCESS"
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ERROR = "ERROR"


# Canonical MediaPipe 33 Landmark Names in sequential index order (0-32)
CANONICAL_POSE_LANDMARK_NAMES: tuple[str, ...] = (
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)


@dataclasses.dataclass(frozen=True)
class PoseLandmarkPoint:
    """Individual anatomical landmark point with coordinates and confidence scores."""

    index: int
    name: str
    x: Optional[float]
    y: Optional[float]
    z: Optional[float] = None
    visibility: Optional[float] = None
    presence: Optional[float] = None

    def is_usable(self, min_visibility: float = 0.5) -> bool:
        """Return True if landmark has valid normalized coordinates and sufficient visibility."""
        if self.x is None or self.y is None:
            return False
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            return False
        if self.visibility is not None and self.visibility < min_visibility:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "x": round(self.x, 6) if self.x is not None else None,
            "y": round(self.y, 6) if self.y is not None else None,
            "z": round(self.z, 6) if self.z is not None else None,
            "visibility": round(self.visibility, 4) if self.visibility is not None else None,
            "presence": round(self.presence, 4) if self.presence is not None else None,
        }


@dataclasses.dataclass
class PoseExtractionResult:
    """Deterministic result schema for single-image pose extraction."""

    view_type: str  # "front" or "side"
    image_width: int
    image_height: int
    model_id: str
    status: ExtractionStatus
    landmarks: List[PoseLandmarkPoint] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return self.status == ExtractionStatus.SUCCESS and len(self.landmarks) == 33

    def get_landmark_by_name(self, name: str) -> Optional[PoseLandmarkPoint]:
        """Find a landmark by its canonical anatomical name."""
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        return None

    def get_landmark_by_index(self, index: int) -> Optional[PoseLandmarkPoint]:
        """Find a landmark by its 0-32 canonical index."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire extraction result to a JSON-serializable structure."""
        return {
            "view_type": self.view_type,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_id": self.model_id,
            "status": self.status.value,
            "landmarks": [lm.to_dict() for lm in self.landmarks],
            "warnings": list(self.warnings),
        }
