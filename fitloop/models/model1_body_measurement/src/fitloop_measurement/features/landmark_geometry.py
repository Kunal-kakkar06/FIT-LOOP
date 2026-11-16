"""
features/landmark_geometry.py
-----------------------------
Landmark geometry utilities for converting normalized MediaPipe coordinates
into image pixel space and computing inter-landmark geometric relations.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ..config import LANDMARK_VISIBILITY_THRESHOLD
from ..pose.schemas import PoseExtractionResult, PoseLandmarkPoint
from .schemas import PixelPoint


def convert_landmarks_to_pixels(
    pose_result: PoseExtractionResult,
    min_visibility: float = LANDMARK_VISIBILITY_THRESHOLD,
) -> tuple[List[PixelPoint], list[str]]:
    """Convert normalized landmarks to image-pixel coordinates with validity checks.

    Parameters
    ----------
    pose_result:
        PoseExtractionResult from Stage 3A.
    min_visibility:
        Threshold below which a landmark is considered invalid for geometry.

    Returns
    -------
    tuple[List[PixelPoint], list[str]]
        List of 33 PixelPoint objects and diagnostic warnings.
    """
    warnings: list[str] = []
    w = pose_result.image_width
    h = pose_result.image_height

    if w <= 0 or h <= 0:
        return [], ["Invalid image dimensions for landmark pixel projection."]

    pixel_points: List[PixelPoint] = []

    for lm in pose_result.landmarks:
        # Check basic coordinate existence
        if lm.x is None or lm.y is None:
            pixel_points.append(
                PixelPoint(
                    index=lm.index,
                    name=lm.name,
                    pixel_x=None,
                    pixel_y=None,
                    relative_z=lm.z,
                    visibility=lm.visibility,
                    presence=lm.presence,
                    is_valid=False,
                )
            )
            continue

        # Visibility / presence check
        vis_ok = lm.visibility is None or lm.visibility >= min_visibility
        pres_ok = lm.presence is None or lm.presence >= min_visibility

        # Discrete pixel coordinate conversion: x_px = x_norm * (W - 1), y_px = y_norm * (H - 1)
        px = lm.x * (w - 1)
        py = lm.y * (h - 1)

        # In-bounds validation [0, W - 1] and [0, H - 1]
        in_bounds = 0.0 <= px <= (w - 1) and 0.0 <= py <= (h - 1)
        if not in_bounds:
            warnings.append(
                f"Landmark {lm.name} ({lm.index}) projected outside image bounds ({px:.1f}, {py:.1f})."
            )

        is_valid = vis_ok and pres_ok and in_bounds

        pixel_points.append(
            PixelPoint(
                index=lm.index,
                name=lm.name,
                pixel_x=px,
                pixel_y=py,
                relative_z=lm.z,
                visibility=lm.visibility,
                presence=lm.presence,
                is_valid=is_valid,
            )
        )

    return pixel_points, warnings


def landmark_pixel_distance(p1: PixelPoint, p2: PixelPoint) -> Optional[float]:
    """Calculate Euclidean 2D pixel distance between two valid landmarks."""
    if not (p1.is_valid and p2.is_valid):
        return None
    if p1.pixel_x is None or p1.pixel_y is None or p2.pixel_x is None or p2.pixel_y is None:
        return None
    dx = p1.pixel_x - p2.pixel_x
    dy = p1.pixel_y - p2.pixel_y
    return math.hypot(dx, dy)


def landmark_pixel_midpoint(p1: PixelPoint, p2: PixelPoint) -> Optional[Tuple[float, float]]:
    """Calculate 2D midpoint between two valid landmarks in pixel space."""
    if not (p1.is_valid and p2.is_valid):
        return None
    if p1.pixel_x is None or p1.pixel_y is None or p2.pixel_x is None or p2.pixel_y is None:
        return None
    return ((p1.pixel_x + p2.pixel_x) / 2.0, (p1.pixel_y + p2.pixel_y) / 2.0)


def landmark_horizontal_span(p1: PixelPoint, p2: PixelPoint) -> Optional[float]:
    """Calculate horizontal pixel distance abs(x1 - x2)."""
    if not (p1.is_valid and p2.is_valid):
        return None
    if p1.pixel_x is None or p2.pixel_x is None:
        return None
    return abs(p1.pixel_x - p2.pixel_x)


def landmark_vertical_span(p1: PixelPoint, p2: PixelPoint) -> Optional[float]:
    """Calculate vertical pixel distance abs(y1 - y2)."""
    if not (p1.is_valid and p2.is_valid):
        return None
    if p1.pixel_y is None or p2.pixel_y is None:
        return None
    return abs(p1.pixel_y - p2.pixel_y)
