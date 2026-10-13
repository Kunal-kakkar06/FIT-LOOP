"""
measurements/circumference.py
-----------------------------
Mathematical circumference estimation using Ramanujan's second ellipse perimeter formula.

Mathematical Foundation:
Given semi-major axis 'a' (half front width in cm) and semi-minor axis 'b' (half side depth in cm):
C ≈ π * [ 3(a + b) - sqrt((3a + b)(a + 3b)) ]

Pure mathematical logic decoupled from image inference and tested independently.
"""

from __future__ import annotations

import math
from typing import Optional

from .schemas import (
    AnatomicalLevel,
    CircumferenceMeasurement,
    MeasurementStatus,
)


def ramanujan_ellipse_perimeter(a: float, b: float) -> float:
    """Calculate the perimeter of an ellipse using Ramanujan's second approximation.

    Parameters
    ----------
    a:
        Semi-major axis in centimetres (strictly positive).
    b:
        Semi-minor axis in centimetres (strictly positive).

    Returns
    -------
    float
        Perimeter of the ellipse in centimetres.

    Raises
    ------
    ValueError
        If either axis is non-positive or non-finite.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"Axes must be finite numbers, got a={a}, b={b}")
    if a <= 0 or b <= 0:
        raise ValueError(f"Axes must be strictly positive (> 0), got a={a}, b={b}")

    # Ramanujan's Second Approximation:
    # C ≈ π * [ 3(a + b) - sqrt( (3a + b) * (a + 3b) ) ]
    term1 = 3.0 * (a + b)
    term2 = math.sqrt((3.0 * a + b) * (a + 3.0 * b))
    perimeter = math.pi * (term1 - term2)
    return perimeter


def estimate_circumference_at_level(
    level: AnatomicalLevel,
    front_width_px: Optional[int],
    side_depth_px: Optional[int],
    front_scale_cm_per_pixel: Optional[float],
    side_scale_cm_per_pixel: Optional[float],
    level_warnings: Optional[list[str]] = None,
) -> CircumferenceMeasurement:
    """Compute calibrated physical circumference for an anatomical level.

    Parameters
    ----------
    level:
        Target anatomical location (CHEST, WAIST, ABDOMEN, HIPS).
    front_width_px:
        Full horizontal width in pixels from front view.
    side_depth_px:
        Full horizontal depth in pixels from side view.
    front_scale_cm_per_pixel:
        Metric scale for front view (cm/px).
    side_scale_cm_per_pixel:
        Metric scale for side view (cm/px).
    level_warnings:
        Any warnings arising from slice selection or landmark positioning.

    Returns
    -------
    CircumferenceMeasurement
        Estimated circumference with geometric metadata and explicit caveats.
    """
    warnings: list[str] = list(level_warnings or [])

    # Validate inputs
    if (
        front_width_px is None
        or side_depth_px is None
        or front_scale_cm_per_pixel is None
        or side_scale_cm_per_pixel is None
    ):
        return CircumferenceMeasurement(
            level=level,
            circumference_cm=None,
            status=MeasurementStatus.INVALID_INPUT,
            front_width_px=front_width_px,
            side_depth_px=side_depth_px,
            front_scale_cm_per_pixel=front_scale_cm_per_pixel,
            side_scale_cm_per_pixel=side_scale_cm_per_pixel,
            semi_major_cm=None,
            semi_minor_cm=None,
            warnings=["One or more required widths or scales were missing (None)."],
        )

    if (
        front_width_px <= 0
        or side_depth_px <= 0
        or front_scale_cm_per_pixel <= 0
        or side_scale_cm_per_pixel <= 0
        or not math.isfinite(front_width_px)
        or not math.isfinite(side_depth_px)
        or not math.isfinite(front_scale_cm_per_pixel)
        or not math.isfinite(side_scale_cm_per_pixel)
    ):
        return CircumferenceMeasurement(
            level=level,
            circumference_cm=None,
            status=MeasurementStatus.INVALID_INPUT,
            front_width_px=front_width_px,
            side_depth_px=side_depth_px,
            front_scale_cm_per_pixel=front_scale_cm_per_pixel,
            side_scale_cm_per_pixel=side_scale_cm_per_pixel,
            semi_major_cm=None,
            semi_minor_cm=None,
            warnings=["Widths and scale factors must be strictly positive, non-zero finite numbers."],
        )

    # Physical metric dimensions: semi-major and semi-minor axes in cm
    a = (front_width_px * front_scale_cm_per_pixel) / 2.0
    b = (side_depth_px * side_scale_cm_per_pixel) / 2.0

    try:
        circ_cm = ramanujan_ellipse_perimeter(a, b)
    except Exception as e:
        return CircumferenceMeasurement(
            level=level,
            circumference_cm=None,
            status=MeasurementStatus.INVALID_INPUT,
            front_width_px=front_width_px,
            side_depth_px=side_depth_px,
            front_scale_cm_per_pixel=front_scale_cm_per_pixel,
            side_scale_cm_per_pixel=side_scale_cm_per_pixel,
            semi_major_cm=a,
            semi_minor_cm=b,
            warnings=[f"Failed to calculate Ramanujan perimeter: {e}"],
        )

    # Sanity checks on human body scale
    if circ_cm < 20.0 or circ_cm > 250.0:
        warnings.append(
            f"Calculated circumference ({circ_cm:.1f} cm) is outside typical adult range [20, 250] cm."
        )

    status = MeasurementStatus.PARTIAL if len(warnings) > 0 else MeasurementStatus.SUCCESS

    return CircumferenceMeasurement(
        level=level,
        circumference_cm=circ_cm,
        status=status,
        front_width_px=front_width_px,
        side_depth_px=side_depth_px,
        front_scale_cm_per_pixel=front_scale_cm_per_pixel,
        side_scale_cm_per_pixel=side_scale_cm_per_pixel,
        semi_major_cm=a,
        semi_minor_cm=b,
        warnings=warnings,
    )
