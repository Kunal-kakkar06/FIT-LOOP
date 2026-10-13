"""
segmentation/contours.py
-------------------------
Contour extraction and geometric evaluation from binary foreground masks.

Extracts external contours using OpenCV, identifies the principal body contour,
and calculates bounding parameters without modifying the input mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schemas import BoundingBox, ContourResult


def extract_principal_contour(
    binary_mask: np.ndarray,
    min_area_pixels: float = 1000.0,
) -> tuple[ContourResult | None, list[str]]:
    """Extract external contours and determine the principal body contour.

    Parameters
    ----------
    binary_mask:
        Binary mask (uint8) where foreground is 255 and background is 0.
    min_area_pixels:
        Minimum contour area threshold below which contours are ignored.

    Returns
    -------
    tuple[ContourResult | None, list[str]]
        Extracted principal contour (or None) and any associated warnings.
    """
    warnings: list[str] = []

    if binary_mask is None or binary_mask.size == 0:
        return None, ["Binary mask is empty or None."]

    if len(binary_mask.shape) != 2:
        return None, [f"Binary mask must be 2D (H, W), got shape: {binary_mask.shape}"]

    # cv2.findContours expects uint8
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        binary_mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, ["No external contours found in binary mask."]

    # Filter out tiny noise contours
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_pixels]

    if not valid_contours:
        warnings.append(
            f"All detected contours were smaller than minimum area threshold ({min_area_pixels} px)."
        )
        # Fall back to largest detected contour if any exist
        principal = max(contours, key=cv2.contourArea)
    else:
        # Principal body contour is the one with largest pixel area
        principal = max(valid_contours, key=cv2.contourArea)
        if len(valid_contours) > 1:
            warnings.append(
                f"Multiple significant components found ({len(valid_contours)}). "
                "Selected largest contour as principal body silhouette."
            )

    area = float(cv2.contourArea(principal))
    perimeter = float(cv2.arcLength(principal, closed=True))
    x, y, w, h = cv2.boundingRect(principal)

    result = ContourResult(
        contour_points=principal,
        area=area,
        perimeter=perimeter,
        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
    )

    return result, warnings
