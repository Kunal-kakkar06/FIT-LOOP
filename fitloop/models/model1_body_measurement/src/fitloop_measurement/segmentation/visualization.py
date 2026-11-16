"""
segmentation/visualization.py
-----------------------------
Overlay utilities for segmentation masks, contours, and bounding boxes.
Does not mutate the caller's input array or save files automatically.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schemas import SegmentationResult


def draw_segmentation_overlay(
    bgr_image: np.ndarray,
    segmentation_result: SegmentationResult,
    mask_color: tuple[int, int, int] = (0, 200, 255),       # BGR Amber/Yellow
    contour_color: tuple[int, int, int] = (0, 255, 0),     # BGR Green
    bbox_color: tuple[int, int, int] = (255, 0, 0),        # BGR Blue
    alpha: float = 0.35,
    contour_thickness: int = 2,
    bbox_thickness: int = 2,
) -> np.ndarray:
    """Blend the foreground segmentation mask, principal contour, and bounding box onto the image.

    Parameters
    ----------
    bgr_image:
        Input BGR image array.
    segmentation_result:
        SegmentationResult object.
    mask_color:
        BGR tuple for tinted silhouette overlay.
    contour_color:
        BGR tuple for principal body contour boundary.
    bbox_color:
        BGR tuple for bounding box rectangle.
    alpha:
        Blending transparency factor for mask tint [0.0 = invisible, 1.0 = solid].
    contour_thickness:
        Pixel thickness of contour line.
    bbox_thickness:
        Pixel thickness of bounding box line.

    Returns
    -------
    np.ndarray
        New BGR image with overlays applied (input is never mutated).
    """
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Cannot draw overlay on empty or None image.")

    # Guard: never mutate the caller's image
    annotated = bgr_image.copy()

    if not segmentation_result.is_successful or segmentation_result.mask is None:
        return annotated

    mask = segmentation_result.mask
    h, w = annotated.shape[:2]

    # Ensure mask dimensions align with image
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 1. Semi-transparent colored overlay on foreground pixels
    fg_indices = mask > 0
    if np.any(fg_indices):
        color_layer = np.zeros_like(annotated)
        color_layer[:] = mask_color

        blended = cv2.addWeighted(annotated, 1.0 - alpha, color_layer, alpha, 0)
        annotated[fg_indices] = blended[fg_indices]

    # 2. Draw principal contour
    if segmentation_result.principal_contour is not None:
        contour_pts = segmentation_result.principal_contour.contour_points
        cv2.drawContours(
            annotated,
            [contour_pts],
            contourIdx=-1,
            color=contour_color,
            thickness=contour_thickness,
            lineType=cv2.LINE_AA,
        )

        # 3. Draw bounding box
        bbox = segmentation_result.principal_contour.bounding_box
        pt1 = (bbox.x, bbox.y)
        pt2 = (bbox.x + bbox.width, bbox.y + bbox.height)
        cv2.rectangle(annotated, pt1, pt2, bbox_color, bbox_thickness, cv2.LINE_AA)

    return annotated
