"""
features/silhouette_geometry.py
-------------------------------
Silhouette geometry utilities: horizontal scanline widths, width profiles,
boundary detection, and body-relative vertical normalization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..segmentation.schemas import SegmentationResult
from .schemas import ScanlineSlice


def sample_scanline_width(
    binary_mask: np.ndarray,
    image_y: int,
    silhouette_top_y: Optional[int] = None,
    silhouette_height: Optional[int] = None,
) -> ScanlineSlice:
    """Measure the horizontal silhouette width and boundary coordinates at a single scanline.

    Parameters
    ----------
    binary_mask:
        2D binary mask array (uint8) where foreground pixels are > 0.
    image_y:
        Integer vertical pixel row index.
    silhouette_top_y:
        Top pixel row of the overall bounding box (for relative_y calculation).
    silhouette_height:
        Height of the overall bounding box (for relative_y calculation).

    Returns
    -------
    ScanlineSlice
        Geometric boundary description for row image_y.
    """
    h, w = binary_mask.shape[:2]

    # Calculate relative_y if bounding bounds are provided
    rel_y: Optional[float] = None
    if silhouette_top_y is not None and silhouette_height is not None and silhouette_height > 0:
        rel_y = (image_y - silhouette_top_y) / float(silhouette_height)

    # In-bounds check
    if not (0 <= image_y < h):
        return ScanlineSlice(
            image_y=image_y,
            relative_y=rel_y,
            left_x=None,
            right_x=None,
            width_pixels=None,
            is_valid=False,
            segment_count=0,
        )

    row = binary_mask[image_y, :]
    fg_indices = np.where(row > 0)[0]

    if len(fg_indices) == 0:
        return ScanlineSlice(
            image_y=image_y,
            relative_y=rel_y,
            left_x=None,
            right_x=None,
            width_pixels=0,
            is_valid=False,
            segment_count=0,
        )

    left_x = int(fg_indices[0])
    right_x = int(fg_indices[-1])
    width_pixels = right_x - left_x + 1

    # Count disconnected segments along this scanline
    # Changes between 0 and >0
    diffs = np.diff((row > 0).astype(int))
    # Number of transitions from 0 to 1
    segment_count = int(np.count_nonzero(diffs == 1))
    if row[0] > 0:
        segment_count += 1

    return ScanlineSlice(
        image_y=image_y,
        relative_y=rel_y,
        left_x=left_x,
        right_x=right_x,
        width_pixels=width_pixels,
        is_valid=True,
        segment_count=segment_count,
    )


def extract_vertical_width_profile(
    binary_mask: np.ndarray,
    top_y: int,
    bottom_y: int,
    num_samples: int = 50,
) -> tuple[List[ScanlineSlice], list[str]]:
    """Sample the silhouette width across vertical intervals between top_y and bottom_y.

    Parameters
    ----------
    binary_mask:
        Binary mask array.
    top_y:
        Top pixel boundary of the profile sampling region.
    bottom_y:
        Bottom pixel boundary of the profile sampling region.
    num_samples:
        Number of horizontal slices to evaluate.

    Returns
    -------
    tuple[List[ScanlineSlice], list[str]]
        List of ScanlineSlice samples and diagnostic warnings.
    """
    warnings: list[str] = []
    h = binary_mask.shape[0]

    if bottom_y <= top_y or num_samples <= 0:
        return [], ["Invalid vertical range or num_samples for width profile extraction."]

    total_span = bottom_y - top_y
    y_positions = np.linspace(top_y, bottom_y - 1, num=num_samples, dtype=int)

    slices: List[ScanlineSlice] = []
    discontinuity_count = 0

    for y in y_positions:
        s = sample_scanline_width(
            binary_mask=binary_mask,
            image_y=int(y),
            silhouette_top_y=top_y,
            silhouette_height=total_span,
        )
        slices.append(s)
        if s.segment_count > 1:
            discontinuity_count += 1

    if discontinuity_count > num_samples * 0.3:
        warnings.append(
            f"High number of disconnected scanlines ({discontinuity_count}/{num_samples}). "
            "Limb separation or occlusion detected."
        )

    return slices, warnings
