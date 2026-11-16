# fitloop_measurement/segmentation/__init__.py
"""
segmentation sub-package
------------------------
Silhouette mask segmentation, contour extraction, and visualization for FitLoop Model 1.
"""

from .contours import extract_principal_contour
from .schemas import (
    BoundingBox,
    ContourResult,
    SegmentationResult,
    SegmentationStatus,
)
from .segmenter import BodySegmenter, clean_binary_mask
from .visualization import draw_segmentation_overlay

__all__ = [
    "BodySegmenter",
    "clean_binary_mask",
    "extract_principal_contour",
    "draw_segmentation_overlay",
    "SegmentationResult",
    "SegmentationStatus",
    "ContourResult",
    "BoundingBox",
]
