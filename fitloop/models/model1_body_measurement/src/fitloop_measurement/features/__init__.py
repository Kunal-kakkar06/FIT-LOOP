# fitloop_measurement/features/__init__.py
"""
features sub-package
--------------------
Geometric feature extraction combining pose landmarks and silhouette contours.
"""

from .extractor import GeometricFeatureExtractor
from .landmark_geometry import (
    convert_landmarks_to_pixels,
    landmark_horizontal_span,
    landmark_pixel_distance,
    landmark_pixel_midpoint,
    landmark_vertical_span,
)
from .schemas import (
    FeatureStatus,
    PairedGeometricFeatures,
    PixelPoint,
    ScanlineSlice,
    ViewGeometricFeatures,
)
from .silhouette_geometry import (
    extract_vertical_width_profile,
    sample_scanline_width,
)

__all__ = [
    "GeometricFeatureExtractor",
    "convert_landmarks_to_pixels",
    "landmark_pixel_distance",
    "landmark_pixel_midpoint",
    "landmark_horizontal_span",
    "landmark_vertical_span",
    "sample_scanline_width",
    "extract_vertical_width_profile",
    "FeatureStatus",
    "PixelPoint",
    "ScanlineSlice",
    "ViewGeometricFeatures",
    "PairedGeometricFeatures",
]
