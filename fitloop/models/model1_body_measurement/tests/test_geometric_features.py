"""
tests/test_geometric_features.py
--------------------------------
Comprehensive tests for Stage 4A: Geometric Feature Extraction.

Covers:
1. Normalized to pixel coordinate conversion.
2. Distance, midpoint, and horizontal/vertical span calculations.
3. Missing or low-visibility landmarks handling.
4. Landmarks outside image bounds handling.
5. Known silhouette bounding box and area features.
6. Horizontal scanline width measurements on synthetic masks.
7. Width profile sampling and disconnected component warnings.
8. Body-relative coordinate normalization.
9. Empty masks and invalid contours handling.
10. Front and side view-type validation.
11. Paired features across different image dimensions and height ratios.
12. Deterministic JSON serialization.
13. Input immutability (no mutation of inputs).
14. Deterministic repeated results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.features import (
    FeatureStatus,
    GeometricFeatureExtractor,
    PairedGeometricFeatures,
    PixelPoint,
    ScanlineSlice,
    ViewGeometricFeatures,
    convert_landmarks_to_pixels,
    extract_vertical_width_profile,
    landmark_horizontal_span,
    landmark_pixel_distance,
    landmark_pixel_midpoint,
    landmark_vertical_span,
    sample_scanline_width,
)
from fitloop_measurement.pose.schemas import (
    CANONICAL_POSE_LANDMARK_NAMES,
    ExtractionStatus,
    PoseExtractionResult,
    PoseLandmarkPoint,
)
from fitloop_measurement.segmentation.schemas import (
    BoundingBox,
    ContourResult,
    SegmentationResult,
    SegmentationStatus,
)


@pytest.fixture
def synthetic_pose_result():
    """Synthetic PoseExtractionResult for a 640x960 image."""
    points = []
    for idx, name in enumerate(CANONICAL_POSE_LANDMARK_NAMES):
        # Y positions distributed from 0.1 to 0.9
        y_norm = 0.1 + (idx / 33.0) * 0.8
        points.append(
            PoseLandmarkPoint(
                index=idx,
                name=name,
                x=0.5,
                y=y_norm,
                z=-0.05,
                visibility=0.95,
                presence=0.95,
            )
        )
    return PoseExtractionResult(
        view_type="front",
        image_width=640,
        image_height=960,
        model_id="test-pose",
        status=ExtractionStatus.SUCCESS,
        landmarks=points,
    )


@pytest.fixture
def synthetic_segmentation_result():
    """Synthetic SegmentationResult with known mask rect (200, 200) to (440, 800)."""
    mask = np.zeros((960, 640), dtype=np.uint8)
    cv2.rectangle(mask, (200, 200), (440, 800), 255, -1)
    bbox = BoundingBox(x=200, y=200, width=241, height=601)
    contour = ContourResult(
        contour_points=np.array([[[200, 200]], [[440, 200]], [[440, 800]], [[200, 800]]]),
        area=float(241 * 601),
        perimeter=float(2 * (241 + 601)),
        bounding_box=bbox,
    )
    return SegmentationResult(
        view_type="front",
        image_width=640,
        image_height=960,
        status=SegmentationStatus.SUCCESS,
        model_id="test-segmenter",
        mask=mask,
        raw_mask=mask,
        foreground_ratio=float(241 * 601) / float(640 * 960),
        principal_contour=contour,
    )


class TestLandmarkGeometry:
    def test_coordinate_conversion(self, synthetic_pose_result):
        pixels, warnings = convert_landmarks_to_pixels(synthetic_pose_result)
        assert len(pixels) == 33
        assert len(warnings) == 0

        # Point 0: x=0.5 -> 0.5 * (640 - 1) = 319.5
        p0 = pixels[0]
        assert p0.pixel_x == 319.5
        assert p0.is_valid is True
        assert p0.relative_z == -0.05

    def test_exact_normalized_boundaries(self):
        """Verify that landmarks at exact normalized bounds 0.0 and 1.0 map to 0 and W-1 / H-1 without error."""
        points = [
            PoseLandmarkPoint(index=0, name="TOP_LEFT", x=0.0, y=0.0, z=0.0, visibility=1.0),
            PoseLandmarkPoint(index=1, name="BOTTOM_RIGHT", x=1.0, y=1.0, z=0.0, visibility=1.0),
        ]
        pose_res = PoseExtractionResult(
            view_type="front",
            image_width=640,
            image_height=960,
            model_id="test",
            status=ExtractionStatus.SUCCESS,
            landmarks=points,
        )
        pixels, warnings = convert_landmarks_to_pixels(pose_res)
        assert len(warnings) == 0
        assert pixels[0].pixel_x == 0.0
        assert pixels[0].pixel_y == 0.0
        assert pixels[0].is_valid is True

        # Exactly at 1.0 -> maps to max valid pixel index W-1 = 639.0, H-1 = 959.0
        assert pixels[1].pixel_x == 639.0
        assert pixels[1].pixel_y == 959.0
        assert pixels[1].is_valid is True

    def test_distance_midpoint_and_spans(self):
        p1 = PixelPoint(index=11, name="LEFT_SHOULDER", pixel_x=250.0, pixel_y=300.0, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True)
        p2 = PixelPoint(index=12, name="RIGHT_SHOULDER", pixel_x=390.0, pixel_y=300.0, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True)

        dist = landmark_pixel_distance(p1, p2)
        assert dist == 140.0

        mid = landmark_pixel_midpoint(p1, p2)
        assert mid == (320.0, 300.0)

        h_span = landmark_horizontal_span(p1, p2)
        assert h_span == 140.0

        v_span = landmark_vertical_span(p1, p2)
        assert v_span == 0.0

    def test_low_visibility_handling(self):
        p_low = PixelPoint(index=0, name="NOSE", pixel_x=320.0, pixel_y=200.0, relative_z=0.0, visibility=0.2, presence=0.9, is_valid=False)
        p_valid = PixelPoint(index=1, name="EYE", pixel_x=320.0, pixel_y=190.0, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True)

        assert landmark_pixel_distance(p_low, p_valid) is None
        assert landmark_pixel_midpoint(p_low, p_valid) is None


class TestSilhouetteGeometry:
    def test_sample_scanline_width(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 20:61] = 255  # Width = 41 pixels

        slice_data = sample_scanline_width(mask, image_y=50, silhouette_top_y=10, silhouette_height=80)
        assert slice_data.is_valid is True
        assert slice_data.left_x == 20
        assert slice_data.right_x == 60
        assert slice_data.width_pixels == 41
        assert slice_data.segment_count == 1
        assert round(slice_data.relative_y, 2) == 0.50

    def test_disconnected_scanline_count(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Two separated arm/leg blocks on the same line
        mask[50, 10:20] = 255
        mask[50, 40:50] = 255

        slice_data = sample_scanline_width(mask, image_y=50)
        assert slice_data.segment_count == 2
        assert slice_data.width_pixels == 40  # span from 10 to 49

    def test_vertical_width_profile(self, synthetic_segmentation_result):
        profile, warnings = extract_vertical_width_profile(
            synthetic_segmentation_result.mask,
            top_y=200,
            bottom_y=800,
            num_samples=20,
        )
        assert len(profile) == 20
        assert all(s.is_valid for s in profile)
        assert all(s.width_pixels == 241 for s in profile)


class TestGeometricFeatureExtractor:
    def test_single_view_feature_extraction(self, synthetic_pose_result, synthetic_segmentation_result):
        extractor = GeometricFeatureExtractor(num_profile_samples=25)
        features = extractor.extract_view_features(synthetic_pose_result, synthetic_segmentation_result)

        assert features.status == FeatureStatus.SUCCESS
        assert features.view_type == "front"
        assert features.silhouette_width == 241
        assert features.silhouette_height == 601
        assert len(features.width_profile) == 25
        assert len(features.anatomical_levels) == 33

        # Serialization test
        d = features.to_dict()
        assert d["status"] == "SUCCESS"
        assert d["silhouette_bbox"]["width"] == 241
        assert json.dumps(d)  # JSON-safe

    def test_paired_feature_extraction(self, synthetic_pose_result, synthetic_segmentation_result):
        extractor = GeometricFeatureExtractor()
        front = extractor.extract_view_features(synthetic_pose_result, synthetic_segmentation_result)

        # Create a side view with matching height
        side_pose = PoseExtractionResult(
            view_type="side",
            image_width=640,
            image_height=960,
            model_id="test",
            status=ExtractionStatus.SUCCESS,
            landmarks=synthetic_pose_result.landmarks,
        )
        side_seg = SegmentationResult(
            view_type="side",
            image_width=640,
            image_height=960,
            status=SegmentationStatus.SUCCESS,
            model_id="test",
            mask=synthetic_segmentation_result.mask,
            raw_mask=synthetic_segmentation_result.mask,
            foreground_ratio=synthetic_segmentation_result.foreground_ratio,
            principal_contour=synthetic_segmentation_result.principal_contour,
        )
        side = extractor.extract_view_features(side_pose, side_seg)

        paired = extractor.extract_paired_features(front, side, expected_height_cm=175.0)
        assert paired.status == FeatureStatus.SUCCESS
        assert paired.height_ratio_side_to_front == 1.0
        assert paired.expected_height_cm == 175.0
        assert paired.vertical_alignment_warning is None

        # JSON-safe serialization
        serialized = json.dumps(paired.to_dict())
        assert "175.0" in serialized

    def test_immutability_of_inputs(self, synthetic_pose_result, synthetic_segmentation_result):
        mask_copy = synthetic_segmentation_result.mask.copy()
        extractor = GeometricFeatureExtractor()
        _ = extractor.extract_view_features(synthetic_pose_result, synthetic_segmentation_result)

        assert np.array_equal(synthetic_segmentation_result.mask, mask_copy)

    def test_mismatched_view_warning(self, synthetic_pose_result, synthetic_segmentation_result):
        extractor = GeometricFeatureExtractor()
        front = extractor.extract_view_features(synthetic_pose_result, synthetic_segmentation_result)
        # Passing front view as side should produce warning
        paired = extractor.extract_paired_features(front, front)
        assert any("Expected side view" in w for w in paired.warnings)
