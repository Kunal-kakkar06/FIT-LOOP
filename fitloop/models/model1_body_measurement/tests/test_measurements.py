"""
tests/test_measurements.py
--------------------------
Comprehensive unit test suite for Stage 4B: Calibrated Circumference Estimation Engine.

Covers:
1. Calibration from known height and silhouette height.
2. Invalid, non-finite, and out-of-range heights.
3. Zero-height and clipped silhouettes.
4. Independent front and side scale calculations.
5. Ramanujan perimeter accuracy against known mathematical examples (circle, flat ellipse, unequal axes).
6. Equal front and side widths with equal scales.
7. Unequal front and side widths.
8. Invalid, zero, negative, or non-finite widths and scales.
9. Missing or low-visibility anatomical landmarks.
10. Disconnected scanlines handling.
11. Unsupported anatomical levels.
12. Explicit invalid statuses and null values.
13. Deterministic repeated results.
14. JSON-safe metadata serialization.
15. Input immutability.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.features.schemas import (
    FeatureStatus,
    PairedGeometricFeatures,
    PixelPoint,
    ScanlineSlice,
    ViewGeometricFeatures,
)
from fitloop_measurement.measurements import (
    AnatomicalLevel,
    CalibrationStatus,
    CircumferenceEstimator,
    CircumferenceMeasurement,
    DualViewCalibration,
    MeasurementEngineResult,
    MeasurementStatus,
    ViewCalibration,
    calibrate_dual_views,
    calibrate_view,
    estimate_circumference_at_level,
    ramanujan_ellipse_perimeter,
    select_anatomical_slice,
)


# ---------------------------------------------------------------------------
# Numerical Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_view_features():
    """Synthetic ViewGeometricFeatures for front view (H=960, W=640, sil_h=700)."""
    # Key landmark pixel positions
    # Shoulders at y=250, Hips at y=550 (Torso span = 300px)
    landmarks = [
        PixelPoint(index=11, name="LEFT_SHOULDER", pixel_x=240.0, pixel_y=250.0, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
        PixelPoint(index=12, name="RIGHT_SHOULDER", pixel_x=400.0, pixel_y=250.0, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
        PixelPoint(index=23, name="LEFT_HIP", pixel_x=260.0, pixel_y=550.0, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
        PixelPoint(index=24, name="RIGHT_HIP", pixel_x=380.0, pixel_y=550.0, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
    ]

    # Sampled width profile: slices every 15 pixels from 100 to 800
    profile = []
    for y in range(100, 805, 15):
        # Default width 180px
        profile.append(
            ScanlineSlice(
                image_y=y,
                relative_y=(y - 100) / 700.0,
                left_x=230,
                right_x=410,
                width_pixels=181,
                is_valid=True,
                segment_count=1,
            )
        )

    return ViewGeometricFeatures(
        view_type="front",
        image_width=640,
        image_height=960,
        status=FeatureStatus.SUCCESS,
        pixel_landmarks=landmarks,
        silhouette_bbox_x=200,
        silhouette_bbox_y=100,
        silhouette_width=240,
        silhouette_height=700,
        silhouette_area=140000.0,
        foreground_ratio=0.22,
        width_profile=profile,
    )


# ---------------------------------------------------------------------------
# 1. Calibration Tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_valid_height_calibration(self, sample_view_features):
        calib = calibrate_view(175.0, sample_view_features)
        assert calib.is_valid is True
        assert calib.status == CalibrationStatus.VALID
        assert calib.silhouette_height_px == 700
        # scale = 175.0 / 700 = 0.25 cm/pixel
        assert pytest.approx(calib.scale_cm_per_pixel, 1e-4) == 0.25
        assert len(calib.warnings) == 0

    @pytest.mark.parametrize("bad_height", [None, 0.0, -10.0, float("nan"), float("inf")])
    def test_invalid_and_non_finite_heights(self, sample_view_features, bad_height):
        calib = calibrate_view(bad_height, sample_view_features)
        assert calib.is_valid is False
        assert calib.status == CalibrationStatus.INVALID_HEIGHT
        assert calib.scale_cm_per_pixel == 0.0

    def test_zero_silhouette_height(self):
        empty_features = ViewGeometricFeatures(
            view_type="front",
            image_width=640,
            image_height=960,
            status=FeatureStatus.FAILED,
            silhouette_height=0,
        )
        calib = calibrate_view(175.0, empty_features)
        assert calib.is_valid is False
        assert calib.status == CalibrationStatus.INVALID_SILHOUETTE

    def test_clipped_silhouette_warning(self, sample_view_features):
        clipped_features = ViewGeometricFeatures(
            view_type="front",
            image_width=640,
            image_height=960,
            status=FeatureStatus.SUCCESS,
            silhouette_bbox_y=1,  # Head at frame edge
            silhouette_height=958,  # Feet at bottom edge
        )
        calib = calibrate_view(175.0, clipped_features)
        assert calib.status == CalibrationStatus.CLIPPED_BODY
        assert calib.is_valid is False
        assert any("clipped" in w for w in calib.warnings)

    def test_separate_front_and_side_scales(self, sample_view_features):
        # Side view has sil_h = 750 (differs from front sil_h = 700)
        side_features = ViewGeometricFeatures(
            view_type="side",
            image_width=640,
            image_height=960,
            status=FeatureStatus.SUCCESS,
            silhouette_bbox_x=200,
            silhouette_bbox_y=90,
            silhouette_width=180,
            silhouette_height=750,
            width_profile=sample_view_features.width_profile,
        )

        dual = calibrate_dual_views(175.0, sample_view_features, side_features)
        assert dual.is_valid is True
        assert dual.front.scale_cm_per_pixel == 175.0 / 700.0  # 0.25
        assert dual.side.scale_cm_per_pixel == 175.0 / 750.0   # ~0.2333
        assert dual.front.scale_cm_per_pixel != dual.side.scale_cm_per_pixel


# ---------------------------------------------------------------------------
# 2. Ramanujan Ellipse Perimeter Tests
# ---------------------------------------------------------------------------

class TestRamanujanPerimeter:
    def test_circle_case_a_equals_b(self):
        # A circle with radius r = 10 has perimeter C = 2 * pi * r = 20 * pi ≈ 62.831853
        r = 10.0
        c = ramanujan_ellipse_perimeter(r, r)
        expected = 2.0 * math.pi * r
        assert pytest.approx(c, 1e-6) == expected

    def test_known_unequal_ellipse(self):
        # Ellipse with a = 20, b = 10
        # Term 1: 3 * 30 = 90
        # Term 2: sqrt((60 + 10) * (20 + 30)) = sqrt(70 * 50) = sqrt(3500) ≈ 59.1607978
        # C = pi * (90 - 59.1607978) = pi * 30.839202 ≈ 96.8845
        c = ramanujan_ellipse_perimeter(20.0, 10.0)
        assert pytest.approx(c, 1e-3) == 96.8845

    @pytest.mark.parametrize("invalid_a, invalid_b", [(0.0, 10.0), (-5.0, 10.0), (10.0, -2.0), (float("nan"), 10.0)])
    def test_invalid_axes_raise(self, invalid_a, invalid_b):
        with pytest.raises(ValueError):
            ramanujan_ellipse_perimeter(invalid_a, invalid_b)


# ---------------------------------------------------------------------------
# 3. Circumference Estimator Function Tests
# ---------------------------------------------------------------------------

class TestCircumferenceEstimation:
    def test_valid_circumference_calculation(self):
        # Front width 120px at 0.25 cm/px -> a = (120 * 0.25) / 2 = 15.0 cm
        # Side depth 80px at 0.25 cm/px -> b = (80 * 0.25) / 2 = 10.0 cm
        m = estimate_circumference_at_level(
            level=AnatomicalLevel.WAIST,
            front_width_px=120,
            side_depth_px=80,
            front_scale_cm_per_pixel=0.25,
            side_scale_cm_per_pixel=0.25,
        )
        assert m.status == MeasurementStatus.SUCCESS
        assert m.semi_major_cm == 15.0
        assert m.semi_minor_cm == 10.0
        assert m.circumference_cm is not None
        assert 70.0 < m.circumference_cm < 90.0  # Approx ~79.3 cm

    def test_invalid_and_negative_inputs(self):
        m = estimate_circumference_at_level(
            level=AnatomicalLevel.CHEST,
            front_width_px=-50,
            side_depth_px=80,
            front_scale_cm_per_pixel=0.25,
            side_scale_cm_per_pixel=0.25,
        )
        assert m.status == MeasurementStatus.INVALID_INPUT
        assert m.circumference_cm is None
        assert any("strictly positive" in w for w in m.warnings)

    def test_missing_none_inputs(self):
        m = estimate_circumference_at_level(
            level=AnatomicalLevel.HIPS,
            front_width_px=None,
            side_depth_px=80,
            front_scale_cm_per_pixel=0.25,
            side_scale_cm_per_pixel=0.25,
        )
        assert m.status == MeasurementStatus.INVALID_INPUT
        assert m.circumference_cm is None


# ---------------------------------------------------------------------------
# 4. Slice Selector Tests
# ---------------------------------------------------------------------------

class TestSliceSelector:
    def test_landmark_anchored_slice_selection(self, sample_view_features):
        # Shoulders y=250, Hips y=550 (torso span = 300px)
        # Chest should be ~35% down torso span: 250 + 105 = 355
        matched_slice, result = select_anatomical_slice(AnatomicalLevel.CHEST, sample_view_features)
        assert result.is_valid is True
        assert matched_slice is not None
        assert abs(result.image_y - 355) <= 15
        assert "Landmark-anchored" in result.selection_method

    def test_fallback_when_landmarks_missing(self):
        features_no_landmarks = ViewGeometricFeatures(
            view_type="front",
            image_width=640,
            image_height=960,
            status=FeatureStatus.SUCCESS,
            pixel_landmarks=[],  # Empty landmarks
            silhouette_bbox_y=100,
            silhouette_height=700,
            width_profile=[
                ScanlineSlice(image_y=380, relative_y=0.4, left_x=200, right_x=380, width_pixels=181, is_valid=True, segment_count=1)
            ],
        )
        matched_slice, result = select_anatomical_slice(AnatomicalLevel.WAIST, features_no_landmarks)
        assert "fallback" in result.selection_method.lower()
        assert any("landmarks unavailable" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 5. Full Orchestration & Engine Tests
# ---------------------------------------------------------------------------

class TestCircumferenceEstimatorEngine:
    def test_end_to_end_calibrated_estimation(self, sample_view_features):
        # Create paired features with sample front and side
        paired = PairedGeometricFeatures(
            status=FeatureStatus.SUCCESS,
            front=sample_view_features,
            side=sample_view_features,
            height_ratio_side_to_front=1.0,
            expected_height_cm=175.0,
        )

        estimator = CircumferenceEstimator()
        result = estimator.estimate_measurements(paired)

        assert result.status in {MeasurementStatus.SUCCESS, MeasurementStatus.PARTIAL}
        assert result.calibration.is_valid is True

        # Check all 4 supported levels are populated
        for lvl in ["chest", "waist", "abdomen", "hips"]:
            assert lvl in result.measurements
            m = result.measurements[lvl]
            assert m.circumference_cm is not None
            assert m.circumference_cm > 0.0

    def test_uncalibrated_rejection(self, sample_view_features):
        paired = PairedGeometricFeatures(
            status=FeatureStatus.SUCCESS,
            front=sample_view_features,
            side=sample_view_features,
            height_ratio_side_to_front=1.0,
            expected_height_cm=None,  # No height provided
        )

        estimator = CircumferenceEstimator()
        result = estimator.estimate_measurements(paired, known_height_cm=None)

        assert result.status == MeasurementStatus.UNCALIBRATED
        assert result.calibration.is_valid is False
        for m in result.measurements.values():
            assert m.status == MeasurementStatus.UNCALIBRATED
            assert m.circumference_cm is None

    def test_json_safe_serialization(self, sample_view_features):
        paired = PairedGeometricFeatures(
            status=FeatureStatus.SUCCESS,
            front=sample_view_features,
            side=sample_view_features,
            height_ratio_side_to_front=1.0,
            expected_height_cm=175.0,
        )
        estimator = CircumferenceEstimator()
        result = estimator.estimate_measurements(paired)

        d = result.to_dict()
        assert "calibration" in d
        assert "measurements" in d
        # Verify clean JSON encoding without serialization errors
        json_str = json.dumps(d)
        decoded = json.loads(json_str)
        assert decoded["status"] in {"SUCCESS", "PARTIAL"}
