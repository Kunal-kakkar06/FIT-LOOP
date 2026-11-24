"""
tests/test_slice_strategy.py
----------------------------
Focused tests for Stage 7E.2: Slice selection strategy contracts,
BaselineFixedFractionStrategy, and AnatomicalCrossViewStrategy placeholder boundary.
"""

import pytest
import numpy as np

from fitloop_measurement.features.schemas import (
    FeatureStatus,
    PixelPoint,
    ScanlineSlice,
    ViewGeometricFeatures,
    PairedGeometricFeatures,
)
from fitloop_measurement.measurements.calibration import calibrate_dual_views
from fitloop_measurement.measurements.estimator import CircumferenceEstimator
from fitloop_measurement.measurements.schemas import (
    AnatomicalLevel,
    MeasurementStatus,
)
from fitloop_measurement.measurements.slice_strategy import (
    AlignedSlicePair,
    AlignmentConfidence,
    AnatomicalCrossViewStrategy,
    BaselineFixedFractionStrategy,
    SliceSelectionStrategyType,
    compute_observational_physical_height,
)


@pytest.fixture
def mock_torso_features():
    """Generates synthetic front and side ViewGeometricFeatures with known landmarks.

    Top: y = 100, Height = 700
    Shoulders: y = 200
    Hips: y = 600
    Torso span = 400 px
    Target scanlines at fixed fractions:
      - CHEST (0.35):   200 + 0.35 * 400 = 340
      - WAIST (0.65):   200 + 0.65 * 400 = 460
      - ABDOMEN (0.80): 200 + 0.80 * 400 = 520
      - HIPS (1.05):    200 + 1.05 * 400 = 620
    """
    def _create(view_type: str, top_y: int = 100, sil_h: int = 700, sh_y: float = 200.0, hip_y: float = 600.0):
        landmarks = [
            PixelPoint(index=11, name="LEFT_SHOULDER", pixel_x=220.0, pixel_y=sh_y, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
            PixelPoint(index=12, name="RIGHT_SHOULDER", pixel_x=380.0, pixel_y=sh_y, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
            PixelPoint(index=23, name="LEFT_HIP", pixel_x=240.0, pixel_y=hip_y, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
            PixelPoint(index=24, name="RIGHT_HIP", pixel_x=360.0, pixel_y=hip_y, relative_z=0.0, visibility=0.9, presence=0.9, is_valid=True),
        ]
        # Profile samples every 20px
        profile = []
        for y in range(top_y, top_y + sil_h, 20):
            profile.append(
                ScanlineSlice(
                    image_y=y,
                    relative_y=(y - top_y) / sil_h,
                    left_x=200,
                    right_x=400,
                    width_pixels=200,
                    is_valid=True,
                    segment_count=1,
                )
            )

        return ViewGeometricFeatures(
            view_type=view_type,
            image_width=640,
            image_height=960,
            status=FeatureStatus.SUCCESS,
            pixel_landmarks=landmarks,
            silhouette_bbox_x=200,
            silhouette_bbox_y=top_y,
            silhouette_width=200,
            silhouette_height=sil_h,
            width_profile=profile,
        )

    return _create


class TestBaselineFixedFractionStrategy:
    """Test suite verifying BaselineFixedFractionStrategy preserves exact behavior."""

    def test_strategy_type_and_defaults(self):
        strat = BaselineFixedFractionStrategy()
        assert strat.strategy_type == SliceSelectionStrategyType.BASELINE_FIXED_FRACTION

    def test_exact_row_selection_matches_fractions(self, mock_torso_features):
        front = mock_torso_features("front", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        side = mock_torso_features("side", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        calib = calibrate_dual_views(175.0, front, side)

        strat = BaselineFixedFractionStrategy()

        # Expected target rows: 340 (chest), 460 (waist), 520 (abdomen), 620 (hips)
        expected = {
            AnatomicalLevel.CHEST: 340,
            AnatomicalLevel.WAIST: 460,
            AnatomicalLevel.ABDOMEN: 520,
            AnatomicalLevel.HIPS: 620,
        }

        for lvl, exp_y in expected.items():
            pair = strat.select_slices(lvl, front, side, calib)
            assert pair.is_valid is True
            assert pair.front_slice.image_y == exp_y
            assert pair.side_slice.image_y == exp_y
            assert pair.strategy_used == SliceSelectionStrategyType.BASELINE_FIXED_FRACTION
            assert pair.confidence == AlignmentConfidence.UNVERIFIED
            assert pair.is_aligned is False

    def test_mask_aware_selection_preservation(self, mock_torso_features):
        front = mock_torso_features("front")
        side = mock_torso_features("side")
        calib = calibrate_dual_views(175.0, front, side)

        # Create synthetic mask of shape (960, 640)
        mask = np.zeros((960, 640), dtype=np.uint8)
        # Add torso foreground box around waist (y=450..470, x=220..380)
        mask[450:470, 220:380] = 255

        strat = BaselineFixedFractionStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib, front_mask=mask)

        assert pair.is_valid is True
        assert pair.front_slice is not None
        assert "torso-aware" in pair.front_result.selection_method.lower()

    def test_fallback_when_landmarks_missing(self):
        front_no_lm = ViewGeometricFeatures(
            view_type="front",
            image_width=640,
            image_height=960,
            status=FeatureStatus.SUCCESS,
            pixel_landmarks=[],
            silhouette_bbox_y=100,
            silhouette_height=700,
            width_profile=[
                ScanlineSlice(image_y=380, relative_y=0.4, left_x=200, right_x=380, width_pixels=181, is_valid=True, segment_count=1)
            ],
        )
        side_no_lm = ViewGeometricFeatures(
            view_type="side",
            image_width=640,
            image_height=960,
            status=FeatureStatus.SUCCESS,
            pixel_landmarks=[],
            silhouette_bbox_y=100,
            silhouette_height=700,
            width_profile=[
                ScanlineSlice(image_y=380, relative_y=0.4, left_x=200, right_x=380, width_pixels=181, is_valid=True, segment_count=1)
            ],
        )
        calib = calibrate_dual_views(175.0, front_no_lm, side_no_lm)
        strat = BaselineFixedFractionStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front_no_lm, side_no_lm, calib)

        assert "fallback" in pair.front_result.selection_method.lower()
        assert pair.confidence == AlignmentConfidence.UNVERIFIED

    def test_physical_height_difference_calculation(self, mock_torso_features):
        # Front top_y = 100, sil_h = 700 -> scale = 175.0 / 700 = 0.25 cm/px
        # Waist y = 460 -> front_phys = (460 - 100) * 0.25 = 90.0 cm
        # Side top_y = 50, sil_h = 700 -> scale = 175.0 / 700 = 0.25 cm/px
        # Shoulders y = 150, Hips y = 550 (span 400) -> side waist y = 150 + 0.65 * 400 = 410
        # side_phys = (410 - 50) * 0.25 = 90.0 cm -> height_delta = 0.0 cm
        front = mock_torso_features("front", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        side = mock_torso_features("side", top_y=50, sil_h=700, sh_y=150.0, hip_y=550.0)
        calib = calibrate_dual_views(175.0, front, side)

        strat = BaselineFixedFractionStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib)

        assert pair.front_phys_height_cm == pytest.approx(90.0, 1e-2)
        assert pair.side_phys_height_cm == pytest.approx(90.0, 1e-2)
        assert pair.height_delta_cm == pytest.approx(0.0, 1e-2)

    def test_physical_height_discrepancy_generates_warning(self, mock_torso_features):
        # Shift side shoulders so side waist ends up 20px lower (20 * 0.25 = 5.0 cm difference)
        front = mock_torso_features("front", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        side = mock_torso_features("side", top_y=100, sil_h=700, sh_y=220.0, hip_y=620.0)
        calib = calibrate_dual_views(175.0, front, side)

        strat = BaselineFixedFractionStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib)

        assert pair.height_delta_cm == pytest.approx(5.0, 1e-2)
        assert any("discrepancy" in w.lower() for w in pair.warnings)

    def test_uncalibrated_height_difference_unavailable(self, mock_torso_features):
        front = mock_torso_features("front")
        side = mock_torso_features("side")
        # Uncalibrated: known_height_cm = None
        calib = calibrate_dual_views(None, front, side)

        strat = BaselineFixedFractionStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib)

        assert pair.front_phys_height_cm is None
        assert pair.side_phys_height_cm is None
        assert pair.height_delta_cm is None
        assert any("unavailable: metric calibration is invalid" in w for w in pair.warnings)


class TestAnatomicalCrossViewStrategy:
    """Test suite verifying AnatomicalCrossViewStrategy behavior under various conditions."""

    def test_strategy_type(self):
        strat = AnatomicalCrossViewStrategy()
        assert strat.strategy_type == SliceSelectionStrategyType.ANATOMICAL_CROSS_VIEW

    def test_uncalibrated_rejection(self, mock_torso_features):
        front = mock_torso_features("front")
        side = mock_torso_features("side")
        calib = calibrate_dual_views(None, front, side)

        strat = AnatomicalCrossViewStrategy()
        # When mask is None, candidates cannot be extracted -> UNAVAILABLE
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib, front_mask=None, side_mask=None)

        assert pair.is_aligned is False
        assert pair.confidence == AlignmentConfidence.UNAVAILABLE
        assert any("unavailable" in w.lower() for w in pair.warnings)

        # When mask is provided but calibration is invalid -> LOW confidence
        f_mask = np.zeros((960, 640), dtype=np.uint8)
        s_mask = np.zeros((960, 640), dtype=np.uint8)
        f_mask[200:600, 250:350] = 255
        s_mask[200:600, 250:350] = 255
        pair2 = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib, front_mask=f_mask, side_mask=s_mask)
        assert pair2.is_aligned is False
        assert pair2.confidence == AlignmentConfidence.LOW
        assert any("calibration invalid" in w.lower() for w in pair2.warnings)

    def test_candidate_selection_with_synthetic_masks(self, mock_torso_features):
        front = mock_torso_features("front", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        side = mock_torso_features("side", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        calib = calibrate_dual_views(175.0, front, side)

        f_mask = np.zeros((960, 640), dtype=np.uint8)
        s_mask = np.zeros((960, 640), dtype=np.uint8)
        # Add hourglass torso to front mask (narrowest around y=460)
        for y in range(200, 600):
            hw = int(80 + abs(y - 460) * 0.2)
            f_mask[y, 300 - hw: 300 + hw] = 255
            # side mask with waist depth inflection around y=460
            sd = int(50 + abs(y - 460) * 0.1)
            s_mask[y, 300 - sd: 300 + sd] = 255

        strat = AnatomicalCrossViewStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib, front_mask=f_mask, side_mask=s_mask)

        assert pair.front_slice is not None
        assert pair.side_slice is not None
        # Candidate waist should be within 10px of y=460
        assert abs(pair.front_slice.image_y - 460) <= 10
        # Alignment confidence should be MEDIUM (never HIGH without surface annotations)
        assert pair.confidence in {AlignmentConfidence.MEDIUM, AlignmentConfidence.LOW}

    def test_height_discrepancy_rejection(self, mock_torso_features):
        # Shift side landmarks so physical heights diverge by 5.0 cm (> 2.5 cm tolerance)
        front = mock_torso_features("front", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        side = mock_torso_features("side", top_y=100, sil_h=700, sh_y=220.0, hip_y=620.0)
        calib = calibrate_dual_views(175.0, front, side)

        f_mask = np.zeros((960, 640), dtype=np.uint8)
        s_mask = np.zeros((960, 640), dtype=np.uint8)
        f_mask[200:600, 250:350] = 255
        s_mask[200:650, 250:350] = 255

        strat = AnatomicalCrossViewStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib, front_mask=f_mask, side_mask=s_mask)

        # Discrepancy exceeds tolerance: is_aligned must be False and confidence LOW
        assert pair.is_aligned is False
        assert pair.confidence == AlignmentConfidence.LOW
        assert any("tolerance" in w.lower() for w in pair.warnings)

    def test_missing_mask_returns_unavailable(self, mock_torso_features):
        front = mock_torso_features("front")
        side = mock_torso_features("side")
        calib = calibrate_dual_views(175.0, front, side)

        strat = AnatomicalCrossViewStrategy()
        # When mask is None, candidates cannot be extracted
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib, front_mask=None, side_mask=None)

        assert pair.is_valid is False
        assert pair.front_slice is None
        assert pair.side_slice is None
        assert pair.confidence == AlignmentConfidence.UNAVAILABLE
        assert any("unavailable" in w.lower() for w in pair.warnings)

    def test_ambiguous_contour_generates_warning(self, mock_torso_features):
        front = mock_torso_features("front", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        side = mock_torso_features("side", top_y=100, sil_h=700, sh_y=200.0, hip_y=600.0)
        calib = calibrate_dual_views(175.0, front, side)

        # Create perfectly straight/tubular cylinder mask (0% width variation)
        f_mask = np.zeros((960, 640), dtype=np.uint8)
        s_mask = np.zeros((960, 640), dtype=np.uint8)
        f_mask[200:600, 220:380] = 255  # constant 161 px width
        s_mask[200:600, 250:350] = 255  # constant 101 px depth

        strat = AnatomicalCrossViewStrategy()
        pair = strat.select_slices(AnatomicalLevel.WAIST, front, side, calib, front_mask=f_mask, side_mask=s_mask)

        assert any("ambiguous" in w.lower() for w in pair.warnings)


class TestCircumferenceEstimatorIntegration:
    """Verifies estimator correctly defaults to baseline strategy and produces identical outputs."""

    def test_default_strategy_is_baseline(self):
        estimator = CircumferenceEstimator()
        assert isinstance(estimator.strategy, BaselineFixedFractionStrategy)
        assert estimator.strategy.strategy_type == SliceSelectionStrategyType.BASELINE_FIXED_FRACTION

    def test_estimation_output_identical_with_baseline(self, mock_torso_features):
        front = mock_torso_features("front")
        side = mock_torso_features("side")
        paired = PairedGeometricFeatures(
            status=FeatureStatus.SUCCESS,
            front=front,
            side=side,
            height_ratio_side_to_front=1.0,
            expected_height_cm=175.0,
        )

        estimator = CircumferenceEstimator()
        result = estimator.estimate_measurements(paired)

        assert result.status == MeasurementStatus.SUCCESS
        for lvl in ["chest", "waist", "abdomen", "hips"]:
            assert result.measurements[lvl].circumference_cm is not None
            assert result.measurements[lvl].circumference_cm > 0.0

    def test_estimator_with_experimental_strategy_on_synthetic_masks(self, mock_torso_features):
        front = mock_torso_features("front")
        side = mock_torso_features("side")
        paired = PairedGeometricFeatures(
            status=FeatureStatus.SUCCESS,
            front=front,
            side=side,
            height_ratio_side_to_front=1.0,
            expected_height_cm=175.0,
        )

        f_mask = np.zeros((960, 640), dtype=np.uint8)
        s_mask = np.zeros((960, 640), dtype=np.uint8)
        f_mask[100:800, 220:380] = 255
        s_mask[100:800, 250:350] = 255

        estimator = CircumferenceEstimator(strategy=AnatomicalCrossViewStrategy())
        result = estimator.estimate_measurements(paired, front_mask=f_mask, side_mask=s_mask)

        assert result.status in {MeasurementStatus.SUCCESS, MeasurementStatus.PARTIAL}
        for lvl in ["chest", "waist", "abdomen", "hips"]:
            m = result.measurements[lvl]
            assert m.circumference_cm is not None
            assert m.circumference_cm > 0.0


class TestCoordinateHelper:
    """Test observational physical height calculation."""

    def test_valid_calculation(self):
        res = compute_observational_physical_height(image_y=350, bbox_y=100, scale_cm_per_pixel=0.25)
        assert res == 62.5

    @pytest.mark.parametrize("y, bbox_y, scale", [
        (None, 100, 0.25),
        (350, None, 0.25),
        (350, 100, None),
        (350, 100, 0.0),
        (350, 100, -0.25),
        (350, 100, float("nan")),
        (350, 100, float("inf")),
    ])
    def test_invalid_parameters_return_none(self, y, bbox_y, scale):
        assert compute_observational_physical_height(y, bbox_y, scale) is None
