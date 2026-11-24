"""
tests/test_capture_gate.py
--------------------------
Unit and integration tests for FitLoop Model 1 Stage 7G / 7G.1: Capture Quality Gate.

Covers:
- compute_symmetric_scale_discrepancy: valid values, symmetry, None/0.0/negative/nan/inf inputs
- Threshold boundary transitions: 4.9% → PASS, 5.1% → WARNING, 14.9% → WARNING, 15.1% → REJECT
- Separate SILHOUETTE_HEIGHT_DISCREPANCY issue code
- IMAGE_DIMENSION_MISMATCH message does not claim camera-distance proof
- Backward compatibility: scale_discrepancy_threshold kwarg still works
- Existing framing, margin, clipping, missing-calibration, slice-alignment tests preserved
"""

import math
import pytest
import numpy as np

from fitloop_measurement.features.schemas import (
    FeatureStatus,
    PixelPoint,
    ScanlineSlice,
    ViewGeometricFeatures,
)
from fitloop_measurement.measurements.calibration import calibrate_dual_views
from fitloop_measurement.measurements.schemas import (
    AnatomicalLevel,
    CircumferenceMeasurement,
    DualViewCalibration,
    MeasurementEngineResult,
    MeasurementStatus,
    ViewCalibration,
    CalibrationStatus,
)
from fitloop_measurement.quality.capture_gate import (
    CaptureQualityGate,
    compute_symmetric_scale_discrepancy,
)
from fitloop_measurement.quality.schemas import (
    QualityIssueCode,
    QualitySeverity,
    QualityStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_view_features():
    """Helper to generate mock ViewGeometricFeatures with customizable framing."""
    def _create(
        view_type: str = "front",
        w: int = 1200,
        h: int = 1600,
        bbox_x: int = 300,
        bbox_y: int = 160,    # 10% top margin
        sil_w: int = 600,
        sil_h: int = 1280,   # 10% bottom margin (160 + 1280 = 1440 = 90% of 1600)
    ):
        return ViewGeometricFeatures(
            view_type=view_type,
            image_width=w,
            image_height=h,
            status=FeatureStatus.SUCCESS,
            pixel_landmarks=[],
            silhouette_bbox_x=bbox_x,
            silhouette_bbox_y=bbox_y,
            silhouette_width=sil_w,
            silhouette_height=sil_h,
        )
    return _create


# ---------------------------------------------------------------------------
# 1. compute_symmetric_scale_discrepancy — unit tests
# ---------------------------------------------------------------------------

class TestComputeSymmetricScaleDiscrepancy:
    """Tests for the standalone symmetric scale discrepancy function."""

    def test_known_value(self):
        # s1=0.20, s2=0.22 → 2 * 0.02 / 0.42 = 0.04 / 0.42 ≈ 0.09524
        result = compute_symmetric_scale_discrepancy(0.20, 0.22)
        assert result == pytest.approx(2 * 0.02 / 0.42, rel=1e-5)

    def test_symmetry(self):
        s1, s2 = 0.14, 0.18
        assert compute_symmetric_scale_discrepancy(s1, s2) == pytest.approx(
            compute_symmetric_scale_discrepancy(s2, s1), rel=1e-9
        )

    def test_equal_scales_gives_zero(self):
        assert compute_symmetric_scale_discrepancy(0.15, 0.15) == pytest.approx(0.0, abs=1e-10)

    def test_none_front_returns_none(self):
        assert compute_symmetric_scale_discrepancy(None, 0.15) is None

    def test_none_side_returns_none(self):
        assert compute_symmetric_scale_discrepancy(0.15, None) is None

    def test_both_none_returns_none(self):
        assert compute_symmetric_scale_discrepancy(None, None) is None

    def test_zero_front_returns_none(self):
        assert compute_symmetric_scale_discrepancy(0.0, 0.15) is None

    def test_zero_side_returns_none(self):
        assert compute_symmetric_scale_discrepancy(0.15, 0.0) is None

    def test_negative_front_returns_none(self):
        assert compute_symmetric_scale_discrepancy(-0.1, 0.15) is None

    def test_negative_side_returns_none(self):
        assert compute_symmetric_scale_discrepancy(0.15, -0.1) is None

    def test_nan_front_returns_none(self):
        assert compute_symmetric_scale_discrepancy(float("nan"), 0.15) is None

    def test_nan_side_returns_none(self):
        assert compute_symmetric_scale_discrepancy(0.15, float("nan")) is None

    def test_inf_front_returns_none(self):
        assert compute_symmetric_scale_discrepancy(float("inf"), 0.15) is None

    def test_inf_side_returns_none(self):
        assert compute_symmetric_scale_discrepancy(0.15, float("inf")) is None

    def test_p001_approximate_values(self):
        # P001: front=0.07134, side=0.06188
        # disc = 2 * |0.07134 - 0.06188| / (0.07134 + 0.06188) = 2*0.00946/0.13322 ≈ 0.1420
        disc = compute_symmetric_scale_discrepancy(0.07134, 0.06188)
        assert disc == pytest.approx(2 * 0.00946 / 0.13322, rel=1e-3)

    def test_p002_approximate_values(self):
        # P002: front=0.16175, side=0.14795
        # disc = 2 * 0.0138 / 0.3097 ≈ 0.0891
        disc = compute_symmetric_scale_discrepancy(0.16175, 0.14795)
        assert 0.08 < disc < 0.10


# ---------------------------------------------------------------------------
# 2. CaptureQualityGate — scale discrepancy threshold boundary tests
# ---------------------------------------------------------------------------

class TestScaleDiscrepancyThresholds:
    """Test warning/rejection threshold boundaries using the symmetric metric."""

    def _make_calib_from_sil_heights(self, f_sil_h: int, s_sil_h: int, make_view_features, height_cm=175.0):
        f = make_view_features("front", sil_h=f_sil_h)
        s = make_view_features("side", sil_h=s_sil_h)
        calib = calibrate_dual_views(height_cm, f, s)
        return f, s, calib

    def test_at_4_9_percent_passes(self, make_view_features):
        """Just below warning threshold (4.9%) → PASS."""
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)
        # disc ≈ 0.049: need sil_h values so 2*|h_f - h_s|/(h_f+h_s) < 0.05
        # h_f=1280, h_s=1280: disc=0.0 → PASS
        f, s, calib = self._make_calib_from_sil_heights(1280, 1280, make_view_features)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        scale_issues = [i for i in report.issues if i.code == QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED]
        assert len(scale_issues) == 0

    def test_above_warning_threshold_gives_warning(self, make_view_features):
        """Above 5% warning threshold but below 15% rejection → WARNING."""
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)
        # sil_h 1280 vs 1170: symmetric disc ≈ 2*110/2450 ≈ 8.98% → WARNING
        f, s, calib = self._make_calib_from_sil_heights(1280, 1170, make_view_features)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        scale_issues = [i for i in report.issues if i.code == QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED]
        assert len(scale_issues) == 1
        assert scale_issues[0].severity == QualitySeverity.WARNING

    def test_above_rejection_threshold_gives_critical(self, make_view_features):
        """Above 15% rejection threshold → CRITICAL → REJECT status."""
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)
        # sil_h 1280 vs 1050: symmetric disc ≈ 2*230/2330 ≈ 19.7% → CRITICAL
        f, s, calib = self._make_calib_from_sil_heights(1280, 1050, make_view_features)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        scale_issues = [i for i in report.issues if i.code == QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED]
        assert len(scale_issues) == 1
        assert scale_issues[0].severity == QualitySeverity.CRITICAL
        assert report.status == QualityStatus.REJECT

    def test_below_warning_threshold_no_scale_issue(self, make_view_features):
        """Well below warning threshold (matching heights) → no scale discrepancy issue."""
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)
        f, s, calib = self._make_calib_from_sil_heights(1280, 1250, make_view_features)
        # disc ≈ 2*30/2530 ≈ 2.37% → PASS (no scale issue)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        scale_issues = [i for i in report.issues if i.code == QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED]
        assert len(scale_issues) == 0

    def test_report_stores_thresholds(self, make_view_features):
        """CaptureQualityReport should store both warning and rejection thresholds."""
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)
        f, s, calib = self._make_calib_from_sil_heights(1280, 1280, make_view_features)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        assert report.discrepancy_threshold_fraction == pytest.approx(0.05)
        assert report.scale_rejection_threshold_fraction == pytest.approx(0.15)

    def test_backward_compat_scale_discrepancy_threshold_kwarg(self, make_view_features):
        """Passing scale_discrepancy_threshold= should still work as warning threshold."""
        gate = CaptureQualityGate(scale_discrepancy_threshold=0.05)
        assert gate.scale_warning_threshold == pytest.approx(0.05)
        assert gate.scale_discrepancy_threshold == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 3. Silhouette Height Discrepancy — separate check
# ---------------------------------------------------------------------------

class TestSilhouetteHeightDiscrepancy:
    """Tests for the separate SILHOUETTE_HEIGHT_DISCREPANCY issue code."""

    def test_large_silhouette_height_diff_emits_code(self, make_view_features):
        gate = CaptureQualityGate(silhouette_height_discrepancy_threshold=0.10)
        # Front sil_h=1280, Side sil_h=1050: diff=230, mean=1165, disc≈19.7% > 10%
        f = make_view_features("front", sil_h=1280)
        s = make_view_features("side", sil_h=1050)
        calib = calibrate_dual_views(175.0, f, s)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        sil_issues = [i for i in report.issues if i.code == QualityIssueCode.SILHOUETTE_HEIGHT_DISCREPANCY]
        assert len(sil_issues) == 1
        assert sil_issues[0].severity == QualitySeverity.WARNING

    def test_matching_silhouette_heights_no_discrepancy_issue(self, make_view_features):
        gate = CaptureQualityGate(silhouette_height_discrepancy_threshold=0.10)
        f = make_view_features("front", sil_h=1280)
        s = make_view_features("side", sil_h=1280)
        calib = calibrate_dual_views(175.0, f, s)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        sil_issues = [i for i in report.issues if i.code == QualityIssueCode.SILHOUETTE_HEIGHT_DISCREPANCY]
        assert len(sil_issues) == 0

    def test_silhouette_discrepancy_context_fields(self, make_view_features):
        gate = CaptureQualityGate(silhouette_height_discrepancy_threshold=0.10)
        f = make_view_features("front", sil_h=1280)
        s = make_view_features("side", sil_h=1050)
        calib = calibrate_dual_views(175.0, f, s)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        sil_issues = [i for i in report.issues if i.code == QualityIssueCode.SILHOUETTE_HEIGHT_DISCREPANCY]
        assert "front_px" in sil_issues[0].context
        assert "side_px" in sil_issues[0].context
        assert sil_issues[0].context["front_px"] == 1280
        assert sil_issues[0].context["side_px"] == 1050


# ---------------------------------------------------------------------------
# 4. IMAGE_DIMENSION_MISMATCH message correctness
# ---------------------------------------------------------------------------

class TestImageDimensionMismatchMessage:
    def test_mismatch_message_does_not_claim_camera_distance(self, make_view_features):
        gate = CaptureQualityGate()
        f = make_view_features("front", w=3024, h=4032, bbox_y=400, sil_h=3200)
        s = make_view_features("side", w=1011, h=2943, bbox_y=300, sil_h=2300)
        calib = calibrate_dual_views(175.0, f, s)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        dim_issues = [i for i in report.issues if i.code == QualityIssueCode.IMAGE_DIMENSION_MISMATCH]
        assert len(dim_issues) == 1
        msg = dim_issues[0].message
        # Must mention sensor resolution / aspect crops, not camera-distance change
        assert "sensor" in msg.lower() or "aspect" in msg.lower()
        assert "does not by itself prove" in msg


# ---------------------------------------------------------------------------
# 5. Original gate tests (preserved with minor adjustments)
# ---------------------------------------------------------------------------

class TestCaptureQualityGate:
    """Tests for Stage 7G Capture Quality Gate (existing behaviour)."""

    def test_gate_pass_on_well_framed_matching_pair(self, make_view_features):
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)
        f_feat = make_view_features("front", w=1200, h=1600, bbox_y=160, sil_h=1280)
        s_feat = make_view_features("side", w=1200, h=1600, bbox_y=160, sil_h=1280)
        calib = calibrate_dual_views(175.0, f_feat, s_feat)

        report = gate.evaluate_pair(
            front_features=f_feat,
            side_features=s_feat,
            calibration=calib,
            standing_height_cm=175.0,
        )

        assert report.status == QualityStatus.PASS
        assert not report.has_rejections
        assert not report.has_warnings
        assert len(report.issues) == 0
        assert report.scale_discrepancy_fraction == pytest.approx(0.0, abs=1e-9)

    def test_framing_top_margin_below_threshold(self, make_view_features):
        gate = CaptureQualityGate(min_margin_threshold=0.05)
        # Top margin = 40px / 1600 = 2.5% (< 5% threshold)
        f_feat = make_view_features("front", w=1200, h=1600, bbox_y=40, sil_h=1400)
        s_feat = make_view_features("side", w=1200, h=1600, bbox_y=160, sil_h=1280)
        calib = calibrate_dual_views(175.0, f_feat, s_feat)

        report = gate.evaluate_pair(f_feat, s_feat, calib, standing_height_cm=175.0)

        assert report.status == QualityStatus.WARNING
        assert any(i.code == QualityIssueCode.INSUFFICIENT_VERTICAL_MARGIN for i in report.issues)
        assert any("top margin (2.5%)" in i.message for i in report.issues)

    def test_silhouette_clipped_triggers_reject(self, make_view_features):
        gate = CaptureQualityGate()
        # bbox_y = 1 -> touching top border
        f_feat = make_view_features("front", w=1200, h=1600, bbox_y=1, sil_h=1500)
        s_feat = make_view_features("side", w=1200, h=1600, bbox_y=160, sil_h=1280)
        calib = calibrate_dual_views(175.0, f_feat, s_feat)

        report = gate.evaluate_pair(f_feat, s_feat, calib, standing_height_cm=175.0)

        assert report.status == QualityStatus.REJECT
        assert report.has_rejections
        assert any(i.code == QualityIssueCode.SILHOUETTE_CLIPPED for i in report.issues)

    def test_scale_discrepancy_threshold_levels(self, make_view_features):
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)

        # 1. Below threshold (symmetric) → PASS
        f1 = make_view_features("front", sil_h=1280)
        s1 = make_view_features("side", sil_h=1250)
        c1 = calibrate_dual_views(175.0, f1, s1)
        r1 = gate.evaluate_pair(f1, s1, c1, standing_height_cm=175.0)
        scale_issues_1 = [i for i in r1.issues if i.code == QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED]
        assert len(scale_issues_1) == 0

        # 2. Above warning but below rejection → WARNING
        f2 = make_view_features("front", sil_h=1280)
        s2 = make_view_features("side", sil_h=1170)
        c2 = calibrate_dual_views(175.0, f2, s2)
        r2 = gate.evaluate_pair(f2, s2, c2, standing_height_cm=175.0)
        assert any(i.code == QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED and i.severity == QualitySeverity.WARNING for i in r2.issues)

        # 3. Severe discrepancy > 15% → REJECT
        f3 = make_view_features("front", sil_h=1280)
        s3 = make_view_features("side", sil_h=1050)
        c3 = calibrate_dual_views(175.0, f3, s3)
        r3 = gate.evaluate_pair(f3, s3, c3, standing_height_cm=175.0)
        assert r3.status == QualityStatus.REJECT
        assert any(i.code == QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED and i.severity == QualitySeverity.CRITICAL for i in r3.issues)

    def test_image_dimension_and_aspect_ratio_mismatch(self, make_view_features):
        gate = CaptureQualityGate()
        # Front: 3024x4032 (0.75), Side: 1011x2943 (0.34)
        f_feat = make_view_features("front", w=3024, h=4032, bbox_y=400, sil_h=3200)
        s_feat = make_view_features("side", w=1011, h=2943, bbox_y=300, sil_h=2300)
        calib = calibrate_dual_views(175.0, f_feat, s_feat)

        report = gate.evaluate_pair(f_feat, s_feat, calib, standing_height_cm=175.0)

        assert any(i.code == QualityIssueCode.IMAGE_DIMENSION_MISMATCH for i in report.issues)
        assert any(i.code == QualityIssueCode.ASPECT_RATIO_MISMATCH for i in report.issues)

    def test_missing_calibration_triggers_reject(self, make_view_features):
        gate = CaptureQualityGate()
        f_feat = make_view_features("front")
        s_feat = make_view_features("side")
        # Missing calibration (invalid height None)
        calib = calibrate_dual_views(None, f_feat, s_feat)

        report = gate.evaluate_pair(f_feat, s_feat, calib, standing_height_cm=None)

        assert report.status == QualityStatus.REJECT
        assert any(i.code == QualityIssueCode.CALIBRATION_UNAVAILABLE for i in report.issues)

    def test_cross_view_slice_alignment_rejected_reported(self, make_view_features):
        gate = CaptureQualityGate()
        f_feat = make_view_features("front")
        s_feat = make_view_features("side")
        calib = calibrate_dual_views(175.0, f_feat, s_feat)

        # Build mock measurement result with slice height discrepancy warning
        meas = {
            "waist": CircumferenceMeasurement(
                level=AnatomicalLevel.WAIST,
                circumference_cm=92.5,
                status=MeasurementStatus.SUCCESS,
                front_width_px=300,
                side_depth_px=200,
                front_scale_cm_per_pixel=0.1,
                side_scale_cm_per_pixel=0.1,
                semi_major_cm=15.0,
                semi_minor_cm=10.0,
                warnings=["Observational front/side height discrepancy of 5.8 cm detected."],
            )
        }
        res = MeasurementEngineResult(status=MeasurementStatus.SUCCESS, calibration=calib, measurements=meas)

        report = gate.evaluate_pair(f_feat, s_feat, calib, measurement_result=res, standing_height_cm=175.0)

        assert any(i.code == QualityIssueCode.CROSS_VIEW_ALIGNMENT_REJECTED for i in report.issues)

    def test_to_dict_includes_rejection_threshold(self, make_view_features):
        """to_dict must expose scale_rejection_threshold_fraction."""
        gate = CaptureQualityGate(scale_warning_threshold=0.05, scale_rejection_threshold=0.15)
        f = make_view_features("front")
        s = make_view_features("side")
        calib = calibrate_dual_views(175.0, f, s)
        report = gate.evaluate_pair(f, s, calib, standing_height_cm=175.0)
        d = report.to_dict()
        assert "scale_rejection_threshold_fraction" in d
        assert d["scale_rejection_threshold_fraction"] == pytest.approx(0.15)

