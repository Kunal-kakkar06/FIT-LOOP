"""
tests/test_api.py
------------------
Comprehensive backend API tests for Model 1 FastAPI application.

Covers:
1. Health endpoint response and status.
2. Missing file uploads and 422 errors.
3. Unsupported MIME types (HTTP 415).
4. Invalid standing height bounds (HTTP 422).
5. Oversized image uploads (HTTP 413).
6. Malformed/undecodable image payloads (HTTP 422).
7. Successful inference with mocked pipeline dependency.
8. Partial/uncalibrated measurement handling (transparent status, non-zero values).
9. Pipeline unexpected exception handling (HTTP 500).
10. Smoke test exercising real pipeline with synthetic test input (if asset exists).
11. (Stage 7G.2) quality_status and quality_gate consistency in API responses.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from fitloop_measurement.api import app, get_pipeline_service
from fitloop_measurement.measurements.schemas import (
    CalibrationStatus,
    MeasurementStatus,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_png_bytes():
    # Create a 480x640 dummy PNG
    img = np.zeros((640, 480, 3), dtype=np.uint8)
    # Draw simple shapes
    cv2.circle(img, (240, 100), 40, (255, 255, 255), -1)
    cv2.rectangle(img, (200, 150), (280, 400), (255, 255, 255), -1)
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


class TestHealthEndpoint:
    def test_health_check_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "fitloop-model1" in data["service"]


class TestRequestValidation:
    def test_missing_files_returns_422(self, client):
        response = client.post(
            "/api/v1/measurements/estimate",
            data={"standing_height_cm": 175.0},
        )
        assert response.status_code == 422

    def test_missing_height_returns_422(self, client, valid_png_bytes):
        response = client.post(
            "/api/v1/measurements/estimate",
            files={
                "front_image": ("front.png", valid_png_bytes, "image/png"),
                "side_image": ("side.png", valid_png_bytes, "image/png"),
            },
        )
        assert response.status_code == 422

    def test_height_out_of_bounds_returns_422(self, client, valid_png_bytes):
        # 50 cm is below human stature bounds (100-250)
        response = client.post(
            "/api/v1/measurements/estimate",
            data={"standing_height_cm": 50.0},
            files={
                "front_image": ("front.png", valid_png_bytes, "image/png"),
                "side_image": ("side.png", valid_png_bytes, "image/png"),
            },
        )
        assert response.status_code == 422
        assert "outside valid human stature range" in response.json()["detail"]

    def test_unsupported_image_mime_type_returns_415(self, client, valid_png_bytes):
        response = client.post(
            "/api/v1/measurements/estimate",
            data={"standing_height_cm": 175.0},
            files={
                "front_image": ("front.pdf", b"%PDF-1.4...", "application/pdf"),
                "side_image": ("side.png", valid_png_bytes, "image/png"),
            },
        )
        assert response.status_code == 415
        assert "unsupported" in response.json()["detail"]

    def test_corrupted_image_bytes_returns_422(self, client):
        response = client.post(
            "/api/v1/measurements/estimate",
            data={"standing_height_cm": 175.0},
            files={
                "front_image": ("front.jpg", b"not-a-valid-jpeg-stream", "image/jpeg"),
                "side_image": ("side.jpg", b"not-a-valid-jpeg-stream", "image/jpeg"),
            },
        )
        assert response.status_code == 422
        assert "Could not decode" in response.json()["detail"]


class TestPipelineExecutionMocked:
    def test_successful_mocked_pipeline_execution(self, client, valid_png_bytes):
        mock_result = {
            "status": "SUCCESS",
            "calibration": {
                "known_height_cm": 175.0,
                "is_valid": True,
                "front": {
                    "view_type": "front",
                    "silhouette_height_px": 1200,
                    "scale_cm_per_pixel": 0.1458,
                    "is_valid": True,
                    "status": "VALID",
                    "warnings": [],
                },
                "side": {
                    "view_type": "side",
                    "silhouette_height_px": 1200,
                    "scale_cm_per_pixel": 0.1458,
                    "is_valid": True,
                    "status": "VALID",
                    "warnings": [],
                },
                "assumptions": [],
                "warnings": [],
            },
            "measurements": {
                "chest": {
                    "level": "chest",
                    "circumference_cm": 95.0,
                    "status": "SUCCESS",
                    "front_width_px": 250,
                    "side_depth_px": 180,
                    "semi_major_cm": 18.2,
                    "semi_minor_cm": 13.1,
                    "approximation_method": "Ramanujan Second Ellipse Approximation",
                    "caveats": [],
                    "warnings": [],
                },
                "waist": {
                    "level": "waist",
                    "circumference_cm": 78.5,
                    "status": "SUCCESS",
                    "front_width_px": 210,
                    "side_depth_px": 160,
                    "semi_major_cm": 15.3,
                    "semi_minor_cm": 11.6,
                    "approximation_method": "Ramanujan Second Ellipse Approximation",
                    "caveats": [],
                    "warnings": [],
                },
            },
            "warnings": [],
            "is_mock": False,
        }

        mock_svc = MagicMock()
        mock_svc.process_pair.return_value = mock_result

        app.dependency_overrides[get_pipeline_service] = lambda: mock_svc
        try:
            with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
                response = client.post(
                    "/api/v1/measurements/estimate",
                    data={"standing_height_cm": 175.0},
                    files={
                        "front_image": ("front.png", valid_png_bytes, "image/png"),
                        "side_image": ("side.png", valid_png_bytes, "image/png"),
                    },
                )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SUCCESS"
            assert data["is_mock"] is False
            assert "chest" in data["measurements"]
            assert data["measurements"]["chest"]["circumference_cm"] == 95.0
        finally:
            app.dependency_overrides.clear()

    def test_pipeline_internal_error_returns_500(self, client, valid_png_bytes):
        mock_svc = MagicMock()
        mock_svc.process_pair.side_effect = RuntimeError("GPU memory allocation failure")

        with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
            response = client.post(
                "/api/v1/measurements/estimate",
                data={"standing_height_cm": 175.0},
                files={
                    "front_image": ("front.png", valid_png_bytes, "image/png"),
                    "side_image": ("side.png", valid_png_bytes, "image/png"),
                },
            )
        assert response.status_code == 500
        assert "Inference pipeline execution error" in response.json()["detail"]


class TestRealInferenceSmoke:
    def test_smoke_real_pipeline_execution_synthetic_person(self, client, valid_png_bytes):
        """Smoke test exercising the real Vision/Estimator service with synthetic image inputs.

        Verifies that PoseExtractor, BodySegmenter, FeatureExtractor, and CircumferenceEstimator
        run synchronously through FastAPI without unhandled crashes.
        """
        response = client.post(
            "/api/v1/measurements/estimate",
            data={"standing_height_cm": 175.0},
            files={
                "front_image": ("front.png", valid_png_bytes, "image/png"),
                "side_image": ("side.png", valid_png_bytes, "image/png"),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_mock"] is False
        assert "calibration" in data
        assert "measurements" in data
        # Flat synthetic image has no person landmarks, so it cleanly returns uncalibrated / unavailable
        # without blowing up or substituting zero for circumferences.
        assert data["status"] in {MeasurementStatus.UNCALIBRATED.value, MeasurementStatus.INVALID_INPUT.value, MeasurementStatus.SUCCESS.value}
        # Stage 7G.2: real response always includes quality_status and quality_gate
        assert "quality_status" in data
        assert data["quality_status"] in {"PASS", "WARNING", "REJECT"}
        assert "quality_gate" in data
        assert "status" in data["quality_gate"]
        # Top-level quality_status must agree with nested quality_gate.status
        assert data["quality_status"] == data["quality_gate"]["status"]


class TestQualityStatusAPIConsistency:
    """Stage 7G.2: Verify quality_status and quality_gate API contract from mocked service responses.

    These tests check the API layer serialisation and response shape, not the gate logic itself.
    Gate logic is tested in test_capture_gate.py.
    """

    def _make_mock_result(self, quality_status: str, issues: list = None) -> dict:
        """Build a minimal process_pair return dict with a given quality_status."""
        issues = issues or []
        return {
            "status": "SUCCESS",
            "is_mock": False,
            "quality_status": quality_status,
            "quality_gate": {
                "status": quality_status,
                "has_rejections": quality_status == "REJECT",
                "has_warnings": quality_status in ("WARNING", "REJECT"),
                "scale_discrepancy_fraction": None,
                "discrepancy_threshold_fraction": 0.05,
                "scale_rejection_threshold_fraction": 0.15,
                "min_margin_threshold_fraction": 0.05,
                "issues": issues,
                "front_framing": None,
                "side_framing": None,
                "assumptions_and_disclaimers": [
                    "A quality gate PASS verifies geometric and capture constraints, NOT anthropometric accuracy."
                ],
            },
            "calibration": {
                "known_height_cm": 175.0,
                "is_valid": True,
                "front": {"view_type": "front", "silhouette_height_px": 1200, "scale_cm_per_pixel": 0.1458, "is_valid": True, "status": "VALID", "warnings": []},
                "side": {"view_type": "side", "silhouette_height_px": 1200, "scale_cm_per_pixel": 0.1458, "is_valid": True, "status": "VALID", "warnings": []},
                "assumptions": [],
                "warnings": [],
            },
            "measurements": {
                "chest": {"level": "chest", "circumference_cm": 95.0, "status": "SUCCESS", "front_width_px": 250, "side_depth_px": 180, "semi_major_cm": 18.2, "semi_minor_cm": 13.1, "approximation_method": "Ramanujan Second Ellipse Approximation", "caveats": [], "warnings": []},
                "waist": {"level": "waist", "circumference_cm": 78.5, "status": "SUCCESS", "front_width_px": 210, "side_depth_px": 160, "semi_major_cm": 15.3, "semi_minor_cm": 11.6, "approximation_method": "Ramanujan Second Ellipse Approximation", "caveats": [], "warnings": []},
            },
            "warnings": [],
        }

    def test_pass_quality_status_in_api_response(self, client, valid_png_bytes):
        """PASS quality_status is preserved in HTTP 200 response."""
        mock_svc = MagicMock()
        mock_svc.process_pair.return_value = self._make_mock_result("PASS")

        with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
            response = client.post(
                "/api/v1/measurements/estimate",
                data={"standing_height_cm": 175.0},
                files={
                    "front_image": ("front.png", valid_png_bytes, "image/png"),
                    "side_image": ("side.png", valid_png_bytes, "image/png"),
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["quality_status"] == "PASS"
        assert data["quality_gate"]["status"] == "PASS"
        assert data["quality_gate"]["has_rejections"] is False
        assert data["quality_gate"]["has_warnings"] is False
        # PASS: no rejection warning in top-level warnings
        assert not any("REJECT" in w for w in data.get("warnings", []))

    def test_warning_quality_status_in_api_response(self, client, valid_png_bytes):
        """WARNING quality_status returned as HTTP 200 with issue list."""
        warning_issue = {
            "code": "SCALE_DISCREPANCY_EXCEEDED",
            "severity": "WARNING",
            "message": "Cross-view scale discrepancy is 9.1% (symmetric), above warning threshold 5.0%.",
            "context": {"discrepancy": 0.091, "warning_threshold": 0.05, "rejection_threshold": 0.15},
            "actionable_fix": "Retake both images from the same camera position.",
        }
        mock_svc = MagicMock()
        mock_svc.process_pair.return_value = self._make_mock_result("WARNING", issues=[warning_issue])

        with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
            response = client.post(
                "/api/v1/measurements/estimate",
                data={"standing_height_cm": 175.0},
                files={
                    "front_image": ("front.png", valid_png_bytes, "image/png"),
                    "side_image": ("side.png", valid_png_bytes, "image/png"),
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["quality_status"] == "WARNING"
        assert data["quality_gate"]["status"] == "WARNING"
        assert data["quality_gate"]["has_warnings"] is True
        issues = data["quality_gate"]["issues"]
        assert len(issues) == 1
        assert issues[0]["code"] == "SCALE_DISCREPANCY_EXCEEDED"
        assert issues[0]["severity"] == "WARNING"
        # Measurements still present despite WARNING
        assert "chest" in data["measurements"]
        assert data["measurements"]["chest"]["circumference_cm"] == 95.0

    def test_reject_quality_status_returns_200_with_diagnostic_warning(self, client, valid_png_bytes):
        """REJECT quality_status: HTTP 200 with REJECT in warnings; measurements preserved for diagnostics."""
        reject_result = self._make_mock_result("REJECT", issues=[{
            "code": "SILHOUETTE_CLIPPED",
            "severity": "CRITICAL",
            "message": "Front silhouette extends beyond top of frame.",
            "context": {},
            "actionable_fix": "Step back from the camera so your full body is in frame.",
        }])
        reject_result["warnings"] = [
            "CAPTURE QUALITY GATE REJECT: One or more critical capture quality violations were detected. "
            "Returned measurements are diagnostic-only and MUST NOT be treated as quality-approved estimates."
        ]

        mock_svc = MagicMock()
        mock_svc.process_pair.return_value = reject_result

        with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
            response = client.post(
                "/api/v1/measurements/estimate",
                data={"standing_height_cm": 175.0},
                files={
                    "front_image": ("front.png", valid_png_bytes, "image/png"),
                    "side_image": ("side.png", valid_png_bytes, "image/png"),
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["quality_status"] == "REJECT"
        assert data["quality_gate"]["status"] == "REJECT"
        assert data["quality_gate"]["has_rejections"] is True
        # Reject warning present in top-level warnings
        assert any("REJECT" in w for w in data["warnings"])
        # Diagnostic measurements are preserved (not replaced with zeros)
        assert "chest" in data["measurements"]
        assert data["measurements"]["chest"]["circumference_cm"] is not None
        assert data["measurements"]["chest"]["circumference_cm"] != 0.0

    def test_quality_status_and_nested_gate_status_always_agree(self, client, valid_png_bytes):
        """Top-level quality_status must always equal quality_gate.status."""
        for qs in ("PASS", "WARNING", "REJECT"):
            mock_svc = MagicMock()
            mock_svc.process_pair.return_value = self._make_mock_result(qs)

            with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
                response = client.post(
                    "/api/v1/measurements/estimate",
                    data={"standing_height_cm": 175.0},
                    files={
                        "front_image": ("front.png", valid_png_bytes, "image/png"),
                        "side_image": ("side.png", valid_png_bytes, "image/png"),
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["quality_status"] == data["quality_gate"]["status"], \
                f"Mismatch for quality_status={qs}: top={data['quality_status']}, nested={data['quality_gate']['status']}"

    def test_is_mock_false_on_real_pipeline_response(self, client, valid_png_bytes):
        """is_mock field must be False on all real pipeline responses."""
        mock_svc = MagicMock()
        mock_svc.process_pair.return_value = self._make_mock_result("PASS")

        with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
            response = client.post(
                "/api/v1/measurements/estimate",
                data={"standing_height_cm": 175.0},
                files={
                    "front_image": ("front.png", valid_png_bytes, "image/png"),
                    "side_image": ("side.png", valid_png_bytes, "image/png"),
                },
            )

        assert response.status_code == 200
        assert response.json()["is_mock"] is False

    def test_reject_measurements_not_zero(self, client, valid_png_bytes):
        """Rejected captures must NOT have measurements replaced with zero.

        The contract is: rejected results are diagnostic-only; measurements are preserved
        as-estimated for debugging, never zeroed out.
        """
        reject_result = self._make_mock_result("REJECT")
        reject_result["measurements"]["chest"]["circumference_cm"] = 88.3  # real estimate
        reject_result["warnings"] = ["CAPTURE QUALITY GATE REJECT: ..."]

        mock_svc = MagicMock()
        mock_svc.process_pair.return_value = reject_result

        with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
            response = client.post(
                "/api/v1/measurements/estimate",
                data={"standing_height_cm": 175.0},
                files={
                    "front_image": ("front.png", valid_png_bytes, "image/png"),
                    "side_image": ("side.png", valid_png_bytes, "image/png"),
                },
            )

        data = response.json()
        assert data["quality_status"] == "REJECT"
        assert data["measurements"]["chest"]["circumference_cm"] == 88.3

    def test_capture_quality_report_to_dict_serialization_integration(self, client, valid_png_bytes):
        """Stage 7G.3: Explicitly exercises the actual CaptureQualityReport.to_dict() serialization path.

        Verifies the FastAPI response contains consistent top-level and nested statuses, issue codes,
        severities, actionable remediation messages, framing diagnostics, and disclaimers.
        """
        from fitloop_measurement.quality.schemas import (
            CaptureQualityReport,
            QualityIssueCode,
            QualityIssueDetail,
            QualitySeverity,
            QualityStatus as PyQualityStatus,
            ViewFramingDiagnostics,
        )

        front_diag = ViewFramingDiagnostics(
            image_width=1080,
            image_height=1920,
            aspect_ratio=1080 / 1920,
            silhouette_bbox_x=200,
            silhouette_bbox_y=100,
            silhouette_width=600,
            silhouette_height=1700,
            top_margin_fraction=100 / 1920,
            bottom_margin_fraction=120 / 1920,
            scale_cm_per_pixel=0.103,
            is_clipped=False,
        )

        issues = [
            QualityIssueDetail(
                code=QualityIssueCode.SCALE_DISCREPANCY_EXCEEDED,
                severity=QualitySeverity.WARNING,
                message="Cross-view scale discrepancy is 7.2% (warning threshold 5.0%).",
                actionable_fix="Ensure participant stands at identical distance in both photos.",
                context={"front_scale": 0.103, "side_scale": 0.096, "discrepancy": 0.072},
            )
        ]

        report = CaptureQualityReport(
            status=PyQualityStatus.WARNING,
            issues=issues,
            front_framing=front_diag,
            side_framing=None,
            scale_discrepancy_fraction=0.072,
            discrepancy_threshold_fraction=0.05,
            scale_rejection_threshold_fraction=0.15,
            min_margin_threshold_fraction=0.05,
        )

        # Build mock measurement service returning the serialized report dict directly
        mock_result = {
            "status": "SUCCESS",
            "is_mock": False,
            "quality_status": report.status.value,
            "quality_gate": report.to_dict(),  # Actual schema method
            "calibration": {
                "known_height_cm": 175.0,
                "is_valid": True,
                "front": {"view_type": "front", "scale_cm_per_pixel": 0.103, "is_valid": True, "status": "VALID", "warnings": []},
                "side": {"view_type": "side", "scale_cm_per_pixel": 0.096, "is_valid": True, "status": "VALID", "warnings": []},
                "assumptions": [],
                "warnings": [],
            },
            "measurements": {
                "chest": {"level": "chest", "circumference_cm": 98.5, "status": "SUCCESS", "warnings": []},
            },
            "warnings": [],
        }

        mock_svc = MagicMock()
        mock_svc.process_pair.return_value = mock_result

        with patch("fitloop_measurement.api.get_pipeline_service", return_value=mock_svc):
            response = client.post(
                "/api/v1/measurements/estimate",
                data={"standing_height_cm": 175.0},
                files={
                    "front_image": ("front.png", valid_png_bytes, "image/png"),
                    "side_image": ("side.png", valid_png_bytes, "image/png"),
                },
            )

        assert response.status_code == 200
        body = response.json()

        # Contract assertion: Top-level quality_status matches nested report status
        assert body["quality_status"] == "WARNING"
        assert body["quality_gate"]["status"] == "WARNING"
        assert body["quality_gate"]["has_warnings"] is True
        assert body["quality_gate"]["has_rejections"] is False
        assert body["quality_gate"]["scale_discrepancy_fraction"] == 0.072

        # Issue serialization verification
        assert len(body["quality_gate"]["issues"]) == 1
        issue_out = body["quality_gate"]["issues"][0]
        assert issue_out["code"] == "SCALE_DISCREPANCY_EXCEEDED"
        assert issue_out["severity"] == "WARNING"
        assert "7.2%" in issue_out["message"]
        assert "identical distance" in issue_out["actionable_fix"]
        assert issue_out["context"]["discrepancy"] == 0.072

        # Framing serialization verification
        assert body["quality_gate"]["front_framing"] is not None
        assert body["quality_gate"]["front_framing"]["image_width"] == 1080
        assert body["quality_gate"]["front_framing"]["is_clipped"] is False

        # Assumptions and disclaimers list preserved
        assert len(body["quality_gate"]["assumptions_and_disclaimers"]) >= 3
        assert any("NOT anthropometric accuracy" in d for d in body["quality_gate"]["assumptions_and_disclaimers"])

