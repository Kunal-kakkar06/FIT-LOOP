"""
tests/test_pair_checker.py
--------------------------
Unit tests for fitloop_measurement.quality.PairedQualityChecker.

Test coverage:
1. Both views passing + valid height.
2. Missing front image.
3. Missing side image.
4. Unreadable / unsupported image formats.
5. One view failing quality checks (front fails, side passes).
6. One view failing quality checks (front passes, side fails).
7. Invalid reference height (missing, out-of-range low, out-of-range high).
8. Both views failing for different reasons.
9. Machine-readable error codes and actionable guidance verification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.quality.checker import QualityReport
from fitloop_measurement.quality.pair_checker import (
    ErrorCode,
    PairedQualityChecker,
    PairedQualityReport,
)


@pytest.fixture
def mock_checker():
    """Mock single-image QualityChecker."""
    return MagicMock()


@pytest.fixture
def dummy_img():
    """Standard synthetic test image."""
    return np.zeros((960, 640, 3), dtype=np.uint8)


class TestPairedQualityChecker:
    def test_both_views_passing(self, mock_checker, dummy_img):
        mock_checker.run.return_value = QualityReport(passed=True, issues=[], info={"resolution": "640×960"})
        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=dummy_img,
            side_image=dummy_img,
            subject_height_cm=175.0,
        )

        assert report.passed
        assert report.height_validated
        assert not report.has_front_failure
        assert not report.has_side_failure
        assert len(report.front_result.issues) == 0
        assert len(report.side_result.issues) == 0
        assert mock_checker.run.call_count == 2

    def test_missing_front_image(self, mock_checker, dummy_img):
        mock_checker.run.return_value = QualityReport(passed=True, issues=[])
        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=None,
            side_image=dummy_img,
            subject_height_cm=175.0,
        )

        assert not report.passed
        assert report.has_front_failure
        assert not report.has_side_failure
        assert any(i.code == ErrorCode.MISSING_IMAGE for i in report.front_result.issues)

    def test_missing_side_image(self, mock_checker, dummy_img):
        mock_checker.run.return_value = QualityReport(passed=True, issues=[])
        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=dummy_img,
            side_image=None,
            subject_height_cm=175.0,
        )

        assert not report.passed
        assert not report.has_front_failure
        assert report.has_side_failure
        assert any(i.code == ErrorCode.MISSING_IMAGE for i in report.side_result.issues)

    def test_unreadable_or_unsupported_images(self, mock_checker, tmp_path):
        bad_file = tmp_path / "corrupt.bmp"
        bad_file.write_bytes(b"invalid data")

        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=bad_file,
            side_image=Path("/nonexistent/file.jpg"),
            subject_height_cm=175.0,
        )

        assert not report.passed
        assert report.has_front_failure
        assert report.has_side_failure
        assert any(i.code == ErrorCode.UNREADABLE_IMAGE for i in report.front_result.issues)
        assert any(i.code == ErrorCode.UNREADABLE_IMAGE for i in report.side_result.issues)

    def test_front_view_fails_quality(self, mock_checker, dummy_img):
        def side_effect(img, subject_height_cm=None):
            if mock_checker.run.call_count == 1:
                return QualityReport(passed=False, issues=["Body midpoint is 20% off-centre"])
            return QualityReport(passed=True, issues=[])

        mock_checker.run.side_effect = side_effect
        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=dummy_img,
            side_image=dummy_img,
            subject_height_cm=175.0,
        )

        assert not report.passed
        assert report.has_front_failure
        assert not report.has_side_failure
        assert any(i.code == ErrorCode.QUALITY_CHECK_FAILED for i in report.front_result.issues)
        assert "Center the subject" in report.front_result.issues[0].actionable_fix

    def test_side_view_fails_quality(self, mock_checker, dummy_img):
        def side_effect(img, subject_height_cm=None):
            if mock_checker.run.call_count == 1:
                return QualityReport(passed=True, issues=[])
            return QualityReport(passed=False, issues=["No person detected. Ensure full body is visible."])

        mock_checker.run.side_effect = side_effect
        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=dummy_img,
            side_image=dummy_img,
            subject_height_cm=175.0,
        )

        assert not report.passed
        assert not report.has_front_failure
        assert report.has_side_failure
        assert any(i.code == ErrorCode.QUALITY_CHECK_FAILED for i in report.side_result.issues)

    @pytest.mark.parametrize("invalid_height", [None, 50.0, 300.0])
    def test_invalid_reference_height(self, mock_checker, dummy_img, invalid_height):
        mock_checker.run.return_value = QualityReport(passed=True, issues=[])
        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=dummy_img,
            side_image=dummy_img,
            subject_height_cm=invalid_height,
        )

        assert not report.passed
        assert not report.height_validated
        assert any(i.code == ErrorCode.INVALID_HEIGHT for i in report.height_issues)

    def test_both_views_failing_for_different_reasons(self, mock_checker, dummy_img):
        # Front has centering failure, Side has framing failure
        def side_effect(img, subject_height_cm=None):
            if mock_checker.run.call_count == 1:
                return QualityReport(passed=False, issues=["Body midpoint is 25% off-centre"])
            return QualityReport(passed=False, issues=["Body occupies only 50% of image height"])

        mock_checker.run.side_effect = side_effect
        checker = PairedQualityChecker(quality_checker=mock_checker)

        report = checker.validate_pair(
            front_image=dummy_img,
            side_image=dummy_img,
            subject_height_cm=180.0,
        )

        assert not report.passed
        assert report.has_front_failure
        assert report.has_side_failure

        front_issue = report.front_result.issues[0]
        side_issue = report.side_result.issues[0]

        assert "off-centre" in front_issue.message
        assert "occupies only" in side_issue.message
        assert front_issue.actionable_fix != side_issue.actionable_fix
