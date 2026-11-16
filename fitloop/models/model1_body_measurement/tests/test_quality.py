"""
tests/test_quality.py
---------------------
Unit tests for fitloop_measurement.quality.QualityChecker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.quality.checker import QualityChecker, QualityReport


def _make_landmark(x: float = 0.5, y: float = 0.5, visibility: float = 1.0):
    lm = MagicMock()
    lm.x, lm.y, lm.visibility = x, y, visibility
    return lm


def _mock_pose_results(detected: bool = True):
    mock = MagicMock()
    if not detected:
        mock.pose_landmarks = []
        return mock

    lms = [_make_landmark(0.5, 0.05 + i * (0.90 / 32)) for i in range(33)]
    lms[23] = _make_landmark(0.48, 0.7)  # LEFT_HIP
    lms[24] = _make_landmark(0.52, 0.7)  # RIGHT_HIP
    mock.pose_landmarks = [lms]
    return mock


class TestQualityReport:
    def test_passed_when_no_issues(self):
        r = QualityReport(passed=True)
        assert r.passed
        assert r.issues == []

    def test_str_contains_status(self):
        r = QualityReport(passed=False, issues=["Something wrong"])
        assert "FAILED" in str(r)
        assert "Something wrong" in str(r)


class TestQualityChecker:
    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_fails_when_no_person_detected(self, mock_exists, mock_create):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _mock_pose_results(detected=False)

        checker = QualityChecker()
        img = np.zeros((960, 640, 3), dtype=np.uint8)
        report = checker.run(img, subject_height_cm=175.0)

        assert not report.passed
        assert any("No person" in i for i in report.issues)

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_fails_without_height(self, mock_exists, mock_create):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _mock_pose_results(detected=True)

        checker = QualityChecker()
        img = np.zeros((960, 640, 3), dtype=np.uint8)
        report = checker.run(img, subject_height_cm=None)

        assert not report.passed
        assert any("height_cm" in i for i in report.issues)

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_fails_with_implausible_height(self, mock_exists, mock_create):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _mock_pose_results(detected=True)

        checker = QualityChecker()
        img = np.zeros((960, 640, 3), dtype=np.uint8)
        report = checker.run(img, subject_height_cm=5.0)

        assert not report.passed
        assert any("plausible" in i for i in report.issues)

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_passes_with_valid_inputs(self, mock_exists, mock_create):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _mock_pose_results(detected=True)

        checker = QualityChecker()
        img = np.zeros((960, 640, 3), dtype=np.uint8)
        report = checker.run(img, subject_height_cm=175.0)

        assert report.passed, f"Expected pass, got issues: {report.issues}"
