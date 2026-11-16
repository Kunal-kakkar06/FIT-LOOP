"""
tests/test_pose_extractor.py
----------------------------
Comprehensive tests for Stage 3A: MediaPipe Pose Landmark Extraction.

Test coverage:
1. Canonical 33-landmark ordering and names match MediaPipe specification.
2. Normalized coordinates validity and unusable coordinate handling.
3. Missing visibility or presence handling.
4. No-person detection status and output formatting.
5. Unreadable image input error status and warning generation.
6. Deterministic JSON serialization schema compliance.
7. Visualization safety (non-mutation of input array).
8. Context manager lifecycle and explicit resource cleanup.
9. Working directory independence for model asset resolution.
10. Optional smoke test loading the real Heavy model asset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.config import DEFAULT_MODEL_PATH
from fitloop_measurement.pose import (
    CANONICAL_POSE_LANDMARK_NAMES,
    ExtractionStatus,
    PoseExtractor,
    PoseLandmarkPoint,
    ViewType,
    draw_pose_landmarks,
)


def _make_mock_raw_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0, vis: float = 0.9, pres: float = 0.9):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = vis
    lm.presence = pres
    return lm


def _make_mock_detection_result(detected: bool = True):
    result = MagicMock()
    if not detected:
        result.pose_landmarks = []
        return result

    raw_lms = []
    for i in range(33):
        raw_lms.append(_make_mock_raw_landmark(x=0.5, y=i / 33.0, z=-0.1, vis=0.95, pres=0.95))
    result.pose_landmarks = [raw_lms]
    return result


@pytest.fixture
def dummy_bgr_img():
    return np.full((960, 640, 3), 120, dtype=np.uint8)


class TestPoseSchemasAndOrdering:
    def test_canonical_landmark_count_and_names(self):
        assert len(CANONICAL_POSE_LANDMARK_NAMES) == 33
        assert CANONICAL_POSE_LANDMARK_NAMES[0] == "NOSE"
        assert CANONICAL_POSE_LANDMARK_NAMES[11] == "LEFT_SHOULDER"
        assert CANONICAL_POSE_LANDMARK_NAMES[12] == "RIGHT_SHOULDER"
        assert CANONICAL_POSE_LANDMARK_NAMES[23] == "LEFT_HIP"
        assert CANONICAL_POSE_LANDMARK_NAMES[24] == "RIGHT_HIP"
        assert CANONICAL_POSE_LANDMARK_NAMES[32] == "RIGHT_FOOT_INDEX"

    def test_landmark_usability_filtering(self):
        valid = PoseLandmarkPoint(index=0, name="NOSE", x=0.5, y=0.5, z=0.0, visibility=0.9)
        assert valid.is_usable(min_visibility=0.5)

        low_vis = PoseLandmarkPoint(index=1, name="LEFT_EYE", x=0.5, y=0.5, z=0.0, visibility=0.2)
        assert not low_vis.is_usable(min_visibility=0.5)

        out_of_bounds = PoseLandmarkPoint(index=2, name="RIGHT_EYE", x=1.5, y=0.5, z=0.0, visibility=0.9)
        assert not out_of_bounds.is_usable(min_visibility=0.5)

        missing_coords = PoseLandmarkPoint(index=3, name="NOSE", x=None, y=None, z=None)
        assert not missing_coords.is_usable(min_visibility=0.5)


class TestPoseExtractorMocked:
    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_successful_extraction_front_view(self, mock_exists, mock_create, dummy_bgr_img):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _make_mock_detection_result(detected=True)

        extractor = PoseExtractor()
        result = extractor.extract(dummy_bgr_img, view_type=ViewType.FRONT)

        assert result.is_successful
        assert result.status == ExtractionStatus.SUCCESS
        assert result.view_type == "front"
        assert result.image_width == 640
        assert result.image_height == 960
        assert len(result.landmarks) == 33

        # Check sequential order matches specification
        for idx, lm in enumerate(result.landmarks):
            assert lm.index == idx
            assert lm.name == CANONICAL_POSE_LANDMARK_NAMES[idx]
            assert 0.0 <= lm.x <= 1.0
            assert 0.0 <= lm.y <= 1.0

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_no_person_detected(self, mock_exists, mock_create, dummy_bgr_img):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _make_mock_detection_result(detected=False)

        extractor = PoseExtractor()
        result = extractor.extract(dummy_bgr_img, view_type="side")

        assert not result.is_successful
        assert result.status == ExtractionStatus.NO_PERSON_DETECTED
        assert len(result.landmarks) == 0
        assert any("No person detected" in w for w in result.warnings)

    def test_unreadable_image_path(self):
        extractor = PoseExtractor()
        result = extractor.extract("/nonexistent/photo.jpg", view_type="front")

        assert result.status == ExtractionStatus.ERROR
        assert len(result.landmarks) == 0
        assert any("Failed to load image" in w for w in result.warnings)

    def test_invalid_view_type_raises_value_error(self, dummy_bgr_img):
        extractor = PoseExtractor()
        with pytest.raises(ValueError, match="Invalid view_type"):
            extractor.extract(dummy_bgr_img, view_type="diagonal")

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_json_serialization_compatibility(self, mock_exists, mock_create, dummy_bgr_img):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _make_mock_detection_result(detected=True)

        extractor = PoseExtractor()
        result = extractor.extract(dummy_bgr_img, view_type="front")

        out_dict = result.to_dict()
        assert out_dict["view_type"] == "front"
        assert out_dict["status"] == "SUCCESS"
        assert len(out_dict["landmarks"]) == 33

        # Must cleanly serialize to standard JSON string
        json_str = json.dumps(out_dict)
        deserialized = json.loads(json_str)
        assert deserialized["landmarks"][0]["name"] == "NOSE"
        assert isinstance(deserialized["landmarks"][0]["x"], float)

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_resource_cleanup_in_context_manager(self, mock_exists, mock_create, dummy_bgr_img):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_landmarker.detect.return_value = _make_mock_detection_result(detected=True)

        with PoseExtractor() as extractor:
            extractor.extract(dummy_bgr_img)
            assert extractor._landmarker is not None

        mock_landmarker.close.assert_called_once()
        assert extractor._landmarker is None


class TestPoseVisualization:
    def test_visualization_does_not_mutate_original(self, dummy_bgr_img):
        original_copy = dummy_bgr_img.copy()

        # Build a synthetic successful result
        points = [
            PoseLandmarkPoint(index=i, name=CANONICAL_POSE_LANDMARK_NAMES[i], x=0.5, y=0.5, visibility=0.9)
            for i in range(33)
        ]
        from fitloop_measurement.pose.schemas import PoseExtractionResult

        result = PoseExtractionResult(
            view_type="front",
            image_width=640,
            image_height=960,
            model_id="test-model",
            status=ExtractionStatus.SUCCESS,
            landmarks=points,
        )

        annotated = draw_pose_landmarks(dummy_bgr_img, result)

        assert not np.array_equal(dummy_bgr_img, annotated), "Annotated image should have visual markings"
        assert np.array_equal(dummy_bgr_img, original_copy), "Source input image must remain unmodified"

    def test_visualization_on_failed_result_returns_copy_unaltered(self, dummy_bgr_img):
        from fitloop_measurement.pose.schemas import PoseExtractionResult

        result = PoseExtractionResult(
            view_type="front",
            image_width=640,
            image_height=960,
            model_id="test-model",
            status=ExtractionStatus.NO_PERSON_DETECTED,
            landmarks=[],
        )

        annotated = draw_pose_landmarks(dummy_bgr_img, result)
        assert np.array_equal(dummy_bgr_img, annotated)


class TestModelAssetResolutionAndSmoke:
    def test_default_model_path_resolves_and_exists(self):
        assert DEFAULT_MODEL_PATH.is_absolute()
        assert DEFAULT_MODEL_PATH.exists()
        assert DEFAULT_MODEL_PATH.stat().st_size > 10 * 1024 * 1024  # > 10 MB

    def test_smoke_load_real_heavy_model_asset(self, dummy_bgr_img):
        """Smoke test loading and evaluating synthetic input with the real downloaded Heavy model."""
        with PoseExtractor() as extractor:
            # Synthetic flat image has no person, should cleanly return NO_PERSON_DETECTED without crashing
            res = extractor.extract(dummy_bgr_img, view_type="front")
            assert res.status in {ExtractionStatus.NO_PERSON_DETECTED, ExtractionStatus.SUCCESS}
            assert res.model_id == "mediapipe-pose-landmarker-heavy-v1"
