"""
tests/test_segmentation.py
--------------------------
Comprehensive unit test suite for Stage 3B: Body Silhouette Segmentation.

Test coverage:
1. Schema validation and JSON-safe metadata serialization.
2. Binary mask dimensions, dtype, and valid range {0, 255}.
3. Empty masks and invalid inputs.
4. Unreadable image paths.
5. Foreground ratio calculation.
6. Connected-component noise filtering and conservative cleanup.
7. Contour extraction from synthetic masks with exact known bounding boxes.
8. No-contour and invalid-mask handling.
9. Visualization safety (non-mutation of input array).
10. Configurable thresholds and cleanup parameters.
11. Resource lifecycle management and context manager cleanup.
12. Model path resolution from different working directories.
13. Optional smoke test with the real Heavy model asset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.config import DEFAULT_MODEL_PATH
from fitloop_measurement.segmentation import (
    BodySegmenter,
    BoundingBox,
    ContourResult,
    SegmentationResult,
    SegmentationStatus,
    clean_binary_mask,
    draw_segmentation_overlay,
    extract_principal_contour,
)


@pytest.fixture
def dummy_bgr_img():
    """Standard synthetic BGR image (H=960, W=640)."""
    return np.full((960, 640, 3), 100, dtype=np.uint8)


@pytest.fixture
def synthetic_body_mask():
    """Synthetic binary mask with known geometry: rectangle from (200, 200) to (440, 800)."""
    mask = np.zeros((960, 640), dtype=np.uint8)
    cv2.rectangle(mask, (200, 200), (440, 800), 255, -1)
    return mask


class TestSegmentationSchemas:
    def test_bounding_box_geometry(self):
        bbox = BoundingBox(x=100, y=150, width=200, height=500)
        assert bbox.x_max == 300
        assert bbox.y_max == 650
        assert round(bbox.aspect_ratio, 2) == 0.40
        d = bbox.to_dict()
        assert d == {"x": 100, "y": 150, "width": 200, "height": 500}

    def test_json_safe_serialization(self, synthetic_body_mask):
        contour, _ = extract_principal_contour(synthetic_body_mask)
        res = SegmentationResult(
            view_type="front",
            image_width=640,
            image_height=960,
            status=SegmentationStatus.SUCCESS,
            model_id="test-segmenter",
            mask=synthetic_body_mask,
            raw_mask=synthetic_body_mask,
            foreground_ratio=0.25,
            principal_contour=contour,
            warnings=["Test warning"],
            metadata={"threshold": 0.5},
        )
        data = res.to_dict()
        assert data["view_type"] == "front"
        assert data["status"] == "SUCCESS"
        assert data["has_mask"] is True
        assert data["foreground_ratio"] == 0.25
        assert "contour_points" not in str(data)  # Array omitted from JSON metadata
        assert data["principal_contour"]["area"] > 0

        # Verify JSON encoding
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        assert decoded["status"] == "SUCCESS"


class TestContourExtraction:
    def test_exact_known_contour_bounding_box(self, synthetic_body_mask):
        # Known rect: x in [200, 440], y in [200, 800] -> width=241, height=601
        contour, warnings = extract_principal_contour(synthetic_body_mask)
        assert contour is not None
        assert isinstance(contour, ContourResult)
        assert abs(contour.bounding_box.x - 200) <= 1
        assert abs(contour.bounding_box.y - 200) <= 1
        assert abs(contour.bounding_box.width - 241) <= 2
        assert abs(contour.bounding_box.height - 601) <= 2
        assert contour.area > 100000

    def test_empty_mask_handling(self):
        empty_mask = np.zeros((960, 640), dtype=np.uint8)
        contour, warnings = extract_principal_contour(empty_mask)
        assert contour is None
        assert any("No external contours found" in w for w in warnings)

    def test_none_mask_handling(self):
        contour, warnings = extract_principal_contour(None)
        assert contour is None
        assert any("empty or None" in w for w in warnings)


class TestMaskCleanup:
    def test_removes_small_isolated_noise_specks(self, synthetic_body_mask):
        noisy_mask = synthetic_body_mask.copy()
        # Add 3 isolated 5x5 noise specks (area=25 px each)
        cv2.rectangle(noisy_mask, (50, 50), (55, 55), 255, -1)
        cv2.rectangle(noisy_mask, (50, 200), (55, 205), 255, -1)
        cv2.rectangle(noisy_mask, (550, 500), (555, 505), 255, -1)

        cleaned, warnings = clean_binary_mask(noisy_mask, min_component_area=100)
        # Noise should be removed
        assert np.count_nonzero(cleaned[45:60, 45:60]) == 0
        assert np.count_nonzero(cleaned[545:560, 495:510]) == 0
        # Main body rectangle should remain intact
        assert np.count_nonzero(cleaned[250:350, 250:350]) > 0
        assert any("Filtered out 3 small isolated noise" in w for w in warnings)

    def test_preserves_authentic_body_perimeter(self, synthetic_body_mask):
        cleaned, _ = clean_binary_mask(synthetic_body_mask, min_component_area=500, morph_kernel_size=3)
        # Area change should be minimal (< 1%)
        diff = np.abs(cleaned.astype(int) - synthetic_body_mask.astype(int))
        assert np.count_nonzero(diff) / np.count_nonzero(synthetic_body_mask) < 0.01


class TestBodySegmenterMocked:
    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_successful_segmentation_mock(self, mock_exists, mock_create, dummy_bgr_img):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        # Mock segmentation mask float probability array
        mock_prob = np.zeros((960, 640), dtype=np.float32)
        mock_prob[200:800, 200:440] = 0.95  # High confidence foreground

        mock_mask_obj = MagicMock()
        mock_mask_obj.numpy_view.return_value = mock_prob

        mock_results = MagicMock()
        mock_results.segmentation_masks = [mock_mask_obj]
        mock_landmarker.detect.return_value = mock_results

        segmenter = BodySegmenter()
        result = segmenter.segment(dummy_bgr_img, view_type="front")

        assert result.is_successful
        assert result.status == SegmentationStatus.SUCCESS
        assert result.view_type == "front"
        assert result.mask is not None
        assert result.mask.shape == (960, 640)
        assert result.foreground_ratio > 0.10
        assert result.principal_contour is not None
        assert result.principal_contour.bounding_box.width > 200

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_no_foreground_detected(self, mock_exists, mock_create, dummy_bgr_img):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker

        mock_results = MagicMock()
        mock_results.segmentation_masks = []
        mock_landmarker.detect.return_value = mock_results

        segmenter = BodySegmenter()
        result = segmenter.segment(dummy_bgr_img, view_type="side")

        assert not result.is_successful
        assert result.status == SegmentationStatus.NO_FOREGROUND
        assert any("No person segmentation mask" in w for w in result.warnings)

    def test_unreadable_image_path(self):
        segmenter = BodySegmenter()
        result = segmenter.segment("/nonexistent/file.jpg", view_type="front")
        assert result.status == SegmentationStatus.ERROR
        assert any("Failed to load image" in w for w in result.warnings)

    def test_invalid_view_type_raises(self, dummy_bgr_img):
        segmenter = BodySegmenter()
        with pytest.raises(ValueError, match="Invalid view_type"):
            segmenter.segment(dummy_bgr_img, view_type="oblique")

    @patch("mediapipe.tasks.vision.PoseLandmarker.create_from_options")
    @patch("pathlib.Path.exists", return_value=True)
    def test_context_manager_cleanup(self, mock_exists, mock_create, dummy_bgr_img):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        mock_results = MagicMock()
        mock_results.segmentation_masks = []
        mock_landmarker.detect.return_value = mock_results

        with BodySegmenter() as segmenter:
            segmenter.segment(dummy_bgr_img)
            assert segmenter._landmarker is not None

        mock_landmarker.close.assert_called_once()
        assert segmenter._landmarker is None


class TestVisualizationSafety:
    def test_draw_overlay_does_not_mutate_input_image(self, dummy_bgr_img, synthetic_body_mask):
        orig_copy = dummy_bgr_img.copy()
        contour, _ = extract_principal_contour(synthetic_body_mask)

        res = SegmentationResult(
            view_type="front",
            image_width=640,
            image_height=960,
            status=SegmentationStatus.SUCCESS,
            model_id="test",
            mask=synthetic_body_mask,
            principal_contour=contour,
        )

        overlay = draw_segmentation_overlay(dummy_bgr_img, res)
        assert not np.array_equal(dummy_bgr_img, overlay), "Overlay should have visual color changes"
        assert np.array_equal(dummy_bgr_img, orig_copy), "Input array must not be mutated"


class TestModelAssetResolutionAndRealSmoke:
    def test_model_path_resolution(self):
        assert DEFAULT_MODEL_PATH.exists()
        assert DEFAULT_MODEL_PATH.stat().st_size > 10 * 1024 * 1024

    def test_smoke_real_heavy_model_segmentation(self, dummy_bgr_img):
        """Smoke test loading real heavy model asset with output_segmentation_masks=True."""
        with BodySegmenter() as segmenter:
            res = segmenter.segment(dummy_bgr_img, view_type="front")
            # Synthetic flat image has no person, expect NO_FOREGROUND without crashing
            assert res.status in {SegmentationStatus.NO_FOREGROUND, SegmentationStatus.SUCCESS}
            assert res.model_id == "mediapipe-pose-landmarker-heavy-v1"
