"""
service.py
----------
Orchestrates the Model 1 body measurement pipeline end-to-end:
1. Validates inputs and decode front/side images via OpenCV.
2. Extracts 33 canonical MediaPipe pose landmarks (PoseExtractor).
3. Segments body silhouette and extracts principal contour (BodySegmenter).
4. Extracts deterministic geometric features and cross-view alignment (GeometricFeatureExtractor).
5. Calibrates independent metric scales and estimates circumferences via Ramanujan ellipse (CircumferenceEstimator).

Never fabricates measurements or replaces failed predictions with zeros.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from fitloop_measurement.config import HEIGHT_MAX_CM, HEIGHT_MIN_CM
from fitloop_measurement.features import GeometricFeatureExtractor
from fitloop_measurement.measurements import CircumferenceEstimator
from fitloop_measurement.measurements.schemas import (
    MeasurementEngineResult,
    MeasurementStatus,
)
from fitloop_measurement.pose import PoseExtractor
from fitloop_measurement.segmentation import BodySegmenter

logger = logging.getLogger(__name__)


def decode_image_bytes(image_bytes: bytes, view_name: str) -> np.ndarray:
    """Decode raw image bytes into a BGR uint8 NumPy array.

    Raises
    ------
    ValueError:
        If bytes are empty, truncated, or cannot be decoded as an image.
    """
    if not image_bytes:
        raise ValueError(f"Received empty file buffer for {view_name} image.")

    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr_img is None or bgr_img.size == 0:
        raise ValueError(f"Could not decode {view_name} image. File may be corrupted or not a valid image format.")

    h, w = bgr_img.shape[:2]
    if w < 480 or h < 640:
        raise ValueError(
            f"{view_name.capitalize()} image resolution ({w}x{h}) is below minimum requirement of 480x640."
        )

    return bgr_img


class MeasurementPipelineService:
    """Orchestrates end-to-end Model 1 execution."""

    def __init__(self) -> None:
        self.pose_extractor = PoseExtractor()
        self.segmenter = BodySegmenter()
        self.feature_extractor = GeometricFeatureExtractor()
        self.estimator = CircumferenceEstimator()

    def process_pair(
        self,
        front_image: np.ndarray,
        side_image: np.ndarray,
        standing_height_cm: float,
    ) -> Dict[str, Any]:
        """Execute end-to-end measurement estimation on paired BGR images.

        Parameters
        ----------
        front_image:
            Decoded BGR NumPy array for front perspective.
        side_image:
            Decoded BGR NumPy array for side perspective.
        standing_height_cm:
            Barefoot standing height in centimetres.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable measurement result dictionary matching frontend contract.
        """
        # Validate height range
        if standing_height_cm < HEIGHT_MIN_CM or standing_height_cm > HEIGHT_MAX_CM:
            raise ValueError(
                f"Standing height ({standing_height_cm} cm) is outside valid human stature range [{HEIGHT_MIN_CM}, {HEIGHT_MAX_CM}] cm."
            )

        pipeline_warnings: List[str] = []

        # 1. Pose Extraction (Stage 3A)
        front_pose = self.pose_extractor.extract(front_image, view_type="front")
        side_pose = self.pose_extractor.extract(side_image, view_type="side")

        if not front_pose.is_successful:
            pipeline_warnings.append(f"Front pose detection issue: {front_pose.status.value}")
        if not side_pose.is_successful:
            pipeline_warnings.append(f"Side pose detection issue: {side_pose.status.value}")

        # 2. Silhouette Segmentation (Stage 3B)
        front_seg = self.segmenter.segment(front_image, view_type="front")
        side_seg = self.segmenter.segment(side_image, view_type="side")

        if not front_seg.is_successful:
            pipeline_warnings.append(f"Front silhouette segmentation issue: {front_seg.status.value}")
        if not side_seg.is_successful:
            pipeline_warnings.append(f"Side silhouette segmentation issue: {side_seg.status.value}")

        # 3. Geometric Feature Extraction (Stage 4A)
        front_geom = self.feature_extractor.extract_view_features(front_pose, front_seg)
        side_geom = self.feature_extractor.extract_view_features(side_pose, side_seg)

        paired_geom = self.feature_extractor.extract_paired_features(
            front_features=front_geom,
            side_features=side_geom,
            expected_height_cm=standing_height_cm,
        )

        # 4. Calibrated Circumference Estimation (Stage 4B)
        result: MeasurementEngineResult = self.estimator.estimate_measurements(
            paired_features=paired_geom,
            known_height_cm=standing_height_cm,
            front_mask=front_seg.mask if front_seg.is_successful else None,
            side_mask=side_seg.mask if side_seg.is_successful else None,
        )

        # 5. Capture Quality Gate (Stage 7G)
        # Note: The quality gate runs here after feature extraction and estimation so that it
        # can inspect the full pipeline diagnostics (including cross-view slice alignment
        # and calibration consistency) while preserving diagnostic measurements for debugging.
        quality_gate = getattr(self, "quality_gate", None)
        if quality_gate is None:
            from fitloop_measurement.quality import CaptureQualityGate
            self.quality_gate = CaptureQualityGate()
            quality_gate = self.quality_gate

        quality_report = quality_gate.evaluate_pair(
            front_features=front_geom,
            side_features=side_geom,
            calibration=result.calibration,
            front_pose=front_pose,
            side_pose=side_pose,
            front_seg=front_seg,
            side_seg=side_seg,
            measurement_result=result,
            standing_height_cm=standing_height_cm,
        )

        out_dict = result.to_dict()
        out_dict["is_mock"] = False
        out_dict["quality_gate"] = quality_report.to_dict()
        out_dict["quality_status"] = quality_report.status.value

        if quality_report.status.value == "REJECT":
            pipeline_warnings.append(
                "CAPTURE QUALITY GATE REJECT: One or more critical capture quality violations were detected. "
                "Returned measurements are diagnostic-only and MUST NOT be treated as quality-approved estimates."
            )

        out_dict["warnings"] = list(set(out_dict.get("warnings", []) + pipeline_warnings))

        return out_dict

    def close(self) -> None:
        """Release underlying vision resources."""
        if hasattr(self.pose_extractor, "close"):
            self.pose_extractor.close()
        if hasattr(self.segmenter, "close"):
            self.segmenter.close()

    def __enter__(self) -> "MeasurementPipelineService":
        return self

    def __exit__(self, *_) -> None:
        self.close()
