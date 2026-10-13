"""
pose/extractor.py
-----------------
MediaPipe Tasks PoseLandmarker extraction engine.

Extracts canonical 33 body landmarks from single front or side photographs,
supporting path inputs or raw BGR numpy arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import mediapipe as mp
import numpy as np

from ..config import (
    DEFAULT_MODEL_PATH,
    LANDMARK_VISIBILITY_THRESHOLD,
    MEDIAPIPE_MIN_DETECTION_CONF,
    MEDIAPIPE_MIN_PRESENCE_CONF,
    MEDIAPIPE_MIN_TRACKING_CONF,
)
from ..image_io import ImageLoadError, image_to_rgb, load_image
from .schemas import (
    CANONICAL_POSE_LANDMARK_NAMES,
    ExtractionStatus,
    PoseExtractionResult,
    PoseLandmarkPoint,
    ViewType,
)

logger = logging.getLogger(__name__)


class PoseExtractor:
    """Manages lifecycle of MediaPipe PoseLandmarker for extracting 33 body keypoints."""

    MODEL_IDENTIFIER: str = "mediapipe-pose-landmarker-heavy-v1"

    def __init__(
        self,
        model_asset_path: Optional[Union[str, Path]] = None,
        min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONF,
        min_presence_confidence: float = MEDIAPIPE_MIN_PRESENCE_CONF,
        min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONF,
    ) -> None:
        self.model_path = Path(model_asset_path or DEFAULT_MODEL_PATH).resolve()
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._landmarker = None

    def _get_landmarker(self):
        """Lazy-initialize MediaPipe Tasks PoseLandmarker."""
        if self._landmarker is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"MediaPipe PoseLandmarker asset not found at: {self.model_path}. "
                    "Ensure models/pose_landmarker_heavy.task is present."
                )

            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
        return self._landmarker

    def extract(
        self,
        image_input: Union[str, Path, np.ndarray],
        view_type: Union[ViewType, str] = ViewType.FRONT,
    ) -> PoseExtractionResult:
        """Extract canonical 33 pose landmarks from a single view.

        Parameters
        ----------
        image_input:
            File path or BGR uint8 numpy array.
        view_type:
            Perspective of the capture ('front' or 'side').

        Returns
        -------
        PoseExtractionResult
            Structured result containing canonical 33 landmarks or explicit failure status.
        """
        # Normalize view_type
        if isinstance(view_type, ViewType):
            v_type_str = view_type.value
        else:
            v_type_str = str(view_type).lower().strip()
            if v_type_str not in {"front", "side"}:
                raise ValueError(f"Invalid view_type '{view_type}'. Must be 'front' or 'side'.")

        # Load / resolve image
        if isinstance(image_input, (str, Path)):
            try:
                bgr_image = load_image(image_input)
            except ImageLoadError as e:
                return PoseExtractionResult(
                    view_type=v_type_str,
                    image_width=0,
                    image_height=0,
                    model_id=self.MODEL_IDENTIFIER,
                    status=ExtractionStatus.ERROR,
                    landmarks=[],
                    warnings=[f"Failed to load image: {e}"],
                )
            except Exception as e:
                return PoseExtractionResult(
                    view_type=v_type_str,
                    image_width=0,
                    image_height=0,
                    model_id=self.MODEL_IDENTIFIER,
                    status=ExtractionStatus.ERROR,
                    landmarks=[],
                    warnings=[f"Unexpected error loading image: {e}"],
                )
        elif isinstance(image_input, np.ndarray):
            bgr_image = image_input
        else:
            return PoseExtractionResult(
                view_type=v_type_str,
                image_width=0,
                image_height=0,
                model_id=self.MODEL_IDENTIFIER,
                status=ExtractionStatus.ERROR,
                landmarks=[],
                warnings=[f"Unsupported image input type: {type(image_input)}"],
            )

        h, w = bgr_image.shape[:2]

        # Convert to MediaPipe Image
        rgb = image_to_rgb(bgr_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            landmarker = self._get_landmarker()
            results = landmarker.detect(mp_image)
        except Exception as e:
            logger.error("MediaPipe detection failed: %s", e)
            return PoseExtractionResult(
                view_type=v_type_str,
                image_width=w,
                image_height=h,
                model_id=self.MODEL_IDENTIFIER,
                status=ExtractionStatus.ERROR,
                landmarks=[],
                warnings=[f"MediaPipe detection failed: {e}"],
            )

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return PoseExtractionResult(
                view_type=v_type_str,
                image_width=w,
                image_height=h,
                model_id=self.MODEL_IDENTIFIER,
                status=ExtractionStatus.NO_PERSON_DETECTED,
                landmarks=[],
                warnings=["No person detected in the provided image."],
            )

        raw_landmarks = results.pose_landmarks[0]
        extracted_points: list[PoseLandmarkPoint] = []
        warnings: list[str] = []

        low_visibility_count = 0
        for i, canonical_name in enumerate(CANONICAL_POSE_LANDMARK_NAMES):
            if i < len(raw_landmarks):
                lm = raw_landmarks[i]
                vis = getattr(lm, "visibility", None)
                pres = getattr(lm, "presence", None)
                z_val = getattr(lm, "z", None)

                point = PoseLandmarkPoint(
                    index=i,
                    name=canonical_name,
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(z_val) if z_val is not None else None,
                    visibility=float(vis) if vis is not None else None,
                    presence=float(pres) if pres is not None else None,
                )
                if vis is not None and vis < LANDMARK_VISIBILITY_THRESHOLD:
                    low_visibility_count += 1
            else:
                # Landmark missing from output
                point = PoseLandmarkPoint(
                    index=i,
                    name=canonical_name,
                    x=None,
                    y=None,
                    z=None,
                    visibility=None,
                    presence=None,
                )
                warnings.append(f"Landmark {canonical_name} ({i}) was missing from detector output.")

            extracted_points.append(point)

        status = ExtractionStatus.SUCCESS
        if low_visibility_count > 10:
            warnings.append(
                f"{low_visibility_count} of 33 landmarks have low visibility (< {LANDMARK_VISIBILITY_THRESHOLD})."
            )
            # In side view, higher occlusion of contralateral side is expected
            if v_type_str == "side" and low_visibility_count <= 16:
                pass  # Typical anatomical occlusion in side view
            elif low_visibility_count > 16:
                status = ExtractionStatus.LOW_CONFIDENCE

        return PoseExtractionResult(
            view_type=v_type_str,
            image_width=w,
            image_height=h,
            model_id=self.MODEL_IDENTIFIER,
            status=status,
            landmarks=extracted_points,
            warnings=warnings,
        )

    def close(self) -> None:
        """Release underlying landmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()
