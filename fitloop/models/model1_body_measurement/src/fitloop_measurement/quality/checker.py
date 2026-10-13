"""
quality/checker.py
------------------
Image quality gate for the measurement pipeline using MediaPipe Tasks Vision.

Checks (in order):
    1. Image dimensions meet the minimum.
    2. A person is detected and the body occupies sufficient image height.
    3. The body midpoint is near the horizontal centre.
    4. A known height has been supplied and is within plausible range.

Each check appends a human-readable message to ``QualityReport.issues``.
A report is *passed* only when ``QualityReport.issues`` is empty.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np

from ..config import (
    DEFAULT_MODEL_PATH,
    HEIGHT_MAX_CM,
    HEIGHT_MIN_CM,
    LANDMARK_VISIBILITY_THRESHOLD,
    LEFT_HIP_IDX,
    MAX_CENTER_OFFSET_FRACTION,
    MEDIAPIPE_MIN_DETECTION_CONF,
    MEDIAPIPE_MIN_PRESENCE_CONF,
    MEDIAPIPE_MIN_TRACKING_CONF,
    MIN_BODY_HEIGHT_FRACTION,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    NOSE_IDX,
    RIGHT_HIP_IDX,
)
from ..image_io import image_to_rgb

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class QualityReport:
    """Result of a quality check run."""

    passed: bool
    issues: list[str] = dataclasses.field(default_factory=list)
    info: dict = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"QualityReport [{status}]"]
        for issue in self.issues:
            lines.append(f"  ✗ {issue}")
        for k, v in self.info.items():
            lines.append(f"  · {k}: {v}")
        return "\n".join(lines)


class QualityChecker:
    """Stateless quality gate using MediaPipe Tasks PoseLandmarker."""

    def __init__(self, model_asset_path: Optional[str | Path] = None) -> None:
        self.model_path = Path(model_asset_path or DEFAULT_MODEL_PATH)
        self._landmarker = None

    def _get_landmarker(self):
        if self._landmarker is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"PoseLandmarker task model not found at {self.model_path}. "
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
                min_pose_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONF,
                min_pose_presence_confidence=MEDIAPIPE_MIN_PRESENCE_CONF,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONF,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
        return self._landmarker

    def run(
        self,
        bgr_image: np.ndarray,
        subject_height_cm: Optional[float] = None,
    ) -> QualityReport:
        """Run all quality checks against *bgr_image*."""
        issues: list[str] = []
        info: dict = {}

        h, w = bgr_image.shape[:2]
        info["resolution"] = f"{w}×{h}"

        # 1. Minimum resolution
        if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
            issues.append(
                f"Image too small ({w}×{h}). "
                f"Need ≥ {MIN_IMAGE_WIDTH}×{MIN_IMAGE_HEIGHT}."
            )

        # 2. Pose detection
        rgb = image_to_rgb(bgr_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker = self._get_landmarker()
        results = landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            issues.append(
                "No person detected. Ensure the full body is visible against "
                "a plain background and the image is well-lit."
            )
            return QualityReport(passed=False, issues=issues, info=info)

        lm = results.pose_landmarks[0]

        # 3. Body height fraction
        visible = [
            l for l in lm
            if getattr(l, "visibility", 1.0) >= LANDMARK_VISIBILITY_THRESHOLD
        ]
        if not visible:
            issues.append("All landmarks have low visibility — is the person occluded?")
        else:
            ys = [l.y for l in visible]
            body_frac = max(ys) - min(ys)
            info["body_height_fraction"] = f"{body_frac:.2f}"
            if body_frac < MIN_BODY_HEIGHT_FRACTION:
                issues.append(
                    f"Body occupies only {body_frac:.0%} of image height "
                    f"(need ≥ {MIN_BODY_HEIGHT_FRACTION:.0%}). "
                    "Step back or zoom out so the full body fits in the frame."
                )

        # 4. Horizontal centering
        l_hip = lm[LEFT_HIP_IDX]
        r_hip = lm[RIGHT_HIP_IDX]
        mid_x = (l_hip.x + r_hip.x) / 2.0
        offset = abs(mid_x - 0.5)
        info["body_center_offset"] = f"{offset:.2f}"
        if offset > MAX_CENTER_OFFSET_FRACTION:
            issues.append(
                f"Body midpoint is {offset:.0%} off-centre "
                f"(limit {MAX_CENTER_OFFSET_FRACTION:.0%}). "
                "Centre yourself in the frame."
            )

        # 5. Height calibration
        if subject_height_cm is None:
            issues.append(
                "subject_height_cm not supplied. Scale calibration is impossible "
                "without the subject's known height."
            )
        elif not (HEIGHT_MIN_CM <= subject_height_cm <= HEIGHT_MAX_CM):
            issues.append(
                f"subject_height_cm={subject_height_cm} is outside the plausible "
                f"range [{HEIGHT_MIN_CM}, {HEIGHT_MAX_CM}] cm."
            )
        else:
            info["subject_height_cm"] = subject_height_cm

        passed = len(issues) == 0
        report = QualityReport(passed=passed, issues=issues, info=info)
        logger.debug(report)
        return report

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "QualityChecker":
        return self

    def __exit__(self, *_) -> None:
        self.close()
