"""
segmentation/segmenter.py
-------------------------
Body silhouette segmentation engine using MediaPipe Tasks PoseLandmarker.

Generates a binary foreground mask, performs conservative morphological
cleanup (preserving anatomical limb boundaries), and extracts the principal contour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import mediapipe as mp
import numpy as np

from ..config import DEFAULT_MODEL_PATH
from ..image_io import ImageLoadError, image_to_rgb, load_image
from .contours import extract_principal_contour
from .schemas import SegmentationResult, SegmentationStatus

logger = logging.getLogger(__name__)


def clean_binary_mask(
    raw_mask: np.ndarray,
    min_component_area: int = 500,
    morph_kernel_size: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Conservatively clean a binary mask without eroding or dilating body boundaries.

    Technique:
    1. Small component removal (filtering isolated background noise specks).
    2. Subtle 3x3 close operation to fill small pinhole artifacts in solid clothing/torso.
       Note: We avoid large morphological kernels to preserve authentic body outline widths.

    Parameters
    ----------
    raw_mask:
        Binary mask (uint8) with values {0, 255}.
    min_component_area:
        Connected component area threshold below which isolated islands are removed.
    morph_kernel_size:
        Kernel dimension for conservative closing (default 3x3).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Cleaned mask and diagnostic warnings.
    """
    warnings: list[str] = []
    if raw_mask is None or raw_mask.size == 0:
        return raw_mask, ["Empty mask provided to clean_binary_mask."]

    cleaned = (raw_mask > 0).astype(np.uint8) * 255

    # 1. Connected components analysis to remove isolated noise specks
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    if num_labels <= 1:
        # Only background
        return cleaned, ["Mask contains only background."]

    # stats columns: [x, y, width, height, area]
    # Filter components smaller than min_component_area (excluding label 0 = background)
    filtered_mask = np.zeros_like(cleaned)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered_mask[labels == label] = 255

    removed_count = sum(
        1 for label in range(1, num_labels) if stats[label, cv2.CC_STAT_AREA] < min_component_area
    )
    if removed_count > 0:
        warnings.append(
            f"Filtered out {removed_count} small isolated noise island(s) (< {min_component_area} px)."
        )

    # 2. Conservative closing to heal small interior pinholes without altering perimeter
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        closed_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    else:
        closed_mask = filtered_mask

    return closed_mask, warnings


class BodySegmenter:
    """Reusable person silhouette segmenter using MediaPipe PoseLandmarker."""

    MODEL_IDENTIFIER: str = "mediapipe-pose-landmarker-heavy-v1"

    def __init__(
        self,
        model_asset_path: Optional[Union[str, Path]] = None,
        foreground_threshold: float = 0.50,
        enable_mask_cleanup: bool = True,
        min_component_area: int = 500,
        morph_kernel_size: int = 3,
    ) -> None:
        self.model_path = Path(model_asset_path or DEFAULT_MODEL_PATH).resolve()
        self.foreground_threshold = foreground_threshold
        self.enable_mask_cleanup = enable_mask_cleanup
        self.min_component_area = min_component_area
        self.morph_kernel_size = morph_kernel_size
        self._landmarker = None

    def _get_landmarker(self):
        """Lazy-initialize MediaPipe PoseLandmarker with output_segmentation_masks enabled."""
        if self._landmarker is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"PoseLandmarker asset not found at {self.model_path}. "
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
                output_segmentation_masks=True,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
        return self._landmarker

    def segment(
        self,
        image_input: Union[str, Path, np.ndarray],
        view_type: str = "front",
    ) -> SegmentationResult:
        """Extract foreground silhouette mask and contour from a photograph.

        Parameters
        ----------
        image_input:
            Path to image or BGR numpy array.
        view_type:
            Perspective of the image ('front' or 'side').

        Returns
        -------
        SegmentationResult
            Structured result containing binary mask, contour, foreground ratio, and metadata.
        """
        # Validate view_type
        v_type_clean = str(view_type).lower().strip()
        if v_type_clean not in {"front", "side"}:
            raise ValueError(f"Invalid view_type '{view_type}'. Must be 'front' or 'side'.")

        # Load / validate image
        if isinstance(image_input, (str, Path)):
            try:
                bgr_image = load_image(image_input)
            except ImageLoadError as e:
                return SegmentationResult(
                    view_type=v_type_clean,
                    image_width=0,
                    image_height=0,
                    status=SegmentationStatus.ERROR,
                    model_id=self.MODEL_IDENTIFIER,
                    warnings=[f"Failed to load image: {e}"],
                )
            except Exception as e:
                return SegmentationResult(
                    view_type=v_type_clean,
                    image_width=0,
                    image_height=0,
                    status=SegmentationStatus.ERROR,
                    model_id=self.MODEL_IDENTIFIER,
                    warnings=[f"Unexpected error loading image: {e}"],
                )
        elif isinstance(image_input, np.ndarray):
            bgr_image = image_input
        else:
            return SegmentationResult(
                view_type=v_type_clean,
                image_width=0,
                image_height=0,
                status=SegmentationStatus.ERROR,
                model_id=self.MODEL_IDENTIFIER,
                warnings=[f"Unsupported image input type: {type(image_input)}"],
            )

        h, w = bgr_image.shape[:2]
        rgb = image_to_rgb(bgr_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            landmarker = self._get_landmarker()
            detect_results = landmarker.detect(mp_image)
        except Exception as e:
            logger.error("Segmentation inference failed: %s", e)
            return SegmentationResult(
                view_type=v_type_clean,
                image_width=w,
                image_height=h,
                status=SegmentationStatus.ERROR,
                model_id=self.MODEL_IDENTIFIER,
                warnings=[f"Segmentation inference failed: {e}"],
            )

        # Check if segmentation masks were returned
        if not detect_results.segmentation_masks or len(detect_results.segmentation_masks) == 0:
            return SegmentationResult(
                view_type=v_type_clean,
                image_width=w,
                image_height=h,
                status=SegmentationStatus.NO_FOREGROUND,
                model_id=self.MODEL_IDENTIFIER,
                warnings=["No person segmentation mask detected."],
            )

        # Extract float confidence mask
        raw_mask_obj = detect_results.segmentation_masks[0]
        confidence_mask = raw_mask_obj.numpy_view()

        # If mask shape doesn't match original dimensions, resize probability map
        if confidence_mask.shape[:2] != (h, w):
            confidence_mask = cv2.resize(
                confidence_mask, (w, h), interpolation=cv2.INTER_LINEAR
            )

        # Apply configurable foreground threshold
        raw_binary = (confidence_mask >= self.foreground_threshold).astype(np.uint8) * 255

        total_pixels = h * w
        fg_pixels = int(np.count_nonzero(raw_binary))
        fg_ratio = float(fg_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

        if fg_pixels == 0 or fg_ratio < 0.01:
            return SegmentationResult(
                view_type=v_type_clean,
                image_width=w,
                image_height=h,
                status=SegmentationStatus.NO_FOREGROUND,
                model_id=self.MODEL_IDENTIFIER,
                raw_mask=raw_binary,
                mask=raw_binary,
                foreground_ratio=fg_ratio,
                warnings=["Silhouette mask contains negligible or zero foreground."],
            )

        all_warnings: list[str] = []

        # Optional conservative post-processing cleanup
        if self.enable_mask_cleanup:
            cleaned_mask, cleanup_warnings = clean_binary_mask(
                raw_binary,
                min_component_area=self.min_component_area,
                morph_kernel_size=self.morph_kernel_size,
            )
            all_warnings.extend(cleanup_warnings)
        else:
            cleaned_mask = raw_binary

        # Contour extraction
        principal_contour, contour_warnings = extract_principal_contour(cleaned_mask)
        all_warnings.extend(contour_warnings)

        # Low confidence warning if foreground ratio is abnormally small or large
        status = SegmentationStatus.SUCCESS
        if fg_ratio < 0.05:
            all_warnings.append(
                f"Low foreground occupancy ({fg_ratio:.1%}). Person may be too far or occluded."
            )
            status = SegmentationStatus.LOW_CONFIDENCE
        elif fg_ratio > 0.85:
            all_warnings.append(
                f"Abnormally high foreground occupancy ({fg_ratio:.1%}). Background may be leaking into mask."
            )
            status = SegmentationStatus.LOW_CONFIDENCE

        return SegmentationResult(
            view_type=v_type_clean,
            image_width=w,
            image_height=h,
            status=status,
            model_id=self.MODEL_IDENTIFIER,
            mask=cleaned_mask,
            raw_mask=raw_binary,
            foreground_ratio=fg_ratio,
            principal_contour=principal_contour,
            warnings=all_warnings,
            metadata={
                "foreground_threshold": self.foreground_threshold,
                "cleanup_enabled": self.enable_mask_cleanup,
                "min_component_area": self.min_component_area,
                "morph_kernel_size": self.morph_kernel_size,
            },
        )

    def close(self) -> None:
        """Release underlying landmarker graph."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "BodySegmenter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
