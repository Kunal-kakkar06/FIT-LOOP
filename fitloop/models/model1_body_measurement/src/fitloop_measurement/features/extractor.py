"""
features/extractor.py
---------------------
High-level geometric feature extraction engine.

Transforms Stage 3A PoseExtractionResult and Stage 3B SegmentationResult
into ViewGeometricFeatures and aligns front/side captures into PairedGeometricFeatures.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..pose.schemas import PoseExtractionResult
from ..segmentation.schemas import SegmentationResult
from .landmark_geometry import convert_landmarks_to_pixels
from .schemas import (
    FeatureStatus,
    PairedGeometricFeatures,
    ViewGeometricFeatures,
)
from .silhouette_geometry import (
    extract_vertical_width_profile,
    sample_scanline_width,
)

logger = logging.getLogger(__name__)


class GeometricFeatureExtractor:
    """Extracts deterministic geometric features from pose landmarks and silhouette masks."""

    def __init__(
        self,
        min_landmark_visibility: float = 0.50,
        num_profile_samples: int = 50,
    ) -> None:
        self.min_landmark_visibility = min_landmark_visibility
        self.num_profile_samples = num_profile_samples

    def extract_view_features(
        self,
        pose_result: PoseExtractionResult,
        segmentation_result: SegmentationResult,
    ) -> ViewGeometricFeatures:
        """Extract geometric features for a single view (front or side).

        Parameters
        ----------
        pose_result:
            Stage 3A PoseExtractionResult.
        segmentation_result:
            Stage 3B SegmentationResult.

        Returns
        -------
        ViewGeometricFeatures
            Consolidated geometric features combining landmarks and silhouette boundaries.
        """
        warnings: List[str] = []

        # Validate view_type consistency
        if pose_result.view_type != segmentation_result.view_type:
            warnings.append(
                f"Mismatched view_types: pose has '{pose_result.view_type}', segmentation has '{segmentation_result.view_type}'."
            )

        view_type = pose_result.view_type
        w = pose_result.image_width
        h = pose_result.image_height

        # 1. Landmark pixel coordinates
        pixel_landmarks, lm_warnings = convert_landmarks_to_pixels(
            pose_result, min_visibility=self.min_landmark_visibility
        )
        warnings.extend(lm_warnings)

        # 2. Silhouette geometry
        bbox_x, bbox_y, sil_w, sil_h = None, None, None, None
        sil_area = None
        fg_ratio = segmentation_result.foreground_ratio
        width_profile = []
        anatomical_levels = {}

        if (
            segmentation_result.is_successful
            and segmentation_result.mask is not None
            and segmentation_result.principal_contour is not None
        ):
            bbox = segmentation_result.principal_contour.bounding_box
            bbox_x = bbox.x
            bbox_y = bbox.y
            sil_w = bbox.width
            sil_h = bbox.height
            sil_area = segmentation_result.principal_contour.area

            # Extract sampled width profile
            profile, prof_warnings = extract_vertical_width_profile(
                binary_mask=segmentation_result.mask,
                top_y=bbox_y,
                bottom_y=bbox_y + sil_h,
                num_samples=self.num_profile_samples,
            )
            width_profile = profile
            warnings.extend(prof_warnings)

            # Compute body-relative anatomical reference levels
            # convention: relative_y = (pixel_y - bbox_y) / sil_h
            if sil_h > 0:
                for lm in pixel_landmarks:
                    if lm.is_valid and lm.pixel_y is not None:
                        rel_level = (lm.pixel_y - bbox_y) / float(sil_h)
                        anatomical_levels[lm.name] = rel_level
                    else:
                        anatomical_levels[lm.name] = None
        else:
            warnings.append("Silhouette mask or principal contour was unavailable for geometric profiling.")

        status = FeatureStatus.SUCCESS
        if len(warnings) > 0 and (sil_w is None or len(pixel_landmarks) == 0):
            status = FeatureStatus.FAILED
        elif len(warnings) > 0 or any(lm.pixel_x is None for lm in pixel_landmarks):
            status = FeatureStatus.PARTIAL

        return ViewGeometricFeatures(
            view_type=view_type,
            image_width=w,
            image_height=h,
            status=status,
            pixel_landmarks=pixel_landmarks,
            silhouette_bbox_x=bbox_x,
            silhouette_bbox_y=bbox_y,
            silhouette_width=sil_w,
            silhouette_height=sil_h,
            silhouette_area=sil_area,
            foreground_ratio=fg_ratio,
            anatomical_levels=anatomical_levels,
            width_profile=width_profile,
            warnings=warnings,
        )

    def extract_paired_features(
        self,
        front_features: ViewGeometricFeatures,
        side_features: ViewGeometricFeatures,
        expected_height_cm: Optional[float] = None,
    ) -> PairedGeometricFeatures:
        """Combine and cross-evaluate geometric features across front and side perspectives.

        Parameters
        ----------
        front_features:
            Extracted ViewGeometricFeatures for front view.
        side_features:
            Extracted ViewGeometricFeatures for side view.
        expected_height_cm:
            User reference height in cm (if available from Stage 1/2).

        Returns
        -------
        PairedGeometricFeatures
            Joint container with cross-view scale and height coherence checks.
        """
        warnings: List[str] = []
        vert_warning: Optional[str] = None
        height_ratio: Optional[float] = None

        if front_features.view_type != "front":
            warnings.append(f"Expected front view, but received '{front_features.view_type}'.")
        if side_features.view_type != "side":
            warnings.append(f"Expected side view, but received '{side_features.view_type}'.")

        # 1. Compare visible silhouette heights between front and side
        if front_features.silhouette_height and side_features.silhouette_height:
            h_front = float(front_features.silhouette_height)
            h_side = float(side_features.silhouette_height)

            if h_front > 0:
                height_ratio = h_side / h_front
                # If silhouette height varies by > 15%, camera distance or subject posture differed
                delta = abs(h_front - h_side) / h_front
                if delta > 0.15:
                    vert_warning = (
                        f"Visible silhouette height differs by {delta:.1%} between front ({h_front:.0f}px) "
                        f"and side ({h_side:.0f}px). Subject distance from camera likely varied."
                    )
                    warnings.append(vert_warning)

        # 2. Compare anatomical landmark reference levels across views (e.g. hip level)
        # Note: Equal silhouette heights do NOT guarantee identical anatomical alignment
        # due to pelvic tilt, thoracic curvature, head posture, and camera perspective.
        f_hip = front_features.anatomical_levels.get("LEFT_HIP") or front_features.anatomical_levels.get("RIGHT_HIP")
        s_hip = side_features.anatomical_levels.get("LEFT_HIP") or side_features.anatomical_levels.get("RIGHT_HIP")
        if f_hip is not None and s_hip is not None:
            hip_diff = abs(f_hip - s_hip)
            if hip_diff > 0.08:  # > 8% relative vertical discrepancy
                warnings.append(
                    f"Anatomical hip level differs between views by {hip_diff:.1%} of body height. "
                    "Posture variation, pelvic tilt, or camera angle difference detected."
                )

        status = FeatureStatus.SUCCESS
        if front_features.status == FeatureStatus.FAILED or side_features.status == FeatureStatus.FAILED:
            status = FeatureStatus.FAILED
        elif (
            front_features.status == FeatureStatus.PARTIAL
            or side_features.status == FeatureStatus.PARTIAL
            or vert_warning is not None
            or any("Anatomical" in w for w in warnings)
        ):
            status = FeatureStatus.PARTIAL

        return PairedGeometricFeatures(
            status=status,
            front=front_features,
            side=side_features,
            height_ratio_side_to_front=height_ratio,
            vertical_alignment_warning=vert_warning,
            expected_height_cm=expected_height_cm,
            warnings=warnings,
        )
