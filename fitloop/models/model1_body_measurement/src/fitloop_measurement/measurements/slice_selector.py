"""
measurements/slice_selector.py
------------------------------
Anatomical slice selection identifying vertical y-levels on front and side views.

Anatomical Level Definitions:
- CHEST: Inferior to armpits (axillary fold level), located between shoulders and midway to hips (~30% down torso).
- WAIST: Narrowest abdominal indentation; between inferior rib margin and iliac crest (~60% down torso).
- ABDOMEN: Maximum abdominal protrusion level; navel height (~75% down torso).
- HIPS: Level of maximum buttock prominence / greater trochanters (~hip joint level).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..features.schemas import ScanlineSlice, ViewGeometricFeatures
from .schemas import AnatomicalLevel, SliceSelectionResult


def _find_slice_at_y(width_profile: List[ScanlineSlice], target_y: int) -> Optional[ScanlineSlice]:
    """Find the closest valid ScanlineSlice to target_y in width_profile."""
    if not width_profile:
        return None
    # Find slice with minimum distance to target_y
    closest = min(width_profile, key=lambda s: abs(s.image_y - target_y))
    return closest


def select_anatomical_slice(
    level: AnatomicalLevel,
    view_features: ViewGeometricFeatures,
) -> tuple[Optional[ScanlineSlice], SliceSelectionResult]:
    """Identify the corresponding anatomical scanline slice for a given view."""
    warnings: list[str] = []
    view_type = view_features.view_type

    # Landmark references
    l_shoulder = view_features.get_landmark_by_name("LEFT_SHOULDER")
    r_shoulder = view_features.get_landmark_by_name("RIGHT_SHOULDER")
    l_hip = view_features.get_landmark_by_name("LEFT_HIP")
    r_hip = view_features.get_landmark_by_name("RIGHT_HIP")

    # Determine reference shoulder and hip vertical pixel positions
    shoulder_y: Optional[float] = None
    if l_shoulder and l_shoulder.is_valid and r_shoulder and r_shoulder.is_valid:
        shoulder_y = (l_shoulder.pixel_y + r_shoulder.pixel_y) / 2.0
    elif l_shoulder and l_shoulder.is_valid:
        shoulder_y = l_shoulder.pixel_y
    elif r_shoulder and r_shoulder.is_valid:
        shoulder_y = r_shoulder.pixel_y

    hip_y: Optional[float] = None
    if l_hip and l_hip.is_valid and r_hip and r_hip.is_valid:
        hip_y = (l_hip.pixel_y + r_hip.pixel_y) / 2.0
    elif l_hip and l_hip.is_valid:
        hip_y = l_hip.pixel_y
    elif r_hip and r_hip.is_valid:
        hip_y = r_hip.pixel_y

    ref_landmarks: list[str] = []
    target_y: Optional[int] = None
    selection_method = ""

    # Check if torso landmarks exist
    if shoulder_y is not None and hip_y is not None and hip_y > shoulder_y:
        torso_span = hip_y - shoulder_y
        ref_landmarks = ["SHOULDER", "HIP"]

        if level == AnatomicalLevel.CHEST:
            # Chest: ~35% down from shoulder to hip (axillary level)
            target_y = int(round(shoulder_y + 0.35 * torso_span))
            selection_method = "Landmark-anchored (35% torso span below shoulders)"
        elif level == AnatomicalLevel.WAIST:
            # Waist: ~65% down torso span
            target_y = int(round(shoulder_y + 0.65 * torso_span))
            selection_method = "Landmark-anchored (65% torso span below shoulders)"
        elif level == AnatomicalLevel.ABDOMEN:
            # Abdomen: ~80% down torso span (navel / suprailiac level)
            target_y = int(round(shoulder_y + 0.80 * torso_span))
            selection_method = "Landmark-anchored (80% torso span below shoulders)"
        elif level == AnatomicalLevel.HIPS:
            # Hips: ~105% down torso span (greater trochanter level)
            target_y = int(round(shoulder_y + 1.05 * torso_span))
            selection_method = "Landmark-anchored (at/slightly below hip joints)"
        else:
            warnings.append(f"Unsupported level: {level}")
    else:
        # Fallback to body-relative silhouette bounds if landmarks are unavailable
        warnings.append("Shoulder or hip landmarks unavailable; falling back to silhouette height heuristic.")
        top_y = view_features.silhouette_bbox_y
        sil_h = view_features.silhouette_height
        if top_y is not None and sil_h is not None and sil_h > 0:
            if level == AnatomicalLevel.CHEST:
                target_y = int(round(top_y + 0.28 * sil_h))
                selection_method = "Heuristic fallback (28% body height)"
            elif level == AnatomicalLevel.WAIST:
                target_y = int(round(top_y + 0.40 * sil_h))
                selection_method = "Heuristic fallback (40% body height)"
            elif level == AnatomicalLevel.ABDOMEN:
                target_y = int(round(top_y + 0.46 * sil_h))
                selection_method = "Heuristic fallback (46% body height)"
            elif level == AnatomicalLevel.HIPS:
                target_y = int(round(top_y + 0.52 * sil_h))
                selection_method = "Heuristic fallback (52% body height)"

    if target_y is None:
        slice_result = SliceSelectionResult(
            view_type=view_type,
            level=level,
            image_y=None,
            relative_y=None,
            selection_method="None",
            reference_landmarks=[],
            is_valid=False,
            warnings=["Could not determine target scanline y-coordinate."],
        )
        return None, slice_result

    # Match target_y against extracted width profile
    matched_slice = _find_slice_at_y(view_features.width_profile, target_y)

    if matched_slice is None or not matched_slice.is_valid:
        warnings.append(f"No valid scanline found at y={target_y}.")
        slice_result = SliceSelectionResult(
            view_type=view_type,
            level=level,
            image_y=target_y,
            relative_y=None,
            selection_method=selection_method,
            reference_landmarks=ref_landmarks,
            is_valid=False,
            warnings=warnings,
        )
        return None, slice_result

    # Flag if the scanline has disconnected segments (e.g. arm cross-section intersecting torso)
    if matched_slice.segment_count > 1:
        warnings.append(
            f"Scanline at y={matched_slice.image_y} contains {matched_slice.segment_count} disconnected segments (arm or garment separation)."
        )

    slice_result = SliceSelectionResult(
        view_type=view_type,
        level=level,
        image_y=matched_slice.image_y,
        relative_y=matched_slice.relative_y,
        selection_method=selection_method,
        reference_landmarks=ref_landmarks,
        is_valid=True,
        warnings=warnings,
    )

    return matched_slice, slice_result
