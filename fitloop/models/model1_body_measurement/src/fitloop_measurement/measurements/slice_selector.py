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
from ..features.silhouette_geometry import (
    sample_scanline_width,
    torso_aware_scanline_width,
)
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

    # ── Determine scanline width using pose-aware extraction ─────────────────
    # For front view: use torso_aware_scanline_width with shoulder + hip midpoint X
    # guidance so that arm segments can be distinguished from the torso.
    # For side view: the body is presented in profile, arms are typically
    # in front of or behind the torso and do not create lateral arm-torso
    # fusion artefacts — standard sample_scanline_width is appropriate.
    mask = getattr(view_features, '_mask_ref', None)  # not on schema; looked up below

    # Retrieve segmentation mask from width_profile source (closest existing slice)
    matched_slice: ScanlineSlice | None = _find_slice_at_y(view_features.width_profile, target_y)

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

    # ── Apply torso-aware correction for front view with multiple segments ───
    # The width_profile was built with sample_scanline_width (widest-segment fallback).
    # When the front-view scanline has >1 segment AND we have landmark guidance,
    # re-run torso_aware_scanline_width directly on the mask for a better estimate.
    if (
        view_type == "front"
        and matched_slice.segment_count > 1
        and (shoulder_y is not None or hip_y is not None)
    ):
        # We need the actual binary mask. It is stored on the SegmentationResult
        # but is not part of ViewGeometricFeatures schema.  Instead, we reconstruct
        # the scanline from width_profile geometry.  Because the profile was built
        # with sample_scanline_width, we need the raw mask to run torso_aware.
        # ViewGeometricFeatures does NOT carry the mask.  Consequently we pass the
        # width_profile slice through to the estimator unchanged but augment warnings
        # with the selection_reason so diagnostics are informative.
        #
        # NOTE: Full torso_aware integration requires the mask to be carried on
        # ViewGeometricFeatures or passed separately.  This is tracked as a
        # known limitation in the Phase 2 design document; the mask threading is
        # implemented in the new select_anatomical_slice_from_mask() companion below.
        warnings.append(
            f"[{level.value}] Front-view scanline at y={matched_slice.image_y} has "
            f"{matched_slice.segment_count} segments; torso-aware re-evaluation "
            "requires mask reference (see select_anatomical_slice_from_mask)."
        )

    # Flag if the scanline has disconnected segments
    if matched_slice.segment_count > 1:
        warnings.append(
            f"Scanline at y={matched_slice.image_y} contains {matched_slice.segment_count} "
            "disconnected segments (arm or garment separation)."
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


def select_anatomical_slice_from_mask(
    level: AnatomicalLevel,
    view_features: ViewGeometricFeatures,
    binary_mask,
) -> "tuple[ScanlineSlice | None, SliceSelectionResult]":
    """Select anatomical slice using torso_aware_scanline_width on the raw binary mask.

    This is the pose-robust companion to select_anatomical_slice().  It accepts
    the segmentation binary mask directly so that torso_aware_scanline_width() can
    re-evaluate the scanline with full landmark guidance, rather than relying on
    the pre-sampled width_profile.

    Parameters
    ----------
    level         : Target anatomical level.
    view_features : Extracted ViewGeometricFeatures (provides landmarks + bbox).
    binary_mask   : Raw uint8 binary segmentation mask aligned to the image.

    Returns
    -------
    (ScanlineSlice | None, SliceSelectionResult)
        Same structure as select_anatomical_slice() for compatibility.
    """
    import numpy as np
    from ..features.silhouette_geometry import TorsoWidthResult

    warnings_out: list[str] = []
    view_type = view_features.view_type

    # ── Reuse the same y-target logic ────────────────────────────────────────
    l_shoulder = view_features.get_landmark_by_name("LEFT_SHOULDER")
    r_shoulder = view_features.get_landmark_by_name("RIGHT_SHOULDER")
    l_hip      = view_features.get_landmark_by_name("LEFT_HIP")
    r_hip      = view_features.get_landmark_by_name("RIGHT_HIP")
    l_elbow    = view_features.get_landmark_by_name("LEFT_ELBOW")
    r_elbow    = view_features.get_landmark_by_name("RIGHT_ELBOW")
    l_wrist    = view_features.get_landmark_by_name("LEFT_WRIST")
    r_wrist    = view_features.get_landmark_by_name("RIGHT_WRIST")

    def _mid_y(a, b):
        if a and a.is_valid and b and b.is_valid:
            return (a.pixel_y + b.pixel_y) / 2.0
        if a and a.is_valid: return a.pixel_y
        if b and b.is_valid: return b.pixel_y
        return None

    def _mid_x(a, b):
        if a and a.is_valid and b and b.is_valid:
            return (a.pixel_x + b.pixel_x) / 2.0
        if a and a.is_valid: return a.pixel_x
        if b and b.is_valid: return b.pixel_x
        return None

    shoulder_y = _mid_y(l_shoulder, r_shoulder)
    hip_y      = _mid_y(l_hip, r_hip)
    shoulder_mid_x = _mid_x(l_shoulder, r_shoulder)
    hip_mid_x      = _mid_x(l_hip, r_hip)

    elbow_xs = [lm.pixel_x for lm in (l_elbow, r_elbow)
                if lm and lm.is_valid and lm.pixel_x is not None]
    wrist_xs = [lm.pixel_x for lm in (l_wrist, r_wrist)
                if lm and lm.is_valid and lm.pixel_x is not None]

    ref_landmarks: list[str] = []
    target_y = None
    selection_method = ""

    if shoulder_y is not None and hip_y is not None and hip_y > shoulder_y:
        torso_span = hip_y - shoulder_y
        ref_landmarks = ["SHOULDER", "HIP"]
        offsets = {
            AnatomicalLevel.CHEST:   (0.35, "Landmark-anchored (35% torso span)"),
            AnatomicalLevel.WAIST:   (0.65, "Landmark-anchored (65% torso span)"),
            AnatomicalLevel.ABDOMEN: (0.80, "Landmark-anchored (80% torso span)"),
            AnatomicalLevel.HIPS:    (1.05, "Landmark-anchored (at hip joints)"),
        }
        if level in offsets:
            frac, selection_method = offsets[level]
            target_y = int(round(shoulder_y + frac * torso_span))
    else:
        warnings_out.append("Shoulder or hip landmarks unavailable; falling back to silhouette heuristic.")
        top_y = view_features.silhouette_bbox_y
        sil_h = view_features.silhouette_height
        if top_y is not None and sil_h is not None and sil_h > 0:
            fb = {AnatomicalLevel.CHEST: 0.28, AnatomicalLevel.WAIST: 0.40,
                  AnatomicalLevel.ABDOMEN: 0.46, AnatomicalLevel.HIPS: 0.52}
            if level in fb:
                target_y = int(round(top_y + fb[level] * sil_h))
                selection_method = f"Heuristic fallback ({fb[level]:.0%} body height)"

    if target_y is None:
        return None, SliceSelectionResult(
            view_type=view_type, level=level, image_y=None, relative_y=None,
            selection_method="None", reference_landmarks=[],
            is_valid=False, warnings=["Could not determine target scanline y-coordinate."],
        )

    shoulder_width = None
    if l_shoulder and l_shoulder.is_valid and r_shoulder and r_shoulder.is_valid:
        if l_shoulder.pixel_x is not None and r_shoulder.pixel_x is not None:
            shoulder_width = abs(l_shoulder.pixel_x - r_shoulder.pixel_x)

    hip_width = None
    if l_hip and l_hip.is_valid and r_hip and r_hip.is_valid:
        if l_hip.pixel_x is not None and r_hip.pixel_x is not None:
            hip_width = abs(l_hip.pixel_x - r_hip.pixel_x)

    # ── Torso-aware extraction directly from mask ────────────────────────────
    bbox_y = view_features.silhouette_bbox_y
    sil_h  = view_features.silhouette_height

    twr: TorsoWidthResult = torso_aware_scanline_width(
        binary_mask=binary_mask,
        image_y=target_y,
        shoulder_mid_x=shoulder_mid_x if view_type == "front" else None,
        hip_mid_x=hip_mid_x if view_type == "front" else None,
        elbow_xs=elbow_xs if view_type == "front" else None,
        wrist_xs=wrist_xs if view_type == "front" else None,
        silhouette_top_y=bbox_y,
        silhouette_height=sil_h,
        shoulder_width=shoulder_width if view_type == "front" else None,
        hip_width=hip_width if view_type == "front" else None,
    )

    all_warnings = warnings_out + twr.warnings

    if not twr.is_valid:
        return None, SliceSelectionResult(
            view_type=view_type, level=level, image_y=target_y, relative_y=None,
            selection_method=selection_method, reference_landmarks=ref_landmarks,
            is_valid=False, warnings=all_warnings,
        )

    scanline_slice = twr.to_scanline_slice()

    return scanline_slice, SliceSelectionResult(
        view_type=view_type, level=level,
        image_y=scanline_slice.image_y, relative_y=scanline_slice.relative_y,
        selection_method=f"{selection_method} [torso-aware: {twr.selection_reason}]",
        reference_landmarks=ref_landmarks,
        is_valid=True, warnings=all_warnings,
    )
