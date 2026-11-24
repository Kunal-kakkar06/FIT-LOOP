"""
measurements/slice_strategy.py
------------------------------
Strategy abstraction, typed contracts, and baseline implementation for
anatomical slice selection and cross-view coordinate reconciliation.

Coordinate Contract & Assumptions:
----------------------------------
1. Front and side views are captured from distinct camera perspectives.
2. In the baseline implementation, target y-scanlines are selected independently
   within each view using landmark-anchored torso fractions:
     - CHEST:   0.35 * torso_span below shoulders
     - WAIST:   0.65 * torso_span below shoulders
     - ABDOMEN: 0.80 * torso_span below shoulders
     - HIPS:    1.05 * torso_span below shoulders
3. Equal normalized torso fractions do NOT imply equal anatomical heights.
4. Independent height-based calibration provides a scale (cm/pixel) relating
   pixels in that specific view to real-world centimetres, referenced from the
   silhouette bounding box top (y_top) to silhouette bounding box bottom.
5. Calibrated physical height from silhouette vertex:
     y_phys_cm = (image_y - silhouette_bbox_y) * scale_cm_per_pixel
   A physical height difference:
     height_delta_cm = abs(front_phys_cm - side_phys_cm)
   can ONLY be calculated when BOTH views possess:
     a) A valid silhouette bounding box top (bbox_y)
     b) A valid, positive scale (scale_cm_per_pixel)
     c) An unclipped head apex and standing upright posture.
   Even when both views satisfy these criteria, a height delta is strictly
   observational and does not by itself prove anatomical correspondence
   (due to pitch/tilt, forward lean, pelvic posture, or spinal curvature).
"""

from __future__ import annotations

import abc
import dataclasses
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from ..features.schemas import ScanlineSlice, ViewGeometricFeatures
from ..features.silhouette_geometry import (
    sample_scanline_width,
    torso_aware_scanline_width,
)
from .schemas import (
    AnatomicalLevel,
    DualViewCalibration,
    SliceSelectionResult,
)
from .slice_selector import (
    select_anatomical_slice,
    select_anatomical_slice_from_mask,
)


class SliceSelectionStrategyType(str, Enum):
    """Supported slice selection strategy identifiers."""

    BASELINE_FIXED_FRACTION = "baseline_fixed_fraction"
    ANATOMICAL_CROSS_VIEW = "anatomical_cross_view"


class AlignmentConfidence(str, Enum):
    """Confidence level in cross-view anatomical slice correspondence."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNVERIFIED = "UNVERIFIED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclasses.dataclass(frozen=True)
class AlignedSlicePair:
    """Typed container representing coordinated front and side anatomical slices.

    Captures the selected scanlines, slice selection diagnostics, calibrated
    physical coordinates, and cross-view alignment metadata.
    """

    level: AnatomicalLevel
    front_slice: Optional[ScanlineSlice]
    side_slice: Optional[ScanlineSlice]
    front_result: SliceSelectionResult
    side_result: SliceSelectionResult
    # Observational physical coordinates (cm from silhouette top)
    front_phys_height_cm: Optional[float]
    side_phys_height_cm: Optional[float]
    height_delta_cm: Optional[float]
    # Alignment diagnostics
    is_aligned: bool
    confidence: AlignmentConfidence
    strategy_used: SliceSelectionStrategyType
    warnings: List[str] = dataclasses.field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if both front and side slices are valid and present."""
        return (
            self.front_slice is not None
            and self.side_slice is not None
            and self.front_result.is_valid
            and self.side_result.is_valid
            and self.front_slice.is_valid
            and self.side_slice.is_valid
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation of the aligned slice pair."""
        return {
            "level": self.level.value,
            "front_slice": self.front_slice.to_dict() if self.front_slice else None,
            "side_slice": self.side_slice.to_dict() if self.side_slice else None,
            "front_result": self.front_result.to_dict(),
            "side_result": self.side_result.to_dict(),
            "front_phys_height_cm": (
                round(self.front_phys_height_cm, 2)
                if self.front_phys_height_cm is not None
                else None
            ),
            "side_phys_height_cm": (
                round(self.side_phys_height_cm, 2)
                if self.side_phys_height_cm is not None
                else None
            ),
            "height_delta_cm": (
                round(self.height_delta_cm, 2)
                if self.height_delta_cm is not None
                else None
            ),
            "is_aligned": self.is_aligned,
            "confidence": self.confidence.value,
            "strategy_used": self.strategy_used.value,
            "warnings": list(self.warnings),
        }


class ISliceSelectionStrategy(Protocol):
    """Protocol for slice selection strategies."""

    @property
    def strategy_type(self) -> SliceSelectionStrategyType:
        """Return the unique identifier for this strategy."""
        ...

    def select_slices(
        self,
        level: AnatomicalLevel,
        front_features: ViewGeometricFeatures,
        side_features: ViewGeometricFeatures,
        calibration: DualViewCalibration,
        front_mask: Optional[np.ndarray] = None,
        side_mask: Optional[np.ndarray] = None,
    ) -> AlignedSlicePair:
        """Select coordinated front and side slices for the given anatomical level."""
        ...


def compute_observational_physical_height(
    image_y: Optional[int],
    bbox_y: Optional[int],
    scale_cm_per_pixel: Optional[float],
) -> Optional[float]:
    """Calculate physical distance in cm from silhouette top.

    Returns None if any required coordinate or scale is missing, non-finite,
    or non-positive.
    """
    if (
        image_y is None
        or bbox_y is None
        or scale_cm_per_pixel is None
        or not math.isfinite(scale_cm_per_pixel)
        or scale_cm_per_pixel <= 0.0
    ):
        return None
    return float(image_y - bbox_y) * float(scale_cm_per_pixel)


class BaselineFixedFractionStrategy:
    """Preserves current production behavior.

    Selects scanline rows independently in front and side views using existing
    torso-span multipliers (or silhouette fallbacks). Observational physical
    height delta is computed for diagnostic tracking without shifting scanlines.
    Returns UNVERIFIED confidence because independent fixed fractions do not
    verify anatomical cross-view correspondence.
    """

    @property
    def strategy_type(self) -> SliceSelectionStrategyType:
        return SliceSelectionStrategyType.BASELINE_FIXED_FRACTION

    def select_slices(
        self,
        level: AnatomicalLevel,
        front_features: ViewGeometricFeatures,
        side_features: ViewGeometricFeatures,
        calibration: DualViewCalibration,
        front_mask: Optional[np.ndarray] = None,
        side_mask: Optional[np.ndarray] = None,
    ) -> AlignedSlicePair:
        """Select front and side slices via production slice selector functions."""
        warnings: List[str] = []

        # 1. Front view selection (preserves mask-aware path)
        if front_mask is not None:
            f_slice, f_result = select_anatomical_slice_from_mask(
                level, front_features, front_mask
            )
        else:
            f_slice, f_result = select_anatomical_slice(level, front_features)

        # 2. Side view selection
        if side_mask is not None:
            s_slice, s_result = select_anatomical_slice_from_mask(
                level, side_features, side_mask
            )
        else:
            s_slice, s_result = select_anatomical_slice(level, side_features)

        warnings.extend([f"[Front] {w}" for w in f_result.warnings])
        warnings.extend([f"[Side] {w}" for w in s_result.warnings])

        # 3. Observational physical height calculation
        # Only compute if calibration is valid and silhouette bbox top is available
        front_scale = (
            calibration.front.scale_cm_per_pixel
            if calibration.front.is_valid
            else None
        )
        side_scale = (
            calibration.side.scale_cm_per_pixel
            if calibration.side.is_valid
            else None
        )

        f_phys: Optional[float] = None
        s_phys: Optional[float] = None
        height_delta: Optional[float] = None

        if f_slice is not None and f_slice.is_valid:
            f_phys = compute_observational_physical_height(
                f_slice.image_y,
                front_features.silhouette_bbox_y,
                front_scale,
            )

        if s_slice is not None and s_slice.is_valid:
            s_phys = compute_observational_physical_height(
                s_slice.image_y,
                side_features.silhouette_bbox_y,
                side_scale,
            )

        if f_phys is not None and s_phys is not None:
            height_delta = abs(f_phys - s_phys)
            # Add informational notice if discrepancy is noticeable (> 2.5 cm)
            if height_delta > 2.5:
                warnings.append(
                    f"Observational front/side height discrepancy of {height_delta:.1f} cm detected "
                    f"(Front: {f_phys:.1f} cm, Side: {s_phys:.1f} cm from silhouette apex). "
                    "Baseline fixed-fraction does not reconcile cross-view scanline heights."
                )
        else:
            if not calibration.is_valid:
                warnings.append(
                    "Cross-view physical height comparison unavailable: metric calibration is invalid."
                )
            elif front_features.silhouette_bbox_y is None or side_features.silhouette_bbox_y is None:
                warnings.append(
                    "Cross-view physical height comparison unavailable: silhouette bbox top is missing."
                )

        return AlignedSlicePair(
            level=level,
            front_slice=f_slice,
            side_slice=s_slice,
            front_result=f_result,
            side_result=s_result,
            front_phys_height_cm=f_phys,
            side_phys_height_cm=s_phys,
            height_delta_cm=height_delta,
            is_aligned=False,  # Baseline fixed fractions do not establish validated alignment
            confidence=AlignmentConfidence.UNVERIFIED,
            strategy_used=self.strategy_type,
            warnings=warnings,
        )


class AnatomicalCrossViewStrategy:
    """Experimental cross-view anatomical alignment strategy (research only).

    Candidate Anchoring Logic & Coordinate Contract:
    -----------------------------------------------
    1. Candidate Rows:
       - CHEST:
         Front: Axillary fold / inferior border of pectoralis/axilla region (candidate search
         in 25%-40% torso span).
         Side: Axillary line level (25%-40% side torso span).
       - WAIST:
         Front: Searches for lateral contour narrowing (minimum width) within lumbar band
         (45%-75% torso span). If contour is flat, ambiguous, or monotonic, marks candidate ambiguous.
         Side: Searches for lordotic/depth inflection or projects calibrated physical height.
       - ABDOMEN:
         Front: Lower abdominal expansion level (75%-88% torso span).
         Side: Anterior abdominal protrusion peak (maximum anterior extent / depth peak).
       - HIPS:
         Front: Maximum pelvic/trochanter width near hip joint level (95%-115% torso span).
         Side: Maximum buttock/gluteal prominence (95%-115% torso span).
    2. Coordinate Gating & Rejection:
       - Cross-view correspondence is verified using observational physical height from
         the silhouette apex (calibrated cm).
       - If physical height delta exceeds tolerance (MAX_HEIGHT_TOLERANCE_CM = 2.5 cm),
         the candidate pair is rejected or marked LOW confidence.
       - NEVER claims HIGH confidence without independent anatomical reference annotations.
       - If either view lacks unambiguous contour features or valid calibration, marks the
         level UNAVAILABLE or LOW confidence with explicit reasons. Never silently falls back
         to baseline while claiming alignment succeeded.
    """

    MAX_HEIGHT_TOLERANCE_CM: float = 2.5

    @property
    def strategy_type(self) -> SliceSelectionStrategyType:
        return SliceSelectionStrategyType.ANATOMICAL_CROSS_VIEW

    def _get_torso_landmarks(self, view_features: ViewGeometricFeatures):
        l_sh = view_features.get_landmark_by_name("LEFT_SHOULDER")
        r_sh = view_features.get_landmark_by_name("RIGHT_SHOULDER")
        l_hp = view_features.get_landmark_by_name("LEFT_HIP")
        r_hp = view_features.get_landmark_by_name("RIGHT_HIP")
        l_el = view_features.get_landmark_by_name("LEFT_ELBOW")
        r_el = view_features.get_landmark_by_name("RIGHT_ELBOW")
        l_wr = view_features.get_landmark_by_name("LEFT_WRIST")
        r_wr = view_features.get_landmark_by_name("RIGHT_WRIST")

        sh_y = None
        if l_sh and l_sh.is_valid and r_sh and r_sh.is_valid:
            sh_y = (l_sh.pixel_y + r_sh.pixel_y) / 2.0
        elif l_sh and l_sh.is_valid:
            sh_y = l_sh.pixel_y
        elif r_sh and r_sh.is_valid:
            sh_y = r_sh.pixel_y

        hp_y = None
        if l_hp and l_hp.is_valid and r_hp and r_hp.is_valid:
            hp_y = (l_hp.pixel_y + r_hp.pixel_y) / 2.0
        elif l_hp and l_hp.is_valid:
            hp_y = l_hp.pixel_y
        elif r_hp and r_hp.is_valid:
            hp_y = r_hp.pixel_y

        sh_mid_x = None
        if l_sh and l_sh.is_valid and r_sh and r_sh.is_valid:
            sh_mid_x = (l_sh.pixel_x + r_sh.pixel_x) / 2.0
        elif l_sh and l_sh.is_valid:
            sh_mid_x = l_sh.pixel_x
        elif r_sh and r_sh.is_valid:
            sh_mid_x = r_sh.pixel_x

        hp_mid_x = None
        if l_hp and l_hp.is_valid and r_hp and r_hp.is_valid:
            hp_mid_x = (l_hp.pixel_x + r_hp.pixel_x) / 2.0
        elif l_hp and l_hp.is_valid:
            hp_mid_x = l_hp.pixel_x
        elif r_hp and r_hp.is_valid:
            hp_mid_x = r_hp.pixel_x

        sh_w = None
        if l_sh and l_sh.is_valid and r_sh and r_sh.is_valid:
            if l_sh.pixel_x is not None and r_sh.pixel_x is not None:
                sh_w = abs(l_sh.pixel_x - r_sh.pixel_x)

        hp_w = None
        if l_hp and l_hp.is_valid and r_hp and r_hp.is_valid:
            if l_hp.pixel_x is not None and r_hp.pixel_x is not None:
                hp_w = abs(l_hp.pixel_x - r_hp.pixel_x)

        el_xs = [lm.pixel_x for lm in (l_el, r_el) if lm and lm.is_valid and lm.pixel_x is not None]
        wr_xs = [lm.pixel_x for lm in (l_wr, r_wr) if lm and lm.is_valid and lm.pixel_x is not None]

        return {
            "sh_y": sh_y,
            "hp_y": hp_y,
            "sh_mid_x": sh_mid_x,
            "hp_mid_x": hp_mid_x,
            "sh_w": sh_w,
            "hp_w": hp_w,
            "el_xs": el_xs,
            "wr_xs": wr_xs,
        }

    def _find_front_candidate(
        self,
        level: AnatomicalLevel,
        features: ViewGeometricFeatures,
        mask: Optional[np.ndarray],
        lm_info: dict,
    ) -> Tuple[Optional[int], str, List[str]]:
        """Identify candidate image row on front view using contour/geometric evidence."""
        warnings: List[str] = []
        sh_y = lm_info["sh_y"]
        hp_y = lm_info["hp_y"]

        if sh_y is None or hp_y is None or hp_y <= sh_y:
            warnings.append("[Front] Landmark torso span unavailable; cannot establish candidate search window.")
            return None, "None", warnings

        span = hp_y - sh_y

        if mask is None:
            warnings.append("[Front] Segmentation mask unavailable for candidate contour analysis.")
            return None, "None", warnings

        # 1. CHEST
        if level == AnatomicalLevel.CHEST:
            # Axillary region: 30% to 38% torso span
            target_y = int(round(sh_y + 0.35 * span))
            return target_y, "Front axillary landmark level (~35% torso span)", warnings

        # 2. WAIST: Search for lateral narrowing (minimum width) in lumbar range (45% - 75%)
        elif level == AnatomicalLevel.WAIST:
            min_r = int(round(sh_y + 0.45 * span))
            max_r = int(round(sh_y + 0.75 * span))
            rows = range(min_r, max_r, max(1, int(span * 0.01)))
            widths = []
            valid_rows = []

            for r in rows:
                twr = torso_aware_scanline_width(
                    binary_mask=mask,
                    image_y=r,
                    shoulder_mid_x=lm_info["sh_mid_x"],
                    hip_mid_x=lm_info["hp_mid_x"],
                    elbow_xs=lm_info["el_xs"],
                    wrist_xs=lm_info["wr_xs"],
                    silhouette_top_y=features.silhouette_bbox_y,
                    silhouette_height=features.silhouette_height,
                    shoulder_width=lm_info["sh_w"],
                    hip_width=lm_info["hp_w"],
                )
                if twr.is_valid and twr.width_pixels is not None:
                    widths.append(twr.width_pixels)
                    valid_rows.append(r)

            if not widths:
                warnings.append("[Front] No valid torso scanlines in waist search band.")
                return None, "None", warnings

            min_w = min(widths)
            max_w = max(widths)
            # Check contour curvature: if width variation across lumbar band is < 2.5%,
            # contour is ambiguous / cylindrical (e.g. loose clothing or tubular torso)
            if max_w > 0 and (max_w - min_w) / float(max_w) < 0.025:
                warnings.append(
                    f"[Front] Lateral waist indentation is ambiguous (variation {((max_w - min_w) / max_w):.1%} < 2.5%). "
                    "Contour lacks distinct narrowing."
                )

            best_idx = int(np.argmin(widths))
            best_y = valid_rows[best_idx]
            frac = (best_y - sh_y) / span
            return best_y, f"Front lateral contour minimum (narrowest waist at {frac:.1%} torso span)", warnings

        # 3. ABDOMEN: Lower abdominal expansion level (75% - 85%)
        elif level == AnatomicalLevel.ABDOMEN:
            target_y = int(round(sh_y + 0.80 * span))
            return target_y, "Front infra-umbilical landmark level (~80% torso span)", warnings

        # 4. HIPS: Maximum pelvic prominence near hip joints (95% - 110%)
        elif level == AnatomicalLevel.HIPS:
            target_y = int(round(sh_y + 1.05 * span))
            return target_y, "Front trochanteric / hip joint level (~105% torso span)", warnings

        return None, "None", [f"Unsupported level: {level}"]

    def _find_side_candidate(
        self,
        level: AnatomicalLevel,
        features: ViewGeometricFeatures,
        mask: Optional[np.ndarray],
        lm_info: dict,
    ) -> Tuple[Optional[int], str, List[str]]:
        """Identify candidate image row on side view using profile contour evidence."""
        warnings: List[str] = []
        sh_y = lm_info["sh_y"]
        hp_y = lm_info["hp_y"]

        if sh_y is None or hp_y is None or hp_y <= sh_y:
            warnings.append("[Side] Landmark torso span unavailable; cannot establish candidate search window.")
            return None, "None", warnings

        span = hp_y - sh_y

        if mask is None:
            warnings.append("[Side] Segmentation mask unavailable for candidate contour analysis.")
            return None, "None", warnings

        # 1. CHEST
        if level == AnatomicalLevel.CHEST:
            target_y = int(round(sh_y + 0.35 * span))
            return target_y, "Side axillary line level (~35% torso span)", warnings

        # 2. WAIST: Search for minimum sagittal depth or lordotic curve (45% - 75%)
        elif level == AnatomicalLevel.WAIST:
            min_r = int(round(sh_y + 0.45 * span))
            max_r = int(round(sh_y + 0.75 * span))
            rows = range(min_r, max_r, max(1, int(span * 0.01)))
            depths = []
            valid_rows = []

            for r in rows:
                sl = sample_scanline_width(
                    binary_mask=mask,
                    image_y=r,
                    silhouette_top_y=features.silhouette_bbox_y,
                    silhouette_height=features.silhouette_height,
                )
                if sl.is_valid and sl.width_pixels is not None:
                    depths.append(sl.width_pixels)
                    valid_rows.append(r)

            if not depths:
                warnings.append("[Side] No valid profile scanlines in waist search band.")
                return None, "None", warnings

            min_d = min(depths)
            max_d = max(depths)
            if max_d > 0 and (max_d - min_d) / float(max_d) < 0.025:
                warnings.append(
                    f"[Side] Sagittal waist inflection is ambiguous (variation {((max_d - min_d) / max_d):.1%} < 2.5%)."
                )

            best_idx = int(np.argmin(depths))
            best_y = valid_rows[best_idx]
            frac = (best_y - sh_y) / span
            return best_y, f"Side sagittal depth minimum ({frac:.1%} torso span)", warnings

        # 3. ABDOMEN: Anterior abdominal protrusion peak (65% - 85%)
        elif level == AnatomicalLevel.ABDOMEN:
            min_r = int(round(sh_y + 0.65 * span))
            max_r = int(round(sh_y + 0.88 * span))
            rows = range(min_r, max_r, max(1, int(span * 0.01)))
            depths = []
            valid_rows = []

            for r in rows:
                sl = sample_scanline_width(
                    binary_mask=mask,
                    image_y=r,
                    silhouette_top_y=features.silhouette_bbox_y,
                    silhouette_height=features.silhouette_height,
                )
                if sl.is_valid and sl.width_pixels is not None:
                    depths.append(sl.width_pixels)
                    valid_rows.append(r)

            if not depths:
                warnings.append("[Side] No valid profile scanlines in abdomen search band.")
                return None, "None", warnings

            best_idx = int(np.argmax(depths))
            best_y = valid_rows[best_idx]
            frac = (best_y - sh_y) / span
            return best_y, f"Side anterior protrusion / max depth peak ({frac:.1%} torso span)", warnings

        # 4. HIPS: Maximum gluteal prominence (95% - 110%)
        elif level == AnatomicalLevel.HIPS:
            target_y = int(round(sh_y + 1.05 * span))
            return target_y, "Side gluteal / trochanteric prominence (~105% torso span)", warnings

        return None, "None", [f"Unsupported level: {level}"]

    def select_slices(
        self,
        level: AnatomicalLevel,
        front_features: ViewGeometricFeatures,
        side_features: ViewGeometricFeatures,
        calibration: DualViewCalibration,
        front_mask: Optional[np.ndarray] = None,
        side_mask: Optional[np.ndarray] = None,
    ) -> AlignedSlicePair:
        """Select cross-view coordinated anatomical slices with conservative uncertainty gating."""
        warnings: List[str] = []

        f_lm = self._get_torso_landmarks(front_features)
        s_lm = self._get_torso_landmarks(side_features)

        # 1. Independent candidate row detection
        f_cand_y, f_method, f_cand_warn = self._find_front_candidate(level, front_features, front_mask, f_lm)
        s_cand_y, s_method, s_cand_warn = self._find_side_candidate(level, side_features, side_mask, s_lm)

        warnings.extend(f_cand_warn)
        warnings.extend(s_cand_warn)

        # If candidates could not be identified, return explicit UNAVAILABLE result
        if f_cand_y is None or s_cand_y is None:
            unavail_f = SliceSelectionResult(
                view_type="front",
                level=level,
                image_y=f_cand_y,
                relative_y=None,
                selection_method=f_method,
                reference_landmarks=[],
                is_valid=False,
                warnings=f_cand_warn or ["Front candidate row could not be identified."],
            )
            unavail_s = SliceSelectionResult(
                view_type="side",
                level=level,
                image_y=s_cand_y,
                relative_y=None,
                selection_method=s_method,
                reference_landmarks=[],
                is_valid=False,
                warnings=s_cand_warn or ["Side candidate row could not be identified."],
            )
            return AlignedSlicePair(
                level=level,
                front_slice=None,
                side_slice=None,
                front_result=unavail_f,
                side_result=unavail_s,
                front_phys_height_cm=None,
                side_phys_height_cm=None,
                height_delta_cm=None,
                is_aligned=False,
                confidence=AlignmentConfidence.UNAVAILABLE,
                strategy_used=self.strategy_type,
                warnings=warnings or ["Cross-view candidate extraction unavailable."],
            )

        # 2. Extract scanline slices at candidate rows using valid mask-aware extraction
        if front_mask is not None:
            twr = torso_aware_scanline_width(
                binary_mask=front_mask,
                image_y=f_cand_y,
                shoulder_mid_x=f_lm["sh_mid_x"],
                hip_mid_x=f_lm["hp_mid_x"],
                elbow_xs=f_lm["el_xs"],
                wrist_xs=f_lm["wr_xs"],
                silhouette_top_y=front_features.silhouette_bbox_y,
                silhouette_height=front_features.silhouette_height,
                shoulder_width=f_lm["sh_w"],
                hip_width=f_lm["hp_w"],
            )
            f_slice = twr.to_scanline_slice() if twr.is_valid else None
            warnings.extend([f"[Front] {w}" for w in twr.warnings])
        else:
            f_slice = sample_scanline_width(
                binary_mask=front_mask,
                image_y=f_cand_y,
                silhouette_top_y=front_features.silhouette_bbox_y,
                silhouette_height=front_features.silhouette_height,
            )

        if side_mask is not None:
            s_slice = sample_scanline_width(
                binary_mask=side_mask,
                image_y=s_cand_y,
                silhouette_top_y=side_features.silhouette_bbox_y,
                silhouette_height=side_features.silhouette_height,
            )
        else:
            s_slice = None

        f_result = SliceSelectionResult(
            view_type="front",
            level=level,
            image_y=f_slice.image_y if f_slice else f_cand_y,
            relative_y=f_slice.relative_y if f_slice else None,
            selection_method=f_method,
            reference_landmarks=["SHOULDER", "HIP"],
            is_valid=(f_slice is not None and f_slice.is_valid),
            warnings=f_cand_warn,
        )
        s_result = SliceSelectionResult(
            view_type="side",
            level=level,
            image_y=s_slice.image_y if s_slice else s_cand_y,
            relative_y=s_slice.relative_y if s_slice else None,
            selection_method=s_method,
            reference_landmarks=["SHOULDER", "HIP"],
            is_valid=(s_slice is not None and s_slice.is_valid),
            warnings=s_cand_warn,
        )

        # 3. Coordinate reconciliation via calibrated physical height from silhouette apex
        f_scale = calibration.front.scale_cm_per_pixel if calibration.front.is_valid else None
        s_scale = calibration.side.scale_cm_per_pixel if calibration.side.is_valid else None

        f_phys = compute_observational_physical_height(f_cand_y, front_features.silhouette_bbox_y, f_scale)
        s_phys = compute_observational_physical_height(s_cand_y, side_features.silhouette_bbox_y, s_scale)

        height_delta = None
        if f_phys is not None and s_phys is not None:
            height_delta = abs(f_phys - s_phys)

        # 4. Uncertainty & Alignment Gating
        is_aligned = False
        confidence = AlignmentConfidence.LOW

        if not calibration.is_valid or height_delta is None:
            confidence = AlignmentConfidence.LOW
            warnings.append("Calibration invalid or unverified; cross-view physical height comparison unavailable.")
        elif height_delta > self.MAX_HEIGHT_TOLERANCE_CM:
            # Physical heights diverge beyond tolerance: reject alignment claim
            is_aligned = False
            confidence = AlignmentConfidence.LOW
            warnings.append(
                f"Candidate cross-view height discrepancy ({height_delta:.1f} cm) exceeds tolerance "
                f"({self.MAX_HEIGHT_TOLERANCE_CM} cm). Candidate correspondence rejected."
            )
        else:
            # Physical heights agree within tolerance
            is_aligned = True
            # Never claim HIGH confidence without independent anatomical reference annotations
            confidence = AlignmentConfidence.MEDIUM
            warnings.append(
                f"Candidate cross-view physical heights agree within {height_delta:.1f} cm (<= {self.MAX_HEIGHT_TOLERANCE_CM} cm). "
                "Experimental alignment provisional; unverified against manual surface landmarks."
            )

        return AlignedSlicePair(
            level=level,
            front_slice=f_slice,
            side_slice=s_slice,
            front_result=f_result,
            side_result=s_result,
            front_phys_height_cm=f_phys,
            side_phys_height_cm=s_phys,
            height_delta_cm=height_delta,
            is_aligned=is_aligned,
            confidence=confidence,
            strategy_used=self.strategy_type,
            warnings=warnings,
        )

