"""
features/silhouette_geometry.py
-------------------------------
Silhouette geometry utilities: horizontal scanline widths, width profiles,
boundary detection, and body-relative vertical normalization.

Pose-robust architecture (Phase 2)
-----------------------------------
The module now exposes two levels of scanline width extraction:

  sample_scanline_width()
      Original baseline: segments the scanline and selects the widest
      contiguous foreground run.  Used for the width profile and as the
      fallback path when landmark guidance is unavailable.

  torso_aware_scanline_width()
      Pose-robust path: accepts anatomical landmark X-coordinates for the
      scanline level (shoulder midpoint, hip midpoint, elbow, wrist).  It
      computes a TorsoXWindow that brackets the expected torso extent and
      scores each foreground interval by its overlap with that window.
      The interval with the highest torso-overlap is selected; intervals
      outside the window are rejected and recorded for diagnostics.

      If no landmark guidance can be established, falls back to widest-
      segment selection identically to sample_scanline_width().

      The function never silently substitutes an arm width or outermost
      span.  If no interval overlaps the torso window at all, the result
      is marked unavailable with an explicit reason string.

Selection confidence categories (not probabilities)
-----------------------------------------------------
  LANDMARK_GUIDED   — torso window derived from valid landmarks; an
                      interval overlapping the window was found.
  WIDEST_FALLBACK   — no landmark guidance; widest segment chosen.
  WINDOW_NO_OVERLAP — landmark window established but no foreground
                      interval overlaps it; result is unavailable.
  SINGLE_SEGMENT    — exactly one foreground interval on the scanline;
                      selected without ambiguity regardless of landmark
                      guidance.
  OUT_OF_BOUNDS     — scanline y is outside image bounds.
  EMPTY_SCANLINE    — no foreground pixels at this y level.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..segmentation.schemas import SegmentationResult
from .schemas import ScanlineSlice


def _extract_contiguous_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """Return a list of (start_x, end_x) inclusive intervals for each contiguous
    foreground run in *row* (a 1-D binary mask slice where foreground > 0).

    The list is sorted by run width descending so callers can take the widest
    run first.
    """
    fg = (row > 0).astype(np.int8)
    diffs = np.diff(fg, prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1  # inclusive end
    runs = [(int(s), int(e)) for s, e in zip(starts, ends)]
    # Sort widest-first so the caller can pick runs[0] as the torso candidate
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs


def sample_scanline_width(
    binary_mask: np.ndarray,
    image_y: int,
    silhouette_top_y: Optional[int] = None,
    silhouette_height: Optional[int] = None,
) -> ScanlineSlice:
    """Measure the horizontal silhouette width and boundary coordinates at a single scanline.

    Parameters
    ----------
    binary_mask:
        2D binary mask array (uint8) where foreground pixels are > 0.
    image_y:
        Integer vertical pixel row index.
    silhouette_top_y:
        Top pixel row of the overall bounding box (for relative_y calculation).
    silhouette_height:
        Height of the overall bounding box (for relative_y calculation).

    Returns
    -------
    ScanlineSlice
        Geometric boundary description for row image_y.
    """
    h, w = binary_mask.shape[:2]

    # Calculate relative_y if bounding bounds are provided
    rel_y: Optional[float] = None
    if silhouette_top_y is not None and silhouette_height is not None and silhouette_height > 0:
        rel_y = (image_y - silhouette_top_y) / float(silhouette_height)

    # In-bounds check
    if not (0 <= image_y < h):
        return ScanlineSlice(
            image_y=image_y,
            relative_y=rel_y,
            left_x=None,
            right_x=None,
            width_pixels=None,
            is_valid=False,
            segment_count=0,
        )

    row = binary_mask[image_y, :]
    fg_indices = np.where(row > 0)[0]

    if len(fg_indices) == 0:
        return ScanlineSlice(
            image_y=image_y,
            relative_y=rel_y,
            left_x=None,
            right_x=None,
            width_pixels=0,
            is_valid=False,
            segment_count=0,
        )

    # Enumerate all contiguous foreground runs
    runs = _extract_contiguous_runs(row)
    segment_count = len(runs)

    if segment_count == 1:
        # Single contiguous region — use it directly
        left_x, right_x = runs[0]
    else:
        # Multiple disconnected segments (e.g. torso + arms).
        # Selecting the widest run is the most conservative estimate: the torso
        # is almost always the largest contiguous foreground region on the
        # scanline.  Using the absolute outermost endpoints (fg_indices[0] to
        # fg_indices[-1]) would span background gaps and arm regions, yielding
        # severely inflated widths.  The caller already receives segment_count
        # and will emit a PARTIAL warning via slice_selector.
        left_x, right_x = runs[0]  # widest run (sorted descending by width)

    width_pixels = right_x - left_x + 1

    return ScanlineSlice(
        image_y=image_y,
        relative_y=rel_y,
        left_x=left_x,
        right_x=right_x,
        width_pixels=width_pixels,
        is_valid=True,
        segment_count=segment_count,
    )


def extract_vertical_width_profile(
    binary_mask: np.ndarray,
    top_y: int,
    bottom_y: int,
    num_samples: int = 50,
) -> tuple[List[ScanlineSlice], list[str]]:
    """Sample the silhouette width across vertical intervals between top_y and bottom_y.

    Parameters
    ----------
    binary_mask:
        Binary mask array.
    top_y:
        Top pixel boundary of the profile sampling region.
    bottom_y:
        Bottom pixel boundary of the profile sampling region.
    num_samples:
        Number of horizontal slices to evaluate.

    Returns
    -------
    tuple[List[ScanlineSlice], list[str]]
        List of ScanlineSlice samples and diagnostic warnings.
    """
    warnings: list[str] = []
    h = binary_mask.shape[0]

    if bottom_y <= top_y or num_samples <= 0:
        return [], ["Invalid vertical range or num_samples for width profile extraction."]

    total_span = bottom_y - top_y
    y_positions = np.linspace(top_y, bottom_y - 1, num=num_samples, dtype=int)

    slices: List[ScanlineSlice] = []
    discontinuity_count = 0

    for y in y_positions:
        s = sample_scanline_width(
            binary_mask=binary_mask,
            image_y=int(y),
            silhouette_top_y=top_y,
            silhouette_height=total_span,
        )
        slices.append(s)
        if s.segment_count > 1:
            discontinuity_count += 1

    if discontinuity_count > num_samples * 0.3:
        warnings.append(
            f"High number of disconnected scanlines ({discontinuity_count}/{num_samples}). "
            "Limb separation or occlusion detected."
        )

    return slices, warnings


# ── Phase 2: Pose-robust torso boundary extraction ────────────────────────────


@dataclasses.dataclass(frozen=True)
class TorsoXWindow:
    """Horizontal pixel range expected to contain the torso at a given scanline y.

    Derived from anatomical landmarks; not a statistical confidence interval.

    Attributes
    ----------
    x_min, x_max : inclusive pixel columns of the expected torso extent.
    margin_px    : padding added to each side of the raw landmark span.
    source       : human-readable description of which landmarks defined the window.
    is_valid     : False when insufficient landmarks were available.
    """

    x_min: int
    x_max: int
    margin_px: int
    source: str
    is_valid: bool

    def width(self) -> int:
        return max(0, self.x_max - self.x_min + 1)

    def overlap_pixels(self, run_start: int, run_end: int) -> int:
        """Return the number of pixels that [run_start, run_end] overlaps with this window."""
        lo = max(self.x_min, run_start)
        hi = min(self.x_max, run_end)
        return max(0, hi - lo + 1)

    def overlap_fraction(self, run_start: int, run_end: int) -> float:
        """Overlap as a fraction of the run width, in [0.0, 1.0]."""
        run_width = run_end - run_start + 1
        if run_width <= 0:
            return 0.0
        return self.overlap_pixels(run_start, run_end) / run_width


def _build_torso_x_window(
    shoulder_mid_x: Optional[float],
    hip_mid_x: Optional[float],
    elbow_xs: Optional[List[float]],
    wrist_xs: Optional[List[float]],
    image_width: int,
    margin_fraction: float = 0.25,
    shoulder_width: Optional[float] = None,
    hip_width: Optional[float] = None,
) -> TorsoXWindow:
    """Derive the expected horizontal torso extent from pose landmark X positions.

    Strategy (in priority order)
    -----------------------------
    1.  If shoulder_width or hip_width is provided, use that to directly size
        the torso half-width around the torso centerline (midpoints of shoulders/hips).
    2.  If both shoulder and hip midpoints exist, use their center as the torso
        centerline, and estimate half-width from shoulder/hip width or fall back
        to ~15% of image width.
    3.  If only one of shoulder or hip is available, use it as the single
        reference centre and apply a margin of 1.5× estimated half-width.
    4.  If no landmarks are available at all, return an invalid window so
        the caller can fall back to widest-segment selection.

    Parameters
    ----------
    shoulder_mid_x : Pixel X of the shoulder midpoint (None if unavailable).
    hip_mid_x      : Pixel X of the hip midpoint (None if unavailable).
    elbow_xs       : List of pixel X positions of visible elbows (may be empty).
    wrist_xs       : List of pixel X positions of visible wrists (may be empty).
    image_width    : Full image width in pixels.
    margin_fraction: Fraction of the raw window width to add as padding on each side.
    shoulder_width : Horizontal pixel distance between shoulders (|r_sh - l_sh|).
    hip_width      : Horizontal pixel distance between hips (|r_hip - l_hip|).

    Returns
    -------
    TorsoXWindow
    """
    if shoulder_mid_x is None and hip_mid_x is None:
        return TorsoXWindow(x_min=0, x_max=image_width - 1,
                            margin_px=0, source="none", is_valid=False)

    # Raw horizontal reference points for the torso centre
    ref_xs = [x for x in (shoulder_mid_x, hip_mid_x) if x is not None]
    raw_centre = sum(ref_xs) / len(ref_xs)

    # Determine half-width representing the torso radius
    widths = [w for w in (shoulder_width, hip_width) if w is not None and w > 0]
    if widths:
        # Landmark-derived torso half width (inter-shoulder or inter-hip / 2)
        raw_half = max(max(widths) / 2.0, 15.0)
        source = "inter-landmark width"
    elif shoulder_mid_x is not None and hip_mid_x is not None:
        # Fall back to image-relative estimate if lateral width points not provided
        raw_half = max(image_width * 0.15, 25.0)
        source = "shoulder_mid + hip_mid"
    elif shoulder_mid_x is not None:
        raw_half = max(image_width * 0.15, 25.0)
        source = "shoulder_mid (hip unavailable)"
    else:
        raw_half = max(image_width * 0.15, 25.0)
        source = "hip_mid (shoulder unavailable)"

    margin_px = max(5, int(round(raw_half * margin_fraction)))

    # The window is symmetric around the torso centre.
    x_min_raw = raw_centre - raw_half - margin_px
    x_max_raw = raw_centre + raw_half + margin_px

    x_min = max(0, int(round(x_min_raw)))
    x_max = min(image_width - 1, int(round(x_max_raw)))

    if x_min >= x_max:
        return TorsoXWindow(x_min=0, x_max=image_width - 1,
                            margin_px=0, source="degenerate_window", is_valid=False)

    return TorsoXWindow(
        x_min=x_min,
        x_max=x_max,
        margin_px=margin_px,
        source=source,
        is_valid=True,
    )


@dataclasses.dataclass(frozen=True)
class TorsoWidthResult:
    """Result of pose-robust torso boundary extraction at a single scanline.

    Attributes
    ----------
    image_y          : Scanline row index.
    relative_y       : Body-relative y in [0, 1] if bounding info was provided.
    left_x, right_x  : Inclusive pixel columns of the selected torso interval.
                       None when extraction failed.
    width_pixels     : right_x - left_x + 1, or None.
    segment_count    : Total disconnected foreground intervals on this scanline.
    selected_interval: Index into the all_intervals list that was chosen,
                       or None.
    all_intervals    : All (start_x, end_x) foreground runs on this scanline.
    rejected_intervals: Runs that were NOT selected (for diagnostics).
    torso_window     : The TorsoXWindow used for selection.
    selection_reason : One of the confidence category strings from the module
                       docstring (LANDMARK_GUIDED, WIDEST_FALLBACK, etc.).
    is_valid         : True only when a torso interval was identified.
    warnings         : Human-readable warnings; never empty when is_valid=False.
    """

    image_y: int
    relative_y: Optional[float]
    left_x: Optional[int]
    right_x: Optional[int]
    width_pixels: Optional[int]
    segment_count: int
    selected_interval: Optional[int]          # index into all_intervals
    all_intervals: List[Tuple[int, int]]
    rejected_intervals: List[Tuple[int, int]]
    torso_window: TorsoXWindow
    selection_reason: str
    is_valid: bool
    warnings: List[str]

    def to_scanline_slice(self) -> ScanlineSlice:
        """Convert to a ScanlineSlice for compatibility with downstream consumers."""
        return ScanlineSlice(
            image_y=self.image_y,
            relative_y=self.relative_y,
            left_x=self.left_x,
            right_x=self.right_x,
            width_pixels=self.width_pixels,
            is_valid=self.is_valid,
            segment_count=self.segment_count,
        )


def torso_aware_scanline_width(
    binary_mask: np.ndarray,
    image_y: int,
    shoulder_mid_x: Optional[float] = None,
    hip_mid_x: Optional[float] = None,
    elbow_xs: Optional[List[float]] = None,
    wrist_xs: Optional[List[float]] = None,
    silhouette_top_y: Optional[int] = None,
    silhouette_height: Optional[int] = None,
    torso_window_margin_fraction: float = 0.25,
    min_overlap_fraction: float = 0.10,
    shoulder_width: Optional[float] = None,
    hip_width: Optional[float] = None,
) -> TorsoWidthResult:
    """Extract the torso width at a single scanline using anatomical landmark guidance.

    The function builds a TorsoXWindow from landmark X-coordinates,
    evaluates all disconnected foreground intervals, selects the one with
    the highest overlap with the torso window, and returns diagnostic
    information about every interval that was considered.

    When no landmark guidance is available (all landmark args None), the
    function falls back to widest-segment selection.  It never silently
    returns an outermost-span width or combines disjoint arm and torso pixels.

    Parameters
    ----------
    binary_mask              : 2D binary uint8 mask (foreground > 0).
    image_y                  : Scanline row to sample.
    shoulder_mid_x           : Pixel X of shoulder midpoint at this level.
    hip_mid_x                : Pixel X of hip midpoint at this level.
    elbow_xs                 : Pixel X positions of visible elbows (may be []).
    wrist_xs                 : Pixel X positions of visible wrists (may be []).
    silhouette_top_y         : Bounding-box top row (for relative_y calculation).
    silhouette_height        : Bounding-box height in pixels (for relative_y).
    torso_window_margin_fraction : Fraction of torso half-width added as margin.
    min_overlap_fraction     : Minimum overlap fraction for an interval to be
                               considered a torso candidate.
    shoulder_width           : Horizontal pixel distance between shoulders.
    hip_width                : Horizontal pixel distance between hips.

    Returns
    -------
    TorsoWidthResult
    """
    h, w = binary_mask.shape[:2]

    rel_y: Optional[float] = None
    if silhouette_top_y is not None and silhouette_height is not None and silhouette_height > 0:
        rel_y = (image_y - silhouette_top_y) / float(silhouette_height)

    def _no_result(reason: str, warns: List[str], window: Optional[TorsoXWindow] = None) -> TorsoWidthResult:
        return TorsoWidthResult(
            image_y=image_y, relative_y=rel_y,
            left_x=None, right_x=None, width_pixels=None,
            segment_count=0, selected_interval=None,
            all_intervals=[], rejected_intervals=[],
            torso_window=window or TorsoXWindow(
                x_min=0, x_max=w - 1, margin_px=0, source="none", is_valid=False),
            selection_reason=reason,
            is_valid=False, warnings=warns,
        )

    # Out-of-bounds guard
    if not (0 <= image_y < h):
        return _no_result("OUT_OF_BOUNDS",
                          [f"Scanline y={image_y} is outside image bounds (h={h})."])

    row = binary_mask[image_y, :]
    # All contiguous runs, widest-first
    runs_widest_first = _extract_contiguous_runs(row)
    segment_count = len(runs_widest_first)

    if segment_count == 0:
        return _no_result("EMPTY_SCANLINE",
                          [f"No foreground pixels at scanline y={image_y}."])

    # Build torso window from landmark guidance
    torso_window = _build_torso_x_window(
        shoulder_mid_x=shoulder_mid_x,
        hip_mid_x=hip_mid_x,
        elbow_xs=list(elbow_xs or []),
        wrist_xs=list(wrist_xs or []),
        image_width=w,
        margin_fraction=torso_window_margin_fraction,
        shoulder_width=shoulder_width,
        hip_width=hip_width,
    )

    # ── Single segment: unambiguous regardless of landmark guidance ──────────
    if segment_count == 1:
        s, e = runs_widest_first[0]
        return TorsoWidthResult(
            image_y=image_y, relative_y=rel_y,
            left_x=s, right_x=e, width_pixels=e - s + 1,
            segment_count=1, selected_interval=0,
            all_intervals=runs_widest_first, rejected_intervals=[],
            torso_window=torso_window,
            selection_reason="SINGLE_SEGMENT",
            is_valid=True, warnings=[],
        )

    # ── Multiple segments ─────────────────────────────────────────────────────
    warnings: List[str] = [
        f"Scanline at y={image_y} contains {segment_count} disconnected segments "
        "(arm or garment separation)."
    ]

    # Fallback: no landmark guidance → widest segment
    if not torso_window.is_valid:
        s, e = runs_widest_first[0]
        return TorsoWidthResult(
            image_y=image_y, relative_y=rel_y,
            left_x=s, right_x=e, width_pixels=e - s + 1,
            segment_count=segment_count, selected_interval=0,
            all_intervals=runs_widest_first,
            rejected_intervals=runs_widest_first[1:],
            torso_window=torso_window,
            selection_reason="WIDEST_FALLBACK",
            is_valid=True,
            warnings=warnings + [
                "No landmark guidance available; widest segment selected as torso estimate."
            ],
        )

    # Score each interval by overlap fraction with torso window
    # Sort original list by index to track which is which; runs already widest-first
    scored: List[Tuple[float, int, int, int]] = []  # (overlap_frac, orig_idx, start, end)
    for idx, (s, e) in enumerate(runs_widest_first):
        ov = torso_window.overlap_fraction(s, e)
        scored.append((ov, idx, s, e))

    # Pick the interval with the best torso-window overlap that meets threshold
    scored.sort(key=lambda t: t[0], reverse=True)
    best_ov, best_orig_idx, best_s, best_e = scored[0]

    if best_ov < min_overlap_fraction:
        # No interval overlaps the torso window enough — return unavailable
        return _no_result(
            "WINDOW_NO_OVERLAP",
            warnings + [
                f"No foreground interval overlaps the torso window "
                f"[{torso_window.x_min}, {torso_window.x_max}] "
                f"by at least {min_overlap_fraction:.0%}. "
                f"Best overlap was {best_ov:.1%} for interval [{best_s}, {best_e}]. "
                "Measurement withheld; retake with arms at sides."
            ],
            torso_window,
        )

    selected_orig_idx_set = {best_orig_idx}
    rejected = [(s, e) for idx, (s, e) in enumerate(runs_widest_first)
                if idx not in selected_orig_idx_set]

    return TorsoWidthResult(
        image_y=image_y, relative_y=rel_y,
        left_x=best_s, right_x=best_e, width_pixels=best_e - best_s + 1,
        segment_count=segment_count, selected_interval=best_orig_idx,
        all_intervals=runs_widest_first, rejected_intervals=rejected,
        torso_window=torso_window,
        selection_reason="LANDMARK_GUIDED",
        is_valid=True,
        warnings=warnings + [
            f"Torso window [{torso_window.x_min}, {torso_window.x_max}] "
            f"(source: {torso_window.source}); "
            f"selected interval [{best_s}, {best_e}] "
            f"with {best_ov:.1%} overlap. "
            f"{len(rejected)} interval(s) rejected as non-torso."
        ],
    )
