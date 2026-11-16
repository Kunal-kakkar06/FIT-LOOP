"""
quality/pair_checker.py
-----------------------
Paired-image validation layer for FitLoop Model 1.

Accepts front and side views and produces a structured, machine-readable report
identifying whether each view satisfies framing, quality, and height-calibration
prerequisites for downstream 3D reconstruction and measurement inference.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..config import HEIGHT_MAX_CM, HEIGHT_MIN_CM
from ..image_io import ImageLoadError, load_image
from .checker import QualityChecker, QualityReport


class ErrorCode(str, Enum):
    """Machine-readable error codes for paired-image quality validation."""

    MISSING_IMAGE = "MISSING_IMAGE"
    UNREADABLE_IMAGE = "UNREADABLE_IMAGE"
    INVALID_IMAGE_TYPE = "INVALID_IMAGE_TYPE"
    INVALID_HEIGHT = "INVALID_HEIGHT"
    QUALITY_CHECK_FAILED = "QUALITY_CHECK_FAILED"


@dataclasses.dataclass
class QualityIssue:
    """Structured issue identifying code, message, and actionable recommendation."""

    code: ErrorCode
    message: str
    actionable_fix: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code.value,
            "message": self.message,
            "actionable_fix": self.actionable_fix,
        }


@dataclasses.dataclass
class ViewValidationResult:
    """Validation outcome for a single view (front or side)."""

    view_name: str  # "front" or "side"
    passed: bool
    issues: List[QualityIssue] = dataclasses.field(default_factory=list)
    quality_report: Optional[QualityReport] = None
    diagnostics: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_name": self.view_name,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "diagnostics": self.diagnostics,
        }


@dataclasses.dataclass
class PairedQualityReport:
    """Consolidated report across front and side captures and reference height."""

    passed: bool
    front_result: ViewValidationResult
    side_result: ViewValidationResult
    height_validated: bool
    height_issues: List[QualityIssue] = dataclasses.field(default_factory=list)
    pair_issues: List[QualityIssue] = dataclasses.field(default_factory=list)

    @property
    def has_front_failure(self) -> bool:
        return not self.front_result.passed

    @property
    def has_side_failure(self) -> bool:
        return not self.side_result.passed

    # Backwards-compatibility helpers for Stage 1 consumers
    @property
    def front_report(self) -> Optional[QualityReport]:
        return self.front_result.quality_report

    @property
    def side_report(self) -> Optional[QualityReport]:
        return self.side_result.quality_report

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "height_validated": self.height_validated,
            "height_issues": [i.to_dict() for i in self.height_issues],
            "pair_issues": [i.to_dict() for i in self.pair_issues],
            "front": self.front_result.to_dict(),
            "side": self.side_result.to_dict(),
        }

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"PairedQualityReport [{status}]"]

        for issue in self.height_issues:
            lines.append(f"  ✗ [Height] {issue.message} -> Fix: {issue.actionable_fix}")

        for issue in self.pair_issues:
            lines.append(f"  ✗ [Pair] {issue.message} -> Fix: {issue.actionable_fix}")

        # Front summary
        f_status = "PASSED" if self.front_result.passed else "FAILED"
        lines.append(f"  --- Front View [{f_status}] ---")
        for issue in self.front_result.issues:
            lines.append(f"    ✗ [{issue.code.value}] {issue.message} -> Fix: {issue.actionable_fix}")
        for k, v in self.front_result.diagnostics.items():
            lines.append(f"    · {k}: {v}")

        # Side summary
        s_status = "PASSED" if self.side_result.passed else "FAILED"
        lines.append(f"  --- Side View [{s_status}] ---")
        for issue in self.side_result.issues:
            lines.append(f"    ✗ [{issue.code.value}] {issue.message} -> Fix: {issue.actionable_fix}")
        for k, v in self.side_result.diagnostics.items():
            lines.append(f"    · {k}: {v}")

        return "\n".join(lines)


class PairedQualityChecker:
    """Validates front and side captures independently and verifies input calibration scale."""

    def __init__(self, quality_checker: Optional[QualityChecker] = None) -> None:
        self._checker = quality_checker or QualityChecker()
        self._owns_checker = quality_checker is None

    def _validate_height(self, subject_height_cm: Optional[float]) -> tuple[bool, List[QualityIssue]]:
        issues: List[QualityIssue] = []
        if subject_height_cm is None:
            issues.append(
                QualityIssue(
                    code=ErrorCode.INVALID_HEIGHT,
                    message="Subject standing height is required for metric scale calibration.",
                    actionable_fix="Please provide the user's standing height in centimetres (e.g. 175.0).",
                )
            )
            return False, issues

        if not (HEIGHT_MIN_CM <= subject_height_cm <= HEIGHT_MAX_CM):
            issues.append(
                QualityIssue(
                    code=ErrorCode.INVALID_HEIGHT,
                    message=f"Entered height ({subject_height_cm} cm) is outside plausible human range [{HEIGHT_MIN_CM}, {HEIGHT_MAX_CM}] cm.",
                    actionable_fix=f"Check the height value and provide a measurement between {HEIGHT_MIN_CM} and {HEIGHT_MAX_CM} cm.",
                )
            )
            return False, issues

        return True, issues

    def _evaluate_single_view(
        self,
        image_input: Union[str, Path, np.ndarray, None],
        view_name: str,
        subject_height_cm: Optional[float],
    ) -> ViewValidationResult:
        issues: List[QualityIssue] = []
        diagnostics: Dict[str, Any] = {}

        # 1. Presence check
        if image_input is None:
            issues.append(
                QualityIssue(
                    code=ErrorCode.MISSING_IMAGE,
                    message=f"{view_name.capitalize()} image was not provided.",
                    actionable_fix=f"Please capture and submit a {view_name} view image.",
                )
            )
            return ViewValidationResult(
                view_name=view_name,
                passed=False,
                issues=issues,
                quality_report=None,
                diagnostics=diagnostics,
            )

        # 2. Loading / Type check
        bgr_image: Optional[np.ndarray] = None
        if isinstance(image_input, (str, Path)):
            try:
                bgr_image = load_image(image_input)
            except ImageLoadError as e:
                issues.append(
                    QualityIssue(
                        code=ErrorCode.UNREADABLE_IMAGE,
                        message=f"{view_name.capitalize()} image could not be loaded: {e}",
                        actionable_fix="Ensure file exists, is in supported format (.jpg, .jpeg, .png, .webp) and is ≥ 480×640.",
                    )
                )
                return ViewValidationResult(
                    view_name=view_name,
                    passed=False,
                    issues=issues,
                    quality_report=QualityReport(passed=False, issues=[str(e)]),
                    diagnostics=diagnostics,
                )
            except Exception as e:
                issues.append(
                    QualityIssue(
                        code=ErrorCode.UNREADABLE_IMAGE,
                        message=f"Unexpected error loading {view_name} image: {e}",
                        actionable_fix="Check the image file for corruption.",
                    )
                )
                return ViewValidationResult(
                    view_name=view_name,
                    passed=False,
                    issues=issues,
                    quality_report=QualityReport(passed=False, issues=[str(e)]),
                    diagnostics=diagnostics,
                )
        elif isinstance(image_input, np.ndarray):
            bgr_image = image_input
        else:
            issues.append(
                QualityIssue(
                    code=ErrorCode.INVALID_IMAGE_TYPE,
                    message=f"Invalid image type for {view_name}: {type(image_input)}",
                    actionable_fix="Pass a filepath (str or Path) or a valid BGR numpy array.",
                )
            )
            return ViewValidationResult(
                view_name=view_name,
                passed=False,
                issues=issues,
                quality_report=None,
                diagnostics=diagnostics,
            )

        # 3. Quality evaluation using existing QualityChecker
        # Note: height validation is handled separately at pair level,
        # but passed to checker so single-image quality report stays consistent.
        q_report = self._checker.run(bgr_image, subject_height_cm=subject_height_cm)
        diagnostics.update(q_report.info)

        # Map single-image checker issues to structured issues
        for issue_text in q_report.issues:
            # Avoid repeating height issues in view report if already tracked at pair level
            if "height_cm" in issue_text:
                continue

            # Produce contextual actionable guidance
            if "too small" in issue_text:
                action = "Re-upload a higher resolution photograph (at least 480x640)."
            elif "No person detected" in issue_text:
                action = f"Ensure the user's full body is clearly visible in the {view_name} capture with unobstructed lighting."
            elif "Body occupies only" in issue_text:
                action = "Step back or adjust camera framing so the subject occupies at least 70% of frame height."
            elif "off-centre" in issue_text:
                action = "Center the subject horizontally within the frame."
            else:
                action = "Ensure subject is clearly visible in neutral standing posture against a plain background."

            issues.append(
                QualityIssue(
                    code=ErrorCode.QUALITY_CHECK_FAILED,
                    message=f"{view_name.capitalize()} view: {issue_text}",
                    actionable_fix=action,
                )
            )

        # Passed if there are no view-specific issues and q_report didn't fail on visual checks
        view_passed = len(issues) == 0

        return ViewValidationResult(
            view_name=view_name,
            passed=view_passed,
            issues=issues,
            quality_report=q_report,
            diagnostics=diagnostics,
        )

    def validate_pair(
        self,
        front_image: Union[str, Path, np.ndarray, None],
        side_image: Union[str, Path, np.ndarray, None],
        subject_height_cm: Optional[float] = None,
    ) -> PairedQualityReport:
        """Independently validate front and side views and verify height input.

        Parameters
        ----------
        front_image:
            Front perspective image file path or BGR array.
        side_image:
            Side perspective image file path or BGR array.
        subject_height_cm:
            User standing height in cm (prerequisite for metric calibration).

        Returns
        -------
        PairedQualityReport
            Combined validation report with view-specific diagnostic breakdowns.
        """
        # Validate height separately
        height_ok, height_issues = self._validate_height(subject_height_cm)

        # Independently validate front and side views
        front_result = self._evaluate_single_view(
            front_image, view_name="front", subject_height_cm=subject_height_cm
        )
        side_result = self._evaluate_single_view(
            side_image, view_name="side", subject_height_cm=subject_height_cm
        )

        pair_issues: List[QualityIssue] = []

        all_passed = height_ok and front_result.passed and side_result.passed

        return PairedQualityReport(
            passed=all_passed,
            front_result=front_result,
            side_result=side_result,
            height_validated=height_ok,
            height_issues=height_issues,
            pair_issues=pair_issues,
        )

    def close(self) -> None:
        if self._owns_checker:
            self._checker.close()

    def __enter__(self) -> "PairedQualityChecker":
        return self

    def __exit__(self, *_) -> None:
        self.close()
