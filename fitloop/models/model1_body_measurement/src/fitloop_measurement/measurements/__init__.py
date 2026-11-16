# fitloop_measurement/measurements/__init__.py
"""
measurements sub-package
------------------------
Calibrated circumference estimation engine for FitLoop Model 1.
"""

from .calibration import calibrate_dual_views, calibrate_view
from .circumference import estimate_circumference_at_level, ramanujan_ellipse_perimeter
from .estimator import CircumferenceEstimator
from .schemas import (
    AnatomicalLevel,
    CalibrationStatus,
    CircumferenceMeasurement,
    DualViewCalibration,
    MeasurementEngineResult,
    MeasurementStatus,
    SliceSelectionResult,
    ViewCalibration,
)
from .slice_selector import select_anatomical_slice

__all__ = [
    "CircumferenceEstimator",
    "calibrate_view",
    "calibrate_dual_views",
    "ramanujan_ellipse_perimeter",
    "estimate_circumference_at_level",
    "select_anatomical_slice",
    "AnatomicalLevel",
    "CalibrationStatus",
    "MeasurementStatus",
    "ViewCalibration",
    "DualViewCalibration",
    "SliceSelectionResult",
    "CircumferenceMeasurement",
    "MeasurementEngineResult",
]
