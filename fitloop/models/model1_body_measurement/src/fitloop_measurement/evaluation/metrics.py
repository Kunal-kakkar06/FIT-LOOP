"""
evaluation/metrics.py
---------------------
Quantitative evaluation metrics for ground-truth comparison.

Calculates:
- Sample counts (total, valid predictions, failed predictions, coverage ratio).
- Accuracy metrics among valid predictions (MAE, RMSE, signed bias, median absolute error, 90th percentile).
- Precision thresholds (% within ±1.0 cm, % within ±2.0 cm).
- Transparent denominator reporting (both valid-prediction and overall totals).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional

import numpy as np

from .schemas import ComparisonRecord


@dataclasses.dataclass(frozen=True)
class MeasurementMetricSummary:
    """Metrics calculated independently for a single measurement dimension."""

    measurement_name: str
    total_samples: int
    valid_predictions_count: int
    failed_predictions_count: int
    coverage_rate: float            # valid / total

    # Accuracy metrics (computed over valid predictions)
    mae_cm: Optional[float]
    rmse_cm: Optional[float]
    signed_bias_cm: Optional[float] # mean(prediction - reference)
    median_ae_cm: Optional[float]
    p90_ae_cm: Optional[float]      # 90th percentile absolute error

    pct_within_1cm_valid: Optional[float] # % of valid predictions within ±1 cm
    pct_within_2cm_valid: Optional[float] # % of valid predictions within ±2 cm
    pct_within_1cm_total: Optional[float] # % of all test instances within ±1 cm
    pct_within_2cm_total: Optional[float] # % of all test instances within ±2 cm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurement_name": self.measurement_name,
            "total_samples": self.total_samples,
            "valid_predictions_count": self.valid_predictions_count,
            "failed_predictions_count": self.failed_predictions_count,
            "coverage_rate": round(self.coverage_rate, 4),
            "mae_cm": round(self.mae_cm, 3) if self.mae_cm is not None else None,
            "rmse_cm": round(self.rmse_cm, 3) if self.rmse_cm is not None else None,
            "signed_bias_cm": round(self.signed_bias_cm, 3) if self.signed_bias_cm is not None else None,
            "median_ae_cm": round(self.median_ae_cm, 3) if self.median_ae_cm is not None else None,
            "p90_ae_cm": round(self.p90_ae_cm, 3) if self.p90_ae_cm is not None else None,
            "pct_within_1cm_valid": round(self.pct_within_1cm_valid, 2) if self.pct_within_1cm_valid is not None else None,
            "pct_within_2cm_valid": round(self.pct_within_2cm_valid, 2) if self.pct_within_2cm_valid is not None else None,
            "pct_within_1cm_total": round(self.pct_within_1cm_total, 2) if self.pct_within_1cm_total is not None else None,
            "pct_within_2cm_total": round(self.pct_within_2cm_total, 2) if self.pct_within_2cm_total is not None else None,
        }


def calculate_metrics_for_records(
    measurement_name: str,
    records: List[ComparisonRecord],
) -> MeasurementMetricSummary:
    """Calculate error metrics for a specific measurement from comparison records.

    Parameters
    ----------
    measurement_name:
        Name of the evaluated dimension (e.g. "waist").
    records:
        List of ComparisonRecords for this measurement.

    Returns
    -------
    MeasurementMetricSummary
        Independent error and coverage statistics.
    """
    total = len(records)
    if total == 0:
        return MeasurementMetricSummary(
            measurement_name=measurement_name,
            total_samples=0,
            valid_predictions_count=0,
            failed_predictions_count=0,
            coverage_rate=0.0,
            mae_cm=None,
            rmse_cm=None,
            signed_bias_cm=None,
            median_ae_cm=None,
            p90_ae_cm=None,
            pct_within_1cm_valid=None,
            pct_within_2cm_valid=None,
            pct_within_1cm_total=None,
            pct_within_2cm_total=None,
        )

    valid_records = [r for r in records if r.is_valid_prediction and r.predicted_cm is not None]
    n_valid = len(valid_records)
    n_failed = total - n_valid
    coverage = float(n_valid) / float(total)

    if n_valid == 0:
        return MeasurementMetricSummary(
            measurement_name=measurement_name,
            total_samples=total,
            valid_predictions_count=0,
            failed_predictions_count=n_failed,
            coverage_rate=0.0,
            mae_cm=None,
            rmse_cm=None,
            signed_bias_cm=None,
            median_ae_cm=None,
            p90_ae_cm=None,
            pct_within_1cm_valid=None,
            pct_within_2cm_valid=None,
            pct_within_1cm_total=0.0,
            pct_within_2cm_total=0.0,
        )

    errors = np.array([r.error_cm for r in valid_records], dtype=float)
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    median_ae = float(np.median(abs_errors))
    p90_ae = float(np.percentile(abs_errors, 90))

    within_1cm_count = int(np.count_nonzero(abs_errors <= 1.0))
    within_2cm_count = int(np.count_nonzero(abs_errors <= 2.0))

    pct_within_1cm_valid = (within_1cm_count / n_valid) * 100.0
    pct_within_2cm_valid = (within_2cm_count / n_valid) * 100.0

    pct_within_1cm_total = (within_1cm_count / total) * 100.0
    pct_within_2cm_total = (within_2cm_count / total) * 100.0

    return MeasurementMetricSummary(
        measurement_name=measurement_name,
        total_samples=total,
        valid_predictions_count=n_valid,
        failed_predictions_count=n_failed,
        coverage_rate=coverage,
        mae_cm=mae,
        rmse_cm=rmse,
        signed_bias_cm=bias,
        median_ae_cm=median_ae,
        p90_ae_cm=p90_ae,
        pct_within_1cm_valid=pct_within_1cm_valid,
        pct_within_2cm_valid=pct_within_2cm_valid,
        pct_within_1cm_total=pct_within_1cm_total,
        pct_within_2cm_total=pct_within_2cm_total,
    )
