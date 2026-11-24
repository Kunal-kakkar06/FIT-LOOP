"""
evaluation/runner.py
--------------------
Evaluation runner matching reference ground-truth records with model predictions.

Ensures:
- Robust key-based matching on (participant_id, session_id, measurement_name); never matches by row order.
- Explicit detection of unmatched references and missing predictions.
- Per-measurement reporting (never averaging over dissimilar body regions).
- Subgroup analyses (e.g. clothing conditions, height ranges).
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Optional

from .metrics import MeasurementMetricSummary, calculate_metrics_for_records
from .schemas import (
    ClothingCondition,
    ComparisonRecord,
    ModelPredictionRecord,
    ParticipantSessionRecord,
)


class EvaluationReport:
    """Consolidated ground-truth evaluation report across all measurements and subgroups."""

    def __init__(
        self,
        comparisons: List[ComparisonRecord],
        unmatched_references: List[str],
        unmatched_predictions: List[str],
        model_version: str = "model1_v0.1.0",
    ) -> None:
        self.comparisons = comparisons
        self.unmatched_references = unmatched_references
        self.unmatched_predictions = unmatched_predictions
        self.model_version = model_version

        # Calculate metrics per measurement
        self.measurement_metrics: Dict[str, MeasurementMetricSummary] = {}
        unique_measurements = sorted(list({c.measurement_name for c in comparisons}))
        for m_name in unique_measurements:
            sub = [c for c in comparisons if c.measurement_name == m_name]
            self.measurement_metrics[m_name] = calculate_metrics_for_records(m_name, sub)

    def get_clothing_subgroup_metrics(self) -> Dict[str, Dict[str, MeasurementMetricSummary]]:
        """Calculate per-measurement metrics split by clothing condition."""
        subgroups: Dict[str, Dict[str, MeasurementMetricSummary]] = {}
        conditions = {c.clothing_condition for c in self.comparisons}

        for cond in conditions:
            sub_records = [c for c in self.comparisons if c.clothing_condition == cond]
            m_dict = {}
            for m_name in self.measurement_metrics.keys():
                m_sub = [c for c in sub_records if c.measurement_name == m_name]
                m_dict[m_name] = calculate_metrics_for_records(m_name, m_sub)
            subgroups[cond.value] = m_dict

        return subgroups

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "total_comparisons": len(self.comparisons),
            "unmatched_references_count": len(self.unmatched_references),
            "unmatched_predictions_count": len(self.unmatched_predictions),
            "unmatched_references": list(self.unmatched_references),
            "unmatched_predictions": list(self.unmatched_predictions),
            "per_measurement_metrics": {
                k: v.to_dict() for k, v in self.measurement_metrics.items()
            },
            "clothing_subgroups": {
                cond: {m_name: m_summary.to_dict() for m_name, m_summary in m_dict.items()}
                for cond, m_dict in self.get_clothing_subgroup_metrics().items()
            },
        }

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        """Export tabular metric summary rows for reporting."""
        rows = []
        for m_name, m in self.measurement_metrics.items():
            row = m.to_dict()
            row["model_version"] = self.model_version
            rows.append(row)
        return rows


def run_evaluation(
    reference_sessions: List[ParticipantSessionRecord],
    predictions: List[ModelPredictionRecord],
    model_version: str = "model1_v0.1.0",
) -> EvaluationReport:
    """Match ground-truth records with model predictions and compute evaluation metrics.

    Matching is strictly performed by composite key: (participant_id, session_id, measurement_name).
    """
    # Index predictions by composite key and reject duplicates
    pred_map: Dict[tuple[str, str, str], ModelPredictionRecord] = {}
    for p in predictions:
        key = (p.participant_id.strip(), p.session_id.strip(), p.measurement_name.strip().lower())
        if key in pred_map:
            raise ValueError(f"Duplicate prediction record encountered for key: {key}")
        pred_map[key] = p

    comparisons: List[ComparisonRecord] = []
    unmatched_refs: List[str] = []
    matched_pred_keys: set[tuple[str, str, str]] = set()

    for session in reference_sessions:
        p_id = session.participant_id.strip()
        s_id = session.session_id.strip()

        for m_name, m_ref in session.measurements.items():
            key = (p_id, s_id, m_name.strip().lower())

            if key in pred_map:
                matched_pred_keys.add(key)
                pred = pred_map[key]

                is_valid = pred.predicted_cm is not None and pred.prediction_status in {"SUCCESS", "PARTIAL"}
                error = (pred.predicted_cm - m_ref.reference_value_cm) if is_valid else None
                abs_err = abs(error) if error is not None else None

                comparisons.append(
                    ComparisonRecord(
                        participant_id=p_id,
                        session_id=s_id,
                        measurement_name=m_name.lower(),
                        reference_cm=m_ref.reference_value_cm,
                        predicted_cm=pred.predicted_cm,
                        is_valid_prediction=is_valid,
                        prediction_status=pred.prediction_status,
                        clothing_condition=session.clothing_condition,
                        standing_height_cm=session.standing_height_cm,
                        error_cm=error,
                        abs_error_cm=abs_err,
                        warnings=pred.warnings,
                    )
                )
            else:
                unmatched_refs.append(f"{p_id}/{s_id}/{m_name}")
                # Record missing prediction
                comparisons.append(
                    ComparisonRecord(
                        participant_id=p_id,
                        session_id=s_id,
                        measurement_name=m_name.lower(),
                        reference_cm=m_ref.reference_value_cm,
                        predicted_cm=None,
                        is_valid_prediction=False,
                        prediction_status="MISSING_PREDICTION",
                        clothing_condition=session.clothing_condition,
                        standing_height_cm=session.standing_height_cm,
                        error_cm=None,
                        abs_error_cm=None,
                        warnings=["Model prediction was not supplied for this record."],
                    )
                )

    # Detect unmatched predictions
    unmatched_preds = []
    for key, pred in pred_map.items():
        if key not in matched_pred_keys:
            unmatched_preds.append(f"{key[0]}/{key[1]}/{key[2]}")

    return EvaluationReport(
        comparisons=comparisons,
        unmatched_references=unmatched_refs,
        unmatched_predictions=unmatched_preds,
        model_version=model_version,
    )
