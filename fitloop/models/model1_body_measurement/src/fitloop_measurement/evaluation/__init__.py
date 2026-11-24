# fitloop_measurement/evaluation/__init__.py
"""
evaluation sub-package
-----------------------
Ground-truth validation and quantitative evaluation framework for FitLoop Model 1.
"""

from .dataset import (
    PILOT_CSV_HEADERS,
    DatasetValidationError,
    create_manual_measurement,
    load_ground_truth_csv,
    load_predictions_json,
    parse_api_response_to_predictions,
    save_predictions_json,
    split_dataset_by_participant,
    validate_participant_session,
    write_pilot_csv_template,
)
from .metrics import (
    MeasurementMetricSummary,
    calculate_metrics_for_records,
)
from .runner import (
    EvaluationReport,
    run_evaluation,
)
from .schemas import (
    ClothingCondition,
    ComparisonRecord,
    DerivationMethod,
    ManualMeasurementRecord,
    ModelPredictionRecord,
    ParticipantSessionRecord,
)

__all__ = [
    "ClothingCondition",
    "DerivationMethod",
    "ManualMeasurementRecord",
    "ParticipantSessionRecord",
    "ModelPredictionRecord",
    "ComparisonRecord",
    "MeasurementMetricSummary",
    "EvaluationReport",
    "DatasetValidationError",
    "create_manual_measurement",
    "validate_participant_session",
    "split_dataset_by_participant",
    "calculate_metrics_for_records",
    "run_evaluation",
    "parse_api_response_to_predictions",
    "save_predictions_json",
    "load_predictions_json",
]
