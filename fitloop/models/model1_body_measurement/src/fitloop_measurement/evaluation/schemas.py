"""
evaluation/schemas.py
---------------------
Data schemas for the Ground-Truth Evaluation Framework (Stage 6).

Covers:
- ClothingCondition: Standardized capture apparel categories.
- ManualMeasurementRecord: Repeated anthropometric readings with derivation methods.
- ParticipantSessionRecord: Complete reference session for one participant.
- ModelPredictionRecord: Point-in-time prediction output from Model 1.
- ComparisonRecord: Paired record uniting ground truth and prediction for a single dimension.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional


class ClothingCondition(str, Enum):
    """Subject apparel condition during photograph capture."""

    FITTED = "fitted"              # Compression garments / form-fitting athletic wear
    STANDARD_CASUAL = "standard"    # Everyday casual wear (t-shirt, jeans, etc.)
    LOOSE = "loose"                # Oversized / baggy clothing
    UNKNOWN = "unknown"


class DerivationMethod(str, Enum):
    """Method used to derive final reference value from repeated manual measurements."""

    MEAN_DUAL = "mean_dual"         # Mean of 2 readings differing by <= tolerance
    MEAN_TRIPLE = "mean_triple"     # Mean of 3 readings when first 2 diverged
    MEDIAN_TRIPLE = "median_triple"
    SINGLE_READING = "single_reading"


@dataclasses.dataclass(frozen=True)
class ManualMeasurementRecord:
    """Repeated manual anthropometric tape measurements for a single dimension."""

    measurement_name: str          # e.g., "chest", "waist", "abdomen", "hips"
    readings_cm: List[float]       # At least 2 independent readings
    reference_value_cm: float      # Consolidated ground-truth reference value
    derivation_method: DerivationMethod
    inter_reading_diff_cm: float   # Max difference between readings
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurement_name": self.measurement_name,
            "readings_cm": list(self.readings_cm),
            "reference_value_cm": round(self.reference_value_cm, 2),
            "derivation_method": self.derivation_method.value,
            "inter_reading_diff_cm": round(self.inter_reading_diff_cm, 2),
            "notes": self.notes,
        }


@dataclasses.dataclass(frozen=True)
class ParticipantSessionRecord:
    """Ground-truth session record for a pseudonymous participant."""

    participant_id: str            # Pseudonymous identifier (e.g. "P001", never personal names)
    session_id: str                # Session / scan identifier
    standing_height_cm: float      # Verified manual standing stature
    clothing_condition: ClothingCondition
    measurements: Dict[str, ManualMeasurementRecord]  # Keyed by measurement name
    front_image_path: Optional[str] = None           # Path to front photo
    side_image_path: Optional[str] = None            # Path to side photo
    protocol_version: str = "1.0"
    capture_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "standing_height_cm": self.standing_height_cm,
            "clothing_condition": self.clothing_condition.value,
            "front_image_path": self.front_image_path,
            "side_image_path": self.side_image_path,
            "protocol_version": self.protocol_version,
            "measurements": {k: v.to_dict() for k, v in self.measurements.items()},
            "capture_notes": self.capture_notes,
        }


@dataclasses.dataclass(frozen=True)
class ModelPredictionRecord:
    """Container for model predictions on a specific session."""

    participant_id: str
    session_id: str
    measurement_name: str
    predicted_cm: Optional[float]
    prediction_status: str         # e.g. "SUCCESS", "PARTIAL", "UNCALIBRATED"
    warnings: List[str] = dataclasses.field(default_factory=list)
    model_version: str = "model1_v0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "measurement_name": self.measurement_name,
            "predicted_cm": round(self.predicted_cm, 2) if self.predicted_cm is not None else None,
            "prediction_status": self.prediction_status,
            "warnings": list(self.warnings),
            "model_version": self.model_version,
        }


@dataclasses.dataclass(frozen=True)
class ComparisonRecord:
    """Unified record pairing reference ground truth and model prediction."""

    participant_id: str
    session_id: str
    measurement_name: str
    reference_cm: float
    predicted_cm: Optional[float]
    is_valid_prediction: bool
    prediction_status: str
    clothing_condition: ClothingCondition
    standing_height_cm: float
    error_cm: Optional[float]      # predicted_cm - reference_cm (Signed error)
    abs_error_cm: Optional[float]  # abs(error_cm)
    warnings: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "measurement_name": self.measurement_name,
            "reference_cm": round(self.reference_cm, 2),
            "predicted_cm": round(self.predicted_cm, 2) if self.predicted_cm is not None else None,
            "is_valid_prediction": self.is_valid_prediction,
            "prediction_status": self.prediction_status,
            "clothing_condition": self.clothing_condition.value,
            "standing_height_cm": self.standing_height_cm,
            "error_cm": round(self.error_cm, 2) if self.error_cm is not None else None,
            "abs_error_cm": round(self.abs_error_cm, 2) if self.abs_error_cm is not None else None,
            "warnings": list(self.warnings),
        }
