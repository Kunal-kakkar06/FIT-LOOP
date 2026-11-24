"""
evaluation/dataset.py
---------------------
Ground-truth dataset parser, validator, and partition utility.

Enforces:
- Strict numerical validity and sanity bounds.
- Rejection of duplicate keys (participant + session + measurement).
- Explicit missing values (no silent substitution with 0.0).
- Participant-level data partitioning to prevent session leakage across calibration/test splits.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from ..config import HEIGHT_MAX_CM, HEIGHT_MIN_CM
from .schemas import (
    ClothingCondition,
    DerivationMethod,
    ManualMeasurementRecord,
    ModelPredictionRecord,
    ParticipantSessionRecord,
)

# Anthropometric sanity bounds for adult human circumferences [cm]
PLAUSIBLE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "chest": (50.0, 180.0),
    "waist": (45.0, 170.0),
    "abdomen": (45.0, 180.0),
    "hips": (55.0, 190.0),
    "neck": (25.0, 60.0),
    "bicep": (18.0, 65.0),
    "thigh": (30.0, 95.0),
    "wrist": (12.0, 30.0),
    "ankle": (15.0, 40.0),
}


class DatasetValidationError(Exception):
    """Raised when ground-truth dataset records fail validation."""


def create_manual_measurement(
    measurement_name: str,
    readings_cm: List[float],
    max_tolerance_cm: float = 0.5,
    notes: Optional[str] = None,
) -> ManualMeasurementRecord:
    """Create a validated ManualMeasurementRecord from repeated tape readings.

    Protocol Rule:
    - At least 2 readings required.
    - If diff <= max_tolerance_cm, reference value is the arithmetic mean.
    - If diff > max_tolerance_cm and 3 readings exist, reference value is the 3-reading mean or median.
    """
    m_name = measurement_name.lower().strip()

    if not readings_cm or len(readings_cm) < 2:
        raise DatasetValidationError(
            f"Measurement '{m_name}' must have at least 2 repeated readings, got: {readings_cm}"
        )

    for r in readings_cm:
        if r is None or not math.isfinite(r) or r <= 0:
            raise DatasetValidationError(f"Invalid non-finite or non-positive reading in {m_name}: {r}")

    diff = max(readings_cm) - min(readings_cm)

    if len(readings_cm) == 2:
        if diff > max_tolerance_cm:
            # Note warning if tolerance exceeded without third reading
            derivation = DerivationMethod.MEAN_DUAL
        else:
            derivation = DerivationMethod.MEAN_DUAL
        ref_val = sum(readings_cm) / 2.0
    else:
        derivation = DerivationMethod.MEAN_TRIPLE
        ref_val = sum(readings_cm) / float(len(readings_cm))

    # Bounds check
    if m_name in PLAUSIBLE_BOUNDS:
        min_b, max_b = PLAUSIBLE_BOUNDS[m_name]
        if not (min_b <= ref_val <= max_b):
            raise DatasetValidationError(
                f"Reference value for {m_name} ({ref_val:.1f} cm) outside plausible range [{min_b}, {max_b}] cm."
            )

    return ManualMeasurementRecord(
        measurement_name=m_name,
        readings_cm=readings_cm,
        reference_value_cm=ref_val,
        derivation_method=derivation,
        inter_reading_diff_cm=diff,
        notes=notes,
    )


def validate_participant_session(record: ParticipantSessionRecord) -> list[str]:
    """Validate a ParticipantSessionRecord against strict protocol constraints."""
    warnings: list[str] = []

    if not record.participant_id or not record.participant_id.strip():
        raise DatasetValidationError("participant_id cannot be empty.")

    if not record.session_id or not record.session_id.strip():
        raise DatasetValidationError("session_id cannot be empty.")

    # Stature check
    h = record.standing_height_cm
    if h is None or not math.isfinite(h) or not (HEIGHT_MIN_CM <= h <= HEIGHT_MAX_CM):
        raise DatasetValidationError(
            f"Standing height ({h} cm) outside valid human range [{HEIGHT_MIN_CM}, {HEIGHT_MAX_CM}] cm."
        )

    if not record.measurements:
        warnings.append(f"Session '{record.session_id}' has no measurement records.")

    # Image path validation (if provided)
    if record.front_image_path is not None:
        p_front = Path(record.front_image_path)
        if not p_front.exists():
            warnings.append(f"Front image file does not exist: {record.front_image_path}")

    if record.side_image_path is not None:
        p_side = Path(record.side_image_path)
        if not p_side.exists():
            warnings.append(f"Side image file does not exist: {record.side_image_path}")

    return warnings


def split_dataset_by_participant(
    records: List[ParticipantSessionRecord],
    calibration_ratio: float = 0.40,
    seed: int = 42,
) -> Tuple[List[ParticipantSessionRecord], List[ParticipantSessionRecord]]:
    """Partition dataset into Calibration and Evaluation subsets strictly by participant ID.

    Prevents participant leakage: All sessions of a given participant belong exclusively
    to either the calibration split or the evaluation split.
    """
    import random

    if not records:
        return [], []

    # Get unique participant IDs
    participant_ids = sorted(list({r.participant_id for r in records}))

    rng = random.Random(seed)
    rng.shuffle(participant_ids)

    n_calib = max(1, int(round(len(participant_ids) * calibration_ratio)))
    calib_pids = set(participant_ids[:n_calib])
    eval_pids = set(participant_ids[n_calib:])

    calib_records = [r for r in records if r.participant_id in calib_pids]
    eval_records = [r for r in records if r.participant_id in eval_pids]

    # Verify zero leakage
    overlap = set(r.participant_id for r in calib_records) & set(r.participant_id for r in eval_records)
    if overlap:
        raise RuntimeError(f"Participant leakage detected across splits: {overlap}")

    return calib_records, eval_records


PILOT_CSV_HEADERS: list[str] = [
    "participant_id",
    "session_id",
    "standing_height_cm",
    "clothing_condition",
    "front_image_path",
    "side_image_path",
    "chest_r1",
    "chest_r2",
    "chest_r3",
    "waist_r1",
    "waist_r2",
    "waist_r3",
    "abdomen_r1",
    "abdomen_r2",
    "abdomen_r3",
    "hips_r1",
    "hips_r2",
    "hips_r3",
    "capture_notes",
]


def write_pilot_csv_template(output_path: str | Path) -> None:
    """Generate a clean CSV template for pilot data collection."""
    import csv

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PILOT_CSV_HEADERS)


def load_ground_truth_csv(csv_path: str | Path) -> List[ParticipantSessionRecord]:
    """Load and strictly validate participant records from a pilot CSV table."""
    import csv

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth CSV file not found: {path}")

    sessions: List[ParticipantSessionRecord] = []
    seen_keys: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            p_id = (row.get("participant_id") or "").strip()
            s_id = (row.get("session_id") or "").strip()
            if not p_id or not s_id:
                raise DatasetValidationError(f"Row {row_idx}: participant_id and session_id are required.")

            key = f"{p_id}_{s_id}"
            if key in seen_keys:
                raise DatasetValidationError(f"Row {row_idx}: Duplicate session key '{key}'.")
            seen_keys.add(key)

            # Stature height
            raw_h = (row.get("standing_height_cm") or "").strip()
            if not raw_h:
                raise DatasetValidationError(f"Row {row_idx} ({key}): standing_height_cm is required.")
            try:
                h_val = float(raw_h)
            except ValueError:
                raise DatasetValidationError(f"Row {row_idx} ({key}): invalid height value '{raw_h}'.")

            # Clothing condition
            raw_cond = (row.get("clothing_condition") or "fitted").strip().lower()
            try:
                cond = ClothingCondition(raw_cond)
            except ValueError:
                cond = ClothingCondition.UNKNOWN

            # Parse each of the 4 supported dimensions
            measurements: Dict[str, ManualMeasurementRecord] = {}
            for dim in ["chest", "waist", "abdomen", "hips"]:
                readings: list[float] = []
                for r_idx in [1, 2, 3]:
                    val_str = (row.get(f"{dim}_r{r_idx}") or "").strip()
                    if val_str:
                        try:
                            readings.append(float(val_str))
                        except ValueError:
                            raise DatasetValidationError(
                                f"Row {row_idx} ({key}): invalid reading for {dim}_r{r_idx}: '{val_str}'"
                            )

                if readings:
                    if len(readings) < 2:
                        raise DatasetValidationError(
                            f"Row {row_idx} ({key}): '{dim}' must have at least 2 repeated readings (got {len(readings)})."
                        )
                    rec = create_manual_measurement(dim, readings)
                    measurements[dim] = rec

            def _clean_path(raw_path: Optional[str]) -> Optional[str]:
                if not raw_path:
                    return None
                p = raw_path.strip()
                if p.startswith("file://"):
                    p = unquote(urlparse(p).path)
                return p or None

            session = ParticipantSessionRecord(
                participant_id=p_id,
                session_id=s_id,
                standing_height_cm=h_val,
                clothing_condition=cond,
                front_image_path=_clean_path(row.get("front_image_path")),
                side_image_path=_clean_path(row.get("side_image_path")),
                measurements=measurements,
                capture_notes=(row.get("capture_notes") or "").strip() or None,
            )
            validate_participant_session(session)
            sessions.append(session)

    return sessions


def parse_api_response_to_predictions(
    api_response: Dict[str, Any],
    participant_id: str,
    session_id: str,
    model_version: str = "model1_v0.1.0",
) -> List[ModelPredictionRecord]:
    """Parse real FastAPI / MeasurementPipelineService response directly into evaluator records.

    Preserves exact prediction statuses (e.g. SUCCESS, PARTIAL, UNCALIBRATED, ANATOMICAL_LEVEL_UNAVAILABLE)
    and warnings without transforming failed or missing measurements into 0.0.

    Parameters
    ----------
    api_response:
        Dictionary returned by the API or service.process_pair().
    participant_id:
        Pseudonymous participant identifier.
    session_id:
        Unique session / trial identifier.
    model_version:
        Model pipeline version identifier.

    Returns
    -------
    List[ModelPredictionRecord]
        List of prediction records ready for run_evaluation().
    """
    records: List[ModelPredictionRecord] = []
    measurements_dict = api_response.get("measurements", {})

    for level_key in ["chest", "waist", "abdomen", "hips"]:
        m_data = measurements_dict.get(level_key)
        if m_data is None:
            records.append(
                ModelPredictionRecord(
                    participant_id=participant_id,
                    session_id=session_id,
                    measurement_name=level_key,
                    predicted_cm=None,
                    prediction_status="MISSING_LEVEL",
                    warnings=["Measurement level was not returned in API output."],
                    model_version=model_version,
                )
            )
            continue

        raw_cm = m_data.get("circumference_cm")
        status_val = m_data.get("status", "UNKNOWN")
        warnings_list = list(m_data.get("warnings", []))

        records.append(
            ModelPredictionRecord(
                participant_id=participant_id,
                session_id=session_id,
                measurement_name=level_key,
                predicted_cm=float(raw_cm) if raw_cm is not None else None,
                prediction_status=status_val,
                warnings=warnings_list,
                model_version=model_version,
            )
        )

    return records


def save_predictions_json(predictions: List[ModelPredictionRecord], output_path: str | Path) -> None:
    """Save model predictions to a reproducible machine-readable JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [p.to_dict() for p in predictions]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_predictions_json(input_path: str | Path) -> List[ModelPredictionRecord]:
    """Reload model predictions from a machine-readable JSON file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Invalid predictions file: expected a JSON list of prediction objects.")

    predictions: List[ModelPredictionRecord] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for item in data:
        p_id = item.get("participant_id", "").strip()
        s_id = item.get("session_id", "").strip()
        m_name = item.get("measurement_name", "").strip().lower()

        if not p_id or not s_id or not m_name:
            raise ValueError(f"Prediction item missing required keys: {item}")

        key = (p_id, s_id, m_name)
        if key in seen_keys:
            raise ValueError(f"Duplicate prediction record encountered in file for key: {key}")
        seen_keys.add(key)

        predictions.append(
            ModelPredictionRecord(
                participant_id=p_id,
                session_id=s_id,
                measurement_name=m_name,
                predicted_cm=float(item["predicted_cm"]) if item.get("predicted_cm") is not None else None,
                prediction_status=item.get("prediction_status", "UNKNOWN"),
                warnings=list(item.get("warnings", [])),
                model_version=item.get("model_version", "model1_v0.1.0"),
            )
        )

    return predictions


