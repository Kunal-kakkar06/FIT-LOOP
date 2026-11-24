"""
tests/test_evaluation.py
------------------------
Comprehensive tests for Stage 6: Ground-Truth Evaluation Framework.

Covers:
1. Valid manual measurement creation with repeated reading tolerance.
2. Participant session validation and bounds checking.
3. Strict participant-level partitioning (zero participant leakage).
4. Error metrics calculations (MAE, RMSE, signed bias, median AE, 90th percentile).
5. Precision threshold percentages (±1 cm and ±2 cm) with transparent denominators.
6. Evaluation runner key matching (participant_id + session_id + measurement_name).
7. Detection of unmatched references and missing predictions.
8. Subgroup metric reporting (by clothing condition).
9. Handling empty datasets and all-failed predictions (no fabricated 0-error).
10. Deterministic JSON and CSV serialization.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitloop_measurement.evaluation import (
    ClothingCondition,
    DatasetValidationError,
    DerivationMethod,
    EvaluationReport,
    ManualMeasurementRecord,
    ModelPredictionRecord,
    ParticipantSessionRecord,
    calculate_metrics_for_records,
    create_manual_measurement,
    run_evaluation,
    split_dataset_by_participant,
    validate_participant_session,
)
from fitloop_measurement.evaluation.schemas import ComparisonRecord


# ---------------------------------------------------------------------------
# 1. Dataset & Manual Measurement Tests
# ---------------------------------------------------------------------------

class TestManualMeasurementDataset:
    def test_valid_two_readings_within_tolerance(self):
        m = create_manual_measurement("waist", [75.0, 75.2], max_tolerance_cm=0.5)
        assert m.reference_value_cm == 75.1
        assert m.inter_reading_diff_cm == pytest.approx(0.2)
        assert m.derivation_method == DerivationMethod.MEAN_DUAL

    def test_three_readings_averaged(self):
        m = create_manual_measurement("chest", [92.0, 93.0, 92.5], max_tolerance_cm=0.5)
        assert m.reference_value_cm == pytest.approx(92.5)
        assert m.derivation_method == DerivationMethod.MEAN_TRIPLE

    def test_fewer_than_two_readings_raises(self):
        with pytest.raises(DatasetValidationError, match="at least 2 repeated readings"):
            create_manual_measurement("hips", [95.0])

    def test_non_finite_reading_raises(self):
        with pytest.raises(DatasetValidationError, match="non-finite"):
            create_manual_measurement("waist", [75.0, float("nan")])

    def test_implausible_out_of_bounds_raises(self):
        with pytest.raises(DatasetValidationError, match="outside plausible range"):
            create_manual_measurement("waist", [10.0, 10.0])  # Impossible 10 cm adult waist

    def test_participant_session_validation(self):
        m_waist = create_manual_measurement("waist", [78.0, 78.2])
        session = ParticipantSessionRecord(
            participant_id="P001",
            session_id="S01",
            standing_height_cm=175.0,
            clothing_condition=ClothingCondition.FITTED,
            measurements={"waist": m_waist},
        )
        warnings = validate_participant_session(session)
        assert len(warnings) == 0

    def test_invalid_session_height_raises(self):
        session = ParticipantSessionRecord(
            participant_id="P001",
            session_id="S01",
            standing_height_cm=50.0,  # Invalid stature
            clothing_condition=ClothingCondition.FITTED,
            measurements={},
        )
        with pytest.raises(DatasetValidationError, match="outside valid human range"):
            validate_participant_session(session)


# ---------------------------------------------------------------------------
# 2. Partitioning & Leakage Prevention Tests
# ---------------------------------------------------------------------------

class TestDatasetPartitioning:
    def test_participant_level_split_zero_leakage(self):
        # 10 participants, 2 sessions each (total 20 sessions)
        records = []
        for i in range(10):
            pid = f"P{i:03d}"
            m = create_manual_measurement("waist", [70.0 + i, 70.2 + i])
            # Session 1: fitted, Session 2: standard
            records.append(
                ParticipantSessionRecord(
                    participant_id=pid,
                    session_id=f"{pid}_fitted",
                    standing_height_cm=170.0 + i,
                    clothing_condition=ClothingCondition.FITTED,
                    measurements={"waist": m},
                )
            )
            records.append(
                ParticipantSessionRecord(
                    participant_id=pid,
                    session_id=f"{pid}_standard",
                    standing_height_cm=170.0 + i,
                    clothing_condition=ClothingCondition.STANDARD_CASUAL,
                    measurements={"waist": m},
                )
            )

        calib, eval_set = split_dataset_by_participant(records, calibration_ratio=0.40, seed=123)

        calib_pids = set(r.participant_id for r in calib)
        eval_pids = set(r.participant_id for r in eval_set)

        # Ensure both splits exist
        assert len(calib_pids) == 4
        assert len(eval_pids) == 6

        # Zero participant leakage
        assert calib_pids.isdisjoint(eval_pids)
        assert len(calib) == 8
        assert len(eval_set) == 12


# ---------------------------------------------------------------------------
# 3. Metrics Calculation Tests
# ---------------------------------------------------------------------------

class TestMetricsEngine:
    def test_exact_metric_calculations_known_sample(self):
        # 5 samples with known errors:
        # Ref: 80, Pred: 81 -> error = +1.0
        # Ref: 80, Pred: 82 -> error = +2.0
        # Ref: 80, Pred: 79 -> error = -1.0
        # Ref: 80, Pred: 83 -> error = +3.0
        # Ref: 80, Pred: 80 -> error =  0.0
        # Absolute errors: [1.0, 2.0, 1.0, 3.0, 0.0]
        # MAE: (1 + 2 + 1 + 3 + 0) / 5 = 1.4 cm
        # Errors squared: [1, 4, 1, 9, 0] -> sum = 15 -> RMSE = sqrt(3) ≈ 1.732 cm
        # Bias: (1 + 2 - 1 + 3 + 0) / 5 = +1.0 cm
        # Median AE: sorted [0, 1, 1, 2, 3] -> median = 1.0 cm
        # <= 1.0 cm count: 3 of 5 (60.0%)
        # <= 2.0 cm count: 4 of 5 (80.0%)
        records = [
            ComparisonRecord("P1", "S1", "waist", 80.0, 81.0, True, "SUCCESS", ClothingCondition.FITTED, 175.0, 1.0, 1.0),
            ComparisonRecord("P2", "S1", "waist", 80.0, 82.0, True, "SUCCESS", ClothingCondition.FITTED, 175.0, 2.0, 2.0),
            ComparisonRecord("P3", "S1", "waist", 80.0, 79.0, True, "SUCCESS", ClothingCondition.FITTED, 175.0, -1.0, 1.0),
            ComparisonRecord("P4", "S1", "waist", 80.0, 83.0, True, "SUCCESS", ClothingCondition.FITTED, 175.0, 3.0, 3.0),
            ComparisonRecord("P5", "S1", "waist", 80.0, 80.0, True, "SUCCESS", ClothingCondition.FITTED, 175.0, 0.0, 0.0),
        ]

        summary = calculate_metrics_for_records("waist", records)
        assert summary.total_samples == 5
        assert summary.valid_predictions_count == 5
        assert summary.coverage_rate == 1.0
        assert pytest.approx(summary.mae_cm, 1e-3) == 1.400
        assert pytest.approx(summary.rmse_cm, 1e-3) == 1.732
        assert pytest.approx(summary.signed_bias_cm, 1e-3) == 1.000
        assert summary.median_ae_cm == 1.0
        assert summary.pct_within_1cm_valid == 60.0
        assert summary.pct_within_2cm_valid == 80.0

    def test_handles_failed_predictions_gracefully(self):
        # 3 total samples: 1 valid (error = +1.0), 2 failed
        records = [
            ComparisonRecord("P1", "S1", "waist", 80.0, 81.0, True, "SUCCESS", ClothingCondition.FITTED, 175.0, 1.0, 1.0),
            ComparisonRecord("P2", "S1", "waist", 80.0, None, False, "UNCALIBRATED", ClothingCondition.FITTED, 175.0, None, None),
            ComparisonRecord("P3", "S1", "waist", 80.0, None, False, "INVALID_INPUT", ClothingCondition.FITTED, 175.0, None, None),
        ]

        summary = calculate_metrics_for_records("waist", records)
        assert summary.total_samples == 3
        assert summary.valid_predictions_count == 1
        assert summary.failed_predictions_count == 2
        assert pytest.approx(summary.coverage_rate, 1e-2) == 0.33

        # Accurate MAE on valid predictions
        assert summary.mae_cm == 1.0
        # Denominator distinction:
        assert summary.pct_within_1cm_valid == 100.0  # 1 of 1 valid
        assert pytest.approx(summary.pct_within_1cm_total, 1e-1) == 33.3  # 1 of 3 total

    def test_all_failed_predictions_returns_unavailable_metrics(self):
        records = [
            ComparisonRecord("P1", "S1", "waist", 80.0, None, False, "UNCALIBRATED", ClothingCondition.FITTED, 175.0, None, None)
        ]
        summary = calculate_metrics_for_records("waist", records)
        assert summary.total_samples == 1
        assert summary.valid_predictions_count == 0
        assert summary.failed_predictions_count == 1
        assert summary.mae_cm is None
        assert summary.rmse_cm is None


# ---------------------------------------------------------------------------
# 4. Evaluation Runner Tests
# ---------------------------------------------------------------------------

class TestEvaluationRunner:
    def test_key_matching_and_unmatched_tracking(self):
        m_waist = create_manual_measurement("waist", [75.0, 75.0])
        m_chest = create_manual_measurement("chest", [90.0, 90.0])

        session = ParticipantSessionRecord(
            participant_id="P001",
            session_id="S01",
            standing_height_cm=175.0,
            clothing_condition=ClothingCondition.FITTED,
            measurements={"waist": m_waist, "chest": m_chest},
        )

        predictions = [
            # Matched waist
            ModelPredictionRecord("P001", "S01", "waist", 76.5, "SUCCESS"),
            # Unmatched prediction (nonexistent measurement "bicep")
            ModelPredictionRecord("P001", "S01", "bicep", 32.0, "SUCCESS"),
        ]

        report = run_evaluation([session], predictions)
        assert len(report.comparisons) == 2  # waist and chest
        # Waist matched
        c_waist = next(c for c in report.comparisons if c.measurement_name == "waist")
        assert c_waist.is_valid_prediction is True
        assert c_waist.error_cm == 1.5

        # Chest missing prediction
        c_chest = next(c for c in report.comparisons if c.measurement_name == "chest")
        assert c_chest.is_valid_prediction is False
        assert c_chest.prediction_status == "MISSING_PREDICTION"

        # Check unmatched lists
        assert "P001/S01/chest" in report.unmatched_references
        assert "P001/S01/bicep" in report.unmatched_predictions

    def test_json_and_csv_export(self):
        m = create_manual_measurement("waist", [75.0, 75.0])
        session = ParticipantSessionRecord(
            participant_id="P001",
            session_id="S01",
            standing_height_cm=175.0,
            clothing_condition=ClothingCondition.FITTED,
            measurements={"waist": m},
        )
        predictions = [ModelPredictionRecord("P001", "S01", "waist", 76.0, "SUCCESS")]

        report = run_evaluation([session], predictions)
        data = report.to_dict()
        assert "per_measurement_metrics" in data
        assert "waist" in data["per_measurement_metrics"]
        assert json.dumps(data)

        csv_rows = report.to_csv_rows()
        assert len(csv_rows) == 1
        assert csv_rows[0]["measurement_name"] == "waist"
        assert csv_rows[0]["mae_cm"] == 1.0


# ---------------------------------------------------------------------------
# 5. Pilot CSV Template & Ground Truth Loading Tests
# ---------------------------------------------------------------------------

class TestPilotCSVIO:
    def test_write_pilot_csv_template(self, tmp_path):
        from fitloop_measurement.evaluation.dataset import PILOT_CSV_HEADERS, write_pilot_csv_template

        csv_file = tmp_path / "pilot_template.csv"
        write_pilot_csv_template(csv_file)
        assert csv_file.exists()

        content = csv_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(content) == 1
        headers = content[0].split(",")
        assert headers == PILOT_CSV_HEADERS

    def test_load_ground_truth_csv_valid_and_repeated_readings(self, tmp_path):
        import csv
        from fitloop_measurement.evaluation.dataset import PILOT_CSV_HEADERS, load_ground_truth_csv

        csv_file = tmp_path / "pilot_data.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(PILOT_CSV_HEADERS)
            writer.writerow([
                "P001", "S01", "175.5", "fitted", "", "",
                "95.0", "95.2", "95.1",
                "80.0", "80.2", "",
                "85.0", "85.0", "",
                "98.0", "98.2", "98.1",
                "Clean pilot test record"
            ])

        records = load_ground_truth_csv(csv_file)
        assert len(records) == 1
        rec = records[0]
        assert rec.participant_id == "P001"
        assert rec.session_id == "S01"
        assert rec.standing_height_cm == 175.5
        assert rec.clothing_condition == ClothingCondition.FITTED
        assert "chest" in rec.measurements
        assert pytest.approx(rec.measurements["chest"].reference_value_cm, 1e-2) == 95.1
        assert "waist" in rec.measurements
        assert pytest.approx(rec.measurements["waist"].reference_value_cm, 1e-2) == 80.1
        assert "abdomen" in rec.measurements
        assert pytest.approx(rec.measurements["abdomen"].reference_value_cm, 1e-2) == 85.0
        assert "hips" in rec.measurements
        assert pytest.approx(rec.measurements["hips"].reference_value_cm, 1e-2) == 98.1

    def test_load_ground_truth_csv_single_reading_fails_protocol(self, tmp_path):
        import csv
        from fitloop_measurement.evaluation.dataset import PILOT_CSV_HEADERS, load_ground_truth_csv

        csv_file = tmp_path / "pilot_single.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(PILOT_CSV_HEADERS)
            writer.writerow([
                "P001", "S01", "175.5", "fitted", "", "",
                "95.0", "", "",  # Only 1 reading for chest
                "", "", "",
                "", "", "",
                "", "", "",
                ""
            ])

        with pytest.raises(DatasetValidationError, match="at least 2 repeated readings"):
            load_ground_truth_csv(csv_file)

    def test_parse_api_response_preserves_partial_and_failed_measurements(self):
        from fitloop_measurement.evaluation import parse_api_response_to_predictions

        # Realistic API response matching the actual output observed in Model 1
        sample_api_output = {
            "status": "PARTIAL",
            "calibration": {"is_valid": True},
            "measurements": {
                "chest": {
                    "level": "chest",
                    "circumference_cm": 103.2,
                    "status": "SUCCESS",
                    "warnings": [],
                },
                "waist": {
                    "level": "waist",
                    "circumference_cm": 110.55,
                    "status": "SUCCESS",
                    "warnings": [],
                },
                "abdomen": {
                    "level": "abdomen",
                    "circumference_cm": 111.63,
                    "status": "PARTIAL",
                    "warnings": ["[Side] Scanline at y=1300 contains 2 disconnected segments."],
                },
                "hips": {
                    "level": "hips",
                    "circumference_cm": None,
                    "status": "ANATOMICAL_LEVEL_UNAVAILABLE",
                    "warnings": ["Could not locate corresponding anatomical slice in both views."],
                },
            },
            "warnings": [],
            "is_mock": False,
        }

        preds = parse_api_response_to_predictions(
            sample_api_output,
            participant_id="P001",
            session_id="S01",
            model_version="model1_pilot_v1",
        )

        assert len(preds) == 4
        pred_dict = {p.measurement_name: p for p in preds}

        # Chest: SUCCESS with float value
        assert pred_dict["chest"].predicted_cm == 103.2
        assert pred_dict["chest"].prediction_status == "SUCCESS"
        assert len(pred_dict["chest"].warnings) == 0

        # Abdomen: PARTIAL with float value and warnings preserved
        assert pred_dict["abdomen"].predicted_cm == 111.63
        assert pred_dict["abdomen"].prediction_status == "PARTIAL"
        assert len(pred_dict["abdomen"].warnings) == 1
        assert "disconnected segments" in pred_dict["abdomen"].warnings[0]

        # Hips: Failed prediction remains None, NOT converted to 0.0
        assert pred_dict["hips"].predicted_cm is None
        assert pred_dict["hips"].prediction_status == "ANATOMICAL_LEVEL_UNAVAILABLE"
        assert len(pred_dict["hips"].warnings) == 1

    def test_prediction_serialization_and_reload(self, tmp_path):
        from fitloop_measurement.evaluation import load_predictions_json, save_predictions_json

        original_preds = [
            ModelPredictionRecord(
                participant_id="P001",
                session_id="S01",
                measurement_name="chest",
                predicted_cm=95.4,
                prediction_status="SUCCESS",
                warnings=[],
            ),
            ModelPredictionRecord(
                participant_id="P001",
                session_id="S01",
                measurement_name="waist",
                predicted_cm=82.1,
                prediction_status="PARTIAL",
                warnings=["Slice warning"],
            ),
            ModelPredictionRecord(
                participant_id="P001",
                session_id="S01",
                measurement_name="hips",
                predicted_cm=None,
                prediction_status="ANATOMICAL_LEVEL_UNAVAILABLE",
                warnings=["Missing"],
            ),
        ]

        save_path = tmp_path / "preds.json"
        save_predictions_json(original_preds, save_path)
        assert save_path.exists()

        reloaded = load_predictions_json(save_path)
        assert len(reloaded) == 3
        assert reloaded[0].participant_id == "P001"
        assert reloaded[0].predicted_cm == 95.4
        assert reloaded[0].prediction_status == "SUCCESS"
        assert reloaded[1].prediction_status == "PARTIAL"
        assert reloaded[1].warnings == ["Slice warning"]
        assert reloaded[2].predicted_cm is None
        assert reloaded[2].prediction_status == "ANATOMICAL_LEVEL_UNAVAILABLE"

    def test_duplicate_prediction_records_rejected(self):
        duplicate_preds = [
            ModelPredictionRecord(
                participant_id="P001",
                session_id="S01",
                measurement_name="waist",
                predicted_cm=80.0,
                prediction_status="SUCCESS",
            ),
            ModelPredictionRecord(
                participant_id="P001",
                session_id="S01",
                measurement_name="waist",  # Duplicate key
                predicted_cm=81.0,
                prediction_status="SUCCESS",
            ),
        ]

        session = ParticipantSessionRecord(
            participant_id="P001",
            session_id="S01",
            standing_height_cm=175.0,
            clothing_condition=ClothingCondition.FITTED,
            measurements={"waist": create_manual_measurement("waist", [80.0, 80.2])},
        )

        with pytest.raises(ValueError, match="Duplicate prediction record encountered"):
            run_evaluation([session], duplicate_preds)

    def test_reproducible_pilot_runner_with_imported_predictions(self, tmp_path):
        import csv
        import sys
        from pathlib import Path
        repo_root = str(Path(__file__).resolve().parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from fitloop_measurement.evaluation import PILOT_CSV_HEADERS, save_predictions_json
        from scripts.run_pilot_evaluation import run_pilot

        csv_file = tmp_path / "pilot_gt.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(PILOT_CSV_HEADERS)
            writer.writerow([
                "P002", "S01", "180.0", "standard", "", "",
                "100.0", "100.2", "",
                "85.0", "85.2", "",
                "90.0", "90.0", "",
                "105.0", "105.4", "",
                "Synthetic pilot test record"
            ])

        preds = [
            ModelPredictionRecord(
                participant_id="P002",
                session_id="S01",
                measurement_name="chest",
                predicted_cm=101.1,
                prediction_status="SUCCESS",
            ),
            ModelPredictionRecord(
                participant_id="P002",
                session_id="S01",
                measurement_name="waist",
                predicted_cm=85.6,
                prediction_status="PARTIAL",
                warnings=["Irregular contour"],
            ),
            ModelPredictionRecord(
                participant_id="P002",
                session_id="S01",
                measurement_name="abdomen",
                predicted_cm=None,
                prediction_status="ANATOMICAL_LEVEL_UNAVAILABLE",
            ),
            ModelPredictionRecord(
                participant_id="P002",
                session_id="S01",
                measurement_name="hips",
                predicted_cm=106.2,
                prediction_status="SUCCESS",
            ),
        ]
        preds_path = tmp_path / "imported_preds.json"
        save_predictions_json(preds, preds_path)

        out_json = tmp_path / "report.json"
        out_csv = tmp_path / "summary.csv"

        report = run_pilot(
            gt_csv_path=csv_file,
            output_report_path=out_json,
            output_csv_path=out_csv,
            import_predictions_path=preds_path,
        )

        assert out_json.exists()
        assert out_csv.exists()

        rep_dict = report.to_dict()
        assert rep_dict["total_comparisons"] == 4
        assert rep_dict["unmatched_references_count"] == 0
        assert rep_dict["unmatched_predictions_count"] == 0

        # Abdomen had null prediction -> coverage should be 0, failed=1, valid=0
        ab_m = rep_dict["per_measurement_metrics"]["abdomen"]
        assert ab_m["valid_predictions_count"] == 0
        assert ab_m["failed_predictions_count"] == 1
        assert ab_m["coverage_rate"] == 0.0
        assert ab_m["mae_cm"] is None

        # Chest had valid prediction: ref = 100.1, pred = 101.1 -> abs error = 1.0 cm
        ch_m = rep_dict["per_measurement_metrics"]["chest"]
        assert ch_m["valid_predictions_count"] == 1
        assert ch_m["coverage_rate"] == 1.0
        assert ch_m["mae_cm"] == pytest.approx(1.0, 1e-2)
        assert ch_m["pct_within_1cm_valid"] == 100.0



