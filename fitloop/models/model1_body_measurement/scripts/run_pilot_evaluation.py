"""
scripts/run_pilot_evaluation.py
-------------------------------
Reproducible pilot runner for Model 1 evaluation.

Workflow:
1. Loads ground-truth session records from a pilot CSV (or generates a sample template).
2. For sessions with image paths, executes inference via MeasurementPipelineService
   (or parses provided API output JSONs).
3. Directly transforms API outputs into ModelPredictionRecords without converting
   failures/missing slices to 0.0 or altering PARTIAL status semantics.
4. Executes run_evaluation() to match ground-truth references with model predictions
   and calculates per-measurement error and coverage statistics.
5. Emits structured JSON summary and markdown report.

Usage:
    PYTHONPATH=src /opt/anaconda3/envs/ml_env/bin/python scripts/run_pilot_evaluation.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import List, Optional

# Add project src to sys.path
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fitloop_measurement.evaluation import (
    EvaluationReport,
    ModelPredictionRecord,
    ParticipantSessionRecord,
    load_ground_truth_csv,
    load_predictions_json,
    parse_api_response_to_predictions,
    run_evaluation,
    save_predictions_json,
    write_pilot_csv_template,
)
from fitloop_measurement.service import MeasurementPipelineService, decode_image_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pilot_runner")


def run_pilot(
    gt_csv_path: Path,
    output_report_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    save_predictions_path: Optional[Path] = None,
    import_predictions_path: Optional[Path] = None,
    model_version: str = "model1_v0.1.0",
) -> EvaluationReport:
    """Execute reproducible pilot evaluation from a ground-truth CSV."""
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found at: {gt_csv_path}")

    logger.info("Loading ground-truth records from %s", gt_csv_path)
    sessions: List[ParticipantSessionRecord] = load_ground_truth_csv(gt_csv_path)
    logger.info("Loaded %d validated participant session(s).", len(sessions))

    predictions: List[ModelPredictionRecord] = []

    if import_predictions_path:
        logger.info("Importing pre-computed predictions from %s", import_predictions_path)
        predictions = load_predictions_json(import_predictions_path)
        logger.info("Loaded %d prediction record(s) from disk.", len(predictions))
    else:
        with MeasurementPipelineService() as pipeline:
            for idx, session in enumerate(sessions, start=1):
                p_id = session.participant_id
                s_id = session.session_id
                logger.info("[%d/%d] Running inference on session %s / %s...", idx, len(sessions), p_id, s_id)

                if not session.front_image_path or not session.side_image_path:
                    logger.warning(
                        "Session %s / %s is missing image path(s). Recording missing predictions.",
                        p_id,
                        s_id,
                    )
                    for dim in ["chest", "waist", "abdomen", "hips"]:
                        predictions.append(
                            ModelPredictionRecord(
                                participant_id=p_id,
                                session_id=s_id,
                                measurement_name=dim,
                                predicted_cm=None,
                                prediction_status="MISSING_INPUT_IMAGES",
                                warnings=["Image files not provided in ground-truth record."],
                                model_version=model_version,
                            )
                        )
                    continue

                front_path = Path(session.front_image_path)
                side_path = Path(session.side_image_path)

                if not front_path.exists() or not side_path.exists():
                    logger.error(
                        "One or both image files do not exist: front=%s, side=%s",
                        front_path,
                        side_path,
                    )
                    for dim in ["chest", "waist", "abdomen", "hips"]:
                        predictions.append(
                            ModelPredictionRecord(
                                participant_id=p_id,
                                session_id=s_id,
                                measurement_name=dim,
                                predicted_cm=None,
                                prediction_status="IMAGE_FILE_NOT_FOUND",
                                warnings=[f"Image missing on disk: front={front_path.exists()}, side={side_path.exists()}"],
                                model_version=model_version,
                            )
                        )
                    continue

                try:
                    with open(front_path, "rb") as f_front, open(side_path, "rb") as f_side:
                        front_bgr = decode_image_bytes(f_front.read(), "front")
                        side_bgr = decode_image_bytes(f_side.read(), "side")

                    # Real model pipeline execution (no changes to formulas or parameters)
                    api_dict = pipeline.process_pair(
                        front_image=front_bgr,
                        side_image=side_bgr,
                        standing_height_cm=session.standing_height_cm,
                    )

                    # Direct conversion into evaluator schemas without zero-substitution
                    session_preds = parse_api_response_to_predictions(
                        api_response=api_dict,
                        participant_id=p_id,
                        session_id=s_id,
                        model_version=model_version,
                    )
                    predictions.extend(session_preds)

                except Exception as e:
                    logger.exception("Pipeline failure on session %s / %s: %s", p_id, s_id, e)
                    for dim in ["chest", "waist", "abdomen", "hips"]:
                        predictions.append(
                            ModelPredictionRecord(
                                participant_id=p_id,
                                session_id=s_id,
                                measurement_name=dim,
                                predicted_cm=None,
                                prediction_status="PIPELINE_ERROR",
                                warnings=[str(e)],
                                model_version=model_version,
                            )
                        )

        if save_predictions_path:
            save_predictions_json(predictions, save_predictions_path)
            logger.info("Saved %d predictions to %s", len(predictions), save_predictions_path)

    logger.info("Running evaluation matching and metric calculations...")
    report: EvaluationReport = run_evaluation(
        reference_sessions=sessions,
        predictions=predictions,
        model_version=model_version,
    )

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved evaluation JSON report to %s", output_report_path)

    if output_csv_path:
        import csv
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = report.to_csv_rows()
        if rows:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Saved evaluation CSV summary to %s", output_csv_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="FitLoop Model 1 Pilot Evaluation Runner")
    parser.add_argument("--gt-csv", type=Path, help="Path to ground-truth CSV table")
    parser.add_argument("--output-json", type=Path, default=Path("pilot_evaluation_report.json"), help="Output JSON path")
    parser.add_argument("--output-csv", type=Path, help="Output summary CSV path")
    parser.add_argument("--save-predictions", type=Path, help="Export raw model predictions to JSON for re-evaluation")
    parser.add_argument("--import-predictions", type=Path, help="Import pre-computed predictions JSON instead of re-running inference")
    parser.add_argument("--generate-template", type=Path, help="Generate an empty pilot CSV template at the specified path")
    parser.add_argument("--model-version", type=str, default="model1_v0.1.0", help="Model version tag")

    args = parser.parse_args()

    if args.generate_template:
        write_pilot_csv_template(args.generate_template)
        print(f"Generated clean pilot CSV template at: {args.generate_template}")
        return

    if not args.gt_csv:
        parser.error("Either --gt-csv or --generate-template is required.")

    report = run_pilot(
        gt_csv_path=args.gt_csv,
        output_report_path=args.output_json,
        output_csv_path=args.output_csv,
        save_predictions_path=args.save_predictions,
        import_predictions_path=args.import_predictions,
        model_version=args.model_version,
    )

    report_data = report.to_dict()
    print("\n" + "=" * 80)
    print(f"PILOT EVALUATION SUMMARY (Model Version: {report_data['model_version']})")
    print("=" * 80)
    print(f"Total comparisons: {report_data['total_comparisons']}")
    print(f"Unmatched references: {report_data['unmatched_references_count']}")
    print(f"Unmatched predictions: {report_data['unmatched_predictions_count']}")
    print("-" * 80)
    print("Per-Measurement Metrics (Denominators clearly identified):")
    print(f"{'Site':<10} {'Valid/Total':<14} {'Coverage':<10} {'MAE (cm)':<10} {'RMSE (cm)':<10} {'Bias (cm)':<10} {'<=1cm':<10} {'<=2cm':<10}")
    print("-" * 80)
    for dim, m in report_data["per_measurement_metrics"].items():
        counts = f"{m['valid_predictions_count']}/{m['total_samples']}"
        cov = f"{m['coverage_rate']:.1%}"
        mae = f"{m['mae_cm']:.2f}" if m['mae_cm'] is not None else "N/A"
        rmse = f"{m['rmse_cm']:.2f}" if m['rmse_cm'] is not None else "N/A"
        bias = f"{m['signed_bias_cm']:.2f}" if m['signed_bias_cm'] is not None else "N/A"
        p1 = f"{m['pct_within_1cm_valid']:.1f}%" if m['pct_within_1cm_valid'] is not None else "N/A"
        p2 = f"{m['pct_within_2cm_valid']:.1f}%" if m['pct_within_2cm_valid'] is not None else "N/A"
        print(f"{dim.upper():<10} {counts:<14} {cov:<10} {mae:<10} {rmse:<10} {bias:<10} {p1:<10} {p2:<10}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
