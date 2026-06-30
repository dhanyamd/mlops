"""Fraud detection orchestration — Prefect flow with quality gates + drift detection.

Flow order:
  ingest → quality gate (raw data) → train → drift check → alert if drifted
"""

from __future__ import annotations

import pandas as pd
from prefect import flow, task

from fraud_detection.data.ingest import main as ingest_fraud
from fraud_detection.pipelines.training import FraudTrainingConfig, FraudTrainingPipeline
from shared.config import DATA_ROOT
from shared.data_quality.quality import DataQualityError, validate_fraud_raw
from shared.monitoring.drift import detect_dataset_drift, should_retrain
from shared.observability.logging import get_logger
from shared.observability.metrics import FEATURE_DRIFT_SCORE, PIPELINE_RUNS_TOTAL

log = get_logger(__name__)

RAW_KEY = "fraud_detection/offline/raw_transactions.parquet"
FEATURES_KEY = "fraud_detection/offline/transaction_features.parquet"


@task(name="ingest-fraud-clickhouse-qdrant")
def task_ingest() -> dict:
    return ingest_fraud()


@task(name="quality-gate-fraud-raw")
def task_quality_gate() -> bool:
    """Run data quality gate on raw transactions before training."""
    from shared.clients import S3Client
    s3 = S3Client()
    try:
        df = s3.read_df(RAW_KEY)
    except Exception as exc:
        log.warning("quality_gate_skip_no_s3_parquet", key=RAW_KEY, error=str(exc))
        return True

    result = validate_fraud_raw(df)
    if not result.passed:
        raise DataQualityError(
            f"Fraud training blocked — raw data quality failed:\n{result.summary}"
        )
    log.info("quality_gate_passed", checkpoint="fraud_raw_transactions", rows=len(df))
    return True


@task(name="train-fraud-xgboost")
def task_train() -> dict:
    return FraudTrainingPipeline(FraudTrainingConfig(raw_key=RAW_KEY)).run()


@task(name="drift-check-fraud")
def task_drift_check() -> dict:
    """Check feature drift between training data and most recent data."""
    from shared.clients import S3Client
    s3 = S3Client()
    try:
        df = s3.read_df(FEATURES_KEY)
    except Exception as exc:
        log.warning("drift_check_skip_no_s3_parquet", key=FEATURES_KEY, error=str(exc))
        return {"drifted": False, "reason": "no features file in S3"}

    # Split into reference (first 80%) and current (last 20%) — mirrors time split
    cutoff = int(len(df) * 0.8)
    reference = df.iloc[:cutoff]
    current = df.iloc[cutoff:]

    numeric_cols = ["amount", "txn_count_1h", "txn_count_24h", "velocity_ratio", "amount_deviation"]
    reports = detect_dataset_drift(reference, current, numeric_columns=numeric_cols)

    # Emit per-feature drift scores to Prometheus
    for r in reports:
        FEATURE_DRIFT_SCORE.labels(feature_name=r.feature, project="fraud").set(r.statistic)
        if r.drifted:
            log.warning("feature_drift_detected", feature=r.feature, ks_stat=r.statistic)

    # Generate Evidently HTML report
    try:
        from shared.monitoring.evidently_reports import DriftDashboardGenerator
        dashboard = DriftDashboardGenerator("fraud_detection")
        dashboard.generate_drift_report(
            reference=reference[numeric_cols],
            current=current[numeric_cols],
            target_column=None
        )
    except Exception as e:
        log.warning("failed_to_generate_evidently_drift_report", error=str(e))

    drifted = should_retrain(reports, min_drifted=2)
    return {
        "drifted": drifted,
        "drift_reports": [
            {"feature": r.feature, "statistic": round(r.statistic, 4), "drifted": r.drifted}
            for r in reports
        ],
    }


@flow(name="fraud-detection-training", log_prints=True)
def training_flow() -> dict:
    stats = task_ingest()
    task_quality_gate()
    metrics = task_train()
    drift = task_drift_check()

    PIPELINE_RUNS_TOTAL.labels(pipeline="fraud_training_flow", status="success").inc()

    return {"ingest": stats, "metrics": metrics, "drift": drift}


if __name__ == "__main__":
    training_flow()

