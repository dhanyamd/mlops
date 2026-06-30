"""Prefect orchestration — demand forecasting production pipeline.

Flow order:
  ingest → ETL → quality gate → preprocess → features → quality gate → train → drift check → inference
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from prefect import flow, task

from demand_forecasting.data.ingest import main as ingest_demand
from demand_forecasting.etl.pipeline import ETLConfig, ETLPipeline
from demand_forecasting.pipelines.feature_engineering import (
    FeatureEngineeringConfig,
    FeatureEngineeringPipeline,
)
from demand_forecasting.pipelines.inference import InferenceConfig, InferencePipeline
from demand_forecasting.pipelines.preprocessing import PreprocessingConfig, PreprocessingPipeline
from demand_forecasting.pipelines.training import TrainingConfig, TrainingPipeline
from shared.config import DATA_LAKE
from shared.data_quality.quality import DataQualityError, validate_demand_features
from shared.monitoring.drift import detect_dataset_drift, should_retrain
from shared.observability.logging import get_logger
from shared.observability.metrics import FEATURE_DRIFT_SCORE, PIPELINE_RUNS_TOTAL

log = get_logger(__name__)

BASE = DATA_LAKE / "demand_forecasting"


@task(name="ingest-demand-data")
def task_ingest_demand() -> dict:
    return ingest_demand()


@task(name="etl-clickhouse-postgres")
def task_etl() -> str:
    out_key = "demand_forecasting/processed/unified.parquet"
    ETLPipeline(ETLConfig(output_key=out_key)).run()
    return out_key


@task(name="preprocess")
def task_preprocess(unified_key: str) -> str:
    out_key = "demand_forecasting/processed/cleaned.parquet"
    PreprocessingPipeline(PreprocessingConfig(input_key=unified_key, output_key=out_key)).run()
    return out_key


@task(name="feature-engineering")
def task_features(cleaned_key: str) -> str:
    out_key = "demand_forecasting/features/features.parquet"
    FeatureEngineeringPipeline(FeatureEngineeringConfig(input_key=cleaned_key, output_key=out_key)).run()
    return out_key


@task(name="quality-gate-demand-features")
def task_quality_gate(features_key: str) -> bool:
    """Validate engineered features before training."""
    from shared.clients import S3Client
    s3 = S3Client()
    df = s3.read_df(features_key)
    result = validate_demand_features(df)
    if not result.passed:
        raise DataQualityError(
            f"Demand training blocked — feature quality failed:\n{result.summary}"
        )
    log.info("quality_gate_passed", checkpoint="demand_features", rows=len(df))
    return True


@task(name="train-demand-model")
def task_train(features_key: str) -> dict:
    return TrainingPipeline(TrainingConfig(input_key=features_key)).run()


@task(name="drift-check-demand")
def task_drift_check(features_key: str) -> dict:
    """Check feature drift between training reference and recent data."""
    from shared.clients import S3Client
    s3 = S3Client()
    df = s3.read_df(features_key)
    cutoff = int(len(df) * 0.8)
    reference = df.iloc[:cutoff]
    current = df.iloc[cutoff:]

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in {"quantity_sold", "revenue"}]
    reports = detect_dataset_drift(reference, current, numeric_columns=numeric_cols[:10])

    for r in reports:
        FEATURE_DRIFT_SCORE.labels(feature_name=r.feature, project="demand").set(r.statistic)
        if r.drifted:
            log.warning("feature_drift_detected", feature=r.feature, ks_stat=r.statistic)

    # Generate Evidently HTML report
    try:
        from shared.monitoring.evidently_reports import DriftDashboardGenerator
        dashboard = DriftDashboardGenerator("demand_forecasting")
        dashboard.generate_drift_report(
            reference=reference[numeric_cols[:10]],
            current=current[numeric_cols[:10]],
            target_column=None
        )
    except Exception as e:
        log.warning("failed_to_generate_evidently_drift_report", error=str(e))

    drifted = should_retrain(reports, min_drifted=2)
    return {"drifted": drifted, "features_checked": len(reports)}


@task(name="batch-inference-clickhouse")
def task_inference(features_key: str) -> int:
    forecasts = InferencePipeline(InferenceConfig(features_key=features_key)).run()
    return len(forecasts)


@flow(name="demand-forecasting-pipeline", log_prints=True)
def demand_pipeline() -> dict:
    task_ingest_demand()
    unified_key = task_etl()
    cleaned_key = task_preprocess(unified_key)
    features_key = task_features(cleaned_key)
    task_quality_gate(features_key)
    metrics = task_train(features_key)
    drift = task_drift_check(features_key)
    n = task_inference(features_key)

    PIPELINE_RUNS_TOTAL.labels(pipeline="demand_forecasting_flow", status="success").inc()

    return {"metrics": metrics, "forecasts": n, "drift": drift}


if __name__ == "__main__":
    demand_pipeline()

