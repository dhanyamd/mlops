"""Prefect orchestration — demand forecasting production pipeline."""

from __future__ import annotations

from pathlib import Path

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

BASE = DATA_LAKE / "demand_forecasting"


@task(name="ingest-demand-data")
def task_ingest_demand() -> dict:
    return ingest_demand()


@task(name="etl-clickhouse-postgres")
def task_etl() -> Path:
    out = BASE / "processed" / "unified.parquet"
    ETLPipeline(ETLConfig(output_path=out)).run()
    return out


@task(name="preprocess")
def task_preprocess(unified: Path) -> Path:
    out = BASE / "processed" / "cleaned.parquet"
    PreprocessingPipeline(PreprocessingConfig(input_path=unified, output_path=out)).run()
    return out


@task(name="feature-engineering")
def task_features(cleaned: Path) -> Path:
    out = BASE / "features" / "features.parquet"
    FeatureEngineeringPipeline(FeatureEngineeringConfig(input_path=cleaned, output_path=out)).run()
    return out


@task(name="train-demand-model")
def task_train(features: Path) -> dict:
    return TrainingPipeline(TrainingConfig(input_path=features)).run()


@task(name="batch-inference-clickhouse")
def task_inference(features: Path) -> int:
    forecasts = InferencePipeline(InferenceConfig(features_path=features)).run()
    return len(forecasts)


@flow(name="demand-forecasting-pipeline", log_prints=True)
def demand_pipeline() -> dict:
    task_ingest_demand()
    unified = task_etl()
    cleaned = task_preprocess(unified)
    features = task_features(cleaned)
    metrics = task_train(features)
    n = task_inference(features)
    return {"metrics": metrics, "forecasts": n}


if __name__ == "__main__":
    demand_pipeline()
