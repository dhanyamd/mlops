"""Fraud detection orchestration."""

from __future__ import annotations

from prefect import flow, task

from fraud_detection.data.ingest import main as ingest_fraud
from fraud_detection.pipelines.training import FraudTrainingConfig, FraudTrainingPipeline


@task(name="ingest-fraud-clickhouse-qdrant")
def task_ingest() -> dict:
    return ingest_fraud()


@task(name="train-fraud-xgboost")
def task_train() -> dict:
    return FraudTrainingPipeline(FraudTrainingConfig()).run()


@flow(name="fraud-detection-training", log_prints=True)
def training_flow() -> dict:
    stats = task_ingest()
    metrics = task_train()
    return {"ingest": stats, "metrics": metrics}


if __name__ == "__main__":
    training_flow()
