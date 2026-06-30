"""Prefect deployments for fraud detection."""

from prefect import serve
from orchestration import training_flow

if __name__ == "__main__":
    # Define a deployment that runs daily at midnight
    fraud_daily_deployment = training_flow.to_deployment(
        name="fraud-daily-training",
        cron="0 0 * * *",
        tags=["fraud", "daily"],
        description="Daily training pipeline for fraud detection models.",
        version="1.0",
    )

    # Serve the deployment
    serve(fraud_daily_deployment)
