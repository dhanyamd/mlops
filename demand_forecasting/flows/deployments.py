"""Prefect deployments for demand forecasting."""

from prefect import serve
from orchestration import demand_forecasting_flow

if __name__ == "__main__":
    # Define a deployment that runs weekly on Sunday at midnight
    demand_weekly_deployment = demand_forecasting_flow.to_deployment(
        name="demand-weekly-forecasting",
        cron="0 0 * * 0",
        tags=["demand", "weekly"],
        description="Weekly forecasting pipeline for demand models.",
        version="1.0",
    )

    # Serve the deployment
    serve(demand_weekly_deployment)
