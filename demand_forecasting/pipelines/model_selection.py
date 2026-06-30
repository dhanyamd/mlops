"""Model selection and champion-challenger lifecycle management for demand forecasting.

Why automated model selection:
  In production, data distributions shift and the best model type changes.
  A static model selection choice leads to decay. An automated model selection
  harness trains multiple architectures (LightGBM, XGBoost, Prophet), compares
  them on a validation set using business metrics, and handles deployment/registration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from demand_forecasting.pipelines.model_zoo import (
    BaseForecaster,
    LightGBMForecaster,
    ProphetForecaster,
    XGBoostForecaster,
)
from shared.mlflow_utils import promote_model, setup_mlflow
from shared.observability.logging import get_logger

log = get_logger(__name__)


class ModelSelector:
    """Trains, tunes, compares, and registers the best model in MLflow."""

    def __init__(
        self,
        experiment_name: str = "demand_forecasting",
        champion_alias: str = "champion",
        challenger_alias: str = "challenger",
    ):
        self.experiment_name = experiment_name
        self.champion_alias = champion_alias
        self.challenger_alias = challenger_alias
        setup_mlflow()

    def select_best_model(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        model_name: str = "demand_forecast_model",
    ) -> tuple[BaseForecaster, dict[str, float]]:
        """Fit candidates, log results to MLflow, and determine the champion.

        Enters a parent run, executes nested/child runs for each candidate,
        then promotes the winner to champion and the runner-up to challenger.
        """
        mlflow.set_experiment(self.experiment_name)

        # Candidates definition
        candidates = [
            XGBoostForecaster(n_estimators=150, max_depth=6, learning_rate=0.08),
            LightGBMForecaster(n_estimators=150, max_depth=6, learning_rate=0.08),
            ProphetForecaster(),
        ]

        best_metric = float("inf")  # We'll use RMSE as the primary decision metric
        best_model = None
        best_metrics = {}

        candidate_results = []

        # Parent run for this selection cycle
        with mlflow.start_run(run_name="model_selection_harness") as parent_run:
            log.info("model_selection_started", parent_run_id=parent_run.info.run_id)

            for forecaster in candidates:
                # Child run for each model
                with mlflow.start_run(run_name=f"candidate_{forecaster.name}", nested=True) as child_run:
                    log.info("training_candidate", model_name=forecaster.name)

                    try:
                        # Fit model
                        forecaster.fit(train_df, feature_cols, target_col)

                        # Evaluate model
                        eval_metrics = forecaster.evaluate(test_df, feature_cols, target_col)
                        log.info(
                            "candidate_evaluated",
                            model_name=forecaster.name,
                            rmse=eval_metrics["rmse"],
                            mae=eval_metrics["mae"],
                            mape=eval_metrics["mape"],
                        )

                        # Log parameters and metrics to MLflow
                        mlflow.log_param("model_type", forecaster.name)
                        mlflow.log_params({
                            "n_features": len(feature_cols),
                            "train_rows": len(train_df),
                            "test_rows": len(test_df),
                        })
                        mlflow.log_metrics(eval_metrics)

                        # Log the model artifact
                        if forecaster.name in ["XGBoost", "LightGBM"]:
                            mlflow.xgboost.log_model(
                                forecaster.model,
                                artifact_path="model",
                            )
                        else:
                            # Prophet/fallback models logged as generic python function
                            mlflow.pyfunc.log_model(
                                artifact_path="model",
                                python_model=forecaster,
                            )

                        candidate_results.append({
                            "forecaster": forecaster,
                            "metrics": eval_metrics,
                            "run_id": child_run.info.run_id,
                        })

                    except Exception as exc:
                        log.error("candidate_failed", model_name=forecaster.name, error=str(exc))
                        continue

            # Check if we have at least one successful model
            if not candidate_results:
                raise RuntimeError("All candidate models failed training.")

            # Sort candidates by RMSE (ascending)
            candidate_results.sort(key=lambda x: x["metrics"]["rmse"])

            # Determine winner and runner-up
            winner = candidate_results[0]
            best_model = winner["forecaster"]
            best_metrics = winner["metrics"]

            log.info(
                "model_selection_winner",
                model_name=best_model.name,
                run_id=winner["run_id"],
                rmse=best_metrics["rmse"],
            )

            # Register winner in model registry
            winner_uri = f"runs:/{winner['run_id']}/model"
            winner_reg = mlflow.register_model(winner_uri, model_name)

            # Check if there is an existing champion
            client = mlflow.tracking.MlflowClient()
            has_existing_champion = False
            try:
                # Look up versions with champion alias
                champ_version = client.get_model_version_by_alias(model_name, self.champion_alias)
                has_existing_champion = True
            except mlflow.exceptions.MlflowException:
                log.info("no_existing_champion_found", model_name=model_name)

            if has_existing_champion:
                # Evaluate champion on the holdout window to perform a champion-challenger promotion check
                log.info("champion_challenger_eval", champion_version=champ_version.version)
                # Load existing champion
                champ_model = mlflow.pyfunc.load_model(f"models:/{model_name}@{self.champion_alias}")

                # Compute champion RMSE
                if best_model.name in ["XGBoost", "LightGBM"]:
                    champ_preds = champ_model.predict(test_df[feature_cols])
                else:
                    champ_preds = champ_model.predict(test_df)

                y_true = test_df[target_col].values
                champ_rmse = float(np.sqrt(mean_squared_error(y_true, champ_preds)))
                log.info("champion_rmse", rmse=champ_rmse)

                # Promotion logic: challenger must beat champion by > 5% relative RMSE reduction
                improvement = (champ_rmse - best_metrics["rmse"]) / max(champ_rmse, 1e-5)
                if improvement > 0.05:
                    log.info(
                        "promoting_challenger_to_champion",
                        improvement_pct=round(improvement * 100, 2),
                        new_version=winner_reg.version,
                    )
                    # Set the alias champion to the new model
                    client.set_registered_model_alias(model_name, self.champion_alias, winner_reg.version)
                    # Set the old champion to challenger
                    client.set_registered_model_alias(model_name, self.challenger_alias, champ_version.version)
                else:
                    log.info(
                        "champion_retains_crown",
                        improvement_pct=round(improvement * 100, 2),
                        champion_version=champ_version.version,
                        challenger_version=winner_reg.version,
                    )
                    # Winner becomes the official challenger
                    client.set_registered_model_alias(model_name, self.challenger_alias, winner_reg.version)
            else:
                # First model trained, set as champion directly
                log.info("initial_champion_registered", version=winner_reg.version)
                client.set_registered_model_alias(model_name, self.champion_alias, winner_reg.version)

                # Register the runner-up as challenger if we have one
                if len(candidate_results) > 1:
                    runner_up = candidate_results[1]
                    runner_up_uri = f"runs:/{runner_up['run_id']}/model"
                    runner_up_reg = mlflow.register_model(runner_up_uri, model_name)
                    client.set_registered_model_alias(model_name, self.challenger_alias, runner_up_reg.version)
                    log.info("initial_challenger_registered", version=runner_up_reg.version)

            # Generate and log SHAP and Model Card explainability artifacts
            try:
                from shared.monitoring.explainability import SHAPExplainer
                from shared.monitoring.model_card import ModelCardGenerator

                # If best_model has an underlying tree model, use it for SHAP
                if winner["forecaster"].name in ["XGBoost", "LightGBM"]:
                    explainer = SHAPExplainer(winner["forecaster"].model, train_df[feature_cols])
                    shap_plot_path = "artifacts/demand_forecasting/shap_summary.png"
                    explainer.generate_summary_plot(test_df[feature_cols], shap_plot_path)
                    mlflow.log_artifact(shap_plot_path)
                    top_features = list(explainer.explain_prediction(test_df[feature_cols].head(1)).keys())
                else:
                    top_features = ["date"] # Prophet is date-based

                card_gen = ModelCardGenerator("demand_forecasting")
                card_metadata = {
                    "model_name": f"Demand Forecasting {winner['forecaster'].name} Model",
                    "model_version": f"v{winner_reg.version}",
                    "model_type": "Regression / Time-Series",
                    "framework": winner["forecaster"].name,
                    "intended_use": "Aggregate store-item demand forecasting.",
                    "out_of_scope": "Individual user purchase recommendations, real-time transaction scoring.",
                    "training_data_source": "ClickHouse demand.sales",
                    "dataset_size": len(train_df) + len(test_df),
                    "features": feature_cols,
                    "metrics": winner["metrics"],
                    "top_features": top_features,
                }
                card_path = card_gen.generate_card(card_metadata)
                mlflow.log_artifact(card_path)
            except Exception as e:
                log.warning("failed_to_generate_explainability_artifacts", error=str(e))

            # Log champion selection metadata to parent run
            mlflow.log_params({
                "champion_model_type": best_model.name,
                "champion_run_id": winner["run_id"],
                "champion_rmse": best_metrics["rmse"],
            })

        return best_model, best_metrics
