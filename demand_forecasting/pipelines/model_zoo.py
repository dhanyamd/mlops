"""Model Zoo for demand forecasting: BaseForecaster, Prophet, LightGBM, and XGBoost.

Why Multi-Model Forecasting:
  No single ML model fits all demand forecasting scenarios:
  - Prophet: Excellent for catching yearly/weekly seasonality, trend shifts,
    and holiday effects on aggregate/regional levels.
  - LightGBM: High-performance gradient boosting. Faster to train and consumes
    less memory than XGBoost, natively handles categorical features.
  - XGBoost: Classic robust regressor for tabular/time-series tabular features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from shared.observability.logging import get_logger

log = get_logger(__name__)

# Graceful import check for prophet (avoid pipeline crash if dependency is missing)
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    log.warning("prophet_not_installed", msg="Prophet library is not installed. ProphetForecaster will use a trend/seasonality baseline fallback.")


class BaseForecaster(ABC):
    """Abstract Base Class for all forecasters in the zoo."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> BaseForecaster:
        """Fit the forecasting model on the training dataframe."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Generate demand forecasts for the input dataframe."""
        pass

    def evaluate(self, df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict[str, float]:
        """Evaluate the model and return common regression metrics."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} must be fitted before evaluation.")

        preds = self.predict(df, feature_cols)
        y_true = df[target_col].values

        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float(r2_score(y_true, preds))
        mape = float(np.mean(np.abs((y_true - preds) / np.maximum(y_true, 1.0))))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        }


class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor forecaster wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, **kwargs):
        super().__init__("XGBoost")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> XGBoostForecaster:
        X = train_df[feature_cols]
        y = train_df[target_col]
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        X = df[feature_cols]
        preds = self.model.predict(X)
        return np.maximum(0, preds)  # Demand cannot be negative


class LightGBMForecaster(BaseForecaster):
    """LightGBM regressor forecaster wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, **kwargs):
        super().__init__("LightGBM")
        # Inline import to avoid dependency issues if lightgbm is missing
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **kwargs
        )

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> LightGBMForecaster:
        X = train_df[feature_cols]
        y = train_df[target_col]
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        X = df[feature_cols]
        preds = self.model.predict(X)
        return np.maximum(0, preds)


class ProphetForecaster(BaseForecaster, mlflow.pyfunc.PythonModel):
    """Prophet forecaster wrapper with fallback to a daily seasonal trend baseline."""

    def __init__(self, **kwargs):
        BaseForecaster.__init__(self, "Prophet")
        self.kwargs = kwargs
        self.model = None
        self.fallback_coefs_ = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> ProphetForecaster:
        if "date" not in train_df.columns:
            raise ValueError("Prophet requires a 'date' column in the input dataframe.")

        if PROPHET_AVAILABLE:
            # Prepare Prophet format dataframe: ds and y
            prophet_df = pd.DataFrame({
                "ds": pd.to_datetime(train_df["date"]),
                "y": train_df[target_col]
            })
            # Fit Prophet model
            self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, **self.kwargs)
            # Add US holidays if needed (simplified)
            self.model.add_country_holidays(country_name='US')
            self.model.fit(prophet_df)
        else:
            # Fallback trend + weekday seasonality model using numpy/pandas
            log.warning("prophet_fallback_fitting", msg="Using simple linear trend + weekday mean fallback.")
            df = train_df.copy()
            df["date_idx"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["date"]).min()).dt.days
            df["weekday"] = pd.to_datetime(df["date"]).dt.weekday

            # Linear trend
            X_trend = np.vstack([np.ones(len(df)), df["date_idx"]]).T
            y = df[target_col].values
            coefs, _, _, _ = np.linalg.lstsq(X_trend, y, rcond=None)
            self.fallback_coefs_ = coefs

            # Weekday deviations
            pred_trend = X_trend @ coefs
            residuals = y - pred_trend
            df["residual"] = residuals
            self.fallback_weekday_map_ = df.groupby("weekday")["residual"].mean().to_dict()

        self.is_fitted = True
        return self

    def predict(self, *args, **kwargs) -> np.ndarray:
        """Dual-use predict method supporting both:
          1. predict(df, feature_cols) -> Custom Zoo call
          2. predict(context, model_input) -> MLflow PyFunc call
        """
        if len(args) == 2:
            # Determine if args[0] is MLflow context
            # In MLflow, args[0] is mlflow.pyfunc.model.PythonModelContext
            if hasattr(args[0], "artifacts") or args[0] is None:
                # MLflow PyFunc: context, model_input
                df = args[1]
            else:
                # Custom Zoo: df, feature_cols
                df = args[0]
        elif len(args) == 1:
            df = args[0]
        else:
            df = kwargs.get("df", kwargs.get("model_input"))

        if df is None:
            raise ValueError("Input dataframe (df or model_input) must be provided to predict.")

        if "date" not in df.columns:
            raise ValueError("Prophet requires a 'date' column in the input dataframe.")

        if PROPHET_AVAILABLE and self.model is not None:
            prophet_df = pd.DataFrame({
                "ds": pd.to_datetime(df["date"])
            })
            forecast = self.model.predict(prophet_df)
            return np.maximum(0, forecast["yhat"].values)
        else:
            # Generate fallback predictions
            date_idx = (pd.to_datetime(df["date"]) - pd.to_datetime(df["date"]).min()).dt.days
            X_trend = np.vstack([np.ones(len(df)), date_idx]).T
            pred_trend = X_trend @ self.fallback_coefs_

            weekdays = pd.to_datetime(df["date"]).dt.weekday
            weekday_adjustments = np.array([self.fallback_weekday_map_.get(w, 0.0) for w in weekdays])

            preds = pred_trend + weekday_adjustments
            return np.maximum(0, preds)
