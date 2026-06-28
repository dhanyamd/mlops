"""Unit tests — pipeline logic with inline fixtures (no Docker required)."""

from pathlib import Path

import pandas as pd
import pytest

from demand_forecasting.pipelines.feature_engineering import (
    FeatureEngineeringConfig,
    FeatureEngineeringPipeline,
)
from demand_forecasting.pipelines.preprocessing import PreprocessingConfig, PreprocessingPipeline
from shared.monitoring.drift import detect_univariate_drift


@pytest.fixture
def sample_unified(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for d in dates:
        for store in range(1, 3):
            for item in range(1, 4):
                rows.append(
                    {
                        "date": d,
                        "store_id": store,
                        "item_id": item,
                        "product_id": f"S{store:02d}_I{item:02d}",
                        "quantity_sold": 10 + store + item,
                        "revenue": 100.0,
                        "category": "grocery",
                        "unit_price": 10.0,
                        "cost": 5.0,
                        "is_holiday": False,
                        "has_promotion": 0,
                        "discount_pct": 0.0,
                        "channel": "web",
                    }
                )
    path = tmp_path / "unified.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_preprocessing(sample_unified: Path, tmp_path: Path) -> None:
    out = tmp_path / "cleaned.parquet"
    result = PreprocessingPipeline(
        PreprocessingConfig(input_path=sample_unified, output_path=out)
    ).run()
    assert len(result) > 0
    assert result["quantity_sold"].isna().sum() == 0


def test_feature_engineering(sample_unified: Path, tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned.parquet"
    PreprocessingPipeline(
        PreprocessingConfig(input_path=sample_unified, output_path=cleaned)
    ).run()
    out = tmp_path / "features.parquet"
    result = FeatureEngineeringPipeline(
        FeatureEngineeringConfig(input_path=cleaned, output_path=out)
    ).run()
    assert "lag_7" in result.columns
    assert len(result) > 0


def test_drift_detection() -> None:
    ref = pd.Series([1, 2, 3, 4, 5], name="feature_a")
    cur = pd.Series([10, 11, 12, 13, 14], name="feature_a")
    report = detect_univariate_drift(ref, cur, threshold=0.05)
    assert bool(report.drifted) is True
