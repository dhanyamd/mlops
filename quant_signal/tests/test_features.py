"""Feature-engineering tests — hermetic (no Snowflake, no Spark, no network).

Covers ``flows.features.compute_features`` (the single source of truth for the
Spark batch and the pandas fallback): the rolling math, no-lookahead
guarantee, per-symbol isolation, null-before-warm-up behavior, and the
``ingest.store.write_features`` routing into the GOLD mart.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from config.settings import Settings
from db.snowflake import SnowflakeClient
from flows.features import FEATURE_COLUMNS, compute_features
from ingest.store import write_features


def _settings(**overrides) -> Settings:
    base = {
        "snowflake_account": "GULXCKK-PI01025",
        "snowflake_user": "devuser",
        "snowflake_password": "pw",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def _bars(symbols: list[str] = ["AAPL"], n: int = 30, base: float = 100.0) -> pd.DataFrame:
    """Deterministic ascending-price bars (each close = 1% above the previous)."""
    start = dt.datetime(2026, 1, 2, 14, 0)
    rows = []
    for sym in symbols:
        price = base
        for i in range(n):
            if i > 0:
                price = price * 1.01
            rows.append(
                {
                    "symbol": sym,
                    "ts": start + dt.timedelta(minutes=1 * i),
                    "timeframe": "1Min",
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 1000.0 + i,
                    "provider": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def test_empty_input_returns_empty() -> None:
    out = compute_features(pd.DataFrame(columns=["symbol", "ts", "close"]))
    assert out.empty


def test_returns_math_matches_definition() -> None:
    bars = _bars(n=5)
    out = compute_features(bars).set_index("ts").sort_index()
    closes = bars["close"].to_numpy()
    expected_log = np.log(closes[1:] / closes[:-1])
    expected_ret1 = closes[1:] / closes[:-1] - 1.0
    pdt.assert_series_equal(
        out["log_return"].iloc[1:].astype(float),
        pd.Series(expected_log, index=out.index[1:]),
        check_names=False,
    )
    pdt.assert_series_equal(
        out["ret_1"].iloc[1:].astype(float),
        pd.Series(expected_ret1, index=out.index[1:]),
        check_names=False,
    )
    # First row has no predecessor -> null (never fabricated).
    assert pd.isna(out["log_return"].iloc[0])
    assert pd.isna(out["ret_1"].iloc[0])


def test_features_are_null_until_warmup() -> None:
    out = compute_features(_bars(n=30))
    warm = out[out["symbol"] == "AAPL"].reset_index(drop=True)
    # vol_20 / zscore_20 need 5 non-NaN log returns (log_return[0] is NaN, so
    # the first valid std lands on row 5); mom_20 needs a full 20-bar window.
    assert warm["vol_20"].iloc[:5].isna().all()
    assert not pd.isna(warm["vol_20"].iloc[5])
    assert warm["mom_20"].iloc[:19].isna().all()
    assert not pd.isna(warm["mom_20"].iloc[20])


def test_features_are_computed_per_symbol_not_across() -> None:
    # Two symbols with the same prices: identical features even when interleaved.
    bars = _bars(symbols=["AAPL", "MSFT"], n=30)
    out = compute_features(bars)
    a = out[out["symbol"] == "AAPL"].reset_index(drop=True)
    b = out[out["symbol"] == "MSFT"].reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        pdt.assert_series_equal(a[col].astype(float), b[col].astype(float), check_names=False)


def test_no_lookahead_prefix_is_stable() -> None:
    # Features for the first N rows must not change when future bars are added.
    bars = _bars(n=30)
    full = compute_features(bars)
    prefix = compute_features(bars.iloc[:20])
    for col in FEATURE_COLUMNS:
        pdt.assert_series_equal(
            full[col].iloc[:20].astype(float),
            prefix[col].astype(float),
            check_names=False,
        )


def test_output_contract_columns() -> None:
    out = compute_features(_bars())
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    assert {"symbol", "ts", "close"} <= set(out.columns)


def test_write_features_routes_to_gold_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_upsert(self, df, table_name, merge_keys, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(table_name=table_name, merge_keys=merge_keys, schema=kwargs["schema"])
        return len(df)

    monkeypatch.setattr(SnowflakeClient, "upsert_df", fake_upsert)
    n = write_features(compute_features(_bars(n=8)), _settings())
    assert n == 8
    assert captured["table_name"] == "FEATURES"
    assert captured["merge_keys"] == ["symbol", "timeframe", "ts"]
    assert captured["schema"] == "GOLD"


def test_write_features_empty_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    def fake_upsert(self, df, table_name, merge_keys, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return 0

    monkeypatch.setattr(SnowflakeClient, "upsert_df", fake_upsert)
    assert write_features(pd.DataFrame(), _settings()) == 0
    assert calls == []
