"""Backfill aggregation is hermetic: exact Flink TUMBLE(1H) OHLCV semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.backfill_feature_windows import hourly_windows

_HOUR = 3_600_000
_START = 1_800_000_000_000


def _bars(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_ohlcv_aggregation_matches_first_max_min_last_sum() -> None:
    bars = _bars(
        [
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START,
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "volume": 1.0,
            },
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START + 60_000,
                "open": 10.5,
                "high": 13.0,
                "low": 10.0,
                "close": 12.0,
                "volume": 2.0,
            },
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START + 120_000,
                "open": 12.0,
                "high": 12.5,
                "low": 11.5,
                "close": 11.5,
                "volume": 3.0,
            },
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START + _HOUR,
                "open": 99.0,
                "high": 99.0,
                "low": 99.0,
                "close": 99.0,
                "volume": 9.0,
            },
        ]
    )
    out = hourly_windows(bars)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["symbol"] == "BTCUSDT"
    assert row["window_start_ms"] == _START
    assert row["window_end_ms"] == _START + _HOUR
    assert row["open"] == 10.0
    assert row["high"] == 13.0
    assert row["low"] == 9.5
    assert row["close"] == 11.5
    assert row["volume"] == 6.0
    assert row["bar_count"] == 3


def test_vwap_is_volume_weighted() -> None:
    bars = _bars(
        [
            {
                "symbol": "ETHUSDT",
                "ts_ms": _START,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 100.0,
                "volume": 1.0,
            },
            {
                "symbol": "ETHUSDT",
                "ts_ms": _START + 60_000,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 200.0,
                "volume": 3.0,
            },
        ]
    )
    row = hourly_windows(bars).iloc[0]
    assert row["vwap"] == pytest.approx((100.0 * 1.0 + 200.0 * 3.0) / 4.0)


def test_in_progress_hour_is_dropped() -> None:
    # The latest hour bucket only has 2 of 60 bars — Flink would not have emitted
    # it yet, so the backfill must not either.
    bars = _bars(
        [
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 1.0,
            },
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START + 60_000,
                "open": 10.0,
                "high": 12.0,
                "low": 9.0,
                "close": 11.0,
                "volume": 2.0,
            },
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START + _HOUR,
                "open": 99.0,
                "high": 99.0,
                "low": 99.0,
                "close": 99.0,
                "volume": 9.0,
            },
        ]
    )
    assert len(hourly_windows(bars)) == 1  # the _START hour survives, the partial one does not


def test_empty_input_returns_empty_schema() -> None:
    out = hourly_windows(_bars([]))
    assert out.empty
    assert list(out.columns) == [
        "symbol",
        "window_start_ms",
        "window_end_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "bar_count",
    ]


def test_multiple_symbols_stay_separate() -> None:
    bars = _bars(
        [
            {
                "symbol": "BTCUSDT",
                "ts_ms": _START,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            },
            {
                "symbol": "ETHUSDT",
                "ts_ms": _START,
                "open": 2.0,
                "high": 2.0,
                "low": 2.0,
                "close": 2.0,
                "volume": 1.0,
            },
        ]
    )
    out = hourly_windows(bars)
    assert set(out["symbol"]) == {"BTCUSDT", "ETHUSDT"}
