"""Dashboard API tests — hermetic (no Snowflake connection required).

Covers the two things the API adds on top of the already-tested PEAD math:
  1. JSON-safe serialization (NaN/NaT/datetime handling).
  2. Endpoint wiring + bound-parameter SQL against a fake client.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.db as db
from api.main import app


class _FakeClient:
    """Returns canned DataFrames and records the SQL/params it was called with."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.calls: list[tuple[str, tuple | None]] = []

    def query_df(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
        self.calls.append((sql, params))
        return self.df


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "_client", None)


# ── JSON-safe serialization ──────────────────────────────────────────────────


def test_records_converts_nan_nat_and_datetimes() -> None:
    df = pd.DataFrame(
        {
            "D": [pd.Timestamp("2024-01-02"), pd.NaT, None],
            "F": [1.5, float("nan"), 2.0],
            "S": ["a", "b", None],
        }
    )
    rows = db._records(df)
    assert rows == [
        {"D": "2024-01-02T00:00:00", "F": 1.5, "S": "a"},
        {"D": None, "F": None, "S": "b"},
        {"D": None, "F": 2.0, "S": None},
    ]


def test_market_bars_returns_oldest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "TRADE_DATE": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")],
            "DAY_CLOSE": [200.0, 190.0],
            "VOLUME": [100, 90],
        }
    )
    fake = _FakeClient(df)
    monkeypatch.setattr(db, "_get_client", lambda: fake)
    rows = db.market_bars("AAPL", days=2)
    assert [r["TRADE_DATE"] for r in rows] == ["2024-01-02T00:00:00", "2024-01-03T00:00:00"]
    sql, params = fake.calls[0]
    assert "GOLD_DAILY_BARS" in sql and params == ("AAPL", 2)


def test_fundamentals_binds_ticker_and_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(pd.DataFrame({"TICKER": ["AAPL"]}))
    monkeypatch.setattr(db, "_get_client", lambda: fake)
    db.fundamentals("aapl", metric="NetIncomeLoss")
    sql, params = fake.calls[0]
    assert "SILVER_COMPANY_FACTS" in sql
    assert "ORDER BY METRIC, FISCAL_YEAR, FILED_AT" in sql
    assert params == ("AAPL", "NetIncomeLoss")


def test_pipeline_metrics_optional_flow_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(pd.DataFrame())
    monkeypatch.setattr(db, "_get_client", lambda: fake)
    db.pipeline_metrics(flow="ingest-market-data", limit=50)
    _, params = fake.calls[0]
    assert params == ("ingest-market-data", 50)


def test_macro_series_optional_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(pd.DataFrame())
    monkeypatch.setattr(db, "_get_client", lambda: fake)
    db.macro_series(limit=10)
    _, params = fake.calls[0]
    assert params == (10,)


# ── Endpoints ────────────────────────────────────────────────────────────────


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_tickers_metrics_and_macro_series_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "default_tickers", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(db, "default_metrics", lambda: ["Revenues", "NetIncomeLoss"])
    monkeypatch.setattr(db, "default_macro_series", lambda: ["VIXCLS", "DGS10"])
    with TestClient(app) as client:
        assert client.get("/api/tickers").json()["tickers"] == ["AAPL", "MSFT"]
        assert client.get("/api/metrics").json()["metrics"] == ["Revenues", "NetIncomeLoss"]
        assert client.get("/api/macro/series").json()["series"] == ["VIXCLS", "DGS10"]


def test_pead_endpoint_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "api.main.compute_pead",
        lambda metric, windows, min_prior, quintiles: {
            "metric": metric,
            "windows": windows,
            "n_events": 0,
        },
    )
    monkeypatch.setattr(db, "_get_client", lambda: _FakeClient(pd.DataFrame()))
    with TestClient(app) as client:
        resp = client.get("/api/pead?metric=NetIncomeLoss&windows=1,5&quintiles=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["windows"] == [1, 5]


def test_market_endpoint_uppercases_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "TRADE_DATE": [pd.Timestamp("2024-01-02")],
            "DAY_OPEN": [1.0],
            "DAY_HIGH": [2.0],
            "DAY_LOW": [0.5],
            "DAY_CLOSE": [1.5],
            "VOLUME": [10],
        }
    )
    fake = _FakeClient(df)
    monkeypatch.setattr(db, "_get_client", lambda: fake)
    with TestClient(app) as client:
        resp = client.get("/api/market/aapl?days=1")
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "AAPL"
    assert resp.json()["count"] == 1
