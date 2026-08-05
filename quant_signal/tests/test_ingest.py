"""Ingestion layer tests — schemas, quality gate, providers, store, flows.

No live Snowflake and no network: every external call is mocked or offline.
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pandas as pd
import pandas.testing as pdt
import pytest
from pydantic import ValidationError

from config.settings import Settings
from db.snowflake import SnowflakeClient
from ingest.providers.binance import BinanceBarProvider
from ingest.providers.fred import FredProvider
from ingest.providers.sec_edgar import (
    EdgarFundamentalsProvider,
    _extract_annual_filings,
)
from ingest.providers.synthetic import SyntheticBarProvider
from ingest.providers.yahoo import YahooBarProvider
from ingest.quality import validate_bars, validate_facts, validate_macro
from ingest.schemas import CompanyFact, EquityBar, MacroObservation
from ingest.store import write_company_facts, write_equity_bars, write_macro


def _settings(**overrides) -> Settings:
    base = {
        "snowflake_account": "GULXCKK-PI01025",
        "snowflake_user": "devuser",
        "snowflake_password": "pw",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def _bar_payload(**overrides) -> dict:
    payload = {
        "symbol": "aapl",
        "ts": dt.datetime(2026, 1, 2, 14, 31),
        "timeframe": "1Min",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
        "provider": "synthetic",
        "loaded_at": dt.datetime(2026, 1, 2, 20, 0),
    }
    payload.update(overrides)
    return payload


# ── Schemas ─────────────────────────────────────────────────────────────────


def test_equity_bar_valid_and_uppercases_symbol() -> None:
    bar = EquityBar.model_validate(_bar_payload())
    assert bar.symbol == "AAPL"
    assert bar.close == 100.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"high": 98.0, "low": 101.0},  # high < low
        {"close": 102.0, "high": 101.0},  # close > high
        {"open": 98.0, "low": 99.0},  # open < low
        {"volume": -5},  # negative volume
        {"close": -1.0},  # negative price
        {"ts": "not-a-date"},
    ],
)
def test_equity_bar_rejects_invalid(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        EquityBar.model_validate(_bar_payload(**overrides))


def test_company_fact_valid() -> None:
    fact = CompanyFact.model_validate(
        {
            "ticker": "aapl",
            "cik": "0000320193",
            "metric": "Revenues",
            "fiscal_year": 2024,
            "value": 391035000000.0,
            "filed_at": dt.date(2024, 11, 1),
            "loaded_at": dt.datetime(2026, 1, 2),
        }
    )
    assert fact.ticker == "AAPL"


# ── Quality gate ────────────────────────────────────────────────────────────


def test_validate_bars_splits_good_and_bad_with_reason() -> None:
    df = pd.DataFrame(
        [
            _bar_payload(),
            _bar_payload(symbol="MSFT", high=98.0, low=101.0),  # impossible OHLC
        ]
    )
    good, bad = validate_bars(df)
    assert len(good) == 1
    assert good.iloc[0]["symbol"] == "AAPL"
    assert len(bad) == 1
    assert "reason" in bad.columns
    assert "low must be" in bad.iloc[0]["reason"]


def test_validate_facts_rejects_bad_metric() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "metric": "Revenues",
                "fiscal_year": 2024,
                "value": 1.0,
                "filed_at": dt.date(2025, 1, 30),
                "loaded_at": dt.datetime(2026, 1, 2),
            },
            {
                "ticker": "MSFT",
                "cik": "bad-cik",
                "metric": "Revenues",
                "fiscal_year": 2024,
                "value": 1.0,
                "filed_at": dt.date(2025, 1, 30),
                "loaded_at": dt.datetime(2026, 1, 2),
            },
        ]
    )
    good, bad = validate_facts(df)
    assert len(good) == 1
    assert len(bad) == 1 and "cik" in bad.iloc[0]["reason"]


# ── Synthetic provider ──────────────────────────────────────────────────────


_BAR_COLUMNS = {
    "symbol",
    "ts",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
    "loaded_at",
}


def test_synthetic_provider_column_contract() -> None:
    df = SyntheticBarProvider().fetch_bars(["AAPL"], days=2)
    assert set(df.columns) == _BAR_COLUMNS
    assert not df.empty
    assert df["ts"].dt.tz is None  # timestamp_ntz-compatible


def test_synthetic_provider_is_deterministic() -> None:
    # Data rows must be reproducible; loaded_at is wall-clock metadata and is
    # excluded from the comparison.
    a = (
        SyntheticBarProvider(seed=7)
        .fetch_bars(["AAPL", "NVDA"], days=3)
        .drop(columns=["loaded_at"])
    )
    b = (
        SyntheticBarProvider(seed=7)
        .fetch_bars(["AAPL", "NVDA"], days=3)
        .drop(columns=["loaded_at"])
    )
    pdt.assert_frame_equal(a, b)


def test_synthetic_provider_ohlc_is_consistent() -> None:
    df = SyntheticBarProvider().fetch_bars(["AAPL"], days=3)
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["volume"] > 0).all()


# ── SEC EDGAR provider (network mocked) ─────────────────────────────────────


_SAMPLE_FACTS = {
    "facts": {
        "us-gaap": {
            # AAPL FY2018: the top line is tagged "Revenues" that year. The 10-K
            # also carries comparative prior-year rows (1.0 / 2.0) and a per-share
            # duplicate (265.595) — all of which must be ignored; the fiscal-year
            # figure is the max-`end` row (265,595M).
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "end": "2016-09-24",
                            "val": 1.0,
                            "fy": 2018,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2018-11-05",
                        },
                        {
                            "end": "2017-09-30",
                            "val": 2.0,
                            "fy": 2018,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2018-11-05",
                        },
                        {
                            "end": "2018-09-29",
                            "val": 265595000000.0,
                            "fy": 2018,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2018-11-05",
                        },
                    ],
                    "USD_per_Share": [
                        {
                            "end": "2018-09-29",
                            "val": 265.595,
                            "fy": 2018,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2018-11-05",
                        },
                    ],
                }
            },
            # Post-2018 AAPL tags revenue under the ASC 606 concept.
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        {
                            "end": "2019-09-28",
                            "val": 260174000000.0,
                            "fy": 2019,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2019-10-31",
                        },
                        {
                            "end": "2020-09-26",
                            "val": 274515000000.0,
                            "fy": 2020,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2020-10-30",
                        },
                    ]
                }
            },
            # Pre-2018 AAPL tagged revenue as SalesRevenueNet.
            "SalesRevenueNet": {
                "units": {
                    "USD": [
                        {
                            "end": "2015-09-26",
                            "val": 3.0,
                            "fy": 2017,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2017-11-03",
                        },
                        {
                            "end": "2016-09-24",
                            "val": 4.0,
                            "fy": 2017,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2017-11-03",
                        },
                        {
                            "end": "2017-09-30",
                            "val": 229234000000.0,
                            "fy": 2017,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2017-11-03",
                        },
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {
                            "end": "2020-09-26",
                            "val": 57411000000.0,
                            "fy": 2020,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2020-10-30",
                        },
                        # restatement filed later — must survive as its own row
                        {
                            "end": "2020-09-26",
                            "val": 57411000000.0,
                            "fy": 2020,
                            "fp": "FY",
                            "form": "10-K/A",
                            "filed": "2021-02-15",
                        },
                        # quarterly — must be ignored
                        {"end": "2020-06-27", "val": 123.0, "fp": "Q2", "form": "10-Q"},
                    ]
                }
            },
        }
    }
}


def test_extract_annual_filings_keeps_only_fiscal_year_end() -> None:
    # Comparative prior-year rows (3.0 / 4.0) live inside the same 10-K and must
    # NOT become separate annual figures — only the max-`end` row survives.
    rows = _extract_annual_filings(_SAMPLE_FACTS, ["SalesRevenueNet"])
    assert rows == [(2017, 229234000000.0, dt.date(2017, 11, 3))]


def test_extract_annual_filings_unions_concepts_and_prefers_primary_unit() -> None:
    # AAPL switched XBRL concepts across time (SalesRevenueNet -> Revenues ->
    # RevenueFromContractWithCustomer...). The union must span all of them, and
    # the primary "USD" unit wins over "USD_per_Share".
    rows = _extract_annual_filings(
        _SAMPLE_FACTS,
        ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    )
    assert len(rows) == 4
    by_fy = {fy: val for fy, val, _ in rows}
    assert by_fy[2017] == 229234000000.0
    assert by_fy[2018] == 265595000000.0
    assert by_fy[2019] == 260174000000.0
    assert by_fy[2020] == 274515000000.0


def test_extract_annual_filings_keeps_restatements_as_own_rows() -> None:
    rows = _extract_annual_filings(_SAMPLE_FACTS, ["NetIncomeLoss"])
    assert len(rows) == 2
    assert (2020, 57411000000.0, dt.date(2020, 10, 30)) in rows
    assert (2020, 57411000000.0, dt.date(2021, 2, 15)) in rows


def test_extract_annual_filings_unknown_concepts_yield_empty() -> None:
    assert _extract_annual_filings(_SAMPLE_FACTS, ["DoesNotExist"]) == []
    assert _extract_annual_filings(_SAMPLE_FACTS, []) == []


def test_fetch_facts_lands_valid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: object) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self._payload

    sec_ticker_map = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "APPLE INC"},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }

    def fake_get(url, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        if "company_tickers.json" in url:
            return _FakeResponse(sec_ticker_map)
        return _FakeResponse(_SAMPLE_FACTS)

    monkeypatch.setattr("ingest.providers.sec_edgar.requests.get", fake_get)
    df = EdgarFundamentalsProvider(user_agent="Test t@example.com").fetch_facts(
        ["AAPL"], metrics=["Revenues", "NetIncomeLoss"]
    )
    assert list(df.columns) == [
        "ticker",
        "cik",
        "metric",
        "fiscal_year",
        "value",
        "unit",
        "filed_at",
        "loaded_at",
    ]
    # Revenues spans the concept switch (SalesRevenueNet -> Revenues ->
    # RevenueFromContractWithCustomer...) as a single metric → 4 fiscal years.
    rev = df[df["metric"] == "Revenues"]
    assert set(rev["fiscal_year"]) == {2017, 2018, 2019, 2020}
    assert rev[rev["fiscal_year"] == 2020]["value"].iloc[0] == 274515000000.0
    assert set(rev["filed_at"]) == {
        dt.date(2017, 11, 3),
        dt.date(2018, 11, 5),
        dt.date(2019, 10, 31),
        dt.date(2020, 10, 30),
    }
    assert rev["cik"].iloc[0] == "0000320193"
    # Restatement survives as its own point-in-time row.
    assert len(df[df["metric"] == "NetIncomeLoss"]) == 2


# ── Store (SnowflakeClient mocked) ──────────────────────────────────────────


def test_write_equity_bars_uses_upsert_on_natural_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_upsert(self, df, table_name, merge_keys, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(table_name=table_name, merge_keys=merge_keys, schema=kwargs["schema"])
        return len(df)

    monkeypatch.setattr(SnowflakeClient, "upsert_df", fake_upsert)
    n = write_equity_bars(SyntheticBarProvider().fetch_bars(["AAPL"], days=1), _settings())
    assert n > 0
    assert captured["table_name"] == "EQUITY_BARS"
    assert captured["merge_keys"] == ["symbol", "timeframe", "ts"]
    assert captured["schema"] == "BRONZE"


def test_write_company_facts_uses_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_upsert(self, df, table_name, merge_keys, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(table_name=table_name, merge_keys=merge_keys)
        return len(df)

    monkeypatch.setattr(SnowflakeClient, "upsert_df", fake_upsert)
    df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "metric": "Revenues",
                "fiscal_year": 2024,
                "value": 1.0,
                "filed_at": dt.date(2025, 1, 30),
                "loaded_at": dt.datetime(2026, 1, 2),
            }
        ]
    )
    assert write_company_facts(df, _settings()) == 1
    assert captured["table_name"] == "COMPANY_FACTS"
    assert captured["merge_keys"] == ["ticker", "metric", "fiscal_year", "filed_at"]


# ── Flow (write tasks mocked, offline) ──────────────────────────────────────


def test_ingest_market_data_flow_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    import flows.ingest_market_data as flow_mod

    written: dict = {}

    def fake_write_bronze(df: pd.DataFrame) -> int:  # type: ignore[no-untyped-def]
        written["bronze"] = len(df)
        return len(df)

    def fake_quarantine(df: pd.DataFrame, source: str) -> int:  # type: ignore[no-untyped-def]
        written["quarantine"] = len(df)
        return len(df)

    monkeypatch.setattr(flow_mod, "write_bronze", fake_write_bronze)
    monkeypatch.setattr(flow_mod, "quarantine", fake_quarantine)

    # .fn() runs the underlying function directly (no ephemeral Prefect server,
    # no network) — this is the maintainer-documented way to test flows.
    result = flow_mod.ingest_market_data.fn(provider_name="synthetic", symbols=["AAPL"], days=2)
    assert result["fetched"] > 0
    assert result["written"] == result["fetched"]
    assert result["quarantined"] == 0
    assert written["bronze"] == result["fetched"]


def test_ingest_market_data_flow_quarantines_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    import flows.ingest_market_data as flow_mod

    written: dict = {}

    def fake_fetch(provider_name: str, symbols: list[str], days: int) -> pd.DataFrame:
        # One impossible OHLC bar (high < low) → must go to quarantine.
        return pd.DataFrame([_bar_payload(symbol="BAD", high=98.0, low=101.0)])

    def fake_write_bronze(df: pd.DataFrame) -> int:  # type: ignore[no-untyped-def]
        written["bronze"] = len(df)
        return len(df)

    def fake_quarantine(df: pd.DataFrame, source: str) -> int:  # type: ignore[no-untyped-def]
        written["quarantine"] = len(df)
        return len(df)

    monkeypatch.setattr(flow_mod, "fetch_bars", fake_fetch)
    monkeypatch.setattr(flow_mod, "write_bronze", fake_write_bronze)
    monkeypatch.setattr(flow_mod, "quarantine", fake_quarantine)

    result = flow_mod.ingest_market_data.fn(symbols=["BAD"], days=1)
    assert result["fetched"] == 1
    assert result["written"] == 0  # nothing valid reached Bronze
    assert result["quarantined"] == 1  # the bad row is quarantined, not dropped
    assert written["quarantine"] == 1


def test_ingest_market_data_routes_crypto_to_own_table(monkeypatch: pytest.MonkeyPatch) -> None:
    import flows.ingest_market_data as flow_mod

    routed: list[str] = []

    def fake_fetch(provider_name: str, symbols: list[str], days: int) -> pd.DataFrame:
        return pd.DataFrame([_bar_payload(symbol="BTCUSDT")])

    def fake_write_bronze(df: pd.DataFrame) -> int:  # type: ignore[no-untyped-def]
        routed.append("equity")
        return len(df)

    def fake_write_crypto(df: pd.DataFrame) -> int:  # type: ignore[no-untyped-def]
        routed.append("crypto")
        return len(df)

    def fake_quarantine(df: pd.DataFrame, source: str) -> int:  # type: ignore[no-untyped-def]
        return len(df)

    monkeypatch.setattr(flow_mod, "fetch_bars", fake_fetch)
    monkeypatch.setattr(flow_mod, "write_bronze", fake_write_bronze)
    monkeypatch.setattr(flow_mod, "write_bronze_crypto", fake_write_crypto)
    monkeypatch.setattr(flow_mod, "quarantine", fake_quarantine)

    result = flow_mod.ingest_market_data.fn(provider_name="binance", symbols=["BTCUSDT"], days=1)
    assert routed == ["crypto"]  # crypto must NEVER land in EQUITY_BARS
    assert result["written"] == 1


# ── Real providers (network mocked, offline) ────────────────────────────────


def test_macro_observation_valid_and_uppercases_series() -> None:
    obs = MacroObservation.model_validate(
        {
            "series_id": "vixcls",
            "date": dt.date(2026, 8, 1),
            "value": 17.24,
            "loaded_at": dt.datetime(2026, 8, 5),
        }
    )
    assert obs.series_id == "VIXCLS"
    assert obs.date == dt.date(2026, 8, 1)


def test_validate_macro_splits_bad_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "series_id": "VIXCLS",
                "date": dt.date(2026, 8, 1),
                "value": 17.24,
                "loaded_at": dt.datetime(2026, 8, 5),
            },
            {
                "series_id": "bad id!",
                "date": dt.date(2026, 8, 1),
                "value": 17.24,
                "loaded_at": dt.datetime(2026, 8, 5),
            },
        ]
    )
    good, bad = validate_macro(df)
    assert len(good) == 1
    assert len(bad) == 1 and "series_id" in bad.iloc[0]["reason"]


def test_write_macro_uses_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_upsert(self, df, table_name, merge_keys, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(table_name=table_name, merge_keys=merge_keys)
        return len(df)

    monkeypatch.setattr(SnowflakeClient, "upsert_df", fake_upsert)
    df = pd.DataFrame(
        [
            {
                "series_id": "VIXCLS",
                "date": dt.date(2026, 8, 1),
                "value": 17.24,
                "loaded_at": dt.datetime(2026, 8, 5),
            }
        ]
    )
    assert write_macro(df, _settings()) == 1
    assert captured["table_name"] == "FRED_MACRO"
    assert captured["merge_keys"] == ["series_id", "date"]


def test_fred_provider_parses_csv_and_drops_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self._text = text

        def raise_for_status(self) -> None:
            return None

        @property
        def text(self) -> str:
            return self._text

    csv_text = "observation_date,VIXCLS\n2026-07-30,16.5\n2026-07-31,.\n2026-08-03,17.2\n"

    def fake_get(url, params=None, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        assert params["id"] == "VIXCLS"
        return _FakeResponse(csv_text)

    monkeypatch.setattr("ingest.providers.fred.requests.get", fake_get)
    df = FredProvider().fetch_observations(["vixcls"])
    assert list(df.columns) == ["series_id", "date", "value", "loaded_at"]
    assert len(df) == 2  # "." observation is a gap, not a bad row
    assert df.iloc[0]["series_id"] == "VIXCLS"
    assert df.iloc[0]["date"] == dt.date(2026, 7, 30)
    assert df.iloc[0]["value"] == 16.5


def test_yahoo_provider_parses_daily_bars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    import json

    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "AAPL"},
                    "timestamp": [1785369600, 1785456000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 102.0],
                                "high": [101.0, 103.0],
                                "low": [99.0, 101.0],
                                "close": [100.5, 102.5],
                                "volume": [1000, 2000],
                            }
                        ]
                    },
                }
            ]
        }
    }

    def fake_get(self, url, params=None, timeout=None):  # type: ignore[no-untyped-def]
        class _R:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return json.loads(json.dumps(payload))

        return _R()

    monkeypatch.setattr("curl_cffi.requests.Session.get", fake_get)
    df = YahooBarProvider(cache_dir=tmp_path).fetch_bars(["aapl"], days=5)
    assert set(df.columns) == _BAR_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["timeframe"] == "1D"
    assert df.iloc[0]["provider"] == "yahoo"
    assert df.iloc[0]["symbol"] == "AAPL"
    assert df["ts"].dt.tz is None


def test_yahoo_provider_skips_missing_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    import json

    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "AAPL"},
                    "timestamp": [1785369600, 1785456000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, None],
                                "high": [101.0, None],
                                "low": [99.0, None],
                                "close": [100.5, None],
                                "volume": [1000, None],
                            }
                        ]
                    },
                }
            ]
        }
    }

    def fake_get(self, url, params=None, timeout=None):  # type: ignore[no-untyped-def]
        class _R:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return json.loads(json.dumps(payload))

        return _R()

    monkeypatch.setattr("curl_cffi.requests.Session.get", fake_get)
    df = YahooBarProvider(cache_dir=tmp_path).fetch_bars(["aapl"], days=5)
    assert len(df) == 1  # the no-data day is a gap, not a bad row


def test_binance_provider_paginates_minute_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    now_ms = 1785900000000
    pages = [
        [
            [
                now_ms - 120000,
                "100.0",
                "101.0",
                "99.5",
                "100.5",
                "1.25",
                now_ms - 60000,
                0,
                0,
                0,
                0,
                "0",
            ],
            [now_ms - 60000, "100.5", "102.0", "100.0", "101.5", "0.75", now_ms, 0, 0, 0, 0, "0"],
        ],
        [],
    ]

    def fake_get(url, params=None, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        class _R:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> list[list[object]]:
                return json.loads(json.dumps(pages.pop(0)))

        return _R()

    monkeypatch.setattr("ingest.providers.binance.requests.get", fake_get)
    df = BinanceBarProvider().fetch_bars(["btcusdt"], days=1)
    assert set(df.columns) == _BAR_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["timeframe"] == "1Min"
    assert df.iloc[0]["provider"] == "binance"
    assert df.iloc[0]["symbol"] == "BTCUSDT"
    assert df.iloc[0]["volume"] == 1.25  # fractional crypto volume stays float


# ── Macro flow (write tasks mocked, offline) ────────────────────────────────


def test_ingest_macro_data_flow_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    import flows.ingest_macro_data as flow_mod

    written: dict = {}

    def fake_fetch(series_ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "series_id": "VIXCLS",
                    "date": dt.date(2026, 8, 1),
                    "value": 17.24,
                    "loaded_at": dt.datetime(2026, 8, 5),
                },
                {
                    "series_id": "BAD ID!",
                    "date": dt.date(2026, 8, 1),
                    "value": 17.24,
                    "loaded_at": dt.datetime(2026, 8, 5),
                },
            ]
        )

    def fake_write_bronze(df: pd.DataFrame) -> int:  # type: ignore[no-untyped-def]
        written["bronze"] = len(df)
        return len(df)

    def fake_quarantine(df: pd.DataFrame, source: str) -> int:  # type: ignore[no-untyped-def]
        written["quarantine"] = len(df)
        return len(df)

    monkeypatch.setattr(flow_mod, "fetch_macro", fake_fetch)
    monkeypatch.setattr(flow_mod, "write_bronze", fake_write_bronze)
    monkeypatch.setattr(flow_mod, "quarantine", fake_quarantine)

    result = flow_mod.ingest_macro_data.fn(series_ids=["VIXCLS"])
    assert result["fetched"] == 2
    assert result["written"] == 1
    assert result["quarantined"] == 1
    assert written["bronze"] == 1 and written["quarantine"] == 1
