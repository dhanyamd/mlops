"""Yahoo Finance equity bars provider — REAL daily OHLCV, no API key.

Uses the public ``query1/query2.finance.yahoo.com`` chart endpoint. This is an
unofficial endpoint (Yahoo has no documented public API), so it is fragile and
ToS-gray; it is used here because the user requires real, keyless, downloadable
data. Data is real exchange OHLCV.

IMPORTANT — verified live in this repo's history: Yahoo began blocking plain
``requests``-library calls (429 Too Many Requests, TLS fingerprint detection).
The community-wide fix (yfinance switched to it) is ``curl_cffi`` with Chrome
TLS impersonation. This provider therefore: impersonates Chrome, alternates
query1/query2 hosts, backs off exponentially on 429, throttles per symbol, and
caches responses to disk so scheduled reruns don't re-trip the rate limiter.

Output columns match the contract: ``symbol, ts, timeframe, open, high, low,
close, volume, provider, loaded_at``. ``ts`` is the trading date at UTC
midnight so ``ts::date`` == trade date regardless of exchange timezone.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
from curl_cffi import requests as cr

from ingest.providers.base import BarProvider

_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
_CHART_PATH = "/v8/finance/chart"
# Be polite: one request per symbol, spaced out. Yahoo rate-limits by IP
# (community reports ~100-360 req/hour); a burst trips the limiter fast.
_REQUEST_INTERVAL_S = 1.0
_MAX_TRIES = 5
_429_BACKOFF_S = 30  # per-attempt wait when Yahoo says "Too Many Requests"
_CACHE_TTL_S = 12 * 3600  # reuse a symbol's response for 12h (scheduled reruns)

# Yahoo range strings are coarse buckets; map our "last N days" to the nearest
# range that fully covers N days.
_RANGE_BY_DAYS = (
    (5, "5d"),
    (30, "1mo"),
    (90, "3mo"),
    (180, "6mo"),
    (365, "1y"),
    (730, "2y"),
    (1825, "5y"),
)


def _range_for(days: int) -> str:
    for cap, label in _RANGE_BY_DAYS:
        if days <= cap:
            return label
    return "max"


class YahooBarProvider(BarProvider):
    """Daily US-equity OHLCV bars from Yahoo Finance (keyless)."""

    name = "yahoo"
    timeframe = "1D"

    def __init__(self, timeout: int = 30, cache_dir: Path | None = None) -> None:
        self._timeout = timeout
        self._cache_dir = cache_dir or (Path.home() / ".cache" / "quant_signal" / "yahoo")
        self._session = cr.Session(impersonate="chrome")

    def _cache_path(self, symbol: str, range_: str) -> Path:
        return self._cache_dir / f"{symbol}_{range_}.json"

    def _load_cache(self, symbol: str, range_: str) -> dict | None:
        path = self._cache_path(symbol, range_)
        try:
            if not path.is_file() or time.time() - path.stat().st_mtime > _CACHE_TTL_S:
                return None
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _store_cache(self, symbol: str, range_: str, payload: dict) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(symbol, range_).write_text(json.dumps(payload))
        except OSError:
            pass  # cache is an optimization; never fail a fetch over it

    def _fetch(self, symbol: str, range_: str) -> dict:
        cached = self._load_cache(symbol, range_)
        if cached is not None:
            return cached
        params = {
            "range": range_,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,split",
        }
        last_exc: Exception | None = None
        for attempt in range(_MAX_TRIES):
            host = _HOSTS[attempt % len(_HOSTS)]
            try:
                resp = self._session.get(
                    f"{host}{_CHART_PATH}/{symbol}", params=params, timeout=self._timeout
                )
                resp.raise_for_status()
                payload = resp.json()
                self._store_cache(symbol, range_, payload)
                return payload
            except Exception as exc:  # noqa: BLE001 - provider flakiness
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                wait = _429_BACKOFF_S if status == 429 else 1.5 * (attempt + 1)
                time.sleep(wait)
        raise RuntimeError(
            f"yahoo fetch failed for {symbol!r} after {_MAX_TRIES} tries: {last_exc}"
        )

    def _fetch_symbol(self, symbol: str, days: int) -> list[tuple]:
        range_ = _range_for(days)
        payload = self._fetch(symbol, range_)
        result = payload.get("chart", {}).get("result") or []
        if not result:
            raise RuntimeError(f"yahoo returned no data for {symbol!r}")
        first = result[0]
        timestamps = first.get("timestamp") or []
        quote = (first.get("indicators", {}).get("quote") or [{}])[0]
        opened = quote.get("open")
        if timestamps and opened is None:
            opened = [None] * len(timestamps)
        ts_utc = pd.Series(pd.to_datetime(timestamps, unit="s", utc=True))
        dates = ts_utc.dt.tz_convert("America/New_York").dt.date
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        rows: list[tuple] = []
        for i, d in enumerate(dates):
            close = _safe_float(quote.get("close"), i)
            if close is None:
                continue  # non-trading day / missing data — a gap, not a bad row
            open_ = _safe_float(quote.get("open"), i)
            high = _safe_float(quote.get("high"), i)
            low = _safe_float(quote.get("low"), i)
            volume = _safe_float(quote.get("volume"), i) or 0.0
            if open_ is None or high is None or low is None:
                continue
            rows.append(
                (
                    symbol.upper(),
                    dt.datetime.combine(d, dt.time(0, 0)),
                    self.timeframe,
                    open_,
                    high,
                    low,
                    close,
                    volume,
                    self.name,
                    now,
                )
            )
        return rows

    def fetch_bars(self, symbols: list[str], days: int) -> pd.DataFrame:
        if not symbols or days < 1:
            return pd.DataFrame(
                columns=[
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
                ]
            )
        rows: list[tuple] = []
        for symbol in symbols:
            rows.extend(self._fetch_symbol(symbol, days))
            time.sleep(_REQUEST_INTERVAL_S)
        return pd.DataFrame(
            rows,
            columns=[
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
            ],
        )


def _safe_float(values: list[object] | None, index: int) -> float | None:
    if not values or index >= len(values) or values[index] is None:
        return None
    return float(values[index])
