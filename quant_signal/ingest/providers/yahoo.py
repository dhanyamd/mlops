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

Long-history gotcha (verified live): ``range=max`` silently returns 3mo
(quarterly) bars, so history beyond the 5y bucket is fetched with the
undocumented-but-stable ``period1``/``period2`` params in overlapping daily
windows (``_fetch_period``), deduplicated on the trading date.

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
# Verified live: Yahoo's chart API downgrades range=max to 3mo (quarterly)
# granularity, silently. For history beyond the 5y range bucket we paginate with
# the undocumented-but-stable period1/period2 params, which return true daily
# bars (verified: 19y window -> ~4,800 daily points, dataGranularity=1d).
_PERIOD_THRESHOLD_DAYS = 1825
_CHUNK_DAYS = 15 * 365  # safe window size for one daily request
_CHUNK_OVERLAP_DAYS = 60  # overlap so a chunk boundary never drops a day
# Below this a 400 window is treated as "no data" (predates the ticker's IPO)
# instead of recursing forever.
_MIN_WINDOW_DAYS = 120


class _Yahoo400Error(RuntimeError):
    """Yahoo returned HTTP 400 for a window — deterministic (e.g. predates the
    ticker's IPO), so it must not be retried; the caller splits the window."""


def _range_for(days: int) -> str:
    for cap, label in _RANGE_BY_DAYS:
        if days <= cap:
            return label
    return "5y"


def _period_windows(days: int) -> list[tuple[int, int]]:
    """Non-overlapping (period1, period2) chunks covering the last ``days``.

    Each chunk requests 1d bars over ``_CHUNK_DAYS`` with a small overlap; the
    provider dedupes on the trading date. Anchored to ``now`` so reruns within
    the cache TTL reuse the same keys.
    """
    now = int(time.time())
    end = now
    windows: list[tuple[int, int]] = []
    covered = 0
    while covered < days:
        chunk = min(_CHUNK_DAYS, days - covered)
        start = end - chunk * 86400
        windows.append((start, end))
        covered += chunk
        end = start - _CHUNK_OVERLAP_DAYS * 86400
    return windows


def _merge_payloads(symbol: str, payloads: list[dict]) -> dict:
    """Concatenate several chart payloads into one (timestamps + quote fields)."""
    ts: list[int] = []
    quote: dict[str, list[object]] = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }
    for payload in payloads:
        result = payload.get("chart", {}).get("result") or []
        if not result:
            continue
        first = result[0]
        chunk_ts = first.get("timestamp") or []
        chunk_quote = (first.get("indicators", {}).get("quote") or [{}])[0]
        ts.extend(chunk_ts)
        for field in quote:
            values = chunk_quote.get(field)
            if values is None:
                values = [None] * len(chunk_ts)
            quote[field].extend(values)
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": symbol},
                    "timestamp": ts,
                    "indicators": {"quote": [quote]},
                }
            ]
        }
    }


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

    def _request(self, symbol: str, cache_key: str, params: dict) -> dict:
        """GET a chart payload with Chrome impersonation, retries and disk cache."""
        cached = self._load_cache(symbol, cache_key)
        if cached is not None:
            return cached
        last_exc: Exception | None = None
        for attempt in range(_MAX_TRIES):
            host = _HOSTS[attempt % len(_HOSTS)]
            try:
                resp = self._session.get(
                    f"{host}{_CHART_PATH}/{symbol}", params=params, timeout=self._timeout
                )
                resp.raise_for_status()
                payload = resp.json()
                self._store_cache(symbol, cache_key, payload)
                return payload
            except Exception as exc:  # noqa: BLE001 - provider flakiness
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 400:
                    # Deterministic: retrying won't help (e.g. window predates the
                    # ticker's IPO). Surface it so the caller can split the window.
                    raise _Yahoo400Error(f"yahoo returned 400 for {symbol!r} {cache_key}") from exc
                last_exc = exc
                wait = _429_BACKOFF_S if status == 429 else 1.5 * (attempt + 1)
                time.sleep(wait)
        raise RuntimeError(
            f"yahoo fetch failed for {symbol!r} after {_MAX_TRIES} tries: {last_exc}"
        )

    def _fetch(self, symbol: str, range_: str) -> dict:
        return self._request(
            symbol,
            range_,
            {
                "range": range_,
                "interval": "1d",
                "includePrePost": "false",
                "events": "div,split",
            },
        )

    def _fetch_period(self, symbol: str, days: int) -> dict:
        """True daily bars over the last ``days`` via period1/period2 windows.

        Yahoo silently downgrades ``range=max`` to 3mo granularity, so long
        history is fetched in overlapping daily windows (deduped downstream).
        A window that predates a ticker's IPO gets HTTP 400; those windows are
        split in half recursively until ``_MIN_WINDOW_DAYS``, below which the
        window is treated as "no data" (not yet listed).
        """
        payloads = [
            self._fetch_window(symbol, period1, period2)
            for period1, period2 in _period_windows(days)
        ]
        payloads = [p for p in payloads if p is not None]
        if not payloads:
            raise RuntimeError(f"yahoo returned no data for {symbol!r}")
        return _merge_payloads(symbol, payloads)

    def _fetch_window(self, symbol: str, period1: int, period2: int) -> dict | None:
        """Fetch one daily window; on 400 (predates IPO) split it in half."""
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,split",
        }
        try:
            return self._request(symbol, f"p{period1}_{period2}", params)
        except _Yahoo400Error:
            span_days = (period2 - period1) // 86400
            if span_days <= _MIN_WINDOW_DAYS:
                return None  # predates the ticker's IPO — no data here
            mid = period1 + (period2 - period1) // 2
            left = self._fetch_window(symbol, period1, mid)
            right = self._fetch_window(symbol, mid, period2)
            return _merge_payloads(symbol, [p for p in (left, right) if p is not None])

    def _fetch_symbol(self, symbol: str, days: int) -> list[tuple]:
        if days > _PERIOD_THRESHOLD_DAYS:
            payload = self._fetch_period(symbol, days)
        else:
            payload = self._fetch(symbol, _range_for(days))
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
        seen: set[dt.date] = set()
        for i, d in enumerate(dates):
            if d in seen:
                continue  # overlap across period windows — keep the first occurrence
            seen.add(d)
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


def _merge_payloads(symbol: str, payloads: list[dict]) -> dict:
    """Concatenate several chart payloads into one (timestamps + quote fields)."""
    ts: list[int] = []
    quote: dict[str, list[object]] = {
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
    }
    for payload in payloads:
        result = payload.get("chart", {}).get("result") or []
        if not result:
            continue
        first = result[0]
        chunk_ts = first.get("timestamp") or []
        chunk_quote = (first.get("indicators", {}).get("quote") or [{}])[0]
        ts.extend(chunk_ts)
        for field in quote:
            values = chunk_quote.get(field)
            if values is None:
                values = [None] * len(chunk_ts)
            quote[field].extend(values)
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": symbol},
                    "timestamp": ts,
                    "indicators": {"quote": [quote]},
                }
            ]
        }
    }


def _safe_float(values: list[object] | None, index: int) -> float | None:
    if not values or index >= len(values) or values[index] is None:
        return None
    return float(values[index])
