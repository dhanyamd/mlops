"""Alpaca Market Data equity bars provider — official US-equity upgrade (IEX feed).

Unlike the keyless providers (Yahoo/Binance/FRED/EDGAR), Alpaca requires a free
API key pair (register at alpaca.markets). The free tier serves the **IEX
feed** (~2.5% of consolidated US equity volume), documented here as the
production-grade step up from Yahoo: official endpoint, no ToS ambiguity.

Endpoint (verified against the public docs):

    GET https://data.alpaca.markets/v2/stocks/{symbol}/bars
        ?timeframe=1Day&feed=iex&adjustment=raw&start=...&limit=10000

Auth is two headers (``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY``). Responses
paginate via ``next_page_token``. Bars are returned as ``{t,o,h,l,c,v,n,vw}``
with ``t`` an RFC-3339 UTC timestamp.

Timezone contract — daily bars are anchored at **New York midnight** (the start
of the trading day), so ``t`` like ``2023-09-29T04:00:00Z`` (ET is UTC-4 in
summer) belongs to the 2023-09-29 trading day. To keep the repo-wide invariant
``ts::date == trade date`` (same as the Yahoo provider), this provider converts
each bar's UTC timestamp to the America/New_York date and stores ``ts`` as that
date at UTC midnight.

Output columns match the bar contract: ``symbol, ts, timeframe, open, high,
low, close, volume, provider, loaded_at``.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

from ingest.providers.base import BarProvider

_BASE = "https://data.alpaca.markets/v2/stocks"
# Raw (unadjusted) prices: ``adjustment=raw`` keeps splits/dividends untouched
# so the raw tape matches the other providers' handling. ``feed=iex`` is the
# free feed (~2.5% of consolidated volume); a paid subscription can switch to
# ``feed=sip`` for the full consolidated tape without code changes.
_FEED = "iex"
_TIMEFRAME = "1Day"
_PAGE = 10_000  # max bars per request
_MAX_TRIES = 3
_REQUEST_INTERVAL_S = 0.2  # be polite across the symbols in one batch


class AlpacaBarProvider(BarProvider):
    """Daily US-equity OHLCV bars from Alpaca Market Data (IEX feed, API key)."""

    name = "alpaca"
    timeframe = "1D"

    def __init__(self, api_key: str, api_secret: str, timeout: int = 30) -> None:
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        self._timeout = timeout

    def _get(self, url: str, params: dict[str, object]) -> dict:
        """GET a bars page with auth headers and bounded retries."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_TRIES):
            try:
                resp = requests.get(
                    url, params=params, headers=self._headers, timeout=self._timeout
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - provider flakiness
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"alpaca fetch failed after {_MAX_TRIES} tries: {last_exc}")

    def _fetch_symbol(self, symbol: str, days: int) -> list[tuple]:
        # ``start`` covers the trailing ``days``; one page holds ~27y of daily
        # bars so this is normally a single request (pagination is still
        # handled for very long windows).
        start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        rows: list[tuple] = []
        page_token: str | None = None
        while True:
            params: dict[str, object] = {
                "timeframe": _TIMEFRAME,
                "feed": _FEED,
                "adjustment": "raw",
                "start": start,
                "limit": _PAGE,
            }
            if page_token:
                params["page_token"] = page_token
            payload = self._get(f"{_BASE}/{symbol}/bars", params)
            for bar in payload.get("bars") or []:
                # Daily bar timestamps are the trading day's New York midnight in
                # UTC; normalize to the NY calendar date so ``ts::date`` is the
                # trade date regardless of DST.
                t_utc = pd.Timestamp(bar["t"]).tz_convert("America/New_York")
                rows.append(
                    (
                        symbol.upper(),
                        dt.datetime.combine(t_utc.date(), dt.time(0, 0)),
                        self.timeframe,
                        float(bar["o"]),
                        float(bar["h"]),
                        float(bar["l"]),
                        float(bar["c"]),
                        float(bar["v"]),
                        self.name,
                        dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
                    )
                )
            page_token = payload.get("next_page_token")
            if not page_token:
                break
            time.sleep(_REQUEST_INTERVAL_S)
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
