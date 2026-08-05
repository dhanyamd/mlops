"""Binance public market-data provider — REAL crypto OHLCV, no API key.

Public klines endpoint (verified live): ``/api/v3/klines`` returns minute bars
since ~2017, paginated at 1000 bars/request. Single venue (Binance), so volume
is exchange volume, not consolidated tape; pairs are uppercase like ``BTCUSDT``.
Fractional base-asset volume is kept as float.

Output columns match the bar contract: ``symbol, ts, timeframe, open, high,
low, close, volume, provider, loaded_at``.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

from ingest.providers.base import BarProvider

_BASE = "https://api.binance.com/api/v3/klines"
_HEADERS = {"User-Agent": "quant-signal-research/1.0"}
_REQUEST_INTERVAL_S = 0.2
_MAX_TRIES = 3
_PAGE = 1000  # max bars per request


class BinanceBarProvider(BarProvider):
    """Minute OHLCV bars from Binance (keyless, since 2017)."""

    name = "binance"
    timeframe = "1Min"

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def _get(self, params: dict[str, object]) -> list[list[object]]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_TRIES):
            try:
                resp = requests.get(_BASE, params=params, headers=_HEADERS, timeout=self._timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - network flakiness
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"binance fetch failed after {_MAX_TRIES} tries: {last_exc}")

    def _fetch_symbol(self, symbol: str, days: int) -> list[tuple]:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - days * 86_400_000
        # Kline: [openTime_ms, open, high, low, close, volume, closeTime, ...]
        rows: list[tuple] = []
        cursor = start_ms
        while cursor < now_ms:
            page = self._get(
                {
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": cursor,
                    "endTime": now_ms,
                    "limit": _PAGE,
                }
            )
            if not page:
                break
            for k in page:
                rows.append(
                    (
                        symbol,
                        pd.to_datetime(k[0], unit="ms", utc=True).tz_localize(None),
                        self.timeframe,
                        float(k[1]),
                        float(k[2]),
                        float(k[3]),
                        float(k[4]),
                        float(k[5]),
                        self.name,
                        dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
                    )
                )
            cursor = int(page[-1][0]) + 1
            if len(page) < _PAGE:
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
            rows.extend(self._fetch_symbol(symbol.upper(), days))
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
