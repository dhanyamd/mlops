"""Bybit spot public market-data provider — REAL crypto OHLCV, no API key.

Public klines endpoint (verified live): ``api.bybit.com/v5/market/kline``
returns spot 1m bars (``category=spot``), paginated at 1000 bars/request.
Single venue (Bybit spot), so volume is exchange volume, not consolidated
tape; pairs are uppercase like ``BTCUSDT`` (same naming as the Binance
provider), and spot volume is in the base coin (e.g. BTC). Fractional
base-asset volume is kept as float.

The endpoint is public and keyless; ``api.bybit.com`` is the global mainnet
host (region-specific mirrors like ``api.bybit.id`` for Indonesia exist and
can be passed via ``base_url``). Bars come back newest-first, so the provider
sorts ascending to match the bar contract.

Output columns match the bar contract: ``symbol, ts, timeframe, open, high,
low, close, volume, provider, loaded_at``.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

from ingest.providers.base import BarProvider

_BASE = "https://api.bybit.com"
_HEADERS = {"User-Agent": "quant-signal-research/1.0"}
_REQUEST_INTERVAL_S = 0.2
_MAX_TRIES = 3
_PAGE = 1000  # max bars per request


class BybitBarProvider(BarProvider):
    """Minute OHLCV spot bars from Bybit (keyless, since ~2020)."""

    name = "bybit"
    timeframe = "1Min"

    def __init__(self, timeout: int = 30, base_url: str = _BASE) -> None:
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    def _get(self, params: dict[str, object]) -> dict:
        """GET the kline endpoint with retries; surface API error codes."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_TRIES):
            try:
                resp = requests.get(
                    f"{self._base_url}/v5/market/kline",
                    params=params,
                    headers=_HEADERS,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001 - network flakiness
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if payload.get("retCode") != 0:
                # Hard error (bad symbol, rate limit, ...): don't retry silently.
                raise RuntimeError(
                    f"bybit kline error retCode={payload.get('retCode')} "
                    f"retMsg={payload.get('retMsg')!r}"
                )
            return payload
        raise RuntimeError(f"bybit fetch failed after {_MAX_TRIES} tries: {last_exc}")

    def _fetch_symbol(self, symbol: str, days: int, minutes: int | None = None) -> list[tuple]:
        now_ms = int(time.time() * 1000)
        # ``minutes`` overrides ``days``: fetch only the trailing window (used by
        # the live-stream poller so each poll is a handful of bars, not a day).
        start_ms = now_ms - minutes * 60_000 if minutes else now_ms - days * 86_400_000
        # Kline row: [startTime_ms, open, high, low, close, volume, turnover].
        rows: list[tuple] = []
        cursor = start_ms
        while cursor < now_ms:
            page = self._get(
                {
                    "category": "spot",
                    "symbol": symbol,
                    "interval": "1",
                    "start": cursor,
                    "end": now_ms,
                    "limit": _PAGE,
                }
            )
            bars = (page.get("result") or {}).get("list") or []
            if not bars:
                break
            # Newest-first; sort ascending so rows land oldest→newest and the
            # pagination cursor walks forward without duplicates.
            bars = sorted(bars, key=lambda k: int(k[0]))
            for k in bars:
                rows.append(
                    (
                        symbol,
                        pd.to_datetime(int(k[0]), unit="ms", utc=True).tz_localize(None),
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
            oldest_start = int(bars[0][0])
            if len(bars) < _PAGE or oldest_start + 60_000 >= now_ms:
                break
            cursor = oldest_start + 1
            time.sleep(_REQUEST_INTERVAL_S)
        return rows

    def fetch_bars(self, symbols: list[str], days: int, minutes: int | None = None) -> pd.DataFrame:
        if not symbols or (days < 1 and not minutes):
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
            rows.extend(self._fetch_symbol(symbol.upper(), days, minutes))
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
