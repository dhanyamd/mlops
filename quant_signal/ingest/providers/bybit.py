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
import logging
import time
from collections.abc import Callable

import pandas as pd
import requests

from config.logging import get_logger
from ingest.providers.base import BarProvider

logger = get_logger(__name__)

_BASE = "https://api.bybit.com"
_HEADERS = {"User-Agent": "quant-signal-research/1.0"}
_REQUEST_INTERVAL_S = 0.2
_MAX_TRIES = 3
_PAGE = 1000  # max bars per request
# Real-time poll (15s cadence, 45s hard deadline): a single request must not be
# allowed to stall the whole window. 10s bounds a wedged/illiquid pair; the next
# poll retries it naturally, so we never sacrifice pipeline liveness for one
# symbol (e.g. ZECUSDT, which returns no 1m klines and hangs with a time window).
_LIVE_TIMEOUT_S = 10
# Live path fast-fails a dead symbol after one retry; warm-start backfill keeps
# the full retry budget (transient blips during history load matter more there).
_LIVE_TRIES = 1


class BybitBarProvider(BarProvider):
    """Minute OHLCV spot bars from Bybit (keyless, since ~2020)."""

    name = "bybit"
    timeframe = "1Min"

    def __init__(self, timeout: int = _LIVE_TIMEOUT_S, base_url: str = _BASE) -> None:
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    def _get(self, params: dict[str, object], retries: int | None = None) -> dict:
        """GET the kline endpoint with retries; surface API error codes.

        ``retries`` overrides the default budget (used by the live poller to
        fast-fail a dead symbol instead of stalling the whole window).
        """
        last_exc: Exception | None = None
        for attempt in range(retries if retries is not None else _MAX_TRIES):
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

    def _fetch_symbol(
        self,
        symbol: str,
        days: int,
        minutes: int | None = None,
        *,
        on_skip: Callable[[str, str], None] | None = None,
    ) -> list[tuple]:
        now_ms = int(time.time() * 1000)
        # ``minutes`` overrides ``days``: fetch only the trailing window (used by
        # the live-stream poller so each poll is a handful of bars, not a day).
        start_ms = now_ms - minutes * 60_000 if minutes else now_ms - days * 86_400_000
        # Kline row: [startTime_ms, open, high, low, close, volume, turnover].
        rows: list[tuple] = []
        cursor = start_ms
        while cursor < now_ms:
            try:
                page = self._get(
                    {
                        "category": "spot",
                        "symbol": symbol,
                        "interval": "1",
                        "start": cursor,
                        "end": now_ms,
                        "limit": _PAGE,
                    },
                    retries=_LIVE_TRIES,
                )
            except Exception as exc:  # noqa: BLE001 - one dead/illiquid pair must
                # not stall the whole poll window; skip it and let the next cycle
                # retry. Surfaced loudly so a systematically-broken symbol is
                # visible without freezing ingestion.
                logger.warning(
                    "bybit 1m kline for %s failed (%s); skipping this symbol this poll",
                    symbol,
                    type(exc).__name__,
                )
                if on_skip is not None:
                    on_skip(symbol, f"timeout:{type(exc).__name__}")
                return rows
            bars = (page.get("result") or {}).get("list") or []
            if not bars:
                break
            # Newest-first; sort ascending so rows land oldest→newest and the
            # pagination cursor walks forward without duplicates.
            bars = sorted(bars, key=lambda k: int(k[0]))
            # Bybit returns the *latest available* bars when a window is empty
            # instead of honoring start/end (observed: ZECUSDT's 1m klines are
            # frozen in Feb-2025, so a recent window yields stale bars). Keep
            # only bars inside [start_ms, now_ms); if the window is empty, stop —
            # otherwise we'd re-fetch identical stale pages forever (cursor never
            # reaches now_ms) and wedge the whole poll.
            in_window = [k for k in bars if start_ms <= int(k[0]) < now_ms]
            if not in_window:
                # Frozen/illiquid pair (e.g. ZECUSDT's 1m klines stop in 2025):
                # Bybit returns the latest available bars outside our window.
                # Skip the symbol rather than loop forever on stale data.
                if on_skip is not None:
                    on_skip(symbol, "stale:no_bars_in_window")
                break
            bars = in_window
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

    def fetch_klines_1h(
        self, symbol: str, limit: int = 1000, *, end_ms: int | None = None,
        category: str = "spot",
    ) -> list[tuple[int, float, float]]:
        """Hourly OHLCV closes for warm-start (interval=60), oldest→newest.

        Returns ``(window_end_ms, close, volume)`` for up to ``limit`` bars,
        paginating backward (Bybit caps a page at 1000). Seeds the live xs_rel14
        signal's close history so its 336h lookback — and the UP-UP regime gate's
        1346-bar BTC requirement — are satisfied on the first live bar (no
        ~56-day dead wait). Keyless public endpoint.
        """
        now_ms = int(time.time() * 1000)
        end = end_ms if end_ms is not None else now_ms
        collected: list[list] = []
        remaining = min(limit, 2000)
        while remaining > 0 and end > 0:
            page = self._get(
                {
                    # ``category`` defaults to spot for the existing callers.
                    # A PERPETUALS book must pass "linear": many perps (e.g.
                    # 1000LUNCUSDT) have no spot pair at all, and warm-start
                    # dies with retCode=10001 'Not supported symbols'. The price
                    # history should come from the instrument actually traded.
                    "category": category,
                    "symbol": symbol.upper(),
                    "interval": "60",
                    "end": end,
                    "limit": min(remaining, _PAGE),
                }
            )
            bars = (page.get("result") or {}).get("list") or []
            if not bars:
                break
            bars = sorted(bars, key=lambda r: int(r[0]))
            collected = bars + collected
            oldest = int(bars[0][0])
            end = oldest - 1
            remaining -= len(bars)
            if len(bars) < _PAGE:
                break
            time.sleep(_REQUEST_INTERVAL_S)
        out: list[tuple[int, float, float]] = []
        for k in collected:
            # kline row: [startTime_ms, open, high, low, close, volume, turnover]
            start = int(k[0])
            out.append((start + 3_600_000, float(k[4]), float(k[5])))
        return out

    def fetch_bars(
        self,
        symbols: list[str],
        days: int,
        minutes: int | None = None,
        *,
        on_skip: Callable[[str, str], None] | None = None,
    ) -> pd.DataFrame:
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
            rows.extend(self._fetch_symbol(symbol.upper(), days, minutes, on_skip=on_skip))
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
