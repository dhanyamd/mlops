"""Deterministic synthetic OHLCV bars — the offline/test/demo source.

Not a toy: it exists so the *entire* platform (validation → quarantine →
Bronze → dbt → Silver → Gold) can be exercised end-to-end with zero external
dependencies and fully reproducible data. Same seed + symbols + days == same
bars, which makes tests deterministic and demos repeatable.

Output columns match the contract: ``symbol, ts, open, high, low, close,
volume, loaded_at`` (tz-naive UTC, compatible with Snowflake TIMESTAMP_NTZ).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ingest.providers.base import BarProvider

# 09:30-16:00 ET == 14:30-21:00 UTC during US daylight time. We operate in
# naive UTC and skip weekends, which is good enough for synthetic data.
_MARKET_OPEN = dt.time(14, 30)
_BARS_PER_DAY = 390  # 6.5 hours of minute bars


class SyntheticBarProvider(BarProvider):
    name = "synthetic"

    def __init__(self, seed: int = 42, base_prices: dict[str, float] | None = None) -> None:
        self._seed = seed
        self._base_prices = base_prices or {"AAPL": 150.0, "MSFT": 410.0, "NVDA": 120.0}

    def _rng_for(self, symbol: str) -> np.random.Generator:
        # Derive a per-symbol seed so re-fetching the same symbol is stable
        # regardless of call order or other symbols.
        salt = sum(ord(ch) for ch in symbol)
        return np.random.default_rng(self._seed + salt)

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
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        today = now.date()
        rows: list[tuple] = []
        for symbol in symbols:
            rng = self._rng_for(symbol)
            price = self._base_prices.get(symbol, 100.0)
            close = price
            for offset in range(days, 0, -1):
                day = today - dt.timedelta(days=offset)
                if day.weekday() >= 5:  # skip weekends
                    continue
                ts = dt.datetime.combine(day, _MARKET_OPEN)
                for _ in range(_BARS_PER_DAY):
                    open_ = close
                    close = open_ * (1.0 + float(rng.normal(0.0, 0.0008)))
                    high = max(open_, close) * (1.0 + abs(float(rng.normal(0.0, 0.0003))))
                    low = min(open_, close) * (1.0 - abs(float(rng.normal(0.0, 0.0003))))
                    volume = int(rng.integers(100, 10_000))
                    rows.append(
                        (
                            symbol,
                            ts,
                            "1Min",
                            round(open_, 4),
                            round(high, 4),
                            round(low, 4),
                            round(close, 4),
                            volume,
                            self.name,
                            now,
                        )
                    )
                    ts += dt.timedelta(minutes=1)
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
