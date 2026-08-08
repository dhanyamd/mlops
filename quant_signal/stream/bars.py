"""Bar payload conversion shared by the producer and consumers.

Every Kafka payload carries both ``ts`` (epoch ms — Flink's event time) and
``ts_iso`` (the display/dedupe key in-memory consumers like the API hub use).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _num(value: Any) -> float | None:
    if value is None or value != value:  # None / NaN
        return None
    return float(value)


def df_to_bars(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Provider rows → JSON-safe bar payloads for the Kafka bus."""
    bars: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        ts = row.get("ts")
        if ts is None:
            continue
        try:
            stamp = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
        except (TypeError, ValueError):
            continue
        bars.append(
            {
                "symbol": str(row["symbol"]).upper(),
                "ts": int(stamp.value // 1_000_000),
                "ts_iso": stamp.isoformat(),
                "timeframe": str(row.get("timeframe") or "1Min"),
                "open": _num(row.get("open")),
                "high": _num(row.get("high")),
                "low": _num(row.get("low")),
                "close": _num(row.get("close")),
                "volume": _num(row.get("volume")),
                "provider": str(row.get("provider") or "unknown"),
            }
        )
    return bars
