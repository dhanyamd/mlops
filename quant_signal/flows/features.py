"""Rolling/as-of feature engineering over OHLCV bars — pure pandas.

The single source of truth for feature definitions used by the Spark batch
(``flows/feature_engineering.py``) and the no-Java pandas fallback path. Kept
dependency-free (pandas only) so the math is hermetic-unit-testable without a
Spark runtime.

Lookahead discipline: every feature at row ``ts`` is computed from the trailing
window *up to and including* that row — realized features, no future bars.
A label for supervised training would be a *forward* return added downstream.

Features
--------
log_return          ln(close[t] / close[t-1])
ret_1               close[t] / close[t-1] - 1
vol_20              rolling std of log_return (20 bars, min 5)
mom_20              close[t] / close[t-20] - 1
zscore_20           (close[t] - rolling mean) / rolling std (20 bars)
volume_zscore_20    (volume[t] - rolling mean) / rolling std (20 bars)

``None`` (null) where there is not enough history — never a fabricated 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "ret_1",
    "vol_20",
    "mom_20",
    "zscore_20",
    "volume_zscore_20",
]


def compute_features(
    bars: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int = 5,
) -> pd.DataFrame:
    """Compute rolling/as-of features per symbol from a bars DataFrame.

    Input contract: columns ``symbol, ts, close`` (optionally ``volume`` and
    any other bar fields, which pass through untouched). Rows may arrive in
    any order; the function sorts by ``(symbol, ts)``.
    """
    if bars.empty:
        return bars.copy()

    df = bars.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    groups = []
    for _, group in df.groupby("symbol", sort=False):
        g = group.sort_values("ts").reset_index(drop=True)
        close = g["close"].astype(float)
        log_return = np.log(close / close.shift(1))

        g["log_return"] = log_return
        g["ret_1"] = close / close.shift(1) - 1.0
        g["vol_20"] = log_return.rolling(window, min_periods=min_periods).std()
        g["mom_20"] = close / close.shift(window) - 1.0
        rolling_mean = close.rolling(window, min_periods=min_periods).mean()
        rolling_std = close.rolling(window, min_periods=min_periods).std()
        g["zscore_20"] = (close - rolling_mean) / rolling_std

        if "volume" in g.columns:
            volume = g["volume"].astype(float)
            v_mean = volume.rolling(window, min_periods=min_periods).mean()
            v_std = volume.rolling(window, min_periods=min_periods).std()
            g["volume_zscore_20"] = (volume - v_mean) / v_std
        else:
            g["volume_zscore_20"] = np.nan

        groups.append(g)

    return pd.concat(groups, ignore_index=True).sort_values(["symbol", "ts"]).reset_index(drop=True)
