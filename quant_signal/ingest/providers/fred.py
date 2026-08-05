"""FRED macro provider — REAL St. Louis Fed data via the public CSV download.

The FRED JSON API needs a key, but the St. Louis Fed publishes every series
as a plain CSV at ``fredgraph.csv?id=SERIES`` (no auth). We verified live that
VIXCLS (1990-), CPIAUCSL (1947-), FEDFUNDS (1954-), UNRATE, DGS2 and DGS10 all
resolve. The CDN drops requests intermittently, so retries with backoff are
mandatory (observed ~50% transient HTTP 000 during verification).

Output columns: ``series_id, date, value, loaded_at``. FRED uses "." for
missing observations — those are dropped (a gap, not a bad row).
"""

from __future__ import annotations

import datetime as dt
import io
import time

import pandas as pd
import requests

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {"User-Agent": "quant-signal-research research@example.com"}
_REQUEST_INTERVAL_S = 0.5
_MAX_TRIES = 4


class FredProvider:
    """Daily/monthly macro observations from FRED (keyless)."""

    name = "fred"

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def _fetch_csv(self, series_id: str) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(_MAX_TRIES):
            try:
                resp = requests.get(
                    _FRED_CSV,
                    params={"id": series_id},
                    headers=_HEADERS,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                # First column is observation_date, second is the value (the
                # column header is the series id itself).
                df = pd.read_csv(io.StringIO(resp.text))
                if list(df.columns)[0] != "observation_date":
                    raise RuntimeError(f"unexpected FRED CSV shape for {series_id}")
                return df
            except Exception as exc:  # noqa: BLE001 - network/CDN flakiness
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"fred fetch failed for {series_id} after {_MAX_TRIES} tries: {last_exc}"
        )

    def fetch_observations(self, series_ids: list[str] | tuple[str, ...]) -> pd.DataFrame:
        """Latest observations per (series_id, date); missing values dropped."""
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        rows: list[tuple] = []
        for series_id in series_ids:
            sid = str(series_id).strip().upper()
            if not sid:
                continue
            df = self._fetch_csv(sid)
            value_col = df.columns[1]
            obs = (
                df.rename(columns={df.columns[0]: "date", value_col: "value"})
                .assign(
                    date=lambda d: pd.to_datetime(d["date"]).dt.date,
                    value=lambda d: pd.to_numeric(d["value"], errors="coerce"),
                )
                .dropna(subset=["value"])
            )
            for date_, value in zip(obs["date"], obs["value"]):
                rows.append((sid, date_, float(value), now))
            time.sleep(_REQUEST_INTERVAL_S)
        return pd.DataFrame(rows, columns=["series_id", "date", "value", "loaded_at"])
