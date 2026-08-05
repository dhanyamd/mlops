"""Contract-first schema validation at the ingestion edge.

Every row must satisfy these Pydantic contracts before it may land in Bronze.
Rows that fail are routed to QUARANTINE with the validation reason attached
(never silently dropped, never allowed to reach Silver). This is the platform's
"clean, reliable data" guarantee: bad data is *visible*, not lost.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator, model_validator

# Bar aggregation supported by the platform. Providers land bars at exactly
# one of these; mixing frequencies without a discriminator is a data-quality
# trap (daily and minute volumes must never be summed together).
_TIMEFRAMES = ("1Min", "1D")


class EquityBar(BaseModel):
    """A single OHLCV bar (minute or daily). Field names MUST stay uppercase-safe:
    ingestion normalizes to UPPERCASE before landing, matching dbt's unquoted
    identifier folding."""

    symbol: str = Field(min_length=1, max_length=16, pattern=r"^[A-Z0-9.\-^]+$")
    ts: dt.datetime
    timeframe: str = Field(pattern=r"^(1Min|1D)$")
    open: float = Field(ge=0.0)
    high: float = Field(ge=0.0)
    low: float = Field(ge=0.0)
    close: float = Field(ge=0.0)
    # float (not int) because crypto providers report fractional base volume.
    volume: float = Field(ge=0.0)
    provider: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_]+$")
    loaded_at: dt.datetime

    @field_validator("symbol", mode="before")
    @classmethod
    def _uppercase_symbol(cls, value: object) -> object:
        return str(value).strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _ohlc_consistent(self) -> "EquityBar":
        # low <= open, close <= high. Any bar violating this is physically
        # impossible and must never reach Silver.
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("low must be <= open and close <= high")
        return self


class MacroObservation(BaseModel):
    """One observation of a FRED macro series (daily or monthly, real data).

    ``date`` is the observation date and ``value`` the published value; FRED
    uses "." for missing observations, which the provider drops (a gap, not a
    bad row)."""

    series_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Z0-9_]+$")
    date: dt.date
    value: float
    loaded_at: dt.datetime

    @field_validator("series_id", mode="before")
    @classmethod
    def _uppercase_series(cls, value: object) -> object:
        return str(value).strip().upper() if isinstance(value, str) else value


class CompanyFact(BaseModel):
    """One annual fundamental from SEC EDGAR company facts (us-gaap).

    ``filed_at`` is the SEC filing date — the point-in-time anchor. A value is
    only "known" once its filing lands, so as-of features must filter
    ``filed_at <= as_of``. Restatements (10-K/A) arrive on later dates and are
    kept as separate rows; ``(ticker, metric, fiscal_year, filed_at)`` is the
    natural key.
    """

    ticker: str = Field(min_length=1, max_length=16, pattern=r"^[A-Z0-9.\-^]+$")
    cik: str = Field(pattern=r"^\d{10}$")
    metric: str = Field(min_length=1, max_length=64)
    fiscal_year: int = Field(ge=2000, le=2100)
    value: float
    unit: str = "USD"
    filed_at: dt.date
    loaded_at: dt.datetime

    @field_validator("ticker", mode="before")
    @classmethod
    def _uppercase_ticker(cls, value: object) -> object:
        return str(value).strip().upper() if isinstance(value, str) else value
