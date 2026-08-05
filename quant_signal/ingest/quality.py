"""Quality gate between providers and Bronze.

Validates every row against the Pydantic contract and splits the batch into
(valid, invalid). Invalid rows are NOT dropped — they carry a ``reason`` column
and are written to QUARANTINE so a human can inspect and repair the source.
"""

from __future__ import annotations

import pandas as pd
from pydantic import ValidationError

from ingest.schemas import CompanyFact, EquityBar, MacroObservation


def _reason(exc: ValidationError) -> str:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first["loc"]) or "row"
    return f"{loc}: {first['msg']}"


def validate_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a bar frame into (valid, invalid-with-reason)."""
    good: list[dict] = []
    bad: list[dict] = []
    for row in df.to_dict("records"):
        try:
            good.append(EquityBar.model_validate(row).model_dump())
        except ValidationError as exc:  # fail loud, not silent
            bad.append({**row, "reason": _reason(exc)})
    return pd.DataFrame(good), pd.DataFrame(bad)


def validate_facts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a fundamentals frame into (valid, invalid-with-reason)."""
    good: list[dict] = []
    bad: list[dict] = []
    for row in df.to_dict("records"):
        try:
            good.append(CompanyFact.model_validate(row).model_dump())
        except ValidationError as exc:
            bad.append({**row, "reason": _reason(exc)})
    return pd.DataFrame(good), pd.DataFrame(bad)


def validate_macro(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a macro-observations frame into (valid, invalid-with-reason)."""
    good: list[dict] = []
    bad: list[dict] = []
    for row in df.to_dict("records"):
        try:
            good.append(MacroObservation.model_validate(row).model_dump())
        except ValidationError as exc:
            bad.append({**row, "reason": _reason(exc)})
    return pd.DataFrame(good), pd.DataFrame(bad)
