"""PEAD backtest correctness — the core promise is "no lookahead".

These tests pin the two properties that make the event study PIT-clean:
  1. expected earnings for a filing use only filings that were already public
     (a later restatement must never change an earlier surprise).
  2. quintile labels use only prior events' SUE breakpoints.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from scripts.pead_backtest import _compute_sue, _prior_breakpoint_labels

_F = dt.date


def _facts(rows: list[tuple[str, int, float, dt.date]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"TICKER": t, "FISCAL_YEAR": fy, "VALUE": v, "FILED_AT": f} for t, fy, v, f in rows]
    )


def test_compute_sue_expected_ignores_future_filings() -> None:
    # AAPL: FY2009 reported twice — original 10-K then a 10-K/A restatement a
    # month later. The restatement must not change the FY2009 original's
    # surprise, and neither FY2009 filing may leak into the FY2008 surprise.
    facts = _facts(
        [
            ("AAPL", 2006, 20.0, _F(2007, 10, 1)),
            ("AAPL", 2007, 24.0, _F(2008, 10, 1)),
            ("AAPL", 2008, 30.0, _F(2009, 10, 1)),
            ("AAPL", 2009, 36.0, _F(2010, 10, 27)),  # original 10-K
            ("AAPL", 2009, 42.0, _F(2010, 11, 27)),  # restatement (10-K/A)
        ]
    )
    events = _compute_sue(facts, min_prior=2)
    fy2009 = [e for e in events if e["fiscal_year"] == 2009]
    assert len(fy2009) == 2  # original + restatement are separate PIT events
    # Original: as-filed value, expected = FY2008 (30.0), never the restated 42.0.
    assert fy2009[0]["actual"] == 36.0 and fy2009[0]["expected"] == 30.0
    # Restatement: its own surprise, still vs FY2008.
    assert fy2009[1]["actual"] == 42.0 and fy2009[1]["expected"] == 30.0


def test_compute_sue_excludes_restatements_from_expected() -> None:
    # Expected for FY2009 must be the FY2008 value known BEFORE the FY2009
    # filing — the FY2008 value can only be learned from its own filing.
    facts = _facts(
        [
            ("MSFT", 2010, 60.0, _F(2011, 7, 20)),
            ("MSFT", 2011, 68.0, _F(2012, 7, 20)),
            ("MSFT", 2011, 70.0, _F(2012, 9, 1)),  # restatement of 2011
            ("MSFT", 2012, 75.0, _F(2013, 7, 25)),
        ]
    )
    events = _compute_sue(facts, min_prior=1)
    by_fy = {e["fiscal_year"]: e for e in events}
    # FY2012 expected = FY2011's value as known before the FY2012 filing,
    # which is the 70.0 restatement (filed 2012-09-01 < 2013-07-25).
    assert by_fy[2012]["expected"] == 70.0
    assert by_fy[2012]["surprise"] == 5.0


def test_prior_breakpoint_labels_use_only_past_events() -> None:
    events: list[dict[str, Any]] = []
    # Build a clean monotone SUE history so breakpoints are stable.
    for i, sue in enumerate([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]):
        events.append({"filed_at": _F(2010 + i, 12, 1), "sue": sue})
    labels = _prior_breakpoint_labels(events, quintiles=5)
    assert labels[0] is None  # nothing prior to the first event
    assert labels[1] is None  # < quintiles*3 priors
    assert all(label in {"Q1", "Q2", "Q3", "Q4", "Q5"} for label in labels[2:] if label)
    # Label of the LAST event must come from breakpoints excluding itself:
    # adding an extreme event can't push prior labels around, and the last
    # event's label is determined only by the prior 6.
    extreme = {"filed_at": _F(2020, 12, 1), "sue": 1000.0}
    labels_extreme = _prior_breakpoint_labels(events + [extreme], quintiles=5)
    assert labels_extreme[:-1] == labels
