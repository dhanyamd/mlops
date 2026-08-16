"""SRP live book -- assembles live frames and scores them through the shared module.

This is the ONLY place the live path differs from research, and it differs in
data plumbing only: it seeds history from the research caches, appends fresh
observations from the live feeds, and hands the resulting frames to
``scripts.srp_strategy``. The strategy logic itself is never reimplemented here.

WHY A SEED IS REQUIRED
----------------------
SRP ranks every factor against the symbol's OWN trailing 52 weeks. The live
REST surfaces cannot supply that:

    positioning (openInterestHist etc.)  ~30 days
    klines                               plenty, but 52w x 112 symbols is a slow pull

So history comes from the research caches written by
``scripts/backfill_positioning.py`` and ``scripts/backfill_intraday_features.py``,
and the live feeds extend it. Both feeds were verified to agree with those
caches exactly (0.000e+00 on every field) once partial days are excluded, so the
join is seamless rather than a splice of two different measurements.

FAIL CLOSED
-----------
A symbol missing any required input is EXCLUDED from the cross-section, not
imputed. If too few symbols survive, the book returns all-FLAT. A missed trade
costs nothing; a fabricated one costs money.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.srp_strategy import (
    SRPConfig,
    build_factors,
    directions,
    factor_book_weights,
    factor_scores,
    srp_weights,
)

logger = logging.getLogger(__name__)

_DAY_MS = 86_400_000
_INTRADAY_FIELDS = ("q", "rsj", "ofi", "cpv")
# Ticket-shape factors need 5-MINUTE bars: at 1h a day splits into ~12 bars per
# side, below the 20-ticket minimum the shape estimator requires (measured: 0%
# computable at 1h, 100% at 5m). They are therefore seeded from a separate 5m
# cache while the other four keep using 1h, where they measured BETTER
# (Sharpe 2.60 vs 2.37).
_TICKET_FIELDS = ("tskew_dir", "tku")


def _load_intraday_cache(path: str, symbols: set[str],
                         fields: tuple[str, ...] = _INTRADAY_FIELDS) -> dict[str, pd.DataFrame]:
    out: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    p = Path(path)
    if not p.is_dir():
        return {}
    for f_ in sorted(p.glob("*.json")):
        if f_.stem not in symbols:
            continue
        try:
            recs = json.loads(f_.read_text())
        except Exception:
            continue
        if not recs:
            continue
        idx = pd.to_datetime([r["t"] for r in recs], unit="ms", utc=True)
        for fld in fields:
            s = pd.Series([r.get(fld) for r in recs], index=idx, dtype=float)
            if s.notna().sum() > 20:
                out[fld][f_.stem] = s
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items() if v}


def _load_positioning_cache(path: str, symbols: set[str]) -> dict[str, pd.DataFrame]:
    keys = ("sum_open_interest", "sum_toptrader_long_short_ratio_mean",
            "count_long_short_ratio_mean")
    out: dict[str, dict[str, pd.Series]] = {k: {} for k in keys}
    p = Path(path)
    if not p.is_dir():
        return {}
    for f_ in sorted(p.glob("*.json")):
        if f_.stem not in symbols:
            continue
        try:
            recs = json.loads(f_.read_text())
        except Exception:
            continue
        if len(recs) < 20:
            continue
        idx = pd.to_datetime([r["day"] for r in recs], utc=True)
        for k in keys:
            s = pd.Series([r.get(k) for r in recs], index=idx, dtype=float)
            if s.notna().sum() > 10:
                out[k][f_.stem] = s
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items() if v}


def _append(base: pd.DataFrame, sym: str, pairs: list[tuple[int, float]]) -> pd.DataFrame:
    """Merge live observations into a cached frame. Live wins on collision."""
    if not pairs:
        return base
    idx = pd.to_datetime([t for t, _ in pairs], unit="ms", utc=True)
    s = pd.Series([v for _, v in pairs], index=idx, dtype=float)
    if sym not in base.columns:
        base = base.reindex(columns=list(base.columns) + [sym])
    col = base[sym].reindex(base.index.union(s.index))
    col.loc[s.index] = s.values
    base = base.reindex(index=col.index)
    base[sym] = col
    return base.sort_index()


class SRPBook:
    """Builds live frames and produces LONG/SHORT/FLAT per symbol."""

    def __init__(
        self,
        universe: list[str],
        *,
        intraday_cache: str = "/tmp/quant_cache/intraday_1h",
        ticket_cache: str = "/tmp/quant_cache/intraday3",
        positioning_cache: str = "/tmp/quant_cache/positioning_daily",
        cfg: SRPConfig | None = None,
        min_symbols: int = 20,
    ) -> None:
        self.universe = [s.upper() for s in universe]
        self.cfg = cfg or SRPConfig()
        self.min_symbols = min_symbols
        syms = set(self.universe)
        self.intraday = _load_intraday_cache(intraday_cache, syms)
        self.ticket = _load_intraday_cache(ticket_cache, syms, _TICKET_FIELDS)
        self.positioning = _load_positioning_cache(positioning_cache, syms)
        logger.info(
            "SRPBook seeded: intraday %s, positioning %s",
            {k: v.shape for k, v in self.intraday.items()},
            {k: v.shape for k, v in self.positioning.items()},
        )

    # ---- live ingestion -------------------------------------------------
    def update_positioning(self, live: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
        m = {"open_interest": "sum_open_interest",
             "top_ls": "sum_toptrader_long_short_ratio_mean",
             "all_ls": "count_long_short_ratio_mean"}
        for sym, rec in (live or {}).items():
            for src, dst in m.items():
                if dst not in self.positioning:
                    self.positioning[dst] = pd.DataFrame()
                self.positioning[dst] = _append(self.positioning[dst], sym, rec.get(src, []))

    def update_intraday(self, live: dict[str, dict[int, dict]]) -> None:
        for sym, days in (live or {}).items():
            for fld in _INTRADAY_FIELDS:
                pairs = [(d * _DAY_MS, rec.get(fld)) for d, rec in sorted(days.items())
                         if rec.get(fld) is not None]
                if not pairs:
                    continue
                if fld not in self.intraday:
                    self.intraday[fld] = pd.DataFrame()
                self.intraday[fld] = _append(self.intraday[fld], sym, pairs)

    # ---- scoring --------------------------------------------------------
    def score(
        self,
        weekly_close: pd.DataFrame,
        weekly_volume: pd.DataFrame,
        weekly_funding: pd.DataFrame,
        prev_books: dict[str, pd.Series] | None = None,
    ) -> tuple[dict[str, str], dict[str, pd.Series], pd.Series | None]:
        """-> (directions, books to carry forward, portfolio weights)."""
        need_pos = ("sum_open_interest", "sum_toptrader_long_short_ratio_mean",
                    "count_long_short_ratio_mean")
        if not all(k in self.positioning and not self.positioning[k].empty for k in need_pos):
            logger.warning("SRP: positioning frames missing -> FLAT")
            return {s: "FLAT" for s in self.universe}, {}, None
        if not all(f in self.intraday and not self.intraday[f].empty for f in _INTRADAY_FIELDS):
            logger.warning("SRP: intraday frames missing -> FLAT")
            return {s: "FLAT" for s in self.universe}, {}, None

        cols = [
            s for s in self.universe
            if s in weekly_close.columns
            and s in self.positioning["sum_open_interest"].columns
            and s in self.intraday["cpv"].columns
        ]
        if len(cols) < self.min_symbols:
            logger.warning("SRP: only %d symbols with full data -> FLAT", len(cols))
            return {s: "FLAT" for s in self.universe}, {}, None

        grid = weekly_close.index

        def G(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(columns=cols).reindex(df.index.union(grid)).ffill().reindex(grid)

        raw = build_factors(
            weekly_close=weekly_close[cols],
            weekly_volume=weekly_volume[cols],
            intraday={f: G(self.intraday[f]) for f in _INTRADAY_FIELDS},
            open_interest=G(self.positioning["sum_open_interest"]),
            top_ls=G(self.positioning["sum_toptrader_long_short_ratio_mean"]),
            all_ls=G(self.positioning["count_long_short_ratio_mean"]),
            ticket={f: G(self.ticket[f]) for f in _TICKET_FIELDS
                    if f in self.ticket and not self.ticket[f].empty} or None,
        )
        S = factor_scores(raw, self.cfg)
        fund = weekly_funding.reindex(index=grid, columns=cols).fillna(0.0)

        # Per-factor book returns drive the risk-parity weights. Rebuilt from the
        # same point-in-time frames rather than cached, so a restart cannot
        # resume with stale weights.
        fwd = (weekly_close[cols].shift(-1) / weekly_close[cols] - 1.0).clip(upper=1.0)
        ffwd = fund.shift(-1)
        BR: dict[str, pd.Series] = {}
        for name, sc in S.items():
            rets, ridx, prev = [], [], None
            for w in sc.index:
                if w not in fwd.index:
                    continue
                t = factor_book_weights(sc, fund, self.cfg, prev, w, cols)
                if t is None:
                    prev = None
                    continue
                r = float((t * fwd.loc[w]).reindex(cols).sum(skipna=True)) - float(
                    (t * ffwd.loc[w]).reindex(cols).sum(skipna=True))
                rets.append(r)
                ridx.append(w)
                prev = t
            BR[name] = pd.Series(rets, index=ridx)
        # dropna(how="any") would require ALL nine factors to have a return in
        # the same week, and the intersection is set by the SHORTEST factor.
        # AVOL is shortest by construction (12w volume sum, then a 52w
        # self-referential rank consumes 64 of the live registry's ~112 weeks),
        # so an intersection join collapsed 87 usable weeks to 20 -- below the
        # 26 the risk model needs, leaving the book permanently FLAT. Research
        # never hit this because 363 weeks of history hid it.
        #
        # Each factor's inverse-vol weight is estimated from ITS OWN column, so
        # a missing value elsewhere is not a reason to discard the row.
        # risk_parity_weights renormalises across whatever is available.
        br = pd.DataFrame(BR).dropna(how="all")
        if br.empty:
            logger.warning("SRP: no book-return history -> FLAT")
            return {s: "FLAT" for s in self.universe}, {}, None

        stamp = br.index[-1]
        port, books = srp_weights(S, fund, br, cols, stamp, prev_books, self.cfg)
        if port.abs().sum() == 0:
            logger.warning("SRP: risk model not warm at %s -> FLAT", stamp)
            return {s: "FLAT" for s in self.universe}, {}, None

        d = directions(port)
        out = {s: d.get(s, "FLAT") for s in self.universe}
        logger.info(
            "SRP scored %s: %d LONG, %d SHORT, %d FLAT (gross %.3f, net %+.2e)",
            stamp.date(), sum(1 for v in out.values() if v == "LONG"),
            sum(1 for v in out.values() if v == "SHORT"),
            sum(1 for v in out.values() if v == "FLAT"),
            float(port.abs().sum()), float(port.sum()),
        )
        return out, books, port
