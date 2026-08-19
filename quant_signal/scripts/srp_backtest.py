"""SRP backtest -- the committed, reusable evaluator.

Until now the headline number (annualised 2.40 net) came from an ad-hoc script
that was never committed. That is a reproducibility hole independent of whether
the number is right: a referee cannot re-run it, and neither can we after the
scratch file is gone. This module is that evaluator, in the repository, calling
``scripts.srp_strategy`` exactly the way the live path does.

It is also the engine the multiple-testing correction runs on. Every
configuration ``run()`` evaluates is written to ``scripts.trial_registry`` by
``scripts.srp_sweep``, so the N and sd that ``scripts.srp_dsr`` deflates by are
produced by executed code rather than recalled by a human.

RETURN CONSTRUCTION
-------------------
Identical to the gate in ``scripts/srp_parity.py``, which is the version whose
point-in-time behaviour is asserted:

    r_t = w_t . fwd_t                      price return of the held book
        - w_t . funding_{t+1}              perpetual funding actually paid
        - |w_t - w_{t-1}| . maker_bps      cost of getting there

Maker cost is scaled by liquidity: the cheapest quintile by dollar volume pays
1bp, the dearest 5bp, interpolated on the cross-sectional dollar-volume rank.
Forward returns are clipped at +100% so a single listing spike cannot manufacture
a Sharpe.

DATA IS LOADED ONCE
-------------------
``SRPData.load()`` is slow (112 symbols x 363 weeks x several sources). A sweep
runs hundreds of configs against ONE loaded panel, and the expensive
intermediates -- raw factors, self-referential scores -- are memoised on the
sub-config they actually depend on, so varying ``vol_window`` does not recompute
ranks that cannot have changed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.factor_core import ts_rank_pit, xs_rank
from scripts.research_fas_clean import _liquidity_mask, load
from scripts.research_intraday import load_intraday
from scripts.srp_strategy import (
    FACTORS,
    TICKET_FACTORS,
    SRPConfig,
    build_factors,
    factor_book_weights,
    srp_weights,
)

CACHE = "/tmp/quant_cache"
ALL_FACTORS = tuple(FACTORS) + tuple(TICKET_FACTORS)

# The three construction choices, as INDEPENDENT switches. SRP is
# (self, books, riskparity); the conventional pipeline every factor paper uses
# is (cross, blend, equal). Making them orthogonal rather than a single
# srp/conventional flag means the sweep decomposes WHICH choice earns the gap,
# and it lets the baseline be tuned over the same hyperparameter grid as the
# strategy -- without that, "1.03 vs 2.40" compares a tuned method to an untuned
# one and a referee is right to throw it out.
RANKINGS = ("self", "cross", "cross_z")  # own window / peers / peers-after-own-standardisation
COMBINES = ("books", "blend")     # one book per factor vs one blended score
WEIGHTINGS = ("riskparity", "equal")


def _load_positioning(d: str) -> dict[str, pd.DataFrame]:
    P: dict[str, dict] = {}
    for p in sorted(Path(d).glob("*.json")):
        try:
            recs = json.loads(p.read_text())
        except Exception:
            continue
        if len(recs) < 50:
            continue
        idx = pd.to_datetime([x["day"] for x in recs], utc=True)
        for f in ("sum_open_interest", "sum_toptrader_long_short_ratio_mean",
                  "count_long_short_ratio_mean"):
            s = pd.Series([x.get(f) for x in recs], index=idx, dtype=float)
            if s.notna().sum() > 40:
                P.setdefault(f, {})[p.stem] = s
    return {k: pd.DataFrame(v).sort_index() for k, v in P.items()}


def _load_ticket(d: str, cols: set[str]) -> dict[str, pd.DataFrame]:
    """5-minute-derived ticket shape (TSKD/TKU inputs). Absent at 1h by design."""
    out: dict[str, dict] = {}
    p = Path(d)
    if not p.is_dir():
        return {}
    for f_ in sorted(p.glob("*.json")):
        if f_.stem not in cols:
            continue
        try:
            recs = json.loads(f_.read_text())
        except Exception:
            continue
        if not recs:
            continue
        idx = pd.to_datetime([r["t"] for r in recs], unit="ms", utc=True)
        for fld in ("tskew_dir", "tku"):
            s = pd.Series([r.get(fld) for r in recs], index=idx, dtype=float)
            if s.notna().sum() > 20:
                out.setdefault(fld, {})[f_.stem] = s
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items() if v}


@dataclass
class SRPData:
    """The loaded panel. Immutable across a sweep."""

    cw: pd.DataFrame          # weekly close
    vw: pd.DataFrame          # weekly base volume
    fund: pd.DataFrame        # weekly funding, aligned
    cols: list[str]
    intraday: dict[str, pd.DataFrame]
    ticket: dict[str, pd.DataFrame]
    open_interest: pd.DataFrame
    top_ls: pd.DataFrame
    all_ls: pd.DataFrame
    maker_bps: pd.DataFrame
    fwd: pd.DataFrame
    fwd_funding: pd.DataFrame
    _raw: dict = field(default_factory=dict, repr=False)
    _scores: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(
        cls,
        cache: str = f"{CACHE}/fas_broad.json",
        intraday: str = f"{CACHE}/intraday_1h",
        positioning: str = f"{CACHE}/positioning_daily",
        ticket: str = f"{CACHE}/intraday3",
        week_anchor: str = "W-MON",
        source: str = "file",
        as_of: str | None = None,
    ) -> "SRPData":
        # ``week_anchor`` rebuilds the whole panel on a different rebalance
        # weekday; the daily positioning/intraday sources are reindexed onto
        # whatever grid results, so the seven anchors are genuinely independent
        # schedules rather than a shifted view of one. Used by srp_walkforward's
        # timing-luck test.
        #
        # ``source`` picks where the price/volume/funding panel comes from. The
        # two are asserted identical by ``scripts.panel_parity``, so this is a
        # choice of transport, not of data: "file" reads the local cache, and
        # "snowflake" reads the warehouse, which is what makes a run reproducible
        # by someone who does not have this laptop's /tmp. ``as_of`` truncates
        # warehouse history so a historical study re-runs on the panel it was
        # actually decided on rather than on everything landed since.
        if source == "snowflake":
            from scripts.warehouse_panel import load_from_snowflake

            cw, vw, aw, _dcl, _dvl = load_from_snowflake(week_anchor, as_of=as_of)
        elif source == "file":
            cw, vw, aw, _dcl, _dvl = load(cache, week_anchor)
        else:
            raise ValueError(f"unknown panel source {source!r} (file|snowflake)")
        sym = _liquidity_mask(cw, vw)
        fr = load_intraday(intraday)
        P = _load_positioning(positioning)
        cols = [c for c in sym
                if c in P["sum_open_interest"].columns and c in fr["cpv"].columns]

        def G(d: pd.DataFrame) -> pd.DataFrame:
            keep = [c for c in cols if c in d.columns]
            return (d[keep].reindex(d.index.union(cw.index)).ffill()
                    .reindex(cw.index).reindex(columns=cols))

        px = cw[cols]
        usd = (vw[cols] * px).replace(0, np.nan)
        rk = usd.rank(axis=1, pct=True)
        fundw = aw.reindex(index=cw.index, columns=cols).fillna(0.0)
        return cls(
            cw=px,
            vw=vw[cols],
            fund=fundw,
            cols=cols,
            intraday={k: G(fr[k]) for k in ("q", "rsj", "ofi", "cpv")},
            ticket={k: G(v) for k, v in _load_ticket(ticket, set(cols)).items()},
            open_interest=G(P["sum_open_interest"]),
            top_ls=G(P["sum_toptrader_long_short_ratio_mean"]),
            all_ls=G(P["count_long_short_ratio_mean"]),
            # cheapest quintile 1bp -> dearest 5bp, on the dollar-volume rank
            maker_bps=1.0 + 4.0 * (1.0 - rk),
            fwd=(px.shift(-1) / px - 1.0).clip(upper=1.0),
            fwd_funding=fundw.shift(-1),
        )

    def subset(self, symbols: list[str]) -> "SRPData":
        """A copy restricted to ``symbols`` -- for held-out-universe testing.

        The cross-section is rebuilt from scratch rather than masked: ranks,
        quintile boundaries and the funding tilt are all computed WITHIN the
        cross-section, so a book restricted to 32 symbols must re-rank those 32
        against each other. Masking a 112-symbol book would leak the excluded
        symbols into every rank and defeat the purpose of the split.

        Memoised intermediates are deliberately NOT carried over.
        """
        keep = [s for s in symbols if s in self.cols]
        if len(keep) < 10:
            raise ValueError(f"subset too small: {len(keep)} symbols")

        def C(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(columns=keep)

        px = C(self.cw)
        # maker cost is a CROSS-SECTIONAL dollar-volume rank, so it must be
        # recomputed within the subset, not sliced from the full universe.
        usd = (C(self.vw) * px).replace(0, np.nan)
        rk = usd.rank(axis=1, pct=True)
        fundw = C(self.fund)
        return SRPData(
            cw=px,
            vw=C(self.vw),
            fund=fundw,
            cols=keep,
            intraday={k: C(v) for k, v in self.intraday.items()},
            ticket={k: C(v) for k, v in self.ticket.items()},
            open_interest=C(self.open_interest),
            top_ls=C(self.top_ls),
            all_ls=C(self.all_ls),
            maker_bps=1.0 + 4.0 * (1.0 - rk),
            fwd=(px.shift(-1) / px - 1.0).clip(upper=1.0),
            fwd_funding=fundw.shift(-1),
        )

    # -- memoised intermediates -------------------------------------------
    def raw(self, smooth: int, use_ticket: bool) -> dict[str, pd.DataFrame]:
        key = (smooth, use_ticket)
        if key not in self._raw:
            self._raw[key] = build_factors(
                weekly_close=self.cw,
                weekly_volume=self.vw,
                intraday=self.intraday,
                open_interest=self.open_interest,
                top_ls=self.top_ls,
                all_ls=self.all_ls,
                smooth=smooth,
                ticket=self.ticket if (use_ticket and self.ticket) else None,
            )
        return self._raw[key]

    def scores(self, cfg: SRPConfig, smooth: int, use_ticket: bool,
               ranking: str = "self") -> dict[str, pd.DataFrame]:
        """Factor scores under the chosen ranking.

        "self"  -- each symbol against its OWN trailing window (point-in-time).
        "cross" -- each symbol against its peers that week, the conventional
                   pipeline. Both are point-in-time; ``xs_rank`` uses only the
                   current row, ``ts_rank_pit`` only trailing rows.
        """
        key = (smooth, use_ticket, cfg.rank_window, cfg.rank_min_periods, ranking)
        if key not in self._scores:
            raw = self.raw(smooth, use_ticket)
            if ranking == "cross":
                self._scores[key] = {k: xs_rank(v) for k, v in raw.items()}
            elif ranking == "cross_z":
                # MECHANISM TEST. Cross-sectional ranking compares raw values
                # across assets whose natural scales differ by orders of
                # magnitude (BTC's order flow vs a small alt's), so part of what
                # it ranks is asset identity rather than signal. Standardising
                # each asset against its OWN trailing window first removes that,
                # then ranks cross-sectionally. If this recovers the
                # self-referential edge, scale heterogeneity IS the mechanism;
                # if it does not, something else drives it. Rolling and trailing,
                # so it stays point-in-time.
                out = {}
                for k, v in raw.items():
                    r = v.rolling(cfg.rank_window, min_periods=cfg.rank_min_periods)
                    z = (v - r.mean()) / r.std().replace(0, np.nan)
                    out[k] = xs_rank(z)
                self._scores[key] = out
            else:
                self._scores[key] = {
                    k: ts_rank_pit(v, window=cfg.rank_window,
                                   min_periods=cfg.rank_min_periods)
                    for k, v in raw.items()
                }
        return self._scores[key]


@dataclass
class SRPResult:
    returns: pd.Series          # weekly NET returns of the combined book
    sharpe_weekly: float
    sharpe_ann: float
    n_obs: int
    gross_mean: float
    turnover_mean: float
    active_rebalances: int
    total_rebalances: int


def _book_returns(data: SRPData, S: dict[str, pd.DataFrame], cfg: SRPConfig,
                  costs: bool) -> pd.DataFrame:
    """Per-factor book returns -- the input the risk model weights by."""
    BR: dict[str, pd.Series] = {}
    for name, sc in S.items():
        rets, ridx, prev = [], [], None
        for w in sc.index:
            if w not in data.fwd.index:
                continue
            t = factor_book_weights(sc, data.fund, cfg, prev, w, data.cols)
            if t is None:
                prev = None
                continue
            r = float((t * data.fwd.loc[w]).reindex(data.cols).sum(skipna=True))
            r -= float((t * data.fwd_funding.loc[w]).reindex(data.cols).sum(skipna=True))
            if costs and prev is not None:
                r -= float(((t - prev).abs()
                            * data.maker_bps.loc[w].reindex(data.cols).fillna(5.0)
                            / 1e4).sum())
            rets.append(r)
            ridx.append(w)
            prev = t
        if rets:
            BR[name] = pd.Series(rets, index=ridx)
    if not BR:
        return pd.DataFrame()
    # dropna(how="all"): an intersection join is set by the SHORTEST factor
    # (AVOL, whose 12w volume sum plus a 52w rank consumes ~64 weeks) and
    # discards rows every other factor could have traded. risk_parity_weights
    # renormalises over whatever is present per row.
    return pd.DataFrame(BR).dropna(how="all")


def run(
    data: SRPData,
    cfg: SRPConfig | None = None,
    *,
    smooth: int = 20,
    factors: tuple[str, ...] | None = None,
    costs: bool = True,
    ranking: str = "self",
    combine: str = "books",
    weighting: str = "riskparity",
) -> SRPResult:
    """Evaluate ONE configuration. Returns the weekly net return series.

    ``factors`` selects the subset of books to combine; None means all available
    (the ticket factors are included only when the 5m cache supplied them).
    ``ranking``/``combine``/``weighting`` select the construction -- see the
    module constants. SRP is ("self", "books", "riskparity"); the conventional
    pipeline is ("cross", "blend", "equal").
    """
    cfg = cfg or SRPConfig()
    if ranking not in RANKINGS or combine not in COMBINES or weighting not in WEIGHTINGS:
        raise ValueError(f"bad construction ({ranking}, {combine}, {weighting})")
    use_ticket = factors is None or any(f in TICKET_FACTORS for f in factors)
    S_all = data.scores(cfg, smooth, use_ticket, ranking)
    if factors is not None:
        S = {k: v for k, v in S_all.items() if k in factors}
    else:
        S = dict(S_all)
    if not S:
        return SRPResult(pd.Series(dtype=float), float("nan"), float("nan"),
                         0, 0.0, 0.0, 0, 0)

    if combine == "blend":
        # The conventional pipeline: average the scores into ONE composite and
        # trade a single quintile book off it. This is what discards the
        # factors' independence before a trade is placed -- nine near-orthogonal
        # signals are forced through one ranking.
        stack = pd.concat(S.values(), keys=list(S))
        S = {"COMPOSITE": stack.groupby(level=1).mean().reindex(data.cw.index)}

    # Equal weighting means the risk model is never consulted, so the
    # "stay FLAT until inverse-vol is estimable" guard must not fire either.
    if weighting == "equal":
        cfg = replace(cfg, require_risk_parity=False)

    BR = _book_returns(data, S, cfg, costs)
    if BR.empty:
        return SRPResult(pd.Series(dtype=float), float("nan"), float("nan"),
                         0, 0.0, 0.0, 0, len(BR))

    rets, ridx, grosses, turns = [], [], [], []
    prev_books: dict[str, pd.Series] | None = None
    prev_port: pd.Series | None = None
    for w in BR.index:
        # Equal weighting withholds the book-return history entirely, so
        # srp_weights falls through to 1/n rather than inverse vol.
        hist = None if weighting == "equal" else BR.loc[:w]
        port, books = srp_weights(S, data.fund, hist, data.cols, w, prev_books, cfg)
        prev_books = books or prev_books
        if port.abs().sum() == 0 or w not in data.fwd.index:
            continue
        r = float((port * data.fwd.loc[w]).reindex(data.cols).sum(skipna=True))
        r -= float((port * data.fwd_funding.loc[w]).reindex(data.cols).sum(skipna=True))
        if costs and prev_port is not None:
            d = (port - prev_port).abs()
            r -= float((d * data.maker_bps.loc[w].reindex(data.cols).fillna(5.0) / 1e4).sum())
            turns.append(float(d.sum()))
        grosses.append(float(port.abs().sum()))
        rets.append(r)
        ridx.append(w)
        prev_port = port

    s = pd.Series(rets, index=ridx)
    if len(s) < 2 or s.std() == 0:
        return SRPResult(s, float("nan"), float("nan"), len(s),
                         float(np.mean(grosses)) if grosses else 0.0,
                         float(np.mean(turns)) if turns else 0.0,
                         len(s), len(BR))
    sw = float(s.mean() / s.std())
    return SRPResult(
        returns=s,
        sharpe_weekly=sw,
        sharpe_ann=sw * math.sqrt(52),
        n_obs=len(s),
        gross_mean=float(np.mean(grosses)) if grosses else 0.0,
        turnover_mean=float(np.mean(turns)) if turns else 0.0,
        active_rebalances=len(s),
        total_rebalances=len(BR),
    )


def config_dict(cfg: SRPConfig, smooth: int, factors: tuple[str, ...] | None,
                costs: bool, ranking: str = "self", combine: str = "books",
                weighting: str = "riskparity") -> dict:
    """The canonical description of a trial, for the registry.

    Every knob that can change the result must appear here: two configs that
    hash equal but score differently would corrupt both N and the sd estimate.
    """
    d = asdict(cfg)
    d["smooth"] = smooth
    d["factors"] = sorted(factors) if factors is not None else sorted(ALL_FACTORS)
    d["costs"] = bool(costs)
    d["ranking"] = ranking
    d["combine"] = combine
    d["weighting"] = weighting
    return d


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="run the SRP backtest once")
    ap.add_argument("--cache", default=f"{CACHE}/fas_broad.json")
    ap.add_argument("--intraday", default=f"{CACHE}/intraday_1h")
    ap.add_argument("--positioning", default=f"{CACHE}/positioning_daily")
    ap.add_argument("--ticket", default=f"{CACHE}/intraday3")
    ap.add_argument("--no-costs", action="store_true")
    ap.add_argument("--factors", default=None,
                    help="comma-separated subset; default = all available")
    ap.add_argument("--source", choices=["file", "snowflake"], default="file",
                    help="where the price/volume/funding panel comes from; the "
                         "two are asserted identical by scripts.panel_parity")
    ap.add_argument("--as-of", default=None,
                    help="ISO date; truncate warehouse history (snowflake only)")
    a = ap.parse_args()

    data = SRPData.load(a.cache, a.intraday, a.positioning, a.ticket,
                        source=a.source, as_of=a.as_of)
    print(f"universe {len(data.cols)} symbols, {len(data.cw)} weeks, "
          f"panel from {a.source}, "
          f"ticket factors {'present' if data.ticket else 'ABSENT'}")
    facs = tuple(a.factors.split(",")) if a.factors else None
    res = run(data, SRPConfig(), factors=facs, costs=not a.no_costs)
    print(f"\n  weekly observations : {res.n_obs}")
    print(f"  weekly Sharpe       : {res.sharpe_weekly:.4f}")
    print(f"  annualised Sharpe   : {res.sharpe_ann:.3f}")
    print(f"  t-statistic         : {res.sharpe_weekly * math.sqrt(res.n_obs):.2f}")
    print(f"  mean gross          : {res.gross_mean:.3f}")
    print(f"  mean turnover       : {res.turnover_mean:.3f}")
    print(f"  active rebalances   : {res.active_rebalances}/{res.total_rebalances}")


if __name__ == "__main__":
    main()
