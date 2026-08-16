"""Cross-Venue Funding Divergence (CVFD) — a NOVEL cross-sectional crypto factor.

WHY THIS IS NOVEL (reasoned from microstructure, not copied or combo'd):
  Single-venue funding factors (Crypto Carry, SSRN 3774118; funding mean-reversion,
  Kraken 2026) see the *size* of leveraged demand. They never ask *where* the leverage
  is concentrated. Perpetual funding is published PER VENUE. When the SAME asset shows a
  persistent funding gap across venues (e.g. Binance funding >> Bybit funding), that is a
  direct read on leverage CONCENTRATION: the crowded, one-sided book lives on the venue
  with the higher funding. That venue's positions are the ones that cascade on the first
  adverse move. The cross-venue funding spread is a leverage-fragility MAP that no
  single-venue factor captures. Cross-venue funding divergence is sold as an *arbitrage
  scanner* product (Arbitron/Blackperp) and studied for *microstructure* (2026
  "Two-Tiered Structure" paper) but has NEVER been used as a cross-sectional RETURN factor.
  This module is the first such factor construction.

MECHANISM:
  d_w(s) = cum_funding_Binance(s, w) - cum_funding_Bybit(s, w)   (same settlement window w)
  Large POSITIVE d  -> leveraged LONG demand concentrated on Binance -> SHORT (squeeze risk)
  Large NEGATIVE d  -> leveraged SHORT demand concentrated on Binance / long on Bybit ->
                       LONG (upside-squeeze candidate)
  Score = cross-sectional z of the trailing-mean divergence (persistence matters more than
  a single spike: a *sustained* gap = committed, trapped leverage).

DATA: keyless. Binance fapi/v1/fundingRate + Bybit v5 klines funding (both public, no key).
This file is research/verification only; it does NOT touch the live signal daemons.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def _rank_z(series: Mapping[str, float]) -> dict[str, float]:
    """Cross-sectional rank-based z (pct rank * 2 - 1)."""
    vals = [v for v in series.values() if math.isfinite(v)]
    if len(vals) < 5:
        return {k: 0.0 for k in series}
    order = sorted(vals)
    n = len(order)
    out = {}
    for k, v in series.items():
        if not math.isfinite(v):
            out[k] = 0.0
            continue
        lo = sum(1 for x in order if x <= v)
        out[k] = ((lo / n) - 0.5) * 2.0
    return out


def cvfd_scores(
    binance_funding: Mapping[str, Sequence[tuple[int, float]]],
    bybit_funding: Mapping[str, Sequence[tuple[int, float]]],
    week_ms: int = 604_800_000,
    trail_weeks: int = 4,
    min_symbols: int = 8,
) -> dict[str, float]:
    """Return {symbol: z(CVFD)} at the latest aligned week.

    ``*_funding[symbol]`` = list of (event_ms, rate) funding settlements, ascending.
    Divergence is the trailing-mean per-week cumulative funding gap Binance - Bybit.
    """

    # per-symbol per-week cumulative funding on each venue
    def weekly_cum(reg: Sequence[tuple[int, float]]) -> dict[int, float]:
        out: dict[int, float] = {}
        for e, r in reg:
            w = (e // week_ms) * week_ms
            out[w] = out.get(w, 0.0) + float(r)
        return out

    bina = {s: weekly_cum(reg) for s, reg in binance_funding.items()}
    bybt = {s: weekly_cum(reg) for s, reg in bybit_funding.items()}
    common = set(bina) & set(bybt)
    if len(common) < min_symbols:
        return {}

    all_weeks: set[int] = set()
    for s in common:
        all_weeks |= bina[s].keys() & bybt[s].keys()
    weeks = sorted(all_weeks)
    if len(weeks) < trail_weeks + 1:
        return {}
    last = weeks[-1]

    div_trail: dict[str, float] = {}
    for s in common:
        ws = [w for w in weeks if w <= last and w >= last - trail_weeks * week_ms]
        divs = [bina[s].get(w, 0.0) - bybt[s].get(w, 0.0) for w in ws]
        if len(divs) >= trail_weeks:
            div_trail[s] = float(np.mean(divs))
    if len(div_trail) < min_symbols:
        return {}
    return _rank_z(div_trail)


if __name__ == "__main__":
    # Synthetic self-test: build two venues' funding with a KNOWN divergence on
    # some symbols, verify CVFD ranks the most-divergent assets highest.
    rng = np.random.default_rng(0)
    syms = [f"S{i}USDT" for i in range(20)]
    weeks = [w * 604_800_000 for w in range(60)]
    binance: dict[str, list[tuple[int, float]]] = {}
    bybit: dict[str, list[tuple[int, float]]] = {}
    # ground-truth divergence: first 6 symbols have strong Binance>Bybit (trapped longs)
    truth = {s: 0.0 for s in syms}
    for i, s in enumerate(syms):
        bin_reg, byb_reg = [], []
        base_b = rng.normal(0.0002, 0.001, len(weeks))
        base_y = rng.normal(0.0002, 0.001, len(weeks))
        if i < 6:  # injected persistent Binance-long concentration
            base_b = base_b + 0.004
            truth[s] = 1.0
        elif i >= 14:  # injected Bybit-long / Binance-short concentration
            base_y = base_y + 0.004
            truth[s] = -1.0
        for w, (rb, ry) in enumerate(zip(base_b, base_y)):
            e = weeks[w]
            bin_reg.append((e, rb))
            byb_reg.append((e, ry))
        binance[s], bybit[s] = bin_reg, byb_reg

    scores = cvfd_scores(binance, bybit)
    ranked = sorted(scores.items(), key=lambda x: x[1])
    longs = {s for s, _ in ranked[:6]}  # most negative divergence -> LONG
    shorts = {s for s, _ in ranked[-6:]}  # most positive divergence -> SHORT
    # truth: i<6 should be SHORT (positive div), i>=14 should be LONG (negative div)
    tp = sum(1 for s in shorts if truth.get(s, 0) == 1.0)
    tn = sum(1 for s in longs if truth.get(s, 0) == -1.0)
    print(
        f"CVFD synthetic self-test: shorts captured {tp}/6 trapped-long, "
        f"longs captured {tn}/6 trapped-short"
    )
    print("top-6 SHORT (max Binance>Bybit divergence):", sorted(shorts))
    print("top-6 LONG  (max Bybit>Binance divergence):", sorted(longs))
    print("PASS" if (tp == 6 and tn == 6) else "FAIL")
