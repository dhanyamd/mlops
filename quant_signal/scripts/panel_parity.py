"""Gate: the warehouse panel and the file panel are the same panel.

The reason this exists rather than a README sentence. Once research can read
Snowflake, there are two sources for the same input, and two sources that are
"basically the same" are worse than one source -- a result becomes ambiguous
about which panel produced it, and a discrepancy shows up as an unexplained
Sharpe drift months later rather than as a failure here.

So this asserts equality on the frames themselves, cell by cell, not on a
summary statistic. Shapes, index, columns, and every value. A single bar that
landed in one and not the other fails the gate.

    uv run python -m scripts.panel_parity
    uv run python -m scripts.panel_parity --as-of 2025-06-01
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config.logging import configure_logging
from config.settings import get_settings
from scripts.research_fas_clean import load as load_file
from scripts.warehouse_panel import load_from_snowflake

FRAMES = ("weekly_close", "weekly_volume", "weekly_funding", "daily_close", "daily_volume")


def compare(a: pd.DataFrame, b: pd.DataFrame, name: str) -> tuple[bool, str]:
    """Element-wise equality, tolerant only of float representation.

    NaNs must match position-for-position: a NaN in one panel and a number in
    the other means a bar is missing from a source, which is exactly the failure
    this is here to catch.
    """
    if a.shape != b.shape:
        return False, f"{name}: shape {a.shape} vs {b.shape}"

    cols_a, cols_b = sorted(a.columns), sorted(b.columns)
    if cols_a != cols_b:
        only_file = sorted(set(cols_a) - set(cols_b))[:5]
        only_wh = sorted(set(cols_b) - set(cols_a))[:5]
        return False, f"{name}: columns differ (file-only {only_file}, warehouse-only {only_wh})"

    a = a[cols_a].sort_index()
    b = b[cols_a].sort_index()
    if not a.index.equals(b.index):
        return False, f"{name}: index differs ({a.index.min()}..{a.index.max()} vs {b.index.min()}..{b.index.max()})"

    if not (a.isna() == b.isna()).all().all():
        n = int((a.isna() != b.isna()).sum().sum())
        return False, f"{name}: {n} cells present in one panel but not the other"

    delta = (a - b).abs().max().max()
    delta = 0.0 if pd.isna(delta) else float(delta)
    if delta > 1e-9:
        return False, f"{name}: max abs difference {delta:.3e}"
    return True, f"{name}: identical ({a.shape[0]}x{a.shape[1]}, max delta {delta:.1e})"


def _derived_checks(file_panel, wh_panel) -> list[tuple[str, str, bool]]:
    """Compare the objects the strategy is actually computed from.

    The liquidity mask is the traded universe; the forward return and forward
    funding frames are the two terms of the backtest's return. If these three
    match, any deterministic strategy over them returns the same series, which
    is the property the headline Sharpe depends on.
    """
    from scripts.research_fas_clean import _liquidity_mask

    (f_cw, f_vw, f_aw, *_), (w_cw, w_vw, w_aw, *_) = file_panel, wh_panel
    results: list[tuple[str, str, bool]] = []

    f_syms, w_syms = _liquidity_mask(f_cw, f_vw), _liquidity_mask(w_cw, w_vw)
    if set(f_syms) != set(w_syms):
        diff = sorted(set(f_syms) ^ set(w_syms))[:8]
        results.append(("liquidity_mask", f"{len(f_syms)} vs {len(w_syms)} symbols, differ on {diff}", False))
        return results

    # Membership is the invariant that matters; ORDER is not. The mask is built
    # by walking ``weekly_close.columns``, and that order comes from JSON key
    # order on the file path and from GROUP BY on the warehouse path. Every
    # downstream frame is reindexed onto this same list, and the operations over
    # it (cross-sectional rank, sum) are order-invariant, so a different order is
    # a different spelling of the same universe. Said out loud here rather than
    # silently sorted, because sorting would change the file path's column order
    # too and move a published number for no reason.
    order_note = "" if f_syms == w_syms else " (different column order, same set)"
    results.append(("liquidity_mask", f"same {len(f_syms)} tradable symbols{order_note}", True))

    # Deliberately use each source's OWN ordering below: ``compare`` aligns
    # columns by name, so if ordering did leak into the arithmetic this is where
    # it would show up as a non-zero delta.
    f_px, w_px = f_cw[f_syms], w_cw[w_syms]
    checks = {
        "forward_returns": (
            (f_px.shift(-1) / f_px - 1.0).clip(upper=1.0),
            (w_px.shift(-1) / w_px - 1.0).clip(upper=1.0),
        ),
        "forward_funding": (
            f_aw.reindex(index=f_cw.index, columns=f_syms).fillna(0.0).shift(-1),
            w_aw.reindex(index=w_cw.index, columns=w_syms).fillna(0.0).shift(-1),
        ),
    }
    for name, (a, b) in checks.items():
        ok, message = compare(a, b, name)
        results.append((name, message.split(": ", 1)[1], ok))
    return results


def main() -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--week-anchor", default="W-MON")
    args = ap.parse_args()

    cache = args.cache or get_settings().srp_weekly_cache
    file_panel = load_file(cache, args.week_anchor)
    wh_panel = load_from_snowflake(args.week_anchor, as_of=args.as_of)

    print(f"file      {cache}")
    print("warehouse QUANT.SILVER.SILVER_CRYPTO_{PANEL_BARS,FUNDING}")
    print(f"anchor    {args.week_anchor}   as_of {args.as_of or 'latest'}\n")

    failures = 0
    for name, f_df, w_df in zip(FRAMES, file_panel, wh_panel):
        ok, message = compare(f_df, w_df, name)
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
        failures += not ok

    # Raw frames matching is necessary but not sufficient. What the strategy
    # actually consumes is the tradable universe and the forward returns derived
    # from those frames, and a difference could in principle appear there --
    # the liquidity screen is a threshold, so two panels differing below the
    # float tolerance above could still select different symbols. Check the
    # derived objects too, because those are what a Sharpe is computed from.
    for name, message, ok in _derived_checks(file_panel, wh_panel):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {message}")
        failures += not ok

    print()
    if failures:
        print(f"PARITY FAILED on {failures} check(s)")
        return 1
    print("PARITY HOLDS on all 5 raw frames and all 3 derived objects —")
    print("the warehouse panel IS the research panel, so a backtest is")
    print("indifferent to which source it was loaded from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
