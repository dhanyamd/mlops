"""CI-runnable version of the SRP deploy gate (``scripts/srp_parity.py``).

The full gate asserts against the real 27MB research cache, which is not in the
repository, so CI cannot run it. The assertions it makes are structural rather
than statistical, though: point-in-time integrity, determinism, dollar
neutrality, bounded gross and non-emptiness are properties of the *code*, and a
synthetic panel exercises every one of them. What CI cannot check is that the
real data still reproduces the paper's numbers; that stays in the full gate.

This exists because the failure it guards against has happened twice in this
project: a streaming reimplementation that drifted to 0.147 rank correlation
against research, and a look-ahead bug that survived because research and live
called the same wrong helper. Both were invisible to a test suite that never
compared the two paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.factor_core as fc
from scripts.srp_strategy import (
    SRPConfig,
    build_factors,
    factor_book_weights,
    factor_scores,
    srp_weights,
)

WEEKS = 160
SYMBOLS = 24
SEED = 20260817


def _panel(rng: np.random.Generator, index: pd.DatetimeIndex,
           cols: list[str], *, base: float, scale: float) -> pd.DataFrame:
    """A positive, smoothly varying panel — the shape every input arrives in."""
    walk = rng.normal(0.0, scale, size=(len(index), len(cols))).cumsum(axis=0)
    return pd.DataFrame(base * np.exp(walk / base), index=index, columns=cols)


@pytest.fixture(scope="module")
def synthetic() -> dict:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2021-01-04", periods=WEEKS, freq="W-MON", tz="UTC")
    cols = [f"SYM{i:02d}USDT" for i in range(SYMBOLS)]

    close = _panel(rng, idx, cols, base=100.0, scale=6.0)
    volume = _panel(rng, idx, cols, base=1e6, scale=4e4)
    intraday = {
        k: pd.DataFrame(rng.normal(0.0, 1.0, (WEEKS, SYMBOLS)), index=idx, columns=cols)
        for k in ("q", "rsj", "ofi", "cpv")
    }
    raw = build_factors(
        weekly_close=close,
        weekly_volume=volume,
        intraday=intraday,
        open_interest=_panel(rng, idx, cols, base=5e6, scale=2e5),
        top_ls=pd.DataFrame(rng.uniform(0.6, 1.8, (WEEKS, SYMBOLS)), index=idx, columns=cols),
        all_ls=pd.DataFrame(rng.uniform(0.6, 1.8, (WEEKS, SYMBOLS)), index=idx, columns=cols),
    )
    cfg = SRPConfig()
    scores = factor_scores(raw, cfg)
    funding = pd.DataFrame(
        rng.normal(0.0, 3e-4, (WEEKS, SYMBOLS)), index=idx, columns=cols
    )
    return {"raw": raw, "scores": scores, "funding": funding,
            "cols": cols, "cfg": cfg, "close": close}


def test_factors_are_point_in_time(synthetic):
    """Recomputing a score with future rows deleted must reproduce it exactly.

    This is the assertion that would have caught the shared-helper look-ahead
    bug: both callers agreed with each other, and neither was compared to a
    truncated reference.
    """
    cfg = synthetic["cfg"]
    for name, df in synthetic["raw"].items():
        leak = fc.leak_test(
            lambda d: fc.ts_rank_pit(
                d, window=cfg.rank_window, min_periods=cfg.rank_min_periods
            ),
            df.dropna(how="all"),
            at=60,
            truncate=100,
        )
        assert not np.isfinite(leak) or abs(leak) < 1e-9, f"{name} leaks {leak:.2e}"


def _book_returns(synthetic) -> pd.DataFrame:
    scores, cols, cfg = synthetic["scores"], synthetic["cols"], synthetic["cfg"]
    funding, close = synthetic["funding"], synthetic["close"]
    fwd = (close.shift(-1) / close - 1.0).clip(upper=1.0)
    out: dict[str, pd.Series] = {}
    for name, sc in scores.items():
        rets, ridx, prev = [], [], None
        for w in sc.index:
            if w not in fwd.index:
                continue
            tgt = factor_book_weights(sc, funding, cfg, prev, w, cols)
            if tgt is None:
                prev = None
                continue
            rets.append(float((tgt * fwd.loc[w]).reindex(cols).sum(skipna=True)))
            ridx.append(w)
            prev = tgt
        out[name] = pd.Series(rets, index=ridx)
    # dropna(how="all"), never a plain dropna(): an intersection join lets the
    # shortest factor silently truncate the evaluation sample.
    return pd.DataFrame(out).dropna(how="all")


def test_books_are_non_empty(synthetic):
    """A config that returns FLAT everywhere is a failure, not a pass."""
    br = _book_returns(synthetic)
    assert len(br) > 50, f"only {len(br)} rebalances produced a book"
    assert br.notna().any().any(), "every factor book is empty"


def test_weights_are_deterministic(synthetic):
    br = _book_returns(synthetic)
    stamp = br.index[-1]
    args = (synthetic["scores"], synthetic["funding"], br.loc[:stamp],
            synthetic["cols"], stamp, None, synthetic["cfg"])
    first, _ = srp_weights(*args)
    second, _ = srp_weights(*args)
    assert float((first - second).abs().max()) == 0.0


def test_book_is_dollar_neutral_and_bounded(synthetic):
    br = _book_returns(synthetic)
    stamp = br.index[-1]
    weights, _ = srp_weights(
        synthetic["scores"], synthetic["funding"], br.loc[:stamp],
        synthetic["cols"], stamp, None, synthetic["cfg"],
    )
    assert abs(float(weights.sum())) < 1e-9, "book is not dollar-neutral"
    assert float(weights.abs().sum()) <= 2.0 + 1e-9, "gross exposure exceeds bound"
    assert float(weights.abs().sum()) > 0.0, "book is flat"
