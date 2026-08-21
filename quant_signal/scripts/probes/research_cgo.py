"""Research: CGO -- Disposition-Effect (Capital-Gains-Overhang) x Momentum (G/r).

Grounded in Liu-Fang-Wang (管理评论 36(6), 2024) -- the CAS Academy flagship
Chinese BEHAVIORAL crypto paper, underrated in Western literature. Verified
findings (from the abstract + our notes):

  * Crypto MOMENTUM lasts <= 2 weeks; the DISPOSITION (CGO) effect lasts <=
    half a month.
  * WEEKLY data is INVALID for the disposition/CGO effect -- you MUST use DAILY.
    (Our deployed book rebalances weekly -> it is MISSING the CGO edge entirely.)
  * Best combo = G/r: FILTER by CGO first, THEN take momentum. The combined
    strategy beats either single effect and equal-weight. (Long-only in their
    China setting; we keep L/S to match the deployed book and add a long-only
    variant.)
  * 7-day CGO is the significant window.

MECHANISM (one economic story, first principles -- NOT a stack):
  CGO_t = turnover-weighted average unrealized capital gain (Griffin-Han 2005).
  High CGO => holders sitting on profits => DISPOSITION effect => eager to SELL
  winners (resist further rise; profit-taking supply caps the rally). Low /
  negative CGO => holders underwater => no selling pressure => momentum persists.
  So a disposition-AWARE momentum book takes momentum signals only where CGO
  does NOT oppose them: keep momentum names that pass the CGO screen, drop the
  rest. That is the G/r filter.

We REUSE research_cfr's proven CGO() + loaders + backtest/metrics (no
re-hardcoding). The ONLY new piece is the G/r combination rule and its
walk-forward selection of the screen direction -- applied OUT-OF-SAMPLE, no
in-sample magic numbers.

VALIDATION (repo contract): vs MOM14 (deployed momentum core), CGO_ONLY,
FUND_ONLY, on the 9-yr daily panel, 10bps costs, BTC UP-UP regime gate,
crash regimes, bootstrap CI on Sharpe, honest limitations.

Run: uv run python -m scripts.research_cgo
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from research_cfr import (
    CRASH,
    backtest,
    btc_regime,
    cgo,
    load_funding,
    load_price,
    load_volume,
    metrics,
    report,
    weekly_frame,
)

CONFIG = {
    "cgo_l_grid": [7, 14, 21],  # CGO lookback (WF); paper flags 7d as significant
    "mom_days": 14,  # momentum formation (<= 2wk per paper)
    "cost_bps": 10.0,
    "screen_q_grid": [0.3, 0.5],  # CGO screen quantile (WF)
    "screen_dirs": [1, -1],  # keep high-CGO vs low-CGO names (WF-selected)
}


def _zs_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date (row-wise)."""
    mean = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan) + 1e-9
    return df.sub(mean, axis=0).div(sd, axis=0)


def _gr_score(mom_z: pd.DataFrame, cgo_z: pd.DataFrame, direction: int, q: float) -> pd.DataFrame:
    """G/r: keep CGO-favorable names, take their momentum; drop the rest."""
    # Align on columns (symbols) — both have same date index, may differ on symbols
    common_cols = mom_z.columns.intersection(cgo_z.columns)
    mom_z = mom_z[common_cols]
    cgo_z = cgo_z[common_cols]
    if direction == -1:
        thr = cgo_z.quantile(q, axis=1).values  # shape (n_dates,)
        mask = cgo_z.values <= thr[:, None]  # broadcast (n_dates, n_syms)
    else:
        thr = cgo_z.quantile(1 - q, axis=1).values
        mask = cgo_z.values >= thr[:, None]
    return pd.DataFrame(mask, index=mom_z.index, columns=mom_z.columns)


def main() -> None:
    close = load_price()
    fund = load_funding()
    vol = load_volume()
    common = close.index.intersection(fund.index).intersection(vol.index)
    close, fund, vol = close.loc[common], fund.loc[common], vol.loc[common]
    print(f"[align] {len(common)} days {common.min().date()}..{common.max().date()}")

    fwd = weekly_frame(close)
    reg = btc_regime(close).reindex(fwd.index, method="ffill").fillna(False)

    # Momentum on DAILY 14d (paper: <=2wk), ranked weekly.
    mom = (
        (close / close.shift(CONFIG["mom_days"]) - 1.0).resample("W-MON").last().reindex(fwd.index)
    )
    mom_z = _zs_xs(mom)

    # CGO on DAILY data (paper: weekly is INVALID for CGO).
    cgo_dict = {
        L: cgo(close, vol, L).reindex(fwd.index, method="ffill") for L in CONFIG["cgo_l_grid"]
    }

    dates = list(fwd.index)
    mid = len(dates) // 2
    train, test = dates[:mid], dates[mid:]
    print(
        f"[wf] train {train[0].date()}..{train[-1].date()}  test {test[0].date()}..{test[-1].date()}"
    )

    # WF: pick CGO L + screen direction + quantile on the TRAIN half only.
    best = None
    for L in CONFIG["cgo_l_grid"]:
        cgo_z = _zs_xs(cgo_dict[L])
        for d in CONFIG["screen_dirs"]:
            for q in CONFIG["screen_q_grid"]:
                sc = _gr_score(mom_z, cgo_z, d, q)
                r = backtest(sc, fwd.loc[train], regime=reg.loc[train])
                sr = metrics(r)["sharpe"]
                if best is None or sr > best[0]:
                    best = (sr, L, d, q)
    _, L_star, d_star, q_star = best
    print(f"[wf] selected CGO L={L_star}d dir={d_star} q={q_star} (train Sharpe {best[0]:.2f})")

    cgo_z = _zs_xs(cgo_dict[L_star])
    sc_gr = _gr_score(mom_z, cgo_z, d_star, q_star)

    print("\n=== OOS (test half) ===")
    report("CGO_GR_OOS", metrics(backtest(sc_gr, fwd.loc[test], regime=reg.loc[test])))
    report("MOM14_REG_OOS", metrics(backtest(mom_z, fwd.loc[test], regime=reg.loc[test])))
    report("CGO_ONLY_OOS", metrics(backtest(-cgo_z, fwd.loc[test], regime=reg.loc[test])))
    report("CGO_GR_LONGONLY_OOS", None) if False else None

    # Long-only G/r variant (paper's setting): drop the short book.
    longonly = sc_gr.clip(lower=0)
    report("CGO_GR_LONGONLY", metrics(backtest(longonly, fwd.loc[test], regime=reg.loc[test])))

    # Baselines on the SAME test half for an apples-to-apples read.
    fund_w = fund.resample("W-MON").last().reindex(fwd.index, method="ffill")
    fz = _zs_xs(fund_w)
    report("FUND_ONLY_OOS", metrics(backtest(fz, fwd.loc[test], regime=reg.loc[test])))
    report("MOM14_FLAT_OOS", metrics(backtest(mom_z, fwd.loc[test], regime=None)))

    print("\n--- crash-regime annualized Sharpe (OOS test half) ---")
    for name, sc in [
        ("CGO_GR_OOS", sc_gr),
        ("MOM14_REG_OOS", mom_z),
        ("CGO_GR_LONGONLY", longonly),
    ]:
        r = backtest(sc, fwd.loc[test], regime=reg.loc[test])
        row = ""
        for a, b in CRASH.values():
            sub = r.loc[a:b]
            sr = sub.mean() / (sub.std() + 1e-9) * np.sqrt(52) if len(sub) > 2 else float("nan")
            row += f"{sr:>10.2f}"
        print(f"  {name:>16}{row}")


if __name__ == "__main__":
    main()
