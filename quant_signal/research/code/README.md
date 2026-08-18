# Replication code

Every numerical claim in *Every Asset Its Own Benchmark: Market-Neutral Alpha in
Perpetual Futures* is produced by the code in this directory. Nothing here is
illustrative: each script named below was executed to generate the corresponding
table, and re-running it reproduces the figure exactly.

Scripts abandoned during development are **not** included. Those constructions
are enumerated in Appendix B of the paper, but they contribute no reported
result and their code is not part of the replication set.

---

## Which script produces which result

| paper | result | script |
|---|---|---|
| Table 2 | Factor definitions | `srp_strategy.py`, `backfill_intraday_features.py` |
| §4.4 | Trade-size asymmetry (TSKD) construction | `backfill_intraday_features.py` (`daily_features`) |
| §4.5 | Ranking primitives, leak test | `factor_core.py` |
| Table 3 | Ranking frame, with and without carry | `srp_backtest.py` (`ranking=cross\|self`) |
| Table 4 | Mechanism decomposition, 41% | `srp_backtest.py` (`ranking=cross_z`) |
| Table 5 | Carry as a factor | `srp_backtest.py` (`funding_tilt=0`) |
| Table 6 | Construction, paired and best-of-grid | `srp_ablation.py` over `srp_sweep.py` |
| Table 7 | Factor-adjusted alpha, Newey–West | `srp_factor_regression.py` |
| Table 8 | Leave-one-out contributions | `tskd_audit.py` |
| Table 9 | Drawdown, tails, 2022 collapse | `srp_drawdown.py` |
| Table 10 | Walk-forward by rebalance anchor | `srp_walkforward.py --mode both` |
| Table 11 | Fold geometry | `srp_walkforward.py` |
| Table 12 | Held-out universe, matched breadth | `srp_holdout.py` |
| Table 13 | Deflated Sharpe ratio | `srp_dsr.py` reading `trial_registry.py` |
| Table 14 | Transaction-cost robustness | `srp_backtest.py` (cost curve scaled) |
| Table 15 | Research–production equivalence gate | `srp_parity.py` |
| Table 16 | Capacity | `srp_backtest.py` liquidity panel |
| Appendix A | Summary of statistical evidence | all of the above |

---

## Reproducing the headline numbers

```bash
python -m scripts.srp_parity              # the gate: 0 FAIL, 0/4,480 mismatches
python -m scripts.srp_backtest            # 2.161, t 5.03, 282 rebalances
python -m scripts.srp_factor_regression   # alpha 27.05%, market beta -0.001
python -m scripts.srp_drawdown            # max DD -7.55%, 2022: mkt -82.5% vs +9.4%
python -m scripts.srp_walkforward --mode both   # OOS 2.068 across seven anchors
python -m scripts.srp_holdout --splits 8  # held-out universe, 83% at matched breadth
python -m scripts.tskd_audit              # TSKD 3rd of 11
python -m scripts.srp_ablation            # construction gap +0.665, t 5.82
python -m scripts.srp_dsr                 # DSR 0.9885 (N=211) / 0.9475 (N=1,286)
```

All are deterministic: identical inputs give identical outputs, which
`srp_parity.py` asserts directly.

---

## Structure

**The strategy itself**

- `srp_strategy.py` — the single source of truth. A pure function: frames in,
  target weights out. No I/O, no globals, no environment reads. Research and the
  live path both call it; neither reimplements it.
- `factor_core.py` — the two ranking primitives kept deliberately separate
  (`xs_rank` cannot leak; `ts_rank_pit` is trailing) plus `leak_test`, which
  recomputes a historical score with future rows deleted and requires an exact
  match.

**Evaluation**

- `srp_backtest.py` — the committed evaluator. Loads the panel once and memoises
  intermediates on the sub-config they actually depend on.
- `srp_walkforward.py` — walk-forward selection and seven-anchor timing luck.
- `srp_holdout.py` — held-out universe with the matched-breadth control.
- `srp_factor_regression.py` — regression on market, size, momentum with
  Newey–West standard errors.
- `srp_drawdown.py` — equity curve, tails, and the 2022 window.
- `tskd_audit.py` — leave-one-out contributions on one matched sample.
- `srp_ablation.py` — paired and best-of-grid construction comparison.

**Search discipline**

- `trial_registry.py` — append-only JSONL: config hash, Sharpe, observation
  count, git SHA. Written by the code performing each evaluation.
- `srp_sweep.py` — executes the search space and logs every cell, including
  those that fail.
- `srp_dsr.py` — deflated Sharpe, reading N and dispersion from the registry
  rather than from recollection.
- `deflated_sharpe.py` — the PSR and expected-maximum-Sharpe primitives.

**Verification**

- `srp_parity.py` — the deploy gate. Asserts point-in-time integrity,
  determinism, dollar-neutrality, non-emptiness, and that the live path produces
  identical positions. Exits non-zero on any failure.

**Data**

- `backfill/` — acquisition CLIs for bars, funding and positioning. All sources
  are public and keyless.
- `backfill_intraday_features.py` — reduces intraday bars to one record per UTC
  day. Used by both research and the live feed, which is why the two agree.
- `research_fas_clean.py`, `research_intraday.py` — panel loaders.

**Live path**

- `srp_live.py` — data plumbing only. Seeds history from the research caches,
  appends live observations, and hands the frames to `srp_strategy`. It contains
  no strategy logic, which is what makes the parity assertion meaningful.

---

## Data

No proprietary or paid data is used. Price, volume and funding come from a public
exchange bulk archive; positioning from a keyless public metrics endpoint. The
panel is 112 USDT-margined perpetual futures over 363 weeks.

## Note on paths

These files are copied from `scripts/` and `stream/` in the parent repository and
retain their original `scripts.*` import paths. Run them from the repository
root rather than from this directory.
