# scripts/

Everything that runs as a command rather than as part of the live system. The
live path is `stream/`, `api/` and `ingest/`; nothing here is imported by a
running service except the files at the top level, which are libraries the
services depend on.

## Where the strategy actually lives

Start here. These are the files that define and evaluate SRP.

| file | what it is |
|---|---|
| `srp_strategy.py` | **The strategy.** 11 factors, self-referential ranking, one book per factor, risk-parity combination. Imported by the live scorer, so research and production cannot diverge. |
| `srp_backtest.py` | The committed evaluator. Produces the headline Sharpe. `--source file\|snowflake`. |
| `srp_walkforward.py` | Walk-forward selection and rebalance-timing luck across seven weekday anchors. |
| `srp_holdout.py` | Held-out universe splits. |
| `srp_dsr.py` | Deflated Sharpe, deflated by trials actually executed. |
| `srp_sweep.py` | The configuration sweep; every run it evaluates is written to `trial_registry.py`. |
| `srp_ablation.py` | Which construction choice earns the gap (ranking / books / weighting). |
| `srp_parity.py` | Live scoring reproduces research scoring, bar for bar. |
| `srp_factor_regression.py` | Alpha against market, size and momentum. |
| `srp_drawdown.py` | Drawdown and regime behaviour. |
| `tskd_audit.py` | The eleventh factor, standalone and leave-one-out. |

Supporting libraries, also top level because the live path imports them:

| file | what it is |
|---|---|
| `factor_core.py` | `ts_rank_pit` and `xs_rank` — the two ranking primitives, and the point-in-time guarantee. |
| `research_fas_clean.py` | Loads the panel and resamples it. `build_frames` is shared by the file and warehouse paths. |
| `warehouse_panel.py` | The same panel, read from Snowflake, with an `as_of` cutoff. |
| `panel_parity.py` | Gate: the warehouse panel and the file panel are identical. |
| `trial_registry.py` | Every configuration ever evaluated. What the deflated Sharpe deflates by. |
| `deflated_sharpe.py` | Bailey & López de Prado DSR. |
| `stream_watchdog.py` | Liveness checks for the running services. |

## Subdirectories

| directory | contents |
|---|---|
| `probes/` | Abandoned research. Ideas that were tested and did not survive: disposition effects, washout reversal, cross-sectional tails, and about thirty more. Kept deliberately — they are the denominator in the multiple-testing correction, and a strategy claim is only as honest as the record of what else was tried. Nothing imports them. |
| `backtests/` | Standalone backtests for strategies other than SRP (carry, PEAD, washout), plus portfolio construction and factor IC. |
| `data/` | Fetching and landing: exchange pulls, the Snowflake research-panel loader. |
| `backfill/` | Historical backfills for the caches the live scorer seeds from. |
| `ops/` | Running the system: service installation, dbt runs, replay, quality checks, track record. |

## Related

`research/code/` is a separate, frozen replication package for the paper, with
its own README mapping each script to the table it produces. It is not this
directory and is not imported by anything — do not delete it, the manuscript
cites it by name.
