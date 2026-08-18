# Realtime architecture in depth

Design decisions, failure modes, and the outages that motivated them. The [README](../README.md) has the overview; this has the detail.

See also [architecture.md](architecture.md) for the batch platform and [RUNBOOK.md](RUNBOOK.md) for operating it.

---

## Two clocks, two execution engines

SRP and the online predictor are different strategies on different clocks, and
conflating them was the one real blocker to getting SRP traded.

The hourly engine matches a stored signal to a bar by `window_end_ms` and refuses
to act when they differ. SRP scores on a **weekly** bar, so its records could
never match an hourly window. Echoing the hourly stamp would have made it trade
immediately — and would have asserted that SRP scored on a window it never saw,
which is exactly the class of untruth the parity gate exists to catch.

So each strategy runs its own execution instance, sharing one venue adapter:

| | clock | signal | ledger | cost filter |
|---|---|---|---|---|
| River / asym | per feature window | `predicted_return` + direction | `execution:crypto:1h` | λ·fee band |
| **SRP** | **weekly** | target weight + direction | `execution:srp` | **disabled** |

The cost filter is disabled for SRP deliberately. It exists to stop a
continuously-updating forecast from churning; SRP has no forecast to threshold,
and its turnover is already bounded by the strategy's own cap. Applying a
forecast band to a portfolio weight would compare quantities with different
units. Two things had to change in the engine for a weight-driven strategy to
work at all: the entry gate required a non-null `predicted_return` even when the
filter was off, and a fresh process had no previous weekly bar to advance from,
so it held forever without logging anything.

**SRP is weekly by economics, not by limitation.** Its rank window is 52 weeks;
recomputing hourly would shift that window by one part in 8,736 and produce the
same book, then pay fees to trade it. There is no hourly information in a
52-week trailing rank. The continuous slot is the predictor's job.

---

## Who actually produces each topic

Two producers feed the feature bus, and knowing which is which matters — a
topic that is *fresh* is not necessarily fresh *from the source you assume*.
Verified by consuming each topic and checking the VWAP signature (Flink computes
a true volume-weighted price; the feed publishes a typical price):

| Topic | Producer | State |
|---|---|---|
| `crypto.bars.raw` | `stream/producer` | live — 1-min venue bars |
| `crypto.features.5m` | **Flink SQL** | live — volume-weighted VWAP |
| `crypto.features.1h` | **`stream/feature_feed.py`** | live — 20/20 sampled messages carry the feed's signature, none Flink's |

**The hourly path bypasses Flink**, and that was a deliberate response to a real
outage. The 1h river stalled while every process still reported healthy: nothing
crashed, no error was logged, and the dashboard rendered a `window_end`
**312 days** in the past. A pipeline producing nothing looks identical to one
that is merely quiet.

`stream/feature_feed.py` is therefore the hourly upstream: it fetches completed
1h bars from the venue's keyless public endpoint and publishes them to
`crypto.features.1h` in the **exact Flink schema**, field for field, so every
downstream consumer runs unchanged. Flink continues to serve the 5-minute
windows. Two defences were added at the same time:

- a **staleness guard** in the signal path, so a bar whose `window_end` is older
  than a bounded age can never trigger a rebalance;
- **per-symbol venue-category discovery** — `BTCUSDT` exists as both spot and
  linear perp while `RLCUSDT`/`STORJUSDT`/`XMRUSDT` are perp-only, so hardcoding
  either choice starves half the universe (measured: 1 of 6 test symbols
  fetching). The category is discovered on first use and remembered.

The generalisable lesson: **liveness has to be asserted against a clock, not
inferred from the absence of errors** — and "which producer wrote this?" is a
question worth being able to answer, which is why the table above was verified
by inspecting message contents rather than by trusting the architecture diagram.

---

---

## Data infrastructure: storage, transport, and failure modes

Two pipelines feed the strategy, and they are engineered to different contracts.

**Offline path — builds history, optimised for completeness.**

```
data.binance.vision (bulk archive)  +  keyless Binance/Bybit REST
   │
   ├─ scripts/backfill/backfill_ohlcv.py ────────→ hourly bars, resampled to weekly
   ├─ scripts/backfill_intraday_features.py ────→ 1h AND 5m klines reduced to ONE
   │                                               record per UTC day
   └─ scripts/backfill/backfill_positioning.py ─→ futures metrics (open interest,
                                                   long/short ratios), daily
```

**Online path — extends that history, optimised for freshness.**

```
Bybit keyless REST → stream/feature_feed.py → Kafka (crypto.features.1h)
                                                 │
   stream/asym_signal.py ◄────────────────────────┘   consumes the bus,
      │                                                maintains a weekly registry
      └─ stream/srp_live.py ── seeds from the offline caches, appends live via
                               intraday_feed.py + positioning_feed.py
                                                 │
   Redis (prediction:crypto:asym:1h) ◄───────────┘
      │
   stream/execution.py + bybit_demo.py → REAL orders (maker-first, market fallback)
      │
   Redis (execution:crypto:1h) → FastAPI :8000 → Next.js :3000
```

**Why each store.** The feature bus is **Kafka** because it is a replayable log —
a consumer can be restarted and re-read from an offset, which matters when the
signal daemon crashes mid-window. The signal→execution handoff is **Redis**
because execution needs the *current* target, not the history; a key-value read
is the right primitive and keeps the API under 500 ms without touching Kafka or
the warehouse. **Snowflake and the Iceberg lake are not in the crypto trading
path at all** — they serve the batch/equities side (Yahoo, EDGAR, FRED, Alpaca).

**The seed-and-extend problem.** Live REST surfaces expose roughly 30 days of
positioning history; the strategy needs 52 weeks of trailing data per symbol. So
live history is *seeded* from the offline caches and extended forward. That join
is only sound if both sources measure the same quantity, which was verified
rather than assumed — **0.000e+00 across 7 fields over 26 days.** Reaching that
number required fixing two real defects:

| Defect | Why it was invisible | Fix |
|---|---|---|
| Partial-day bars | the current day always has fewer bars than a complete one, so the reduction silently differed between live and research | require a full bar count per interval before emitting a day |
| REST/archive timestamp offset | REST stamps a window at its **end**; the bulk archive labels by occurrence — a clean one-period shift that looked like plausible data | subtract one period on the REST path |

**Failure modes this pipeline is built against**, each learned from an outage:

- **Silent starvation.** A producer that stops writing looks exactly like a quiet
  market. Liveness is asserted against a wall clock — a bar older than a bounded
  age cannot trigger a rebalance, and staleness is surfaced on a health strip
  rather than inferred from an absence of errors.
- **Wrong-venue lookups.** `BTCUSDT` exists as both spot and linear perp;
  `RLCUSDT`, `STORJUSDT` and `XMRUSDT` are perp-only. Hardcoding either category
  starved the universe (measured: 1 of 6 symbols fetching). Category is now
  **discovered per symbol** on first use and cached.
- **State latched on failure.** A transient all-FLAT selection once froze the
  book for a full week because the failure was latched as if it were a decision.
  Only a non-FLAT selection is now latched.
- **Throughput mistaken for breakage.** A 76-symbol book filling alphabetically
  looks stalled. Poll budgets are configurable rather than fixed (19 min → 5 min
  for a full cycle), and partial fills are reported as progress, not failure.

**Deployment.** Locally the daemons run under **launchd** (`com.quantsignal.*`),
each with its own environment and restart policy. On AWS the same processes run
as **ECS Fargate** services against **MSK** and **ElastiCache**, defined in
Terraform with remote state.

---

## Two strategies, two execution paths

The platform started by trying to **predict returns** with an online learner and
later added a **systematic factor strategy**. Both run, and both now trade — on
different clocks, through separate execution engines that share one venue
adapter (see [Two clocks, two execution engines](#two-clocks-two-execution-engines)).

The hourly engine trades whichever forecast-driven signal `STREAM_STRATEGY`
selects; SRP is driven independently by `stream/srp_execution.py` on its weekly
rebalance:

```
stream/predictor.py    River online learner  → prediction:crypto:1h:*        (102 keys)
stream/asym_signal.py  SRP factor strategy   → prediction:crypto:asym:1h:*   (105 keys)
                                                        │
                            STREAM_STRATEGY=asym ───────┘
                                                        ▼
stream/execution.py  ──→  REAL orders  ──→  execution:crypto:1h:*  (shared ledger)
```

**Why the switch.** The online learner is asked to forecast a single asset's
return — the hardest version of the problem, and the one Gu, Kelly & Xiu (2020)
show is weakest. The factor book never forecasts a return at all: it ranks assets
and holds a dollar-neutral spread, so it only needs the *ordering* to be right.
Neither model was allowed to trade on a hunch; the learner had to clear the
promotion gate, and the factor book had to clear out-of-sample validation and the
parity gate.

**Why the learner still runs.** It writes to its own prefix, so it produces a
continuous live comparison against the strategy that trades — an A/B arm at zero
risk. The execution ledger stays on the shared `execution:crypto:1h` key, so the
dashboard is unchanged regardless of which signal is promoted.

That indirection is the point: **swapping the live strategy is a config change,
not a rewrite.** Nothing downstream of the prediction key knows or cares which
model produced it.

---

---

## Research ↔ production parity

The most expensive failure in a quant platform is not a crash. It is a live
system that trades **something other than what was validated**, while every
dashboard reports success. This repository has hit that failure twice, so the
defence against it is built in rather than bolted on.

**What went wrong, both times.** A streaming reimplementation of the book drifted
to **0.147 rank correlation** against research with **27% selection overlap** —
chance is ~20%. It had been trading a different strategy for weeks. Separately, a
look-ahead bug survived for months because research and live called the *same
wrong helper*: `df.apply(_rank_z)` uses pandas' default `axis=0`, so each week
was ranked against that symbol's **entire history, future included**. Deleting
future rows changed a historical score for **98 of 112 symbols**. Both
implementations agreed perfectly — on the wrong answer.

**Defence 1 — one strategy, one implementation.** `scripts/srp_strategy.py` is a
pure function: frames in, target weights out. No I/O, no globals, no environment
reads. Research and the live daemon both call it; neither reimplements it.

**Defence 2 — a gate that fails the build, not the P&L.** `scripts/srp_parity.py`
exits non-zero, so it can block a deploy:

```
universe 112 symbols, 363 weeks
  1. point-in-time     leak 0.00e+00 on every factor
  2. determinism       identical inputs -> identical weights
  3. neutrality        max |net| 9.02e-17 ; gross bounded by 2.0
  4. no silent empty   308 rebalances built, active on 282
  5. research == live  0 direction mismatches / 4,480
=== 0 FAIL ===
```

Assertion 1 catches look-ahead by recomputing a historical score with future rows
deleted and requiring an exact match. Assertion 4 exists because a config bug
once produced an all-FLAT book that **looked like a clean run** — a test suite
that cannot distinguish "correct" from "did nothing" is not a test suite.

**Defence 3 — feeds reconciled before they are trusted.** The live path seeds
history from research caches and extends it with live REST. That join is only
valid if both sources measure the same thing, so it was verified rather than
assumed — **0.000e+00 across 7 fields over 26 days**. Getting there surfaced two
real defects:

| Defect | Symptom | Fix |
|---|---|---|
| Partial-day bars | live/research divergence on the current day | require a complete bar count per interval before emitting |
| REST/archive day offset | positioning silently one day stale | REST stamps at interval **end**, the archive labels by occurrence — subtract one period |

**Defence 4 — measured beats remembered.** `scripts/trial_registry.py` is an
append-only JSONL log — config hash, Sharpe, observation count, git SHA — written
by the code that runs each backtest. Multiple-testing corrections then read the
trial count and dispersion **from executed runs** rather than from a number a
researcher typed. This exists because an audit found four headline figures in
this repo with no artifact behind them at all; one turned out to be a pair of
tail-index statistics copied from a document about an unrelated strategy.

Point-in-time discipline is enforced by `scripts/factor_core.py`, which
separates the two ranking primitives that had been silently conflated:
`xs_rank` (within one date, cannot leak) and `ts_rank_pit` (trailing window,
ends at the current bar), with `leak_test` as the guard for any new factor.

---

---

