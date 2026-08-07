# System design: quant_signal data platform

High-level infra view: latency budgets, bottlenecks, and the target architecture.
Feature work (PEAD etc.) is a vertical slice on top; the emphasis here is the
*pipeline as a system* — how data moves from public sources to serving, how fast
that is allowed to be, and where the wall-clock time actually goes.

## 1. System context

- **Inputs** (all real, keyless): Yahoo (US equity daily OHLCV, unofficial),
  Binance (crypto minute OHLCV, public API), FRED (macro CSVs), SEC EDGAR
  (company facts, public API, `filed_at` point-in-time timeline).
- **Core** (this repo): orchestration (Prefect), provider fetch → contract
  validation gate → Bronze (raw) → dbt Silver (clean/typed/PIT) → Gold (marts);
  invalid rows go to QUARANTINE, never dropped.
- **Consumers**: research scripts (PEAD event study); future feature store,
  model serving, and the sellable data products (lookahead audit, SUE feed).

```
 public sources            core                          consumers
 ─────────────             ──────                        ─────────
 Yahoo ──┐                 ┌─────────── Prefect ───┐
 Binance ┼── providers ───►│  fetch → validate      │──► Bronze ─► dbt ─► Silver ─► Gold ─► scripts
 FRED   ─┤                 │  → upsert / quarantine │                          │
 EDGAR  ─┘                 └────────────────────────┘                          └──► (future) feature store / serving
```

## 2. Measured latency (live runs, 2026-08-06)

| Stage | Measured | Notes |
|---|---|---|
| Equity cold backfill (28 sym × 19y) | ~5 min | fetch-bound; 84+ requests, 1 s throttle, 12 h disk cache |
| Equity warm daily run | **71 s wall** | fetch+validate ~60 s, Snowflake MERGE 12.9 s / 132,980 rows, quarantine 2.8 s |
| Fundamentals backfill (28 tickers) | 35 s | EDGAR fetch 23 s, upsert 7 s / 1,286 rows; SEC 0.15 s/req |
| dbt full build | 49 s | 82 objects (16 incremental, 6 tables, 13 views, 45 tests) |
| Interactive SELECT on marts | 1–4 s | dominated by Snowflake session/metadata, not query |
| Validation gate (pandas) | ~1 s | in-process, contract checks |
| **Streaming E2E (M3)** | **bar → Redis ≈ 2 s** | producer poll flush → Kafka → materializer → Redis; Flink features fire ≤ 5 min + 2 min watermark on the window's end |

## 3. Latency budgets & SLAs by tier

| Tier | Budget | SLO | Notes |
|---|---|---|---|
| Ingest → Bronze | < 5 min | 99% ≤ 10 min | warm path is fetch-bound; external rate limits set the floor |
| Bronze → Silver/Gold (dbt) | < 5 min | 99% ≤ 10 min | currently 49 s |
| **E2E daily refresh** | **< 10 min** | **99% ≤ 15 min** | ingest + dbt today ≈ 2 min wall |
| Interactive analytics query | < 2 s | p95 < 5 s | acceptable for OLAP, NOT for serving |
| Online feature lookup | **p95 < 500 ms** | p99 < 1 s | **cannot be served from Snowflake** — M3's Redis online store hits this (API reads Redis only) |
| Streaming bar → online store | p95 < 5 s | p99 < 10 s | producer flush → Kafka → materializer → Redis (measured ≈ 2 s) |
| Streaming window feature | ≤ 5 min + 2 min watermark | p99 ≤ 8 min | event-time TUMBLE: a window fires ≤ 2 min after it closes |
| Backfill / replay | no SLA | bounded | chunked, resumable, idempotent (MERGE upsert) |

The honest bottleneck split today: **fetch ≫ compute**. External rate limits
(Yahoo ~100–360 req/hr/IP, SEC 0.15 s/req) dominate cold/backfill wall-clock;
steady-state daily runs amortize via the 12 h disk cache, and compute (validation,
MERGE, dbt) is already seconds-scale.

## 4. Target architecture

- **Warm path (daily)**: scheduler → Prefect work pool (persistent server, not the
  current inline temp server) → ingest flow → dbt build. Target < 10 min E2E.
  Idempotent MERGE upserts make failed batches safe to re-run.
- **Cold / backfill path**: already chunked (period1/period2 windows, IPO-aware
  splitting) and resumable; parallelize *across symbols* with a per-provider
  rate-limit token bucket instead of the current sequential 1 s throttle.
- **Near-real-time path (the latency showcase)**: crypto minute bars are the only
  source that *arrives* continuously. M2 (done 2026-08-06): an in-API poller
  (`api/stream.py`) fetches Binance minute bars and serves them live over
  `/ws/market` + REST. M3 (done 2026-08-07): standalone producer → **Redpanda
  (Kafka API) → Flink SQL (5m event-time windows, checkpointed) → Redis online
  store**; the API serves bars + features from Redis at p95 < 500 ms. Kafka →
  Snowflake persistence (Snowpipe Streaming) is the documented next hop.
- **Serving path (features)**: the M3 Redis online store serves live 1m bars and
  5m window features (OHLCV, VWAP, bar_count) at the sub-500 ms budget. The batch
  as-of feature tables for equities (incremental on `filed_at`, restatements
  preserved) remain a dbt milestone — the no-lookahead moat the sellable products
  depend on.
- **Observability**: per-stage structlog events already exist (`validation_gate`,
  `snowflake_upsert`, `ingest_market_data_complete`, all with `elapsed_ms`).
  Formalize them into a `PIPELINE_METRICS` table (write stage timings after each
  run) + Elementary freshness/volume; alert on SLO breach. `query_tag` already
  attributes warehouse spend per pipeline via `QUERY_HISTORY`.

## 5. Design decisions & trade-offs

| Decision | Choice | Rationale / trade-off |
|---|---|---|
| Store | Warehouse-native (Snowflake) | Operational simplicity at this scale; revisit Iceberg/parquet + object store for cold Bronze archives when Bronze ≫ Silver |
| Equity/fundamentals cadence | Batch (daily) | Sources are T+1 by nature; streaming adds latency-complexity for no freshness win |
| Crypto cadence | Streaming (Kafka/Flink/Redis, M3) | Only source that arrives continuously; the near-real-time showcase. Kafka→Snowflake (Snowpipe Streaming) still to wire for persistence |
| Correctness | Quarantine over drop; MERGE upsert; PIT `filed_at` timeline | No lookahead, no silent data loss — the product moat |
| Message bus | Redpanda (Kafka API) + confluent-kafka | Industry-standard ingestion bus; Flink source/sink; online store on Redis |
| Feature serving | Redis online store (stream) + dbt as-of tables (batch) | OLAP (Snowflake) can't hit online p95 < 500 ms |

## 6. Infra-first roadmap

1. **Latency telemetry — DONE (2026-08-06).** `ingest/metrics.py` + the
   `scripts/run_dbt.py` wrapper persist per-stage `elapsed_ms` into
   `BRONZE.PIPELINE_METRICS` (source → `silver_pipeline_metrics`); every ingest
   flow and `make dbt-run` records fetch/validate/write/quarantine/dbt-build.
   Live first run: ingest ≈ 85 s + dbt-build ≈ 30 s → E2E ≈ 2 min, inside the
   < 10 min budget. Known nuance: the dbt-build row lands *after* the silver
   table materializes, so silver picks it up on the next build (query Bronze for
   same-run visibility). Next: SLO-breach alert on this table.
2. **Persistent orchestration** — Prefect work pool + server (today: inline temp
   server per run), scheduled warm path.
3. **Near-real-time crypto path** — M2 live stream (done): in-API Binance
   poller → WebSocket. M3 (done): standalone producer → Redpanda → Flink 5m
   windows → Redis online store → API. Next: Snowpipe Streaming persistence +
   freshness monitor, then the M3.5 prediction layer (online learning +
   conformal intervals + Monte Carlo) served into the Signal Terminal UI.
4. **Feature store / as-of serving** (M2) — dbt as-of features + lookup layer,
   p95 < 500 ms target.
5. **Storage tiering** — archive cold Bronze to object store; keep marts in
   Snowflake.

Parked (features, second-class): quarterly 10-Q PEAD extension, MLflow tracking,
productization of the lookahead audit / SUE feed.
