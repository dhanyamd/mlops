# quant_signal — production data platform architecture

A real-data quant platform: ingest → contract gate → Bronze → dbt (Silver/Gold)
with observability and enforced contracts. **No toy data**: every production
source is a real, keyless, publicly downloadable endpoint verified live.

## Data sources and quality tiers

| Source    | Asset class   | Grain    | Keyless | Quality tier                                   |
| --------- | ------------- | -------- | ------- | ---------------------------------------------- |
| SEC EDGAR | fundamentals  | annual   | yes     | institutional (official government API)        |
| FRED      | macro         | daily/mo | yes     | institutional (official St. Louis Fed CSV)     |
| Binance   | crypto        | minute   | yes     | high for a single venue (self-reported volume) |
| Yahoo     | US equities   | daily    | yes     | research-grade (unofficial endpoint, ToS-gray) |
| synthetic | n/a           | n/a      | n/a     | OFFLINE/TEST ONLY — never a production source  |

All production providers are **keyless** — verified live in this repo's history
(FRED `fredgraph.csv` HTTP 200; SEC `company_tickers.json` + `companyfacts`
HTTP 200; Binance public klines HTTP 200; Yahoo chart API HTTP 200 after the
`curl_cffi` Chrome-TLS fix, previously 429 with plain `requests`).

Production upgrade path (documented, not yet wired): Alpaca free IEX feed for
equities (~2.5% of consolidated volume, free key), paid Alpaca SIP for
production-grade equity coverage.

## Architecture

```
providers (yahoo/binance/fred/sec_edgar)
   │  fetch (real HTTP, retries, throttling, disk cache where needed)
   ▼
contract gate (pydantic validate_bars / validate_macro / validate_facts)
   ├── valid   → BRONZE (MERGE upsert on natural key)
   └── invalid → QUARANTINE (never dropped)
                        │
dbt (data as code)      ▼
   SILVER  — dedup, typing, clean core (contracts enforced)
   GOLD    — analytics marts (daily OHLCV rollups)
```

### Medallion layers

- **BRONZE** — raw landed data, one table per asset class, **never mixed**:
  `EQUITY_BARS`, `CRYPTO_BARS`, `FRED_MACRO`, `COMPANY_FACTS`.
  Types are explicit (`TIMESTAMP_NTZ`, `DATE`, `FLOAT`, `TEXT`) — guaranteed by
  `write_pandas(use_logical_type=True)` (root-caused live: without it Snowflake
  binds datetimes as epoch-nanos and auto-creates `NUMBER` columns).
- **SILVER** — deduplicated, contract-typed models: `silver_equity_bars`,
  `silver_crypto_bars`, `silver_fred_macro`. Every column declares a contract
  (`not_null`, `data_type`, `accepted_values`) enforced at build time.
- **GOLD** — consumer-ready: `gold_daily_bars` (per-symbol daily OHLCV rollup
  grouped by `(symbol, timeframe, date)`).

### Anti-pollution rules

- One Bronze table per asset class; Binance rows never touch `EQUITY_BARS`
  (routing by provider in the flow, tested).
- Different grains never summed across timeframes (`timeframe` is a column;
  Gold groups by it).
- Quarantine over drop: any row breaking the contract lands in `QUARANTINE`
  with its source, so nothing is silently lost.
- Idempotency: MERGE upsert on natural keys (`symbol,timeframe,ts` for bars;
  `series_id,date` for macro; `ticker,metric,fiscal_year` for facts) — reruns
  are safe.

## Configuration — no hardcoded business values

Every business/operational value comes from the environment (`.env` → typed
`Settings` in `config/settings.py`):

- `INGEST_DEFAULT_PROVIDER`, `INGEST_DEFAULT_DAYS`, `INGEST_DEFAULT_SYMBOLS`,
  `INGEST_DEFAULT_CRYPTO_SYMBOLS`, `INGEST_DEFAULT_MACRO_SERIES`,
  `INGEST_DEFAULT_TICKERS`, `INGEST_DEFAULT_METRICS`, `YAHOO_CACHE_DIR`
- Snowflake connection/auth (MFA token-cached, key-pair alternative), dbt
  `DBT_*` mirror vars, `EDGAR_USER_AGENT`, `LOG_LEVEL`

Flows resolve defaults from `Settings`, and the inline CLI entrypoints use
stdlib `argparse` to override (defaults `None` → env). Nothing business-related
is hardcoded in flows or providers. The only literals in provider code are
immutable third-party API facts (URLs, rate-limit intervals, host
alternation) — not config.

Secrets: `.env` is gitignored; `DBT_ENV_SECRET_*` scrubs the dbt password from
logs; Snowflake secrets are `Field(repr=False)`.

## Orchestration

- **Prefect** flows: `ingest_market_data` (equity/crypto routing),
  `ingest_macro_data`, `ingest_fundamentals`. Task-level retries + backoff.
- Runs inline (no server) via `make ingest | ingest-crypto | ingest-macro |
  ingest-fundamentals`; as a deployment they'd run on a Prefect work pool.
- **Note (verified live):** Prefect 3.8 removed the `python flow.py -- --param`
  passthrough — flows use explicit `argparse` entrypoints instead.
- **dbt** via `make dbt-run` (parse → build → tests). Elementary package
  provides source-freshness + data-monitoring alerts (isolated in the
  `ELEMENTARY` schema so raw BRONZE stays clean).

## Cost / credit attribution

Every query is tagged: Snowflake client sends `query_tag`, dbt uses
`query_tag: quant_signal_dbt`. Spend is attributable per pipeline via
`ACCOUNT_USAGE.QUERY_HISTORY` by `QUERY_TAG`.

## Honest gaps vs. real quant firms

What we deliberately do NOT have yet (documented, not hidden):

- **Feature store / point-in-time joins** — no Feast/Featureform yet. EDGAR
  fundamentals ARE a filing-date timeline now (every 10-K/20-F with its
  `filed_at`, restatements preserved, fiscal year derived from the period end),
  but as-of joins to prices live in M2.
- **Experiment tracking** — no MLflow; model runs aren't versioned/logged.
- **Drift & anomaly monitoring** — Elementary covers table-level freshness/
  volume; model drift monitors come later.
- **Streaming** — M3 built a real-time layer (Redpanda/Kafka → Flink SQL →
  Redis) for crypto minute bars. It is **not** yet wired to Snowflake:
  `BRONZE.CRYPTO_BARS` still gets its rows from the batch/producer best-effort
  writes. Kafka → Snowflake via Snowpipe Streaming is the documented upgrade.
  No ML serving on the stream yet (M3.5).

## Real-time roadmap (M2–M4)

The real-time layer is the near-real-time showcase: Binance minute bars arrive
continuously, Flink computes event-time features, Redis serves them at
sub-500ms, and the dashboard visualizes them — demonstrating the full serving
path for a real-time data product without touching Kafka or Snowflake at read
time.

```
BinanceProducer → Redpanda (crypto.bars.raw) → Flink SQL 5m TUMBLE (checkpointed)
   → Redpanda (crypto.features.5m) → materializer → Redis (online store)
   → API /api/market/live (hub) + /api/market/features (Redis)
```

- **M2 — Live stream v1 (2026-08-06, DONE).** `api/stream.py` runs an in-API
  poller that fetches recent Binance minute bars, upserts them to
  `BRONZE.CRYPTO_BARS` (best-effort), keeps a per-symbol ring buffer, and fans
  deltas out over `/ws/market`. Hermetic tests cover ingest/dedupe/broadcast
  and the REST/WS endpoints. Demo-grade by design — superseded by M3's
  standalone producer.
- **M3 — Streaming-native ingestion + feature pipeline (2026-08-07, DONE).**
  Standalone `stream/producer.py` publishes Binance minute bars to Redpanda
  (Kafka API); a **Flink SQL job** (`stream/flink/jobs/crypto_features.sql`)
  computes 5-minute event-time TUMBLE windows (OHLCV, VWAP, bar_count) with
  checkpoints and out-of-order tolerance; `stream/materializer.py` lands live
  bars + window features into a Redis online store. The API serves them from
  Redis. KafkaBus/RedisKV are the production clients; FakeBus/FakeKV are
  test-only doubles. Event time (not processing time) drives the windows —
  watermarks tolerate the producer's in-progress-minute re-publishes. Run the
  stack with `make stream-infra` + `make stream-topics` + `make
  stream-flink-submit` (see README).
- **M3.5 — Prediction + Signal Terminal (NEXT).** Online learning (River)
  model fed by the Flink feature stream, **conformal prediction intervals**
  with self-measured coverage, and a **Monte Carlo** forward-simulation engine
  (paths seeded by Flink realized-vol) — predictions and quantile bands served
  from Redis and visualized in the Next.js Signal Terminal (live MC fan chart +
  feature panel + signal gauge + live P&L vs buy-and-hold, backtest-linked to
  Snowflake). Event Study + Macro as secondary tabs.
- **M4 — Breadth (trimmed).** Only the cheap wins: Spark feature-engineering
  batch over `BRONZE` and CI dbt against a CI Snowflake schema. Prefect
  orchestration and Snowpipe Streaming persistence are parked.

## M4+ roadmap

1. **Prediction layer (M3.5)** — online learning (River) on the Flink feature
   stream + conformal prediction intervals + Monte Carlo forward simulation,
   served from Redis into the Signal Terminal UI.
2. **Alpaca IEX provider** — production-grade equity upgrade (free key, ~2.5%
   of consolidated volume documented).
3. **MLflow** experiment tracking + model registry for forecasting/fraud
   models (the sibling `fraud_detection/` project).
4. **Spark** (local cluster) for feature engineering at scale over `BRONZE`.
5. **Snowpipe Streaming** — persist the Kafka crypto stream into Snowflake
   without batch best-effort writes (documented connector path).
6. **CI**: enable the `dbt-build` job against a CI Snowflake schema when the
   repo has DBT secrets.

System-level view (latency budgets, bottlenecks, target architecture, infra-first
roadmap): see `docs/system_design.md`.

## Sellable product ideas (parked — build after PIT + feature store)

The moat is real data + point-in-time correctness + contracts + monitoring.
Parked productizations, in rough value order:

1. **Lookahead-bias audit (SaaS/consulting)** — "your backtests are lying to
   you." Score a fund's fundamentals data (as-filed vs restated) and quantify
   how much reported alpha is a restatement artifact. People pay for the
   diagnosis. Vendors like FactSet/Refinitiv serve latest-restated numbers, so
   this is exactly the gap our `filed_at` timeline fills.
2. **As-filed earnings-surprise feed** — standardized unexpected earnings (SUE)
   measured against the as-filed figure + PEAD (post-earnings-announcement
   drift). Dataset tier ~$10-50k/yr/seat; hedge-fund-facing feeds $100k+/yr.
3. **Research subscription** — monthly restatement-adjusted earnings-drift
   factor for a fixed universe, with the PIT data layer as the proof of no
   lookahead bias.

First milestone if pursued: reproduce the PEAD anomaly on a few tickers with
as-filed numbers (e.g. AAPL FY2009's $36.5B -> $42.9B restatement), then
expand the universe.

## PEAD backtest (delivered)

`scripts/pead_backtest.py` runs a real event study on the PIT data layer:

- **Events** = SEC EDGAR earnings filings, dated by `filed_at` (the day the
  10-K became public), as-filed values (restatements are separate events).
- **SUE** per Bernard & Thomas (1989): expected = seasonal random walk (prior
  fiscal year's last-known value strictly from earlier filings), surprise =
  actual − expected, standardized by the σ of the last 8 surprises.
- **Ranking** uses prior events' SUE breakpoints (Mohanram 2009) so quintile
  assignment is implementable at the event date — no lookahead anywhere.
- **CAR** = ticker buy-and-hold return minus equal-weight universe return over
  [+0, +h] trading days from the first trading day at/after `filed_at`.

Live result (27 mega-caps, 273 filings, ~19y daily prices): **no significant
drift**. Bad-news (bottom-SUE) filings drift down (car20 ≈ −1.7%), but the
top-SUE quintile shows no drift up; the Q5−Q1 spread is +0.4/+0.9/+0.9 bps at
1/5/20 days (t < 2). This is the expected finding, not a bug: Bernard & Thomas
document drift concentrated in small caps and on **quarterly** announcements;
annual bottom-line net income for mega-caps is the noisiest, most efficiently
priced setting. The pipeline itself is the deliverable — PIT-correct SUE/CAR
with zero lookahead, unit-tested (`tests/test_pead.py`). To reproduce the
textbook drift, extend the EDGAR provider to quarterly 10-Q filings (fp != FY)
and compute EPS-based surprises.

## Quality gates

- `make check` = `ruff` + `pytest` (110 tests, offline — all network/DB mocked).
- `make dbt-parse` validates models/contracts without a Snowflake connection.
- CI (`.github/workflows/quant-signal-ci.yml`) runs lint + test + dbt parse on
  every PR touching `quant_signal/`.
