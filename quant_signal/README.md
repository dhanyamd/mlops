# quant_signal

Production-grade quant signal platform: **Snowflake-backed pipelines** with
**Spark**, data quality, and full observability. Built from research on how
quant research houses (Two Sigma / Man AHL patterns) actually run data
infrastructure.

> Status: **Full build — M0–M5 plus the prediction promotion gate.** Config,
> Snowflake client, structured logging, idempotent bootstrap, dbt with enforced
> contracts, real ingestion (Yahoo/EDGAR/FRED/Binance/Alpaca) into
> Bronze→Silver→Gold with latency telemetry, the PEAD event study, and a
> read-only FastAPI + Next.js dashboard. A production streaming layer — Redpanda
> (Kafka API) → Flink SQL (5m event-time windows) → Redis online store → API —
> powers real-time crypto features, online prediction with conformal intervals,
> Monte Carlo simulation, and QuantPad-style strategy validation. A **promotion
> gate** (progressive validation + Deflated Sharpe) replays real feature
> history and decides, honestly, whether a model may actually trade. The Signal
> Terminal renders the full Monte Carlo visualization stack: a forward fan
> chart with the 10k-simulated path spaghetti (QuantPad "all paths"), an
> outcome-colored equity fan, an efficiency-cloud pass-probability landscape
> (pain vs. gain scatter), and terminal / drawdown distributions. No code is
> hardcoded: every credential and connection value comes from the environment.

## Non-negotiables

- **Nothing hardcoded.** All config via `pydantic-settings` from env vars
  (`config/settings.py`). Secrets use `Field(repr=False)` (never logged),
  dbt's `DBT_ENV_SECRET_*` prefix (auto-scrubbed from logs), and every query
  carries a `query_tag` for credit/cost attribution.
- **Fail fast.** A missing auth method or a malformed account identifier
  raises `ValidationError` at startup — misconfig never silently half-runs.
- **Reproducible infra.** `make bootstrap` idempotently creates the database,
  warehouse, schemas, and least-privilege roles (`QUANT_INGEST` /
  `QUANT_TRANSFORMER` / `QUANT_READER`).
- **Quality as contract.** Every silver/gold dbt model must declare a full
  column contract (`contract: enforced: true`) or the build fails.
- **Structured logs.** JSON lines via `structlog`; secrets never logged.

## Stack (this folder)

| Concern | Tool |
|---|---|
| Config | `pydantic-settings` |
| Warehouse | Snowflake (`snowflake-connector-python`) |
| Transform + quality | `dbt-core` + `dbt-snowflake` (contracts, tests) |
| Statistical tests | `dbt_expectations` |
| Data observability | `elementary` (anomaly/schema-drift detection) |
| Stream broker | Redpanda (Kafka API) — `confluent-kafka` producer/consumer |
| Stream processing | Flink SQL (`flink:1.19.3`), checkpointed event-time windows |
| Online store | Redis 7 (`redis-py`), bounded feature lists |
| Orchestration (later) | Prefect |
| Distributed compute (later) | Spark + `spark-snowflake` connector |

## Layout

```
quant_signal/
├── config/
│   ├── settings.py         # pydantic-settings: env-driven, secrets masked
│   └── logging.py          # structlog JSON logging (level from env)
├── db/
│   ├── snowflake.py        # SnowflakeClient: query/insert/ping, query_tag
│   ├── bootstrap.py        # idempotent infra bootstrap (make bootstrap)
│   └── sql/bootstrap.sql   # DB, warehouse, schemas, least-privilege roles
├── dbt/
│   ├── profiles.yml        # creds ONLY via env_var(...), never plaintext
│   ├── dbt_project.yml     # silver/gold: contracts enforced at project level
│   ├── packages.yml        # dbt_expectations + elementary
│   └── models/             # sources.yml + silver/gold models with contracts
├── api/
│   ├── main.py             # FastAPI: /api/market, /pead, /fundamentals, /ws/market
│   ├── db.py               # read-only query layer over Silver/Gold
│   └── stream.py           # live stream hub: Kafka consumer + ring buffer + WS fan-out
├── stream/                 # M3 streaming stack (all bus/KV logic testable via fakes)
│   ├── producer.py         # standalone Binance minute-bar producer → Kafka
│   ├── materializer.py     # Kafka → Redis online store (live bars + 5m features)
│   ├── predictor.py        # online River model + conformal intervals → Redis
│   ├── predictive_eval.py  # promotion gate: progressive validation + Deflated Sharpe
│   ├── mlflow_tracking.py  # MLflow tracking for validation + gate runs (optional extra)
│   ├── simulation.py       # Monte Carlo forward fan chart (percentiles, VaR/ES)
│   ├── strategy_mc.py      # QuantPad-style strategy pass-probability bootstrap
│   ├── bus.py              # MessageBus (KafkaBus / FakeBus for hermetic tests)
│   ├── kv.py               # KVStore (RedisKV / FakeKV)
│   ├── bars.py             # provider DataFrames → JSON bar payloads
│   └── flink/              # Dockerfile + crypto_features.sql (5m event-time windows)
├── docker-compose.yml      # redpanda + redis + flink-jobmanager/taskmanager
├── scripts/                # ping.py, run_dbt.py, pead_backtest.py, seed_stream_demo.py
├── ui/                     # Next.js dashboard (Market/Fundamentals/PEAD/...)
├── tests/                  # config + connection-param + API tests (no live DB)
└── Makefile                # setup / lint / test / bootstrap / dbt / api / ui / stream-*
```

## Setup (needs a Snowflake trial account)

1. Sign up at `signup.snowflake.com` (free trial, $400 credits, no card).
2. In Snowsight, create a warehouse `QUANT_WH` (XS, auto-suspend 60s) under
   **Admin → Warehouses** (or note an existing one like `COMPUTE_WH`).
3. Copy `.env.example` → `.env` and fill in your Snowflake values (both the
   `SNOWFLAKE_*` and matching `DBT_*` variables). Your password is the one you
   set at signup — Snowflake never shows it back.
4. Install, verify, bootstrap:

```bash
cd quant_signal
make setup        # uv sync (runtime + dev deps)
make dbt-setup    # uv sync (adds dbt-core + adapter)
make check        # ruff + pytest — no Snowflake needed
make bootstrap    # creates QUANT DB, QUANT_WH, schemas, roles (idempotent)
make ping         # verifies the live connection (SELECT 1)
make dbt-debug    # verifies the dbt connection
make dbt-deps     # installs dbt_expectations + elementary
make dbt-run      # builds silver/gold with contracts + tests
```

## Using the client

```python
from db.snowflake import SnowflakeClient

client = SnowflakeClient()
print(client.ping())                       # True once connected
df = client.query_df("SELECT 1 AS one")
client.insert_df(df, table_name="my_table")  # appends into QUANT.BRONZE
```

## Dashboard (live UI over the marts)

The read-only dashboard serves the *same* numbers the CLI shows — market bars,
PIT fundamentals, the PEAD event study, macro series, and pipeline latency.
Nothing is mocked; every page queries Silver/Gold in real time.

```bash
make api   # FastAPI on :8000 (needs .env; MFA passcode via env or prompt)
make ui    # Next.js on :3000 — proxies /api/* to :8000, so browsers stay same-origin
```

Open `http://localhost:3000`. The API is also directly usable:
`curl localhost:8000/api/pead`, `/api/market/AAPL?days=750`, `/api/fundamentals/AAPL`,
`/api/metrics/pipeline`, `/api/macro?series=VIXCLS`,
`/api/market/gate/BTCUSDT`, and `/api/market/validation/BTCUSDT?track=true`
(track logs the run to MLflow). The PEAD endpoint recomputes
the ~10s event study at most once per 60s (env `API_PEAD_CACHE_TTL_SECONDS`).

### Live market stream (near-real-time showcase)

While the API runs it also drives a **live minute-bar stream** from Binance
(`api/stream.py`): a background poller fetches recent crypto bars every
`STREAM_POLL_SECONDS`, upserts them to `BRONZE.CRYPTO_BARS` (best-effort —
a Snowflake outage degrades to a warning, never kills the stream), and
broadcasts deltas over WebSocket. No hardcoded instruments: the symbol set is
`INGEST_DEFAULT_CRYPTO_SYMBOLS` from the environment.

- Snapshot REST: `curl localhost:8000/api/market/live/BTCUSDT`
- Live WebSocket: `wscat -c 'ws://localhost:8000/ws/market?symbol=BTCUSDT'`

The stream is on by default; set `STREAM_ENABLED=false` for a pure query API.

### Streaming stack (M3): Kafka → Flink → Redis

The **real-time feature pipeline** runs alongside the Snowflake batch layer:
a standalone producer publishes Binance minute bars to Redpanda (Kafka API),
a **Flink SQL job** computes 5-minute event-time windows (OHLCV, VWAP,
bar count) with checkpoints, and a **materializer** lands both live bars and
window features into a Redis **online store**. The API serves them from Redis
(<500ms, never touching Kafka or Snowflake):

```
BinanceProducer → Redpanda(crypto.bars.raw) → Flink SQL 5m TUMBLE
   → Redpanda(crypto.features.5m) → materializer → Redis → /api/market/*
```

Bring the stack up (needs Docker):

```bash
make stream-infra            # docker compose up -d --build (redpanda, redis, flink)
make stream-topics           # create the Kafka topics (Flink requires them to exist)
make stream-flink-submit     # deploy the crypto_features.sql job (detached)
make stream-flink-status     # check job state (expect RUNNING)
```

Then run the live ingestion + online store in two terminals:

```bash
make stream-producer         # poll Binance → publish raw bars (real market data)
make stream-materializer     # consume raw + features → Redis
```

For a fast end-to-end check without waiting on Binance, seed synthetic bars
into the current + previous 5-minute buckets (the Flink window fires within
~2 minutes): `make stream-seed`. Verify with:

```bash
curl localhost:8000/api/market/live/BTCUSDT         # hub ring buffer (from Kafka)
curl localhost:8000/api/market/features/BTCUSDT     # 5m features (from Redis)
```

Notes:
- The Kafka topics **must exist** before the Flink job starts (`make stream-topics`
  is idempotent); Redpanda auto-creates topics on produce, but Flink's source
  enumerator fails with `UnknownTopicOrPartitionException` if they're absent.
- Redis maps to host **:6380** so it never collides with a host Redis on :6379
  (`STREAM_REDIS_URL=redis://localhost:6380`).
- The Flink image pins the Kafka connector JAR with correct ownership
  (`chown flink:flink`, `chmod 644`) and a writable checkpoint dir — both are
  required or the job restarts on `ClassNotFoundException`/`IOException`.
- The old M2 in-API poller is demo-grade; the standalone `stream-producer` +
  Kafka path is the production ingestion route. Streaming writes to Snowflake
  only best-effort today (Kafka → Snowflake via Snowpipe Streaming is planned).

### Prediction promotion gate (validation before trading)

The research literature is blunt: single-asset, next-window return prediction
is the *weakest* return-prediction setting (Gu, Kelly & Xiu 2020), and naive
backtesting is "the most dangerous tool in finance" (López de Prado). So before
the live predictor is allowed to emit LONG/SHORT, `stream/predictive_eval.py`
replays its exact learn-then-predict loop over the stored feature-window
history and scores it honestly:

- **progressive validation** (test-then-train, River's canonical protocol);
- **transaction-cost-adjusted P&L** — taker cost charged per position flip, vs
  buy-and-hold;
- **IC + direction accuracy**; **naive baselines** (predict-zero and
  persistence) reported as MASE skill;
- **conformal coverage** vs nominal alpha (the ACI contract);
- **stability across contiguous blocks** (signal must not be one lucky period);
- **Deflated Sharpe Ratio** charged for the number of trials the model has gone
  through — the multiple-testing correction (Bailey & López de Prado).

The verdict is `passes_gate()`: a model may learn but must not trade until it
clears every check. It is served by `/api/market/gate/{symbol}` and recorded to
MLflow with `?track=true` (filterable by a `passes` 0/1 metric). All thresholds
are env-driven (`STREAM_GATE_*` in `config/settings.py`); the default is
deliberately strict — most real models will fail, which is the point.

## Notes

- **No API key.** Snowflake authenticates with account + user + password (or
  RSA key-pair via `SNOWFLAKE_PRIVATE_KEY_FILE`). Snowflake is SaaS-only; your
  machine talks to your cloud trial account.
- **MFA.** New Snowflake accounts require MFA for password logins. Enroll in
  Snowsight (user menu → Settings → Authentication → Duo or an authenticator
  app — **not** a passkey, which can't be used programmatically), set
  `SNOWFLAKE_USE_MFA=true`, and the connector uses `username_password_mfa`
  with Duo push. The first connection prompts once; `ALLOW_CLIENT_MFA_CACHING`
  (set by `make bootstrap`) caches the token ~4h. For fully non-interactive
  automation later, use key-pair auth instead (MFA doesn't apply to it).
- **Spark → Snowflake** (next milestones): the `spark-snowflake` connector has
  no native stream sink — you write via `foreachBatch`. Streaming holds
  sessions/stages open and burns credits; keep the warehouse small +
  auto-suspend.
