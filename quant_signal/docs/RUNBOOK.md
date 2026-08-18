# Runbook

Operating the stack locally and on AWS. The [README](../README.md) covers what the system is; this covers running it.

---

## Dashboard + live market stream

```bash
make api   # FastAPI on :8000 (needs .env; MFA passcode via env or prompt)
make ui    # Next.js on :3000 — proxies /api/* to :8000, same-origin
```

Open `http://localhost:3000`. Direct API calls:

```bash
curl localhost:8000/api/pead
curl localhost:8000/api/market/AAPL?days=750
curl localhost:8000/api/fundamentals/AAPL
curl localhost:8000/api/metrics/pipeline
curl localhost:8000/api/market/live/BTCUSDT          # live bars (from Kafka)
curl localhost:8000/api/market/features/BTCUSDT      # 5m features (from Redis)
curl localhost:8000/api/market/gate/BTCUSDT          # promotion-gate verdict
curl localhost:8000/api/market/quality               # data quality + lineage
curl localhost:8000/api/market/research/BTCUSDT      # research leaderboard
```

Live WebSocket: `wscat -c 'ws://localhost:8000/ws/market?symbol=BTCUSDT'`.
The stream is on by default; set `STREAM_ENABLED=false` for a pure query API.

---

## Streaming stack (Kafka → Flink → Redis)

```bash
make stream-infra            # docker compose up -d --build (redpanda, redis, flink)
make stream-topics           # create Kafka topics (Flink requires them to exist)
make stream-flink-submit     # deploy crypto_features.sql (5m windows, detached)
make stream-flink-submit-1h  # deploy crypto_features_1h.sql (1h windows)
make stream-flink-status     # expect RUNNING
```

Then, in two terminals:

```bash
make stream-producer         # poll Binance → publish raw bars (real market data)
make stream-materializer     # consume raw + features → Redis
make stream-srp              # score the SRP weekly book → srp:weights:*
make stream-srp-execute      # trade that book (paper; --venue bybit-demo for demo fills)
```

For a fast end-to-end check without Binance, seed synthetic bars:
`make stream-seed` (the Flink window fires within ~2 minutes).

> **Notes.** Topics must exist before the Flink job starts. Redis maps to host
> **:6380** (`STREAM_REDIS_URL=redis://localhost:6380`). The Flink image pins
> the Kafka connector JAR with correct ownership and a writable checkpoint dir,
> or the job restarts with `ClassNotFoundException` / `IOException`.

---

## Research harness (Build #1)

Sweep the predictor's config grid offline against the same learn-then-predict
loop the live model uses; every run is tracked to MLflow and per-symbol
leaderboards land in Redis.

```bash
make stream-research
curl localhost:8000/api/market/research/BTCUSDT
```

While the grid is cold the leaderboard honestly reports `winner: None`.

---

## Data quality / SLA (Build #2)

Score the online store across **8 dimensions** (freshness, latency, volume
coverage, completeness, ordering, schema, value sanity, duplicate-free) with a
per-window **lineage manifest**, so a degraded number is traceable to the
source that caused it.

```bash
make stream-quality
curl localhost:8000/api/market/quality
```

Warm-up is reported honestly — a fresh 1h pipeline shows volume `critical`,
never silently ignored.

---

## Data lake — Iceberg (Build #4)

The Snowflake mart (`GOLD.FEATURES`, ...) is versioned to an **Apache Iceberg**
table on S3-compatible object storage. Iceberg is storage-agnostic: the catalog
(SQLite locally, JDBC/Postgres in a shared deployment) tracks schema +
snapshots while Parquet data files live on object storage. Each export is an
idempotent *overwrite* that adds a **new snapshot** — old snapshots stay
readable, so the lake is versioned with time travel, not replaced. The same
tables move to real S3 unchanged by swapping `LAKE_ENDPOINT` (the AWS IaC
storage module already provisions the bucket).

The lake is built to scale:

- **Coarse partitioning.** Tables are partitioned on the coarsest grain that
  prunes the dominant query pattern (`LAKE_PARTITION_BY`, default `SYMBOL` = 32
  low-cardinality values). Iceberg's rule is to stay *well under a few thousand
  partitions* — finer grains turn metadata planning into the scan bottleneck.
  Partitioning is applied by column name at create time (Iceberg specs are
  fixed then; the pyiceberg-recommended `create_table_transaction` +
  `update_spec().add_identity` path, which sidesteps the pyarrow↔Iceberg field-id
  mismatch in apache/iceberg-python#1100).
- **Bounded time travel.** Unbounded snapshot history is the #1 Iceberg metadata
  trap at scale — manifest bookkeeping becomes the latency floor. Every export
  expires snapshots older than `LAKE_SNAPSHOT_RETENTION_HOURS` (7 days default),
  so versioning is bounded regardless of run frequency.

```bash
uv sync --extra lake
make lake-export          # GOLD.FEATURES → gold.features (new snapshot)
make lake-query           # round-trip read back + snapshot history (time travel)
```

Credentials and the bucket come from the environment (`LAKE_ENABLED`,
`LAKE_ACCESS_KEY`, `LAKE_SECRET_KEY`, `LAKE_BUCKET`, `LAKE_PARTITION_BY`,
`LAKE_SNAPSHOT_RETENTION_HOURS`, ...) — never hardcoded.

---

## Prediction promotion gate

Before the live predictor may emit LONG/SHORT, `stream/predictive_eval.py`
replays its exact learn-then-predict loop over stored feature history and
checks: progressive validation, transaction-cost-adjusted P&L, IC / direction
accuracy vs naive baselines, conformal coverage, block stability, and
**Deflated Sharpe** (corrected for multiple testing). A model may learn but not
trade until `passes_gate()` clears everything:

```bash
curl localhost:8000/api/market/validation/BTCUSDT?track=true   # → MLflow
curl localhost:8000/api/market/gate/BTCUSDT
```

The gate needs ≥ `stream_gate_min_windows` (default 100) scored windows before
it can even pass the warm-up. The live Flink 1h job starts from Kafka's
`latest-offset` after a heal, so the online store only accrues windows going
forward — a fresh pipeline would take ~4 days to warm up. Seed the store from
the Snowflake engine of record (the same Bybit minute bars Flink consumes,
aggregated to the identical TUMBLE(1H) OHLCV+VWAP windows):

```bash
make stream-backfill-windows   # or: python -m scripts.backfill_feature_windows [--dry-run]
```

Idempotent: each symbol's key is atomically rebuilt with the newest
`stream_redis_feature_maxlen` (200) closed windows, and the live materializer
keeps appending new windows afterward.

---

## Pipeline health + watchdog

- `GET /api/market/health/summary` → per-stage freshness (produce / features /
  predict / simulate / strategy) rendered as a 6-stage LED strip.
- `scripts/stream_watchdog.py` restarts a stalled Flink job with `--fix`:

```bash
make stream-watchdog          # or: python -m scripts.stream_watchdog --interval 60 --fix
```

---

## Reproduce the validation

Every claim in this repository is behind a script. Nothing below needs a
Snowflake account, an API key, or a paid data feed:

```bash
# 1. The deploy gate. Non-zero exit if research and live disagree.
uv run python -m scripts.srp_parity

# 2. The backtest itself, net of funding and liquidity-scaled maker costs.
uv run python -m scripts.srp_backtest

# 3. Out-of-sample: config chosen on trailing data only, scored on unseen blocks,
#    then repeated on all seven weekday rebalance anchors.
uv run python -m scripts.srp_walkforward --mode both

# 4. Which construction choice earns what — paired, best-of-grid, main effects.
uv run python -m scripts.srp_ablation

# 5. Multiple-testing correction, N and dispersion read from the trial registry.
uv run python -m scripts.trial_registry     # inspect what was actually executed
uv run python -m scripts.srp_dsr
```

The full research record — every number labelled `[VERIFIED]`, `[PENDING]`, or
`[UNVERIFIED]`, including the claims that **failed** audit and why — is in
[`research/SRP_RESEARCH_LOG.md`](../research/SRP_RESEARCH_LOG.md).

---

---

## Snowflake bootstrap (one-time)

1. In Snowsight create a warehouse `QUANT_WH` (XS, auto-suspend 60s).
2. Copy `.env.example` → `.env` and fill in the `SNOWFLAKE_*` and `DBT_*`
   variables.
3. Bootstrap and verify:

```bash
make bootstrap    # idempotent: DB, warehouse, schemas, least-privilege roles
make ping         # SELECT 1 over the live connection
make dbt-debug    # verify the dbt connection
make dbt-deps     # install dbt_expectations + elementary
make dbt-run      # build silver/gold with enforced contracts + tests
```

---

## Using the client

```python
from db.snowflake import SnowflakeClient

client = SnowflakeClient()
print(client.ping())                           # True once connected
df = client.query_df("SELECT 1 AS one")
client.insert_df(df, table_name="my_table")    # appends into QUANT.BRONZE
```

---

---

## AWS deployment

The same platform as managed infrastructure in `infra/terraform/`:
**MSK** (managed Kafka) → **Flink on Fargate** → **S3 checkpoints** →
**ElastiCache Redis** → ECS agents + Next.js UI behind an **ALB**, with
CloudWatch alarms + SNS + a platform dashboard.

Highlights (research-backed): Service Connect for in-cluster DNS, task-role-only
credentials (Bybit demo keys via Secrets Manager — never env or logs), and a
3-broker MSK cluster as the Kafka durability floor.

```
infra/terraform/
├── modules/            networking · storage · msk · iam · ecr · ecs · observability
└── environments/dev/   S3 backend + DynamoDB lock, module wiring, outputs
```

```bash
cd infra/terraform/environments/dev
terraform init          # configures the S3 backend
terraform plan
terraform apply
```

**Full runbook** (backend bootstrap, secrets, Flink job submission via
`ecs exec`): [`infra/terraform/README.md`](../infra/terraform/README.md).

---

---

