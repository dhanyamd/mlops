# Snowpipe Streaming persistence — upgrade path for the real-time layer

Status: **guidance / design** (documented, not yet implemented). The real-time
layer persists `BRONZE.CRYPTO_BARS` today via **best-effort batch writes** from
`stream/producer.py` (see "Honest gaps" in `docs/architecture.md`). This doc
records the upgrade path: get the Kafka stream durably into Snowflake with
Snowpipe Streaming, at ~5–10s end-to-end latency and exactly-once semantics,
without the batch best-effort writes.

## Current state (the gap)

```
BinanceProducer → Redpanda (crypto.bars.raw) ──► Flink (features) → Redis
      │  (best-effort write_crypto_bars per poll)
      └─────────────────────────────────────────► BRONZE.CRYPTO_BARS (upsert)
```

- The producer also calls `write_crypto_bars` on each poll (see
  `stream/producer.py:persist`). This is **best-effort**: a producer restart or
  transient Snowflake/MFA hiccup can drop a window, and every poll is a
  Snowflake round-trip on the hot path.
- Redis is the serving path (sub-500ms reads) and is *not* the durability
  story. Snowflake BRONZE is the analytics/backtest source of truth.

## The upgrade

Run a **Snowpipe Streaming ingest client** against `crypto.bars.raw` as a
sidecar of the stream (no change to producer/Flink/Redis). Three viable
clients, best-first:

### 1. Snowflake Connector for Kafka v4 (recommended)

Ground-up rewrite built on the Snowpipe Streaming high-performance
architecture: up to ~10 GB/s per table, 5–10s latency, exactly-once + ordered
delivery. Requires Kafka Connect (Java 11+) — the one dependency this repo
currently doesn't have.

Key config for this repo (Redpanda is Kafka-API compatible, so it drops in):

```properties
name=snowflake-crypto-bars
connector.class=com.snowflake.kafka.connector.SnowflakeStreamingSinkConnector
# Kafka/Redpanda source
topics=crypto.bars.raw
# Snowflake target — mirror the dbt/dev env vars
snowflake.url.name=<account>.snowflakecomputing.com
snowflake.user.name=<user>
snowflake.private.key=<...>          # v4 requires key-pair auth
snowflake.database.name=QUANT
snowflake.schema.name=BRONZE
snowflake.enable.schematization=true  # map record fields to table columns
# Exactly-once + ordered per partition; dedupe helper columns
snowflake.metadata.topic=true
snowflake.metadata.offset.and.partition=true
snowflake.metadata.createtime=true
```

Notes:

- v4 default `snowflake.enable.schematization=true` maps the bar record's
  fields (symbol, ts, timeframe, OHLCV, volume, provider, loaded_at) onto the
  existing `BRONZE.CRYPTO_BARS` columns. Pre-create the table with the exact
  contract so the connector matches it (matching the medallion discipline).
- Daily-idempotency stays intact: the table's natural key
  `(symbol, timeframe, ts)` MERGE-upsert in the batch path is untouched; the
  stream only *adds* rows. A bar re-published by the producer (the
  in-progress-minute re-publish Flink already tolerates) is deduped with a
  MERGE or `QUALIFY` on `RECORD_METADATA:offset` if it ever lands twice.
- Monitor lag with JMX: `persisted-in-snowflake-offset` vs
  `latest-consumer-offset` (Snowpipe Streaming auto-flushes ~1s; the connector
  buffers up to `buffer.flush.time`, default 10s).

### 2. Spark Structured Streaming (fits the M4 Spark batch)

The M4 `flows/feature_engineering.py --source snowflake` path already brings
PySpark + the spark-snowflake connector. A small structured-streaming job is
the natural sidecar: `readStream` from Redpanda (Kafka source) →
`writeStream` with `format("snowflake")`. Same Snowpipe Streaming plumbing
underneath; keeps the stack Java-free except Spark (Java is already an M4
dependency for the batch). Good middle ground if Kafka Connect feels heavy.

### 3. Snowflake Ingest SDK (Python)

Direct `snowflake.ingest` channel API from the producer: open one
**long-lived** channel per symbol, `insertRows` per poll, track
`getLatestCommittedOffsetToken` as the source of truth on recovery. Lowest
infra, but you re-implement exactly-once recovery yourself — only worth it
when Kafka Connect and Spark are both out of scope.

## Best practices to carry in (from Snowpipe Streaming docs)

- **Long-lived, deterministically named channels** (e.g. `crypto-bars-<env>-<symbol>`);
  never open/close per poll.
- **Metadata columns** `CHANNEL_ID` + `STREAM_OFFSET` on rows → enables
  gap-detection and replay (query for gaps in `STREAM_OFFSET`).
- **Offset tokens are the source of truth**: on restart, reopen the channel and
  resume from the latest committed offset token (not the Kafka offset).
- **Handle 409 (channel invalidated) and 429 (throttle)** with exponential
  backoff; wrap `insertRows` in try/except.
- **Match by column name** (`MATCH_BY_COLUMN_NAME`) instead of a VARIANT blob
  — bill only for values, and keep the typed bar contract.
- **Durable, cheap**: Snowpipe Streaming flushes ~1s by default
  (`MAX_CLIENT_LAG`); the high-performance client enables jemalloc, ZSTD
  compression, and Prometheus metrics (`SS_ENABLE_METRICS=true`).

## What does NOT change

- Redis stays the low-latency serving store (Flink features + online store).
- The batch MERGE upsert path stays for the equity providers and as the
  reconciliation/backfill path.
- The dbt Silver/Gold models, contracts, and the `CI_*` build stay exactly as
  they are — the stream now just feeds BRONZE more reliably.
