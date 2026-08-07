-- Crypto 5m window features from the raw Binance minute-bar topic.
--
-- Real-time stream processing (event-time windows, checkpointed state):
--   crypto.bars.raw  (1m OHLCV from the Kafka ingestion bus)
--     ──> TUMBLE(5 MINUTE) keyed by symbol ──>
--   crypto.features.5m  (windowed OHLCV + VWAP + bar_count)
--
-- Event time is the bar's openTime (epoch ms). Watermarks tolerate the
-- in-progress-minute updates the producer re-publishes. State is checkpointed
-- every 30s so the job resumes from the last consistent point on failure.

SET 'execution.checkpointing.interval' = '30s';
SET 'execution.checkpointing.externalized-checkpoint-retention' = 'RETAIN_ON_CANCELLATION';
SET 'table.local-time-zone' = 'UTC';

CREATE TABLE raw_bars (
  symbol STRING,
  ts BIGINT,
  ts_ltz AS TO_TIMESTAMP_LTZ(ts, 3),
  `open` DOUBLE,
  `high` DOUBLE,
  `low` DOUBLE,
  `close` DOUBLE,
  volume DOUBLE,
  WATERMARK FOR ts_ltz AS ts_ltz - INTERVAL '2' MINUTE
) WITH (
  'connector' = 'kafka',
  'topic' = 'crypto.bars.raw',
  'properties.bootstrap.servers' = 'redpanda:29092',
  'properties.group.id' = 'flink-crypto-features',
  'format' = 'json',
  'scan.startup.mode' = 'latest-offset'
);

CREATE TABLE features (
  symbol STRING,
  window_start_ms BIGINT,
  window_end_ms BIGINT,
  `open` DOUBLE,
  `high` DOUBLE,
  `low` DOUBLE,
  `close` DOUBLE,
  volume DOUBLE,
  vwap DOUBLE,
  bar_count BIGINT
) WITH (
  'connector' = 'kafka',
  'topic' = 'crypto.features.5m',
  'properties.bootstrap.servers' = 'redpanda:29092',
  'format' = 'json'
);

INSERT INTO features
SELECT
  symbol,
  UNIX_TIMESTAMP(CAST(TUMBLE_START(ts_ltz, INTERVAL '5' MINUTE) AS STRING)) * 1000 AS window_start_ms,
  UNIX_TIMESTAMP(CAST(TUMBLE_END(ts_ltz, INTERVAL '5' MINUTE) AS STRING)) * 1000 AS window_end_ms,
  FIRST_VALUE(`open`) AS `open`,
  MAX(`high`) AS `high`,
  MIN(`low`) AS `low`,
  LAST_VALUE(`close`) AS `close`,
  SUM(volume) AS volume,
  SUM(`close` * volume) / SUM(volume) AS vwap,
  COUNT(*) AS bar_count
FROM raw_bars
GROUP BY symbol, TUMBLE(ts_ltz, INTERVAL '5' MINUTE);
