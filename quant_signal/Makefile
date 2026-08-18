PYTHON := uv run
DBT := cd dbt && uv run dbt
# dbt reads profiles/dbt_project from the dbt/ dir; credentials come from ./.env
ENV := set -a; . ./.env; set +a;

.PHONY: setup lint test check dbt-setup dbt-deps dbt-debug dbt-run dbt-parse bootstrap ping ingest ingest-macro ingest-fundamentals api ui-setup ui stream-infra stream-topics stream-flink-submit stream-flink-submit-1h stream-flink-status stream-producer stream-materializer stream-predictor stream-simulator stream-execution stream-seed stream-watchdog stream-research stream-quality

## Environment ────────────────────────────────────────────────────────────────

setup: ## Create venv + install all extras (runtime, dev, dbt, api, spark, mlflow)
	uv sync --all-extras

dbt-setup: ## Install dbt-core + dbt-snowflake adapter (with dev tools)
	uv sync --all-extras

## Quality gates ──────────────────────────────────────────────────────────────

lint: ## Ruff lint check
	uv run ruff check .

test: ## Run pytest (no Snowflake account required)
	uv run pytest

check: lint test ## CI-style quality gate

## Snowflake ──────────────────────────────────────────────────────────────────

bootstrap: ## Create DB, warehouse, schemas, roles (idempotent, needs .env)
	$(ENV) $(PYTHON) python -m db.bootstrap

ping: ## Verify the Snowflake connection (SELECT 1)
	$(ENV) $(PYTHON) python scripts/ping.py

## Ingestion ─────────────────────────────────────────────────────────────────

ingest: ## Land REAL daily equity bars (default provider/symbols from Settings/env)
	$(ENV) $(PYTHON) python flows/ingest_market_data.py

ingest-crypto: ## Land REAL crypto minute bars (Binance, keyless)
	$(ENV) $(PYTHON) python flows/ingest_market_data.py --provider binance --days 3

ingest-macro: ## Land REAL FRED macro series (keyless CSV download)
	$(ENV) $(PYTHON) python flows/ingest_macro_data.py

ingest-fundamentals: ## Land REAL SEC EDGAR fundamentals (keyless, needs EDGAR_USER_AGENT)
	$(ENV) $(PYTHON) python flows/ingest_fundamentals.py

## Feature engineering (M4) ───────────────────────────────────────────────────

features: ## Compute rolling/as-of features over BRONZE bars -> GOLD.FEATURES (no Java needed)
	$(ENV) $(PYTHON) python flows/feature_engineering.py --source pandas

features-spark: ## Same batch via PySpark (needs Java 17+; connector JAR auto-fetched by Spark)
	$(ENV) $(PYTHON) python flows/feature_engineering.py --source snowflake

## Data lake (Iceberg on MinIO/S3) ─────────────────────────────────────────────

lake-export: ## Version the Snowflake mart to an Iceberg table on MinIO (needs `lake` extra)
	$(ENV) $(PYTHON) python flows/lake_export.py --mode export

lake-query: ## Round-trip verify: read the Iceberg table back + show snapshot history
	$(ENV) $(PYTHON) python flows/lake_export.py --mode verify

## dbt ────────────────────────────────────────────────────────────────────────

dbt-deps: ## Install dbt packages (dbt_expectations, elementary)
	$(ENV) $(DBT) deps --project-dir . --profiles-dir .

dbt-debug: ## Validate Snowflake connection + profile (no transforms)
	$(ENV) $(DBT) debug --project-dir . --profiles-dir .

dbt-parse: ## Compile the project without running (fast contract check)
	$(ENV) $(DBT) parse --project-dir . --profiles-dir .

dbt-run: ## Build + test all models (silver/gold, contracts enforced) with latency telemetry
	$(ENV) $(PYTHON) python scripts/run_dbt.py build --project-dir . --profiles-dir .

## Warehouse sink (Redis online store → ClickHouse BRONZE) ───────────────────

ch-materialize: ## Snapshot the live Redis online store into ClickHouse (idempotent upsert)
	$(ENV) $(PYTHON) python flows/materialize_clickhouse.py

## Streaming stack (M3: Redpanda/Kafka + Flink + Redis) ──────────────────────

FLINK_JM := quant_signal-flink-jobmanager-1
REDPANDA := quant_signal-redpanda-1

stream-infra: ## Start Redpanda + Redis + Flink (builds the Flink image)
	docker compose up -d --build

stream-topics: ## Create the Kafka topics (idempotent; Flink needs them to exist)
	docker exec $(REDPANDA) rpk topic create crypto.bars.raw crypto.features.5m crypto.features.1h --partitions 1 --replicas 1 --if-not-exists

stream-flink-submit: ## Submit the crypto 5m-window SQL job (detached; benchmark/Alpha-Autopsy book)
	docker exec $(FLINK_JM) bash -c 'sql-client.sh -f /opt/flink/jobs/crypto_features.sql -d'

stream-flink-submit-1h: ## Submit the crypto 1h-window SQL job (detached; live trading clock)
	docker exec $(FLINK_JM) bash -c 'sql-client.sh -f /opt/flink/jobs/crypto_features_1h.sql -d'

stream-flink-status: ## Show Flink job states
	docker exec $(FLINK_JM) bash -c 'curl -s http://localhost:8081/jobs/overview'

stream-producer: ## Run the live Binance ingestion producer (polls + publishes)
	$(ENV) $(PYTHON) python -m stream.producer

stream-materializer: ## Materialize Kafka → Redis online store (run alongside producer)
	$(ENV) $(PYTHON) python -m stream.materializer

stream-predictor: ## Train the online model + emit conformal predictions to Redis
	$(ENV) $(PYTHON) python -m stream.predictor

stream-simulator: ## Run the Monte Carlo forward simulator to Redis
	$(ENV) $(PYTHON) python -m stream.simulation

stream-srp: ## Score the SRP weekly book and publish target weights to Redis
	$(ENV) $(PYTHON) python -m stream.srp_publisher

stream-srp-execute: ## Trade the SRP weekly book (paper fills; --venue bybit-demo for the demo venue)
	$(ENV) $(PYTHON) python -m stream.srp_execution

stream-execution: ## Run the paper-execution engine (fills, slippage, fees, ledger) to Redis
	$(ENV) $(PYTHON) python -m stream.execution

stream-seed: ## Publish synthetic demo bars (fast E2E without waiting on Binance)
	$(ENV) $(PYTHON) python -m scripts.seed_stream_demo

stream-watchdog: ## Detect a stalled Flink pipeline (features behind raw bars); --fix heals it
	$(ENV) $(PYTHON) python -m scripts.stream_watchdog --interval 60 --fix

stream-research: ## Sweep the predictor's config grid offline; persist per-symbol leaderboards
	$(ENV) $(PYTHON) python -m scripts.run_research

stream-backfill-windows: ## Seed the online store's 1h feature history from Snowflake BRONZE (idempotent)
	$(ENV) $(PYTHON) python -m scripts.backfill_feature_windows

stream-quality: ## Score every symbol's window history + lineage; persist quality summaries
	$(ENV) $(PYTHON) python -m scripts.run_quality

## Dashboard ─────────────────────────────────────────────────────────────────

api: ## Run the read-only dashboard API (FastAPI) on :8000
	$(ENV) $(PYTHON) uvicorn api.main:app --port 8000 --reload

ui-setup: ## Install the Next.js dashboard dependencies
	cd ui && npm install

ui: ## Run the Next.js dashboard on :3000 (proxy /api → :8000; start `make api` first)
	cd ui && npm run dev
