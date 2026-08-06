PYTHON := uv run
DBT := cd dbt && uv run dbt
# dbt reads profiles/dbt_project from the dbt/ dir; credentials come from ./.env
ENV := set -a; . ./.env; set +a;

.PHONY: setup lint test check dbt-setup dbt-deps dbt-debug dbt-run dbt-parse bootstrap ping ingest ingest-macro ingest-fundamentals

## Environment ────────────────────────────────────────────────────────────────

setup: ## Create venv + install runtime + dev deps
	uv sync --extra dev

dbt-setup: ## Install dbt-core + dbt-snowflake adapter (with dev tools)
	uv sync --extra dev --extra dbt

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

## dbt ────────────────────────────────────────────────────────────────────────

dbt-deps: ## Install dbt packages (dbt_expectations, elementary)
	$(ENV) $(DBT) deps --project-dir . --profiles-dir .

dbt-debug: ## Validate Snowflake connection + profile (no transforms)
	$(ENV) $(DBT) debug --project-dir . --profiles-dir .

dbt-parse: ## Compile the project without running (fast contract check)
	$(ENV) $(DBT) parse --project-dir . --profiles-dir .

dbt-run: ## Build + test all models (silver/gold, contracts enforced) with latency telemetry
	$(ENV) $(PYTHON) python scripts/run_dbt.py build --project-dir . --profiles-dir .
