.PHONY: setup sync infra-up infra-down bootstrap demand fraud stream spark dashboard \
        test lint api api-docker quality-check drift-report deploy-flows load-test \
        model-card dvc-repro feast-apply feast-materialize dbt-run dbt-test dbt-docs \
        aws-local tf-local tf-staging tf-destroy

UV := uv

setup: sync
	cp -n .env.example .env 2>/dev/null || true

sync:
	$(UV) sync --all-groups

# ── Infrastructure ─────────────────────────────────────────────────────────────
infra-up:
	docker compose up -d
	@echo "Waiting for platform services..."
	@sleep 25
	@echo ""
	@echo "  Kafka UI:     http://localhost:8080"
	@echo "  ClickHouse:   http://localhost:8123"
	@echo "  Qdrant:       http://localhost:6333/dashboard"
	@echo "  MLflow:       http://localhost:5000"
	@echo "  Spark UI:     http://localhost:8081"
	@echo "  Prefect:      http://localhost:4200"
	@echo "  MinIO:        http://localhost:9001"
	@echo "  Prometheus:   http://localhost:9090"
	@echo "  Grafana:      http://localhost:3000  (admin/admin)"
	@echo "  Jaeger:       http://localhost:16686"

infra-down:
	docker compose down

bootstrap: infra-up
	@echo "Ingesting real datasets into ClickHouse + PostgreSQL + Qdrant..."
	$(UV) run python -m demand_forecasting.data.ingest
	$(UV) run python -m fraud_detection.data.ingest

# ── Pipelines ──────────────────────────────────────────────────────────────────
demand:
	$(UV) run python -m demand_forecasting.flows.orchestration

fraud:
	$(UV) run python -m fraud_detection.flows.orchestration

stream:
	@echo "Starting Spark feature job (background)..."
	docker compose exec -d spark-master spark-submit \
		--master spark://spark-master:7077 \
		--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
		/opt/spark/jobs/feature_streaming.py kafka:29092 redis
	sleep 5
	$(UV) run python -m fraud_detection.streaming.producer --count 200 &
	sleep 3
	$(UV) run python -m fraud_detection.streaming.inference_service --timeout 60

spark:
	docker compose exec spark-master spark-submit \
		--master spark://spark-master:7077 \
		--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
		/opt/spark/jobs/feature_streaming.py kafka:29092 redis

dashboard:
	$(UV) run streamlit run demand_forecasting/app/streamlit_app.py

# ── Pillar 3: FastAPI ──────────────────────────────────────────────────────────
api:
	$(UV) run uvicorn fraud_detection.api.app:app --host 0.0.0.0 --port 8000 --reload

api-docker:
	docker build -f Dockerfile.api -t fraud-api:latest .
	docker run -p 8000:8000 --env-file .env --network mlops_default fraud-api:latest

# ── Pillar 1: Data Quality ─────────────────────────────────────────────────────
quality-check:
	$(UV) run python -m shared.data_quality.quality

# ── Pillar 7: Drift Monitoring ────────────────────────────────────────────────
drift-report:
	$(UV) run python -m shared.monitoring.evidently_reports

# ── Pillar 11: Prefect Deployments ────────────────────────────────────────────
deploy-flows:
	$(UV) run python fraud_detection/flows/deployments.py
	$(UV) run python demand_forecasting/flows/deployments.py

# ── Pillar 12: Load Testing ───────────────────────────────────────────────────
load-test:
	$(UV) run locust -f fraud_detection/tests/locustfile.py \
		--host http://localhost:8000 \
		--users 50 --spawn-rate 5 --run-time 60s --headless

# ── Pillar 13: Model Cards ────────────────────────────────────────────────────
model-card:
	$(UV) run python -m shared.monitoring.model_card

# ── Pillar 10: DVC ────────────────────────────────────────────────────────────
dvc-repro:
	$(UV) run dvc repro

# ── Pillar 16: Feature Store ──────────────────────────────────────────────────
feast-apply:
	cd fraud_detection/feature_repo && $(UV) run feast apply

feast-materialize:
	cd fraud_detection/feature_repo && \
		$(UV) run feast materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)

# ── Pillar 17: Data Warehouse (dbt) ───────────────────────────────────────────
dbt-run:
	cd infra/dbt && dbt run

dbt-test:
	cd infra/dbt && dbt test

dbt-docs:
	cd infra/dbt && dbt docs generate && dbt docs serve

# ── Pillar 15: AWS / Terraform ────────────────────────────────────────────────
aws-local:
	docker compose up -d localstack
	@sleep 5
	bash infra/scripts/init_localstack.sh

tf-local:
	cd infra/terraform/environments/local && tflocal init && tflocal apply -auto-approve

tf-staging:
	cd infra/terraform/environments/staging && terraform init && terraform apply

tf-destroy:
	cd infra/terraform/environments/local && tflocal destroy -auto-approve

# ── Tests & Lint ──────────────────────────────────────────────────────────────
test:
	$(UV) run pytest -v

lint:
	$(UV) run ruff check .
