.PHONY: setup sync infra-up infra-down bootstrap demand fraud stream spark dashboard test lint

UV := uv

setup: sync
	cp -n .env.example .env 2>/dev/null || true

sync:
	$(UV) sync

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

infra-down:
	docker compose down

bootstrap: infra-up
	@echo "Ingesting real datasets into ClickHouse + PostgreSQL + Qdrant..."
	$(UV) run python -m demand_forecasting.data.ingest
	$(UV) run python -m fraud_detection.data.ingest

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

test:
	$(UV) run pytest -v

lint:
	$(UV) run ruff check .
