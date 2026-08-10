# App image — the Python platform: all agents (producer, materializer,
# predictor, execution, simulation, watchdog) AND the FastAPI dashboard are
# different `python -m ...` commands of this one image. CI builds + pushes it
# to ECR; ECS task definitions pick the command per service.

FROM python:3.12-slim

# uv for locked, reproducible installs (same tooling as CI).
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_PROJECT_ENVIRONMENT=/venv

WORKDIR /app

# 1) Locked deps first (layer-cached across source edits). Default extras
#    (streaming/online-store) + api (fastapi/uvicorn). No dev/dbt/spark/mlflow
#    in the runtime image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra api

# 2) Source. Running from the tree (PYTHONPATH=/app) keeps `python -m stream.*`
#    working exactly like the local launchd agents.
COPY . .

# Healthcheck on the dashboard API (cheap, exercises the running process).
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
