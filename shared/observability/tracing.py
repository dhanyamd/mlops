"""OpenTelemetry distributed tracing setup.

What distributed tracing solves:
  A fraud scoring request involves:
    Kafka consumer → Redis lookup → XGBoost inference → Qdrant search → Kafka write

  Without tracing: you see 4 separate log lines with no connection.
  With tracing: you see ONE trace with 4 child spans, each with duration and context.
  This tells you exactly which step is slow.

Concepts:
  Trace  — the complete journey of one request (one fraud scoring operation)
  Span   — one step in that journey (e.g. "redis_feature_lookup")
  Parent span — the outer operation that contains child spans

How it connects to logs:
  OTel injects trace_id and span_id into structlog context.
  Every log line inside a span automatically contains trace_id.
  In Grafana: click an alert → jump to trace → see all spans → jump to logs.

Backend:
  Local: OTLP exporter → Jaeger (http://localhost:16686)
  Production: OTLP → Jaeger / Datadog / Honeycomb / AWS X-Ray

Usage:
    from shared.observability.tracing import setup_tracing, get_tracer

    setup_tracing(service_name="fraud-inference")
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("redis_feature_lookup") as span:
        span.set_attribute("card_id", card_id)
        features = redis.get_features(card_id)
        span.set_attribute("cache_hit", features is not None)
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def setup_tracing(
    service_name: str,
    otlp_endpoint: str = "http://localhost:4317",
    export_to_console: bool = False,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing for the given service.

    Args:
        service_name:    Identifies this service in Jaeger UI (e.g. "fraud-inference").
        otlp_endpoint:   OTLP collector endpoint. Jaeger listens on 4317 by default.
        export_to_console: If True, also print spans to stdout (useful for debugging).

    Returns:
        The configured TracerProvider (usually not needed by callers).
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Primary exporter: OTLP → Jaeger
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if export_to_console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer for instrumenting a specific module.

    Usage:
        tracer = get_tracer(__name__)

        with tracer.start_as_current_span("score_transaction") as span:
            span.set_attribute("amount", txn["amount"])
            score = model.predict(...)
            span.set_attribute("fraud_score", score)
    """
    return trace.get_tracer(name)


def get_current_span_context() -> dict[str, str]:
    """Extract trace_id and span_id from the current span for log correlation.

    Inject this into structlog context so every log line inside a traced
    operation automatically includes the trace and span IDs.

    Usage:
        log = log.bind(**get_current_span_context())
        log.info("feature_fetched", cache_hit=True)
        # → {"trace_id": "abc123", "span_id": "def456", "event": "feature_fetched", ...}
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return {}
