"""OpenTelemetry setup. Spans go to a local Jaeger via OTLP/HTTP, to the console, or nowhere."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

DEFAULT_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"  # Jaeger all-in-one OTLP/HTTP

_configured = False


def setup_tracing(exporter: str, endpoint: str | None, service_name: str) -> None:
    """Install the global tracer provider once per process."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint or DEFAULT_OTLP_ENDPOINT)))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer():
    return trace.get_tracer("tracewise")


def shutdown() -> None:
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
