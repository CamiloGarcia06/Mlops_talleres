"""Custom Prometheus metrics for the inference API.

Generic HTTP metrics (latency histograms, status codes, RPS) are emitted by
`prometheus_fastapi_instrumentator`. Here we only declare custom series
specific to the inference business logic.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS_TOTAL = Counter(
    "inference_predictions_total",
    "Total number of predictions served, labelled by class.",
    ["prediction"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Time spent inside the model.predict call.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MODEL_INFO = Gauge(
    "inference_model_info",
    "Constant gauge exposing the currently loaded model name/version/alias.",
    ["model_name", "model_version", "model_alias"],
)


def set_model_info(name: str, version: str, alias: str) -> None:
    """Reset previous label set and publish the active one as 1."""
    MODEL_INFO.clear()
    MODEL_INFO.labels(model_name=name, model_version=version, model_alias=alias).set(1)
