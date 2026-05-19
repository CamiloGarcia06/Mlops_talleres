"""Compare a freshly registered candidate to the current `champion` and
optionally promote it.

The champion is identified by an MLflow Model Registry alias (default
`champion`). Stages are deprecated since MLflow 2.9, so aliases are the
preferred mechanism. The selection metric is read from the run that backs
the candidate version (`primary_metric`).
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import export_aws_env, load

logger = logging.getLogger(__name__)


def _client() -> MlflowClient:
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def _metric(client: MlflowClient, run_id: str, name: str) -> float:
    return float(client.get_run(run_id).data.metrics.get(name, float("-inf")))


def compare(candidate: dict) -> dict:
    """Inputs the dict returned by `train.run()`. Returns a comparison summary."""
    settings = load()
    client = _client()

    raw_metric = candidate.get("metric")
    candidate_metric = float(raw_metric) if raw_metric is not None else 0.0

    try:
        champion_version = client.get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        champion_metric = _metric(client, champion_version.run_id, "primary_metric")
        champion_v = champion_version.version
    except Exception:
        champion_v = None
        champion_metric = None

    decision = "promote" if champion_metric is None or candidate_metric > champion_metric else "keep"
    summary = {
        "candidate_version": candidate.get("version"),
        "candidate_metric": candidate_metric,
        "champion_version": champion_v,
        "champion_metric": champion_metric,
        "decision": decision,
    }
    logger.info("compare: %s", summary)
    return summary


def _best(candidates: list[dict]) -> dict:
    valid = [c for c in candidates if c and c.get("version") is not None]
    if not valid:
        raise ValueError("no valid candidates to promote (all missing 'version')")
    return max(valid, key=lambda c: float(c.get("metric") or float("-inf")))


def promote_best(candidates: list[dict]) -> dict:
    """Pick the best candidate by `metric` and promote it via `promote()`.

    Used by the Airflow DAG when several `train_*` tasks run in parallel and
    feed their results into a single promotion step. The winner is the one
    with the highest `primary_metric` (F1 by default).
    """
    winner = _best(candidates)
    logger.info(
        "selected best candidate: run_id=%s version=%s metric=%.4f (out of %d)",
        winner.get("run_id"), winner.get("version"), winner.get("metric"), len(candidates),
    )
    result = promote(winner)
    result["selected_run_id"] = winner.get("run_id")
    result["selected_version"] = winner.get("version")
    result["candidates_considered"] = len(candidates)
    return result


def promote(candidate: dict) -> dict:
    """Promote the candidate to `champion` if it beats the current champion."""
    settings = load()
    client = _client()

    decision = compare(candidate)
    if decision["decision"] == "promote" and candidate.get("version"):
        client.set_registered_model_alias(
            name=settings.registered_model_name,
            alias=settings.champion_alias,
            version=str(candidate["version"]),
        )
        logger.info(
            "promoted version %s to alias '%s'",
            candidate["version"],
            settings.champion_alias,
        )
        decision["promoted"] = True
    else:
        logger.info("champion not changed: %s", decision)
        decision["promoted"] = False
    return decision


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("promote requires a candidate dict; invoke via the CLI 'all' command")
