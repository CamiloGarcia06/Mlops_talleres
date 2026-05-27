"""Cliente HTTP para la API de inferencia y consulta de auditoria."""

from __future__ import annotations

import os
from typing import Any

import psycopg2
import requests

_DEFAULT_API_URL = "http://api:8000"


def _base() -> str:
    return os.environ.get("API_URL", _DEFAULT_API_URL).rstrip("/")


def _pg_dsn() -> str:
    return os.environ.get(
        "PG_DSN",
        "postgresql://mlops_user:mlops_pass_2026@postgres-service.mlops.svc.cluster.local:5432/mlops",
    )


def get_model_info(timeout: int = 30) -> dict[str, Any]:
    resp = requests.get(f"{_base()}/model-info", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def predict(features: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    resp = requests.post(
        f"{_base()}/predict",
        json={"features": features},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def get_audit_history(limit: int = 50) -> list[dict]:
    try:
        with psycopg2.connect(_pg_dsn()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, batch_id, run_timestamp, rows_received, rows_after_dedup, "
                "schema_ok, quality_ok, drift_detected, training_decision, training_reason, "
                "mlflow_run_id, model_registered, promotion_decision, promotion_reason, "
                "champion_metric_before, champion_metric_after, status "
                "FROM audit.batch_log ORDER BY run_timestamp DESC LIMIT %s",
                (limit,),
            )
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except Exception:
        return []
