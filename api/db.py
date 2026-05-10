"""Inference logger.

Writes every prediction to `inference.predictions`. The table is created by
`pipeline/db/migrations.py` (phase 3). Failures here are logged but never
break the response to the client (degraded mode).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import psycopg2
from psycopg2.extras import Json

from api.config import load

logger = logging.getLogger(__name__)

INSERT_SQL = """
INSERT INTO inference.predictions
    (request_id, input_payload, prediction, score, model_name, model_version, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def log_inference(
    request_id: str,
    input_payload: Dict[str, Any],
    prediction: int,
    score: float | None,
    model_name: str,
    model_version: str,
    latency_ms: float,
) -> None:
    try:
        with psycopg2.connect(load().pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                INSERT_SQL,
                (
                    request_id,
                    Json(input_payload),
                    int(prediction),
                    None if score is None else float(score),
                    model_name,
                    model_version,
                    float(latency_ms),
                ),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to log inference %s: %s", request_id, e)
