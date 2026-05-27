"""Logger de inferencias hacia inference.predictions."""

from __future__ import annotations

import logging
from typing import Any, Dict

import psycopg2
from psycopg2.extras import Json

from api.config import load

logger = logging.getLogger(__name__)

INSERT_SQL = """
INSERT INTO inference.predictions
    (request_id, input_payload, prediction, model_name, model_version, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s)
"""


def log_inference(
    request_id: str,
    input_payload: Dict[str, Any],
    prediction: float,
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
                    float(prediction),
                    model_name,
                    model_version,
                    float(latency_ms),
                ),
            )
    except Exception as e:
        logger.warning("no se pudo registrar la inferencia %s: %s", request_id, e)
