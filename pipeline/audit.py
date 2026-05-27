"""Operaciones sobre la tabla audit.batch_log."""

from __future__ import annotations

import logging
from typing import Any

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def create_entry(batch_id: str, rows_received: int) -> int:
    """Crea una entrada inicial en audit.batch_log y retorna su id."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO audit.batch_log (batch_id, rows_received, status) "
            "VALUES (%s, %s, 'started') RETURNING id",
            (batch_id, rows_received),
        )
        audit_id = cur.fetchone()[0]
    logger.info("audit entry creada: id=%d batch_id=%s", audit_id, batch_id)
    return audit_id


def update_entry(audit_id: int, **fields: Any) -> None:
    """Actualiza campos de una entrada existente."""
    if not fields:
        return
    set_clauses = ", ".join(f"{k} = %s" for k in fields)
    values = list(fields.values()) + [audit_id]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE audit.batch_log SET {set_clauses} WHERE id = %s",
            values,
        )
    logger.info("audit entry %d actualizada: %s", audit_id, list(fields.keys()))


def get_history(limit: int = 50) -> list[dict]:
    """Retorna las ultimas entradas de auditoria."""
    with connect() as conn, conn.cursor() as cur:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import json
    print(json.dumps(get_history(), indent=2, default=str))
