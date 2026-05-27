"""DDL idempotente para las capas raw / clean / audit / inference.

Aplica `CREATE SCHEMA IF NOT EXISTS` y `CREATE TABLE IF NOT EXISTS`, por lo
que se puede invocar de forma segura en cada ejecucion del DAG sin riesgo
de duplicar objetos ni borrar datos.

Esquemas:
  - raw.properties_raw: fila original como JSONB + metadatos de lote.
  - clean.properties_clean: features transformadas (JSONB) + target (price).
  - audit.batch_log: registro de cada ejecucion del pipeline.
  - inference.predictions: log de cada llamada a /predict.
"""

from __future__ import annotations

import logging

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

DDL = [
    "CREATE SCHEMA IF NOT EXISTS raw",
    "CREATE SCHEMA IF NOT EXISTS clean",
    "CREATE SCHEMA IF NOT EXISTS audit",
    "CREATE SCHEMA IF NOT EXISTS inference",

    # --- RAW ---
    """
    CREATE TABLE IF NOT EXISTS raw.properties_raw (
        row_hash        TEXT PRIMARY KEY,
        batch_id        TEXT NOT NULL,
        load_timestamp  TIMESTAMPTZ NOT NULL DEFAULT now(),
        source          TEXT NOT NULL DEFAULT 'api',
        status          TEXT NOT NULL DEFAULT 'loaded',
        payload         JSONB NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_raw_batch ON raw.properties_raw (batch_id)",

    # --- CLEAN ---
    """
    CREATE TABLE IF NOT EXISTS clean.properties_clean (
        id           BIGSERIAL PRIMARY KEY,
        row_hash     TEXT UNIQUE NOT NULL REFERENCES raw.properties_raw(row_hash),
        batch_id     TEXT NOT NULL,
        processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        split        TEXT,
        features     JSONB NOT NULL,
        target       DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_clean_batch ON clean.properties_clean (batch_id)",
    "CREATE INDEX IF NOT EXISTS ix_clean_split ON clean.properties_clean (split)",

    # --- AUDIT ---
    """
    CREATE TABLE IF NOT EXISTS audit.batch_log (
        id               BIGSERIAL PRIMARY KEY,
        batch_id         TEXT NOT NULL,
        run_timestamp    TIMESTAMPTZ NOT NULL DEFAULT now(),
        rows_received    INT,
        rows_after_dedup INT,
        schema_ok        BOOLEAN,
        quality_ok       BOOLEAN,
        drift_detected   BOOLEAN,
        training_decision BOOLEAN,
        training_reason  TEXT,
        mlflow_run_id    TEXT,
        model_registered BOOLEAN DEFAULT FALSE,
        promotion_decision BOOLEAN,
        promotion_reason TEXT,
        champion_metric_before DOUBLE PRECISION,
        champion_metric_after  DOUBLE PRECISION,
        status           TEXT NOT NULL DEFAULT 'started'
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_audit_batch ON audit.batch_log (batch_id)",

    # --- INFERENCE ---
    """
    CREATE TABLE IF NOT EXISTS inference.predictions (
        request_id    UUID PRIMARY KEY,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        input_payload JSONB NOT NULL,
        prediction    DOUBLE PRECISION,
        model_name    TEXT,
        model_version TEXT,
        latency_ms    DOUBLE PRECISION
    )
    """,
]


def run() -> None:
    """Aplica todas las sentencias DDL en una sola transacción.

    Como cada sentencia usa `IF NOT EXISTS`, ejecutarla varias veces no
    produce errores ni borra datos existentes.
    """
    with connect() as conn, conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    logger.info("migraciones aplicadas: %d sentencias", len(DDL))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
