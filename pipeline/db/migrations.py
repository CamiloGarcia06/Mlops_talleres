"""Idempotent DDL for the raw / clean / inference layers.

Runs `CREATE SCHEMA IF NOT EXISTS` and `CREATE TABLE IF NOT EXISTS` so it can
be invoked safely on every pipeline execution.

Schema decisions:
  - `raw.diabetes_raw` stores the original record as JSONB (`payload`) plus
    audit columns. Using JSONB makes ingestion robust to schema drift in the
    source CSV without requiring DDL changes.
  - `clean.diabetes_clean` keeps a typed `target` plus a JSONB `features`
    map, so feature engineering decisions live in the preprocess module
    rather than in DDL.
  - `inference.predictions` belongs to phase 5; we declare it here so the
    schema is ready when the API arrives.
"""

from __future__ import annotations

import logging

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

DDL = [
    "CREATE SCHEMA IF NOT EXISTS raw",
    "CREATE SCHEMA IF NOT EXISTS clean",
    "CREATE SCHEMA IF NOT EXISTS inference",
    """
    CREATE TABLE IF NOT EXISTS raw.diabetes_raw (
        row_hash        TEXT PRIMARY KEY,
        batch_id        TEXT NOT NULL,
        load_timestamp  TIMESTAMPTZ NOT NULL DEFAULT now(),
        source_file     TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'loaded',
        payload         JSONB NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_raw_batch ON raw.diabetes_raw (batch_id)",
    """
    CREATE TABLE IF NOT EXISTS clean.diabetes_clean (
        id           BIGSERIAL PRIMARY KEY,
        row_hash     TEXT UNIQUE NOT NULL REFERENCES raw.diabetes_raw(row_hash),
        batch_id     TEXT NOT NULL,
        processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        split        TEXT,
        features     JSONB NOT NULL,
        target       INT  NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_clean_batch ON clean.diabetes_clean (batch_id)",
    "CREATE INDEX IF NOT EXISTS ix_clean_split ON clean.diabetes_clean (split)",
    """
    CREATE TABLE IF NOT EXISTS inference.predictions (
        request_id    UUID PRIMARY KEY,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        input_payload JSONB NOT NULL,
        prediction    INT,
        score         DOUBLE PRECISION,
        model_name    TEXT,
        model_version TEXT,
        latency_ms    DOUBLE PRECISION
    )
    """,
]


def run() -> None:
    """Apply all DDL statements idempotently."""
    with connect() as conn, conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    logger.info("migrations applied: %d statements", len(DDL))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
