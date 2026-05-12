"""Incremental CSV -> raw.diabetes_raw ingestion.

Reads the source CSV in chunks of at most `batch_size` rows (capped at 15.000
per project requirement) and inserts each row as a JSONB payload along with
audit columns. Idempotency is enforced via a deterministic SHA-256
`row_hash` and `ON CONFLICT (row_hash) DO NOTHING`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Iterable

import pandas as pd
from psycopg2.extras import Json, execute_batch

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 15_000


def _row_hash(row: dict) -> str:
    """Deterministic hash over canonicalised row content."""
    canonical = json.dumps(
        {k: ("" if pd.isna(v) else str(v)) for k, v in sorted(row.items())},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_source(path: str | None = None) -> str:
    """Verify the source CSV is reachable. Raises FileNotFoundError if not."""
    settings = load()
    csv_path = path or settings.source_csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"source CSV not found: {csv_path}")
    size = os.path.getsize(csv_path)
    logger.info("source csv ok: %s (%d bytes)", csv_path, size)
    return csv_path


def _iter_chunks(csv_path: str, chunk_size: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(csv_path, chunksize=chunk_size)


def load_batch(path: str | None = None, batch_id: str | None = None) -> dict:
    """Load the source CSV into raw.diabetes_raw.

    Returns a summary dict: {batch_id, inserted, duplicates, total_rows}.
    """
    settings = load()
    csv_path = check_source(path)
    chunk_size = min(settings.batch_size, MAX_BATCH_SIZE)
    batch_id = batch_id or uuid.uuid4().hex
    source_file = os.path.basename(csv_path)

    inserted = duplicates = total = 0
    insert_sql = (
        "INSERT INTO raw.diabetes_raw "
        "(row_hash, batch_id, source_file, status, payload) "
        "VALUES (%s, %s, %s, 'loaded', %s) "
        "ON CONFLICT (row_hash) DO NOTHING"
    )

    with connect() as conn, conn.cursor() as cur:
        for chunk_idx, chunk in enumerate(_iter_chunks(csv_path, chunk_size)):
            rows = []
            for _, raw_row in chunk.iterrows():
                row_dict = raw_row.to_dict()
                rh = _row_hash(row_dict)
                clean = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                rows.append((rh, batch_id, source_file, Json(clean)))
            before = _count_rows(cur)
            execute_batch(cur, insert_sql, rows, page_size=1_000)
            after = _count_rows(cur)
            new_rows = after - before
            inserted += new_rows
            duplicates += len(rows) - new_rows
            total += len(rows)
            logger.info(
                "chunk %d: %d rows (inserted=%d duplicates=%d)",
                chunk_idx,
                len(rows),
                new_rows,
                len(rows) - new_rows,
            )

    summary = {
        "batch_id": batch_id,
        "source_file": source_file,
        "inserted": inserted,
        "duplicates": duplicates,
        "total_rows": total,
    }
    logger.info("ingest done: %s", summary)
    return summary


def _count_rows(cur) -> int:
    cur.execute("SELECT count(*) FROM raw.diabetes_raw")
    return cur.fetchone()[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(load_batch())
