"""Incremental CSV -> raw.diabetes_raw ingestion.

Each call loads the next batch of at most `batch_size` rows (capped at 15,000
per project requirement) using the count of already-loaded rows as an offset.
Running the DAG repeatedly advances the cursor, so the full CSV is consumed
across ~7 runs for the 101k-row dataset.
Idempotency is enforced via a deterministic SHA-256 `row_hash` and
`ON CONFLICT (row_hash) DO NOTHING`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid

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



def load_batch(path: str | None = None, batch_id: str | None = None) -> dict:
    """Load the next batch of rows from the source CSV into raw.diabetes_raw.

    Uses the count of already-loaded rows for this source file as an offset,
    so each DAG run advances the cursor by at most `chunk_size` rows.
    Returns a summary dict: {batch_id, inserted, duplicates, total_rows}.
    """
    settings = load()
    csv_path = check_source(path)
    chunk_size = min(settings.batch_size, MAX_BATCH_SIZE)
    batch_id = batch_id or uuid.uuid4().hex
    source_file = os.path.basename(csv_path)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM raw.diabetes_raw WHERE source_file = %s",
            (source_file,),
        )
        offset = cur.fetchone()[0]

    logger.info("current offset for %s: %d rows already loaded", source_file, offset)

    col_names = pd.read_csv(csv_path, nrows=0).columns.tolist()
    chunk = pd.read_csv(
        csv_path,
        skiprows=offset + 1,  # +1 for the header row
        nrows=chunk_size,
        names=col_names,
        header=None,
    )

    if chunk.empty:
        logger.info("no new rows to ingest — CSV fully loaded (offset=%d)", offset)
        summary = {
            "batch_id": batch_id,
            "source_file": source_file,
            "inserted": 0,
            "duplicates": 0,
            "total_rows": 0,
        }
        logger.info("ingest done: %s", summary)
        return summary

    insert_sql = (
        "INSERT INTO raw.diabetes_raw "
        "(row_hash, batch_id, source_file, status, payload) "
        "VALUES (%s, %s, %s, 'loaded', %s) "
        "ON CONFLICT (row_hash) DO NOTHING"
    )

    rows = []
    for _, raw_row in chunk.iterrows():
        row_dict = raw_row.to_dict()
        rh = _row_hash(row_dict)
        clean = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
        rows.append((rh, batch_id, source_file, Json(clean)))

    with connect() as conn, conn.cursor() as cur:
        before = _count_rows(cur)
        execute_batch(cur, insert_sql, rows, page_size=1_000)
        after = _count_rows(cur)

    inserted = after - before
    duplicates = len(rows) - inserted

    logger.info(
        "batch: %d rows (inserted=%d duplicates=%d offset=%d)",
        len(rows), inserted, duplicates, offset,
    )

    summary = {
        "batch_id": batch_id,
        "source_file": source_file,
        "inserted": inserted,
        "duplicates": duplicates,
        "total_rows": len(rows),
    }
    logger.info("ingest done: %s", summary)
    return summary


def _count_rows(cur) -> int:
    cur.execute("SELECT count(*) FROM raw.diabetes_raw")
    return cur.fetchone()[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(load_batch())
