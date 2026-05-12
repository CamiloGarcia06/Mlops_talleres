"""Minimum data-quality checks executed against raw.diabetes_raw.

Returns a report and raises ValueError on hard failures so Airflow flags the
task as failed and downstream steps don't run.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

TARGET_CANDIDATES = ("Outcome", "outcome", "readmitted", "target")


def _detect_critical_keys(columns) -> tuple[str, ...]:
    for c in TARGET_CANDIDATES:
        if c in columns:
            return (c,)
    return ()


CRITICAL_KEYS_DEFAULT: tuple[str, ...] = ()


def _read_batch(batch_id: str | None) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        if batch_id:
            cur.execute(
                "SELECT row_hash, payload FROM raw.diabetes_raw WHERE batch_id = %s",
                (batch_id,),
            )
            rows = cur.fetchall()
            if not rows:
                # All rows were duplicates from a prior run; validate all existing data.
                cur.execute("SELECT row_hash, payload FROM raw.diabetes_raw")
                rows = cur.fetchall()
        else:
            cur.execute("SELECT row_hash, payload FROM raw.diabetes_raw")
            rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no raw rows found (batch_id={batch_id!r})")
    df = pd.DataFrame([{"row_hash": h, **p} for h, p in rows])
    return df


def run(batch_id: str | None = None, critical_keys: Iterable[str] | None = None) -> dict:
    df = _read_batch(batch_id)
    if critical_keys is None:
        critical_keys = _detect_critical_keys(df.columns)
    issues: list[str] = []

    # 1. row count
    if df.empty:
        issues.append("empty dataframe")

    # 2. critical columns must exist and be non-null
    for key in critical_keys:
        if key not in df.columns:
            issues.append(f"missing critical column: {key}")
            continue
        n_null = df[key].isna().sum()
        if n_null:
            issues.append(f"nulls in critical column {key}: {n_null}")

    # 3. duplicates by row_hash should be impossible (UNIQUE) but assert anyway
    if df["row_hash"].duplicated().any():
        issues.append("duplicate row_hash values inside the batch")

    report = {
        "batch_id": batch_id,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "issues": issues,
    }
    if issues:
        logger.error("quality issues: %s", issues)
        raise ValueError(f"data quality failed: {issues}")
    logger.info("quality ok: %s", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
