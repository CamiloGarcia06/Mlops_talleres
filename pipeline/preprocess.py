"""Preprocess raw rows into clean.diabetes_clean.

The output keeps categorical columns as **strings** (not one-hot). One-hot
encoding now lives inside the sklearn Pipeline at training time, so the
encoder is versioned together with the model and the API receives raw
inputs instead of a 141-column pre-encoded payload.

Transformations applied here:
  - target column auto-detected (`Outcome` or `readmitted`) and binarized.
  - numeric columns are imputed with the column median.
  - categorical columns with >20 unique values are dropped (e.g. ICD codes
    with 700+ distinct values would blow up the OneHotEncoder fitted later).
  - features are stored as JSONB so the schema can evolve without DDL.

Re-running on the same batch is idempotent: rows are upserted by `row_hash`.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_batch

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

TARGET_CANDIDATES = ("Outcome", "outcome", "readmitted", "target")
LOW_CARD_MAX = 20


def _detect_target(columns: Iterable[str]) -> str:
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"could not find target column in {list(columns)}")


def _binarize_target(series: pd.Series) -> pd.Series:
    """Map the target to a binary {0,1} integer series."""
    if series.dtype.kind in {"i", "u", "b"}:
        return series.astype(int).clip(0, 1)
    if series.dtype.kind == "f":
        return (series.fillna(0).astype(int) > 0).astype(int)
    # categorical — readmitted within 30 days = positive class
    mapping = {v: 0 for v in series.unique()}
    if "<30" in mapping:
        mapping["<30"] = 1
    elif "YES" in mapping:
        mapping["YES"] = 1
    return series.map(mapping).fillna(0).astype(int)


def _read_batch(batch_id: str | None) -> pd.DataFrame:
    # Always process ALL accumulated raw data so feature schema is consistent
    # across runs.
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, batch_id, payload FROM raw.diabetes_raw "
            "WHERE status = 'loaded'"
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no raw rows to preprocess (batch_id={batch_id!r})")
    return pd.DataFrame([{"row_hash": h, "batch_id": b, **p} for h, b, p in rows])


def run(batch_id: str | None = None) -> dict:
    df = _read_batch(batch_id)
    target_col = _detect_target(df.columns)

    y = _binarize_target(df[target_col])
    feature_df = df.drop(columns=["row_hash", "batch_id", target_col])

    # numeric / categorical split
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    # median imputation for numerics
    for c in numeric_cols:
        median = feature_df[c].median()
        feature_df[c] = feature_df[c].fillna(median)

    # drop high-cardinality categoricals (e.g. ICD diag codes with 700+ values)
    high_card = [c for c in categorical_cols if feature_df[c].nunique() > LOW_CARD_MAX]
    if high_card:
        feature_df = feature_df.drop(columns=high_card)
        categorical_cols = [c for c in categorical_cols if c not in high_card]

    # cast categoricals to plain strings so JSONB stays JSON-serialisable
    for c in categorical_cols:
        feature_df[c] = feature_df[c].astype(str)

    upsert_sql = (
        "INSERT INTO clean.diabetes_clean (row_hash, batch_id, features, target) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (row_hash) DO UPDATE SET "
        "  batch_id = EXCLUDED.batch_id, "
        "  features = EXCLUDED.features, "
        "  target = EXCLUDED.target, "
        "  processed_at = now(), "
        "  split = NULL"
    )
    rows = []
    for row_hash, raw_batch, features, target in zip(
        df["row_hash"], df["batch_id"], feature_df.to_dict(orient="records"), y
    ):
        rows.append((row_hash, batch_id or raw_batch, Json(features), int(target)))

    with connect() as conn, conn.cursor() as cur:
        execute_batch(cur, upsert_sql, rows, page_size=1_000)

    summary = {
        "batch_id": batch_id,
        "rows": len(rows),
        "feature_count": len(feature_df.columns),
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "target_col": target_col,
    }
    logger.info("preprocess done: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
