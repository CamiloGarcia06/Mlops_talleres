"""Preprocess raw rows into clean.diabetes_clean.

Transformations (kept generic so they tolerate either Pima or 130-US schemas):
  - target column auto-detected (`Outcome` or `readmitted`).
  - non-numeric columns are one-hot encoded.
  - missing numeric values are imputed with the column median.
  - the resulting feature row is stored as JSONB so the API can rebuild the
    feature vector without depending on a fixed column list at the DB level.

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
    # categorical — treat the most frequent class as the negative
    mapping = {v: 0 for v in series.unique()}
    if "<30" in mapping:
        mapping["<30"] = 1  # readmitted within 30 days = positive
    elif "YES" in mapping:
        mapping["YES"] = 1
    return series.map(mapping).fillna(0).astype(int)


def _read_batch(batch_id: str | None) -> pd.DataFrame:
    # Always process ALL accumulated raw data so that pd.get_dummies produces
    # a consistent feature schema regardless of which batch triggered the run.
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, batch_id, payload FROM raw.diabetes_raw "
            "WHERE status = 'loaded'"
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no raw rows to preprocess (batch_id={batch_id!r})")
    df = pd.DataFrame([{"row_hash": h, "batch_id": b, **p} for h, b, p in rows])
    return df


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

    # one-hot encode only low-cardinality categoricals; drop high-cardinality ones
    # (e.g. ICD diagnosis codes with 700+ unique values blow up memory)
    LOW_CARD_MAX = 20
    low_card = [c for c in categorical_cols if feature_df[c].nunique() <= LOW_CARD_MAX]
    high_card = [c for c in categorical_cols if feature_df[c].nunique() > LOW_CARD_MAX]
    if high_card:
        feature_df = feature_df.drop(columns=high_card)
    if low_card:
        feature_df = pd.get_dummies(feature_df, columns=low_card, dummy_na=False)
    feature_df = feature_df.astype(float)

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
        "target_col": target_col,
    }
    logger.info("preprocess done: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
