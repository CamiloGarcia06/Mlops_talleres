"""Preprocesa raw.properties_raw y escribe en clean.properties_clean.

Transformaciones:
  - Drop: brokered_by, street (IDs sin valor predictivo).
  - prev_sold_date -> prev_sold_year (int).
  - zip_code -> string (categorica).
  - Imputacion de numericas con mediana.
  - Categoricas a string para encoding en el pipeline de sklearn.
  - Features como JSONB, target como price (DOUBLE PRECISION).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_batch

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

DROP_COLUMNS = ["brokered_by", "street"]
NUMERIC_FEATURES = ["bed", "bath", "acre_lot", "house_size"]
CATEGORICAL_FEATURES = ["status", "city", "state", "zip_code"]
TARGET_COL = "price"


def _read_raw(batch_id: str | None) -> pd.DataFrame:
    # status='loaded' = filas aun no procesadas. Tras preprocesar las marcamos
    # 'processed', de modo que cada corrida solo toca el lote nuevo (O(lote)).
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, batch_id, payload FROM raw.properties_raw "
            "WHERE status = 'loaded'"
        )
        rows = cur.fetchall()
    return pd.DataFrame([{"row_hash": h, "batch_id": b, **p} for h, b, p in rows])


def _extract_year(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.year.fillna(0).astype(int)


def run(batch_id: str | None = None) -> dict:
    df = _read_raw(batch_id)
    if df.empty:
        logger.info("preprocess: no hay filas nuevas que procesar")
        return {"batch_id": batch_id, "rows": 0, "note": "sin filas nuevas"}

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "prev_sold_date" in df.columns:
        df["prev_sold_year"] = _extract_year(df["prev_sold_date"])
        df = df.drop(columns=["prev_sold_date"])

    if TARGET_COL not in df.columns:
        raise ValueError(f"columna target '{TARGET_COL}' no encontrada")

    target = df[TARGET_COL].astype(float)
    meta_cols = ["row_hash", "batch_id", TARGET_COL]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    feature_df = df[feature_cols].copy()

    for col in NUMERIC_FEATURES + ["prev_sold_year"]:
        if col in feature_df.columns:
            median = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median)

    for col in CATEGORICAL_FEATURES:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna("unknown").astype(str)

    upsert_sql = (
        "INSERT INTO clean.properties_clean (row_hash, batch_id, features, target) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (row_hash) DO UPDATE SET "
        "  batch_id = EXCLUDED.batch_id, "
        "  features = EXCLUDED.features, "
        "  target = EXCLUDED.target, "
        "  processed_at = now(), "
        "  split = NULL"
    )

    rows = []
    for row_hash, raw_batch, features, t in zip(
        df["row_hash"], df["batch_id"],
        feature_df.to_dict(orient="records"), target,
    ):
        rows.append((row_hash, batch_id or raw_batch, Json(features), float(t)))

    # Upsert en clean + marca de procesado en raw, atomicos en una transaccion.
    processed_hashes = [r[0] for r in rows]
    with connect() as conn, conn.cursor() as cur:
        execute_batch(cur, upsert_sql, rows, page_size=1_000)
        cur.execute(
            "UPDATE raw.properties_raw SET status = 'processed' WHERE row_hash = ANY(%s)",
            (processed_hashes,),
        )

    summary = {
        "batch_id": batch_id,
        "rows": len(rows),
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
    }
    logger.info("preprocess finalizado: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
