"""Validacion de esquema, calidad, drift y categorias nuevas."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {
    "brokered_by", "status", "price", "bed", "bath", "acre_lot",
    "street", "city", "state", "zip_code", "house_size", "prev_sold_date",
}

NUMERIC_FEATURES = ["price", "bed", "bath", "acre_lot", "house_size"]
CATEGORICAL_FEATURES = ["city", "state", "status", "zip_code"]
KS_THRESHOLD = 0.05
MIN_DRIFT_FEATURES = 2
NEW_CATEGORY_THRESHOLD = 0.05


def _read_batch(batch_id: str | None) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        if batch_id:
            cur.execute(
                "SELECT row_hash, payload FROM raw.properties_raw WHERE batch_id = %s",
                (batch_id,),
            )
        else:
            cur.execute("SELECT row_hash, payload FROM raw.properties_raw")
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no se encontraron filas crudas (batch_id={batch_id!r})")
    return pd.DataFrame([{"row_hash": h, **p} for h, p in rows])


def _read_historical() -> pd.DataFrame | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT features, target FROM clean.properties_clean LIMIT 50000")
        rows = cur.fetchall()
    if not rows:
        return None
    return pd.DataFrame([{**f, "price": t} for f, t in rows])


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    actual = set(df.columns) - {"row_hash"}
    missing = EXPECTED_COLUMNS - actual
    extra = actual - EXPECTED_COLUMNS
    ok = len(missing) == 0
    result = {"schema_ok": ok, "missing_columns": sorted(missing), "extra_columns": sorted(extra)}
    if not ok:
        logger.warning("esquema invalido: %s", result)
    return result


def validate_quality(df: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    total = len(df)

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
    if null_counts:
        issues.append(f"nulos: {null_counts}")

    if "price" in df.columns:
        bad_price = int(((df["price"] <= 0) | (df["price"] > 10_000_000)).sum())
        if bad_price:
            issues.append(f"precios fuera de rango: {bad_price}")

    for col in ["bed", "bath", "house_size"]:
        if col in df.columns:
            negatives = int((df[col] < 0).sum())
            if negatives:
                issues.append(f"{col} negativos: {negatives}")

    dup_hashes = int(df["row_hash"].duplicated().sum()) if "row_hash" in df.columns else 0
    if dup_hashes:
        issues.append(f"row_hash duplicados: {dup_hashes}")

    ok = len(issues) == 0
    result = {"quality_ok": ok, "total_rows": total, "issues": issues}
    if not ok:
        logger.warning("problemas de calidad: %s", issues)
    return result


def detect_drift(df_new: pd.DataFrame, df_hist: pd.DataFrame | None) -> dict[str, Any]:
    if df_hist is None or df_hist.empty:
        return {"drift_detected": False, "reason": "sin datos historicos para comparar", "details": {}}

    drifted = {}
    for col in NUMERIC_FEATURES:
        if col not in df_new.columns or col not in df_hist.columns:
            continue
        new_vals = df_new[col].dropna().values
        hist_vals = df_hist[col].dropna().values
        if len(new_vals) < 10 or len(hist_vals) < 10:
            continue
        stat, pvalue = ks_2samp(new_vals, hist_vals)
        if pvalue < KS_THRESHOLD:
            drifted[col] = {"statistic": round(float(stat), 4), "p_value": round(float(pvalue), 6)}

    detected = len(drifted) >= MIN_DRIFT_FEATURES
    return {"drift_detected": detected, "drifted_features": drifted, "count": len(drifted)}


def detect_new_categories(df_new: pd.DataFrame, df_hist: pd.DataFrame | None) -> dict[str, Any]:
    if df_hist is None or df_hist.empty:
        return {"new_categories": {}, "significant": False, "reason": "sin historico"}

    result = {}
    significant = False
    for col in CATEGORICAL_FEATURES:
        if col not in df_new.columns or col not in df_hist.columns:
            continue
        new_vals = set(df_new[col].dropna().astype(str).unique())
        hist_vals = set(df_hist[col].dropna().astype(str).unique())
        novel = new_vals - hist_vals
        if novel:
            proportion = len(novel) / max(len(new_vals), 1)
            result[col] = {"new_values_count": len(novel), "proportion": round(proportion, 4)}
            if proportion > NEW_CATEGORY_THRESHOLD:
                significant = True

    return {"new_categories": result, "significant": significant}


def run(batch_id: str | None = None) -> dict[str, Any]:
    df = _read_batch(batch_id)
    df_hist = _read_historical()

    schema_result = validate_schema(df)
    quality_result = validate_quality(df)
    drift_result = detect_drift(df, df_hist)
    categories_result = detect_new_categories(df, df_hist)

    report = {
        "batch_id": batch_id,
        "rows": len(df),
        **schema_result,
        **quality_result,
        **drift_result,
        **categories_result,
    }
    logger.info("quality report: %s", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import json
    print(json.dumps(run(), indent=2, default=str))
