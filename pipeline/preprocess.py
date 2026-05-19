"""Preprocesa las filas de raw.diabetes_raw y las escribe en clean.diabetes_clean.

Diseño Patrón 1: las columnas categóricas se dejan como **strings** (no se
hace one-hot aquí). El one-hot encoding vive dentro del `Pipeline` de
sklearn al momento de entrenar, así el encoder queda serializado junto
con el modelo en MLflow y la API recibe features crudas en lugar de un
vector pre-codificado.

Transformaciones aplicadas en este módulo:
  - Detección automática del target (`Outcome` o `readmitted`) y
    binarización a {0, 1}.
  - Imputación de columnas numéricas con la mediana de cada una.
  - Descarte de categóricas con más de 20 valores únicos
    (p. ej. códigos ICD con 700+ valores que harían explotar el
    OneHotEncoder al entrenar).
  - Las features se persisten como JSONB para que el esquema pueda
    evolucionar sin necesidad de DDL.

Reejecutar este paso sobre el mismo batch es idempotente: las filas se
hacen upsert por `row_hash`.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_batch

from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

# Posibles nombres del target soportados por el pipeline.
TARGET_CANDIDATES = ("Outcome", "outcome", "readmitted", "target")
# Umbral máximo de cardinalidad para conservar una categórica.
LOW_CARD_MAX = 20


def _detect_target(columns: Iterable[str]) -> str:
    """Devuelve el nombre de la columna que cumple el rol de target."""
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"no se encontró columna target en {list(columns)}")


def _binarize_target(series: pd.Series) -> pd.Series:
    """Convierte el target a entero binario {0, 1}.

    Maneja tres casos según el tipo de dato:
      - Entero/booleano → se acota a {0, 1}.
      - Flotante → se considera positivo si es > 0.
      - Categórico (caso 130-US): "<30" significa reingreso dentro de
        30 días (clase positiva); "NO" y ">30" se consideran negativos.
    """
    if series.dtype.kind in {"i", "u", "b"}:
        return series.astype(int).clip(0, 1)
    if series.dtype.kind == "f":
        return (series.fillna(0).astype(int) > 0).astype(int)
    # Caso categórico: mapeamos a 0 por defecto y solo elevamos a 1 las
    # categorías que representan reingreso temprano.
    mapping = {v: 0 for v in series.unique()}
    if "<30" in mapping:
        mapping["<30"] = 1
    elif "YES" in mapping:
        mapping["YES"] = 1
    return series.map(mapping).fillna(0).astype(int)


def _read_batch(batch_id: str | None) -> pd.DataFrame:
    """Lee todas las filas crudas con status='loaded'.

    Procesamos siempre TODO el acumulado y no solo el batch nuevo, así el
    esquema de features que ve el encoder es consistente entre
    ejecuciones del DAG.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, batch_id, payload FROM raw.diabetes_raw "
            "WHERE status = 'loaded'"
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no hay filas crudas para preprocesar (batch_id={batch_id!r})")
    return pd.DataFrame([{"row_hash": h, "batch_id": b, **p} for h, b, p in rows])


def run(batch_id: str | None = None) -> dict:
    """Aplica todas las transformaciones y persiste el resultado en clean.diabetes_clean."""
    df = _read_batch(batch_id)
    target_col = _detect_target(df.columns)

    # Separamos target y features.
    y = _binarize_target(df[target_col])
    feature_df = df.drop(columns=["row_hash", "batch_id", target_col])

    # Identificamos qué columnas son numéricas vs. categóricas usando los
    # dtypes que pandas infirió al leer el JSONB.
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    # Imputación de numéricas con la mediana (robusta a outliers).
    for c in numeric_cols:
        median = feature_df[c].median()
        feature_df[c] = feature_df[c].fillna(median)

    # Descartamos categóricas de alta cardinalidad. En el dataset 130-US,
    # las columnas diag_1/2/3 (códigos ICD) tienen 700+ valores únicos y
    # explotarían el OneHotEncoder generando miles de columnas.
    high_card = [c for c in categorical_cols if feature_df[c].nunique() > LOW_CARD_MAX]
    if high_card:
        feature_df = feature_df.drop(columns=high_card)
        categorical_cols = [c for c in categorical_cols if c not in high_card]

    # Casteamos categóricas a string para que JSONB las serialice como
    # texto y el OneHotEncoder las trate de forma consistente.
    for c in categorical_cols:
        feature_df[c] = feature_df[c].astype(str)

    # Upsert por row_hash: reejecuciones del DAG reemplazan la versión
    # anterior de la fila limpia y resetean el split (será reasignado en
    # el paso de split.py).
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
    logger.info("preprocess finalizado: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
