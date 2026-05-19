"""Split estratificado train / val / test (70 / 15 / 15).

El resultado se persiste en la columna `split` de clean.diabetes_clean
para que el módulo de entrenamiento pueda recuperarlo después sin tener
que rehacer la partición. Usamos la semilla configurada (`random_seed`)
para garantizar reproducibilidad.

Reejecutar este paso resetea y reasigna los splits desde cero. Es útil
cuando se reejecuta el DAG después de cambiar el modelo y se quiere
mantener exactamente la misma partición histórica.
"""

from __future__ import annotations

import logging

import pandas as pd
from psycopg2.extras import execute_batch
from sklearn.model_selection import train_test_split

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def run(batch_id: str | None = None) -> dict:
    """Asigna train / val / test a cada fila de clean.diabetes_clean.

    Estratifica por la variable target para preservar la proporción de
    clases en cada subconjunto. Esto importa especialmente en este
    problema porque solo ~11% de las filas son positivos.
    """
    settings = load()

    # Lectura: si nos dan un batch específico intentamos primero ese, y
    # si no hay filas (porque el batch fue 100% duplicado) caemos al
    # acumulado completo.
    with connect() as conn, conn.cursor() as cur:
        if batch_id:
            cur.execute(
                "SELECT id, target FROM clean.diabetes_clean WHERE batch_id = %s",
                (batch_id,),
            )
            rows = cur.fetchall()
            if not rows:
                cur.execute("SELECT id, target FROM clean.diabetes_clean")
                rows = cur.fetchall()
        else:
            cur.execute("SELECT id, target FROM clean.diabetes_clean")
            rows = cur.fetchall()

    if not rows:
        raise ValueError(f"no hay filas limpias para particionar (batch_id={batch_id!r})")

    df = pd.DataFrame(rows, columns=["id", "target"])
    ids = df["id"].tolist()
    y = df["target"].tolist()

    # Primer corte: separamos 70% train del 30% restante (val + test).
    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        ids, y, test_size=0.30, random_state=settings.random_seed, stratify=y,
    )
    # Segundo corte: dividimos ese 30% por la mitad → 15% val, 15% test.
    ids_val, ids_test, _, _ = train_test_split(
        ids_tmp, y_tmp, test_size=0.50, random_state=settings.random_seed, stratify=y_tmp,
    )

    # Armamos la lista de (id, split) que se va a aplicar como UPDATE.
    assignments = (
        [(row_id, "train") for row_id in ids_train]
        + [(row_id, "val") for row_id in ids_val]
        + [(row_id, "test") for row_id in ids_test]
    )

    # Persistimos el split en la tabla. execute_batch envía las filas en
    # bloques de 1000 para reducir el round-trip con Postgres.
    with connect() as conn, conn.cursor() as cur:
        execute_batch(
            cur,
            "UPDATE clean.diabetes_clean SET split = %s WHERE id = %s",
            [(s, i) for i, s in assignments],
            page_size=1_000,
        )

    summary = {
        "batch_id": batch_id,
        "train": len(ids_train),
        "val": len(ids_val),
        "test": len(ids_test),
        "seed": settings.random_seed,
    }
    logger.info("split finalizado: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
