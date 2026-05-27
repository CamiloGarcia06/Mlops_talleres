"""Split train / test (80 / 20) para regresion."""

from __future__ import annotations

import logging

import pandas as pd
from psycopg2.extras import execute_batch
from sklearn.model_selection import train_test_split

from pipeline.config import load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def run(batch_id: str | None = None) -> dict:
    settings = load()

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, target FROM clean.properties_clean")
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"no hay filas limpias para particionar (batch_id={batch_id!r})")

    df = pd.DataFrame(rows, columns=["id", "target"])

    ids_train, ids_test = train_test_split(
        df["id"].tolist(),
        test_size=0.20,
        random_state=settings.random_seed,
    )

    assignments = (
        [(row_id, "train") for row_id in ids_train]
        + [(row_id, "test") for row_id in ids_test]
    )

    with connect() as conn, conn.cursor() as cur:
        execute_batch(
            cur,
            "UPDATE clean.properties_clean SET split = %s WHERE id = %s",
            [(s, i) for i, s in assignments],
            page_size=1_000,
        )

    summary = {
        "batch_id": batch_id,
        "train": len(ids_train),
        "test": len(ids_test),
        "seed": settings.random_seed,
    }
    logger.info("split finalizado: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
