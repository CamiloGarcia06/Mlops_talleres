"""Stratified train/val/test split (70/15/15) persisted onto clean.diabetes_clean.

The split is reproducible via the configured `random_seed`. Re-running this
step over the same batch resets and reassigns splits — useful when the user
re-runs the DAG after promoting a different model and wants identical splits.
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
    settings = load()

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
        raise ValueError(f"no clean rows to split (batch_id={batch_id!r})")

    df = pd.DataFrame(rows, columns=["id", "target"])
    ids = df["id"].tolist()
    y = df["target"].tolist()

    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        ids, y, test_size=0.30, random_state=settings.random_seed, stratify=y,
    )
    ids_val, ids_test, _, _ = train_test_split(
        ids_tmp, y_tmp, test_size=0.50, random_state=settings.random_seed, stratify=y_tmp,
    )

    assignments = (
        [(row_id, "train") for row_id in ids_train]
        + [(row_id, "val") for row_id in ids_val]
        + [(row_id, "test") for row_id in ids_test]
    )

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
    logger.info("split done: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
