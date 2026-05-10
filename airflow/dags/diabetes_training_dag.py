"""Diabetes MLOps training DAG.

Each task is a thin wrapper around a function from the project's `pipeline`
package. The wrappers exist only to:
  - propagate `batch_id` between tasks via XCom
  - keep the DAG file declarative

If a task fails, downstream tasks don't run (default `all_success` rule),
the error is visible in the Airflow UI, and the DAG can be rerun safely
because every step is idempotent (`row_hash` UNIQUE in raw, upserts in
clean, MLflow runs are append-only).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, task

from pipeline import ingest, preprocess, promote, quality, split, train
from pipeline.db import migrations


DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="diabetes_mlops_pipeline",
    description="Ingest -> quality -> preprocess -> split -> train -> promote",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "diabetes"],
)
def diabetes_mlops_pipeline():

    @task()
    def t_migrate() -> None:
        migrations.run()

    @task()
    def t_check_source() -> str:
        return ingest.check_source()

    @task()
    def t_load_batch(_source: str) -> dict:
        return ingest.load_batch()

    @task()
    def t_quality(load_summary: dict) -> dict:
        return quality.run(batch_id=load_summary["batch_id"])

    @task()
    def t_preprocess(load_summary: dict) -> dict:
        return preprocess.run(batch_id=load_summary["batch_id"])

    @task()
    def t_split(load_summary: dict) -> dict:
        return split.run(batch_id=load_summary["batch_id"])

    @task()
    def t_train(load_summary: dict) -> dict:
        return train.run(batch_id=load_summary["batch_id"])

    @task()
    def t_compare(candidate: dict) -> dict:
        return promote.compare(candidate)

    @task()
    def t_promote(candidate: dict) -> dict:
        return promote.promote(candidate)

    migrate = t_migrate()
    src = t_check_source()
    loaded = t_load_batch(src)
    qual = t_quality(loaded)
    prep = t_preprocess(loaded)
    sp = t_split(loaded)
    candidate = t_train(loaded)
    comparison = t_compare(candidate)
    promotion = t_promote(candidate)

    migrate >> src >> loaded >> qual >> prep >> sp >> candidate >> comparison >> promotion


diabetes_mlops_pipeline()
