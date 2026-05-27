"""DAG MLOps de propiedades inmobiliarias.

Flujo con 19 tareas y bifurcaciones explicitas:
  start -> fetch_batch -> store_raw -> validate_schema -> validate_quality
  -> detect_new_categories -> detect_drift -> preprocess_data
  -> decide_training -+-> train_candidate -> evaluate_candidate
                      |   -> register_in_mlflow -> compare_with_production
                      |   -> decide_promotion -+-> promote_model
                      |                        +-> reject_model
                      +-> skip_training
  -> notify_or_log_result -> end
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, task


DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="properties_mlops_pipeline",
    description="Ingesta, validacion, entrenamiento y promocion de modelos de regresion de precios",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "real-estate", "regression"],
)
def properties_mlops_pipeline():

    @task()
    def start() -> dict:
        from pipeline.db import migrations
        migrations.run()
        return {"status": "started"}

    @task()
    def fetch_batch_from_api(**context) -> dict:
        from pipeline.ingest import fetch_batch
        return fetch_batch()

    @task()
    def store_raw_batch(api_response: dict) -> dict:
        from pipeline.ingest import store_batch
        return store_batch(api_response)

    @task()
    def validate_schema(ingest_summary: dict) -> dict:
        from pipeline.quality import _read_batch, validate_schema as _validate_schema
        df = _read_batch(ingest_summary["batch_id"])
        return _validate_schema(df)

    @task()
    def validate_data_quality(ingest_summary: dict) -> dict:
        from pipeline.quality import _read_batch, validate_quality as _validate_quality
        df = _read_batch(ingest_summary["batch_id"])
        return _validate_quality(df)

    @task()
    def detect_new_categories(ingest_summary: dict) -> dict:
        from pipeline.quality import _read_batch, _read_historical, detect_new_categories as _detect
        df = _read_batch(ingest_summary["batch_id"])
        hist = _read_historical()
        return _detect(df, hist)

    @task()
    def detect_data_drift(ingest_summary: dict) -> dict:
        from pipeline.quality import _read_batch, _read_historical, detect_drift
        df = _read_batch(ingest_summary["batch_id"])
        hist = _read_historical()
        return detect_drift(df, hist)

    @task()
    def preprocess_data(ingest_summary: dict) -> dict:
        from pipeline.preprocess import run
        return run(batch_id=ingest_summary["batch_id"])

    @task.branch()
    def decide_training(
        schema_result: dict,
        quality_result: dict,
        drift_result: dict,
        categories_result: dict,
    ) -> str:
        from pipeline.decide import run as decide_run
        quality_report = {
            **schema_result,
            **quality_result,
            **drift_result,
            **categories_result,
        }
        decision = decide_run(quality_report)
        return "train_candidate_model" if decision["should_train"] else "skip_training"

    @task()
    def skip_training(
        schema_result: dict,
        quality_result: dict,
        drift_result: dict,
        categories_result: dict,
    ) -> dict:
        from pipeline.decide import run as decide_run
        quality_report = {**schema_result, **quality_result, **drift_result, **categories_result}
        decision = decide_run(quality_report)
        return {"skipped": True, "reason": decision["reason"]}

    @task()
    def train_candidate_model(ingest_summary: dict,
                              schema_result: dict,
                              quality_result: dict,
                              drift_result: dict,
                              categories_result: dict) -> dict:
        from pipeline.split import run as split_run
        from pipeline.train import run as train_run
        from pipeline.decide import run as decide_run

        batch_id = ingest_summary["batch_id"]
        quality_report = {**schema_result, **quality_result, **drift_result, **categories_result}
        decision = decide_run(quality_report)

        split_run(batch_id=batch_id)
        return train_run(batch_id=batch_id, training_reason=decision["reason"])

    @task()
    def evaluate_candidate_model(candidate: dict) -> dict:
        return {
            "run_id": candidate["run_id"],
            "version": candidate["version"],
            "mae": candidate["metric"],
            "rmse": candidate["rmse"],
            "r2": candidate["r2"],
        }

    @task()
    def register_candidate_in_mlflow(candidate: dict) -> dict:
        return {
            "model_name": candidate["model_name"],
            "version": candidate["version"],
            "run_id": candidate["run_id"],
            "registered": candidate["version"] is not None,
        }

    @task()
    def compare_with_production(candidate: dict) -> dict:
        from pipeline.promote import compare
        return compare(candidate)

    @task.branch()
    def decide_promotion(comparison: dict) -> str:
        return "promote_model" if comparison["decision"] == "promote" else "reject_model"

    @task()
    def promote_model(candidate: dict) -> dict:
        from pipeline.promote import promote
        return promote(candidate)

    @task()
    def reject_model(comparison: dict) -> dict:
        return {"promoted": False, "reason": comparison.get("reason", "no supera al champion")}

    @task(trigger_rule="none_failed_min_one_success")
    def notify_or_log_result(
        ingest_summary: dict,
        schema_result: dict,
        quality_result: dict,
        drift_result: dict,
        categories_result: dict,
    ) -> dict:
        from pipeline.audit import create_entry, update_entry
        from pipeline.decide import run as decide_run

        batch_id = ingest_summary["batch_id"]
        quality_report = {**schema_result, **quality_result, **drift_result, **categories_result}
        decision = decide_run(quality_report)

        audit_id = create_entry(batch_id, ingest_summary["total_rows"])
        update_entry(
            audit_id,
            rows_after_dedup=ingest_summary["inserted"],
            schema_ok=schema_result.get("schema_ok"),
            quality_ok=quality_result.get("quality_ok"),
            drift_detected=drift_result.get("drift_detected"),
            training_decision=decision["should_train"],
            training_reason=decision["reason"],
            status="completed",
        )
        return {"audit_id": audit_id, "batch_id": batch_id}

    @task(trigger_rule="none_failed_min_one_success")
    def end() -> dict:
        return {"status": "finished"}

    # --- Wiring ---
    t_start = start()
    t_fetch = fetch_batch_from_api()
    t_store = store_raw_batch(t_fetch)

    t_schema = validate_schema(t_store)
    t_quality = validate_data_quality(t_store)
    t_categories = detect_new_categories(t_store)
    t_drift = detect_data_drift(t_store)

    t_preprocess = preprocess_data(t_store)

    t_decide = decide_training(t_schema, t_quality, t_drift, t_categories)

    t_skip = skip_training(t_schema, t_quality, t_drift, t_categories)

    t_train = train_candidate_model(t_store, t_schema, t_quality, t_drift, t_categories)
    t_evaluate = evaluate_candidate_model(t_train)
    t_register = register_candidate_in_mlflow(t_train)
    t_compare = compare_with_production(t_train)

    t_decide_promo = decide_promotion(t_compare)

    t_promote = promote_model(t_train)
    t_reject = reject_model(t_compare)

    t_notify = notify_or_log_result(t_store, t_schema, t_quality, t_drift, t_categories)
    t_end = end()

    # Dependencias
    t_start >> t_fetch >> t_store
    t_store >> [t_schema, t_quality, t_categories, t_drift, t_preprocess]
    [t_schema, t_quality, t_categories, t_drift, t_preprocess] >> t_decide
    t_decide >> [t_train, t_skip]
    t_train >> t_evaluate >> t_register >> t_compare >> t_decide_promo
    t_decide_promo >> [t_promote, t_reject]
    [t_promote, t_reject, t_skip] >> t_notify >> t_end


properties_mlops_pipeline()
