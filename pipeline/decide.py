"""Decision automatica de entrenamiento basada en criterios tecnicos."""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import export_aws_env, load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)

VOLUME_INCREASE_THRESHOLD = 0.10


def _champion_training_rows(client: MlflowClient, settings) -> int | None:
    """Recupera n_training_rows del champion actual desde MLflow."""
    try:
        version = client.get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        run = client.get_run(version.run_id)
        val = run.data.params.get("n_training_rows")
        return int(val) if val else None
    except Exception:
        return None


def _current_clean_count() -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM clean.properties_clean")
        return cur.fetchone()[0]


def run(quality_report: dict) -> dict:
    """Evalua si se debe entrenar un nuevo modelo.

    Criterios (cualquiera dispara entrenamiento):
      1. No hay champion (primer lote).
      2. Data drift significativo.
      3. Nuevas categorias frecuentes.
      4. Volumen acumulado aumento >= 10% vs datos del champion.
    """
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    reasons: list[str] = []

    # 1. No hay champion
    try:
        client.get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        has_champion = True
    except Exception:
        has_champion = False
        reasons.append("no hay modelo champion — primer entrenamiento")

    # 2. Drift
    if quality_report.get("drift_detected"):
        drifted = quality_report.get("drifted_features", {})
        reasons.append(f"drift detectado en {len(drifted)} features: {list(drifted.keys())}")

    # 3. Nuevas categorias
    if quality_report.get("significant"):
        cats = quality_report.get("new_categories", {})
        reasons.append(f"categorias nuevas significativas: {cats}")

    # 4. Volumen
    if has_champion:
        champion_rows = _champion_training_rows(client, settings)
        current_rows = _current_clean_count()
        if champion_rows and current_rows > 0:
            increase = (current_rows - champion_rows) / champion_rows
            if increase >= VOLUME_INCREASE_THRESHOLD:
                reasons.append(
                    f"volumen aumento {increase:.1%} (de {champion_rows} a {current_rows})"
                )

    should_train = len(reasons) > 0
    reason_text = "; ".join(reasons) if reasons else "sin cambios significativos"

    result = {
        "should_train": should_train,
        "reason": reason_text,
        "has_champion": has_champion,
    }
    logger.info("decision de entrenamiento: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run({}))
