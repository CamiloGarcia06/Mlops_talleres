"""Compara candidato vs champion y promueve si cumple la regla de regresion.

Regla: promover si MAE baja al menos 3% Y RMSE no empeora mas de 1%.
Si no hay champion previo, se promueve automaticamente.
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import export_aws_env, load

logger = logging.getLogger(__name__)

MAE_IMPROVEMENT_THRESHOLD = 0.03
RMSE_DEGRADATION_THRESHOLD = 0.01


def _client() -> MlflowClient:
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def _get_champion_metrics(client: MlflowClient, settings) -> dict | None:
    try:
        version = client.get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        run = client.get_run(version.run_id)
        return {
            "version": version.version,
            "run_id": version.run_id,
            "mae": float(run.data.metrics.get("test_mae", float("inf"))),
            "rmse": float(run.data.metrics.get("test_rmse", float("inf"))),
        }
    except Exception:
        return None


def compare(candidate: dict) -> dict:
    settings = load()
    client = _client()

    candidate_mae = float(candidate.get("metric", float("inf")))
    candidate_rmse = float(candidate.get("rmse", float("inf")))

    champion = _get_champion_metrics(client, settings)

    if champion is None:
        return {
            "candidate_version": candidate.get("version"),
            "candidate_mae": candidate_mae,
            "candidate_rmse": candidate_rmse,
            "champion_version": None,
            "champion_mae": None,
            "champion_rmse": None,
            "decision": "promote",
            "reason": "no hay champion previo — primer modelo",
        }

    mae_improvement = (champion["mae"] - candidate_mae) / champion["mae"] if champion["mae"] > 0 else 0
    rmse_change = (candidate_rmse - champion["rmse"]) / champion["rmse"] if champion["rmse"] > 0 else 0

    mae_ok = mae_improvement >= MAE_IMPROVEMENT_THRESHOLD
    rmse_ok = rmse_change <= RMSE_DEGRADATION_THRESHOLD

    if mae_ok and rmse_ok:
        decision = "promote"
        reason = f"MAE mejoro {mae_improvement:.1%} (>= {MAE_IMPROVEMENT_THRESHOLD:.0%}) y RMSE cambio {rmse_change:+.1%} (<= {RMSE_DEGRADATION_THRESHOLD:.0%})"
    elif not mae_ok:
        decision = "keep"
        reason = f"MAE mejoro solo {mae_improvement:.1%} (requiere >= {MAE_IMPROVEMENT_THRESHOLD:.0%})"
    else:
        decision = "keep"
        reason = f"RMSE empeoro {rmse_change:+.1%} (maximo permitido {RMSE_DEGRADATION_THRESHOLD:.0%})"

    result = {
        "candidate_version": candidate.get("version"),
        "candidate_mae": candidate_mae,
        "candidate_rmse": candidate_rmse,
        "champion_version": champion["version"],
        "champion_mae": champion["mae"],
        "champion_rmse": champion["rmse"],
        "mae_improvement": round(mae_improvement, 4),
        "rmse_change": round(rmse_change, 4),
        "decision": decision,
        "reason": reason,
    }
    logger.info("comparacion: %s", result)
    return result


def promote(candidate: dict) -> dict:
    settings = load()
    client = _client()

    decision = compare(candidate)
    if decision["decision"] == "promote" and candidate.get("version"):
        client.set_registered_model_alias(
            name=settings.registered_model_name,
            alias=settings.champion_alias,
            version=str(candidate["version"]),
        )
        logger.info("promovida version %s al alias '%s'", candidate["version"], settings.champion_alias)
        decision["promoted"] = True
    else:
        logger.info("champion sin cambios: %s", decision)
        decision["promoted"] = False
    return decision


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("promote requiere un dict de candidato; invocar via CLI 'all'")
