"""Entrena modelos de regresion y los registra en MLflow."""

from __future__ import annotations

import io
import logging
import subprocess
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.config import export_aws_env, load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def _read_clean(batch_id: str | None) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, split, features, target FROM clean.properties_clean "
            "WHERE split IS NOT NULL"
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no hay filas limpias para entrenar (batch_id={batch_id!r})")
    return pd.DataFrame([{"row_hash": h, "split": s, "target": t, **f} for h, s, f, t in rows])


def _split_xy(df: pd.DataFrame, split: str, feature_cols: list[str]):
    sub = df[df["split"] == split]
    return sub[feature_cols].copy(), sub["target"].values


def _build_pipeline(estimator: Any, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", estimator),
    ])


def _candidates(seed: int) -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=10,
            random_state=seed, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=seed,
        ),
    }


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _plot_residuals(y_true, y_pred) -> bytes:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plot_feature_importance(pipeline_obj: Pipeline, feature_names: list[str]) -> bytes | None:
    regressor = pipeline_obj.named_steps["regressor"]
    if not hasattr(regressor, "feature_importances_"):
        return None
    preprocessor = pipeline_obj.named_steps["preprocessor"]
    try:
        transformed_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        transformed_names = [f"f{i}" for i in range(len(regressor.feature_importances_))]

    importances = regressor.feature_importances_
    top_n = min(20, len(importances))
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.barh(range(top_n), importances[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([transformed_names[i] for i in indices], fontsize=8)
    ax.set_title("Feature Importance (top 20)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def run(batch_id: str | None = None, model: str | None = None, training_reason: str | None = None) -> dict:
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    df = _read_clean(batch_id)
    feature_cols = [c for c in df.columns if c not in {"row_hash", "split", "target"}]

    train_df = df[df["split"] == "train"][feature_cols]
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    df[numeric_cols] = df[numeric_cols].astype("float64")

    X_train, y_train = _split_xy(df, "train", feature_cols)
    X_test, y_test = _split_xy(df, "test", feature_cols)

    best = {"run_id": None, "version": None, "metric": float("inf"), "model_name": None}
    commit = _commit_sha()

    all_candidates = _candidates(settings.random_seed)
    if model:
        if model not in all_candidates:
            raise ValueError(f"modelo desconocido {model!r}; opciones: {sorted(all_candidates)}")
        selected = {model: all_candidates[model]}
    else:
        selected = all_candidates

    batch_ids_used = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT batch_id FROM clean.properties_clean WHERE split IS NOT NULL")
        batch_ids_used = [r[0] for r in cur.fetchall()]

    for name, estimator in selected.items():
        with mlflow.start_run(run_name=f"{name}-{batch_id or 'all'}") as run_:
            pipeline = _build_pipeline(estimator, numeric_cols, categorical_cols)
            pipeline.fit(X_train, y_train)

            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            train_metrics = _regression_metrics(y_train, y_train_pred)
            test_metrics = _regression_metrics(y_test, y_test_pred)

            all_metrics = {
                f"train_{k}": v for k, v in train_metrics.items()
            }
            all_metrics.update({
                f"test_{k}": v for k, v in test_metrics.items()
            })

            mlflow.log_params({
                "model_type": name,
                "batch_id": batch_id or "all",
                "seed": settings.random_seed,
                "n_training_rows": len(X_train),
                "n_features": len(feature_cols),
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(categorical_cols),
                "batch_ids_used": ",".join(batch_ids_used),
            })
            mlflow.log_metrics(all_metrics)

            mlflow.set_tags({
                "commit_sha": commit,
                "training_reason": training_reason or "manual",
                "pipeline_version": "2.0",
            })

            residuals_png = _plot_residuals(y_test, y_test_pred)
            mlflow.log_image(plt.imread(io.BytesIO(residuals_png)), "residuals.png")

            fi_png = _plot_feature_importance(pipeline, feature_cols)
            if fi_png:
                mlflow.log_image(plt.imread(io.BytesIO(fi_png)), "feature_importance.png")

            signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=settings.registered_model_name,
                signature=signature,
                pip_requirements=["mlflow", "scikit-learn", "pandas", "numpy"],
            )

            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"run_id='{run_.info.run_id}'")
            version = versions[0].version if versions else None

            logger.info("run %s test_metrics=%s version=%s", name, test_metrics, version)

            # Para regresion, menor MAE es mejor
            if test_metrics["mae"] < best["metric"]:
                best = {
                    "run_id": run_.info.run_id,
                    "version": version,
                    "metric": test_metrics["mae"],
                    "rmse": test_metrics["rmse"],
                    "r2": test_metrics["r2"],
                    "model_name": settings.registered_model_name,
                }

    logger.info("mejor candidato: %s", best)
    return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
