"""Train multiple candidate models and log them to MLflow.

Two baseline candidates are trained: LogisticRegression and RandomForest.
Each model is logged as its own MLflow run with full metrics (accuracy,
precision, recall, F1, ROC-AUC) and registered into the Model Registry.

The metric chosen for selection is `f1` (logged also as `primary_metric`)
because the dataset target is typically imbalanced and the cost of a false
negative (missed positive case) is clinically more relevant than overall
accuracy. See README for the longer justification.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline.config import export_aws_env, load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def _read_clean(batch_id: str | None) -> pd.DataFrame:
    where = "WHERE batch_id = %s AND split IS NOT NULL" if batch_id else "WHERE split IS NOT NULL"
    params = (batch_id,) if batch_id else ()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT row_hash, split, features, target FROM clean.diabetes_clean {where}",
            params,
        )
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"no clean rows ready for training (batch_id={batch_id!r})")
    return pd.DataFrame(
        [
            {"row_hash": h, "split": s, "target": t, **f}
            for (h, s, f, t) in rows
        ]
    )


def _xy(df: pd.DataFrame, split: str, feature_cols: list[str]):
    sub = df[df["split"] == split]
    return sub[feature_cols].values, sub["target"].values


def _candidates(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1
        ),
    }


def run(batch_id: str | None = None) -> dict:
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    df = _read_clean(batch_id)
    feature_cols = [c for c in df.columns if c not in {"row_hash", "split", "target"}]

    X_train, y_train = _xy(df, "train", feature_cols)
    X_val, y_val = _xy(df, "val", feature_cols)
    X_test, y_test = _xy(df, "test", feature_cols)

    best = {"run_id": None, "version": None, "metric": -1.0, "model_name": None}

    for name, model in _candidates(settings.random_seed).items():
        with mlflow.start_run(run_name=f"{name}-{batch_id or 'all'}") as run_:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
            y_test_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "precision": precision_score(y_val, y_val_pred, zero_division=0),
                "recall": recall_score(y_val, y_val_pred, zero_division=0),
                "f1": f1_score(y_val, y_val_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            }
            if y_val_proba is not None and len(np.unique(y_val)) > 1:
                metrics["roc_auc"] = roc_auc_score(y_val, y_val_proba)
            metrics["primary_metric"] = metrics[settings.primary_metric]

            mlflow.log_params({"model_type": name, "batch_id": batch_id or "all", "seed": settings.random_seed})
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=settings.registered_model_name,
            )

            # Resolve the registered version we just created (latest for this run)
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"run_id='{run_.info.run_id}'")
            version = versions[0].version if versions else None

            logger.info("run %s metrics=%s version=%s", name, metrics, version)

            if metrics["primary_metric"] > best["metric"]:
                best = {
                    "run_id": run_.info.run_id,
                    "version": version,
                    "metric": metrics["primary_metric"],
                    "model_name": settings.registered_model_name,
                }

    logger.info("best candidate: %s", best)
    return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run())
