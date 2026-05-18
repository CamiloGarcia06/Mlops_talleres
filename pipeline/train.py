"""Train multiple candidate models and log them to MLflow.

Two baseline candidates are trained: LogisticRegression and RandomForest.
Each candidate is wrapped in a sklearn Pipeline:

    Pipeline([
        ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough",                          numeric_cols),
        ]),
        Classifier(...),
    ])

So the encoder is **serialised together with the model** in MLflow. The
API receives raw features (strings for categoricals, numbers for numerics)
and the pipeline handles encoding at inference time. New categorical
values seen at inference are silently ignored by `handle_unknown="ignore"`.

The metric chosen for selection is `f1` because the dataset target is
imbalanced (~11% positives) and the cost of a false negative (a missed
readmission) is clinically more relevant than overall accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pipeline.config import export_aws_env, load
from pipeline.db.connection import connect

logger = logging.getLogger(__name__)


def _read_clean(batch_id: str | None) -> pd.DataFrame:
    # Always train on ALL accumulated clean rows so the feature schema seen
    # by the encoder is consistent across runs.
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT row_hash, split, features, target FROM clean.diabetes_clean "
            "WHERE split IS NOT NULL"
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


def _split_xy(df: pd.DataFrame, split: str, feature_cols: list[str]):
    sub = df[df["split"] == split]
    return sub[feature_cols].copy(), sub["target"].values


def _build_pipeline(estimator: Any, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])


def _candidates(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=seed, n_jobs=2,
            class_weight="balanced",
        ),
    }


def run(batch_id: str | None = None) -> dict:
    settings = load()
    export_aws_env(settings)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    df = _read_clean(batch_id)
    feature_cols = [c for c in df.columns if c not in {"row_hash", "split", "target"}]

    # numeric vs categorical split based on the actual dtypes coming from the
    # clean JSONB. preprocess.py casts categoricals to str, so anything left
    # as float64/int64 here is genuinely numeric.
    train_df = df[df["split"] == "train"][feature_cols]
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Cast all numerics to float64 so the inferred MLflow signature accepts
    # either int or float from API clients. Without this, ID columns like
    # patient_nbr/encounter_id stay as int64 and MLflow refuses to downcast
    # incoming float64 values at /predict time.
    df[numeric_cols] = df[numeric_cols].astype("float64")

    X_train, y_train = _split_xy(df, "train", feature_cols)
    X_val, y_val = _split_xy(df, "val", feature_cols)
    X_test, y_test = _split_xy(df, "test", feature_cols)

    best = {"run_id": None, "version": None, "metric": -1.0, "model_name": None}

    for name, estimator in _candidates(settings.random_seed).items():
        with mlflow.start_run(run_name=f"{name}-{batch_id or 'all'}") as run_:
            pipeline = _build_pipeline(estimator, numeric_cols, categorical_cols)
            pipeline.fit(X_train, y_train)

            y_val_pred = pipeline.predict(X_val)
            y_val_proba = (
                pipeline.predict_proba(X_val)[:, 1]
                if hasattr(pipeline, "predict_proba") else None
            )
            y_test_pred = pipeline.predict(X_test)

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

            mlflow.log_params({
                "model_type": name,
                "batch_id": batch_id or "all",
                "seed": settings.random_seed,
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(categorical_cols),
            })
            mlflow.log_metrics(metrics)

            # Signature on RAW features so the API knows what schema to send.
            # Two precautions that proved necessary on this cluster:
            #   - no `input_example`: it makes mlflow re-load the freshly
            #     saved model to validate the example, and that round-trip
            #     hangs under low memory pressure.
            #   - explicit `pip_requirements`: bypasses mlflow's automatic
            #     environment detection, which performs HTTP lookups that
            #     hang behind the cluster's egress restrictions.
            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=settings.registered_model_name,
                signature=signature,
                pip_requirements=[
                    "mlflow",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                ],
            )

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
