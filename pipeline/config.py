"""Centralized configuration read from environment variables.

Defaults assume in-cluster service DNS. Locally, override with port-forwards
and the corresponding env vars before invoking the CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    pg_dsn: str
    mlflow_tracking_uri: str
    s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str

    source_csv: str
    batch_size: int
    random_seed: int

    experiment_name: str
    registered_model_name: str
    champion_alias: str
    primary_metric: str  # name of the metric used for selection (logged as `primary_metric`)


def load() -> Settings:
    return Settings(
        pg_dsn=os.environ.get(
            "PG_DSN",
            "postgresql://mlops_user:mlops_pass_2026@postgres-service.mlops.svc.cluster.local:5432/mlops",
        ),
        mlflow_tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://mlflow-service.mlops.svc.cluster.local:5000",
        ),
        s3_endpoint_url=os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL",
            "http://minio-service.mlops.svc.cluster.local:9000",
        ),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        source_csv=os.environ.get("SOURCE_CSV", "/data/Diabetes.csv"),
        batch_size=int(os.environ.get("BATCH_SIZE", "15000")),
        random_seed=int(os.environ.get("RANDOM_SEED", "42")),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT", "diabetes-classification"),
        registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "diabetes-classifier"),
        champion_alias=os.environ.get("CHAMPION_ALIAS", "champion"),
        primary_metric=os.environ.get("PRIMARY_METRIC", "f1"),
    )


def export_aws_env(settings: Settings) -> None:
    """Ensure boto3 / mlflow pick up the configured S3 credentials."""
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.s3_endpoint_url
