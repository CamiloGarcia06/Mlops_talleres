"""API configuration loaded from environment variables."""

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
    registered_model_name: str
    champion_alias: str
    model_cache_ttl_seconds: int


def load() -> Settings:
    s = Settings(
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
        registered_model_name=os.environ.get("MLFLOW_MODEL_NAME", "diabetes-classifier"),
        champion_alias=os.environ.get("CHAMPION_ALIAS", "champion"),
        model_cache_ttl_seconds=int(os.environ.get("MODEL_CACHE_TTL_SECONDS", "300")),
    )
    os.environ["AWS_ACCESS_KEY_ID"] = s.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = s.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s.s3_endpoint_url
    return s
