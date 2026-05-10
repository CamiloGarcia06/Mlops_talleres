"""Thread-safe MLflow model cache with TTL and explicit reload.

Strategy:
  - On first request, the model identified by alias `champion` is downloaded
    from MLflow (artifacts come from MinIO via the S3 endpoint).
  - The loaded model + its registry version is kept in memory.
  - The cache expires after `model_cache_ttl_seconds`; the next prediction
    triggers a re-resolution of the alias. This means a champion promotion
    propagates automatically without redeploy, with at most TTL-seconds of
    staleness.
  - `POST /reload-model` forces an immediate refresh.

The contract intentionally avoids any local file path: the only source of
truth is MLflow's Model Registry.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from api.config import Settings, load

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: Any
    name: str
    version: str
    alias: str
    loaded_at: float


class ModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None
        self._settings: Settings = load()
        mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
        self._client = MlflowClient()

    def get(self) -> LoadedModel:
        with self._lock:
            if self._loaded is None or self._is_expired(self._loaded):
                self._loaded = self._load()
            return self._loaded

    def reload(self) -> LoadedModel:
        with self._lock:
            self._loaded = self._load()
            return self._loaded

    # ----- internals ------------------------------------------------------

    def _is_expired(self, lm: LoadedModel) -> bool:
        ttl = self._settings.model_cache_ttl_seconds
        if ttl <= 0:
            return False
        return (time.time() - lm.loaded_at) > ttl

    def _load(self) -> LoadedModel:
        s = self._settings
        version = self._client.get_model_version_by_alias(
            s.registered_model_name, s.champion_alias
        )
        uri = f"models:/{s.registered_model_name}@{s.champion_alias}"
        logger.info("loading model %s version=%s", uri, version.version)
        model = mlflow.pyfunc.load_model(uri)
        return LoadedModel(
            model=model,
            name=s.registered_model_name,
            version=str(version.version),
            alias=s.champion_alias,
            loaded_at=time.time(),
        )


_cache: ModelCache | None = None


def get_cache() -> ModelCache:
    global _cache
    if _cache is None:
        _cache = ModelCache()
    return _cache
