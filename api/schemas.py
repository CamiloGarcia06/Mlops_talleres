"""Esquemas Pydantic para la API de inferencia de precios."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Diccionario de features que coincide con las usadas al entrenar.",
    )


class PredictResponse(BaseModel):
    request_id: str
    prediction: float
    model_name: str
    model_version: str
    model_alias: str
    processing_time_ms: float


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_alias: str
    loaded_at: float
    cache_ttl_seconds: int


class HealthResponse(BaseModel):
    status: str = "ok"
