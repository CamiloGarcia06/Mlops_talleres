"""Pydantic schemas for the inference API.

The input is intentionally a dict-of-features (`Dict[str, float | int | str]`)
because the `clean.diabetes_clean` table stores features as JSONB and the
exact column set depends on the dataset variant in use (Pima vs. 130-US).
The model itself is a sklearn `Pipeline` whose first step is a
`ColumnTransformer` (or equivalent) trained over the same JSONB record, so
it accepts the dict shape as long as the keys match what was logged at
training time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ..., description="Feature dictionary matching the features used at training time."
    )


class PredictResponse(BaseModel):
    request_id: str
    prediction: int
    score: Optional[float] = None
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
