"""Servicio de inferencia FastAPI para prediccion de precios de propiedades."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from api import db
from api.config import load
from api.metrics import (
    INFERENCE_LATENCY_SECONDS,
    PREDICTIONS_TOTAL,
    set_model_info,
)
from api.model_loader import LoadedModel, get_cache
from api.schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

settings = load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        lm = get_cache().get()
        set_model_info(lm.name, lm.version, lm.alias)
        logger.info("model pre-loaded at startup: %s v%s", lm.name, lm.version)
    except Exception as e:
        logger.warning("startup model pre-load failed (will retry on first request): %s", e)
    yield


app = FastAPI(title="Real Estate Price Inference API", version="1.0.0", lifespan=lifespan)

Instrumentator(
    excluded_handlers=["/metrics", "/health"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    try:
        lm = get_cache().get()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"no model loaded: {e}") from e
    set_model_info(lm.name, lm.version, lm.alias)
    return ModelInfo(
        model_name=lm.name,
        model_version=lm.version,
        model_alias=lm.alias,
        loaded_at=lm.loaded_at,
        cache_ttl_seconds=settings.model_cache_ttl_seconds,
    )


@app.post("/reload-model", response_model=ModelInfo)
def reload_model() -> ModelInfo:
    try:
        lm = get_cache().reload()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"reload failed: {e}") from e
    set_model_info(lm.name, lm.version, lm.alias)
    return ModelInfo(
        model_name=lm.name,
        model_version=lm.version,
        model_alias=lm.alias,
        loaded_at=lm.loaded_at,
        cache_ttl_seconds=settings.model_cache_ttl_seconds,
    )


def _predict_with_model(lm: LoadedModel, features: dict) -> float:
    df = pd.DataFrame([features])
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = df[col].astype("float64")
    with INFERENCE_LATENCY_SECONDS.time():
        raw = lm.model.predict(df)
    return float(raw[0]) if hasattr(raw, "__len__") else float(raw)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        lm = get_cache().get()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"no model available: {e}") from e
    set_model_info(lm.name, lm.version, lm.alias)

    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        prediction = _predict_with_model(lm, req.features)
    except Exception as e:
        logger.exception("prediction failure")
        raise HTTPException(status_code=400, detail=f"invalid features: {e}") from e

    latency_ms = (time.perf_counter() - started) * 1000.0
    PREDICTIONS_TOTAL.labels(prediction="price").inc()

    db.log_inference(
        request_id=request_id,
        input_payload=req.features,
        prediction=prediction,
        model_name=lm.name,
        model_version=lm.version,
        latency_ms=latency_ms,
    )

    return PredictResponse(
        request_id=request_id,
        prediction=prediction,
        model_name=lm.name,
        model_version=lm.version,
        model_alias=lm.alias,
        processing_time_ms=latency_ms,
    )
