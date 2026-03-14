import io
import os
import tempfile

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from minio import Minio

from app.schemas import CovertypeFeatures, PredictionResponse

app = FastAPI(title="Covertype Inference API")

models: dict = {}


def _get_minio():
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=False,
    )


def _load_models() -> None:
    """Load all .joblib models from MinIO."""
    models.clear()
    try:
        client = _get_minio()
        bucket = os.environ["MINIO_BUCKET"]
        objects = client.list_objects(bucket)
        for obj in objects:
            if obj.object_name.endswith(".joblib"):
                name = obj.object_name.replace(".joblib", "")
                response = client.get_object(bucket, obj.object_name)
                data = io.BytesIO(response.read())
                response.close()
                response.release_conn()
                models[name] = joblib.load(data)
    except Exception as e:
        print(f"Warning: Could not load models from MinIO: {e}")


@app.on_event("startup")
def startup() -> None:
    _load_models()


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models)}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CovertypeFeatures):
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded. Run the pipeline first.")

    if features.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{features.model_name}' not found. Available: {list(models.keys())}",
        )

    row = pd.DataFrame([features.model_dump(exclude={"model_name"})])
    pipeline = models[features.model_name]
    cover_type = str(pipeline.predict(row)[0])
    return PredictionResponse(cover_type=cover_type, model_used=features.model_name)


@app.get("/models")
def list_models():
    return {"available": list(models.keys())}


@app.post("/reload")
def reload_models():
    """Reload models from MinIO after training new ones."""
    _load_models()
    return {"status": "reloaded", "models_loaded": len(models)}
