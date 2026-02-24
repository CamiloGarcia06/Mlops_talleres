from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PenguinFeatures, PredictionResponse

app = FastAPI(title="Penguins ML API")

MODELS_DIR = Path("/models")

models: dict = {}


def _load_models() -> None:
    """Load all .joblib models from the shared models directory."""
    models.clear()
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        name = path.stem
        models[name] = joblib.load(path)


@app.on_event("startup")
def startup() -> None:
    _load_models()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(models),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PenguinFeatures):
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded.")

    if features.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{features.model_name}' not found. Available: {list(models.keys())}",
        )

    row = pd.DataFrame([features.model_dump(exclude={"model_name"})])
    pipeline = models[features.model_name]
    species = pipeline.predict(row)[0]
    return PredictionResponse(species=species, model_used=features.model_name)


@app.get("/models")
def list_models():
    return {"available": list(models.keys())}


@app.post("/reload")
def reload_models():
    """Reload models from the shared volume after training new ones."""
    _load_models()
    return {"status": "reloaded", "models_loaded": len(models)}
