from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PenguinFeatures, PredictionResponse

app = FastAPI(title="Penguins ML API")

MODELS_DIR = Path("app/models")

models: dict = {}


@app.on_event("startup")
def load_models() -> None:
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        name = path.stem
        models[name] = joblib.load(path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(models),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PenguinFeatures):
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded. Run train.py first.")

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
