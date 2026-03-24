import os
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

MODEL_NAME = "wine-classifier-production"
models: dict = {}


class WineFeatures(BaseModel):
    model_name: str
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_diluted_wines: float
    proline: float


class PredictionResponse(BaseModel):
    prediction: int
    model_used: str


def _load_models() -> None:
    """Carga todas las versiones del modelo desde MLflow Registry."""
    models.clear()
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for v in versions:
            run = client.get_run(v.run_id)
            run_name = run.info.run_name
            model_uri = f"models:/{MODEL_NAME}/{v.version}"
            models[run_name] = mlflow.sklearn.load_model(model_uri)
            print(f"Cargado: {run_name}  (v{v.version})")
    except Exception as e:
        print(f"Warning: No se pudieron cargar modelos: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    _load_models()
    yield


app = FastAPI(title="Wine Inference API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models)}


@app.get("/models")
def list_models():
    return {"available": list(models.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures):
    if not models:
        raise HTTPException(status_code=503, detail="No hay modelos cargados.")

    if features.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{features.model_name}' no encontrado. Disponibles: {list(models.keys())}",
        )

    row = pd.DataFrame([features.model_dump(exclude={"model_name"})])
    prediction = int(models[features.model_name].predict(row)[0])
    return PredictionResponse(prediction=prediction, model_used=features.model_name)


@app.post("/reload")
def reload_models():
    """Recarga los modelos desde MLflow Registry."""
    _load_models()
    return {"status": "recargado", "models_loaded": len(models)}