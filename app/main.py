from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from palmerpenguins import load_penguins

from app.schemas.predict import PenguinFeatures, PredictionResponse

app = FastAPI(title="Penguins API")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load data once at startup
penguins_df = load_penguins()

# Load trained model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "species_classifier.joblib"
model = joblib.load(MODEL_PATH)


def _df_to_records(df: pd.DataFrame, limit: int):
    # Force object dtype so None stays None (not NaN) when serializing
    df = df.astype(object).where(pd.notnull(df), None)
    records = df.head(limit).to_dict(orient="records")
    return records


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/penguins")
def list_penguins(
    species: Optional[str] = Query(None, description="Filter by species"),
    island: Optional[str] = Query(None, description="Filter by island"),
    sex: Optional[str] = Query(None, description="Filter by sex"),
    limit: int = Query(100, ge=1, le=500),
):
    df = penguins_df.copy()

    if species:
        df = df[df["species"].str.lower() == species.lower()]
    if island:
        df = df[df["island"].str.lower() == island.lower()]
    if sex:
        df = df[df["sex"].str.lower() == sex.lower()]

    payload = {"count": len(df), "data": _df_to_records(df, limit)}
    return JSONResponse(content=jsonable_encoder(payload))


@app.get("/penguins/{row_id}")
def get_penguin(row_id: int):
    if row_id < 0 or row_id >= len(penguins_df):
        raise HTTPException(status_code=404, detail="Penguin not found")
    row = penguins_df.iloc[row_id].to_frame().T
    payload = row.astype(object).where(pd.notnull(row), None).to_dict(orient="records")[0]
    return JSONResponse(content=jsonable_encoder(payload))


@app.post("/predict", response_model=PredictionResponse)
def predict_species(features: PenguinFeatures):
    input_df = pd.DataFrame([{
        "bill_length_mm": features.bill_length_mm,
        "bill_depth_mm": features.bill_depth_mm,
        "flipper_length_mm": features.flipper_length_mm,
        "body_mass_g": features.body_mass_g,
        "island": features.island.lower(),
        "sex": features.sex.lower(),
    }])

    species = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]
    class_names = model.classes_

    return PredictionResponse(
        species=species,
        probabilities={name: round(float(p), 4) for name, p in zip(class_names, probas)},
    )
