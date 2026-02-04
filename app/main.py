from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd
from palmerpenguins import load_penguins

app = FastAPI(title="Penguins API")

# Load data once at startup
penguins_df = load_penguins()


def _df_to_records(df: pd.DataFrame, limit: int):
    # Force object dtype so None stays None (not NaN) when serializing
    df = df.astype(object).where(pd.notnull(df), None)
    records = df.head(limit).to_dict(orient="records")
    return records


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
