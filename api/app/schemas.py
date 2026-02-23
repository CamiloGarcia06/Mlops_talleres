from pydantic import BaseModel


class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: str
    sex: str
    model_name: str


class PredictionResponse(BaseModel):
    species: str
    model_used: str
