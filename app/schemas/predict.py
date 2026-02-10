from pydantic import BaseModel, Field


class PenguinFeatures(BaseModel):
    bill_length_mm: float = Field(..., gt=0, description="Largo del pico en mm")
    bill_depth_mm: float = Field(..., gt=0, description="Profundidad del pico en mm")
    flipper_length_mm: float = Field(..., gt=0, description="Largo de la aleta en mm")
    body_mass_g: float = Field(..., gt=0, description="Masa corporal en gramos")
    island: str = Field(..., description="Isla: torgersen, biscoe o dream")
    sex: str = Field(..., description="Sexo: male, female o unknown")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "bill_length_mm": 39.1,
                "bill_depth_mm": 18.7,
                "flipper_length_mm": 181.0,
                "body_mass_g": 3750.0,
                "island": "torgersen",
                "sex": "male",
            }
        ]
    }}


class PredictionResponse(BaseModel):
    species: str
    probabilities: dict[str, float]
