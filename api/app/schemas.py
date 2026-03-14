from pydantic import BaseModel
from typing import Optional


class CovertypeFeatures(BaseModel):
    Elevation: float
    Aspect: float
    Slope: float
    Horizontal_Distance_To_Hydrology: float
    Vertical_Distance_To_Hydrology: float
    Horizontal_Distance_To_Roadways: float
    Hillshade_9am: float
    Hillshade_Noon: float
    Hillshade_3pm: float
    Horizontal_Distance_To_Fire_Points: float
    Wilderness_Area: str
    Soil_Type: str
    model_name: str = "random_forest"


class PredictionResponse(BaseModel):
    cover_type: str
    model_used: str
