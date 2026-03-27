from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

from app.services.ml_engine import predict_forecast_72h, predict_source
from app.services.whri_calculator import calculate_whri

router = APIRouter(prefix="/api/v1/ml", tags=["ML Intelligence Phase 2"])

# ── Schemas ──
class HistoricalAQIData(BaseModel):
    aqi: List[float] = Field(..., min_items=72, max_items=72)
    pm25: List[float] = Field(..., min_items=72, max_items=72)
    pm10: List[float] = Field(..., min_items=72, max_items=72)
    wind_speed: List[float] = Field(..., min_items=72, max_items=72)
    humidity: List[float] = Field(..., min_items=72, max_items=72)
    temperature: List[float] = Field(..., min_items=72, max_items=72)
    boundary_layer_height: List[float] = Field(..., min_items=72, max_items=72)
    hour_sin: List[float] = Field(..., min_items=72, max_items=72)
    hour_cos: List[float] = Field(..., min_items=72, max_items=72)
    dow_sin: List[float] = Field(..., min_items=72, max_items=72)
    dow_cos: List[float] = Field(..., min_items=72, max_items=72)
    month_sin: List[float] = Field(..., min_items=72, max_items=72)
    month_cos: List[float] = Field(..., min_items=72, max_items=72)

class SourceFeatures(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    wind_speed: float
    wind_sin: float
    wind_cos: float
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    humidity: float
    temperature: float
    pm_ratio: float
    boundary_layer_height: float
    surface_pressure: float

# ── Endpoints ──
@router.post("/forecast/72h")
async def get_72h_forecast(data: HistoricalAQIData):
    """
    Predicts the next 72 hours of AQI based on the last 72 hours of data using the XGBoost Model.
    """
    try:
        df = pd.DataFrame(data.dict())
        forecast_array = predict_forecast_72h(df)
        return {
            "horizon": 72,
            "forecast_aqi": forecast_array,
            "peak_predicted_aqi": max(forecast_array),
            "average_predicted_aqi": round(sum(forecast_array)/72, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting Error: {str(e)}")

@router.post("/source/classify")
async def classify_source(data: SourceFeatures):
    """
    Classifies the main pollution source (Traffic, Dust, Biomass, Industry, Weather Trapped) using XGBoost.
    """
    try:
        predicted_source = predict_source(data.dict())
        return {"predicted_source": predicted_source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification Error: {str(e)}")

@router.get("/whri/{ward_name}")
async def get_whri_score(ward_name: str, aqi_score: float, dengue_risk: float = 0.0, heatwave_risk: float = 0.0):
    """
    Calculates the Ward Health Risk Index (WHRI) combining AQI, Dengue, and Heatwave risks.
    """
    try:
        result = calculate_whri(ward_name, aqi_score, dengue_risk, heatwave_risk)
        return result.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WHRI Calculation Error: {str(e)}")
