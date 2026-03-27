from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.simulator_engine import simulate_policy_impact, SimulationResponse
from app.services.policy_engine import get_policies_for_source
from app.services.aqi_fetcher import fetch_aqicn_all_delhi_stations
from app.services.ward_mapper import get_ward_aqi

router = APIRouter(prefix="/api/v1/admin", tags=["Admin Intelligence Phase 4"])

class SimulationRequest(BaseModel):
    ward_name: str
    policy_action: str
    severity: float

@router.post("/simulate", response_model=SimulationResponse)
async def run_what_if_simulation(request: SimulationRequest):
    """
    **What-If Simulator**: Predicts the new AQI of a ward based on a hypothetical policy action.
    Hooks into LIVE data.
    """
    try:
        # 1. Fetch live stations
        stations = await fetch_aqicn_all_delhi_stations()
        if not stations:
            raise HTTPException(503, "Live data unavailable for simulation.")
            
        # 2. Get live Ward AQI
        ward_data = get_ward_aqi(request.ward_name, stations)
        if ward_data.get("error"):
            raise HTTPException(404, f"Ward '{request.ward_name}' not found.")
            
        current_aqi = float(ward_data.get("aqi", 0))
        if current_aqi == 0:
            raise HTTPException(400, "Unable to find reliable AQI for this ward.")
            
        # 3. Simulate Policy
        result = simulate_policy_impact(
            current_aqi=current_aqi,
            policy_action=request.policy_action,
            severity=request.severity
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation Engine Error: {str(e)}")


@router.get("/policies/{source_classification}")
async def get_actionable_policies(source_classification: str):
    """
    **Policy Engine**: Returns actionable civic interventions for the predicted pollution source.
    Useful for Admin Dashboards advising local MLAs.
    """
    policies = get_policies_for_source(source_classification)
    return {
        "source": source_classification.lower(),
        "total_recommendations": len(policies),
        "actionable_policies": policies
    }
