"""
simulator_engine.py
NagarMitra Phase 4 — What-If Edge Simulator
Calculates hypothetical AQI drops based on proposed civic policies.
"""
from pydantic import BaseModel

class SimulationResponse(BaseModel):
    policy_action: str
    severity_percentage: float
    current_aqi: float
    expected_new_aqi: float
    aqi_drop_points: float
    impact_category: str
    message: str

def simulate_policy_impact(current_aqi: float, policy_action: str, severity: float) -> SimulationResponse:
    """
    Mathematically downscales AQI based on the expected pollution reduction of specific civic actions.
    `severity` is the enforcement percentage (0.0 to 100.0).
    """
    # Max potential AQI reduction coefficients for each policy at 100% enforcement
    POLICY_WEIGHTS = {
        "ban_heavy_vehicles":          0.25, # Traffic contributes heavily to PM2.5/NO2
        "halt_construction":           0.18, # Halts dust resuspension (PM10)
        "stop_biomass_burning":        0.35, # Major PM2.5 and CO contributor
        "deploy_smog_towers":          0.05, # Localized impact only
        "subsidize_public_transit":    0.12, # Gradual traffic reduction
        "odd_even_rule":               0.20  # Immediate 50% private vehicle cut
    }
    
    action = str(policy_action).lower().strip()
    reduction_factor = POLICY_WEIGHTS.get(action, 0.10) # default 10% if unknown
    
    # Calculate the normalized severity (0.0 to 1.0)
    enforcement = max(0.0, min(100.0, float(severity))) / 100.0
    
    # Calculate drop
    aqi_drop = current_aqi * reduction_factor * enforcement
    new_aqi = max(0.0, current_aqi - aqi_drop)
    
    # Evaluate Impact
    if enforcement < 0.2:
        impact = "Low Impact (Poor Enforcement)"
        msg = f"Enforcing this policy at only {severity}% yields minimal results."
    elif aqi_drop > 100:
        impact = "Massive Impact"
        msg = f"This policy dramatically cuts the AQI down securely by {round(aqi_drop)} points!"
    elif aqi_drop > 40:
        impact = "Significant Impact"
        msg = f"Meaningful reduction achieved. Ward AQI drops by {round(aqi_drop)} points."
    else:
        impact = "Moderate Impact"
        msg = f"AQI will drop by {round(aqi_drop)} points. Consider combining with other policies."
        
    return SimulationResponse(
        policy_action=action,
        severity_percentage=severity,
        current_aqi=round(current_aqi, 1),
        expected_new_aqi=round(new_aqi, 1),
        aqi_drop_points=round(aqi_drop, 1),
        impact_category=impact,
        message=msg
    )
