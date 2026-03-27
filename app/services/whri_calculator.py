"""
whri_calculator.py
NagarMitra — Ward Health Risk Index (WHRI) Calculator
"""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class WHRIResult:
    ward_name: str
    whri_score: float
    risk_band: str
    risk_color: str
    aqi_contribution: float
    dengue_contribution: float
    heatwave_contribution: float
    alert_message: str

AQI_BREAKPOINTS = [
    (0,   50,  0,   20),   # Good
    (51,  100, 20,  40),   # Satisfactory
    (101, 200, 40,  60),   # Moderate
    (201, 300, 60,  75),   # Poor
    (301, 400, 75,  90),   # Very Poor
    (401, 500, 90,  100),  # Severe
]

def normalize_aqi(aqi: float) -> float:
    """Converts raw AQI (0-500) to normalized 0-100 score."""
    aqi = max(0, min(500, float(aqi)))
    for aqi_lo, aqi_hi, score_lo, score_hi in AQI_BREAKPOINTS:
        if aqi_lo <= aqi <= aqi_hi:
            ratio = (aqi - aqi_lo) / (aqi_hi - aqi_lo)
            return score_lo + ratio * (score_hi - score_lo)
    return 100.0

def calculate_whri(
    ward_name: str,
    aqi_score: float,
    dengue_risk_score: float = 0.0,
    heatwave_risk_score: float = 0.0,
    aqi_weight: float = 0.40,
    dengue_weight: float = 0.35,
    heatwave_weight: float = 0.25,
) -> WHRIResult:
    
    # Normalize inputs to 0-100
    aqi_norm    = normalize_aqi(aqi_score)
    dengue_norm = max(0, min(100, float(dengue_risk_score)))
    heat_norm   = max(0, min(100, float(heatwave_risk_score)))

    # Compute component contributions
    aqi_contrib    = aqi_norm * aqi_weight
    dengue_contrib = dengue_norm * dengue_weight
    heat_contrib   = heat_norm * heatwave_weight

    # Final WHRI score
    whri = round(max(0, min(100, aqi_contrib + dengue_contrib + heat_contrib)), 2)

    if whri <= 30:
        band, color, msg = "Low", "#2ECC71", "Environmental conditions are within acceptable limits."
    elif whri <= 60:
        band, color, msg = "Moderate", "#F39C12", "Sensitive groups (elderly, children, asthma patients) should take precautions."
    elif whri <= 80:
        band, color, msg = "High", "#E67E22", "Outdoor activity restriction advised. Health advisory in effect."
    else:
        band, color, msg = "Critical", "#E74C3C", "EMERGENCY: Avoid outdoor exposure. Vulnerable groups must stay indoors."

    return WHRIResult(
        ward_name=ward_name, whri_score=whri, risk_band=band, risk_color=color,
        aqi_contribution=round(aqi_contrib, 2), dengue_contribution=round(dengue_contrib, 2),
        heatwave_contribution=round(heat_contrib, 2), alert_message=msg,
    )

def batch_whri(ward_data: list) -> list:
    results = []
    for ward in ward_data:
        results.append(calculate_whri(
            ward_name=ward.get("ward_name", "Unknown"),
            aqi_score=ward.get("aqi_score", 100),
            dengue_risk_score=ward.get("dengue_risk_score", 0),
            heatwave_risk_score=ward.get("heatwave_risk_score", 0),
        ))
    return results
