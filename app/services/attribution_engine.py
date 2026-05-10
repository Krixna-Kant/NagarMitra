"""
NagarMitra - Phase 1
Pollution Source Attribution Engine
Explains WHY AQI is high in a ward using data signals.
No satellite required - uses AQI + weather + time patterns.
"""

import math
from datetime import datetime
from typing import Optional


# ─── Attribution Logic ────────────────────────────────────────────────────────

def attribute_pollution_sources(
    aqi: float,
    pollutants: dict,
    weather: dict,
    ward_name: str,
    timestamp: Optional[str] = None,
) -> dict:
    """
    Given AQI, pollutant breakdown, and weather, return:
    - Dominant pollution source and % contribution
    - Confidence score
    - Human-readable explanation (English + Hindi)
    """
    if aqi is None or aqi < 50:
        return _clean_air_response(aqi, ward_name)

    hour = _get_hour(timestamp)
    wind_speed    = weather.get("wind_speed", 3.0)
    humidity      = weather.get("humidity", 50)
    temperature   = weather.get("temperature", 25)
    weather_main  = weather.get("weather_main", "Clear")

    # --- Compute source scores (0.0–1.0) ---
    scores = {}

    # 1. Traffic: high during rush hours (7-10am, 5-9pm), high PM2.5 + NO2
    traffic_score = _traffic_score(hour, pollutants, wind_speed)
    scores["traffic"] = traffic_score

    # 2. Dust: high PM10 relative to PM2.5, low humidity, medium wind
    dust_score = _dust_score(pollutants, humidity, wind_speed, temperature)
    scores["dust_construction"] = dust_score

    # 3. Biomass/Stubble burning: high PM2.5, high CO, evening/night
    burning_score = _burning_score(hour, pollutants, weather_main)
    scores["biomass_burning"] = burning_score

    # 4. Industrial: high SO2 or NO2, weekday working hours
    industrial_score = _industrial_score(pollutants, hour)
    scores["industrial"] = industrial_score

    # 5. Weather trapping: low wind causes any source to accumulate
    trapping_score = _trapping_score(wind_speed, humidity)
    scores["weather_trapping"] = trapping_score

    # 6. Ward Profile Offsets (Land-use context)
    ward_type = _get_ward_type(ward_name)
    if ward_type == "industrial":
        scores["industrial"] = scores.get("industrial", 0) + 0.25
        scores["dust_construction"] += 0.1
    elif ward_type == "traffic_hub":
        scores["traffic"] += 0.3
    elif ward_type == "biomass_sensitive":
        scores["biomass_burning"] += 0.25

    # Normalize to percentages
    total = sum(scores.values())
    percentages = {k: round((v / total) * 100, 1) if total > 0 else 0
                   for k, v in scores.items()}

    dominant_source = max(percentages, key=percentages.get)
    confidence = _confidence_score(scores, aqi, pollutants)

    explanation_en = _build_explanation_en(dominant_source, percentages, weather, aqi, ward_name)
    explanation_hi = _build_explanation_hi(dominant_source, percentages, weather, aqi, ward_name)

    return {
        "ward":             ward_name,
        "aqi":              aqi,
        "dominant_source":  dominant_source,
        "confidence":       confidence,
        "source_breakdown": percentages,
        "weather_context": {
            "wind_speed_ms": wind_speed,
            "humidity_pct":  humidity,
            "dispersion":    _dispersion_label(wind_speed),
        },
        "explanation": {
            "en": explanation_en,
            "hi": explanation_hi,
        },
    }


# ─── Individual Source Scorers ────────────────────────────────────────────────

def _traffic_score(hour: int, pollutants: dict, wind_speed: float) -> float:
    score = 0.3  # base
    # Rush hour boost
    if 7 <= hour <= 10 or 17 <= hour <= 21:
        score += 0.4
    # NO2 is a traffic indicator
    no2 = pollutants.get("NO2", {})
    if isinstance(no2, dict):
        no2_avg = no2.get("avg", 0) or 0
    else:
        no2_avg = no2 or 0
    if no2_avg > 80:
        score += 0.2
    elif no2_avg > 40:
        score += 0.1
    # Low wind means traffic fumes don't disperse
    if wind_speed < 2:
        score += 0.1
    return min(score, 1.0)


def _dust_score(pollutants: dict, humidity: float, wind_speed: float, temperature: float) -> float:
    score = 0.15  # base
    pm10 = _get_avg(pollutants, "PM10")
    pm25 = _get_avg(pollutants, "PM2.5")

    # High PM10/PM2.5 ratio = coarse particles = dust
    if pm10 and pm25 and pm25 > 0:
        ratio = pm10 / pm25
        if ratio > 2.5:
            score += 0.35
        elif ratio > 1.8:
            score += 0.2

    # Dry conditions favour dust
    if humidity < 40:
        score += 0.15
    elif humidity < 60:
        score += 0.05

    # Medium wind picks up dust but doesn't yet disperse
    if 2 <= wind_speed <= 5:
        score += 0.1

    # High temp → dry ground → more dust
    if temperature > 35:
        score += 0.1

    return min(score, 1.0)


def _burning_score(hour: int, pollutants: dict, weather_main: str) -> float:
    score = 0.1  # base
    # Evening/night burning peaks
    if 18 <= hour <= 23 or 0 <= hour <= 5:
        score += 0.25

    pm25 = _get_avg(pollutants, "PM2.5")
    co   = _get_avg(pollutants, "CO")

    # Biomass burning → very high PM2.5 + CO together
    if pm25 and pm25 > 150:
        score += 0.25
    if co and co > 2:
        score += 0.2

    # Foggy/hazy weather traps smoke
    if weather_main in ("Smoke", "Haze", "Mist"):
        score += 0.15

    return min(score, 1.0)


def _industrial_score(pollutants: dict, hour: int) -> float:
    score = 0.1  # base
    so2 = _get_avg(pollutants, "SO2")
    no2 = _get_avg(pollutants, "NO2")

    if so2 and so2 > 40:
        score += 0.3
    elif so2 and so2 > 20:
        score += 0.15

    # Working hours
    if 8 <= hour <= 18:
        score += 0.1

    # High NO2 but not rush hour → industrial
    if no2 and no2 > 100 and not (7 <= hour <= 10 or 17 <= hour <= 21):
        score += 0.2

    return min(score, 1.0)


def _trapping_score(wind_speed: float, humidity: float) -> float:
    """
    Weather trapping amplifies all sources.
    Not a source itself but shows when weather is making things worse.
    """
    score = 0.0
    if wind_speed < 1.5:
        score += 0.4   # near-calm = severe trapping
    elif wind_speed < 3.0:
        score += 0.2
    if humidity > 80:
        score += 0.15  # humid air traps particles
    return min(score, 0.6)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_avg(pollutants: dict, key: str) -> Optional[float]:
    val = pollutants.get(key) or pollutants.get(key.lower())
    if isinstance(val, dict):
        return val.get("avg")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_hour(timestamp: Optional[str]) -> int:
    if timestamp:
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "")).hour
        except Exception:
            pass
    return datetime.now().hour


def _dispersion_label(wind_speed: float) -> str:
    if wind_speed < 1.5:
        return "Very Poor - pollutants trapped"
    elif wind_speed < 3.0:
        return "Poor - slow dispersion"
    elif wind_speed < 6.0:
        return "Moderate - some dispersion"
    else:
        return "Good - pollutants dispersing"


def _confidence_score(scores: dict, aqi: float, pollutants: dict) -> str:
    """Return High/Medium/Low confidence based on data completeness."""
    has_key_pollutants = any(_get_avg(pollutants, k) for k in ["PM2.5", "PM10", "NO2"])
    max_score = max(scores.values()) if scores else 0
    
    # Lowered thresholds for better qualitative labels
    if has_key_pollutants and max_score > 0.45:
        return "High"
    elif has_key_pollutants and max_score > 0.25:
        return "Medium"
    else:
        return "Low"

def _get_ward_type(ward_name: str) -> str:
    # This is a bit circular, but we can import from ward_mapper if needed
    # Better: the caller passes it. For now, we use a simple dict or local import
    try:
        from app.services.ward_mapper import WARD_PROFILES
        return WARD_PROFILES.get(ward_name, "general")
    except ImportError:
        return "general"


# ─── Explanation Builders ─────────────────────────────────────────────────────

SOURCE_LABELS_EN = {
    "traffic":          "vehicle traffic",
    "dust_construction": "dust / construction activity",
    "biomass_burning":  "biomass / waste burning",
    "industrial":       "industrial emissions",
    "weather_trapping": "weather-trapped pollution",
}

SOURCE_LABELS_HI = {
    "traffic":          "वाहन यातायात",
    "dust_construction": "धूल / निर्माण कार्य",
    "biomass_burning":  "कचरा / फसल जलाना",
    "industrial":       "औद्योगिक उत्सर्जन",
    "weather_trapping": "मौसम जनित प्रदूषण संचय",
}


def _build_explanation_en(dominant: str, pct: dict, weather: dict, aqi: float, ward: str) -> str:
    label = SOURCE_LABELS_EN.get(dominant, dominant)
    wind  = weather.get("wind_speed", 3.0)
    disp  = _dispersion_label(wind)
    top2  = sorted(pct.items(), key=lambda x: x[1], reverse=True)[:2]
    causes = " and ".join(f"{SOURCE_LABELS_EN.get(k, k)} ({v}%)" for k, v in top2)

    return (
        f"AQI in {ward} is {aqi} ({_aqi_word(aqi)}). "
        f"Primary cause: {label} ({pct.get(dominant, 0)}%). "
        f"Main contributors: {causes}. "
        f"Wind conditions: {disp} ({wind} m/s). "
        f"{'Calm winds are preventing pollutants from dispersing.' if wind < 2 else 'Some natural dispersion is occurring.'}"
    )


def _build_explanation_hi(dominant: str, pct: dict, weather: dict, aqi: float, ward: str) -> str:
    label = SOURCE_LABELS_HI.get(dominant, dominant)
    wind  = weather.get("wind_speed", 3.0)
    top2  = sorted(pct.items(), key=lambda x: x[1], reverse=True)[:2]
    causes = " और ".join(f"{SOURCE_LABELS_HI.get(k, k)} ({v}%)" for k, v in top2)

    return (
        f"{ward} में AQI {aqi} ({_aqi_word_hi(aqi)}) है। "
        f"मुख्य कारण: {label} ({pct.get(dominant, 0)}%)। "
        f"प्रमुख स्रोत: {causes}। "
        f"हवा की गति: {wind} m/s — "
        f"{'हवा धीमी है, प्रदूषण फंसा हुआ है।' if wind < 2 else 'हवा से कुछ प्रदूषण फैल रहा है।'}"
    )


def _aqi_word(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"


def _aqi_word_hi(aqi: float) -> str:
    if aqi <= 50:   return "अच्छा"
    if aqi <= 100:  return "संतोषजनक"
    if aqi <= 200:  return "मध्यम"
    if aqi <= 300:  return "खराब"
    if aqi <= 400:  return "बहुत खराब"
    return "गंभीर"


def _clean_air_response(aqi, ward_name):
    return {
        "ward":            ward_name,
        "aqi":             aqi,
        "dominant_source": "none",
        "confidence":      "High",
        "source_breakdown": {},
        "explanation": {
            "en": f"Air quality in {ward_name} is currently Good (AQI: {aqi}). No major pollution sources detected.",
            "hi": f"{ward_name} में वायु गुणवत्ता अभी अच्छी है (AQI: {aqi})। कोई प्रमुख प्रदूषण स्रोत नहीं पाया गया।",
        },
    }