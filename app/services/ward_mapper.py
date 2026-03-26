"""
NagarMitra - Phase 1
Ward Mapper Service
Maps station-level AQI readings to Delhi wards using
Inverse Distance Weighting (IDW) spatial interpolation.
"""

import json
import math
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Load stations config
_STATIONS_PATH = os.path.join(os.path.dirname(__file__), "../../data/delhi_stations.json")

def _load_stations() -> list[dict]:
    with open(_STATIONS_PATH) as f:
        return json.load(f)["stations"]

STATIONS = _load_stations()


# ─── Ward → Nearest Stations ──────────────────────────────────────────────────

# Approximate lat/lon centroids for Delhi's major ward zones
# In production: load from delhi_wards.geojson (data.gov.in)
WARD_CENTROIDS = {
    "Alipur":              (28.8180, 77.1530),
    "Anand Vihar":         (28.6471, 77.3156),
    "Ashok Vihar":         (28.6950, 77.1770),
    "Aya Nagar":           (28.4726, 77.1005),
    "Bawana":              (28.7930, 77.0374),
    "Burari":              (28.7351, 77.2063),
    "Connaught Place":     (28.6315, 77.2167),
    "Daryaganj":           (28.6403, 77.2408),
    "Dilshad Garden":      (28.6798, 77.3162),
    "DTU Campus":          (28.7497, 77.1178),
    "Dwarka Sector 8":     (28.5921, 77.0460),
    "Dwarka Sector 23":    (28.5508, 77.0200),
    "GK I":                (28.5483, 77.2380),
    "Hauz Khas":           (28.5441, 77.1860),
    "ITO":                 (28.6289, 77.2412),
    "Jahangirpuri":        (28.7300, 77.1600),
    "Jasola":              (28.5381, 77.2870),
    "Karol Bagh":          (28.6519, 77.1907),
    "Lajpat Nagar":        (28.5672, 77.2436),
    "Mayur Vihar":         (28.6065, 77.2926),
    "Mehrauli":            (28.5244, 77.1854),
    "Mukherjee Nagar":     (28.7103, 77.2044),
    "Mundka":              (28.6808, 77.0167),
    "Munirka":             (28.5529, 77.1720),
    "Nangloi":             (28.6789, 77.0607),
    "Narela":              (28.8553, 77.0944),
    "Nehru Nagar":         (28.5700, 77.2500),
    "Nizamuddin":          (28.5920, 77.2524),
    "Okhla":               (28.5355, 77.2740),
    "Palam":               (28.5921, 77.0858),
    "Patparganj":          (28.6271, 77.2945),
    "Patel Nagar":         (28.6469, 77.1671),
    "Preet Vihar":         (28.6483, 77.2996),
    "Punjabi Bagh":        (28.6720, 77.1317),
    "R.K. Puram":          (28.5650, 77.1762),
    "Rajouri Garden":      (28.6476, 77.1217),
    "Rohini":              (28.7324, 77.0659),
    "Sarita Vihar":        (28.5250, 77.2920),
    "SDA":                 (28.5444, 77.1889),
    "Seemapuri":           (28.6804, 77.3156),
    "Shadipur":            (28.6490, 77.1515),
    "Shahdara":            (28.6679, 77.2893),
    "Shalimar Bagh":       (28.7116, 77.1648),
    "Sonia Vihar":         (28.7219, 77.2558),
    "South Delhi":         (28.5355, 77.2090),
    "Uttam Nagar":         (28.6217, 77.0554),
    "Vasant Kunj":         (28.5215, 77.1505),
    "Vivek Vihar":         (28.6710, 77.3158),
    "Wazirpur":            (28.7050, 77.1637),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def idw_interpolate(
    target_lat: float,
    target_lon: float,
    station_readings: list[dict],
    power: int = 2,
    max_distance_km: float = 25.0,
    top_k: int = 5,
) -> Optional[float]:
    """
    Inverse Distance Weighting interpolation.
    Returns estimated AQI at (target_lat, target_lon) from nearby station readings.
    """
    weighted_sum = 0.0
    weight_total = 0.0

    distances = []
    for s in station_readings:
        if s.get("aqi") is None or s.get("latitude") is None:
            continue
        dist = haversine_km(target_lat, target_lon, s["latitude"], s["longitude"])
        distances.append((dist, s))

    # Sort by distance, keep closest top_k within max range
    distances.sort(key=lambda x: x[0])
    nearby = [(d, s) for d, s in distances if d <= max_distance_km][:top_k]

    if not nearby:
        return None

    for dist, s in nearby:
        if dist < 0.01:   # exact match
            return s["aqi"]
        w = 1.0 / (dist ** power)
        weighted_sum += w * s["aqi"]
        weight_total += w

    return round(weighted_sum / weight_total, 1) if weight_total > 0 else None


def get_ward_aqi(ward_name: str, station_readings: list[dict]) -> dict:
    """
    Get estimated AQI for a specific Delhi ward.
    Uses IDW interpolation from station readings.
    """
    centroid = WARD_CENTROIDS.get(ward_name)
    if not centroid:
        return {"ward": ward_name, "aqi": None, "error": "Ward not found"}

    lat, lon = centroid
    estimated_aqi = idw_interpolate(lat, lon, station_readings)

    from app.services.aqi_fetcher import _aqi_category
    category, color, health_impact = _aqi_category(estimated_aqi)

    # Find nearest actual station
    nearest = _find_nearest_station(lat, lon, station_readings)

    return {
        "ward":           ward_name,
        "latitude":       lat,
        "longitude":      lon,
        "aqi":            estimated_aqi,
        "category":       category,
        "color":          color,
        "health_impact":  health_impact,
        "nearest_station": nearest.get("station") if nearest else None,
        "nearest_aqi":    nearest.get("aqi") if nearest else None,
        "method":         "IDW_interpolation",
    }


def get_all_wards_aqi(station_readings: list[dict]) -> list[dict]:
    """Get estimated AQI for ALL mapped Delhi wards."""
    results = []
    for ward_name in WARD_CENTROIDS:
        results.append(get_ward_aqi(ward_name, station_readings))
    return sorted(results, key=lambda x: x.get("aqi") or 0, reverse=True)


def _find_nearest_station(lat: float, lon: float, readings: list[dict]) -> Optional[dict]:
    valid = [s for s in readings if s.get("latitude") and s.get("aqi") is not None]
    if not valid:
        return None
    return min(valid, key=lambda s: haversine_km(lat, lon, s["latitude"], s["longitude"]))