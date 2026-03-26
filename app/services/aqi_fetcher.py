"""
NagarMitra - Phase 1
AQI Data Fetcher Service
Fetches live AQI data from CPCB (data.gov.in) and AQICN APIs
"""

import httpx
import json
import os
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_GOV_API_KEY  = os.getenv("DATA_GOV_API_KEY", "")
AQICN_API_TOKEN   = os.getenv("AQICN_API_TOKEN", "")
OPENWEATHER_KEY   = os.getenv("OPENWEATHER_API_KEY", "")

# CPCB dataset resource ID on data.gov.in
CPCB_RESOURCE_ID  = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"


# ─── 1. CPCB via data.gov.in ─────────────────────────────────────────────────

async def fetch_cpcb_delhi(state: str = "Delhi") -> list[dict]:
    """
    Fetch all AQI pollutant readings for Delhi stations from CPCB.
    Returns list of records: {station, lat, lon, pollutant_id, pollutant_avg, ...}
    """
    url = (
        f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"
        f"?api-key={DATA_GOV_API_KEY}"
        f"&format=json"
        f"&limit=500"
        f"&filters[state]={state}"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("records", [])
            logger.info(f"[CPCB] Fetched {len(records)} records for {state}")
            return records
    except Exception as e:
        logger.error(f"[CPCB] Fetch failed: {e}")
        return []


async def get_cpcb_station_aqi(station_name: str) -> Optional[dict]:
    """
    Get all pollutants for a single station and compute composite AQI.
    """
    url = (
        f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"
        f"?api-key={DATA_GOV_API_KEY}"
        f"&format=json"
        f"&limit=50"
        f"&filters[station]={station_name}"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            records = resp.json().get("records", [])
            return _aggregate_station_pollutants(station_name, records)
    except Exception as e:
        logger.error(f"[CPCB] Station fetch failed for {station_name}: {e}")
        return None


def _aggregate_station_pollutants(station_name: str, records: list[dict]) -> dict:
    """
    Aggregate pollutant records into a single station AQI object.
    Calculates composite AQI using CPCB standard (max sub-index method).
    """
    pollutants = {}
    lat, lon = None, None

    for r in records:
        pid  = r.get("pollutant_id", "").upper()
        pavg = _safe_float(r.get("pollutant_avg"))
        pmax = _safe_float(r.get("pollutant_max"))
        pmin = _safe_float(r.get("pollutant_min"))
        if pid and pavg is not None:
            pollutants[pid] = {"avg": pavg, "max": pmax, "min": pmin}
        if not lat and r.get("latitude"):
            lat = _safe_float(r.get("latitude"))
            lon = _safe_float(r.get("longitude"))

    # Compute individual sub-index AQIs per CPCB breakpoints
    sub_indices = {}
    for pid, vals in pollutants.items():
        si = _compute_sub_index(pid, vals["avg"])
        if si is not None:
            sub_indices[pid] = round(si, 1)

    composite_aqi = max(sub_indices.values()) if sub_indices else None
    category, color, health_impact = _aqi_category(composite_aqi)

    return {
        "station":       station_name,
        "latitude":      lat,
        "longitude":     lon,
        "timestamp":     datetime.utcnow().isoformat() + "Z",
        "aqi":           composite_aqi,
        "category":      category,
        "color":         color,
        "health_impact": health_impact,
        "pollutants":    pollutants,
        "sub_indices":   sub_indices,
    }


# ─── 2. AQICN API ─────────────────────────────────────────────────────────────

async def fetch_aqicn_station(station_keyword: str) -> Optional[dict]:
    """
    Fetch live AQI from AQICN for a given keyword (city/station name).
    Used as fallback or for enrichment.
    """
    url = f"https://api.waqi.info/feed/{station_keyword}/?token={AQICN_API_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                logger.warning(f"[AQICN] Bad status for {station_keyword}")
                return None
            d = data["data"]
            return {
                "station":    d.get("city", {}).get("name", station_keyword),
                "latitude":   d.get("city", {}).get("geo", [None, None])[0],
                "longitude":  d.get("city", {}).get("geo", [None, None])[1],
                "timestamp":  d.get("time", {}).get("iso"),
                "aqi":        d.get("aqi"),
                "dominant_pollutant": d.get("dominentpol"),
                "pollutants": {k: v.get("v") for k, v in d.get("iaqi", {}).items()},
                "source":     "aqicn",
            }
    except Exception as e:
        logger.error(f"[AQICN] Fetch failed for {station_keyword}: {e}")
        return None


async def fetch_aqicn_all_delhi_stations() -> list[dict]:
    """
    Fetch AQI for all known Delhi stations from AQICN using lat/lon search.
    Covers the bounding box of Delhi.
    """
    # Delhi bounding box: lat 28.40–28.88, lon 76.84–77.35
    url = (
        f"https://api.waqi.info/map/bounds/"
        f"?latlng=28.40,76.84,28.88,77.35"
        f"&token={AQICN_API_TOKEN}"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                return []
            stations = []
            for s in data.get("data", []):
                aqi_val = s.get("aqi")
                if aqi_val == "-":
                    continue
                category, color, health = _aqi_category(_safe_float(aqi_val))
                stations.append({
                    "station":   s.get("station", {}).get("name"),
                    "latitude":  s.get("lat"),
                    "longitude": s.get("lon"),
                    "aqi":       _safe_float(aqi_val),
                    "category":  category,
                    "color":     color,
                    "timestamp": s.get("station", {}).get("time"),
                    "source":    "aqicn",
                })
            logger.info(f"[AQICN] Fetched {len(stations)} Delhi stations")
            return stations
    except Exception as e:
        logger.error(f"[AQICN] Bounds fetch failed: {e}")
        return []


# ─── 3. Weather (for attribution engine) ─────────────────────────────────────

async def fetch_weather_delhi(lat: float = 28.6139, lon: float = 77.2090) -> dict:
    """
    Fetch current weather for Delhi (wind speed, direction, humidity, temp).
    Critical for pollution source attribution logic.
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            d = resp.json()
            return {
                "temperature":      d["main"]["temp"],
                "humidity":         d["main"]["humidity"],
                "wind_speed":       d["wind"]["speed"],        # m/s
                "wind_direction":   d["wind"].get("deg", 0),   # degrees
                "weather_main":     d["weather"][0]["main"],
                "weather_desc":     d["weather"][0]["description"],
                "visibility":       d.get("visibility", 10000),
                "timestamp":        datetime.utcnow().isoformat() + "Z",
            }
    except Exception as e:
        logger.error(f"[Weather] Fetch failed: {e}")
        # Return defaults so attribution engine still works
        return {
            "temperature": 25, "humidity": 50,
            "wind_speed": 3, "wind_direction": 180,
            "weather_main": "Clear", "weather_desc": "clear sky",
            "visibility": 10000,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


def _compute_sub_index(pollutant: str, concentration: float) -> Optional[float]:
    """
    Compute AQI sub-index for a pollutant using CPCB breakpoints.
    Linear interpolation between breakpoints.
    """
    # (concentration_low, concentration_high, aqi_low, aqi_high)
    BREAKPOINTS = {
        "PM2.5": [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)],
        "PM10":  [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
        "NO2":   [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
        "NH3":   [(0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2400,401,500)],
        "SO2":   [(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2100,401,500)],
        "CO":    [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
        "O3":    [(0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
    }
    bp_list = BREAKPOINTS.get(pollutant.upper())
    if not bp_list:
        return None
    for (cl, ch, il, ih) in bp_list:
        if cl <= concentration <= ch:
            return il + (concentration - cl) * (ih - il) / (ch - cl)
    return None


def _aqi_category(aqi: Optional[float]) -> tuple[str, str, str]:
    """Return (category, hex_color, health_impact) for a given AQI value."""
    if aqi is None:
        return "Unknown", "#808080", "Data unavailable"
    if aqi <= 50:
        return "Good",      "#00B050", "Minimal impact on health"
    elif aqi <= 100:
        return "Satisfactory","#92D050","Minor discomfort to sensitive people"
    elif aqi <= 200:
        return "Moderate",  "#FFFF00", "Breathing discomfort for lung/heart patients"
    elif aqi <= 300:
        return "Poor",      "#FF9900", "Breathing discomfort to most people"
    elif aqi <= 400:
        return "Very Poor", "#FF0000", "Respiratory illness on prolonged exposure"
    else:
        return "Severe",    "#800000", "Serious health impact, affects healthy people"