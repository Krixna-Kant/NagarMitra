"""
NagarMitra - Phase 1
AQI API Routes
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging

from app.services.aqi_fetcher      import (fetch_cpcb_delhi, fetch_aqicn_all_delhi_stations,
                                             fetch_weather_delhi, get_cpcb_station_aqi)
from app.services.ward_mapper       import get_ward_aqi, get_all_wards_aqi, WARD_CENTROIDS
from app.services.attribution_engine import attribute_pollution_sources
from app.services.advisory_engine  import get_health_advisory, get_bulk_advisories

router  = APIRouter(prefix="/api/v1/aqi", tags=["AQI"])
logger  = logging.getLogger(__name__)


# ─── 1. Live All-Delhi Stations ───────────────────────────────────────────────

@router.get("/stations/live")
async def get_live_stations():
    """
    Get live AQI from all Delhi monitoring stations.
    Uses AQICN (fastest) with CPCB as enrichment.
    """
    stations = await fetch_aqicn_all_delhi_stations()
    if not stations:
        raise HTTPException(503, "Unable to fetch live station data. Check API token.")
    return {
        "total_stations": len(stations),
        "source":         "AQICN",
        "stations":       stations,
    }


# ─── 2. Single Station Detail ─────────────────────────────────────────────────

@router.get("/station/{station_name}")
async def get_station_detail(station_name: str):
    """
    Get detailed pollutant breakdown for a specific station (CPCB data).
    Example: /api/v1/aqi/station/Anand Vihar, Delhi - DPCC
    """
    data = await get_cpcb_station_aqi(station_name)
    if not data:
        raise HTTPException(404, f"Station '{station_name}' not found or no data available.")
    return data


# ─── 3. Ward AQI ──────────────────────────────────────────────────────────────

@router.get("/ward/{ward_name}")
async def get_ward_aqi_endpoint(
    ward_name: str,
    lang: str = Query("both", description="Language: en | hi | both"),
    profile: str = Query("general", description="Profile: general | sensitive | children | elderly"),
):
    """
    Get AQI + attribution + health advisory for a specific Delhi ward.
    This is the main citizen-facing endpoint.
    """
    # Fetch live data
    stations  = await fetch_aqicn_all_delhi_stations()
    weather   = await fetch_weather_delhi()

    if not stations:
        raise HTTPException(503, "Live station data unavailable.")

    # Map to ward
    ward_data = get_ward_aqi(ward_name, stations)
    if ward_data.get("error"):
        available = list(WARD_CENTROIDS.keys())
        raise HTTPException(404, {
            "error":            f"Ward '{ward_name}' not found.",
            "available_wards":  available[:20],
            "hint":             "Check /api/v1/aqi/wards for full list",
        })

    # Attribution (why is AQI high?)
    # Get pollutants from nearest station for better attribution
    nearest_station_data = None
    for s in stations:
        if s.get("station") == ward_data.get("nearest_station"):
            nearest_station_data = s
            break
    pollutants = nearest_station_data.get("pollutants", {}) if nearest_station_data else {}

    attribution = attribute_pollution_sources(
        aqi       = ward_data.get("aqi"),
        pollutants= pollutants,
        weather   = weather,
        ward_name = ward_name,
    )

    # Health advisory
    advisory = get_health_advisory(
        aqi       = ward_data.get("aqi"),
        ward_name = ward_name,
        profile   = profile,
        lang      = lang,
    )

    return {
        "ward":        ward_name,
        "aqi_data":    ward_data,
        "weather":     weather,
        "attribution": attribution,
        "advisory":    advisory,
    }


# ─── 4. All Wards Overview ────────────────────────────────────────────────────

@router.get("/wards/all")
async def get_all_wards(
    lang: str = Query("both", description="Language: en | hi | both"),
):
    """
    Get AQI estimates for all mapped Delhi wards (sorted worst-first).
    Use this for the ward heatmap visualization.
    """
    stations = await fetch_aqicn_all_delhi_stations()
    if not stations:
        raise HTTPException(503, "Live data unavailable.")

    all_wards    = get_all_wards_aqi(stations)
    advisories   = get_bulk_advisories(all_wards, lang=lang)

    # Merge advisory into ward data
    advisory_map = {a.get("ward") or a.get("en", {}).get("ward"): a for a in advisories}
    for ward in all_wards:
        ward["advisory"] = advisory_map.get(ward["ward"])

    return {
        "total_wards":  len(all_wards),
        "source":       "IDW interpolation from AQICN stations",
        "worst_ward":   all_wards[0] if all_wards else None,
        "best_ward":    all_wards[-1] if all_wards else None,
        "wards":        all_wards,
    }


# ─── 5. List Available Wards ─────────────────────────────────────────────────

@router.get("/wards")
async def list_wards():
    """List all Delhi wards supported by NagarMitra."""
    return {
        "total": len(WARD_CENTROIDS),
        "wards": list(WARD_CENTROIDS.keys()),
    }


# ─── 6. Full Dashboard Snapshot ──────────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard_snapshot():
    """
    Full dashboard data: live stations + ward overview + weather.
    Used by admin dashboard on load.
    """
    stations = await fetch_aqicn_all_delhi_stations()
    weather  = await fetch_weather_delhi()
    wards    = get_all_wards_aqi(stations) if stations else []

    # Stats
    aqi_values = [w["aqi"] for w in wards if w.get("aqi")]
    avg_aqi    = round(sum(aqi_values) / len(aqi_values), 1) if aqi_values else None

    from app.services.aqi_fetcher import _aqi_category
    avg_cat, avg_color, avg_health = _aqi_category(avg_aqi)

    severe_wards   = [w for w in wards if (w.get("aqi") or 0) > 300]
    moderate_wards = [w for w in wards if 100 < (w.get("aqi") or 0) <= 200]

    return {
        "summary": {
            "delhi_avg_aqi":        avg_aqi,
            "delhi_avg_category":   avg_cat,
            "delhi_avg_color":      avg_color,
            "total_stations_live":  len(stations),
            "wards_severe":         len(severe_wards),
            "wards_moderate":       len(moderate_wards),
        },
        "weather":          weather,
        "live_stations":    stations[:10],    # top 10 for dashboard preview
        "worst_5_wards":    wards[:5],
        "best_5_wards":     wards[-5:],
        "all_wards":        wards,
    }