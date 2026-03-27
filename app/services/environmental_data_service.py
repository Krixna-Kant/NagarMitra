"""
NagarMitra - Environmental Data Service
Collects AQI, dust proxy, wind, humidity, traffic, and construction
signals around route points with short-lived caching.
"""

import asyncio
import logging
import math
import os
import time
from datetime import datetime
from typing import Any, Optional

import httpx

from app.services.aqi_fetcher import fetch_weather_delhi


logger = logging.getLogger(__name__)

AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN", "")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# Very lightweight in-memory cache for hackathon MVP
_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL_SECONDS = {
    "aqi": 300,  # 5 mins
    "wind": 600,  # 10 mins
    "construction": 900,  # 15 mins
    "traffic": 900,  # 15 mins
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _cache_key(prefix: str, lat: float, lon: float, radius_km: float = 0.0) -> str:
    # Round enough to group nearby requests in cache (~1km at 2 decimals)
    lat_bucket = round(lat, 2)
    lon_bucket = round(lon, 2)
    radius_bucket = round(radius_km, 1)
    return f"{prefix}:{lat_bucket}:{lon_bucket}:{radius_bucket}"


def _cache_get(key: str) -> Optional[dict]:
    entry = _CACHE.get(key)
    if not entry:
        return None
    if entry["expires_at"] < time.time():
        _CACHE.pop(key, None)
        return None
    return entry["value"]


def _cache_set(key: str, value: dict, ttl_seconds: int) -> None:
    _CACHE[key] = {
        "value": value,
        "expires_at": time.time() + ttl_seconds,
    }


def _bbox_for_radius(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """
    Return (min_lat, min_lon, max_lat, max_lon) around a point.
    """
    delta_lat = radius_km / 111.0
    lon_divisor = 111.0 * max(math.cos(math.radians(lat)), 0.1)
    delta_lon = radius_km / lon_divisor
    return (
        lat - delta_lat,
        lon - delta_lon,
        lat + delta_lat,
        lon + delta_lon,
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r_earth_km * 2 * math.asin(math.sqrt(a))


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _traffic_time_multiplier() -> float:
    """
    Simple time-of-day/week multiplier for traffic congestion proxy.
    """
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Mon
    is_weekend = weekday >= 5

    if not is_weekend and ((8 <= hour <= 11) or (17 <= hour <= 21)):
        return 1.35  # weekday rush
    if not is_weekend and (12 <= hour <= 16):
        return 1.08  # moderate daytime traffic
    if is_weekend and (11 <= hour <= 22):
        return 1.1
    if 0 <= hour <= 5:
        return 0.62  # late night low traffic
    return 0.9


def _traffic_level(congestion_index: float) -> str:
    if congestion_index >= 0.68:
        return "high"
    if congestion_index >= 0.4:
        return "moderate"
    return "low"


async def _fetch_aqi_in_radius(lat: float, lon: float, radius_km: float) -> dict:
    cache_key = _cache_key("aqi", lat, lon, radius_km)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    fallback = {
        "value": 120.0,
        "source": "fallback",
        "station_count": 0,
        "confidence": 0.2,
    }

    if not AQICN_API_TOKEN:
        _cache_set(cache_key, fallback, _CACHE_TTL_SECONDS["aqi"])
        return fallback

    min_lat, min_lon, max_lat, max_lon = _bbox_for_radius(lat, lon, radius_km)
    url = (
        "https://api.waqi.info/map/bounds/"
        f"?latlng={min_lat},{min_lon},{max_lat},{max_lon}"
        f"&token={AQICN_API_TOKEN}"
    )

    try:
        async with httpx.AsyncClient(timeout=12) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.error(f"[Env:AQI] bounds fetch failed: {exc}")
        _cache_set(cache_key, fallback, _CACHE_TTL_SECONDS["aqi"])
        return fallback

    if payload.get("status") != "ok":
        _cache_set(cache_key, fallback, _CACHE_TTL_SECONDS["aqi"])
        return fallback

    values = []
    for station in payload.get("data", []):
        aqi_raw = station.get("aqi")
        if aqi_raw in (None, "-"):
            continue
        aqi_val = _safe_float(aqi_raw)
        s_lat = _safe_float(station.get("lat"))
        s_lon = _safe_float(station.get("lon"))
        if aqi_val is None or s_lat is None or s_lon is None:
            continue

        # Weight nearer stations more strongly.
        dist = _haversine_km(lat, lon, s_lat, s_lon)
        if dist > max(radius_km * 1.35, radius_km + 0.8):
            continue
        weight = 1.0 / max(dist, 0.3)
        values.append((aqi_val, weight))

    if not values:
        _cache_set(cache_key, fallback, _CACHE_TTL_SECONDS["aqi"])
        return fallback

    weighted_sum = sum(v * w for v, w in values)
    total_weight = sum(w for _, w in values)
    avg_aqi = round(weighted_sum / total_weight, 1) if total_weight else 120.0
    confidence = _clamp(0.45 + (0.08 * len(values)), 0.4, 0.95)

    result = {
        "value": avg_aqi,
        "source": "waqi_bounds",
        "station_count": len(values),
        "confidence": round(confidence, 2),
    }
    _cache_set(cache_key, result, _CACHE_TTL_SECONDS["aqi"])
    return result


async def _fetch_wind(lat: float, lon: float) -> dict:
    cache_key = _cache_key("wind", lat, lon)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "wind_speed_10m,wind_direction_10m,relative_humidity_2m",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()

        current = payload.get("current", {})
        speed = _safe_float(current.get("wind_speed_10m"))
        direction = _safe_float(current.get("wind_direction_10m"))
        humidity = _safe_float(current.get("relative_humidity_2m"))
        if speed is None:
            raise ValueError("missing wind_speed_10m")

        result = {
            "speed_ms": round(speed / 3.6, 2),  # open-meteo uses km/h
            "direction_deg": direction if direction is not None else 0.0,
            "humidity_pct": humidity if humidity is not None else 50.0,
            "source": "open_meteo",
            "confidence": 0.9,
        }
        _cache_set(cache_key, result, _CACHE_TTL_SECONDS["wind"])
        return result

    except Exception as exc:
        logger.warning(f"[Env:Wind] open-meteo failed, fallback openweather: {exc}")
        weather = await fetch_weather_delhi(lat=lat, lon=lon)
        fallback = {
            "speed_ms": float(weather.get("wind_speed", 3.0)),
            "direction_deg": float(weather.get("wind_direction", 0.0)),
            "humidity_pct": float(weather.get("humidity", 50.0)),
            "source": "openweather_fallback",
            "confidence": 0.65,
        }
        _cache_set(cache_key, fallback, _CACHE_TTL_SECONDS["wind"])
        return fallback


async def _fetch_traffic_density(lat: float, lon: float, radius_km: float) -> dict:
    """
    Traffic metric for route segment.
    Prefers TomTom live flow if API key is provided, else falls back to
    free OSM proxy.
    """
    cache_key = _cache_key("traffic", lat, lon, radius_km)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    tomtom_forbidden = False

    if TOMTOM_API_KEY:
        # Approximate area-wise traffic by sampling center + 4 nearby points.
        delta_lat = (radius_km * 0.45) / 111.0
        lon_divisor = 111.0 * max(math.cos(math.radians(lat)), 0.1)
        delta_lon = (radius_km * 0.45) / lon_divisor
        sample_points = [
            (lat, lon),
            (lat + delta_lat, lon),
            (lat - delta_lat, lon),
            (lat, lon + delta_lon),
            (lat, lon - delta_lon),
        ]

        async def _tomtom_fetch(sample_lat: float, sample_lon: float) -> Optional[dict]:
            params = {
                "key": TOMTOM_API_KEY,
                "point": f"{sample_lat},{sample_lon}",
                "unit": "KMPH",
            }
            try:
                async with httpx.AsyncClient(timeout=8) as client:
                    resp = await client.get(TOMTOM_FLOW_URL, params=params)
                    if resp.status_code == 403:
                        return {"_tomtom_forbidden": True}
                    resp.raise_for_status()
                    payload = resp.json().get("flowSegmentData", {})
                current_speed = _safe_float(payload.get("currentSpeed"))
                free_flow_speed = _safe_float(payload.get("freeFlowSpeed"))
                conf = _safe_float(payload.get("confidence")) or 0.6
                if current_speed is None or free_flow_speed is None or free_flow_speed <= 0:
                    return None
                congestion_idx = _clamp(1.0 - (current_speed / free_flow_speed), 0.0, 1.0)
                road_closure = bool(payload.get("roadClosure"))
                if road_closure:
                    congestion_idx = max(congestion_idx, 0.95)
                return {
                    "congestion_index": congestion_idx,
                    "current_speed_kmph": current_speed,
                    "free_flow_speed_kmph": free_flow_speed,
                    "speed_ratio": _clamp(current_speed / free_flow_speed, 0.0, 1.0),
                    "road_closure": road_closure,
                    "confidence": _clamp(conf, 0.2, 1.0),
                }
            except Exception:
                return None

        tomtom_results = await asyncio.gather(*[_tomtom_fetch(s_lat, s_lon) for s_lat, s_lon in sample_points])
        if any(item and item.get("_tomtom_forbidden") for item in tomtom_results):
            tomtom_forbidden = True

        tomtom_results = [
            item
            for item in tomtom_results
            if item is not None and not item.get("_tomtom_forbidden")
        ]
        if tomtom_results:
            avg_congestion = sum(item["congestion_index"] for item in tomtom_results) / len(tomtom_results)
            avg_current_speed = sum(item["current_speed_kmph"] for item in tomtom_results) / len(tomtom_results)
            avg_free_flow_speed = sum(item["free_flow_speed_kmph"] for item in tomtom_results) / len(tomtom_results)
            avg_speed_ratio = sum(item["speed_ratio"] for item in tomtom_results) / len(tomtom_results)
            any_road_closure = any(item["road_closure"] for item in tomtom_results)
            avg_confidence = sum(item["confidence"] for item in tomtom_results) / len(tomtom_results)
            density_equivalent = round(_clamp(avg_congestion * 35.0, 0.1, 35.0), 3)

            result = {
                "markers": len(tomtom_results),
                "density_per_sq_km": density_equivalent,
                "congestion_index": round(avg_congestion, 3),
                "congestion_level": _traffic_level(avg_congestion),
                "current_speed_kmph": round(avg_current_speed, 1),
                "free_flow_speed_kmph": round(avg_free_flow_speed, 1),
                "speed_ratio": round(avg_speed_ratio, 3),
                "road_closure": any_road_closure,
                "source": "tomtom_flow",
                "confidence": round(_clamp(avg_confidence, 0.45, 0.95), 2),
            }
            _cache_set(cache_key, result, _CACHE_TTL_SECONDS["traffic"])
            return result

    radius_m = int(radius_km * 1000)
    query = f"""
[out:json][timeout:12];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary"](around:{radius_m},{lat},{lon});
  node["highway"="traffic_signals"](around:{radius_m},{lat},{lon});
  node["junction"](around:{radius_m},{lat},{lon});
);
out ids;
"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS_URL, data=query.encode("utf-8"))
            resp.raise_for_status()
            payload = resp.json()
            elements = payload.get("elements", [])
    except Exception as exc:
        logger.warning(f"[Env:Traffic] overpass failed: {exc}")
        fallback = {
            "markers": 0,
            "density_per_sq_km": 0.0,
            "congestion_index": 0.12,
            "congestion_level": "low",
            "current_speed_kmph": None,
            "free_flow_speed_kmph": None,
            "speed_ratio": None,
            "road_closure": False,
            "source": "traffic_fallback_tomtom_forbidden" if tomtom_forbidden else "traffic_fallback",
            "confidence": 0.25,
        }
        _cache_set(cache_key, fallback, _CACHE_TTL_SECONDS["traffic"])
        return fallback

    area_sq_km = math.pi * (radius_km ** 2)
    marker_count = len(elements)
    density = round(marker_count / max(area_sq_km, 0.01), 3)
    confidence = _clamp(0.45 + (0.02 * min(marker_count, 12)), 0.45, 0.86)
    raw_congestion = _clamp(density / 18.0, 0.0, 1.0)
    timed_congestion = _clamp(raw_congestion * _traffic_time_multiplier(), 0.0, 1.0)
    result = {
        "markers": marker_count,
        "density_per_sq_km": density,
        "congestion_index": round(timed_congestion, 3),
        "congestion_level": _traffic_level(timed_congestion),
        "speed_ratio": round(1.0 - timed_congestion, 3),
        "road_closure": False,
        "source": "overpass_osm_tomtom_forbidden" if tomtom_forbidden else "overpass_osm",
        "confidence": round(confidence, 2),
    }
    _cache_set(cache_key, result, _CACHE_TTL_SECONDS["traffic"])
    return result


async def _fetch_construction_density(lat: float, lon: float, radius_km: float) -> dict:
    cache_key = _cache_key("construction", lat, lon, radius_km)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    radius_m = int(radius_km * 1000)
    query = f"""
[out:json][timeout:12];
(
  node["building"="construction"](around:{radius_m},{lat},{lon});
  way["building"="construction"](around:{radius_m},{lat},{lon});
  node["landuse"="construction"](around:{radius_m},{lat},{lon});
  way["landuse"="construction"](around:{radius_m},{lat},{lon});
  node["highway"="construction"](around:{radius_m},{lat},{lon});
  way["highway"="construction"](around:{radius_m},{lat},{lon});
);
out ids;
"""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(OVERPASS_URL, data=query.encode("utf-8"))
            resp.raise_for_status()
            payload = resp.json()
            elements = payload.get("elements", [])
    except Exception as exc:
        logger.warning(f"[Env:Construction] overpass failed: {exc}")
        fallback = {
            "sites": 0,
            "density_per_sq_km": 0.0,
            "source": "construction_fallback",
            "confidence": 0.25,
        }
        _cache_set(cache_key, fallback, _CACHE_TTL_SECONDS["construction"])
        return fallback

    area_sq_km = math.pi * (radius_km ** 2)
    count = len(elements)
    density = round(count / max(area_sq_km, 0.01), 3)
    confidence = _clamp(0.45 + (0.03 * min(count, 8)), 0.45, 0.85)

    result = {
        "sites": count,
        "density_per_sq_km": density,
        "source": "overpass_osm",
        "confidence": round(confidence, 2),
    }
    _cache_set(cache_key, result, _CACHE_TTL_SECONDS["construction"])
    return result


def _derive_dust_proxy(
    wind_speed_ms: float,
    humidity_pct: float,
    construction_density: float,
    traffic_density: float,
) -> dict:
    """
    Dust proxy at point (0-1):
    - rises with construction and traffic churn
    - rises with dry air
    - rises with moderate wind that resuspends road dust
    """
    wind_resuspension = 0.0
    if 1.5 <= wind_speed_ms <= 5.5:
        wind_resuspension = 1.0
    elif 5.5 < wind_speed_ms <= 7.0:
        wind_resuspension = 0.7
    elif wind_speed_ms < 1.5:
        wind_resuspension = 0.25

    dryness = _clamp((65.0 - humidity_pct) / 65.0, 0.0, 1.0)
    construction_norm = _clamp(construction_density / 3.2, 0.0, 1.0)
    traffic_norm = _clamp(traffic_density / 30.0, 0.0, 1.0)

    dust_index = _clamp(
        (0.42 * construction_norm)
        + (0.23 * traffic_norm)
        + (0.2 * dryness)
        + (0.15 * wind_resuspension),
        0.0,
        1.0,
    )
    return {
        "index": round(dust_index, 4),
        "dryness_component": round(dryness, 4),
        "wind_resuspension_component": round(wind_resuspension, 4),
        "construction_component": round(construction_norm, 4),
        "traffic_component": round(traffic_norm, 4),
        "source": "derived_proxy",
    }


async def get_environmental_snapshot(lat: float, lon: float, radius_km: float = 2.5) -> dict:
    """
    Fetch AQI, dust proxy, wind, humidity, traffic and construction around a route point.
    """
    radius_km = _clamp(radius_km, 2.0, 3.0)

    aqi_task = _fetch_aqi_in_radius(lat, lon, radius_km)
    wind_task = _fetch_wind(lat, lon)
    traffic_task = _fetch_traffic_density(lat, lon, radius_km)
    construction_task = _fetch_construction_density(lat, lon, radius_km)

    aqi, wind, traffic, construction = await asyncio.gather(
        aqi_task, wind_task, traffic_task, construction_task
    )
    humidity = wind.get("humidity_pct", 50.0)
    dust = _derive_dust_proxy(
        wind_speed_ms=wind.get("speed_ms", 3.0),
        humidity_pct=humidity,
        construction_density=construction.get("density_per_sq_km", 0.0),
        traffic_density=traffic.get("density_per_sq_km", 0.0),
    )

    confidences = [
        aqi.get("confidence", 0.0),
        wind.get("confidence", 0.0),
        traffic.get("confidence", 0.0),
        construction.get("confidence", 0.0),
    ]
    dust_confidence = round(
        (
            wind.get("confidence", 0.0)
            + traffic.get("confidence", 0.0)
            + construction.get("confidence", 0.0)
        )
        / 3.0,
        2,
    )
    overall_conf = round(sum(confidences) / len(confidences), 2)

    return {
        "point": {"latitude": lat, "longitude": lon, "radius_km": radius_km},
        "aqi": aqi,
        "wind": wind,
        "humidity": {
            "value_pct": round(float(humidity), 1),
            "source": wind.get("source", "fallback"),
            "confidence": wind.get("confidence", 0.0),
        },
        "traffic": traffic,
        "construction": construction,
        "dust": {
            **dust,
            "confidence": dust_confidence,
        },
        "confidence": overall_conf,
    }
