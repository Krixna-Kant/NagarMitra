"""
NagarMitra - Route Planning Service
Fetches route alternatives from TomTom (preferred) / OSRM (fallback) and samples points
for environmental scoring along the route corridor.
"""

import logging
import math
import os

import httpx


logger = logging.getLogger(__name__)

OSRM_BASE_URL = "https://router.project-osrm.org/route/v1"
TOMTOM_ROUTING_BASE_URL = "https://api.tomtom.com/routing/1/calculateRoute"
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
PROFILE_MAP = {
    "drive": "driving",
    "driving": "driving",
    "walk": "walking",
    "walking": "walking",
    "bike": "cycling",
    "cycling": "cycling",
}
PROFILE_SPEED_KMPH = {
    "driving": None,  # use provider duration
    "walking": 4.8,
    "cycling": 15.0,
}
TOMTOM_MODE_MAP = {
    "driving": "car",
    "walking": "pedestrian",
    "cycling": "bicycle",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance in kilometers between two geo points."""
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


def _lerp(start: float, end: float, ratio: float) -> float:
    return start + (end - start) * ratio


def _sample_route_points(
    geometry: list[list[float]],
    sample_step_km: float = 0.7,
    max_points: int = 18,
) -> list[dict]:
    """
    Sample points along route geometry roughly every sample_step_km.
    geometry item format: [lon, lat]
    """
    if not geometry:
        return []

    points = []
    total_distance = 0.0

    # Keep first point always
    first_lon, first_lat = geometry[0]
    points.append(
        {
            "index": 0,
            "latitude": first_lat,
            "longitude": first_lon,
            "distance_from_start_km": 0.0,
        }
    )

    next_target = sample_step_km
    sampled_idx = 1

    for i in range(1, len(geometry)):
        prev_lon, prev_lat = geometry[i - 1]
        curr_lon, curr_lat = geometry[i]
        segment_km = haversine_km(prev_lat, prev_lon, curr_lat, curr_lon)

        if segment_km <= 0:
            continue

        while total_distance + segment_km >= next_target and len(points) < max_points:
            # Interpolate point where target distance intersects this segment
            ratio = (next_target - total_distance) / segment_km
            sampled_lat = _lerp(prev_lat, curr_lat, ratio)
            sampled_lon = _lerp(prev_lon, curr_lon, ratio)
            points.append(
                {
                    "index": sampled_idx,
                    "latitude": sampled_lat,
                    "longitude": sampled_lon,
                    "distance_from_start_km": round(next_target, 3),
                }
            )
            sampled_idx += 1
            next_target += sample_step_km

        total_distance += segment_km

    # Keep last point always (unless already very close to last sample)
    last_lon, last_lat = geometry[-1]
    last_distance = round(total_distance, 3)
    if (
        not points
        or haversine_km(
            points[-1]["latitude"], points[-1]["longitude"], last_lat, last_lon
        )
        > 0.15
    ):
        points.append(
            {
                "index": sampled_idx,
                "latitude": last_lat,
                "longitude": last_lon,
                "distance_from_start_km": last_distance,
            }
        )

    return points


def _dedupe_routes(routes: list[dict]) -> list[dict]:
    """
    Remove near-duplicate alternatives based on duration + distance signature.
    """
    seen = set()
    deduped = []
    for route in routes:
        signature = (round(route.get("duration_sec", 0) / 30), round(route.get("distance_m", 0) / 250))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(route)
    return deduped


def _extract_tomtom_geometry(route_payload: dict) -> list[list[float]]:
    """
    Convert TomTom route legs points to [lon, lat] geometry.
    """
    geometry: list[list[float]] = []
    legs = route_payload.get("legs", [])
    for leg in legs:
        for point in leg.get("points", []):
            lat = point.get("latitude")
            lon = point.get("longitude")
            if lat is None or lon is None:
                continue
            coord = [lon, lat]
            if not geometry or geometry[-1] != coord:
                geometry.append(coord)
    return geometry


def _normalize_tomtom_routes(
    routes: list[dict],
    mode: str,
    sample_step_km: float,
    max_sample_points: int,
) -> list[dict]:
    normalized = []
    for idx, route in enumerate(routes):
        summary = route.get("summary", {})
        geometry = _extract_tomtom_geometry(route)
        if len(geometry) < 2:
            continue

        distance_m = round(float(summary.get("lengthInMeters", 0.0)), 2)
        provider_duration_sec = round(float(summary.get("travelTimeInSeconds", 0.0)), 2)
        mode_speed_kmph = PROFILE_SPEED_KMPH.get(mode)

        # Keep provider ETA, but if provider returns an invalid time for non-driving,
        # fallback to profile speed estimate.
        if mode_speed_kmph and provider_duration_sec <= 0 and distance_m > 0:
            duration_sec = round(distance_m / ((mode_speed_kmph * 1000) / 3600), 2)
        else:
            duration_sec = provider_duration_sec

        sampled = _sample_route_points(
            geometry=geometry,
            sample_step_km=sample_step_km,
            max_points=max_sample_points,
        )

        normalized.append(
            {
                "route_id": f"route_{idx + 1}",
                "profile": mode,
                "provider": "tomtom",
                "distance_m": distance_m,
                "duration_sec": duration_sec,
                "provider_duration_sec": provider_duration_sec,
                "traffic_delay_sec": summary.get("trafficDelayInSeconds"),
                "segment_step_km": sample_step_km,
                "geometry": geometry,
                "sample_points": sampled,
            }
        )
    return normalized


async def _get_tomtom_route_alternatives(
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    max_alternatives: int,
    sample_step_km: float,
    max_sample_points: int,
) -> list[dict]:
    if not TOMTOM_API_KEY:
        return []

    o_lat, o_lon = origin
    d_lat, d_lon = destination
    tomtom_mode = TOMTOM_MODE_MAP.get(mode, "car")

    url = f"{TOMTOM_ROUTING_BASE_URL}/{o_lat},{o_lon}:{d_lat},{d_lon}/json"
    params = {
        "key": TOMTOM_API_KEY,
        "travelMode": tomtom_mode,
        "routeType": "fastest",
        "maxAlternatives": max(max_alternatives - 1, 0),
        "computeTravelTimeFor": "all",
    }
    if tomtom_mode == "car":
        params["traffic"] = "true"

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 403:
                logger.warning("[Routing] TomTom routing forbidden for current API key.")
                return []
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning(f"[Routing] TomTom request failed, will fallback to OSRM: {exc}")
        return []

    raw_routes = payload.get("routes", [])
    return _normalize_tomtom_routes(
        raw_routes,
        mode=mode,
        sample_step_km=sample_step_km,
        max_sample_points=max_sample_points,
    )[:max_alternatives]


async def _get_osrm_route_alternatives(
    origin: tuple[float, float],
    destination: tuple[float, float],
    mode: str,
    max_alternatives: int,
    sample_step_km: float,
    max_sample_points: int,
) -> list[dict]:
    o_lat, o_lon = origin
    d_lat, d_lon = destination
    url = (
        f"{OSRM_BASE_URL}/{mode}/{o_lon},{o_lat};{d_lon},{d_lat}"
        f"?alternatives=true&steps=false&overview=full&geometries=geojson"
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.error(f"[Routing] OSRM request failed: {exc}")
        return []

    if payload.get("code") != "Ok":
        logger.warning(f"[Routing] OSRM returned non-ok response: {payload.get('code')}")
        return []

    raw_routes = payload.get("routes", [])
    normalized = []

    for idx, route in enumerate(raw_routes):
        geometry = route.get("geometry", {}).get("coordinates", [])
        if len(geometry) < 2:
            continue

        distance_m = round(route.get("distance", 0.0), 2)
        provider_duration_sec = round(route.get("duration", 0.0), 2)
        mode_speed_kmph = PROFILE_SPEED_KMPH.get(mode)
        if mode_speed_kmph:
            duration_sec = (
                round(distance_m / ((mode_speed_kmph * 1000) / 3600), 2)
                if distance_m > 0
                else provider_duration_sec
            )
        else:
            duration_sec = provider_duration_sec

        sampled = _sample_route_points(
            geometry=geometry,
            sample_step_km=sample_step_km,
            max_points=max_sample_points,
        )
        normalized.append(
            {
                "route_id": f"route_{idx + 1}",
                "profile": mode,
                "provider": "osrm",
                "distance_m": distance_m,
                "duration_sec": duration_sec,
                "provider_duration_sec": provider_duration_sec,
                "segment_step_km": sample_step_km,
                "geometry": geometry,  # [lon, lat]
                "sample_points": sampled,
            }
        )

    normalized = _dedupe_routes(normalized)
    return normalized[:max_alternatives]


async def get_route_alternatives(
    origin: tuple[float, float],
    destination: tuple[float, float],
    profile: str = "driving",
    max_alternatives: int = 3,
    sample_step_km: float = 2.0,
    max_sample_points: int = 18,
) -> list[dict]:
    """
    Get route alternatives between origin and destination.
    Returns normalized route objects with sampled points.
    """
    mode = PROFILE_MAP.get(profile.lower(), "driving")

    tomtom_routes = await _get_tomtom_route_alternatives(
        origin=origin,
        destination=destination,
        mode=mode,
        max_alternatives=max_alternatives,
        sample_step_km=sample_step_km,
        max_sample_points=max_sample_points,
    )
    if tomtom_routes:
        return _dedupe_routes(tomtom_routes)[:max_alternatives]

    osrm_routes = await _get_osrm_route_alternatives(
        origin=origin,
        destination=destination,
        mode=mode,
        max_alternatives=max_alternatives,
        sample_step_km=sample_step_km,
        max_sample_points=max_sample_points,
    )
    return osrm_routes
