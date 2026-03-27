"""
NagarMitra - Pollution Scoring Service
Scores route alternatives using weighted AQI, dust, wind, humidity,
traffic, and construction signals.
"""

import asyncio

from app.services.environmental_data_service import get_environmental_snapshot


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else (num / den)


def _point_pollution_score(
    aqi_value: float,
    dust_index: float,
    wind_speed_ms: float,
    humidity_pct: float,
    traffic_congestion_index: float,
    traffic_density: float,
    construction_density: float,
    predicted_aqi_value: float = 0.0,
    profile: str = "driving",
    w_aqi: float = 0.4,
    w_dust: float = 0.16,
    w_wind: float = 0.12,
    w_humidity: float = 0.1,
    w_traffic: float = 0.12,
    w_construction: float = 0.1,
) -> dict:
    """
    Return normalized point score in [0,1] and components.
    Higher value means higher estimated pollution exposure at this 2km segment.
    """
    adjusted_aqi = predicted_aqi_value if predicted_aqi_value > 0 else aqi_value
    aqi_component = _clamp(adjusted_aqi / 500.0, 0.0, 1.0)
    dust_component = _clamp(dust_index, 0.0, 1.0)
    wind_component = _clamp((6.5 - wind_speed_ms) / 6.5, 0.0, 1.0)
    # Humidity can increase trapping at high values and resuspension risk at low values.
    humidity_low_component = _clamp((35.0 - humidity_pct) / 35.0, 0.0, 1.0)
    humidity_high_component = _clamp((humidity_pct - 78.0) / 22.0, 0.0, 1.0)
    humidity_component = _clamp((0.55 * humidity_low_component) + (0.45 * humidity_high_component), 0.0, 1.0)
    traffic_density_component = _clamp(traffic_density / 30.0, 0.0, 1.0)
    traffic_congestion_component = _clamp(traffic_congestion_index, 0.0, 1.0)
    traffic_component = _clamp(
        (0.72 * traffic_congestion_component) + (0.28 * traffic_density_component),
        0.0,
        1.0,
    )
    construction_component = _clamp(construction_density / 3.0, 0.0, 1.0)

    mode = (profile or "driving").lower()
    traffic_exposure_factor = 0.85 if mode == "driving" else (1.15 if mode == "cycling" else 1.05)
    overall_exposure_factor = 0.9 if mode == "driving" else (1.18 if mode == "cycling" else 1.08)
    traffic_component = _clamp(traffic_component * traffic_exposure_factor, 0.0, 1.0)

    score = (
        w_aqi * aqi_component
        + w_dust * dust_component
        + w_wind * wind_component
        + w_humidity * humidity_component
        + w_traffic * traffic_component
        + w_construction * construction_component
    )
    score = _clamp(score * overall_exposure_factor, 0.0, 1.0)
    return {
        "score": round(_clamp(score, 0.0, 1.0), 4),
        "components": {
            "aqi_component": round(aqi_component, 4),
            "adjusted_aqi": round(adjusted_aqi, 1),
            "dust_component": round(dust_component, 4),
            "wind_component": round(wind_component, 4),
            "humidity_component": round(humidity_component, 4),
            "traffic_component": round(traffic_component, 4),
            "traffic_congestion_component": round(traffic_congestion_component, 4),
            "traffic_density_component": round(traffic_density_component, 4),
            "construction_component": round(construction_component, 4),
        },
    }


def _traffic_level_label(congestion_index: float) -> str:
    if congestion_index >= 0.68:
        return "high"
    if congestion_index >= 0.4:
        return "moderate"
    return "low"


def _route_speed_congestion(route: dict, profile: str) -> float:
    """
    Estimate congestion from route-level observed speed against expected mode speed.
    """
    distance_m = float(route.get("distance_m") or 0.0)
    duration_sec = float(route.get("duration_sec") or 0.0)
    if distance_m <= 0 or duration_sec <= 0:
        return 0.2

    observed_speed_kmph = (distance_m / 1000.0) / (duration_sec / 3600.0)
    mode = (profile or "driving").lower()
    expected_speed_map = {
        "driving": 36.0,
        "cycling": 16.0,
        "walking": 5.0,
    }
    expected_speed = expected_speed_map.get(mode, 36.0)
    return _clamp(1.0 - (observed_speed_kmph / expected_speed), 0.0, 1.0)


async def _score_point(point: dict, radius_km: float, profile: str, route_speed_congestion: float) -> dict:
    lat = point["latitude"]
    lon = point["longitude"]

    try:
        snapshot = await get_environmental_snapshot(lat, lon, radius_km=radius_km)
    except Exception:
        # Provider outage fallback: keep endpoint functional with low confidence.
        snapshot = {
            "aqi": {"value": 120.0, "confidence": 0.2, "source": "fallback"},
            "wind": {"speed_ms": 3.0, "humidity_pct": 50.0, "confidence": 0.2, "source": "fallback"},
            "humidity": {"value_pct": 50.0, "confidence": 0.2, "source": "fallback"},
            "traffic": {
                "density_per_sq_km": 0.0,
                "congestion_index": 0.12,
                "congestion_level": "low",
                "current_speed_kmph": None,
                "free_flow_speed_kmph": None,
                "speed_ratio": 0.88,
                "road_closure": False,
                "confidence": 0.2,
                "source": "fallback",
            },
            "construction": {
                "density_per_sq_km": 0.0,
                "sites": 0,
                "confidence": 0.2,
                "source": "fallback",
            },
            "dust": {"index": 0.2, "confidence": 0.2, "source": "fallback"},
            "confidence": 0.2,
        }

    aqi_value = float(snapshot["aqi"].get("value", 120.0))
    wind_speed = float(snapshot["wind"].get("speed_ms", 3.0))
    humidity_pct = float(snapshot.get("humidity", {}).get("value_pct", 50.0))
    traffic_density = float(snapshot.get("traffic", {}).get("density_per_sq_km", 0.0))
    traffic_congestion_index_raw = float(snapshot.get("traffic", {}).get("congestion_index", 0.12))
    traffic_congestion_index = _clamp(
        (0.75 * traffic_congestion_index_raw) + (0.25 * route_speed_congestion),
        0.0,
        1.0,
    )
    traffic_level = _traffic_level_label(traffic_congestion_index)
    traffic_current_speed = snapshot.get("traffic", {}).get("current_speed_kmph")
    traffic_free_flow_speed = snapshot.get("traffic", {}).get("free_flow_speed_kmph")
    traffic_speed_ratio = snapshot.get("traffic", {}).get("speed_ratio")
    traffic_road_closure = bool(snapshot.get("traffic", {}).get("road_closure", False))
    traffic_source = snapshot.get("traffic", {}).get("source", "unknown")
    construction_density = float(snapshot["construction"].get("density_per_sq_km", 0.0))
    dust_index = float(snapshot.get("dust", {}).get("index", 0.2))
    predicted_aqi = round(aqi_value * (1.0 + (0.22 * traffic_congestion_index)), 1)

    score = _point_pollution_score(
        aqi_value=aqi_value,
        dust_index=dust_index,
        wind_speed_ms=wind_speed,
        humidity_pct=humidity_pct,
        traffic_congestion_index=traffic_congestion_index,
        traffic_density=traffic_density,
        construction_density=construction_density,
        predicted_aqi_value=predicted_aqi,
        profile=profile,
    )

    return {
        "index": point["index"],
        "latitude": lat,
        "longitude": lon,
        "distance_from_start_km": point.get("distance_from_start_km", 0.0),
        "aqi": aqi_value,
        "predicted_aqi": predicted_aqi,
        "dust_index": round(dust_index, 4),
        "wind_speed_ms": wind_speed,
        "humidity_pct": round(humidity_pct, 1),
        "traffic_density_per_sq_km": round(traffic_density, 3),
        "traffic_congestion_index_raw": round(traffic_congestion_index_raw, 3),
        "traffic_congestion_index": round(traffic_congestion_index, 3),
        "traffic_level": traffic_level,
        "route_speed_congestion_index": round(route_speed_congestion, 3),
        "traffic_current_speed_kmph": traffic_current_speed,
        "traffic_free_flow_speed_kmph": traffic_free_flow_speed,
        "traffic_speed_ratio": traffic_speed_ratio,
        "traffic_road_closure": traffic_road_closure,
        "traffic_source": traffic_source,
        "profile": profile,
        "construction_sites": snapshot["construction"].get("sites", 0),
        "construction_density_per_sq_km": construction_density,
        "point_pollution_score": score["score"],
        "components": score["components"],
        "confidence": snapshot.get("confidence", 0.2),
    }


async def score_route_pollution(
    route: dict,
    corridor_radius_km: float = 2.5,
    profile: str = "driving",
    max_concurrency: int = 6,
) -> dict:
    """
    Enrich route with per-point environmental signals and route-level pollution score.
    """
    corridor_radius_km = _clamp(corridor_radius_km, 2.0, 3.0)
    points = route.get("sample_points", [])
    if not points:
        route["pollution"] = {
            "score": 0.0,
            "avg_aqi": None,
            "avg_predicted_aqi": None,
            "avg_dust_index": None,
            "avg_wind_speed_ms": None,
            "avg_humidity_pct": None,
            "avg_traffic_density_per_sq_km": None,
            "avg_traffic_congestion_index": None,
            "avg_traffic_current_speed_kmph": None,
            "avg_traffic_free_flow_speed_kmph": None,
            "traffic_source": None,
            "avg_construction_density_per_sq_km": None,
            "confidence": "Low",
            "risk_segments": [],
        }
        return route

    route_speed_congestion = _route_speed_congestion(route, profile)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(point: dict) -> dict:
        async with semaphore:
            return await _score_point(
                point,
                corridor_radius_km,
                profile,
                route_speed_congestion=route_speed_congestion,
            )

    point_scores = await asyncio.gather(*[_bounded(point) for point in points])

    total_score = sum(p["point_pollution_score"] for p in point_scores)
    route_score = round((total_score / len(point_scores)) * 100, 2)
    avg_aqi = round(sum(p["aqi"] for p in point_scores) / len(point_scores), 1)
    avg_predicted_aqi = round(sum(p["predicted_aqi"] for p in point_scores) / len(point_scores), 1)
    avg_dust = round(sum(p["dust_index"] for p in point_scores) / len(point_scores), 3)
    avg_wind = round(sum(p["wind_speed_ms"] for p in point_scores) / len(point_scores), 2)
    avg_humidity = round(sum(p["humidity_pct"] for p in point_scores) / len(point_scores), 1)
    avg_traffic = round(
        sum(p["traffic_density_per_sq_km"] for p in point_scores) / len(point_scores), 3
    )
    avg_traffic_congestion = round(
        sum(p["traffic_congestion_index"] for p in point_scores) / len(point_scores), 3
    )
    traffic_speed_points = [p for p in point_scores if p.get("traffic_current_speed_kmph") is not None]
    avg_traffic_current_speed = (
        round(sum(float(p["traffic_current_speed_kmph"]) for p in traffic_speed_points) / len(traffic_speed_points), 1)
        if traffic_speed_points
        else None
    )
    free_flow_points = [p for p in point_scores if p.get("traffic_free_flow_speed_kmph") is not None]
    avg_traffic_free_flow_speed = (
        round(sum(float(p["traffic_free_flow_speed_kmph"]) for p in free_flow_points) / len(free_flow_points), 1)
        if free_flow_points
        else None
    )
    avg_construction = round(
        sum(p["construction_density_per_sq_km"] for p in point_scores) / len(point_scores), 3
    )
    conf_value = _safe_div(sum(p["confidence"] for p in point_scores), len(point_scores), 0.2)

    confidence_label = "Low"
    if conf_value >= 0.75:
        confidence_label = "High"
    elif conf_value >= 0.5:
        confidence_label = "Medium"

    risk_segments = sorted(
        point_scores,
        key=lambda item: item["point_pollution_score"],
        reverse=True,
    )[:3]

    route["pollution"] = {
        "score": route_score,  # 0-100, lower is better
        "avg_aqi": avg_aqi,
        "avg_predicted_aqi": avg_predicted_aqi,
        "avg_dust_index": avg_dust,
        "avg_wind_speed_ms": avg_wind,
        "avg_humidity_pct": avg_humidity,
        "avg_traffic_density_per_sq_km": avg_traffic,
        "avg_traffic_congestion_index": avg_traffic_congestion,
        "avg_traffic_current_speed_kmph": avg_traffic_current_speed,
        "avg_traffic_free_flow_speed_kmph": avg_traffic_free_flow_speed,
        "traffic_source": point_scores[0].get("traffic_source"),
        "avg_construction_density_per_sq_km": avg_construction,
        "segment_scores": point_scores,
        "segment_step_km": route.get("segment_step_km", 2.0),
        "route_speed_congestion_index": round(route_speed_congestion, 3),
        "confidence": confidence_label,
        "confidence_value": round(conf_value, 2),
        "risk_segments": risk_segments,
    }
    return route


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-9:
        return [0.0 for _ in values]
    return [round((v - min_v) / (max_v - min_v), 4) for v in values]


def _build_reason(route: dict, best: dict) -> str:
    pollution_delta = round(route["pollution"]["score"] - best["pollution"]["score"], 2)
    eta_delta_min = round((route["duration_sec"] - best["duration_sec"]) / 60.0, 1)

    if route["route_id"] == best["route_id"]:
        return "Best balance of lower pollution exposure and travel time."
    if pollution_delta <= 0 and eta_delta_min > 0:
        return f"Cleaner route but about {abs(eta_delta_min)} min slower than recommended."
    if pollution_delta > 0 and eta_delta_min <= 0:
        return f"Faster route but roughly {pollution_delta} pollution points higher."
    return (
        f"About {abs(eta_delta_min)} min slower and {pollution_delta} points higher pollution "
        "than recommended route."
    )


def rank_routes(scored_routes: list[dict], alpha: float = 0.4) -> list[dict]:
    """
    Rank routes by travel cost and pollution.
    alpha controls travel-time priority:
      alpha=1.0 -> mostly fastest
      alpha=0.0 -> mostly cleanest
    """
    if not scored_routes:
        return []

    alpha = _clamp(alpha, 0.0, 1.0)

    durations = [route.get("duration_sec", 0.0) for route in scored_routes]
    distances = [route.get("distance_m", 0.0) for route in scored_routes]
    pollution = [route.get("pollution", {}).get("score", 0.0) for route in scored_routes]

    n_durations = _normalize(durations)
    n_distances = _normalize(distances)
    n_pollution = _normalize(pollution)

    for idx, route in enumerate(scored_routes):
        travel_cost = round((0.7 * n_durations[idx]) + (0.3 * n_distances[idx]), 4)
        overall = round((alpha * travel_cost) + ((1 - alpha) * n_pollution[idx]), 4)

        route["analytics"] = {
            "travel_cost_normalized": travel_cost,
            "pollution_normalized": n_pollution[idx],
            "overall_score": overall,
            "alpha": alpha,
        }

    ranked = sorted(scored_routes, key=lambda route: route["analytics"]["overall_score"])
    best = ranked[0]

    for route in ranked:
        route["recommendation_reason"] = _build_reason(route, best)

    return ranked


async def score_and_rank_routes(
    routes: list[dict],
    corridor_radius_km: float = 2.5,
    profile: str = "driving",
    alpha: float = 0.4,
) -> list[dict]:
    """
    Full scoring pipeline:
    1) score each route's pollution exposure
    2) rank routes by combined travel + pollution objective
    """
    scored = await asyncio.gather(
        *[
            score_route_pollution(route, corridor_radius_km=corridor_radius_km, profile=profile)
            for route in routes
        ]
    )
    return rank_routes(scored_routes=scored, alpha=alpha)


def build_comparison_summary(ranked_routes: list[dict]) -> dict:
    if not ranked_routes:
        return {
            "preferred_route_id": None,
            "summary": "No routes available to compare.",
        }

    preferred = ranked_routes[0]
    cleanest = min(ranked_routes, key=lambda route: route["pollution"]["score"])
    fastest = min(ranked_routes, key=lambda route: route["duration_sec"])

    summary = (
        f"Recommended {preferred['route_id']} because it provides the best balance of ETA and "
        f"pollution exposure. Cleanest option is {cleanest['route_id']} "
        f"(pollution score {cleanest['pollution']['score']}), while fastest is "
        f"{fastest['route_id']} ({round(fastest['duration_sec'] / 60, 1)} min)."
    )
    return {
        "preferred_route_id": preferred["route_id"],
        "summary": summary,
    }
