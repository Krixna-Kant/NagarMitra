"""
NagarMitra - Pollution-aware Route APIs
"""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.pollution_score_service import (
    build_comparison_summary,
    score_and_rank_routes,
)
from app.services.routing_service import get_route_alternatives


router = APIRouter(prefix="/api/v1/routes", tags=["Routes"])
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")


class Coordinate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class CompareRoutesRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    profile: str = Field(default="driving", pattern="^(drive|driving|walk|walking|bike|cycling)$")
    corridor_radius_km: float = Field(default=2.0, ge=2.0, le=3.0)
    sample_step_km: float = Field(default=2.0, ge=1.0, le=5.0)
    max_routes: int = Field(default=3, ge=1, le=3)
    preference_alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="0.0 cleanest-first, 1.0 fastest-first",
    )


@router.get("/config")
async def get_routes_config():
    """
    Frontend map/routing configuration hints.
    Key is returned because tile services require client-side key usage.
    """
    tomtom_enabled = bool(TOMTOM_API_KEY)
    return {
        "tomtom_enabled": tomtom_enabled,
        "map": {
            "provider": "tomtom" if tomtom_enabled else "openstreetmap",
            "tile_url_template": (
                f"https://api.tomtom.com/map/1/tile/basic/main/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_API_KEY}"
                if tomtom_enabled
                else "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            ),
            "traffic_flow_tile_url_template": (
                f"https://api.tomtom.com/traffic/map/4/tile/flow/relative/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_API_KEY}"
                if tomtom_enabled
                else None
            ),
        },
        "search": {
            "provider": "tomtom" if tomtom_enabled else "nominatim",
            "tomtom_key": TOMTOM_API_KEY if tomtom_enabled else None,
        },
    }


@router.post("/compare")
async def compare_routes(req: CompareRoutesRequest):
    origin = (req.origin.latitude, req.origin.longitude)
    destination = (req.destination.latitude, req.destination.longitude)

    if origin == destination:
        raise HTTPException(400, "Origin and destination cannot be the same.")

    routes = await get_route_alternatives(
        origin=origin,
        destination=destination,
        profile=req.profile,
        max_alternatives=req.max_routes,
        sample_step_km=req.sample_step_km,
    )
    if not routes:
        raise HTTPException(503, "Unable to fetch route alternatives from routing provider.")

    ranked_routes = await score_and_rank_routes(
        routes=routes,
        corridor_radius_km=req.corridor_radius_km,
        profile=req.profile,
        alpha=req.preference_alpha,
    )
    summary = build_comparison_summary(ranked_routes)

    # Convert geometry format for map clients expecting [lat, lon]
    for route in ranked_routes:
        route["geometry_latlon"] = [[lat, lon] for lon, lat in route["geometry"]]

    return {
        "metadata": {
            "profile": req.profile,
            "routes_considered": len(ranked_routes),
            "corridor_radius_km": req.corridor_radius_km,
            "sample_step_km": req.sample_step_km,
            "preference_alpha": req.preference_alpha,
            "route_provider": ranked_routes[0].get("provider") if ranked_routes else None,
        },
        "recommendation": summary,
        "routes": ranked_routes,
    }
