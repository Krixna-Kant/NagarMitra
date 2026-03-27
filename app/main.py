"""
NagarMitra - Phase 1
FastAPI Application Entry Point
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.routes.aqi import router as aqi_router
from app.routes.ml import router as ml_router
from app.routes.admin import router as admin_router
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nagarmitra")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NagarMitra Phase 2 - ML Intelligence Service starting...")
    yield
    logger.info("NagarMitra shutting down.")


app = FastAPI(
    title="NagarMitra API - Phase 1, 2 & 4",
    description="Live Ward AQI, Source Attribution, AI Forecasting, and Admin Dashboard What-If Simulations.",
    version="4.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://nagarmitra.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(aqi_router)
app.include_router(ml_router)
app.include_router(admin_router)

@app.get("/")
async def root():
    return {
        "project":  "NagarMitra",
        "phase":    "Phase 1 - AQI Intelligence",
        "status":   "running",
        "docs":     "/docs",
        "endpoints": {
            "live_stations":   "/api/v1/aqi/stations/live",
            "ward_aqi":        "/api/v1/aqi/ward/{ward_name}",
            "all_wards":       "/api/v1/aqi/wards/all",
            "dashboard":       "/api/v1/aqi/dashboard",
            "list_wards":      "/api/v1/aqi/wards",
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "nagarmitra-phase1"}