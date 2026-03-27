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
from app.bot.telegram_webhook import router as telegram_router

load_dotenv()

from app.routes.aqi import router as aqi_router
from app.routes.routes import router as routes_router
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nagarmitra")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NagarMitra Phase 1 - AQI Intelligence Service starting...")
    yield
    logger.info("NagarMitra shutting down.")


app = FastAPI(
    title="NagarMitra API - Phase 1",
    description="Ward-level AQI intelligence for Delhi: live data, source attribution, health advisories",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "https://nagarmitra.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(aqi_router)
app.include_router(routes_router)
app.include_router(telegram_router)


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
            "routes_config":   "/api/v1/routes/config",
            "compare_routes":  "/api/v1/routes/compare",
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "nagarmitra-phase1"}