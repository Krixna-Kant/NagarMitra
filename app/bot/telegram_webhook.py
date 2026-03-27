"""
NagarMitra - Telegram Webhook Router
Receives Telegram updates, handles commands, location sharing, and
dispatches free-text to the LangGraph ward agent.
"""

import os
import logging
import math
import re
import time
from typing import Optional
from urllib.parse import quote, urlencode

import httpx
from fastapi import APIRouter, Request

from app.bot.ward_agent import run_ward_agent, KNOWN_WARDS
from app.services.ward_mapper import WARD_CENTROIDS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Telegram Bot"])

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
INTERNAL_API = "http://127.0.0.1:8000"
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")
ROUTE_UI_BASE_URL = os.getenv("ROUTE_UI_BASE_URL", "https://theuilink.com")

# ─── Alert throttling ─────────────────────────────────────────────────────────
# {user_id: {"ward": str, "alerted_at": float}}
_last_alert: dict[int, dict] = {}
ALERT_COOLDOWN_SECONDS = 300   # 5 min — don't re-alert same ward within this
AQI_ALERT_THRESHOLD    = 150   # AQI above this triggers a proactive alert
DEFAULT_ROUTE_PROFILE  = "driving"

# {user_id: {"latitude": float, "longitude": float, "chat_id": int, "updated_at": float}}
_last_known_location: dict[int, dict] = {}
ROUTE_INTENT_PATTERN = re.compile(
    r"(?:go to|going to|route to|navigate to|directions to|reach|head to|take me to)\s+(.+)",
    flags=re.IGNORECASE,
)
ROUTE_KEYWORDS = ("route", "directions", "navigate", "go to", "reach", "take me")


# ─── Geospatial helpers ───────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def nearest_ward(lat: float, lon: float) -> tuple[str, float]:
    """Return (ward_name, distance_km) for the closest ward centroid."""
    best_name, best_dist = None, float("inf")
    for name, (wlat, wlon) in WARD_CENTROIDS.items():
        d = _haversine_km(lat, lon, wlat, wlon)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, round(best_dist, 2)


def _should_alert(user_id: int, ward: str) -> bool:
    """True if we haven't alerted this user about this ward recently."""
    last = _last_alert.get(user_id)
    if not last:
        return True
    if last["ward"] != ward:
        return True
    return (time.time() - last["alerted_at"]) > ALERT_COOLDOWN_SECONDS


def _mark_alerted(user_id: int, ward: str):
    _last_alert[user_id] = {"ward": ward, "alerted_at": time.time()}


def _safe(text: str) -> str:
    """Strip markdown special chars from backend-generated text to avoid Telegram parse errors."""
    if not text:
        return ""
    # Replace chars that break Telegram Markdown v1
    return str(text).replace("*", "").replace("_", "").replace("`", "").replace("[", "(").replace("]", ")")


def _remember_user_location(user_id: int, chat_id: int, lat: float, lon: float):
    _last_known_location[user_id] = {
        "latitude": lat,
        "longitude": lon,
        "chat_id": chat_id,
        "updated_at": time.time(),
    }


def _extract_destination_query(text: str) -> Optional[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    match = ROUTE_INTENT_PATTERN.search(cleaned)
    if match:
        destination = match.group(1).strip(" .!?")
        return destination or None

    # Fallback: support messages that are only destination + route keyword.
    lowered = cleaned.lower()
    if any(keyword in lowered for keyword in ROUTE_KEYWORDS):
        parts = re.split(r"\broute\b|\bdirections\b|\bnavigate\b", cleaned, flags=re.IGNORECASE)
        if parts:
            candidate = parts[-1].strip(" .!?")
            if candidate and len(candidate) > 2:
                return candidate
    return None


def _infer_profile_from_text(text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("walk", "walking", "on foot")):
        return "walking"
    if any(token in lowered for token in ("bike", "bicycle", "cycling", "cycle")):
        return "cycling"
    return DEFAULT_ROUTE_PROFILE


def _is_route_request(text: str) -> bool:
    lowered = (text or "").lower()
    if "go to " in lowered or "navigate to " in lowered or "directions to " in lowered:
        return True
    return any(keyword in lowered for keyword in ROUTE_KEYWORDS)


def _normalize_tomtom_result(item: dict) -> Optional[dict]:
    position = item.get("position", {})
    lat = position.get("lat")
    lon = position.get("lon")
    if lat is None or lon is None:
        return None
    address = item.get("address", {})
    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "display_name": (
            address.get("freeformAddress")
            or item.get("poi", {}).get("name")
            or f"{lat}, {lon}"
        ),
    }


def _normalize_nominatim_result(item: dict) -> Optional[dict]:
    lat = item.get("lat")
    lon = item.get("lon")
    if lat is None or lon is None:
        return None
    return {
        "latitude": float(lat),
        "longitude": float(lon),
        "display_name": item.get("display_name", f"{lat}, {lon}"),
    }


async def _search_place_candidates(query: str, limit: int = 6) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    async with httpx.AsyncClient(timeout=20) as client:
        if TOMTOM_API_KEY:
            tomtom_url = f"https://api.tomtom.com/search/2/search/{quote(query)}.json"
            try:
                tt_resp = await client.get(
                    tomtom_url,
                    params={"key": TOMTOM_API_KEY, "limit": limit, "language": "en-IN"},
                    headers={"Accept": "application/json"},
                )
                if tt_resp.status_code == 200:
                    payload = tt_resp.json()
                    items = []
                    for raw in payload.get("results", []):
                        normalized = _normalize_tomtom_result(raw)
                        if normalized:
                            items.append(normalized)
                    if items:
                        return items
            except Exception as exc:
                logger.warning(f"[route_search] TomTom search failed: {exc}")

        fallback_url = "https://nominatim.openstreetmap.org/search"
        fallback_resp = await client.get(
            fallback_url,
            params={
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": limit,
                "q": query,
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "NagarMitraBot/1.0",
            },
        )
        if fallback_resp.status_code != 200:
            return []
        data = fallback_resp.json()
        return [x for x in (_normalize_nominatim_result(raw) for raw in data) if x]


def _pick_closest_candidate(origin_lat: float, origin_lon: float, candidates: list[dict]) -> Optional[dict]:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: _haversine_km(origin_lat, origin_lon, item["latitude"], item["longitude"]),
    )
    best = ranked[0]
    best["distance_from_origin_km"] = round(
        _haversine_km(origin_lat, origin_lon, best["latitude"], best["longitude"]),
        2,
    )
    return best


def _build_route_ui_link(
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    profile: str,
) -> str:
    params = urlencode(
        {
            "from": f"{origin_lat:.6f},{origin_lon:.6f}",
            "to": f"{destination_lat:.6f},{destination_lon:.6f}",
            # Keep both keys for compatibility with external UI integrations.
            "profiling": profile,
            "profile": profile,
        }
    )
    return f"{ROUTE_UI_BASE_URL}?{params}"


# ─── Telegram helpers ─────────────────────────────────────────────────────────

async def send_message(chat_id: int, text: str, parse_mode: str = "Markdown"):
    if not BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set")
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            )
    except Exception as e:
        logger.error(f"[send_message] {e}")


async def send_typing(chat_id: int):
    if not BOT_TOKEN:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{TELEGRAM_API}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
            )
    except Exception:
        pass


async def request_location_keyboard(chat_id: int, text: str):
    """Send a message with a 'Share Live Location' button."""
    if not BOT_TOKEN:
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "reply_markup": {
                        "keyboard": [[{"text": "📍 Share My Location", "request_location": True}]],
                        "resize_keyboard": True,
                        "one_time_keyboard": True,
                    },
                },
            )
    except Exception as e:
        logger.error(f"[request_location_keyboard] {e}")


# ─── Command Handlers ─────────────────────────────────────────────────────────

async def handle_start(chat_id: int, first_name: str):
    msg = (
        f"👋 Hey *{first_name}*! I'm *NagarMitra* 🌍\n\n"
        "I give you real-time air quality data for any Delhi ward.\n\n"
        "*How to use:*\n"
        "• Ask naturally: _What's the AQI in Rohini?_\n"
        "• Or: _Air quality near Connaught Place_\n\n"
        "🧭 Need routes? Ask: _I want to go to DLF CyberHub_ (after sharing location).\n\n"
        "📍 Share your *live location* and I'll alert you whenever you enter a high-pollution zone.\n\n"
        "Use /wards to see all supported areas.\n"
        "Use /location to share your live location.\n\n"
        "_Powered by live CPCB/AQICN sensors + AI_ 🤖"
    )
    await send_message(chat_id, msg)


async def handle_wards(chat_id: int):
    ward_list = "\n".join(f"• {w}" for w in KNOWN_WARDS)
    msg = f"📍 *Supported Delhi Wards ({len(KNOWN_WARDS)}):*\n\n{ward_list}"
    await send_message(chat_id, msg)


async def handle_help(chat_id: int):
    msg = (
        "*NagarMitra Help* 🌫\n\n"
        "Ask your question naturally:\n"
        "— _AQI in Rohini_\n"
        "— _What's the air quality near Hauz Khas?_\n"
        "— _Is Anand Vihar polluted today?_\n\n"
        "For route guidance:\n"
        "— _I want to go to DLF CyberHub_\n"
        "(_share location first using /location_)\n\n"
        "Commands:\n"
        "/start — Welcome message\n"
        "/wards — List all supported wards\n"
        "/location — Share live location for real-time alerts\n"
        "/help — This message\n\n"
        "Data is live from CPCB sensors, interpolated using IDW."
    )
    await send_message(chat_id, msg)


async def handle_location_cmd(chat_id: int):
    await request_location_keyboard(
        chat_id,
        "📍 Tap the button below to share your *live location*.\n\n"
        "I'll alert you whenever you enter a ward with *AQI > 150* (Moderate+).\n\n"
        "_Tip: Choose 'Share Live Location' (not just once) so I can track your movement._"
    )


# ─── Location Handler ─────────────────────────────────────────────────────────

async def handle_location_update(chat_id: int, user_id: int, lat: float, lon: float, is_live: bool):
    """Find nearest ward, fetch AQI, alert if above threshold."""
    _remember_user_location(user_id=user_id, chat_id=chat_id, lat=lat, lon=lon)
    ward_name, dist_km = nearest_ward(lat, lon)

    if dist_km > 20:
        # User is outside Delhi's ward coverage
        if not is_live:  # Only reply for one-shot shares, not every live update
            await send_message(chat_id,
                "📍 You seem to be outside NagarMitra's Delhi ward coverage area.\n"
                "Try asking about a specific ward instead.")
        return

    # Fetch AQI for the nearest ward
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.get(
                f"{INTERNAL_API}/api/v1/aqi/ward/{ward_name}",
                params={"lang": "en", "profile": "general"},
            )
            if resp.status_code != 200:
                if not is_live:
                    await send_message(chat_id, f"⚠️ Couldn't fetch AQI data for *{ward_name}* right now.")
                return
            data = resp.json()
    except Exception as e:
        logger.error(f"[handle_location_update] API error: {e}")
        if not is_live:
            await send_message(chat_id, "⚠️ Unable to reach the AQI service. Please try again.")
        return

    aqi_info  = data.get("aqi_data", {})
    weather   = data.get("weather", {})
    attr      = data.get("attribution", {})
    advisory  = data.get("advisory", {}).get("en", {})
    aqi_val   = aqi_info.get("aqi") or 0
    category  = aqi_info.get("category", "Unknown")

    # Emoji based on AQI
    emoji = "🟢" if aqi_val <= 50 else "🟡" if aqi_val <= 100 else "🟠" if aqi_val <= 200 else "🔴" if aqi_val <= 300 else "🟣"

    if is_live:
        # Live location: only alert if AQI is high AND not recently alerted
        if aqi_val >= AQI_ALERT_THRESHOLD and _should_alert(user_id, ward_name):
            _mark_alerted(user_id, ward_name)
            advisory_text = _safe(advisory.get("message", "Consider limiting outdoor exposure."))
            msg = (
                f"⚠️ *Air Quality Alert!*\n\n"
                f"{emoji} You're near *{ward_name}* — AQI is *{round(aqi_val)}* ({category})\n"
                f"🌡 Temp: {weather.get('temperature')}°C · 💨 Wind: {weather.get('wind_speed')} m/s\n"
                f"🏭 Main source: *{_safe(attr.get('dominant_source', 'Unknown'))}*\n\n"
                f"💡 {advisory_text}"
            )
            await send_message(chat_id, msg)
        # Clean area — stay silent
    else:
        # One-shot location share: always respond with full report
        dist_label  = f"(~{dist_km} km away)" if dist_km > 0.5 else "(you are here)"
        advisory_text = _safe(advisory.get("message", "Stay safe."))
        msg = (
            f"📍 *Nearest ward:* {ward_name} {dist_label}\n\n"
            f"{emoji} *AQI: {round(aqi_val)}* — {category}\n"
            f"🌡 Temp: {weather.get('temperature')}°C · 💧 Humidity: {weather.get('humidity')}%\n"
            f"💨 Wind: {weather.get('wind_speed')} m/s · 👁 Visibility: {weather.get('visibility')} m\n\n"
            f"🏭 *Top source:* {_safe(attr.get('dominant_source', 'Unknown'))} ({_safe(attr.get('confidence', ''))} confidence)\n\n"
            f"💡 {advisory_text}\n\n"
            f"Tip: Share Live Location to get automatic alerts as you move around Delhi."
        )
        await send_message(chat_id, msg)


async def handle_route_request(chat_id: int, user_id: int, text: str):
    location = _last_known_location.get(user_id)
    if not location:
        await request_location_keyboard(
            chat_id,
            "📍 I can plan your best route, but I need your *current location* first.\n\n"
            "Tap below and share location, then ask again like:\n"
            "_I want to go to DLF CyberHub_",
        )
        return

    destination_query = _extract_destination_query(text)
    if not destination_query:
        await send_message(
            chat_id,
            "🧭 Tell me your destination like:\n"
            "_I want to go to DLF CyberHub_",
        )
        return

    profile = _infer_profile_from_text(text)
    origin_lat = location["latitude"]
    origin_lon = location["longitude"]

    try:
        candidates = await _search_place_candidates(destination_query, limit=6)
    except Exception as exc:
        logger.error(f"[handle_route_request] destination search failed: {exc}")
        await send_message(chat_id, "⚠️ I couldn't search destinations right now. Please try again.")
        return

    best_destination = _pick_closest_candidate(origin_lat, origin_lon, candidates)
    if not best_destination:
        await send_message(
            chat_id,
            f"⚠️ I couldn't find a close match for *{_safe(destination_query)}*.\n"
            "Try a more specific place name.",
        )
        return

    compare_payload = {
        "origin": {"latitude": origin_lat, "longitude": origin_lon},
        "destination": {
            "latitude": best_destination["latitude"],
            "longitude": best_destination["longitude"],
        },
        "profile": profile,
        "corridor_radius_km": 2.0,
        "sample_step_km": 2.0,
        "max_routes": 3,
        "preference_alpha": 0.4,
    }

    try:
        async with httpx.AsyncClient(timeout=80) as client:
            resp = await client.post(f"{INTERNAL_API}/api/v1/routes/compare", json=compare_payload)
            if resp.status_code != 200:
                await send_message(chat_id, "⚠️ Route service is busy. Please try again in a minute.")
                return
            compare_data = resp.json()
    except Exception as exc:
        logger.error(f"[handle_route_request] compare API failed: {exc}")
        await send_message(chat_id, "⚠️ Unable to compute route right now. Please try again.")
        return

    routes = compare_data.get("routes", [])
    recommendation = compare_data.get("recommendation", {})
    preferred_route_id = recommendation.get("preferred_route_id")
    recommended_route = next(
        (route for route in routes if route.get("route_id") == preferred_route_id),
        routes[0] if routes else None,
    )
    if not recommended_route:
        await send_message(chat_id, "⚠️ I couldn't build route options for this trip. Please try another destination.")
        return

    duration_min = round(float(recommended_route.get("duration_sec", 0.0)) / 60.0, 1)
    distance_km = round(float(recommended_route.get("distance_m", 0.0)) / 1000.0, 2)
    pollution = recommended_route.get("pollution", {})
    pollution_score = pollution.get("score")
    avg_aqi = pollution.get("avg_predicted_aqi") or pollution.get("avg_aqi")
    pollution_score_label = pollution_score if pollution_score is not None else "n/a"
    avg_aqi_label = avg_aqi if avg_aqi is not None else "n/a"
    reason = _safe(recommended_route.get("recommendation_reason", "Best balance of time and air quality."))
    destination_label = _safe(best_destination.get("display_name", destination_query))
    recommendation_summary = _safe(recommendation.get("summary", "Recommended based on travel time and pollution exposure."))
    ui_link = _build_route_ui_link(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        destination_lat=best_destination["latitude"],
        destination_lon=best_destination["longitude"],
        profile=profile,
    )

    msg = (
        f"✅ I found the *best possible route* to *{destination_label}*.\n\n"
        f"🚗 Mode: *{profile.title()}*\n"
        f"🛣 Route: *{recommended_route.get('route_id', 'route_1').upper()}* (recommended)\n"
        f"⏱ ETA: *{duration_min} min* · 📏 Distance: *{distance_km} km*\n"
        f"🌫 Avg AQI: *{avg_aqi_label}* · Pollution score: *{pollution_score_label}*\n"
        f"📍 Destination match is ~{best_destination.get('distance_from_origin_km', '?')} km from your shared location\n\n"
        f"{recommendation_summary}\n"
        f"Why this route: {reason}\n\n"
        f"🔗 Open full route UI (all alternatives):\n{ui_link}"
    )
    await send_message(chat_id, msg)


# ─── Webhook Endpoint ─────────────────────────────────────────────────────────

@router.post("/webhook")
async def telegram_webhook(req: Request):
    """Receive Telegram updates and dispatch to handlers."""
    try:
        update = await req.json()
    except Exception:
        return {"ok": True}

    # ── Handle edited_message (live location updates) ──
    edited = update.get("edited_message")
    if edited and "location" in edited:
        chat_id  = edited["chat"]["id"]
        user_id  = edited.get("from", {}).get("id", chat_id)
        loc      = edited["location"]
        _remember_user_location(user_id, chat_id, loc["latitude"], loc["longitude"])
        await handle_location_update(
            chat_id, user_id,
            loc["latitude"], loc["longitude"],
            is_live=True,
        )
        return {"ok": True}

    message = update.get("message")
    if not message:
        return {"ok": True}

    chat_id    = message["chat"]["id"]
    user_id    = message.get("from", {}).get("id", chat_id)
    first_name = message.get("from", {}).get("first_name", "there")

    # ── One-shot OR initial live location share ──
    if "location" in message:
        loc = message["location"]
        _remember_user_location(user_id, chat_id, loc["latitude"], loc["longitude"])
        await send_typing(chat_id)
        live_period = loc.get("live_period")  # present = live share, absent = one-shot
        if live_period:
            # Initial live location message — always confirm + show current AQI
            ward_name, dist_km = nearest_ward(loc["latitude"], loc["longitude"])
            try:
                async with httpx.AsyncClient(timeout=25) as client:
                    resp = await client.get(
                        f"{INTERNAL_API}/api/v1/aqi/ward/{ward_name}",
                        params={"lang": "en", "profile": "general"},
                    )
                    data = resp.json() if resp.status_code == 200 else {}
            except Exception:
                data = {}

            aqi_info = data.get("aqi_data", {})
            aqi_val  = aqi_info.get("aqi") or 0
            category = aqi_info.get("category", "Unknown")
            emoji    = "🟢" if aqi_val <= 50 else "🟡" if aqi_val <= 100 else "🟠" if aqi_val <= 200 else "🔴" if aqi_val <= 300 else "🟣"

            await send_message(chat_id,
                f"📡 *Live location tracking started!*\n\n"
                f"📍 Currently near *{ward_name}* {f'(~{dist_km} km)' if dist_km > 0.5 else ''}\n"
                f"{emoji} AQI: *{round(aqi_val)}* — {category}\n\n"
                f"I'll alert you automatically when you enter a zone with AQI > {AQI_ALERT_THRESHOLD}.\n"
                f"_Stop sharing your location in Telegram to turn off alerts._"
            )
        else:
            # True one-shot share — full detailed report
            await handle_location_update(
                chat_id, user_id,
                loc["latitude"], loc["longitude"],
                is_live=False,
            )
        return {"ok": True}


    text = message.get("text", "").strip()
    if not text:
        return {"ok": True}

    # ── Commands ──
    if text.startswith("/start"):
        await handle_start(chat_id, first_name)
        return {"ok": True}

    if text.startswith("/wards"):
        await handle_wards(chat_id)
        return {"ok": True}

    if text.startswith("/location"):
        await handle_location_cmd(chat_id)
        return {"ok": True}

    if text.startswith("/help"):
        await handle_help(chat_id)
        return {"ok": True}

    # ── Route planning intent ──
    if _is_route_request(text):
        await send_typing(chat_id)
        await handle_route_request(chat_id, user_id, text)
        return {"ok": True}

    # ── Natural language location intent ──
    _loc_keywords = ["my location", "track me", "track my", "share location",
                     "where am i", "where i am", "current location", "live location",
                     "aqi here", "aqi near me", "pollution here", "air here"]
    if any(kw in text.lower() for kw in _loc_keywords):
        await handle_location_cmd(chat_id)
        return {"ok": True}

    # ── Natural language → LangGraph Agent ──
    await send_typing(chat_id)
    logger.info(f"[webhook] Agent invoked for: '{text}'")
    reply = await run_ward_agent(text)
    await send_message(chat_id, reply)

    return {"ok": True}