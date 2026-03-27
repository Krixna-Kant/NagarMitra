"""
NagarMitra - Telegram Webhook Router
Receives Telegram updates, handles commands, location sharing, and
dispatches free-text to the LangGraph ward agent.
"""

import os
import logging
import math
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Request

from app.bot.ward_agent import run_ward_agent, KNOWN_WARDS
from app.services.ward_mapper import WARD_CENTROIDS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Telegram Bot"])

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
INTERNAL_API = "http://127.0.0.1:8000"

# ─── Alert throttling ─────────────────────────────────────────────────────────
# {user_id: {"ward": str, "alerted_at": float}}
_last_alert: dict[int, dict] = {}
ALERT_COOLDOWN_SECONDS = 300   # 5 min — don't re-alert same ward within this
AQI_ALERT_THRESHOLD    = 150   # AQI above this triggers a proactive alert


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