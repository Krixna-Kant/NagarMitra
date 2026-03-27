"""
NagarMitra - Telegram Bot
LangGraph agent: detects ward name → fetches live AQI data → LLM summary via OpenRouter
"""

import os
import logging
from typing import TypedDict, Optional

import httpx
from langgraph.graph import StateGraph, END
from app.services.ward_mapper import WARD_CENTROIDS

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Internal base URL — bot calls our own FastAPI
INTERNAL_API_BASE  = "http://127.0.0.1:8000"

# Ward list derived from the live WARD_CENTROIDS map — single source of truth
KNOWN_WARDS: list[str] = sorted(WARD_CENTROIDS.keys())


# ─── State ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_message: str
    ward_name: Optional[str]
    ward_data: Optional[dict]
    reply: Optional[str]
    error: Optional[str]


# ─── Node 1: Detect ward name from user message via LLM ─────────────────────

async def detect_ward(state: AgentState) -> AgentState:
    """Use OpenRouter LLM to extract the Delhi ward name from user's message."""
    user_msg = state["user_message"]

    wards_list = ", ".join(KNOWN_WARDS)

    prompt = f"""You are a ward name extractor for Delhi, India.

Extract the Delhi ward/area name from this user message:
"{user_msg}"

Known wards: {wards_list}

Rules:
- If the user mentions a ward name (or a close variant, e.g. "rohini" → "Rohini"), return EXACTLY the matched ward name from the list above with correct casing.
- If no clear ward is mentioned, return: NONE
- Return ONLY the ward name or NONE. No explanation, no punctuation.

Ward name:"""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://nagarmitra.app",
                    "X-Title": "NagarMitra Bot",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 30,
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            ward = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info(f"[detect_ward] LLM returned: '{ward}'")

            if ward.upper() == "NONE" or not ward:
                return {**state, "ward_name": None, "error": "no_ward"}
            return {**state, "ward_name": ward, "error": None}

    except Exception as e:
        logger.error(f"[detect_ward] Error: {e}")
        return {**state, "ward_name": None, "error": "llm_error"}


# ─── Node 2: Fetch live ward data from our own API ───────────────────────────

async def fetch_ward_data(state: AgentState) -> AgentState:
    """Call /api/v1/aqi/ward/{ward_name} on our own FastAPI backend."""
    ward = state.get("ward_name")
    if not ward:
        return state

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{INTERNAL_API_BASE}/api/v1/aqi/ward/{ward}",
                params={"lang": "en", "profile": "general"},
            )
            if resp.status_code == 404:
                return {**state, "ward_data": None, "error": "ward_not_found"}
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"[fetch_ward_data] Got data for {ward}, AQI={data.get('aqi_data', {}).get('aqi')}")
            return {**state, "ward_data": data, "error": None}

    except Exception as e:
        logger.error(f"[fetch_ward_data] Error: {e}")
        return {**state, "ward_data": None, "error": "api_error"}


# ─── Node 3: Summarise via LLM ───────────────────────────────────────────────

async def summarise_with_llm(state: AgentState) -> AgentState:
    """Ask OpenRouter to write a concise Telegram markdown summary from ward data."""
    data = state.get("ward_data")
    ward = state.get("ward_name")

    if not data:
        return state

    aqi_info   = data.get("aqi_data", {})
    weather    = data.get("weather", {})
    attr       = data.get("attribution", {})
    advisory   = data.get("advisory", {}).get("en", {})

    # Build a structured context for the LLM
    context = f"""
Ward: {ward}
AQI: {aqi_info.get("aqi")} ({aqi_info.get("category")})
Nearest station: {aqi_info.get("nearest_station")}
Method: IDW interpolation from live AQICN sensors

Weather:
  Temperature: {weather.get("temperature")}°C
  Humidity: {weather.get("humidity")}%
  Wind speed: {weather.get("wind_speed")} m/s
  Visibility: {weather.get("visibility")} m
  Conditions: {weather.get("weather_desc")}

Pollution source attribution:
  Dominant source: {attr.get("dominant_source")}
  Confidence: {attr.get("confidence")}
  Breakdown: {attr.get("source_breakdown")}

Health advisory:
  Category: {advisory.get("category")}
  Message: {advisory.get("message")}
  Mask advice: {advisory.get("mask_advice")}
  Actions: {advisory.get("recommended_actions")}
"""

    prompt = f"""You are NagarMitra, an air quality assistant for Delhi citizens.

Here is real-time AQI data for {ward}:
---
{context}
---

Write a SHORT, friendly Telegram message summarising this data. Use Telegram markdown (bold with *text*, no headers, no HTML).

Format:
- One-line AQI status with emoji
- Key weather condition
- Top 2 pollution sources
- One actionable health tip

Keep it under 10 lines. Make it feel human and helpful, not robotic.
"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://nagarmitra.app",
                    "X-Title": "NagarMitra Bot",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.4,
                },
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            return {**state, "reply": reply}

    except Exception as e:
        logger.error(f"[summarise_with_llm] Error: {e}")
        # Fallback: raw data reply without LLM
        aqi_val  = aqi_info.get("aqi")
        category = aqi_info.get("category", "Unknown")
        fallback = (
            f"🌫 *{ward}* — AQI *{aqi_val}* ({category})\n"
            f"Temp: {weather.get('temperature')}°C · Humidity: {weather.get('humidity')}%\n"
            f"Source: {attr.get('dominant_source', 'Unknown')} (dominant)\n"
            f"_{advisory.get('message', 'Stay safe.')}_"
        )
        return {**state, "reply": fallback}


# ─── Routing functions ────────────────────────────────────────────────────────

def route_after_detect(state: AgentState) -> str:
    if state.get("error") == "no_ward":
        return "no_ward"
    if state.get("error"):
        return "error"
    return "fetch"


def route_after_fetch(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    return "summarise"


# ─── Build the LangGraph ─────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("detect_ward",       detect_ward)
    g.add_node("fetch_ward_data",   fetch_ward_data)
    g.add_node("summarise_with_llm", summarise_with_llm)

    # Error / sink nodes (simple pass-through functions)
    def handle_no_ward(state):
        wards_hint = "\n".join(f"• {w}" for w in KNOWN_WARDS[:12])
        return {**state, "reply": (
            "🤔 I couldn't detect a Delhi ward in your message.\n\n"
            "Try something like:\n"
            "*What's the AQI in Rohini?*\n"
            "*Air quality in Connaught Place*\n\n"
            f"Some supported wards:\n{wards_hint}\n...and more!"
        )}

    def handle_error(state):
        err = state.get("error", "unknown")
        if err == "ward_not_found":
            return {**state, "reply": (
                f"⚠️ Ward *{state.get('ward_name')}* not found in NagarMitra's database.\n"
                "Try `/wards` to see all supported wards."
            )}
        return {**state, "reply": "⚠️ Something went wrong. Please try again in a moment."}

    g.add_node("handle_no_ward", handle_no_ward)
    g.add_node("handle_error",   handle_error)

    g.set_entry_point("detect_ward")

    g.add_conditional_edges("detect_ward", route_after_detect, {
        "fetch":   "fetch_ward_data",
        "no_ward": "handle_no_ward",
        "error":   "handle_error",
    })

    g.add_conditional_edges("fetch_ward_data", route_after_fetch, {
        "summarise": "summarise_with_llm",
        "error":     "handle_error",
    })

    g.add_edge("summarise_with_llm", END)
    g.add_edge("handle_no_ward",     END)
    g.add_edge("handle_error",       END)

    return g.compile()


# Singleton compiled graph — imported once at startup
ward_graph = _build_graph()


async def run_ward_agent(user_message: str) -> str:
    """Entry point for the Telegram webhook to invoke the agent."""
    initial_state: AgentState = {
        "user_message": user_message,
        "ward_name":    None,
        "ward_data":    None,
        "reply":        None,
        "error":        None,
    }
    result = await ward_graph.ainvoke(initial_state)
    return result.get("reply", "⚠️ No response generated.")