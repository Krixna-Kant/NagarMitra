# NagarMitra Admin Dashboard - Feature Complete Edition
# Deployed Backend: https://nagarmitra.onrender.com

import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="NagarMitra Admin Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deployment URLs
REMOTE_BACKEND = "https://nagarmitra.onrender.com"
BACKEND_URL = os.getenv("BACKEND_URL", REMOTE_BACKEND)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Custom CSS (Glassmorphism) ---
def local_css():
    st.markdown("""
        <style>
        .stApp { background-color: #0f0f1a; color: #ffffff; }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        [data-testid="stMetric"]:hover { transform: translateY(-5px); border-color: #00d4aa; transition: 0.3s; }
        section[data-testid="stSidebar"] { background-color: #161625; border-right: 1px solid rgba(255, 255, 255, 0.1); }
        .highlight { color: #00d4aa; font-weight: bold; }
        h1, h2, h3 { color: #00d4aa !important; }
        .stDataFrame { background: rgba(255, 255, 255, 0.02); border-radius: 8px; }
        
        /* Tooltip style */
        .pdk-tooltip { background-color: #1a1a2e !important; color: white !important; border: 1px solid #00d4aa !important; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- Data Fetching Layer ---
@st.cache_data(ttl=300)
def fetch_get(endpoint):
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=15)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection Failed: {str(e)}"}

def fetch_post(endpoint, data):
    try:
        response = requests.post(f"{BACKEND_URL}{endpoint}", json=data, timeout=20)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Failed: {str(e)}"}

def get_aqi_color(aqi):
    if not aqi: return "#333333"
    if aqi <= 50: return "#00B050"
    if aqi <= 100: return "#92D050"
    if aqi <= 200: return "#FFFF00"
    if aqi <= 300: return "#FF9900"
    if aqi <= 400: return "#FF0000"
    return "#800000"

# --- Sidebar ---
st.sidebar.markdown(f"# 🌿 <span class='highlight'>NagarMitra</span> Admin", unsafe_allow_html=True)
st.sidebar.caption(f"Backend: {BACKEND_URL.split('//')[-1]}")
st.sidebar.divider()

if 'last_updated' not in st.session_state:
    st.session_state.last_updated = datetime.now().strftime("%H:%M:%S")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🗺️ Heatmap", "📊 Risk Ranking", "🔍 Attribution & Health", "📋 Decisions", "🔮 Simulator", "🤖 AI Advisor"],
)

st.sidebar.divider()
st.sidebar.info(f"Last Updated: {st.session_state.last_updated}")
if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.session_state.last_updated = datetime.now().strftime("%H:%M:%S")
    st.rerun()

# --- Load Baseline Data ---
dashboard_data = fetch_get("/api/v1/aqi/dashboard")
if "error" in dashboard_data:
    st.error(f"⚠️ Connection to **{BACKEND_URL}** failed. Please ensure the server is online.")
    st.stop()

summary = dashboard_data.get("summary", {})
weather = dashboard_data.get("weather", {})
all_wards = dashboard_data.get("all_wards", [])

# 1. 🏠 OVERVIEW
if menu == "🏠 Overview":
    st.title("Admin Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Delhi Avg AQI", summary.get("delhi_avg_aqi", "N/A"))
    col2.metric("Worst Ward", dashboard_data.get("worst_5_wards", [{}])[0].get("ward", "N/A"))
    col3.metric("Best Ward", dashboard_data.get("best_5_wards", [{}])[-1].get("ward", "N/A"))
    col4.metric("Stations Live", summary.get("total_stations_live", 0))
    col5.metric("Severe Wards", summary.get("wards_severe", 0))
    
    st.markdown("### 🌤️ Weather Monitoring")
    w1, w2, w3, w4 = st.columns(4)
    w1.write(f"**Wind:** {weather.get('wind_speed')} m/s")
    w2.write(f"**Humid:** {weather.get('humidity')}%")
    w3.write(f"**Temp:** {weather.get('temperature')}°C")
    w4.write(f"**Status:** {weather.get('weather_main')}")
    st.divider()
    
    st.markdown("### 🚨 High Risk Areas")
    worst_df = pd.DataFrame(dashboard_data.get("worst_5_wards", []))
    if not worst_df.empty:
        st.table(worst_df[['ward', 'aqi', 'category', 'nearest_station']])

# 2. 🗺️ HEATMAP
elif menu == "🗺️ Heatmap":
    st.title("Ward-Level AQI Mesh")
    map_data = [{
        "name": w["ward"], "lat": w["latitude"], "lon": w["longitude"], "aqi": w["aqi"],
        "color": [int(get_aqi_color(w['aqi'])[i:i+2], 16) for i in (1, 3, 5)] + [160]
    } for w in all_wards if w.get("latitude")]
    
    if map_data:
        st.pydeck_chart(pdk.Deck(
            layers=[
                pdk.Layer("HeatmapLayer", data=map_data, get_position="[lon, lat]", get_weight="aqi", radius_pixels=60, intensity=1),
                pdk.Layer("ScatterplotLayer", data=map_data, get_position="[lon, lat]", get_color="color", get_radius=500, pickable=True)
            ],
            initial_view_state=pdk.ViewState(latitude=28.6139, longitude=77.2090, zoom=9.5, pitch=45),
            map_style="mapbox://styles/mapbox/dark-v11",
            tooltip={"text": "{name}\nAQI: {aqi}"}
        ))
    else: st.warning("No spatial data.")

# 3. 📊 RISK RANKING
elif menu == "📊 Risk Ranking":
    st.title("Ward Risk Assessment")
    df_wards = pd.DataFrame(all_wards)
    if not df_wards.empty:
        search = st.text_input("🔍 Search for a Ward", "")
        if search: df_wards = df_wards[df_wards['ward'].str.contains(search, case=False)]
        
        cols = ['ward', 'aqi', 'category', 'nearest_station', 'distance_km']
        st.dataframe(df_wards[[c for c in cols if c in df_wards.columns]].rename(columns={"ward": "Ward", "aqi": "AQI", "category": "Severity"}), use_container_width=True, hide_index=True)

# 4. 🔍 ATTRIBUTION & HEALTH
elif menu == "🔍 Attribution & Health":
    st.title("Ward Root Cause & Health Score")
    selected_ward = st.selectbox("Select Ward", [w['ward'] for w in all_wards])
    
    if selected_ward:
        with st.spinner("Analyzing..."):
            detail = fetch_get(f"/api/v1/aqi/ward/{selected_ward}")
            if "error" not in detail:
                attr = detail.get("attribution", {})
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.subheader(f"Pollution Profile")
                    fig = px.pie(names=list(attr.get("source_breakdown", {}).keys()), values=list(attr.get("source_breakdown", {}).values()), hole=0.5)
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**AI Insight:** {attr.get('explanation', {}).get('en')}")
                
                with c2:
                    st.subheader("🧬 WHRI Score")
                    # Fetch Phase 2 Health Score
                    whri = fetch_get(f"/api/v1/ml/whri/{selected_ward}?aqi_score={detail.get('aqi_data', {}).get('aqi', 100)}")
                    if "error" not in whri:
                        score = whri.get("whri_score", 0)
                        st.metric("Health Risk Index", score, f"{whri.get('risk_band')} Risk")
                        st.markdown(f"**Alert:** {whri.get('alert_message')}")
                        # Simple Progress Bar Gage
                        st.progress(score / 100.0)
                    else: st.warning("WHRI Service unavailable.")
                    
                st.divider()
                st.subheader("💡 Health Advisory")
                adv = detail.get("advisory", {})
                t1, t2 = st.tabs(["English", "हिंदी"])
                t1.info(adv.get("en", {}).get("advisory", "No data"))
                t2.info(adv.get("hi", {}).get("advisory", "कोई डेटा उपलब्ध नहीं है"))
            else: st.error("Detail fetch failed.")

# 5. 📋 DECISIONS
elif menu == "📋 Decisions":
    st.title("Scientific Policy Mandates")
    with st.spinner("Fetching optimal policies..."):
        top_wards = all_wards[:10]
        results = []
        for w in top_wards:
            detail = fetch_get(f"/api/v1/aqi/ward/{w['ward']}")
            if "error" not in detail:
                a = detail.get("attribution", {})
                source = a.get("dominant_source", "unknown")
                # Phase 4 Policy Engine call
                policies = fetch_get(f"/api/v1/admin/policies/{source}")
                top_policy = policies.get("actionable_policies", ["Standard Mitigation"])[0]
                results.append({"Ward": w['ward'], "AQI": w['aqi'], "Source": source.title(), "Recommended Mandate": top_policy})
        
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# 6. 🔮 SIMULATOR
elif menu == "🔮 Simulator":
    st.title("What-If Simulation Engine")
    st.markdown("Predict the impact of policy interventions before deployment.")
    
    col1, col2, col3 = st.columns(3)
    ward = col1.selectbox("Ward to Simulate", [w['ward'] for w in all_wards])
    policy = col2.selectbox("Select Intervention", ["Odd-Even Mandate", "Halt Construction", "Dust Sprinkling", "Industrial Shutdown"])
    intensity = col3.slider("Policy Strictness / Intensity", 0.0, 1.0, 0.5)

    if st.button("🚀 Run Prediction"):
        with st.spinner("Processing in Phase 4 Simulation Engine..."):
            res = fetch_post("/api/v1/admin/simulate", {"ward_name": ward, "policy_action": policy, "severity": intensity})
            if "error" not in res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Current AQI", res.get("current_aqi"))
                c2.metric("Predicted AQI", res.get("impact_prediction", {}).get("after_aqi"), f"-{res.get('impact_prediction', {}).get('pct_reduction')}%")
                c3.metric("Reduction Confidence", f"{res.get('confidence_score')*100}%")
                
                st.success(f"**Prediction Summary:** {res.get('explanation')}")
            else: st.error("Simulation failed.")

# 7. 🤖 AI ADVISOR
elif menu == "🤖 AI Advisor":
    st.title("Admin Research Consultant")
    st.caption("Phase 4 specialized AI researcher for administrative scientific queries.")
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Ask a scientific or administrative question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # Using Phase 4 Consult Endpoint
            res = fetch_post("/api/v1/admin/ai/consult", {"query": prompt})
            answer = res.get("ai_response", "I'm sorry, I couldn't reach the consultant.")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
