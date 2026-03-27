const API_BASE = "http://localhost:8000";

const map = L.map("map").setView([28.6139, 77.209], 11);
const APP_CONFIG = {
  tomtom_enabled: false,
  map: {
    provider: "openstreetmap",
    tile_url_template: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    traffic_flow_tile_url_template: null,
  },
  search: {
    provider: "nominatim",
    tomtom_key: null,
  },
};

const ACTIVE_ROUTE_COLOR = "#1668e3";
let activeRouteLayers = [];
let routeLayerById = {};
let markerLayerGroup = null;
let pollutionOverlayLayer = null;
let trafficOverlayLayer = null;
let baseTileLayer = null;
let currentLocationMarker = null;
let lastRenderedRoutes = [];
let lastPreferredRouteId = null;
let activeRouteId = null;
const selectedPlaces = { origin: null, destination: null };
const suggestionTimers = { origin: null, destination: null };

const formEl = document.getElementById("route-form");
const statusEl = document.getElementById("status");
const recommendationEl = document.getElementById("recommendation");
const cardsEl = document.getElementById("route-cards");
const alphaInput = document.getElementById("alpha-input");
const alphaReadout = document.getElementById("alpha-readout");
const submitBtn = document.getElementById("submit-btn");
const originInputEl = document.getElementById("origin-input");
const destinationInputEl = document.getElementById("destination-input");
const originSuggestionsEl = document.getElementById("origin-suggestions");
const destinationSuggestionsEl = document.getElementById("destination-suggestions");
const trafficOverlayToggleEl = document.getElementById("traffic-overlay-toggle");

function applyBaseMapLayer() {
  if (baseTileLayer) {
    map.removeLayer(baseTileLayer);
    baseTileLayer = null;
  }
  const provider = APP_CONFIG.map.provider || "openstreetmap";
  const attribution =
    provider === "tomtom"
      ? "&copy; TomTom, &copy; OpenStreetMap contributors"
      : "&copy; OpenStreetMap contributors";
  baseTileLayer = L.tileLayer(APP_CONFIG.map.tile_url_template, {
    maxZoom: 20,
    attribution,
  }).addTo(map);
}

async function loadAppConfig() {
  try {
    const resp = await fetch(`${API_BASE}/api/v1/routes/config`);
    if (!resp.ok) {
      throw new Error(`config API failed (${resp.status})`);
    }
    const payload = await resp.json();
    if (payload && payload.map && payload.search) {
      APP_CONFIG.tomtom_enabled = Boolean(payload.tomtom_enabled);
      APP_CONFIG.map = payload.map;
      APP_CONFIG.search = payload.search;
    }
    applyBaseMapLayer();
    setStatus(
      APP_CONFIG.tomtom_enabled
        ? "TomTom map + search enabled."
        : "Using OpenStreetMap fallback map/search."
    );
  } catch {
    applyBaseMapLayer();
  }
}

alphaInput.addEventListener("input", () => {
  const value = Number(alphaInput.value).toFixed(2);
  alphaReadout.textContent = `${value} (0 cleanest to 1 fastest)`;
});

function setStatus(msg) {
  statusEl.textContent = msg;
}

function clearRouteLayers() {
  activeRouteLayers.forEach((layer) => map.removeLayer(layer));
  activeRouteLayers = [];
  routeLayerById = {};
  lastRenderedRoutes = [];
  lastPreferredRouteId = null;
  activeRouteId = null;
  if (markerLayerGroup) {
    map.removeLayer(markerLayerGroup);
    markerLayerGroup = null;
  }
  if (pollutionOverlayLayer) {
    map.removeLayer(pollutionOverlayLayer);
    pollutionOverlayLayer = null;
  }
  if (trafficOverlayLayer) {
    map.removeLayer(trafficOverlayLayer);
    trafficOverlayLayer = null;
  }
}

function hideSuggestions(kind) {
  const listEl = kind === "origin" ? originSuggestionsEl : destinationSuggestionsEl;
  listEl.classList.remove("show");
  listEl.innerHTML = "";
}

function normalizeSuggestion(item) {
  return {
    latitude: Number(item.lat),
    longitude: Number(item.lon),
    displayName: item.display_name,
  };
}

function normalizeTomTomResult(item) {
  const position = item.position || {};
  const address = item.address || {};
  return {
    latitude: Number(position.lat),
    longitude: Number(position.lon),
    displayName:
      address.freeformAddress ||
      item.poi?.name ||
      `${position.lat || ""}, ${position.lon || ""}`,
  };
}

async function fetchPlaceSuggestions(query) {
  if (APP_CONFIG.search.provider === "tomtom" && APP_CONFIG.search.tomtom_key) {
    const tomtomUrl = `https://api.tomtom.com/search/2/search/${encodeURIComponent(query)}.json?key=${
      APP_CONFIG.search.tomtom_key
    }&limit=6&language=en-IN`;
    const tomtomResp = await fetch(tomtomUrl, {
      headers: { Accept: "application/json" },
    });
    if (tomtomResp.ok) {
      const tomtomData = await tomtomResp.json();
      return (tomtomData.results || []).map(normalizeTomTomResult);
    }
  }

  const fallbackUrl =
    "https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&limit=6&q=" +
    encodeURIComponent(query);
  const fallbackResp = await fetch(fallbackUrl, {
    headers: { Accept: "application/json" },
  });
  if (!fallbackResp.ok) {
    throw new Error(`Suggestion search failed (${fallbackResp.status})`);
  }
  const fallbackData = await fallbackResp.json();
  return fallbackData.map(normalizeSuggestion);
}

function renderSuggestions(kind, suggestions) {
  const listEl = kind === "origin" ? originSuggestionsEl : destinationSuggestionsEl;
  listEl.innerHTML = "";
  if (!suggestions.length) {
    hideSuggestions(kind);
    return;
  }

  suggestions.forEach((suggestion) => {
    const item = document.createElement("li");
    item.textContent = suggestion.displayName;
    item.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      const inputEl = kind === "origin" ? originInputEl : destinationInputEl;
      selectedPlaces[kind] = suggestion;
      inputEl.value = suggestion.displayName;
      hideSuggestions(kind);
    });
    listEl.appendChild(item);
  });
  listEl.classList.add("show");
}

function bindAutocomplete(kind, inputEl) {
  inputEl.addEventListener("input", async () => {
    selectedPlaces[kind] = null;
    const query = inputEl.value.trim();
    clearTimeout(suggestionTimers[kind]);

    if (query.length < 3) {
      hideSuggestions(kind);
      return;
    }

    suggestionTimers[kind] = setTimeout(async () => {
      try {
        const suggestions = await fetchPlaceSuggestions(query);
        renderSuggestions(kind, suggestions);
      } catch (err) {
        setStatus(`Suggestion search error: ${err.message}`);
        hideSuggestions(kind);
      }
    }, 250);
  });

  inputEl.addEventListener("focus", () => {
    const kindSuggestions = kind === "origin" ? originSuggestionsEl : destinationSuggestionsEl;
    if (kindSuggestions.children.length) {
      kindSuggestions.classList.add("show");
    }
  });

  inputEl.addEventListener("blur", () => {
    setTimeout(() => hideSuggestions(kind), 120);
  });
}

async function geocodePlace(query) {
  if (APP_CONFIG.search.provider === "tomtom" && APP_CONFIG.search.tomtom_key) {
    const tomtomUrl = `https://api.tomtom.com/search/2/search/${encodeURIComponent(query)}.json?key=${
      APP_CONFIG.search.tomtom_key
    }&limit=1&language=en-IN`;
    const tomtomResp = await fetch(tomtomUrl, {
      headers: { Accept: "application/json" },
    });
    if (tomtomResp.ok) {
      const tomtomData = await tomtomResp.json();
      if ((tomtomData.results || []).length) {
        return normalizeTomTomResult(tomtomData.results[0]);
      }
    }
  }

  const fallbackUrl = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(
    query
  )}`;
  const fallbackResp = await fetch(fallbackUrl, {
    headers: {
      Accept: "application/json",
    },
  });
  if (!fallbackResp.ok) {
    throw new Error(`Geocoding failed (${fallbackResp.status})`);
  }
  const fallbackData = await fallbackResp.json();
  if (!fallbackData.length) {
    throw new Error(`No location match found for: ${query}`);
  }
  return {
    latitude: Number(fallbackData[0].lat),
    longitude: Number(fallbackData[0].lon),
    displayName: fallbackData[0].display_name,
  };
}

async function reverseGeocode(lat, lon) {
  if (APP_CONFIG.search.provider === "tomtom" && APP_CONFIG.search.tomtom_key) {
    const tomtomUrl = `https://api.tomtom.com/search/2/reverseGeocode/${lat},${lon}.json?key=${APP_CONFIG.search.tomtom_key}&language=en-IN`;
    const tomtomResp = await fetch(tomtomUrl, { headers: { Accept: "application/json" } });
    if (tomtomResp.ok) {
      const tomtomData = await tomtomResp.json();
      const address = tomtomData.addresses?.[0]?.address?.freeformAddress;
      if (address) {
        return address;
      }
    }
  }

  const fallbackUrl = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`;
  const fallbackResp = await fetch(fallbackUrl, { headers: { Accept: "application/json" } });
  if (!fallbackResp.ok) {
    return `Current location (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
  }
  const fallbackData = await fallbackResp.json();
  return fallbackData.display_name || `Current location (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
}

async function initializeOriginFromCurrentLocation() {
  if (!navigator.geolocation) {
    return;
  }
  navigator.geolocation.getCurrentPosition(
    async (position) => {
      const { latitude, longitude } = position.coords;
      const label = await reverseGeocode(latitude, longitude);
      selectedPlaces.origin = { latitude, longitude, displayName: label };
      originInputEl.value = label;

      if (currentLocationMarker) {
        map.removeLayer(currentLocationMarker);
      }
      currentLocationMarker = L.marker([latitude, longitude], {
        title: "Current location",
      })
        .addTo(map)
        .bindPopup("You are here")
        .openPopup();

      map.setView([latitude, longitude], 12);
      setStatus("Origin set to your current location. You can change it any time.");
    },
    () => {
      setStatus("Location permission denied. Please type origin manually.");
    },
    {
      enableHighAccuracy: true,
      timeout: 12000,
    }
  );
}

function segmentColor(segment) {
  const score100 = (segment.point_pollution_score || 0) * 100;
  const aqi = Number(segment.aqi || 0);
  if (aqi >= 195 || score100 >= 60) return "#d32f2f"; // high
  if (aqi >= 125 || score100 >= 44) return "#ff8f00"; // moderate
  return "";
}

function updateEndpointMarkers(route) {
  if (markerLayerGroup) {
    map.removeLayer(markerLayerGroup);
    markerLayerGroup = null;
  }
  if (!route || !route.geometry_latlon || route.geometry_latlon.length < 2) {
    return;
  }
  const start = route.geometry_latlon[0];
  const end = route.geometry_latlon[route.geometry_latlon.length - 1];
  markerLayerGroup = L.layerGroup([
    L.marker(start).bindPopup("Origin"),
    L.marker(end).bindPopup("Destination"),
  ]).addTo(map);
}

function renderPollutionOverlay(route) {
  if (pollutionOverlayLayer) {
    map.removeLayer(pollutionOverlayLayer);
    pollutionOverlayLayer = null;
  }
  if (!route || !route.pollution) {
    return;
  }

  const segments = route.pollution.segment_scores || [];
  pollutionOverlayLayer = L.layerGroup();
  const drawQueue = [];

  segments.forEach((segment) => {
    const color = segmentColor(segment);
    if (!color) {
      return;
    }
    drawQueue.push({ segment, color });
  });

  // Ensure user always sees at least top hotspots on active route.
  if (drawQueue.length === 0 && segments.length) {
    const topHotspots = [...segments]
      .sort((a, b) => (b.point_pollution_score || 0) - (a.point_pollution_score || 0))
      .slice(0, 1);
    topHotspots.forEach((segment) => drawQueue.push({ segment, color: "#ff8f00" }));
  }

  drawQueue.forEach(({ segment, color }) => {
    const score100 = ((segment.point_pollution_score || 0) * 100).toFixed(1);
    const segmentRadiusMeters = 900; // smaller visual bubble for cleaner map
    const circle = L.circle([segment.latitude, segment.longitude], {
      radius: segmentRadiusMeters,
      color,
      weight: 1.2,
      opacity: 0.75,
      fillColor: color,
      fillOpacity: color === "#d32f2f" ? 0.18 : 0.12,
    }).bindPopup(
      `Segment ${segment.index} | ${segment.distance_from_start_km} km from start` +
        `<br/>AQI: ${segment.aqi}` +
        `<br/>Predicted AQI (traffic-adjusted): ${Number(segment.predicted_aqi || segment.aqi).toFixed(1)}` +
        `<br/>Dust index: ${Number(segment.dust_index || 0).toFixed(2)}` +
        `<br/>Humidity: ${Number(segment.humidity_pct || 0).toFixed(1)}%` +
        `<br/>Traffic density: ${Number(segment.traffic_density_per_sq_km || 0).toFixed(2)}` +
        `<br/>Construction density: ${Number(segment.construction_density_per_sq_km || 0).toFixed(2)}` +
        `<br/>Pollution score: ${score100}` +
        `<br/>Severity: ${color === "#d32f2f" ? "High" : "Moderate"}`
    );
    pollutionOverlayLayer.addLayer(circle);
  });

  pollutionOverlayLayer.addTo(map);
}

function haversineKm(a, b) {
  const toRad = (v) => (v * Math.PI) / 180;
  const [lat1, lon1] = a;
  const [lat2, lon2] = b;
  const r = 6371;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * r * Math.asin(Math.sqrt(x));
}

function trafficColor(index) {
  const percent = index * 100;
  if (percent > 20) return "#d32f2f"; // heavy
  if (percent > 10) return "#ff8f00"; // moderate
  return "#1e88e5"; // light
}

function trafficLevelFromIndex(index) {
  const percent = index * 100;
  if (percent > 20) return "heavy";
  if (percent > 10) return "moderate";
  return "light";
}

function nearestTrafficSample(samples, distanceKm) {
  if (!samples.length) return null;
  let nearest = samples[0];
  let bestDelta = Math.abs(
    (Number(nearest.distance_from_start_km) || 0) - distanceKm
  );
  for (let i = 1; i < samples.length; i += 1) {
    const candidate = samples[i];
    const delta = Math.abs(
      (Number(candidate.distance_from_start_km) || 0) - distanceKm
    );
    if (delta < bestDelta) {
      nearest = candidate;
      bestDelta = delta;
    }
  }
  return nearest;
}

function tomtomTrafficStyle(sample) {
  const source = String(sample?.traffic_source || "").toLowerCase();
  const fromTomTom = source.includes("tomtom");
  const current = Number(sample?.traffic_current_speed_kmph);
  const freeFlow = Number(sample?.traffic_free_flow_speed_kmph);
  let speedRatio = Number(sample?.traffic_speed_ratio);

  if (!Number.isFinite(speedRatio) && Number.isFinite(current) && Number.isFinite(freeFlow) && freeFlow > 0) {
    speedRatio = current / freeFlow;
  }

  if (fromTomTom && sample?.traffic_road_closure) {
    return { color: "#FDD835", level: "road_closed", label: "Road closed" };
  }

  // Use TomTom relative0 style bands when TomTom speed ratio is available.
  if (fromTomTom && Number.isFinite(speedRatio)) {
    if (speedRatio < 0.15) return { color: "#A50704", level: "heavy", label: "Heavy traffic" };
    if (speedRatio < 0.35) return { color: "#DF4B15", level: "high", label: "High traffic" };
    if (speedRatio < 0.75) return { color: "#E87B3D", level: "moderate", label: "Moderate traffic" };
    return { color: "#1E88E5", level: "light", label: "Light traffic" };
  }

  const congestion = Number(sample?.traffic_congestion_index || 0);
  return {
    color: trafficColor(congestion),
    level: trafficLevelFromIndex(congestion),
    label: "Traffic proxy",
  };
}

function renderTrafficOverlay(route) {
  if (trafficOverlayLayer) {
    map.removeLayer(trafficOverlayLayer);
    trafficOverlayLayer = null;
  }
  if (!trafficOverlayToggleEl.checked) {
    return;
  }

  if (!route || !route.pollution) {
    return;
  }

  const segments = route.pollution.segment_scores || [];
  const geometry = route.geometry_latlon || [];
  if (!segments.length || geometry.length < 2) {
    return;
  }

  trafficOverlayLayer = L.layerGroup();
  let runningDistanceKm = 0;

  for (let i = 1; i < geometry.length; i += 1) {
    const prev = geometry[i - 1];
    const curr = geometry[i];
    const segKm = haversineKm(prev, curr);
    const midKm = runningDistanceKm + segKm / 2;
    const sample = nearestTrafficSample(segments, midKm);
    const style = tomtomTrafficStyle(sample);
    const congestion = Number(sample?.traffic_congestion_index || 0);
    const speedRatio = Number(sample?.traffic_speed_ratio);
    const trafficSource = sample?.traffic_source || "proxy";

    const line = L.polyline([prev, curr], {
      color: style.color,
      weight: 6,
      opacity: 0.86,
      lineCap: "round",
    }).bindPopup(
      `Traffic (${trafficSource}): ${style.label}` +
        `<br/>Congestion index: ${(congestion * 100).toFixed(1)} / 100` +
        `<br/>Speed ratio: ${
          Number.isFinite(speedRatio) ? speedRatio.toFixed(2) : "n/a"
        }` +
        `<br/>Current speed: ${
          sample?.traffic_current_speed_kmph != null
            ? `${Number(sample.traffic_current_speed_kmph).toFixed(1)} km/h`
            : "n/a"
        }` +
        `<br/>Free-flow speed: ${
          sample?.traffic_free_flow_speed_kmph != null
            ? `${Number(sample.traffic_free_flow_speed_kmph).toFixed(1)} km/h`
            : "n/a"
        }` +
        `<br/>Distance on route: ${midKm.toFixed(1)} km`
    );
    trafficOverlayLayer.addLayer(line);
    runningDistanceKm += segKm;
  }

  trafficOverlayLayer.addTo(map);
}

function applyRouteStyles() {
  Object.entries(routeLayerById).forEach(([routeId, layer]) => {
    const isActive = routeId === activeRouteId;
    layer.setStyle({
      color: ACTIVE_ROUTE_COLOR,
      weight: isActive ? 7 : 5,
      opacity: isActive ? 0.96 : 0.28,
    });
    if (isActive) {
      layer.bringToFront();
    }
  });
}

function setActiveRoute(routeId) {
  activeRouteId = routeId;
  applyRouteStyles();
  const activeRoute = lastRenderedRoutes.find((route) => route.route_id === activeRouteId);
  updateEndpointMarkers(activeRoute);
  renderPollutionOverlay(activeRoute);
  renderTrafficOverlay(activeRoute);
  renderCards(lastRenderedRoutes, lastPreferredRouteId, activeRouteId);
}

function renderRoutes(routes, preferredRouteId) {
  if (!routes.length) {
    clearRouteLayers();
    return;
  }
  clearRouteLayers();
  lastRenderedRoutes = routes;
  lastPreferredRouteId = preferredRouteId || routes[0].route_id;
  activeRouteId = lastPreferredRouteId;

  const bounds = [];
  routes.forEach((route) => {
    const latLngs = route.geometry_latlon;
    const polyline = L.polyline(latLngs, {
      color: ACTIVE_ROUTE_COLOR,
      weight: 5,
      opacity: 0.35,
    }).addTo(map);
    polyline.on("click", () => setActiveRoute(route.route_id));
    activeRouteLayers.push(polyline);
    routeLayerById[route.route_id] = polyline;
    bounds.push(...latLngs);
  });

  if (bounds.length) {
    map.fitBounds(bounds, { padding: [35, 35] });
  }

  setActiveRoute(activeRouteId);
}

function pollutionPill(score) {
  if (score <= 30) return '<span class="pill best">Low pollution</span>';
  if (score <= 55) return '<span class="pill warn">Moderate pollution</span>';
  return '<span class="pill warn">High pollution</span>';
}

function renderCards(routes, preferredRouteId, activeId) {
  cardsEl.innerHTML = "";
  routes.forEach((route) => {
    const isPreferred = route.route_id === preferredRouteId;
    const isActive = route.route_id === activeId;
    const mins = (route.duration_sec / 60).toFixed(1);
    const km = (route.distance_m / 1000).toFixed(2);

    const riskPoints = (route.pollution.risk_segments || [])
      .slice(0, 3)
      .map((seg) => `~${seg.distance_from_start_km} km: AQI ${seg.aqi}, score ${(seg.point_pollution_score * 100).toFixed(1)}`)
      .join(" | ");

    const card = document.createElement("article");
    card.className = `route-card ${isActive ? "active" : ""}`;
    card.addEventListener("click", () => setActiveRoute(route.route_id));
    card.innerHTML = `
      <h3>${route.route_id.toUpperCase()} ${isPreferred ? '<span class="pill best">Recommended</span>' : ""} ${isActive ? '<span class="pill active">Active</span>' : ""}</h3>
      <div>ETA: <strong>${mins} min</strong> | Distance: <strong>${km} km</strong></div>
      <div>Pollution score: <strong>${route.pollution.score}</strong> ${pollutionPill(route.pollution.score)}</div>
      <div class="meta-grid">
        <div>Avg AQI: <strong>${route.pollution.avg_aqi}</strong></div>
        <div>Predicted AQI: <strong>${Number(route.pollution.avg_predicted_aqi || route.pollution.avg_aqi || 0).toFixed(1)}</strong></div>
        <div>Avg dust index: <strong>${Number(route.pollution.avg_dust_index || 0).toFixed(2)}</strong></div>
        <div>Wind: <strong>${route.pollution.avg_wind_speed_ms} m/s</strong></div>
        <div>Humidity: <strong>${Number(route.pollution.avg_humidity_pct || 0).toFixed(1)}%</strong></div>
        <div>Traffic density: <strong>${Number(route.pollution.avg_traffic_density_per_sq_km || 0).toFixed(2)}</strong></div>
        <div>Traffic congestion index: <strong>${(Number(route.pollution.avg_traffic_congestion_index || 0) * 100).toFixed(1)}</strong></div>
        <div>Traffic source: <strong>${route.pollution.traffic_source || "proxy"}</strong></div>
        <div>Traffic speed: <strong>${
          route.pollution.avg_traffic_current_speed_kmph != null
            ? `${Number(route.pollution.avg_traffic_current_speed_kmph).toFixed(1)}`
            : "n/a"
        } / ${
      route.pollution.avg_traffic_free_flow_speed_kmph != null
        ? `${Number(route.pollution.avg_traffic_free_flow_speed_kmph).toFixed(1)}`
        : "n/a"
    } km/h</strong></div>
        <div>Construction density: <strong>${Number(route.pollution.avg_construction_density_per_sq_km || 0).toFixed(2)}</strong></div>
        <div>Confidence: <strong>${route.pollution.confidence}</strong> (${route.pollution.confidence_value})</div>
        <div>Segment step: <strong>${route.pollution.segment_step_km || 2} km</strong></div>
        <div>Overlay radius: <strong>0.9 km visual</strong> (2 km analysis)</div>
      </div>
      <div class="seg-list"><strong>Top hotspot segments:</strong> ${riskPoints || "No major hotspots identified"}</div>
      <div class="seg-list"><strong>Why:</strong> ${route.recommendation_reason}</div>
      <div style="margin-top:6px;color:${ACTIVE_ROUTE_COLOR}">Click card/route to activate</div>
    `;
    cardsEl.appendChild(card);
  });
}

async function resolveCoordinates(kind, typedValue) {
  const picked = selectedPlaces[kind];
  if (picked) {
    return picked;
  }
  return geocodePlace(typedValue);
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Geocoding origin and destination...");
  recommendationEl.textContent = "";
  cardsEl.innerHTML = "";
  submitBtn.disabled = true;

  try {
    const originQuery = document.getElementById("origin-input").value.trim();
    const destinationQuery = document.getElementById("destination-input").value.trim();
    const profile = document.getElementById("profile-input").value;
    const corridorRadiusKm = Number(document.getElementById("radius-input").value || 2);
    const alpha = Number(alphaInput.value);

    const [origin, destination] = await Promise.all([
      resolveCoordinates("origin", originQuery),
      resolveCoordinates("destination", destinationQuery),
    ]);

    setStatus("Fetching and scoring route alternatives...");
    const response = await fetch(`${API_BASE}/api/v1/routes/compare`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        origin: {
          latitude: origin.latitude,
          longitude: origin.longitude,
        },
        destination: {
          latitude: destination.latitude,
          longitude: destination.longitude,
        },
        profile,
        corridor_radius_km: corridorRadiusKm,
        sample_step_km: 2.0,
        max_routes: 3,
        preference_alpha: alpha,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Route comparison failed");
    }

    const routes = payload.routes || [];
    renderRoutes(routes, payload.recommendation.preferred_route_id);
    renderCards(routes, payload.recommendation.preferred_route_id, activeRouteId);
    recommendationEl.textContent = payload.recommendation.summary;
    setStatus(
      `Compared ${routes.length} route(s). Preferred: ${payload.recommendation.preferred_route_id}. Provider: ${
        payload.metadata?.route_provider || "unknown"
      }. Click a route to make it active.`
    );
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    clearRouteLayers();
  } finally {
    submitBtn.disabled = false;
  }
});

trafficOverlayToggleEl.addEventListener("change", () => {
  const activeRoute = lastRenderedRoutes.find((route) => route.route_id === activeRouteId);
  renderTrafficOverlay(activeRoute);
});

async function initApp() {
  await loadAppConfig();
  bindAutocomplete("origin", originInputEl);
  bindAutocomplete("destination", destinationInputEl);
  initializeOriginFromCurrentLocation();
}

initApp();
