(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let scenarioData = null;
  let liveSnapshot = null;
  let mapMode = "inspect";
  let mapStatus = null;
  let previousResults = [];
  let socket = null;
  let shownEvaluation = "";
  let initialized = false;
  let loadedResult = false;
  const chartColors = ["#69d6b1", "#73b5fa", "#f0b968", "#c19af5", "#f07977"];

  async function request(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
    return body;
  }

  function titleCase(value) {
    return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function currentFormConfig() {
    if (!scenarioData) throw new Error("Select a scenario first");
    const scenario = structuredClone(scenarioData);
    const rain = Number($("rainInput").value);
    const wind = Number($("windInput").value);
    scenario.weather ||= {};
    scenario.weather.timeline ||= [{ t: 0 }];
    scenario.weather.timeline.forEach((point) => {
      point.rain_mm_h = rain;
      point.wind_mps = wind;
    });
    scenario.mission ||= {};
    scenario.mission.default_speed_mps = Number($("speedInput").value);
    scenario.mission.takeoff_alt_m = Number($("altitudeInput").value);
    scenario.budget ||= {};
    scenario.budget.comm_power_w = Number($("radioPowerInput").value);
    scenario.links.forEach((link) => {
      link.ul_tx_dbm = Math.min(Number(link.ul_tx_dbm ?? 30), Number($("txLimitInput").value));
    });
    const enabledLinks = [...document.querySelectorAll("#radioToggles input:checked")].map((input) => input.value);
    return {
      scenario,
      policy: $("policySelect").value,
      duration_s: Number($("durationInput").value),
      dt: Number($("dtInput").value),
      seed: Number($("seedInput").value),
      seeds: Number($("seedsInput").value),
      enabled_links: enabledLinks,
      radio_power_limit_w: Number($("radioPowerInput").value),
      tx_limit_dbm: Number($("txLimitInput").value),
      predictive_handover: {
        horizon_s: Number($("horizonInput").value),
        snr_margin_db: Number($("marginInput").value),
        ctrl_latency_weight: Number($("latencyWeightInput").value),
        video_quality_weight: Number($("videoWeightInput").value),
        power_weight: Number($("powerWeightInput").value),
        switch_cost: Number($("switchCostInput").value),
      },
      evaluation: {
        control_latency_ms: Number($("ctrlThresholdInput").value),
        video_quality_min: Number($("videoThresholdInput").value),
      },
      map_edits: {
        jammer_eirp_dbm: Number($("jammerStrengthInput").value),
        jammer_bw_mhz: Number($("jammerBandwidthInput").value),
      },
    };
  }

  function populateRadioToggles() {
    const container = $("radioToggles");
    container.replaceChildren();
    (scenarioData?.links || []).forEach((link) => {
      const label = document.createElement("label");
      label.className = "radio-toggle";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = link.name;
      input.checked = true;
      input.addEventListener("change", drawMap);
      label.append(input, document.createTextNode(titleCase(link.name)));
      container.append(label);
    });
  }

  async function loadScenario(name, resetControls = true) {
    scenarioData = await request(`/api/scenarios/${encodeURIComponent(name)}`);
    liveSnapshot = null;
    mapStatus = await request(`/api/map/${encodeURIComponent(name)}?seed=${encodeURIComponent($("seedInput").value || "2026")}`);
    $("scenarioName").textContent = titleCase(scenarioData.name);
    const hasSatellite = scenarioData.links?.some((link) => /sat/i.test(`${link.name} ${link.tech}`));
    $("mapCaption").textContent = hasSatellite
      ? "Scenario geometry · live position · configured satellite link"
      : "Scenario geometry · live position · no satellite link in this simulator preset";
    $("scenarioSelect").value = name;
    if (resetControls) {
      $("durationInput").value = Math.min(Number(scenarioData.duration_s || 60), 60);
      $("dtInput").value = 0.02;
      $("speedInput").value = scenarioData.mission?.default_speed_mps || 12;
      $("altitudeInput").value = scenarioData.mission?.takeoff_alt_m || 50;
      $("radioPowerInput").value = scenarioData.budget?.comm_power_w || 8;
      $("txLimitInput").value = Math.max(...(scenarioData.links || []).map((link) => link.ul_tx_dbm || 30));
      $("rainInput").value = scenarioData.weather?.timeline?.[0]?.rain_mm_h || 0;
      $("windInput").value = scenarioData.weather?.timeline?.[0]?.wind_mps || 0;
    }
    populateRadioToggles();
    drawMap();
  }

  function drawMap() {
    const canvas = $("mapCanvas");
    if (!canvas || !scenarioData) return;
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * ratio);
    canvas.height = Math.round(bounds.height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = bounds.width;
    const height = bounds.height;
    const scenarioMap = makeScenarioMap(scenarioData);
    const liveMap = liveSnapshot?.map || mapStatus;
    const world = liveMap ? { ...scenarioMap, ...liveMap, waypoints: scenarioMap.waypoints, interference: scenarioMap.interference } : scenarioMap;
    const [worldW, worldH] = world.size_m || [2000, 2000];
    const margin = { left: 27, right: 23, top: 21, bottom: 25 };
    const scale = Math.min((width - margin.left - margin.right) / worldW, (height - margin.top - margin.bottom) / worldH);
    const mapW = worldW * scale;
    const mapH = worldH * scale;
    const offsetX = (width - mapW) / 2;
    const offsetY = (height - mapH) / 2;
    const point = (x, y) => [offsetX + x * scale, height - offsetY - y * scale];
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#14212a";
    ctx.fillRect(offsetX, height - offsetY - mapH, mapW, mapH);
    ctx.save();
    ctx.beginPath();
    ctx.rect(offsetX, height - offsetY - mapH, mapW, mapH);
    ctx.clip();
    const terrain = world.terrain_grid;
    if (terrain?.length && terrain[0]?.length) {
      const low = Number(world.terrain_min_m || 0);
      const high = Math.max(low + 0.01, Number(world.terrain_max_m || 1));
      const cellW = mapW / terrain[0].length;
      const cellH = mapH / terrain.length;
      terrain.forEach((row, rowIndex) => row.forEach((elevation, column) => {
        const intensity = Math.max(0, Math.min(1, (elevation - low) / (high - low)));
        const [x, y] = point(column * worldW / row.length, (rowIndex + 1) * worldH / terrain.length);
        ctx.fillStyle = `rgba(${48 + intensity * 36}, ${75 + intensity * 49}, ${77 + intensity * 39}, .52)`;
        ctx.fillRect(x, y, cellW + 1, cellH + 1);
      }));
    }
    ctx.strokeStyle = "#263b43";
    ctx.lineWidth = 1;
    const gridMeters = worldW > 4000 ? 500 : 250;
    for (let x = 0; x <= worldW; x += gridMeters) {
      const [px] = point(x, 0);
      ctx.beginPath();
      ctx.moveTo(px, height - offsetY);
      ctx.lineTo(px, height - offsetY - mapH);
      ctx.stroke();
    }
    for (let y = 0; y <= worldH; y += gridMeters) {
      const [, py] = point(0, y);
      ctx.beginPath();
      ctx.moveTo(offsetX, py);
      ctx.lineTo(offsetX + mapW, py);
      ctx.stroke();
    }
    const buildings = world.buildings || [];
    buildings.forEach((building) => {
      const [x0, y0, x1, y1, elevation] = building;
      const [px, py] = point(x0, y1);
      const w = Math.max(2, (x1 - x0) * scale);
      const h = Math.max(2, (y1 - y0) * scale);
      const shade = Math.max(0, Math.min(1, (elevation || 15) / 100));
      ctx.fillStyle = `rgba(${46 + shade * 15}, ${59 + shade * 12}, ${71 + shade * 10}, .86)`;
      ctx.fillRect(px, py, w, h);
      ctx.strokeStyle = "#516271";
      ctx.strokeRect(px, py, w, h);
    });
    (world.interference || []).forEach((item) => {
      if (!item.pos) return;
      const [px, py] = point(item.pos[0], item.pos[1]);
      const radius = Math.max(15, Math.min(width * 0.19, (worldW > 3000 ? 220 : 130) * scale));
      ctx.fillStyle = "#f0797710";
      ctx.strokeStyle = "#f079775d";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#f07977";
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    });
    const enabled = new Set([...document.querySelectorAll("#radioToggles input:checked")].map((input) => input.value));
    const dronePos = liveSnapshot?.pos;
    if (dronePos) {
      (world.stations || []).forEach((station) => {
        const link = findLinkForStation(station.id);
        if (!link || !enabled.has(link)) return;
        const [sx, sy] = point(station.x, station.y);
        const [dx, dy] = point(dronePos[0], dronePos[1]);
        const quality = liveSnapshot.links?.[link]?.snr_db ?? -5;
        ctx.strokeStyle = quality > 15 ? "#69d6b1a3" : quality > 0 ? "#f0b96899" : "#f0797785";
        ctx.lineWidth = 1.4;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(dx, dy);
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }
    const route = (world.waypoints || []).map((waypoint) => point(waypoint.x, waypoint.y));
    if (route.length) {
      ctx.strokeStyle = "#69d6b166";
      ctx.lineWidth = 1.3;
      ctx.setLineDash([4, 6]);
      ctx.beginPath();
      route.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.stroke();
      ctx.setLineDash([]);
      route.forEach(([x, y], index) => drawWaypoint(ctx, x, y, index + 1));
    }
    (world.stations || []).forEach((station) => {
      const [x, y] = point(station.x, station.y);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = station.kind === "cell" ? "#73b5fa" : "#69d6b1";
      ctx.fillRect(-5, -5, 10, 10);
      ctx.strokeStyle = "#101820";
      ctx.lineWidth = 2;
      ctx.strokeRect(-5, -5, 10, 10);
      ctx.restore();
      ctx.fillStyle = "#a9bbc8";
      ctx.font = "9px 'DM Mono', monospace";
      ctx.fillText(station.id, x + 8, y - 7);
    });
    if (dronePos) {
      const [x, y] = point(dronePos[0], dronePos[1]);
      ctx.fillStyle = "#69d6b127";
      ctx.beginPath();
      ctx.arc(x, y, 13, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#69d6b1";
      ctx.strokeStyle = "#10201c";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#aaffdf";
      ctx.font = "9px 'DM Mono', monospace";
      ctx.fillText("UAV", x + 10, y + 3);
    } else if (world.waypoints?.length) {
      const initial = world.waypoints[0];
      const [x, y] = point(initial.x, initial.y);
      ctx.fillStyle = "#69d6b1";
      ctx.beginPath();
      ctx.arc(x, y, 5.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    $("mapScale").textContent = `${Math.round(worldW)} × ${Math.round(worldH)} m · ${world.terrain || "scenario map"}`;
    canvas._project = (clientX, clientY) => {
      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      return [
        Math.max(0, Math.min(worldW, Math.round((px - offsetX) / scale))),
        Math.max(0, Math.min(worldH, Math.round((height - offsetY - py) / scale))),
      ];
    };
    canvas._pixel = (x, y) => point(x, y);
  }

  function makeScenarioMap(scenario) {
    return {
      size_m: scenario.env?.size_m || [2000, 2000],
      terrain: scenario.env?.terrain?.kind || "terrain",
      stations: (scenario.env?.stations || []).map((station) => ({
        id: station.id, x: station.pos[0], y: station.pos[1], kind: station.kind || "gcs",
      })),
      buildings: (scenario.env?.buildings || []).map((building) => [
        building.x0, building.y0, building.x1, building.y1, building.h,
      ]),
      waypoints: scenario.mission?.waypoints || [],
      interference: scenario.interference || [],
    };
  }

  function findLinkForStation(stationId) {
    return scenarioData?.links?.find((link) => (link.station_ids || []).includes(stationId))?.name;
  }

  function drawWaypoint(ctx, x, y, index) {
    ctx.fillStyle = "#69d6b1";
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#d6fff0";
    ctx.font = "8px 'DM Mono', monospace";
    ctx.fillText(String(index), x + 5, y - 4);
  }

  async function mapClick(event) {
    const point = $("mapCanvas")._project?.(event.clientX, event.clientY);
    if (!point || !scenarioData) return;
    const [x, y] = point;
    $("coordReadout").textContent = `Map coordinates: ${x} m E · ${y} m N`;
    if (mapMode === "waypoint") {
      const waypoints = scenarioData.mission.waypoints;
      const nearest = waypoints.reduce((best, waypoint, index) => {
        const [wx, wy] = $("mapCanvas")._pixel(waypoint.x, waypoint.y);
        const distance = Math.hypot(event.offsetX - wx, event.offsetY - wy);
        return distance < best.distance ? { distance, index } : best;
      }, { distance: Infinity, index: -1 });
      if (nearest.distance <= 18) {
        waypoints[nearest.index].x = x;
        waypoints[nearest.index].y = y;
        $("mapHint").textContent = `Waypoint ${nearest.index + 1} repositioned.`;
      } else {
        waypoints.push({ x, y, z: Number($("altitudeInput").value), speed: 0, hold_s: 0 });
        $("mapHint").textContent = "Waypoint appended to the active mission.";
      }
      drawMap();
    } else if (mapMode === "start") {
      scenarioData.env.home = [x, y];
      $("mapHint").textContent = "Home / start position updated.";
      drawMap();
    } else if (mapMode === "jammer") {
      const duration = Number($("durationInput").value);
      const jamMHz = Number($("jammerBandwidthInput").value);
      scenarioData.interference ||= [];
      scenarioData.interference.push({
        name: `ui_jammer_${scenarioData.interference.length + 1}`,
        kind: "jammer", t0: 0, t1: duration, pos: [x, y, 6],
        eirp_dbm: Number($("jammerStrengthInput").value),
        fc_hz: 5.8e9, bw_hz: jamMHz * 1e6, duty: 1,
      });
      $("mapHint").textContent = "Temporary jammer added to the scenario configuration.";
      drawMap();
    }
  }

  function paintLinkCards(snapshot) {
    const grid = $("linkGrid");
    const links = snapshot?.links;
    if (!links || !Object.keys(links).length) {
      grid.innerHTML = '<div class="empty-state">Configured links will show live values during a run.</div>';
      return;
    }
    grid.innerHTML = "";
    Object.entries(links).forEach(([name, link], index) => {
      const card = document.createElement("article");
      card.className = "link-card";
      const signal = Number(link.snr_db);
      const level = Number.isFinite(signal) ? Math.max(5, Math.min(100, (signal + 10) * 3)) : 0;
      const qualityClass = !Number.isFinite(signal) || signal < 0 ? "bad" : signal < 10 ? "warn" : "good";
      card.innerHTML = `
        <div class="link-title"><span class="link-name">${escapeHtml(titleCase(name))}</span><i class="link-dot ${qualityClass}"></i></div>
        <div class="link-stat"><span>SINR DL</span><strong>${formatNumber(link.snr_db, 1, " dB")}</strong></div>
        <div class="link-stat"><span>RSSI</span><strong>${formatNumber(link.rssi_dbm, 1, " dBm")}</strong></div>
        <div class="link-stat"><span>LATENCY</span><strong>—</strong></div>
        <div class="link-stat"><span>RADIO PWR</span><strong>${formatNumber(link.power_w, 2, " W")}</strong></div>
        <div class="link-stat"><span>LINK STATE</span><strong>${escapeHtml(titleCase(link.state))}</strong></div>
        <div class="link-bar"><span style="width:${level}%;background:${chartColors[index % chartColors.length]}"></span></div>`;
      grid.append(card);
    });
  }

  function paintDashboard(snapshot) {
    liveSnapshot = snapshot || liveSnapshot;
    if (!snapshot) return;
    const time = Math.max(0, Number(snapshot.t || 0));
    $("simTime").textContent = `${String(Math.floor(time / 60)).padStart(2, "0")}:${String(Math.floor(time % 60)).padStart(2, "0")}`;
    $("blindTime").innerHTML = `${Number(snapshot.blind_time_s || 0).toFixed(2)} <small>s</small>`;
    $("controlState").textContent = snapshot.ctrl_reliable ? "Available" : "Lost";
    $("videoState").textContent = snapshot.video_reliable ? "Available" : "Lost";
    $("controlState").classList.toggle("loss", !snapshot.ctrl_reliable);
    $("videoState").classList.toggle("loss", !snapshot.video_reliable);
    $("controlIndicator").className = `indicator ${snapshot.ctrl_reliable ? "ok" : "bad"}`;
    $("videoIndicator").className = `indicator ${snapshot.video_reliable ? "ok" : "bad"}`;
    const decision = snapshot.decision || {};
    $("controlRoute").textContent = (decision.control_links || []).join(" + ") || "—";
    $("videoRoute").textContent = Object.keys(decision.video_shares || {}).join(" + ") || "—";
    $("duplicateRoute").textContent = (decision.duplicate_links || []).join(" + ") || "None";
    $("batteryValue").textContent = `${Number(snapshot.energy_used_wh || 0).toFixed(1)} Wh used · ${Math.round(Number(snapshot.soc || 0) * 100)}%`;
    const socPercent = Math.max(0, Math.min(100, Number(snapshot.soc || 0) * 100));
    $("batteryFill").style.width = `${socPercent}%`;
    $("batteryFill").style.background = socPercent < 20 ? "var(--red)" : socPercent < 40 ? "var(--amber)" : "var(--accent)";
    const outage = decision.outage_prob || {};
    const primary = decision.control_links?.[0];
    const risk = primary ? Number(outage[primary]) : NaN;
    $("controlRisk").textContent = Number.isFinite(risk) ? `${(risk * 100).toFixed(1)}%` : "N/A";
    $("controlRisk").classList.toggle("high-risk", risk > 0.5);
    $("riskHorizon").textContent = decision.horizon_s ? `NEXT ${Number(decision.horizon_s).toFixed(1)}s` : "";
    $("riskNote").textContent = Number.isFinite(risk)
      ? "Control risk is the active policy’s primary-link outage probability. Video / joint probability is not emitted by this policy."
      : "The current policy does not emit a calibrated outage probability.";
    $("advisoryText").textContent = snapshot.blind
      ? "Blind state: neither control nor video reliability currently meets the simulator threshold."
      : snapshot.soc < 0.2
        ? "Low battery state of charge; simulator failsafe may alter the mission."
        : `Policy decision · ${decision.tag || "monitoring"} · flight mode ${snapshot.mode || "—"}`;
    $("decisionStatus").textContent = decision.tag ? String(decision.tag).replaceAll("_", " ").toUpperCase() : "LIVE";
    paintLinkCards(snapshot);
    drawMap();
  }

  function paintHistory(history = []) {
    window._history = history;
    paintChart($("blindChart"), history.map((row) => row.blind_time_s), "#69d6b1", 0);
    const names = Object.keys(liveSnapshot?.links || {});
    paintMultiChart($("sinrChart"), names.map((name) => history.map((row) => row.links?.[name]).filter(Number.isFinite)), names);
    $("timeAxis").textContent = history.length ? `${Number(history[history.length - 1].t).toFixed(1)}s elapsed` : "Awaiting samples";
  }

  function paintChart(canvas, values, color, minimum = null) {
    const context = prepareChart(canvas);
    if (!context) return;
    const { ctx, width, height } = context;
    chartGrid(ctx, width, height);
    if (values.length < 2) return;
    const finite = values.filter(Number.isFinite);
    if (!finite.length) return;
    const min = minimum ?? Math.min(...finite);
    const max = Math.max(min + 0.05, ...finite);
    drawLine(ctx, values, width, height, min, max, color);
  }

  function paintMultiChart(canvas, datasets, names) {
    const context = prepareChart(canvas);
    if (!context) return;
    const { ctx, width, height } = context;
    chartGrid(ctx, width, height);
    const finite = datasets.flat().filter(Number.isFinite);
    if (!finite.length) return;
    const min = Math.min(-10, ...finite);
    const max = Math.max(20, ...finite);
    datasets.forEach((values, index) => {
      if (values.length > 1) drawLine(ctx, values, width, height, min, max, chartColors[index % chartColors.length]);
    });
  }

  function prepareChart(canvas) {
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return null;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * ratio);
    canvas.height = Math.round(bounds.height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, bounds.width, bounds.height);
    return { ctx, width: bounds.width, height: bounds.height };
  }

  function chartGrid(ctx, width, height) {
    ctx.strokeStyle = "#2a3540";
    ctx.lineWidth = 1;
    for (let y = 16; y < height - 8; y += 28) {
      ctx.beginPath();
      ctx.moveTo(10, y);
      ctx.lineTo(width - 9, y);
      ctx.stroke();
    }
  }

  function drawLine(ctx, values, width, height, min, max, color) {
    const plotH = height - 25;
    const plotW = width - 20;
    ctx.beginPath();
    let started = false;
    values.forEach((value, index) => {
      if (!Number.isFinite(value)) { started = false; return; }
      const x = 10 + (index / Math.max(1, values.length - 1)) * plotW;
      const y = 8 + (1 - Math.max(0, Math.min(1, (value - min) / (max - min)))) * plotH;
      if (!started) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      started = true;
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.7;
    ctx.stroke();
  }

  function paintStatus(data) {
    if (!data) return;
    mapStatus = data.snapshot || mapStatus;
    const status = loadedResult && data.status === "idle" ? "finished" : (data.status || "idle");
    const badge = $("runState");
    badge.textContent = loadedResult && status === "finished" ? "LOADED RESULT" : status.toUpperCase();
    badge.className = `status-pill ${status}`;
    $("progressBar").style.width = `${Math.round((data.progress || 0) * 100)}%`;
    $("progressPercent").textContent = `${Math.round((data.progress || 0) * 100)}%`;
    $("progressLabel").textContent = data.message || "Ready when you are";
    $("connectionLabel").textContent = socket?.readyState === WebSocket.OPEN ? "Live connection" : "Reconnecting";
    const active = ["starting", "running", "paused"].includes(status);
    $("startButton").disabled = active || !initialized;
    $("batchButton").disabled = active || !initialized;
    $("pauseButton").disabled = !active || data.batch;
    $("stopButton").disabled = !active;
    $("pauseButton").textContent = data.paused ? "▶ Resume" : "Ⅱ Pause";
    $("pauseButton").dataset.action = data.paused ? "resume" : "pause";
    if (data.snapshot) paintDashboard(data.snapshot);
    if (data.history) paintHistory(data.history);
    if (data.logs) $("consoleLog").textContent = data.logs.join("\n") || "UAV Link Lab ready.";
    const log = $("consoleLog");
    log.scrollTop = log.scrollHeight;
    if (data.evaluation) {
      const fingerprint = `${data.result_dir}:${data.evaluation.analysis || ""}`;
      if (fingerprint !== shownEvaluation) {
        shownEvaluation = fingerprint;
        showEvaluation(data.evaluation, data.metrics, data.result_dir);
      }
    }
    if (data.error) showToast(data.error, true);
  }

  function formatNumber(value, digits = 1, suffix = "") {
    if (value === null || value === undefined || value === "") return "—";
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(digits)}${suffix}` : "—";
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
  }

  function showToast(message, error = false) {
    $("progressLabel").textContent = message;
    $("progressLabel").style.color = error ? "var(--red)" : "var(--accent)";
  }

  async function startRun(batch) {
    try {
      const config = currentFormConfig();
      const validation = await request("/api/dry-run", { method: "POST", body: JSON.stringify({ config, batch }) });
      if (!validation.valid) throw new Error("Configuration was not accepted");
      loadedResult = false;
      await request("/api/runs", { method: "POST", body: JSON.stringify({ config, batch }) });
      showToast(`Validated · ${validation.jobs} simulation job(s)`);
    } catch (error) {
      showToast(error.message, true);
    }
  }

  async function controlRun(action) {
    try {
      await request(`/api/control/${action}`, { method: "POST", body: "{}" });
    } catch (error) {
      showToast(error.message, true);
    }
  }

  async function runDryCheck() {
    try {
      const result = await request("/api/dry-run", {
        method: "POST", body: JSON.stringify({ config: currentFormConfig(), batch: false }),
      });
      showToast(`Configuration valid · ${result.scenario} · ${result.duration_s}s · dt ${result.dt}s`);
      appendLog(`VALID · ${result.scenario}, ${result.jobs} job`);
    } catch (error) {
      showToast(error.message, true);
      appendLog(`INVALID · ${error.message}`);
    }
  }

  function appendLog(line) {
    const log = $("consoleLog");
    log.textContent = `${log.textContent}\n${new Date().toLocaleTimeString()}  ${line}`.split("\n").slice(-100).join("\n");
    log.scrollTop = log.scrollHeight;
  }

  function showEvaluation(evaluation, metrics, resultDir) {
    const target = $("resultsContent");
    if (evaluation.analysis) {
      const root = resultDir || evaluation.report?.split("/evaluation/")[0];
      const analysisPath = evaluation.analysis;
      fetch(`/api/results/analysis?path=${encodeURIComponent(analysisPath)}`)
        .then((response) => response.ok ? response.json() : Promise.reject(new Error("Evaluation analysis unavailable")))
        .then((analysis) => renderAnalysis(target, analysis, evaluation, root, metrics))
        .catch((error) => { target.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`; });
    } else if (metrics) {
      showMetricResult(metrics, target, resultDir);
    }
    refreshResults();
  }

  function renderAnalysis(target, analysis, evaluation, resultDir, liveMetrics) {
    const labels = analysis.labels || Object.keys(analysis.summary || {});
    const summary = analysis.summary || {};
    const proposedLabel = labels.find((label) => /^pred|^cost_predictive/.test(label)) || labels[0];
    const metricRows = labels.map((label) => {
      const row = summary[label] || {};
      return `<tr><td>${escapeHtml(label)}</td><td>${analysis.n_runs?.[label] ?? "—"}</td><td>${formatNumber(row.blind_time_s?.mean, 2)}</td><td>${formatNumber(row.ctrl_lat_p99_ms?.mean, 1)}</td><td>${formatNumber(row.video_quality_mean?.mean, 3)}</td><td>${formatNumber(row.comm_power_mean_w?.mean, 2)}</td></tr>`;
    }).join("");
    const selected = proposedLabel && summary[proposedLabel];
    const kpis = [
      ["BLIND TIME", formatNumber(selected?.blind_time_s?.mean, 2, " s")],
      ["CTRL P99", formatNumber(selected?.ctrl_lat_p99_ms?.mean, 1, " ms")],
      ["VIDEO QUALITY", formatNumber(selected?.video_quality_mean?.mean, 3)],
      ["COMM POWER", formatNumber(selected?.comm_power_mean_w?.mean, 2, " W")],
    ];
    const files = [
      ["Evaluation report", evaluation.report],
      ["Report Markdown", `${resultDir}/evaluation/report.md`],
      ["Metrics CSV", `${resultDir}/evaluation/runs_metrics.csv`],
      ["Headline table", `${resultDir}/evaluation/table_headline.csv`],
      ["Paired comparisons", `${resultDir}/evaluation/paired_comparisons_all.csv`],
      ...((evaluation.figures || []).map((path, index) => [`Figure ${index + 1}`, path])),
    ].filter(([, path]) => path);
    const blindValues = labels.map((label) => summary[label]?.blind_time_s?.mean).filter(Number.isFinite);
    const maxBlind = Math.max(0.01, ...blindValues);
    const blindBars = labels.map((label, index) => {
      const value = summary[label]?.blind_time_s?.mean;
      return Number.isFinite(value) ? `<div class="policy-bar-row"><span>${escapeHtml(label)}</span><div class="policy-bar-track"><span style="width:${Math.max(1, 100 * value / maxBlind)}%;background:${chartColors[index % chartColors.length]}"></span></div><strong class="policy-bar-value">${Number(value).toFixed(2)}s</strong></div>` : "";
    }).join("");
    target.innerHTML = `
      <div class="results-summary">${kpis.map(([label, value]) => `<div class="result-kpi"><span>${label}</span><strong>${value}</strong></div>`).join("")}</div>
      <div class="route-label">BLIND-TIME BY POLICY · LOWER IS BETTER</div><div class="policy-bars">${blindBars || '<div class="empty-state">No policy comparison data.</div>'}</div>
      <div style="overflow:auto"><table class="data-table"><thead><tr><th>POLICY</th><th>N</th><th>BLIND S</th><th>CTRL P99 MS</th><th>VIDEO Q</th><th>RADIO W</th></tr></thead><tbody>${metricRows}</tbody></table></div>
      ${labels.length > 1 && Math.min(...labels.map((label) => analysis.n_runs?.[label] ?? 0)) < 5 ? '<div class="micro-copy">Descriptive paired-seed comparison only; the default evaluation requires at least five matched trials for inferential tests.</div>' : ""}
      <div class="result-actions">${files.map(([label, path]) => `<a href="/api/results/file?path=${encodeURIComponent(path)}" target="_blank" rel="noopener">${escapeHtml(label)} ↗</a>`).join("")}</div>
      ${(evaluation.figures || []).slice(0, 1).map((path) => `<img class="eval-figure" alt="Evaluation figure" src="/api/results/file?path=${encodeURIComponent(path)}">`).join("")}
      ${liveMetrics ? `<div class="empty-state">Latest trial: ${escapeHtml(liveMetrics.run_id || "completed")}</div>` : ""}`;
  }

  function showMetricResult(metrics, target, resultDir, report = null) {
    const fields = [
      ["Blind time", formatNumber(metrics.blind_time_s, 2, " s")],
      ["Control p99", formatNumber(metrics.ctrl_lat_p99_ms, 1, " ms")],
      ["Video quality", formatNumber(metrics.video_quality_mean, 3)],
      ["Radio energy", formatNumber(metrics.energy_comm_wh, 3, " Wh")],
    ];
    target.innerHTML = `<div class="results-summary">${fields.map(([name, value]) => `<div class="result-kpi"><span>${name.toUpperCase()}</span><strong>${value}</strong></div>`).join("")}</div>
      <div class="result-actions"><a href="/api/results/file?path=${encodeURIComponent(`${resultDir}/metrics.json`)}">Run metrics JSON ↓</a>${report ? `<a href="/api/results/file?path=${encodeURIComponent(report)}" target="_blank" rel="noopener">Evaluation report ↗</a>` : ""}</div>`;
  }

  async function refreshResults() {
    try {
      const response = await request("/api/results");
      previousResults = response.results || [];
      const selector = $("previousRunSelect");
      const oldValue = selector.value;
      selector.innerHTML = '<option value="">Choose a saved run…</option>';
      previousResults.forEach((item, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${item.scenario} · ${item.policy} · ${item.run_id}`;
        selector.append(option);
      });
      if ([...selector.options].some((option) => option.value === oldValue)) selector.value = oldValue;
    } catch (error) {
      showToast(`Unable to list previous results: ${error.message}`, true);
    }
  }

  async function loadPreviousResult() {
    const item = previousResults[Number($("previousRunSelect").value)];
    if (!item) return;
    if (item.analysis) {
      try {
        const analysis = await request(`/api/results/analysis?path=${encodeURIComponent(item.analysis)}`);
        const resultDir = item.report?.split("/evaluation/")[0] || item.path;
        renderAnalysis($("resultsContent"), analysis, {
          analysis: item.analysis, report: item.report, figures: item.figures || [],
        }, resultDir, item.metrics);
      } catch (error) {
        showToast(`Could not open evaluation results: ${error.message}`, true);
        showMetricResult(item.metrics, $("resultsContent"), item.path, item.report);
      }
    } else {
      showMetricResult(item.metrics, $("resultsContent"), item.path, item.report);
    }
    const metrics = item.metrics;
    loadedResult = true;
    $("scenarioName").textContent = titleCase(metrics.scenario);
    $("simTime").textContent = `${String(Math.floor(Number(metrics.sim_time_s || 0) / 60)).padStart(2, "0")}:${String(Math.floor(Number(metrics.sim_time_s || 0) % 60)).padStart(2, "0")}`;
    $("blindTime").innerHTML = `${formatNumber(metrics.blind_time_s, 2)} <small>s</small>`;
    $("controlRoute").textContent = "Not stored in summary";
    $("videoRoute").textContent = "Not stored in summary";
    $("duplicateRoute").textContent = "Not stored in summary";
    $("runState").textContent = "LOADED RESULT";
    $("runState").className = "status-pill finished";
  }

  function populatePolicyOptions(policies) {
    const select = $("policySelect");
    select.replaceChildren();
    policies.forEach((policy) => {
      const option = document.createElement("option");
      option.value = policy;
      option.textContent = policy;
      select.append(option);
    });
  }

  async function refreshConfigs() {
    try {
      const response = await request("/api/configs");
      const select = $("savedConfigSelect");
      select.innerHTML = '<option value="">Saved configs…</option>';
      response.configs.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        select.append(option);
      });
    } catch (error) {
      showToast(`Unable to load config list: ${error.message}`, true);
    }
  }

  function applyConfig(config) {
    if (!config || typeof config !== "object" || !config.scenario) throw new Error("JSON config must include a scenario object");
    liveSnapshot = null;
    mapStatus = null;
    scenarioData = structuredClone(config.scenario);
    $("scenarioName").textContent = titleCase(scenarioData.name);
    $("scenarioSelect").value = scenarioData.name;
    $("policySelect").value = config.policy || "Predictive · cost";
    $("durationInput").value = config.duration_s ?? scenarioData.duration_s ?? 60;
    $("dtInput").value = config.dt ?? scenarioData.dt ?? 0.05;
    $("seedInput").value = config.seed ?? 2026;
    $("seedsInput").value = config.seeds ?? 3;
    $("radioPowerInput").value = config.radio_power_limit_w ?? scenarioData.budget?.comm_power_w ?? 8;
    $("speedInput").value = scenarioData.mission?.default_speed_mps ?? 12;
    $("altitudeInput").value = scenarioData.mission?.takeoff_alt_m ?? 50;
    $("rainInput").value = scenarioData.weather?.timeline?.[0]?.rain_mm_h ?? 0;
    $("windInput").value = scenarioData.weather?.timeline?.[0]?.wind_mps ?? 0;
    const predictive = config.predictive_handover || {};
    $("horizonInput").value = predictive.horizon_s ?? 8;
    $("marginInput").value = predictive.snr_margin_db ?? 3;
    $("latencyWeightInput").value = predictive.ctrl_latency_weight ?? 10;
    $("videoWeightInput").value = predictive.video_quality_weight ?? 1;
    $("powerWeightInput").value = predictive.power_weight ?? .05;
    $("switchCostInput").value = predictive.switch_cost ?? .5;
    $("ctrlThresholdInput").value = config.evaluation?.control_latency_ms ?? 150;
    $("videoThresholdInput").value = config.evaluation?.video_quality_min ?? 0;
    $("jammerStrengthInput").value = config.map_edits?.jammer_eirp_dbm ?? 33;
    $("jammerBandwidthInput").value = config.map_edits?.jammer_bw_mhz ?? 40;
    populateRadioToggles();
    const enabled = new Set(config.enabled_links || scenarioData.links.map((link) => link.name));
    document.querySelectorAll("#radioToggles input").forEach((input) => { input.checked = enabled.has(input.value); });
    drawMap();
  }

  async function saveConfig() {
    try {
      const name = prompt("Name this experiment configuration");
      if (!name) return;
      const config = currentFormConfig();
      const response = await request("/api/configs", { method: "POST", body: JSON.stringify({ name, config }) });
      await refreshConfigs();
      $("savedConfigSelect").value = response.name;
      showToast(`Saved config · ${response.name}`);
    } catch (error) {
      showToast(error.message, true);
    }
  }

  function exportConfig() {
    try {
      const config = currentFormConfig();
      const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${scenarioData.name}-uavlink-config.json`;
      link.click();
      URL.revokeObjectURL(link.href);
      showToast("Configuration JSON exported");
    } catch (error) {
      showToast(error.message, true);
    }
  }

  async function loadConfig() {
    const name = $("savedConfigSelect").value;
    if (!name) return;
    try {
      applyConfig(await request(`/api/configs/${encodeURIComponent(name)}`));
      showToast(`Loaded config · ${name}`);
    } catch (error) {
      showToast(error.message, true);
    }
  }

  function resetDefaults() {
    $("policySelect").value = "Predictive · cost";
    $("seedInput").value = 2026;
    $("seedsInput").value = 3;
    $("horizonInput").value = 8;
    $("marginInput").value = 3;
    $("latencyWeightInput").value = 10;
    $("videoWeightInput").value = 1;
    $("powerWeightInput").value = 0.05;
    $("switchCostInput").value = 0.5;
    $("ctrlThresholdInput").value = 150;
    $("videoThresholdInput").value = 0;
    $("jammerStrengthInput").value = 33;
    $("jammerBandwidthInput").value = 40;
    if ($("scenarioSelect").value) loadScenario($("scenarioSelect").value);
    showToast("Controls reset to default values");
  }

  function setMapMode(button) {
    mapMode = button.dataset.mapMode;
    document.querySelectorAll("[data-map-mode]").forEach((item) => item.classList.toggle("selected", item === button));
    const labels = { inspect: "Select a map tool to edit the mission.", waypoint: "Click to append a mission waypoint.", start: "Click to reposition home / mission start.", jammer: "Click to place a temporary jammer." };
    $("mapHint").textContent = labels[mapMode];
  }

  function connectSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
    socket.addEventListener("open", () => { $("connectionLabel").textContent = "Live connection"; });
    socket.addEventListener("message", (event) => {
      try {
        const data = JSON.parse(event.data);
        paintStatus(data);
      } catch (error) {
        showToast(`Invalid live update: ${error.message}`, true);
      }
    });
    socket.addEventListener("close", () => {
      $("connectionLabel").textContent = "Reconnecting";
      setTimeout(connectSocket, 1500);
    });
    socket.addEventListener("error", () => $("connectionLabel").textContent = "Connection error");
  }

  function bindEvents() {
    $("scenarioSelect").addEventListener("change", (event) => loadScenario(event.target.value).catch((error) => showToast(error.message, true)));
    $("mapCanvas").addEventListener("click", mapClick);
    document.querySelectorAll("[data-map-mode]").forEach((button) => button.addEventListener("click", () => setMapMode(button)));
    $("startButton").addEventListener("click", () => startRun(false));
    $("batchButton").addEventListener("click", () => startRun(true));
    $("pauseButton").addEventListener("click", (event) => controlRun(event.currentTarget.dataset.action || "pause"));
    $("stopButton").addEventListener("click", () => controlRun("stop"));
    $("dryRunButton").addEventListener("click", runDryCheck);
    $("selfTestButton").addEventListener("click", async () => {
      try {
        const result = await request("/api/self-test", { method: "POST", body: "{}" });
        appendLog(`SELF-TEST · ${JSON.stringify(result.checks)}`);
        showToast(result.passed ? "Self-test passed" : "Self-test failed", !result.passed);
      } catch (error) { showToast(error.message, true); }
    });
    $("saveConfigButton").addEventListener("click", saveConfig);
    $("exportConfigButton").addEventListener("click", exportConfig);
    $("loadConfigButton").addEventListener("click", loadConfig);
    $("resetButton").addEventListener("click", resetDefaults);
    $("refreshHistory").addEventListener("click", refreshResults);
    $("refreshResults").addEventListener("click", refreshResults);
    $("openRunButton").addEventListener("click", loadPreviousResult);
    $("clearLogButton").addEventListener("click", () => { $("consoleLog").textContent = ""; });
    $("importConfigInput").addEventListener("change", async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      try { applyConfig(JSON.parse(await file.text())); showToast(`Imported ${file.name}`); }
      catch (error) { showToast(`Could not import config: ${error.message}`, true); }
      event.target.value = "";
    });
    $("helpButton").addEventListener("click", () => $("helpDialog").showModal());
    $("closeHelp").addEventListener("click", () => $("helpDialog").close());
    $("closeHelpButton").addEventListener("click", () => $("helpDialog").close());
    window.addEventListener("resize", () => { drawMap(); paintHistory(window._history || []); });
  }

  async function initialize() {
    bindEvents();
    const meta = await request("/api/scenarios");
    const scenarioSelect = $("scenarioSelect");
    meta.scenarios.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = titleCase(name);
      scenarioSelect.append(option);
    });
    populatePolicyOptions(meta.policies);
    $("policySelect").value = "Predictive · cost";
    await loadScenario(meta.scenarios.includes("urban_canyon") ? "urban_canyon" : meta.scenarios[0]);
    await Promise.all([refreshConfigs(), refreshResults()]);
    initialized = true;
    connectSocket();
    const status = await request("/api/status");
    paintStatus(status);
  }

  initialize().catch((error) => {
    $("connectionLabel").textContent = "Initialization error";
    $("progressLabel").textContent = error.message;
    $("progressLabel").style.color = "var(--red)";
  });
})();
