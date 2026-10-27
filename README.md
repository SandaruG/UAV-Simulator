# UAV radio research and study runner

Use [`exhaustive_experiment.py`](./exhaustive_experiment.py) to plan and run multi-seed technology/policy studies. Start with [`study_demo.json`](./study_demo.json) and the [study runner guide](./docs/study-runner.md). It supports the fast `uavlink_sim` backend and the native high-fidelity `drone_multilink` backend; the UI is not involved.

```bash
python3 exhaustive_experiment.py --list-techs
python3 exhaustive_experiment.py --config study_demo.json --dry-run --out results/demo
python3 exhaustive_experiment.py --config study_demo.json --backend high_fidelity --out results/demo-hifi
python3 exhaustive_experiment.py --config study_demo.json --backend high_fidelity --out results/demo-hifi --resume
```

# radioenv — radio environment map for drone link-handover research

A standalone Python module (numpy + scipy only) that builds a **digital twin of an area** — terrain, buildings,
vegetation, cell towers, a LEO constellation, weather and interference — and answers one question for any other
module:

```python
from radioenv import RadioEnvironment, RadioEnvConfig, presets
env = RadioEnvironment(presets.get("suburban", seed=3, weather="light_rain"))
q = env.query("video_58", (x, y, z), t)       # -> LinkQuality at that point and time
```

`(x, y)` are metres East/North of the map centre, `z` is metres above MSL (same datum as the terrain), `t` is seconds
since `area.start_time_utc`. Queries are **stateless and deterministic**: same config + seed + (x, y, z, t) ⇒ same
answer, in any order, in any process. Pass `(N,3)` arrays for vectorised queries.

## What is in the world

| Layer | Implementation |
|---|---|
| Terrain | synthetic fractal DEM, or load SRTM `.hgt`, ESRI ASCII `.asc`, GeoTIFF (needs `rasterio`), `.npy` |
| Buildings | synthetic city (blocks, lots, density-driven heights, sub-centres) or **OSM GeoJSON / Overpass JSON** footprints with `height` / `building:levels` |
| Vegetation | synthetic canopy field or OSM forest/park polygons; seasonal leaf factor from the start date |
| Clutter | open / water / vegetation / suburban / urban raster derived from the above (sets ground reflection, delay spread, man-made noise) |
| Cellular | hex / listed / **OpenCelliD CSV** layouts; 3-sector sites; multi-band (LTE 1.8 GHz + NR 3.5 GHz by default); 3GPP sector antenna with downtilt; diurnal + per-cell load; DL and UL SINR with inter-cell interference; RSRP/RSRQ, neighbour lists, serving-cell override |
| Direct links | GCS ↔ drone at any frequency (915 MHz C2, 5.8 GHz video, …) with antenna patterns |
| LEO satellite | Walker-delta shells (Starlink-like defaults) with J2 drift, or real TLEs (needs `sgp4`); serving-satellite selection every 15 s slot with horizon/obstruction screening, handover interruption, Doppler, sun-in-beam noise, bent-pipe latency |
| Weather | advecting convective rain cells + stratiform background (ITU-R P.838 style k/α), cloud liquid water and fog (P.840), oxygen/water-vapour absorption, tropospheric scintillation |
| Interference | bursty diurnal Wi-Fi, other drones' video/C2 transmitters on orbit paths, **scripted jammers** (noise / narrowband / swept / pulsed, on/off windows, moving), ITU-R P.372 man-made noise |

### Propagation and multipath

Path analysis along the straight A→B profile through the raster: line-of-sight, Fresnel clearance, **multiple
knife-edge diffraction** (Deygout with one edge per crest/roof), building penetration, Weissberger foliage loss,
two-ray ground reflection with clutter-dependent specularity. Cellular links use a *hybrid* model: geometric LOS picks
the 3GPP UMa/RMa LOS or NLOS branch (`cellular.model` can also be `geometric` or pure statistical `3gpp`).
Spatially-correlated log-normal shadowing (decorrelating with distance and altitude, shared across links to the same
node) plus a deterministic plane-wave small-scale fading process (Rician with K from the geometry), so a hovering drone
still fades and a moving one gets physically spaced fades.

> **Multipath is statistical, not ray-traced.** You get delay spread, Rician K, two-ray ground bounce and Doppler-correct
> fading — not specular reflections off individual building walls.

## LinkQuality (what every query returns)

`rssi_dbm, signal_dbm, noise_dbm, interference_dbm, snr_db, sinr_db, pathloss_db, los, shadow_db, fading_db, per,
throughput_mbps, latency_ms, latency_p99_ms, available, ctrl_reliable, video_usable, quality (0–1), outage_prob
(P[SINR < threshold] under fast fading), margin_db`, plus `extra` per link kind — satellite: `sat_index, sat_elev_deg,
range_km, n_visible, doppler_khz, rain_db, cloud_db, handover_interruption, slot_time_left_s`; cellular: `serving_cell,
rsrp_dbm, rsrq_db, cell_load, neighbor_cells, neighbor_rsrp_dbm, ho_margin_db`; direct: `dist_m, fresnel_clearance,
diffraction_db, foliage_db, penetration_db, k_factor_db, delay_spread_ns`.

Convenience flags: `ctrl_reliable` = PER ≤ `reliable_per` and latency (mean and p99) within `latency_limit_ms`;
`video_usable` = throughput ≥ `min_video_mbps` and PER ≤ 10 %. All thresholds live in each `LinkSpec`.

## Everything is configurable

One JSON-serialisable `RadioEnvConfig` tree (`cfg.to_json()`, `RadioEnvConfig.from_json`, `cfg.replace(**{"weather.scenario": "heavy_rain"})`).
Add links with `env.add_link(LinkSpec(...))`; add jammers via `cfg.interference.jammers`. Presets: `demo`, `dense_urban`,
`suburban`, `rural_sparse`, `mountain_valley`.

```
python -m radioenv describe --preset suburban
python -m radioenv config --preset rural_sparse > my.json      # edit, then:  --config my.json
python -m radioenv query --link sat_ku --x 0 --y 0 --z 100 --agl --t 30
python -m radioenv map --link cell_lte1800 --out cov.png --z-agl 100
python examples/demo_flight.py out/                            # flight + blind-time + figures
python tests/test_radioenv.py                                  # 22 checks
```

Useful hooks for the handover study: `env.query_track(positions, times)` evaluates every link along a trajectory;
`fading="none"|"shadow"|"full"` separates mean from instantaneous quality (labels vs. what a modem would see);
`outage_prob`, `margin_db`, `slot_time_left_s`, and the cellular neighbour list give ground-truth features/labels for
a predictor; `query(..., serving_override=...)` lets a handover logic hold a cell instead of always taking the best.

## Verification — what was actually checked

Done and passing (`tests/`, 22 checks): free-space budget equals the analytic link budget to <0.05 dB; −20 dB/decade
slope; knife-edge diffraction equals ITU-R P.526 J(v) to 0.05 dB for clear/grazing/obstructed edges; two-edge case
matches a hand-computed Deygout result (this caught and fixed a real bug: raster cells of one roof were being counted
as separate edges, doubling the loss); ITU rain/cloud magnitudes; determinism across instances and query order;
single-point == batch; jammer on/off windows, frequency selectivity and pulse duty cycle; rain hurts Ku not 915 MHz;
satellite count/elevation/slot-interruption/latency-floor; satellite blocked in street canyons; aerial downlink SINR
degrading with altitude (the known aerial-UE interference effect); OSM GeoJSON and ASCII-DEM loaders; all presets.

**Not verified / be careful before publishing numbers:**
- The sandbox had no route to OSM/SRTM/OpenCelliD/TLE sources, so `fetch_overpass`, the OpenCelliD loader, GeoTIFF and
  TLE paths were never run on real data (GeoJSON + ASCII DEM were, with synthetic files).
- The default world is **synthetic**. It is plausible, not calibrated to any city or operator. 3GPP path-loss
  coefficients were written from memory — check them against TR 38.901 / 36.777 before quoting absolute dB.
- The satellite shells are an analytic Walker constellation, not the real Starlink/OneWeb ephemeris. Terminal and
  satellite parameters (EIRP, G/T, beam sharing) are assumptions.
- Cellular uplink inter-cell interference is simplified (load-driven noise rise plus external sources), and
  man-made noise is reduced by up to 6 dB with altitude by assumption.
- In the shipped presets cellular is rarely the bottleneck at 100 m AGL: the drone has line of sight to many cells and
  a 1.8 GHz uplink budget closes easily. Coverage holes come from terrain, NLOS at low altitude, sparse/partial sites
  (`coverage_fraction`, `cellular.sites`), downlink interference, and power limits — tune those for marginal regimes.
- The ROS 2 node (`ros2/radioenv_node.py`) was only syntax-checked; no ROS 2 install was available.

Timing (one CPU core): build 0.1–6 s; warm single-point query 3–26 ms; 1000-point batch ≈ 0.15 s (direct),
0.6 s (satellite), 1.8 s (cellular, 19 sites).

## Layout

```
radioenv/  config.py geo.py terrain.py propagation.py weather.py interference.py fading.py
           cellular.py satellite.py links.py quality.py environment.py presets.py __main__.py
examples/demo_flight.py     tests/test_radioenv.py     ros2/radioenv_node.py     docs/*.png
```

## Reproducible handover experiments

`run_experiment.py` is the single command-line entry point for the simulator/evaluator
pipeline. It uses `uavlink_sim` for scenario generation, policies, seeded runs, and
logs, then passes the batch directly to `uavlink_eval` for CSV/JSON/Markdown tables
and figures. The `cost`, `predictive`, and `predictive_handover` policy names select
the Part 3 `HandoverController` through the `uavlink_sim.Policy` contract; built-in
policies such as `reactive`, `bonding`, and `cost_predictive` remain available.

```bash
# One scenario/policy/seed
python3 run_experiment.py --scenario urban_canyon --policy cost --seed 8 --out results/study1

# Eight Monte-Carlo seeds for one policy
python3 run_experiment.py --scenario urban_canyon --policy cost --seeds 8 --seed 2026 --out results/study1

# Batch policies and parameter sweeps (the existing batch_example.json schema)
python3 run_experiment.py --batch --config batch_example.json --out results/full_study --workers 4

# Validate without running, or run quick cross-module smoke checks
python3 run_experiment.py --dry-run --scenario urban_canyon --policy predictive_handover
python3 run_experiment.py --self-test
```

Batch JSON can use the existing `uavlink_sim` schema directly, with optional
`predictive_handover` and `evaluation` sections. For example:

```json
{
  "scenario": "urban_canyon",
  "master_seed": 2026,
  "n_trials": 8,
  "baseline": "reactive",
  "overrides": {"dt": 0.05, "duration_s": 120},
  "policies": [
    {"label": "reactive", "name": "reactive"},
    {"label": "pred", "name": "predictive_handover"}
  ],
  "save_ticks": "all",
  "predictive_handover": {
    "policy": "cost",
    "config": {"decision_period_s": 0.3, "safety": {"horizon_s": 4.0}}
  },
  "evaluation": {"plots": {"enabled": true}}
}
```

The policy adapter maps Part 4 link measurements and histories into Part 3
`LinkSample`/`DroneState` inputs and maps the resulting assignments back to Part 4
control, video, and warm-link decisions. Geometry-based prediction is disabled by default because the Part 4 policy contract
does not expose radio endpoint coordinates. To enable it, configure an endpoint for
every simulator link under `predictive_handover.link_specs`, for example
`{"wifi_5g": {"endpoint": [0, 0, 0]}, "lte": {"endpoint": [200, 0, 0]}}`,
and set `predictive_handover.config.predictor.use_geometry` to `true`. Each run logs
under `OUT/logs/`; evaluation artifacts are under `OUT/evaluation/`.

`radioenv` and `drone_multilink.py` remain independent model implementations with
different world, radio, and flight-state contracts from `uavlink_sim`. The unified
runner currently validates/smoke-tests those modules but does not substitute their
physics into Part 4. `uavlink_eval` consumes Part 4's run-log schema. Replacing the
Part 4 environment or drone with the Part 1/2 implementations requires an explicit
physics/backend adapter; treating the models as interchangeable would silently
change the experiment rather than just connect their interfaces.

## Local research UI

Start the visual experiment console from the repository root:

```bash
python3 ui/server.py
```

Open [http://localhost:8000](http://localhost:8000/). The server binds to
`127.0.0.1` and keeps experiment state in memory; run logs, batch summaries,
evaluation reports, figures, and saved JSON configurations are written under
`results/uavlink-ui/` and `results/ui-configs/`. Install the repository
requirements first if FastAPI or Uvicorn is not already available:

```bash
python3 -m pip install -r requirements.txt
```

Choose a scenario and predictive, reactive, or bonding policy, adjust the
mission and link parameters, then start a single simulation or a paired batch.
The map supports click-to-add waypoint, start/home position, and jammer edits.
Runs can be paused, resumed, stopped, and monitored over a WebSocket. Configs
can be saved locally, imported, or exported as JSON. Completed runs are evaluated
through `uavlink_eval`; the result panel links to reports, metric CSVs, tables,
and figures. Batch comparisons include the selected policy plus reactive and
bonding baselines using the same trial seeds.

The interface deliberately reflects the actual simulator boundary: its
selectable scenarios and radio telemetry are provided by `uavlink_sim`. The
shipped presets do not configure a satellite radio, so a satellite coverage
footprint is not drawn for them. `radioenv` and `drone_multilink.py` remain
separate models; the UI does not claim to combine their physics with the Part 4
simulator. Per-radio latency, calibrated video outage probability, and joint
control/video outage probability are not exposed by the simulator policy
snapshot and are marked unavailable rather than synthesized.
