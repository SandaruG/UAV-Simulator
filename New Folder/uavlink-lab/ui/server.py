#!/usr/bin/env python3
"""Local web interface for the UAV multi-link handover simulator."""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import json
import math
import re
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uavlink_eval
import uavlink_sim
import predictive_handover
import run_experiment
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

UI_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
UI_RESULTS_DIR = RESULTS_DIR / "uavlink-ui"
CONFIG_DIR = RESULTS_DIR / "ui-configs"
POLICY_CHOICES = {
    "Predictive · cost": ("predictive_handover", "cost"),
    "Predictive · threshold": ("predictive_handover", "threshold"),
    "Predictive · rule-based": ("predictive_handover", "rule_based"),
    "Reactive": ("reactive", None),
    "Bonding": ("bonding", None),
    "Reference cost-predictive": ("cost_predictive", None),
}

app = FastAPI(title="UAV Link Lab", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")


class RunRequest(BaseModel):
    config: dict[str, Any]
    batch: bool = False


class ConfigRequest(BaseModel):
    config: dict[str, Any]
    name: str


state: dict[str, Any] = {
    "status": "idle", "progress": 0.0, "message": "Ready", "logs": [],
    "snapshot": None, "history": [], "metrics": None, "result_dir": None,
    "error": None, "paused": False, "batch": False, "task": None,
    "pause_event": None, "stop_requested": False,
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _finite_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    state["logs"].append(f"{stamp}  {message}")
    state["logs"] = state["logs"][-400:]


def _result_path(path: str) -> Path:
    candidate = (ROOT / path).resolve()
    if not candidate.is_relative_to(RESULTS_DIR.resolve()) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")
    return candidate


def _make_policy(label: str, predictive: dict) -> dict:
    if label not in POLICY_CHOICES:
        raise ValueError(f"Unknown policy: {label}")
    name, native_policy = POLICY_CHOICES[label]
    if native_policy is None:
        return {"name": name}
    config = {
        "decision_period_s": float(predictive.get("decision_period_s", 0.3)),
        "safety": {
            "horizon_s": float(predictive.get("horizon_s", 8.0)),
            "snr_margin_db": float(predictive.get("snr_margin_db", 3.0)),
            "power_budget_w": float(predictive.get("power_budget_w", 12.0)),
        },
        "weights": {
            "ctrl_latency": float(predictive.get("ctrl_latency_weight", 10.0)),
            "video_quality": float(predictive.get("video_quality_weight", 1.0)),
            "power": float(predictive.get("power_weight", 0.05)),
            "switch_soft": float(predictive.get("switch_cost", 0.5)),
        },
    }
    return {"name": name, "params": {"policy_name": native_policy, "handover_config": config}}


def _prepare_request(request: RunRequest) -> tuple[dict, list[dict], dict]:
    incoming = request.config
    scenario = incoming.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("Configuration must include the scenario object")
    scenario = json.loads(json.dumps(scenario))
    scenario["duration_s"] = float(incoming.get("duration_s", scenario.get("duration_s", 60)))
    scenario["dt"] = float(incoming.get("dt", scenario.get("dt", 0.05)))
    enabled = incoming.get("enabled_links")
    if isinstance(enabled, list):
        scenario["links"] = [link for link in scenario.get("links", []) if link["name"] in enabled]
    if not scenario.get("links"):
        raise ValueError("Enable at least one radio link")
    power_limit = incoming.get("radio_power_limit_w")
    if power_limit is not None:
        scenario.setdefault("budget", {})["comm_power_w"] = float(power_limit)
    tx_limit = incoming.get("tx_limit_dbm")
    if tx_limit is not None:
        for link in scenario.get("links", []):
            link["ul_tx_dbm"] = min(float(link.get("ul_tx_dbm", tx_limit)), float(tx_limit))
    scenario = uavlink_sim.load_scenario(scenario)
    scenario_dict = uavlink_sim.dataclasses.asdict(scenario)
    predictive = incoming.get("predictive_handover", {})
    policy = _make_policy(incoming.get("policy", "Predictive · cost"), predictive)
    if policy["name"] == "predictive_handover":
        predictive_handover.HandoverConfig.from_dict(policy["params"]["handover_config"])
        predictive_handover.make_policy(policy["params"]["policy_name"])
    seed = int(incoming.get("seed", 2026))
    seeds = int(incoming.get("seeds", 3) if request.batch else 1)
    if seed < 0 or seeds < 1 or seeds > 500:
        raise ValueError("Seed must be non-negative and seed count must be between 1 and 500")
    if scenario.duration_s > 7200:
        raise ValueError("Simulation duration is limited to two hours")
    jobs_policy = [policy]
    if request.batch:
        for label in ("Reactive", "Bonding"):
            baseline = _make_policy(label, {})
            if baseline["name"] not in [entry["name"] for entry in jobs_policy]:
                jobs_policy.append(baseline)
    policies = []
    for policy in jobs_policy:
        label = policy["name"]
        if policy["name"] == "predictive_handover":
            label = f"pred_{policy['params'].get('policy_name', 'cost')}"
        policies.append({"label": label, **policy})
    cfg = {
        "name": f"uavlink_ui_{scenario.name}",
        "scenario": scenario_dict,
        "policies": policies,
        "n_trials": seeds,
        "master_seed": seed,
        "baseline": "reactive" if request.batch else None,
        "save_ticks": "all",
        "evaluation": incoming.get("evaluation", {}),
    }
    return cfg, jobs_policy, incoming


def _snapshot(sim: uavlink_sim.SimulationController) -> dict:
    raw = sim.snapshot()
    decision = sim.decision
    links = {}
    for name, runtime in sim.links.items():
        links[name] = {
            "state": runtime.state,
            "rssi_dbm": runtime.rx_dl,
            "snr_db": runtime.sinr_dl,
            "sinr_ul_db": runtime.sinr_ul,
            "per_ctrl": runtime.per_ctrl,
            "power_w": sim.p_link_prev.get(name, 0.0),
            "latency_ms": None,
            "enabled": True,
        }
    raw.update(
        links=links,
        decision={
            "control_links": list(decision.control_links),
            "video_shares": dict(decision.video_shares),
            "duplicate_links": list(decision.control_links[1:]),
            "tag": decision.tag,
            "outage_prob": decision.outage_prob or {},
            "horizon_s": decision.horizon_s,
            "video_bps": decision.video_bps,
        },
        blind_time_s=float(sim.m["blind_t"]),
        blind_status=not sim.ctrl_rel and not sim.vid_rel,
        energy_used_wh=float(sim.scn.drone.battery_wh * sim.scn.drone.start_soc - sim.drone.energy_wh),
        battery_wh=float(sim.drone.energy_wh),
        battery_capacity_wh=float(sim.scn.drone.battery_wh),
        metrics={
            "ctrl_ontime_rate": sim.m["ctrl_ok"] / max(1, sim.m["ctrl_sent"]),
            "video_quality_mean": sim.m["vq_sum"] / max(1, sim.m["vid_n"]),
            "blind_time_s": float(sim.m["blind_t"]),
            "comm_power_w": float(sim.p_comm_w),
        },
        map={
            "size_m": list(sim.scn.env.size_m),
            "stations": [{"id": station.id, "x": station.pos[0], "y": station.pos[1], "kind": station.kind}
                         for station in sim.scn.env.stations],
            "buildings": [
                [float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])]
                for lo, hi in zip(sim.env.b_lo.tolist(), sim.env.b_hi.tolist())
            ],
            "terrain_grid": sim.env.H[::max(1, math.ceil(sim.env.H.shape[0] / 40)),
                                      ::max(1, math.ceil(sim.env.H.shape[1] / 40))].tolist(),
            "terrain_min_m": float(sim.env.H.min()),
            "terrain_max_m": float(sim.env.H.max()),
            "waypoints": [uavlink_sim.dataclasses.asdict(point) for point in sim.scn.mission.waypoints],
            "interference": [uavlink_sim.dataclasses.asdict(item) for item in sim.scn.interference],
            "terrain": sim.scn.env.terrain.kind,
        },
    )
    return _finite_values(raw)


async def _control_tick(sim: uavlink_sim.SimulationController, index: int, total: int, label: str) -> bool:
    if state["stop_requested"]:
        return False
    if state["paused"]:
        await state["pause_event"].wait()
    await asyncio.to_thread(sim.step, 25)
    sample = _snapshot(sim)
    state["snapshot"] = sample
    state["progress"] = min(1.0, (index + sim.t / sim.scn.duration_s) / total)
    state["message"] = f"{label} · {sim.t:.1f} / {sim.scn.duration_s:.0f} s"
    history = state["history"]
    if not history or sample["t"] - history[-1]["t"] >= 0.5:
        history.append({
            "t": sample["t"], "blind_time_s": sample["blind_time_s"],
            "links": {key: value["snr_db"] for key, value in sample["links"].items()},
        })
        del history[:-2500]
    return not state["stop_requested"]


def _write_batch_files(output: Path, cfg: dict, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "batch_config.json").write_text(json.dumps(cfg, indent=2, default=_json_default), encoding="utf-8")
    (output / "runs.jsonl").write_text(
        "".join(json.dumps(row, default=_json_default, allow_nan=True) + "\n" for row in rows), encoding="utf-8"
    )
    columns = sorted({key for row in rows for key in row if key != "error"})
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = uavlink_sim.summarize(rows, cfg.get("baseline"), cfg.get("master_seed", 0))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (output / "summary.md").write_text(uavlink_sim.summary_markdown(summary) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "config_hash": uavlink_sim.chash(cfg), "n_jobs": len(rows),
        "module_version": uavlink_sim.__version__, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")


async def _execute(cfg: dict, incoming: dict) -> None:
    output = UI_RESULTS_DIR / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    rows: list[dict] = []
    jobs = uavlink_sim.make_jobs(cfg, str(output))
    total = len(jobs)
    state.update(status="running", progress=0.0, result_dir=str(output.relative_to(ROOT)),
                 metrics=None, error=None, stop_requested=False, history=[])
    _log(f"Starting {total} run(s) · {cfg['scenario']['name']}")
    try:
        for index, job in enumerate(jobs):
            if state["stop_requested"]:
                break
            scenario = uavlink_sim.deep_merge(job["scenario"], job.get("overrides") or {})
            scenario = uavlink_sim.randomize_scenario(scenario, job.get("randomize") or {}, job["seed"])
            label = f"{job['label']} · seed {job['trial'] + 1}/{cfg['n_trials']}"
            sim = uavlink_sim.SimulationController(
                scenario, job["policy"], job["seed"], out_dir=job["out_dir"],
                log_level=job["log_level"], run_id=job["run_id"],
            )
            _log(f"Running {label}")
            try:
                while not sim.done and not state["stop_requested"]:
                    if not await _control_tick(sim, index, total, label):
                        break
                    await asyncio.sleep(0.025)
                result = await asyncio.to_thread(sim.close)
                row = {
                    **result.metrics,
                    "run_id": job["run_id"], "policy_label": job["label"], "trial": job["trial"],
                    "seed": job["seed"], "crc": result.crc, "tick_hash": result.tick_hash, "error": "",
                }
                rows.append(row)
                state["metrics"] = row
                if state["snapshot"] is not None:
                    state["snapshot"]["done"] = sim.done
                    state["snapshot"]["outcome"] = result.outcome
                _log(f"Completed {label} · blind {result.metrics.get('blind_time_s', 0):.2f}s · {result.outcome}")
            except Exception:
                await asyncio.to_thread(sim.close)
                raise
            state["progress"] = (index + 1) / total
        _write_batch_files(output, cfg, rows)
        if rows:
            state["message"] = "Evaluating saved runs"
            _log("Generating uavlink_eval tables, report, and figures")
            eval_cfg = uavlink_eval.EvalConfig()
            eval_cfg.stats.bootstrap_B = 500
            eval_cfg.stats.permutation_B = 1000
            eval_cfg.definitions.deadline_ms = float(cfg["evaluation"].get("control_latency_ms", 150))
            eval_cfg.definitions.vid_min_quality = float(cfg["evaluation"].get("video_quality_min", 0))
            evaluation = await asyncio.to_thread(
                uavlink_eval.evaluate, str(output), None, eval_cfg, str(output / "evaluation"), False, 1, None, None, True
            )
            state["evaluation"] = {
                "analysis": str((output / "evaluation" / "analysis.json").relative_to(ROOT)),
                "figures": [
                    str((output / "evaluation" / figure).relative_to(ROOT))
                    for figure in evaluation.get("figures", [])
                ],
                "report": str((output / "evaluation" / "report.html").relative_to(ROOT)),
            }
            state["progress"] = 1.0 if not state["stop_requested"] else state["progress"]
        state["status"] = "stopped" if state["stop_requested"] else "finished"
        state["message"] = "Stopped · partial results saved" if state["stop_requested"] else "Run complete"
        _log(state["message"])
    except Exception as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}", message="Run failed")
        _log(f"ERROR · {type(exc).__name__}: {exc}")
        traceback.print_exc()


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/scenarios")
async def scenarios() -> dict:
    return {"scenarios": list(uavlink_sim.PRESETS), "policies": list(POLICY_CHOICES)}


@app.get("/api/scenarios/{name}")
async def scenario(name: str) -> dict:
    if name not in uavlink_sim.PRESETS:
        raise HTTPException(status_code=404, detail="Unknown scenario preset")
    return uavlink_sim.load_scenario_dict(name)


@app.get("/api/map/{name}")
async def scenario_map(name: str, seed: int = 2026) -> dict:
    if name not in uavlink_sim.PRESETS:
        raise HTTPException(status_code=404, detail="Unknown scenario preset")
    if seed < 0:
        raise HTTPException(status_code=422, detail="Seed must be non-negative")
    scenario_cfg = uavlink_sim.load_scenario(name)
    trial_seed = uavlink_sim.trial_seed(seed, 0)
    environment = uavlink_sim.Environment(scenario_cfg.env, trial_seed)
    map_data = {
        "size_m": list(scenario_cfg.env.size_m),
        "stations": [
            {"id": station.id, "x": float(environment.stations[station.id][0]),
             "y": float(environment.stations[station.id][1]), "kind": station.kind}
            for station in scenario_cfg.env.stations
        ],
        "buildings": [
            [float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])]
            for lo, hi in zip(environment.b_lo.tolist(), environment.b_hi.tolist())
        ],
        "terrain_grid": environment.H[::max(1, math.ceil(environment.H.shape[0] / 40)),
                                      ::max(1, math.ceil(environment.H.shape[1] / 40))].tolist(),
        "terrain_min_m": float(environment.H.min()),
        "terrain_max_m": float(environment.H.max()),
        "waypoints": [uavlink_sim.dataclasses.asdict(point) for point in scenario_cfg.mission.waypoints],
        "interference": [uavlink_sim.dataclasses.asdict(item) for item in scenario_cfg.interference],
        "terrain": scenario_cfg.env.terrain.kind,
    }
    return _finite_values(map_data)


@app.get("/api/status")
async def status() -> dict:
    return _finite_values({key: value for key, value in state.items() if key not in ("task", "pause_event")})


@app.post("/api/runs")
async def start_run(request: RunRequest) -> dict:
    if state.get("task") is not None and not state["task"].done():
        raise HTTPException(status_code=409, detail="A run is already active")
    try:
        cfg, _, incoming = _prepare_request(request)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state.update(
        status="starting", progress=0.0, message="Preparing simulation", logs=[],
        error=None, metrics=None, evaluation=None, snapshot=None, paused=False, stop_requested=False,
        batch=request.batch, pause_event=asyncio.Event(),
    )
    state["pause_event"].set()
    state["task"] = asyncio.create_task(_execute(cfg, incoming))
    return {"status": "starting", "jobs": cfg["n_trials"] * len(cfg["policies"])}


@app.post("/api/control/{action}")
async def control(action: str) -> dict:
    if state.get("status") not in ("running", "starting"):
        raise HTTPException(status_code=409, detail="There is no active run")
    if action == "pause":
        state["paused"] = True
        state["pause_event"].clear()
        state["status"] = "paused"
        _log("Simulation paused")
    elif action == "resume":
        state["paused"] = False
        state["pause_event"].set()
        state["status"] = "running"
        _log("Simulation resumed")
    elif action == "stop":
        state["stop_requested"] = True
        state["paused"] = False
        state["pause_event"].set()
        _log("Stop requested; finishing current simulation tick")
    else:
        raise HTTPException(status_code=404, detail="Unknown run control")
    return {"status": state["status"]}


@app.post("/api/dry-run")
async def dry_run(request: RunRequest) -> dict:
    try:
        cfg, _, _ = _prepare_request(request)
        scenario_obj = uavlink_sim.load_scenario(cfg["scenario"])
        for spec in cfg["policies"]:
            uavlink_sim.make_policy(spec)
        return {"valid": True, "scenario": scenario_obj.name, "duration_s": scenario_obj.duration_s,
                "jobs": len(uavlink_sim.make_jobs(cfg)), "dt": scenario_obj.dt}
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/self-test")
async def self_test() -> dict:
    passed = await asyncio.to_thread(run_experiment._self_test)
    return {"checks": {"existing cross-module self-test": passed}, "passed": passed}


@app.get("/api/configs")
async def configs() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return {"configs": sorted(path.stem for path in CONFIG_DIR.glob("*.json"))}


@app.post("/api/configs")
async def save_config(request: ConfigRequest) -> dict:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", request.name).strip("-_")
    if not name or len(name) > 64:
        raise HTTPException(status_code=422, detail="Config name must be 1–64 letters, numbers, hyphens, or underscores")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(request.config, indent=2, allow_nan=False), encoding="utf-8")
    return {"name": name}


@app.get("/api/configs/{name}")
async def load_config(name: str) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise HTTPException(status_code=422, detail="Invalid config name")
    path = CONFIG_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Config not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/results")
async def results() -> dict:
    output = []
    if RESULTS_DIR.exists():
        for metrics_file in RESULTS_DIR.rglob("metrics.json"):
            if metrics_file.is_relative_to(UI_RESULTS_DIR) or metrics_file.parent.name.startswith("run"):
                try:
                    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                    evaluation_dir = metrics_file.parent.parent.parent / "evaluation"
                    output.append({
                        "run_id": metrics.get("run_id", metrics_file.parent.name),
                        "scenario": metrics.get("scenario", "—"),
                        "policy": metrics.get("policy", "—"),
                        "path": str(metrics_file.parent.relative_to(ROOT)),
                        "report": str((evaluation_dir / "report.html").relative_to(ROOT))
                        if (evaluation_dir / "report.html").is_file() else None,
                        "analysis": str((evaluation_dir / "analysis.json").relative_to(ROOT))
                        if (evaluation_dir / "analysis.json").is_file() else None,
                        "figures": [str(path.relative_to(ROOT)) for path in sorted(evaluation_dir.glob("fig*.png"))],
                        "metrics": metrics,
                    })
                except (OSError, json.JSONDecodeError):
                    continue
    return _finite_values({"results": sorted(output, key=lambda item: item["path"], reverse=True)[:100]})


@app.get("/api/results/file")
async def result_file(path: str) -> FileResponse:
    candidate = _result_path(path)
    return FileResponse(candidate)


@app.get("/api/results/analysis")
async def result_analysis(path: str) -> dict:
    candidate = _result_path(path)
    if candidate.suffix.lower() != ".json":
        raise HTTPException(status_code=422, detail="Analysis file must be JSON")
    try:
        return _finite_values(json.loads(candidate.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not read analysis JSON: {exc}") from exc


@app.websocket("/ws")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = {key: value for key, value in state.items() if key not in ("task", "pause_event")}
            await websocket.send_json(_finite_values(data))
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(UI_DIR))
