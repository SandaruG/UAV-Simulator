#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uavlink_sim.py - Simulation Controller & Scenario Runner for multi-link UAV
outage-prediction research.

PURPOSE
    Closed-loop, reproducible experimental platform to test whether a lightweight
    predictor (RSSI/SNR history + position + trajectory) can trigger a soft/hard
    link switch BEFORE both control and video are lost, and to measure the
    reduction in "blind time" versus reactive switching or simple bonding.

WHAT IS IN HERE (single file, depends only on numpy)
    * Scenario model      : JSON/YAML/dict -> typed dataclasses (typos rejected).
    * Environment         : terrain heightmap, buildings (procedural or loaded),
                            ground stations / cell towers, geofence.
    * Weather timeline    : wind + OU gusts, rain (ITU-style attenuation), temperature.
    * Drone model         : kinematics w/ wind, waypoint/terrain-following guidance,
                            tilt-dependent antenna pattern, power & battery model,
                            RTL / hold / land / failsafes, GPS error model.
    * Radio channel       : per link & per station: path loss, terrain knife-edge +
                            building penetration, correlated shadowing, Rician/Rayleigh
                            fading, rain, antenna pattern, asymmetric DL(control)/UL(video)
                            SINR with scheduled/mobile interferers, rate adaptation,
                            PER curves, handover (A3 + TTT), RLF, (re)association delays,
                            queueing, ARQ, latency.
    * Traffic             : 50 Hz control stream w/ deadline, chunked video w/ FEC,
                            bonding/duplication, rolling reliability windows,
                            BLIND-TIME definition (neither control nor video reliable).
    * Fault injection     : scheduled, stochastic (Poisson) and ON-DEMAND (API / CLI).
    * Policy plug-in API  : Policy.decide(Observation) -> Decision. Built-ins:
                            static_primary, reactive, bonding, cost_predictive
                            (reference predictor; replace with yours via "module:Class").
    * Reproducibility     : independent RNG streams (SeedSequence) per subsystem and per
                            link => common random numbers across policies (paired tests);
                            config hash, manifest, tick-log hash, checkpoint/restore.
    * Logging             : per-tick CSV.gz, events.jsonl, decisions.jsonl, metrics.json.
    * Batches             : Monte-Carlo with scenario randomisation, policy sweeps,
                            multiprocessing, resume, paired bootstrap statistics, replay.

CLI (python uavlink_sim.py -h)
    list-presets | make-example | make-batch-example | run | batch | replay | plot | selftest

PROGRAMMATIC
    sim = SimulationController("urban_canyon", policy="cost_predictive", seed=7)
    sim.run(until_s=40); sim.inject_fault("link_blackout", 6.0, link="wifi_5g")
    cp = sim.checkpoint(); sim.run(); res = sim.result(); sim.restore(cp)

A ROS 2 wrapper only needs to call sim.step() from a timer and publish sim.snapshot().
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import gzip
import hashlib
import importlib
import json
import math
import os
import platform
import struct
import sys
import time
import traceback
import types
import typing
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "1.0.0"
C_LIGHT = 299_792_458.0
G = 9.80665
NAN = float("nan")


# =============================================================================
# 1. UTILITIES
# =============================================================================
def db2lin(x: float) -> float:
    return 10.0 ** (x / 10.0)


def lin2db(x: float) -> float:
    return 10.0 * math.log10(x if x > 1e-30 else 1e-30)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def fspl_1m_db(f_hz: float) -> float:
    return 20.0 * math.log10(4.0 * math.pi * f_hz / C_LIGHT)


def noise_dbm(bw_hz: float, nf_db: float) -> float:
    return -174.0 + 10.0 * math.log10(bw_hz) + nf_db


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def cjson(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=_jdefault)


def chash(o) -> str:
    return hashlib.sha256(cjson(o).encode()).hexdigest()[:16]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(d: dict, path: str, value) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# --- PHY abstraction ---------------------------------------------------------
def mcs_threshold_db(eff: float) -> float:
    """SINR (dB) at which a 1500-byte packet has ~10 % PER for spectral efficiency `eff`
    (Shannon gap abstraction: 2.5 dB implementation loss)."""
    return 10.0 * math.log10(2.0 ** eff - 1.0) + 2.5


def per_from_sinr(sinr_db: float, eff: float, bits: float, ref_bits: float = 12000.0,
                  slope_db: float = 0.7) -> float:
    x = (sinr_db - mcs_threshold_db(eff)) / slope_db + math.log(9.0)
    x = clamp(x, -40.0, 40.0)
    p = 1.0 / (1.0 + math.exp(x))
    if p >= 1.0 - 1e-12:
        return 1.0
    return 1.0 - (1.0 - p) ** (bits / ref_bits)


# =============================================================================
# 2. CONFIGURATION DATACLASSES (typed, validated, JSON/YAML loadable)
# =============================================================================
def _L(*v):
    return field(default_factory=lambda: list(v))


def _D(**kw):
    return field(default_factory=lambda: dict(kw))


@dataclass
class StationCfg:
    id: str = "gcs"
    pos: List[float] = _L(0.0, 0.0, 5.0)          # x, y, height AGL (m)
    kind: str = "gcs"                              # gcs | cell | relay


@dataclass
class BuildingCfg:
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 10.0
    y1: float = 10.0
    h: float = 20.0


@dataclass
class TerrainCfg:
    kind: str = "hills"                            # flat | hills | heightmap (via map_file)
    res_m: float = 20.0
    hill_count: int = 8
    hill_height_m: List[float] = _L(0.0, 40.0)
    hill_sigma_m: List[float] = _L(150.0, 500.0)


@dataclass
class ProcBuildCfg:
    count: int = 0
    footprint_m: List[float] = _L(15.0, 60.0)
    height_m: List[float] = _L(10.0, 80.0)
    region: List[float] = _L()                     # [x0,y0,x1,y1]; empty = whole map
    keepout_m: float = 40.0                        # around stations and home


@dataclass
class GeofenceCfg:
    radius_m: float = 0.0                          # 0 = disabled
    max_alt_m: float = 0.0                         # 0 = disabled (AGL)


@dataclass
class EnvCfg:
    size_m: List[float] = _L(2000.0, 2000.0)
    env_seed: Optional[int] = None                 # fixed map across trials if set
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    buildings: List[BuildingCfg] = field(default_factory=list)
    proc_buildings: ProcBuildCfg = field(default_factory=ProcBuildCfg)
    stations: List[StationCfg] = field(default_factory=lambda: [StationCfg()])
    home: List[float] = _L(300.0, 300.0)
    geofence: GeofenceCfg = field(default_factory=GeofenceCfg)
    map_file: Optional[str] = None                 # .json / .npz with heightmap + buildings
    wall_entry_db: float = 8.0
    wall_db_per_m: float = 0.8
    wall_max_db: float = 30.0                      # per building (over-roof diffraction bound)
    max_obstruction_db: float = 80.0
    policy_knows_map: bool = True                  # radio prior uses real geometry


@dataclass
class WaypointCfg:
    x: float = 0.0
    y: float = 0.0
    z: float = 60.0                                # AGL if mission.agl else absolute
    speed: float = 0.0                             # 0 = mission default
    hold_s: float = 0.0


@dataclass
class MissionCfg:
    waypoints: List[WaypointCfg] = field(default_factory=list)
    takeoff_alt_m: float = 40.0
    default_speed_mps: float = 10.0
    accept_radius_m: float = 4.0
    agl: bool = True
    rtl_alt_m: float = 60.0
    rtl_at_end: bool = True
    land_speed_mps: float = 1.5
    lookahead_m: float = 20.0


@dataclass
class DroneCfg:
    mass_kg: float = 2.5
    max_speed_mps: float = 18.0
    max_accel_mps2: float = 4.0
    max_climb_mps: float = 4.0
    max_descent_mps: float = 3.0
    vel_tau_s: float = 0.6
    hover_power_w: float = 220.0
    accel_power_coef: float = 0.15
    cda_m2: float = 0.05
    climb_eff: float = 0.6
    payload_power_w: float = 12.0
    battery_wh: float = 90.0
    start_soc: float = 1.0
    rtl_soc: float = 0.25
    crit_soc: float = 0.10
    cap_temp_coeff: float = 0.01                   # capacity loss per degC below 20
    min_clearance_m: float = 3.0
    gps_sigma_h_m: float = 1.2
    gps_sigma_v_m: float = 2.0
    gps_bias_sigma_m: float = 1.5
    gps_rate_hz: float = 5.0


@dataclass
class LinkCfg:
    name: str = "wifi_5g"
    tech: str = "wifi"
    station_ids: List[str] = _L()                  # empty = all stations
    freq_hz: float = 5.8e9
    bw_hz: float = 20e6
    dl_tx_dbm: float = 24.0                        # station -> drone (control path)
    ul_tx_dbm: float = 20.0                        # drone -> station (video path)
    gs_gain_dbi: float = 12.0
    drone_gain_dbi: float = 2.0
    ant_pattern: str = "dipole"                    # dipole | omni
    nf_db: float = 6.0
    ambient_noise_rise_db: float = 0.0
    pl_exp: float = 2.2
    shadow_sigma_los_db: float = 2.0
    shadow_sigma_nlos_db: float = 6.0
    shadow_decorr_m: float = 40.0
    shadow_tau_s: float = 30.0
    rician_k_db: float = 8.0
    mcs_eff: List[float] = _L(0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    ctrl_eff: float = 0.5
    mac_eff: float = 0.6
    ra_margin_db: float = 2.0
    carries_video: bool = True
    base_latency_ms: float = 6.0
    jitter_ms: float = 3.0
    mac_delay_ms: float = 2.0
    retx_delay_ms: float = 3.0
    max_attempts: int = 4
    activation_delay_s: float = 0.4                # cold -> usable (association)
    rlf_timeout_s: float = 1.0                     # sync lost this long -> link goes cold
    sync_margin_db: float = 2.0
    ho_hyst_db: float = 3.0
    ho_ttt_s: float = 0.16
    ho_interrupt_s: float = 0.05
    ho_fail_p: float = 0.03
    active_w: float = 2.5
    warm_w: float = 0.8
    sleep_w: float = 0.05
    pa_eff: float = 0.25


@dataclass
class WeatherKey:
    t: float = 0.0
    wind_mps: float = 3.0
    wind_dir_deg: float = 270.0                    # direction wind blows FROM (0 = north)
    gust_sigma_mps: float = 1.0
    rain_mm_h: float = 0.0
    temp_c: float = 18.0


@dataclass
class WeatherCfg:
    timeline: List[WeatherKey] = field(default_factory=lambda: [WeatherKey()])
    gust_tau_s: float = 5.0


@dataclass
class InterferenceCfg:
    name: str = "jam0"
    kind: str = "jammer"                           # jammer | pulsed_jammer | wifi_burst | cochannel
    t0: float = 0.0
    t1: float = 1e9
    pos: List[float] = _L(0.0, 0.0, 5.0)           # absolute z
    vel: List[float] = _L(0.0, 0.0, 0.0)           # mobile source (m/s)
    eirp_dbm: float = 30.0
    fc_hz: float = 5.8e9
    bw_hz: float = 20e6
    duty: float = 1.0
    pl_exp: float = 2.5
    aclr_db: float = -45.0


@dataclass
class FaultCfg:
    kind: str = "link_blackout"
    t0: float = 0.0
    duration_s: float = 5.0
    link: str = "*"                                # "*" | "name" | "a,b"
    params: Dict[str, float] = _D()


@dataclass
class FaultProcessCfg:
    kind: str = "link_blackout"
    rate_per_min: float = 0.5
    duration_mean_s: float = 3.0
    duration_min_s: float = 0.5
    duration_max_s: float = 15.0
    link: str = "any"                              # "*" | "any" (random one) | "name"
    params_range: Dict[str, List[float]] = _D()    # uniform ranges
    t_from: float = 0.0
    t_to: Optional[float] = None


@dataclass
class TrafficCfg:
    ctrl_hz: float = 50.0
    ctrl_bytes: int = 64
    hdr_bytes: int = 40
    ctrl_deadline_ms: float = 150.0
    video_max_bps: float = 8e6
    video_deadline_ms: float = 250.0
    mtu_bytes: int = 1400
    fec: float = 0.15
    video_mac_retries: int = 2
    max_queue_ms: float = 300.0
    ctrl_window_s: float = 0.3
    ctrl_ok_ratio: float = 0.8
    ctrl_gap_max_ms: float = 200.0
    vid_window_s: float = 0.5
    vid_ok_ratio: float = 0.7
    freeze_s: float = 0.2
    q_r0_bps: float = 0.25e6


@dataclass
class SensingCfg:
    meas_period_s: float = 0.1
    meas_delay_s: float = 0.05
    meas_sigma_db: float = 1.5
    hist_len: int = 100
    policy_hz: float = 10.0
    decision_delay_s: float = 0.05


@dataclass
class FailsafeCfg:
    action: str = "rtl"                            # none | hold | rtl
    timeout_s: float = 3.0
    resume_s: float = 2.0
    resume: bool = True
    hold_to_rtl_s: float = 8.0


@dataclass
class BudgetCfg:
    comm_power_w: float = 8.0
    spectrum_hz: float = 40e6


@dataclass
class MetricsCfg:
    outage_min_ms: float = 200.0
    pred_threshold: float = 0.5
    pingpong_s: float = 3.0
    latency_limits_ms: List[float] = _L(100.0, 150.0)


@dataclass
class ScenarioCfg:
    name: str = "scenario"
    duration_s: float = 300.0
    dt: float = 0.02
    geometry_period_s: float = 0.1
    env: EnvCfg = field(default_factory=EnvCfg)
    drone: DroneCfg = field(default_factory=DroneCfg)
    mission: MissionCfg = field(default_factory=MissionCfg)
    weather: WeatherCfg = field(default_factory=WeatherCfg)
    links: List[LinkCfg] = field(default_factory=lambda: [LinkCfg()])
    interference: List[InterferenceCfg] = field(default_factory=list)
    faults: List[FaultCfg] = field(default_factory=list)
    fault_processes: List[FaultProcessCfg] = field(default_factory=list)
    traffic: TrafficCfg = field(default_factory=TrafficCfg)
    sensing: SensingCfg = field(default_factory=SensingCfg)
    failsafe: FailsafeCfg = field(default_factory=FailsafeCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)

    def validate(self) -> None:
        if self.dt <= 0 or self.duration_s <= 0:
            raise ValueError("dt and duration_s must be > 0")
        if not self.mission.waypoints:
            raise ValueError("mission needs at least one waypoint")
        if not self.links:
            raise ValueError("at least one link required")
        if 1.0 / self.traffic.ctrl_hz < self.dt * 0.999:
            raise ValueError("control period shorter than dt; reduce dt or ctrl_hz")
        names = [l.name for l in self.links]
        if len(set(names)) != len(names):
            raise ValueError("duplicate link names")
        sids = {s.id for s in self.env.stations}
        for l in self.links:
            for s in l.station_ids:
                if s not in sids:
                    raise ValueError(f"link {l.name}: unknown station {s}")
        for f in self.faults:
            if f.kind not in FAULT_KINDS:
                raise ValueError(f"unknown fault kind {f.kind}; known: {sorted(FAULT_KINDS)}")


def from_dict(cls, d: Optional[dict]):
    """dict -> dataclass tree. Unknown keys raise (catches typos in scenario files)."""
    d = d or {}
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} for {cls.__name__}; valid: {sorted(known)}")
    return cls(**{k: _conv(hints[k], v) for k, v in d.items()})


def _conv(tp, v):
    if v is None:
        return None
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    ut = getattr(types, "UnionType", None)
    if origin is typing.Union or (ut is not None and origin is ut):
        for a in args:
            if a is not type(None):
                return _conv(a, v)
    if dataclasses.is_dataclass(tp) and isinstance(v, dict):
        return from_dict(tp, v)
    if origin in (list, List):
        return [_conv(args[0], x) for x in v] if args else list(v)
    if origin in (dict, Dict):
        return {k: _conv(args[1], x) for k, x in v.items()} if args else dict(v)
    if tp is float and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    return v


def load_scenario_dict(x) -> dict:
    """x: dict | preset name | path to .json/.yml"""
    if isinstance(x, dict):
        return copy.deepcopy(x)
    if isinstance(x, str) and x in PRESETS:
        return PRESETS[x]()
    if isinstance(x, str) and os.path.exists(x):
        with open(x) as f:
            if x.endswith((".yml", ".yaml")):
                import yaml  # optional dependency
                return yaml.safe_load(f)
            return json.load(f)
    raise ValueError(f"cannot load scenario {x!r}; presets: {sorted(PRESETS)}")


def load_scenario(x) -> ScenarioCfg:
    if isinstance(x, ScenarioCfg):
        return x
    s = from_dict(ScenarioCfg, load_scenario_dict(x))
    s.validate()
    return s


# =============================================================================
# 3. RNG BANK - independent streams => common random numbers across policies
# =============================================================================
STREAMS = ("env", "weather", "faults", "sensors", "switch", "policy", "scenario")


class RngBank:
    """Every subsystem (and every link) owns an independent stream derived from the
    trial seed, so changing the POLICY never perturbs weather, faults, fading,
    shadowing, packet-loss draws, or sensor noise. This enables paired comparisons."""

    def __init__(self, seed: int, link_names: Sequence[str]):
        self.seed = int(seed)
        self._g = {n: np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(0, i)))
                   for i, n in enumerate(STREAMS)}
        self.chan, self.pkt = {}, {}
        for ln in link_names:
            h = stable_hash(ln)
            self.chan[ln] = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(1, h, 0)))
            self.pkt[ln] = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(1, h, 1)))

    def __getitem__(self, n: str) -> np.random.Generator:
        return self._g[n]


# =============================================================================
# 4. ENVIRONMENT
# =============================================================================
class Environment:
    def __init__(self, cfg: EnvCfg, seed: int):
        self.cfg = cfg
        s = cfg.env_seed if cfg.env_seed is not None else seed
        rng = np.random.default_rng(np.random.SeedSequence(int(s), spawn_key=(7,)))
        self.sx, self.sy = float(cfg.size_m[0]), float(cfg.size_m[1])
        self.res = float(cfg.terrain.res_m)
        self.nx, self.ny = int(self.sx // self.res) + 2, int(self.sy // self.res) + 2
        H = np.zeros((self.ny, self.nx))
        blds = [[b.x0, b.y0, b.x1, b.y1, b.h] for b in cfg.buildings]
        if cfg.map_file:
            Hm, resm, bm = self._load_map(cfg.map_file)
            if Hm is not None:
                H, self.res = Hm, resm
                self.ny, self.nx = H.shape
            blds += bm
        elif cfg.terrain.kind == "hills":
            X, Y = np.meshgrid(np.arange(self.nx) * self.res, np.arange(self.ny) * self.res)
            lo, hi = cfg.terrain.hill_height_m
            for _ in range(cfg.terrain.hill_count):
                cx, cy = rng.uniform(0, self.sx), rng.uniform(0, self.sy)
                sg = rng.uniform(*cfg.terrain.hill_sigma_m)
                H += rng.uniform(lo, hi) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sg * sg))
        self.H = H
        # stations (z given AGL -> absolute)
        self.stations: Dict[str, np.ndarray] = {}
        for st in cfg.stations:
            x, y, h = st.pos
            self.stations[st.id] = np.array([x, y, self.terrain_at(x, y) + h])
        self.home = np.array([cfg.home[0], cfg.home[1], self.terrain_at(cfg.home[0], cfg.home[1])])
        # procedural buildings
        pb = cfg.proc_buildings
        reg = pb.region or [0.0, 0.0, self.sx, self.sy]
        keep = [(p[0], p[1]) for p in self.stations.values()] + [(self.home[0], self.home[1])]
        for _ in range(pb.count):
            for _try in range(25):
                w, d = rng.uniform(*pb.footprint_m), rng.uniform(*pb.footprint_m)
                x0, y0 = rng.uniform(reg[0], reg[2] - w), rng.uniform(reg[1], reg[3] - d)
                cx, cy = x0 + w / 2, y0 + d / 2
                if all(math.hypot(cx - kx, cy - ky) > pb.keepout_m + max(w, d) / 2 for kx, ky in keep):
                    blds.append([x0, y0, x0 + w, y0 + d, rng.uniform(*pb.height_m)])
                    break
        self.nb = len(blds)
        if blds:
            lo_, hi_ = [], []
            for x0, y0, x1, y1, h in blds:
                zb = self.terrain_at((x0 + x1) / 2, (y0 + y1) / 2)
                lo_.append([min(x0, x1), min(y0, y1), zb])
                hi_.append([max(x0, x1), max(y0, y1), zb + h])
            self.b_lo, self.b_hi = np.array(lo_), np.array(hi_)
        else:
            self.b_lo = self.b_hi = np.zeros((0, 3))

    @staticmethod
    def _load_map(path: str):
        if path.endswith(".npz"):
            z = np.load(path)
            b = z["buildings"].tolist() if "buildings" in z.files else []
            return z["heightmap"].astype(float), float(z["res_m"]), b
        with open(path) as f:
            j = json.load(f)
        hm = j.get("heightmap")
        b = [[x["x0"], x["y0"], x["x1"], x["y1"], x["h"]] if isinstance(x, dict) else list(x)
             for x in j.get("buildings", [])]
        if hm:
            return np.array(hm["data"], float), float(hm["res_m"]), b
        return None, None, b

    def terrain_h(self, x, y):
        gx = np.clip(np.asarray(x, float) / self.res, 0, self.nx - 1.001)
        gy = np.clip(np.asarray(y, float) / self.res, 0, self.ny - 1.001)
        ix, iy = gx.astype(int), gy.astype(int)
        fx, fy = gx - ix, gy - iy
        H = self.H
        return (H[iy, ix] * (1 - fx) * (1 - fy) + H[iy, ix + 1] * fx * (1 - fy)
                + H[iy + 1, ix] * (1 - fx) * fy + H[iy + 1, ix + 1] * fx * fy)

    def terrain_at(self, x: float, y: float) -> float:
        return float(self.terrain_h(x, y))

    def _seg_box_len(self, p0, p1):
        d = p1 - p0
        dd = np.where(np.abs(d) < 1e-9, 1e-9, d)
        t1, t2 = (self.b_lo - p0) / dd, (self.b_hi - p0) / dd
        tmin = np.maximum(np.minimum(t1, t2).max(axis=1), 0.0)
        tmax = np.minimum(np.maximum(t1, t2).min(axis=1), 1.0)
        return np.clip(tmax - tmin, 0.0, None) * float(np.linalg.norm(d))

    def obstruction_db(self, p0: np.ndarray, p1: np.ndarray, freq_hz: float) -> float:
        """Terrain knife-edge diffraction (max Fresnel parameter along the path) +
        building penetration, capped. Deterministic geometry only."""
        cfg = self.cfg
        d = float(np.linalg.norm(p1 - p0))
        if d < 1.0:
            return 0.0
        lam = C_LIGHT / freq_hz
        u = np.arange(1, 25) / 25.0
        pts = p0[None, :] + u[:, None] * (p1 - p0)[None, :]
        excess = self.terrain_h(pts[:, 0], pts[:, 1]) - pts[:, 2]
        v = float((excess * np.sqrt(2.0 / (lam * u * (1 - u) * d))).max())
        Lt = 0.0
        if v > -0.78:
            Lt = 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
        Lb = 0.0
        if self.nb:
            lens = self._seg_box_len(p0, p1)
            m = lens > 0.05
            if m.any():
                Lb = float(np.minimum(cfg.wall_entry_db + cfg.wall_db_per_m * lens[m], cfg.wall_max_db).sum())
        return min(Lt + Lb, cfg.max_obstruction_db)


# =============================================================================
# 5. WEATHER
# =============================================================================
@dataclass
class WxState:
    wind: np.ndarray
    wind_speed: float
    rain_mm_h: float
    temp_c: float


class Weather:
    def __init__(self, cfg: WeatherCfg, rng: np.random.Generator):
        keys = sorted(cfg.timeline, key=lambda k: k.t) or [WeatherKey()]
        self.t = np.array([k.t for k in keys])
        self.f = {n: np.array([getattr(k, n) for k in keys]) for n in
                  ("wind_mps", "wind_dir_deg", "gust_sigma_mps", "rain_mm_h", "temp_c")}
        self.tau, self.rng = cfg.gust_tau_s, rng
        self.gust = np.zeros(2)

    def step(self, t: float, dt: float, extra: np.ndarray) -> WxState:
        w = self.rng.standard_normal(2)
        i = {n: float(np.interp(t, self.t, a)) for n, a in self.f.items()}
        a = math.exp(-dt / self.tau)
        self.gust = a * self.gust + math.sqrt(1 - a * a) * i["gust_sigma_mps"] * w
        th = math.radians(i["wind_dir_deg"])
        base = i["wind_mps"] * np.array([-math.sin(th), -math.cos(th)])
        vec = np.array([base[0] + self.gust[0] + extra[0], base[1] + self.gust[1] + extra[1], 0.0])
        return WxState(vec, float(np.hypot(vec[0], vec[1])), i["rain_mm_h"], i["temp_c"])


def rain_gamma_db_per_km(f_hz: float, rain_mm_h: float) -> float:
    """Approximate specific attenuation (Olsen-Rogers style power law, valid ~3-50 GHz,
    extrapolated; negligible below 3 GHz anyway)."""
    if rain_mm_h <= 0:
        return 0.0
    fg = f_hz / 1e9
    return 4.21e-5 * fg ** 2.42 * rain_mm_h ** (1.41 * fg ** -0.0779)


# =============================================================================
# 6. DRONE (guidance + dynamics + power + failsafe modes) and GPS
# =============================================================================
class Drone:
    AIRBORNE = ("TAKEOFF", "MISSION", "HOLD", "RTL", "LAND")

    def __init__(self, cfg: DroneCfg, mission: MissionCfg, env: Environment):
        self.cfg, self.m, self.env = cfg, mission, env
        self.home = env.home.copy()
        self.pos = self.home.copy()
        self.vel = np.zeros(3)
        self.v_air = np.zeros(3)
        self.axis = np.array([0.0, 0.0, 1.0])
        self.tilt_deg = 0.0
        self.energy_wh = cfg.battery_wh * cfg.start_soc
        self.e_prop_wh = 0.0
        self.p_prop = 0.0
        self.cd = 0.5 * 1.2 * cfg.cda_m2
        self.mode = "TAKEOFF"
        nodes = [(self.home[0], self.home[1], mission.takeoff_alt_m, cfg.max_climb_mps * 0.8, 0.0)]
        for w in mission.waypoints:
            nodes.append((w.x, w.y, w.z, w.speed or mission.default_speed_mps, w.hold_s))
        self.mission_path = nodes
        self.path = nodes
        self.idx = 0
        self.seg_a = np.array([self.home[0], self.home[1], 0.0])
        self.hold_left = 0.0
        self._arrived = False
        self.saved_idx = 0
        self.hold_pt = self.home.copy()
        self.mission_done = False
        self.wp_reached = 0
        self.dist_m = 0.0
        self.clearance_viol = 0
        self.temp_c = 20.0
        self.on_mission_path = True

    # --- helpers
    def _a2z(self, x, y, a):
        return a + (self.env.terrain_at(x, y) if self.m.agl else 0.0)

    def _z2a(self, x, y, z):
        return z - (self.env.terrain_at(x, y) if self.m.agl else 0.0)

    def agl(self) -> float:
        return float(self.pos[2] - self.env.terrain_at(self.pos[0], self.pos[1]))

    def soc(self) -> float:
        cap = self.cfg.battery_wh * (1.0 - self.cfg.cap_temp_coeff * max(0.0, 20.0 - self.temp_c))
        return clamp(self.energy_wh / max(cap, 1e-6), 0.0, 1.0)

    # --- mode commands
    def _set_path(self, nodes, on_mission: bool):
        self.path, self.idx, self.hold_left, self._arrived = nodes, 0, 0.0, False
        self.on_mission_path = on_mission
        p = self.pos
        self.seg_a = np.array([p[0], p[1], self._z2a(p[0], p[1], p[2])])

    def enter_rtl(self, reason: str):
        if self.mode in ("RTL", "LAND", "LANDED", "CRASHED"):
            return []
        if self.on_mission_path:
            self.saved_idx = self.idx
        p, a = self.pos, self._z2a(*self.pos)
        alt = max(a, self.m.rtl_alt_m)
        nodes = [(p[0], p[1], alt, self.cfg.max_climb_mps * 0.8, 0.0),
                 (self.home[0], self.home[1], self.m.rtl_alt_m, self.m.default_speed_mps, 0.0)]
        self._set_path(nodes, False)
        self.mode = "RTL"
        return [("mode", {"mode": "RTL", "reason": reason})]

    def enter_hold(self, reason: str):
        if self.mode in ("LAND", "LANDED", "CRASHED"):
            return []
        if self.on_mission_path and self.mode in ("MISSION", "TAKEOFF"):
            self.saved_idx = self.idx
        self.hold_pt = self.pos.copy()
        self.mode = "HOLD"
        return [("mode", {"mode": "HOLD", "reason": reason})]

    def resume_mission(self):
        if self.mode not in ("HOLD", "RTL") or self.mission_done:
            return []
        self._set_path(self.mission_path, True)
        self.idx = min(self.saved_idx, len(self.mission_path) - 1)
        self.mode = "MISSION" if self.idx > 0 else "TAKEOFF"
        return [("mode", {"mode": self.mode, "reason": "failsafe_resume"})]

    def enter_land(self, reason: str):
        if self.mode in ("LAND", "LANDED", "CRASHED"):
            return []
        self.hold_pt = self.pos.copy()
        self.mode = "LAND"
        return [("mode", {"mode": "LAND", "reason": reason})]

    # --- guidance -> desired ground velocity
    def _guidance(self, dt, ev):
        m, cfg, pos = self.m, self.cfg, self.pos
        if self.mode == "HOLD":
            v = np.clip((self.hold_pt - pos) * 0.5, -3.0, 3.0)
            return v
        if self.mode == "LAND":
            v = np.zeros(3)
            tgt = self.home if self.hold_pt is None else self.hold_pt
            v[:2] = np.clip((tgt[:2] - pos[:2]) * 0.5, -2.0, 2.0)
            v[2] = -m.land_speed_mps
            if self.agl() < 0.25:
                self.mode = "LANDED"
                ev.append(("mode", {"mode": "LANDED", "reason": "touchdown"}))
            return v
        if self.idx >= len(self.path):
            return np.zeros(3)
        nx_, ny_, na, nsp, nhold = self.path[self.idx]
        P = np.array([pos[0], pos[1], self._z2a(pos[0], pos[1], pos[2])])
        B = np.array([nx_, ny_, na])
        dB = float(np.linalg.norm(B - P))
        if dB <= m.accept_radius_m:
            if not self._arrived:
                self._arrived, self.hold_left = True, nhold
            if self.hold_left > 0:
                self.hold_left -= dt
                return np.zeros(3)
            self.seg_a, self._arrived = B, False
            self.idx += 1
            if self.on_mission_path:
                self.wp_reached = max(self.wp_reached, self.idx - 1)
                if self.idx == 1:
                    self.mode = "MISSION"
                ev.append(("wp_reached", {"idx": self.idx - 1}))
            if self.idx >= len(self.path):
                if self.on_mission_path:
                    self.mission_done = True
                    ev.append(("mission_complete", {}))
                    ev += self.enter_rtl("mission_end") if m.rtl_at_end else self.enter_hold("mission_end")
                else:
                    ev += self.enter_land("rtl_arrived")
            return np.zeros(3)
        A, ab = self.seg_a, B - self.seg_a
        Ls = float(np.linalg.norm(ab))
        if Ls > 1e-6:
            u = ab / Ls
            s = min(max(float(np.dot(P - A, u)), 0.0) + m.lookahead_m, Ls)
            Cp = A + u * s
        else:
            Cp = B
        cz = self._a2z(Cp[0], Cp[1], Cp[2])
        dv = np.array([Cp[0] - pos[0], Cp[1] - pos[1], cz - pos[2]])
        dn = float(np.linalg.norm(dv))
        speed = nsp
        if self.idx == len(self.path) - 1 or nhold > 0:
            speed = min(speed, math.sqrt(max(0.0, dB - m.accept_radius_m) * cfg.max_accel_mps2) + 0.5)
        return dv / dn * speed if dn > 1e-6 else np.zeros(3)

    def step(self, dt: float, wind: np.ndarray, temp_c: float, comm_w: float, drain_mult: float):
        ev: list = []
        cfg = self.cfg
        self.temp_c = temp_c
        if self.mode in ("LANDED", "CRASHED"):
            self.vel[:] = 0.0
            self.p_prop = 0.0
            self.energy_wh -= cfg.payload_power_w * 0.3 * dt / 3600.0
            return ev
        v_des = self._guidance(dt, ev)
        a = (v_des - self.vel) / cfg.vel_tau_s
        ah = math.hypot(a[0], a[1])
        if ah > cfg.max_accel_mps2:
            a[0] *= cfg.max_accel_mps2 / ah
            a[1] *= cfg.max_accel_mps2 / ah
        a[2] = clamp(a[2], -0.8 * cfg.max_accel_mps2, 0.8 * cfg.max_accel_mps2)
        self.v_air += a * dt
        sp = math.hypot(self.v_air[0], self.v_air[1])
        if sp > cfg.max_speed_mps:
            self.v_air[0] *= cfg.max_speed_mps / sp
            self.v_air[1] *= cfg.max_speed_mps / sp
            sp = cfg.max_speed_mps
        self.v_air[2] = clamp(self.v_air[2], -cfg.max_descent_mps, cfg.max_climb_mps)
        self.vel = self.v_air + np.array([wind[0], wind[1], 0.0])
        self.pos += self.vel * dt
        self.dist_m += float(np.linalg.norm(self.vel)) * dt
        g = self.env.terrain_at(self.pos[0], self.pos[1])
        if self.pos[2] < g:
            self.pos[2] = g
            self.vel[2] = max(self.vel[2], 0.0)
        near_home = math.hypot(self.pos[0] - self.home[0], self.pos[1] - self.home[1]) < 25.0
        if self.mode in ("MISSION", "HOLD", "RTL") and not near_home and self.agl() < cfg.min_clearance_m:
            self.clearance_viol += 1
        axv = np.array([a[0], a[1], G + a[2]])
        self.axis = axv / float(np.linalg.norm(axv))
        self.tilt_deg = math.degrees(math.acos(clamp(self.axis[2], -1.0, 1.0)))
        vair = math.sqrt(sp * sp + self.v_air[2] ** 2)
        self.p_prop = (cfg.hover_power_w * (1.0 + cfg.accel_power_coef * float(np.linalg.norm(a)) / G)
                       + self.cd * vair ** 3 + cfg.mass_kg * G * max(self.vel[2], 0.0) / cfg.climb_eff)
        used = (self.p_prop + cfg.payload_power_w + comm_w) * drain_mult * dt / 3600.0
        self.energy_wh -= used
        self.e_prop_wh += self.p_prop * dt / 3600.0
        soc = self.soc()
        if self.energy_wh <= 0.0:
            self.mode = "CRASHED"
            ev.append(("mode", {"mode": "CRASHED", "reason": "battery_depleted"}))
        elif soc < cfg.crit_soc and self.mode != "LAND":
            ev += self.enter_land("battery_critical")
        elif soc < cfg.rtl_soc and self.mode in ("MISSION", "TAKEOFF", "HOLD"):
            ev += self.enter_rtl("battery_low")
        return ev

    def forecast(self, horizon: float, step: float) -> np.ndarray:
        """Planned positions at step, 2*step, ... horizon (nominal speed, no wind)."""
        n = max(1, int(round(horizon / step)))
        pos = self.pos
        if self.mode in ("HOLD", "LAND", "LANDED", "CRASHED") or self.idx >= len(self.path):
            return np.tile(pos, (n, 1))
        cur = np.array([pos[0], pos[1], self._z2a(pos[0], pos[1], pos[2])])
        idx, arrived, pend = self.idx, self._arrived, self.hold_left
        out = []
        for _ in range(n):
            rem = step
            while rem > 1e-9 and idx < len(self.path):
                node = self.path[idx]
                B = np.array(node[:3])
                if arrived:
                    use = min(max(pend, 0.0), rem)
                    pend -= use
                    rem -= use
                    if pend <= 1e-9:
                        idx += 1
                        arrived = False
                    continue
                d = float(np.linalg.norm(B - cur))
                sp = max(node[3], 0.1)
                if d <= sp * rem:
                    rem -= d / sp
                    cur, arrived, pend = B.copy(), True, node[4]
                else:
                    cur = cur + (B - cur) * (sp * rem / d)
                    rem = 0.0
            out.append([cur[0], cur[1], self._a2z(cur[0], cur[1], cur[2])])
        return np.array(out)


class GpsModel:
    def __init__(self, cfg: DroneCfg, pos0: np.ndarray):
        self.cfg = cfg
        self.bias = np.zeros(3)
        self.pos_est = pos0.copy()
        self.vel_est = np.zeros(3)
        self.next_fix = 0.0
        self.drift = np.zeros(3)

    def step(self, t, dt, pos, vel, dropout: bool, bias_fault: np.ndarray, z: np.ndarray):
        c = self.cfg
        a = math.exp(-dt / 60.0)
        self.bias = a * self.bias + math.sqrt(1 - a * a) * c.gps_bias_sigma_m * z[0:3]
        if dropout:
            self.pos_est = self.pos_est + self.vel_est * dt
            self.drift += 0.05 * dt * z[0:3]
            self.pos_est = self.pos_est + self.drift * dt
        elif t >= self.next_fix - 1e-9:
            sig = np.array([c.gps_sigma_h_m, c.gps_sigma_h_m, c.gps_sigma_v_m])
            self.pos_est = pos + self.bias + bias_fault + sig * z[3:6]
            self.vel_est = vel + 0.1 * z[6:9]
            self.next_fix += 1.0 / c.gps_rate_hz
            self.drift[:] = 0.0
        else:
            self.pos_est = self.pos_est + self.vel_est * dt


# =============================================================================
# 7. INTERFERENCE SOURCES
# =============================================================================
class Interferer:
    """Own RNG stream (keyed by name) so on-demand injection never perturbs other draws."""

    def __init__(self, cfg: InterferenceCfg, seed: int, uid: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(2, stable_hash(cfg.name), uid)))
        self.pos0 = np.array(cfg.pos, float)
        self.vel = np.array(cfg.vel, float)
        self.on = False
        self.fspl = fspl_1m_db(cfg.fc_hz)

    def step(self, t: float) -> bool:
        u = self.rng.random()
        self.on = (self.cfg.t0 <= t < self.cfg.t1) and (u < self.cfg.duty)
        return self.on

    def pos(self, t: float) -> np.ndarray:
        return self.pos0 + self.vel * (t - self.cfg.t0)

    def power_mw(self, rx, t: float, fc: float, bw: float) -> float:
        c = self.cfg
        lo, hi = max(c.fc_hz - c.bw_hz / 2, fc - bw / 2), min(c.fc_hz + c.bw_hz / 2, fc + bw / 2)
        if hi > lo:
            fac = (hi - lo) / c.bw_hz
        elif abs(c.fc_hz - fc) < (c.bw_hz + bw):
            fac = db2lin(c.aclr_db)
        else:
            return 0.0
        p = self.pos(t)
        d = max(math.sqrt((p[0] - rx[0]) ** 2 + (p[1] - rx[1]) ** 2 + (p[2] - rx[2]) ** 2), 1.0)
        pl = self.fspl + 10.0 * c.pl_exp * math.log10(d)
        return db2lin(c.eirp_dbm - pl) * fac


# =============================================================================
# 8. FAULT INJECTION
# =============================================================================
FAULT_KINDS = {
    "link_blackout": "link carries nothing and loses sync (params: none)",
    "rssi_offset": "extra attenuation on link, both directions (params: db=20)",
    "antenna_failure": "damaged/detuned antenna, extra loss (params: db=15)",
    "latency_spike": "added one-way latency (params: ms=200)",
    "packet_burst_loss": "extra independent loss probability (params: p=0.5)",
    "radio_brownout": "drone TX power reduction, e.g. voltage sag (params: db=6)",
    "handover_failure": "forces radio-link failure on the link at fault start (params: none)",
    "gps_dropout": "no GPS fixes; dead-reckoning with drift (params: none)",
    "gps_bias": "constant GPS offset (params: east_m, north_m, up_m)",
    "telemetry_delay": "extra delay on measurements reaching the policy (params: ms=300)",
    "policy_stall": "policy not executed (compute hiccup); last decision held (params: none)",
    "battery_sag": "extra energy drain multiplier (params: drain_mult=1.8)",
    "wind_gust": "step wind disturbance (params: east=5, north=0)",
    "interference_burst": "spawn interferer at drone (params: eirp_dbm=33, fc_hz, bw_hz, duty=1, offset_m=60)",
}


class LinkFx:
    __slots__ = ("blackout", "loss_db", "lat_ms", "loss_p", "tx_db")

    def __init__(self):
        self.reset()

    def reset(self):
        self.blackout, self.loss_db, self.lat_ms, self.loss_p, self.tx_db = False, 0.0, 0.0, 0.0, 0.0


class GlobalFx:
    def __init__(self):
        self.reset()

    def reset(self):
        self.gps_dropout, self.policy_stall = False, False
        self.gps_bias = np.zeros(3)
        self.tel_delay_ms, self.drain_mult = 0.0, 1.0


@dataclass
class ActiveFault:
    id: int
    spec: FaultCfg
    t0: float
    t1: float
    started: bool = False
    ended: bool = False
    origin: str = "scheduled"


class FaultManager:
    def __init__(self, scheduled: List[FaultCfg], processes: List[FaultProcessCfg],
                 rng: np.random.Generator, duration_s: float, link_names: List[str]):
        self.link_names = list(link_names)
        self.faults: List[ActiveFault] = []
        self._next_id = 0
        self.fx = {n: LinkFx() for n in link_names}
        self.glob = GlobalFx()
        self.wind_extra = np.zeros(2)
        for f in scheduled:
            self._add(f, f.t0, "scheduled")
        for p in processes:
            if p.kind not in FAULT_KINDS:
                raise ValueError(f"unknown fault kind {p.kind}")
            t, tend = p.t_from, min(p.t_to if p.t_to is not None else duration_s, duration_s)
            while p.rate_per_min > 0:
                t += rng.exponential(60.0 / p.rate_per_min)
                if t >= tend:
                    break
                dur = clamp(rng.exponential(p.duration_mean_s), p.duration_min_s, p.duration_max_s)
                prm = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in p.params_range.items()}
                link = p.link if p.link != "any" else self.link_names[int(rng.integers(len(self.link_names)))]
                self._add(FaultCfg(kind=p.kind, t0=t, duration_s=dur, link=link, params=prm), t, "process")

    def _add(self, spec: FaultCfg, t0: float, origin: str) -> ActiveFault:
        af = ActiveFault(self._next_id, spec, t0, t0 + spec.duration_s, origin=origin)
        self._next_id += 1
        self.faults.append(af)
        return af

    def inject(self, spec: FaultCfg, now: float) -> int:
        if spec.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {spec.kind}; known: {sorted(FAULT_KINDS)}")
        return self._add(dataclasses.replace(spec, t0=now), now, "on_demand").id

    def _targets(self, link: str) -> List[str]:
        if link == "*":
            return self.link_names
        return [x.strip() for x in link.split(",") if x.strip() in self.fx]

    def update(self, t: float):
        started, ended, active = [], [], []
        for af in self.faults:
            if not af.started and t >= af.t0 - 1e-9:
                af.started = True
                started.append(af)
            if af.started and not af.ended:
                if t >= af.t1 - 1e-9:
                    af.ended = True
                    ended.append(af)
                else:
                    active.append(af)
        for fx in self.fx.values():
            fx.reset()
        g = self.glob
        g.reset()
        self.wind_extra[:] = 0.0
        for af in active:
            s, k, p = af.spec, af.spec.kind, af.spec.params
            if k in ("link_blackout", "rssi_offset", "antenna_failure", "latency_spike",
                     "packet_burst_loss", "radio_brownout"):
                for n in self._targets(s.link):
                    fx = self.fx[n]
                    if k == "link_blackout":
                        fx.blackout = True
                    elif k == "rssi_offset":
                        fx.loss_db += p.get("db", 20.0)
                    elif k == "antenna_failure":
                        fx.loss_db += p.get("db", 15.0)
                    elif k == "latency_spike":
                        fx.lat_ms += p.get("ms", 200.0)
                    elif k == "packet_burst_loss":
                        fx.loss_p = 1 - (1 - fx.loss_p) * (1 - p.get("p", 0.5))
                    elif k == "radio_brownout":
                        fx.tx_db -= p.get("db", 6.0)
            elif k == "gps_dropout":
                g.gps_dropout = True
            elif k == "gps_bias":
                g.gps_bias = g.gps_bias + np.array([p.get("east_m", 0.0), p.get("north_m", 0.0), p.get("up_m", 0.0)])
            elif k == "telemetry_delay":
                g.tel_delay_ms += p.get("ms", 300.0)
            elif k == "policy_stall":
                g.policy_stall = True
            elif k == "battery_sag":
                g.drain_mult *= p.get("drain_mult", 1.8)
            elif k == "wind_gust":
                self.wind_extra += np.array([p.get("east", 5.0), p.get("north", 0.0)])
        self.n_active = len(active)
        return started, ended


# =============================================================================
# 9. LINK RUNTIME (RF + MAC-ish state machine + traffic outcomes)
# =============================================================================
class _Path:
    __slots__ = ("sid", "shadow", "gre", "gim", "rsrp", "d", "obst", "blocked", "u", "pl_const",
                 "fade_db", "sig_dl", "sig_ul", "sig_dl_slow", "sinr_dl", "sinr_ul")

    def __init__(self, sid):
        self.sid = sid
        self.shadow = self.gre = self.gim = 0.0
        self.rsrp = None
        self.d = None
        self.obst, self.blocked = 0.0, False
        self.u = (0.0, 0.0, 1.0)
        self.pl_const = 0.0
        self.fade_db = self.sig_dl = self.sig_ul = self.sig_dl_slow = 0.0
        self.sinr_dl = self.sinr_ul = -40.0


class LinkRuntime:
    U_CTRL, U_VID = 6, 48

    def __init__(self, cfg: LinkCfg, env: Environment, rng: RngBank, scn: ScenarioCfg):
        self.cfg, self.name = cfg, cfg.name
        tc, se = scn.traffic, scn.sensing
        self.lam = C_LIGHT / cfg.freq_hz
        self.fspl = fspl_1m_db(cfg.freq_hz)
        self.rc, self.rp, self.rs = rng.chan[cfg.name], rng.pkt[cfg.name], rng["switch"]
        sids = cfg.station_ids or list(env.stations)
        self.paths: Dict[str, _Path] = {}
        for sid in sids:
            p = _Path(sid)
            p.shadow = cfg.shadow_sigma_los_db * float(self.rc.standard_normal())
            z = self.rc.standard_normal(2)
            p.gre, p.gim = float(z[0]) * 0.7071, float(z[1]) * 0.7071
            self.paths[sid] = p
        self.serving = sids[0]
        self.noise_dbm = noise_dbm(cfg.bw_hz, cfg.nf_db) + cfg.ambient_noise_rise_db
        self.noise_mw = db2lin(self.noise_dbm)
        self.state = "cold"
        self.countdown = self.rlf_timer = self.ho_timer = 0.0
        self.interrupt_until = -1.0
        self.ho_cand: Optional[str] = None
        self.force_rlf = False
        self.effs = sorted(cfg.mcs_eff)
        self.eff = self.effs[0]
        self.ra_ewma: Optional[float] = None
        self.ra_next = 0.0
        self.sinr_dl = self.sinr_ul = -40.0
        self.rx_dl = self.rx_ul = -150.0
        self.fade_db = self.obst = 0.0
        self.i_dl_mw = 0.0
        self.per_ctrl = self.per_vid = 1.0
        self.cap_bps = 0.0
        self.backlog_b = 0.0
        self.util = 0.0
        self.phy_out = True
        self.u = np.ones(self.U_CTRL + self.U_VID + 1)
        self.zj = 0.0
        self.ctrl_bits = (tc.ctrl_bytes + tc.hdr_bytes) * 8
        offered = tc.ctrl_hz * self.ctrl_bits
        self.ctrl_cap = cfg.bw_hz * cfg.ctrl_eff * cfg.mac_eff
        self.ctrl_cap_factor = min(1.0, self.ctrl_cap / offered)
        self.sync_floor = mcs_threshold_db(self.effs[0]) - cfg.sync_margin_db
        self.out_thr = mcs_threshold_db(cfg.ctrl_eff) - 2.0
        self.rssi_floor_dbm = self.noise_dbm + mcs_threshold_db(cfg.ctrl_eff) + 3.0
        # measurement pipeline
        H = se.hist_len
        self.acc_r = self.acc_s = 0.0
        self.acc_n = 0
        self.pending: deque = deque()
        self.h_rssi = deque([NAN] * H, maxlen=H)
        self.h_snr = deque([NAN] * H, maxlen=H)
        self.h_t = deque([0.0] * H, maxlen=H)
        self.ack = NAN
        self.last_ctrl_t = -1e9
        self.v_sent_b = 0.0

    # ---- RF -------------------------------------------------------------
    def update_rf(self, tick, dt, t, pos, speed, axis, env: Environment, rain, fx: LinkFx,
                  interferers: list, geom_div: int):
        cfg = self.cfg
        px, py, pz = pos
        z_all = self.rc.standard_normal(3 * len(self.paths))
        self.u = self.rp.random(self.U_CTRL + self.U_VID + 1)
        self.zj = float(self.rp.standard_normal())
        i_dl = sum(i.power_mw(pos, t, cfg.freq_hz, cfg.bw_hz) for i in interferers) if interferers else 0.0
        self.i_dl_mw = i_dl
        geom = (tick % geom_div == 0)
        pos_arr = None
        fd = max(speed / self.lam, 2.0)
        xf = 2.0 * math.pi * fd * dt
        rf = math.exp(-xf * xf / 4.0)
        rf_s = math.sqrt(1 - rf * rf)
        rho = math.exp(-(speed * dt) / cfg.shadow_decorr_m - dt / cfg.shadow_tau_s)
        rho_s = math.sqrt(1 - rho * rho)
        for i, (sid, ps) in enumerate(self.paths.items()):
            zs, z1, z2 = z_all[3 * i], z_all[3 * i + 1], z_all[3 * i + 2]
            st = env.stations[sid]
            if geom or ps.d is None:
                if pos_arr is None:
                    pos_arr = np.array([px, py, pz])
                dx, dy, dz = px - st[0], py - st[1], pz - st[2]
                d = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1.0)
                ps.d = d
                ps.obst = env.obstruction_db(st, pos_arr, cfg.freq_hz)
                ps.blocked = ps.obst > 3.0
                ps.u = (-dx / d, -dy / d, -dz / d)       # drone -> station
                ps.pl_const = self.fspl + 10.0 * cfg.pl_exp * math.log10(d) + ps.obst
            sg = cfg.shadow_sigma_nlos_db if ps.blocked else cfg.shadow_sigma_los_db
            ps.shadow = rho * ps.shadow + rho_s * sg * zs
            ps.gre = rf * ps.gre + rf_s * 0.7071 * z1
            ps.gim = rf * ps.gim + rf_s * 0.7071 * z2
            if ps.blocked:
                K = 0.0
            else:
                K = db2lin(cfg.rician_k_db)
            los, sc = math.sqrt(K / (K + 1.0)), math.sqrt(1.0 / (K + 1.0))
            h2 = (los + sc * ps.gre) ** 2 + (sc * ps.gim) ** 2
            ps.fade_db = 10.0 * math.log10(max(h2, 1e-6))
            if cfg.ant_pattern == "dipole":
                c = axis[0] * ps.u[0] + axis[1] * ps.u[1] + axis[2] * ps.u[2]
                g_dr = cfg.drone_gain_dbi + 10.0 * math.log10(max(1.0 - c * c, 0.01))
            else:
                g_dr = cfg.drone_gain_dbi
            rain_db = rain_gamma_db_per_km(cfg.freq_hz, rain) * min(ps.d / 1000.0, 5.0)
            common = ps.shadow + ps.fade_db - (ps.pl_const + rain_db + fx.loss_db) + g_dr + cfg.gs_gain_dbi
            ps.sig_dl = cfg.dl_tx_dbm + common
            ps.sig_ul = cfg.ul_tx_dbm + fx.tx_db + common
            ps.sig_dl_slow = ps.sig_dl - ps.fade_db
            i_ul = sum(i.power_mw(st, t, cfg.freq_hz, cfg.bw_hz) for i in interferers) if interferers else 0.0
            ps.sinr_dl = ps.sig_dl - lin2db(self.noise_mw + i_dl)
            ps.sinr_ul = ps.sig_ul - lin2db(self.noise_mw + i_ul)
        events = []
        # serving station / handover (A3 + time-to-trigger)
        if len(self.paths) > 1:
            for ps in self.paths.values():
                ps.rsrp = ps.sig_dl_slow if ps.rsrp is None else 0.9 * ps.rsrp + 0.1 * ps.sig_dl_slow
            if tick == 0:
                self.serving = max(self.paths.values(), key=lambda p: p.rsrp).sid
            cand = max(self.paths.values(), key=lambda p: p.rsrp)
            if cand.sid != self.serving and cand.rsrp > self.paths[self.serving].rsrp + cfg.ho_hyst_db:
                self.ho_timer = self.ho_timer + dt if self.ho_cand == cand.sid else dt
                self.ho_cand = cand.sid
                if self.ho_timer >= cfg.ho_ttt_s:
                    events.append(("handover", {"link": self.name, "from": self.serving, "to": cand.sid}))
                    self.serving, self.ho_timer, self.ho_cand = cand.sid, 0.0, None
                    self.interrupt_until = t + cfg.ho_interrupt_s
                    if self.state == "warm" and self.rs.random() < cfg.ho_fail_p:
                        self.force_rlf = True
            else:
                self.ho_timer, self.ho_cand = 0.0, None
        sp = self.paths[self.serving]
        self.sinr_dl, self.sinr_ul = sp.sinr_dl, sp.sinr_ul
        self.rx_dl, self.rx_ul, self.fade_db, self.obst = sp.sig_dl, sp.sig_ul, sp.fade_db, sp.obst
        # rate adaptation (delayed/filtered SINR + margin => realistic mismatch)
        self.ra_ewma = sp.sinr_ul if self.ra_ewma is None else 0.9 * self.ra_ewma + 0.1 * sp.sinr_ul
        if t >= self.ra_next - 1e-9:
            self.ra_next = t + 0.1
            est = self.ra_ewma - cfg.ra_margin_db
            best = self.effs[0]
            for e in self.effs:
                if mcs_threshold_db(e) <= est:
                    best = e
            self.eff = best
        self.cap_bps = cfg.bw_hz * self.eff * cfg.mac_eff
        self.per_ctrl = per_from_sinr(self.sinr_dl, cfg.ctrl_eff, self.ctrl_bits)
        self.per_vid = per_from_sinr(self.sinr_ul, self.eff, 8.0 * 1400)
        self.phy_out = fx.blackout or self.sinr_dl < self.out_thr
        self.acc_r += db2lin(sp.sig_dl)
        self.acc_s += db2lin(self.sinr_dl)
        self.acc_n += 1
        return events

    # ---- association / activation state machine ---------------------------
    def update_state(self, t, dt, want_warm: bool, fx: LinkFx):
        ev = []
        cfg = self.cfg
        sync_ok = (min(self.sinr_dl, self.sinr_ul) > self.sync_floor) and not fx.blackout
        if self.state == "cold":
            if want_warm:
                self.state = "warming"
                self.countdown = cfg.activation_delay_s * (0.8 + 0.4 * float(self.rs.random()))
                ev.append(("link_warming", {"link": self.name, "delay_s": round(self.countdown, 3)}))
        elif self.state == "warming":
            if not want_warm:
                self.state = "cold"
            elif sync_ok:
                self.countdown -= dt
                if self.countdown <= 0:
                    self.state, self.rlf_timer = "warm", 0.0
                    ev.append(("link_ready", {"link": self.name}))
        elif self.state == "warm":
            if not want_warm:
                self.state = "cold"
                ev.append(("link_released", {"link": self.name}))
            else:
                self.rlf_timer = 0.0 if sync_ok else self.rlf_timer + dt
                if self.force_rlf or self.rlf_timer >= cfg.rlf_timeout_s:
                    self.state, self.rlf_timer, self.force_rlf = "cold", 0.0, False
                    ev.append(("rlf", {"link": self.name}))
        return ev

    def usable(self, t: float, fx: LinkFx) -> bool:
        return self.state == "warm" and t >= self.interrupt_until and not fx.blackout

    # ---- measurements seen by the policy ------------------------------
    def meas_emit(self, t, nz: np.ndarray, sigma: float, delay_s: float):
        if self.state == "warm" and self.acc_n > 0:
            r = round(lin2db(self.acc_r / self.acc_n) + sigma * float(nz[0]))
            s = round(lin2db(self.acc_s / self.acc_n) + sigma * float(nz[1]))
        else:
            r = s = NAN
        self.pending.append((t + delay_s, t, float(r), float(s)))
        self.acc_r = self.acc_s = 0.0
        self.acc_n = 0

    def meas_flush(self, t):
        while self.pending and self.pending[0][0] <= t + 1e-9:
            _, ts, r, s = self.pending.popleft()
            self.h_rssi.append(r)
            self.h_snr.append(s)
            self.h_t.append(ts)

    def note_ack(self, t, ok: bool):
        v = 1.0 if ok else 0.0
        self.ack = v if math.isnan(self.ack) or t - self.last_ctrl_t > 2.0 else 0.95 * self.ack + 0.05 * v
        self.last_ctrl_t = t

    # ---- traffic -------------------------------------------------------
    def ctrl_attempt(self, t, tc: TrafficCfg, fx: LinkFx) -> Optional[float]:
        """Returns one-way latency (ms) of the control packet if delivered, else None."""
        if not self.usable(t, fx):
            return None
        cfg = self.cfg
        per = 1.0 - (1.0 - self.per_ctrl) * (1.0 - fx.loss_p) * self.ctrl_cap_factor
        k = -1
        for j in range(min(cfg.max_attempts, self.U_CTRL)):
            if self.u[j] >= per:
                k = j
                break
        if k < 0:
            return None
        jit = cfg.jitter_ms * math.exp(0.6 * self.zj - 0.18)
        mac = -math.log(max(1.0 - self.u[-1], 1e-9)) * cfg.mac_delay_ms * (0.2 + self.util)
        ser = self.ctrl_bits / max(self.ctrl_cap, 1.0) * 1000.0
        return cfg.base_latency_ms + fx.lat_ms + jit + mac + k * cfg.retx_delay_ms + ser

    def video_tx(self, t, dt, tc: TrafficCfg, offered_bps: float, fx: LinkFx):
        """One video 'chunk' (dt of video). Returns (n_frag, n_lost, latency_ms)."""
        cfg = self.cfg
        offered_b = offered_bps * (1.0 + tc.fec) * dt / 8.0
        n = max(1, int(math.ceil(offered_b / tc.mtu_bytes)))
        if not self.usable(t, fx):
            self.backlog_b, self.util = 0.0, 0.0
            return n, n, 1e9
        per = 1.0 - (1.0 - self.per_vid) * (1.0 - fx.loss_p)
        eff_cap = max(self.cap_bps * (1.0 - per), 1.0)
        self.backlog_b = max(0.0, self.backlog_b + offered_b - eff_cap * dt / 8.0)
        delay_ms = self.backlog_b * 8.0 / eff_cap * 1000.0
        dropped_b = 0.0
        if delay_ms > tc.max_queue_ms:
            keep = tc.max_queue_ms * eff_cap / 8000.0
            dropped_b, self.backlog_b, delay_ms = self.backlog_b - keep, keep, tc.max_queue_ms
        self.util = clamp(offered_b * 8.0 / dt / max(self.cap_bps, 1.0), 0.0, 1.5)
        p_res = per ** (tc.video_mac_retries + 1)
        m = min(n, self.U_VID)
        lost = float(np.count_nonzero(self.u[self.U_CTRL:self.U_CTRL + m] < p_res)) * (n / m)
        lost = min(n, int(math.ceil(lost)) + int(math.ceil(dropped_b / tc.mtu_bytes)))
        jit = cfg.jitter_ms * math.exp(0.6 * self.zj - 0.18)
        mac = -math.log(max(1.0 - self.u[-1], 1e-9)) * cfg.mac_delay_ms * (0.2 + self.util)
        lat = cfg.base_latency_ms + fx.lat_ms + jit + mac + delay_ms + per / (1 - per + 1e-9) * cfg.retx_delay_ms
        self.v_sent_b += offered_b
        return n, lost, lat

    def video_idle(self):
        self.backlog_b = self.util = 0.0

    def power_w(self, used: bool, offered_ul_bps: float, fx: LinkFx) -> float:
        cfg = self.cfg
        if self.state == "cold":
            return cfg.sleep_w
        if self.state == "warming":
            return cfg.active_w
        if not used:
            return cfg.warm_w
        duty = clamp(offered_ul_bps / max(self.cap_bps * (1.0 - self.per_vid), 1.0), 0.0, 1.0) + 0.02
        ptx = db2lin(cfg.ul_tx_dbm + fx.tx_db) / 1000.0 / cfg.pa_eff * min(duty, 1.0)
        return cfg.active_w + ptx


# =============================================================================
# 10. POLICY API (what a predictor / switching algorithm plugs into)
# =============================================================================
@dataclass
class Decision:
    control_links: List[str] = field(default_factory=list)   # first = primary; >1 = duplication (soft/bonded)
    video_shares: Dict[str, float] = field(default_factory=dict)  # link -> share (normalised); {} = no video
    video_bps: float = 0.0
    warm_links: List[str] = field(default_factory=list)      # kept associated but idle (pre-warm for soft switch)
    tag: str = ""
    outage_prob: Optional[Dict[str, float]] = None           # P(total outage within horizon) per link
    horizon_s: float = 3.0
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkSpec:
    """Public, design-time knowledge about a link (no ground truth)."""
    name: str
    tech: str
    freq_hz: float
    bw_hz: float
    carries_video: bool
    effs: List[float]
    ctrl_eff: float
    mac_eff: float
    noise_dbm: float
    rssi_floor_dbm: float
    base_latency_ms: float
    jitter_ms: float
    retx_delay_ms: float
    exp_lat_ms: float
    activation_delay_s: float
    warm_w: float
    active_w: float
    ul_tx_dbm: float
    pa_eff: float
    stations: List[str]


@dataclass
class PolicyContext:
    links: Dict[str, LinkSpec]
    order: List[str]
    ctrl_deadline_ms: float
    ctrl_hz: float
    video_max_bps: float
    video_q_r0_bps: float
    comm_power_budget_w: float
    spectrum_budget_hz: float
    policy_dt: float
    meas_dt: float
    latency_limits_ms: List[float]


@dataclass
class LinkObs:
    name: str
    state: str                    # cold | warming | warm
    rssi_dbm: float               # last available (NaN if cold / none yet)
    snr_db: float
    rssi_hist: np.ndarray         # oldest->newest, NaN padded
    snr_hist: np.ndarray
    meas_t: np.ndarray
    ack_ratio: float              # recent control ack success on this link (NaN if unused)
    used: bool


@dataclass
class Observation:
    t: float
    pos: np.ndarray               # GPS estimate (noisy, may be stale)
    vel: np.ndarray
    mode: str
    soc: float
    links: Dict[str, LinkObs]
    ctrl_gap_ms: float
    ctrl_ratio: float
    vid_ratio: float
    prev: Decision
    comm_power_w: float
    comm_power_budget_w: float
    meas_dt: float
    forecast: Callable[[float, float], np.ndarray]     # forecast(horizon_s, step_s) -> (n,3) planned positions
    radio_prior: Callable[[str, Any], float]           # nominal DL rx power (dBm) for link at position
    home: np.ndarray


def video_quality(bps: float, r0: float, rmax: float) -> float:
    return math.log1p(max(bps, 0.0) / r0) / math.log1p(rmax / r0)


class Policy:
    name = "policy"

    def __init__(self, **params):
        self.params = params

    def reset(self, ctx: PolicyContext, rng: np.random.Generator) -> None:
        self.ctx, self.rng = ctx, rng

    def initial(self, ctx: PolicyContext) -> Decision:
        L = ctx.order[0]
        V = next((n for n in ctx.order if ctx.links[n].carries_video), None)
        return Decision(control_links=[L], video_shares={V: 1.0} if V else {}, video_bps=2e6, tag="init")

    def decide(self, obs: Observation) -> Decision:
        raise NotImplementedError


def _best_eff(spec: LinkSpec, snr_db: float, margin: float = 2.0) -> float:
    b = spec.effs[0]
    for e in spec.effs:
        if mcs_threshold_db(e) + margin <= snr_db:
            b = e
    return b


class StaticPrimaryPolicy(Policy):
    """No switching at all (lower bound). params: control_link, video_link, video_bps"""
    name = "static_primary"

    def initial(self, ctx):
        c = self.params.get("control_link", ctx.order[0])
        v = self.params.get("video_link", next((n for n in ctx.order if ctx.links[n].carries_video), None))
        self.d = Decision([c], {v: 1.0} if v else {}, self.params.get("video_bps", 3e6), tag="static")
        return self.d

    def decide(self, obs):
        return self.d


class ReactivePolicy(Policy):
    """Switch (hard, to a cold link) only after control health has been bad for hold_ms.
    params: priority (list), hold_ms=400, min_dwell_s=2, cooldown_s=8, video_bps=3e6"""
    name = "reactive"

    def reset(self, ctx, rng):
        super().reset(ctx, rng)
        self.prio = list(self.params.get("priority", ctx.order))
        self.cur = self.prio[0]
        self.t_sw = -1e9
        self.failed: Dict[str, float] = {}

    def _video_for(self, L):
        if self.ctx.links[L].carries_video:
            return {L: 1.0}
        v = next((n for n in self.prio if self.ctx.links[n].carries_video), None)
        return {v: 1.0} if v else {}

    def initial(self, ctx):
        return Decision([self.cur], self._video_for(self.cur), self.params.get("video_bps", 3e6), tag="init")

    def decide(self, obs):
        p = self.params
        hold, dwell, cool = p.get("hold_ms", 400.0), p.get("min_dwell_s", 2.0), p.get("cooldown_s", 8.0)
        tag = "hold"
        if obs.ctrl_gap_ms > hold and obs.t - self.t_sw > dwell:
            self.failed[self.cur] = obs.t
            cands = [L for L in self.prio if L != self.cur and obs.t - self.failed.get(L, -1e9) > cool]
            if cands:
                self.cur, self.t_sw, tag = cands[0], obs.t, "hard_switch"
        return Decision([self.cur], self._video_for(self.cur), p.get("video_bps", 3e6), tag=tag)


class BondingPolicy(Policy):
    """Always-on: control duplicated on every link, video split by estimated capacity.
    params: video_bps_max (default ctx), fec_backoff=0.7"""
    name = "bonding"

    def initial(self, ctx):
        return self._mk(ctx, None)

    def _mk(self, ctx, obs):
        caps = {}
        for L in ctx.order:
            s = ctx.links[L]
            if not s.carries_video:
                continue
            snr = obs.links[L].snr_db if obs is not None and np.isfinite(obs.links[L].snr_db) else 10.0
            caps[L] = max(s.bw_hz * _best_eff(s, snr) * s.mac_eff, 1e3)
        tot = sum(caps.values())
        vmax = self.params.get("video_bps_max", ctx.video_max_bps)
        vb = min(vmax, self.params.get("fec_backoff", 0.7) * tot) if tot else 0.0
        order = sorted(ctx.order, key=lambda n: ctx.links[n].exp_lat_ms)
        return Decision(order, {L: c / tot for L, c in caps.items()} if tot else {}, vb, tag="bond")

    def decide(self, obs):
        return self._mk(self.ctx, obs)


class CostPredictivePolicy(Policy):
    """Reference lightweight predictor + threshold/cost switching logic.

    Forecast: nominal radio map along the PLANNED trajectory + an online residual
    (measured RSSI - prior) blended with a short-term RSSI trend. Interference is
    detected as an SINR deficit vs RSSI-implied SNR. P_out(L) = max_tau Phi(-margin/sigma(tau)).
    Actions by threshold on P_out(primary):  p_warm -> pre-warm best backup,
    p_dup -> duplicate control on backup, p_hard -> switch primary (soft if backup warm).
    Cost J(L) = w_blind*P_out + w_lat*latency_penalty + w_energy*power + switch_pen.
    Control and video roles are chosen independently."""
    name = "cost_predictive"
    DEFAULTS = dict(horizon_s=3.0, step_s=0.25, w_traj=0.6, resid_alpha=0.15, sigma0_db=4.0,
                    sigma_growth_db_per_s=1.5, trend_window_s=1.5, p_warm=0.15, p_dup=0.35, p_hard=0.6,
                    p_release=0.05, dwell_s=3.0, min_gain=0.1, w_blind=10.0, w_q=1.0, w_lat=3.0,
                    w_energy=0.3, switch_pen=0.3, lat_limit_ms=120.0, video_floor_bps=0.5e6,
                    health_gap_ms=150.0, video_bps_max=None)

    def __init__(self, **params):
        bad = set(params) - set(self.DEFAULTS)
        if bad:
            raise ValueError(f"unknown cost_predictive params {sorted(bad)}")
        self.p = {**self.DEFAULTS, **params}
        self.params = params

    def reset(self, ctx, rng):
        super().reset(ctx, rng)
        self.resid = {L: 0.0 for L in ctx.order}
        self.prim = self.vlink = None
        self.t_sw_c = self.t_sw_v = self.t_low = -1e9
        self.warm: set = set()
        self.vmax = self.p["video_bps_max"] or ctx.video_max_bps

    def initial(self, ctx):
        self.prim = min(ctx.order, key=lambda n: ctx.links[n].exp_lat_ms)
        vids = [n for n in ctx.order if ctx.links[n].carries_video]
        self.vlink = vids[0] if vids else None
        return Decision([self.prim], {self.vlink: 1.0} if self.vlink else {}, 2e6, tag="init")

    def _forecast(self, obs, L, pts, taus):
        p, spec, lo = self.p, self.ctx.links[L], obs.links[L]
        prior = np.array([obs.radio_prior(L, q) for q in pts])
        have = lo.state == "warm" and np.isfinite(lo.rssi_dbm)
        deficit = 0.0
        if have:
            err = lo.rssi_dbm - obs.radio_prior(L, obs.pos)
            self.resid[L] += p["resid_alpha"] * (err - self.resid[L])
            if np.isfinite(lo.snr_db):
                deficit = max(0.0, (lo.rssi_dbm - spec.noise_dbm) - lo.snr_db - 3.0)
        pred = prior + self.resid[L]
        sig = p["sigma0_db"] + p["sigma_growth_db_per_s"] * taus
        if have:
            n = max(4, int(p["trend_window_s"] / obs.meas_dt))
            seg, tt = lo.rssi_hist[-n:], lo.meas_t[-n:]
            ok = np.isfinite(seg)
            if ok.sum() >= 4:
                slope = float(np.clip(np.polyfit(tt[ok] - tt[ok][-1], seg[ok], 1)[0], -20.0, 20.0))
                pred = p["w_traj"] * pred + (1 - p["w_traj"]) * (seg[ok][-1] + slope * taus)
        else:
            sig = sig + 3.0
        margin = pred - spec.rssi_floor_dbm - deficit
        pout = float(max(phi(-m / s) for m, s in zip(margin, sig)))
        return pout, pred

    def decide(self, obs):
        p, ctx, t = self.p, self.ctx, obs.t
        pts = obs.forecast(p["horizon_s"], p["step_s"])
        taus = (np.arange(len(pts)) + 1) * p["step_s"]
        pout, qpred, capp = {}, {}, {}
        for L in ctx.order:
            spec = ctx.links[L]
            pout[L], pred = self._forecast(obs, L, pts, taus)
            if spec.carries_video:
                snr = float(np.percentile(pred, 25)) - spec.noise_dbm
                capp[L] = spec.bw_hz * _best_eff(spec, snr, 3.0) * spec.mac_eff
                qpred[L] = video_quality(min(self.vmax, 0.7 * capp[L]), ctx.video_q_r0_bps, ctx.video_max_bps)

        def jc(L, switch=True):
            s = ctx.links[L]
            lat = max(0.0, (s.exp_lat_ms - p["lat_limit_ms"]) / p["lat_limit_ms"])
            cold = 0.0 if obs.links[L].state == "warm" else s.activation_delay_s / p["horizon_s"]
            J = p["w_blind"] * (pout[L] + cold) + p["w_lat"] * lat + p["w_energy"] * s.active_w / ctx.comm_power_budget_w
            return J + (p["switch_pen"] if switch and L != self.prim else 0.0)

        tag = "hold"
        if self.prim is None:
            self.initial(ctx)
        alt = min((L for L in ctx.order if L != self.prim), key=lambda L: jc(L), default=None)
        bad = obs.ctrl_gap_ms > p["health_gap_ms"]
        if alt is not None and (pout[self.prim] >= p["p_hard"] or bad):
            if (t - self.t_sw_c >= p["dwell_s"] or bad) and (jc(alt) + p["min_gain"] < jc(self.prim) or (bad and pout[alt] < pout[self.prim])):
                tag = "soft_switch" if obs.links[alt].state == "warm" else "hard_switch"
                self.prim, self.t_sw_c = alt, t
                alt = min((L for L in ctx.order if L != self.prim), key=lambda L: jc(L), default=None)
        control = [self.prim]
        if alt is not None:
            if pout[self.prim] >= p["p_warm"]:
                self.warm.add(alt)
                self.t_low = -1e9
            if pout[self.prim] >= p["p_dup"]:
                control.append(alt)
                tag = tag if tag != "hold" else "dup"
            if pout[self.prim] < p["p_release"]:
                self.t_low = t if self.t_low < 0 else self.t_low
                if t - self.t_low > p["dwell_s"]:
                    self.warm.clear()
            else:
                self.t_low = -1e9
        self.warm = {L for L in self.warm if L != self.prim}
        # video role
        vids = [L for L in ctx.order if ctx.links[L].carries_video]
        shares, vb = {}, 0.0
        if vids:
            def jv(L):
                return (p["w_blind"] * 0.5 * pout[L] + p["w_q"] * (1 - qpred[L])
                        + p["w_energy"] * ctx.links[L].active_w / ctx.comm_power_budget_w
                        + (p["switch_pen"] if L != self.vlink else 0.0))
            bv = min(vids, key=jv)
            if self.vlink is None or (bv != self.vlink and t - self.t_sw_v >= p["dwell_s"] and jv(bv) + p["min_gain"] < jv(self.vlink)):
                self.vlink, self.t_sw_v = bv, t
            vb = min(self.vmax, 0.7 * capp[self.vlink])
            if pout[self.vlink] >= p["p_dup"]:
                vb = max(p["video_floor_bps"], 0.5 * vb)
            shares = {self.vlink: 1.0}
        return Decision(control, shares, vb, sorted(self.warm), tag, dict(pout), p["horizon_s"],
                        {"resid": {k: round(v, 2) for k, v in self.resid.items()}})


class PredictiveHandoverPolicy(Policy):
    """Adapter for the standalone predictive_handover controller.

    The simulator policy API supplies measured RSSI/SNR histories but not radio
    endpoints, so the standalone predictor defaults to its measurement/trend
    model unless ``handover_config.predictor.use_geometry`` is explicitly set.
    """
    name = "predictive_handover"

    def __init__(self, policy_name: str = "cost", handover_config: Optional[dict] = None,
                 link_specs: Optional[dict] = None, **params):
        if params:
            raise ValueError(f"unknown predictive_handover params {sorted(params)}")
        super().__init__(policy_name=policy_name, handover_config=handover_config or {})
        self.policy_name = policy_name
        self.handover_config = copy.deepcopy(handover_config or {})
        self.link_specs = copy.deepcopy(link_specs or {})
        self._last_sample_t: Dict[str, float] = {}

    def reset(self, ctx: PolicyContext, rng: np.random.Generator) -> None:
        super().reset(ctx, rng)
        import predictive_handover as ph

        valid_cfg = {f.name for f in dataclasses.fields(ph.HandoverConfig)}
        unknown = set(self.handover_config) - valid_cfg
        if unknown:
            raise ValueError(f"unknown handover_config keys {sorted(unknown)}; valid: {sorted(valid_cfg)}")
        section_types = {
            "traffic": ph.TrafficProfile,
            "weights": ph.CostWeights,
            "safety": ph.SafetyMargins,
            "predictor": ph.PredictorConfig,
        }
        for section, section_type in section_types.items():
            values = self.handover_config.get(section, {})
            if not isinstance(values, dict):
                raise ValueError(f"handover_config.{section} must be an object")
            unknown = set(values) - {f.name for f in dataclasses.fields(section_type)}
            if unknown:
                raise ValueError(f"unknown handover_config.{section} keys {sorted(unknown)}")
        cfg = ph.HandoverConfig.from_dict(self.handover_config)
        if "predictor" not in self.handover_config:
            cfg.predictor.use_geometry = False

        specs = []
        valid_link_fields = {f.name for f in dataclasses.fields(ph.LinkSpec)} - {"name"}
        unknown_links = set(self.link_specs) - set(ctx.order)
        if unknown_links:
            raise ValueError(f"link_specs contains unknown simulator links {sorted(unknown_links)}")
        if cfg.predictor.use_geometry:
            missing = [name for name in ctx.order if "endpoint" not in self.link_specs.get(name, {})]
            if missing:
                raise ValueError(f"predictor.use_geometry requires link_specs endpoints for {missing}")
        for name in ctx.order:
            source = ctx.links[name]
            rates = tuple((float(i * 3), max(float(e) * source.bw_hz * source.mac_eff / 1e6, 0.001))
                          for i, e in enumerate(source.effs))
            link_override = self.link_specs.get(name, {})
            if not isinstance(link_override, dict):
                raise ValueError(f"link_specs.{name} must be an object")
            unknown = set(link_override) - valid_link_fields
            if unknown:
                raise ValueError(f"unknown link_specs.{name} keys {sorted(unknown)}")
            link_values = dict(
                name=name,
                band_group=source.tech,
                noise_floor_dbm=source.noise_dbm,
                mcs_table=rates,
                base_latency_ms=source.base_latency_ms,
                supports_video=source.carries_video,
                active_power_w=source.active_w,
                idle_power_w=source.warm_w,
                acquire_time_s=source.activation_delay_s,
                always_warm=False,
            )
            link_values.update(link_override)
            specs.append(ph.LinkSpec(**link_values))
        self.controller = ph.HandoverController(specs, cfg, ph.make_policy(self.policy_name))
        self._ph = ph
        self._last_sample_t = {name: -float("inf") for name in ctx.order}

    def initial(self, ctx: PolicyContext) -> Decision:
        primary = min(ctx.order, key=lambda name: ctx.links[name].exp_lat_ms)
        video = next((name for name in ctx.order if ctx.links[name].carries_video), None)
        active = {primary, video} - {None}
        self.controller.current = self._ph.Assignment(
            control_primary=primary,
            video_primary=video,
            video_level=1 if video else 0,
        )
        for name in active:
            runtime = self.controller.runtime[name]
            runtime.warm = True
            runtime.warming_until = 0.0
        return Decision([primary], {video: 1.0} if video else {}, min(2e6, ctx.video_max_bps), tag="predictive_init")

    def decide(self, obs: Observation) -> Decision:
        ph = self._ph
        self.controller.on_drone_state(ph.DroneState(
            t=float(obs.t),
            pos=tuple(float(x) for x in obs.pos),
            vel=tuple(float(x) for x in obs.vel),
            battery_frac=float(obs.soc),
        ))
        for name, link in obs.links.items():
            for t, rssi, snr in zip(link.meas_t, link.rssi_hist, link.snr_hist):
                t = float(t)
                if t <= self._last_sample_t[name] or not (np.isfinite(rssi) and np.isfinite(snr)):
                    continue
                rssi, snr = float(rssi), float(snr)
                self.controller.on_link_sample(name, ph.LinkSample(
                    t=t,
                    rssi_dbm=rssi,
                    snr_db=snr,
                    noise_dbm=rssi - snr,
                    connected=link.state == "warm",
                ), tuple(float(x) for x in obs.pos))
                self._last_sample_t[name] = t

        result = self.controller.step(float(obs.t))
        assignment = result.assignment
        control = list(assignment.control_links())
        if not control:
            control = list(obs.prev.control_links)

        video_links = []
        if assignment.video_bonded:
            video_links = list(assignment.video_bonded)
            shares = {name: 1.0 / len(video_links) for name in video_links}
        else:
            if assignment.video_primary:
                video_links.append(assignment.video_primary)
            if assignment.video_fallback and assignment.video_fallback not in video_links:
                video_links.append(assignment.video_fallback)
            shares = ({video_links[0]: 0.8, video_links[1]: 0.2}
                      if len(video_links) == 2 else ({video_links[0]: 1.0} if video_links else {}))

        ladder = self.controller.cfg.traffic.video_ladder_kbps
        level = min(max(int(assignment.video_level), 0), len(ladder) - 1)
        video_bps = min(float(ladder[level]) * 1000.0, self.ctx.video_max_bps) if shares else 0.0
        risks = result.diagnostics.get("p_ctrl_fail", {})
        outage_prob = {
            name: float(max(values))
            for name, values in risks.items()
            if values and np.all(np.isfinite(values))
        }
        return Decision(
            control_links=control,
            video_shares=shares,
            video_bps=video_bps,
            warm_links=list(assignment.keep_warm),
            outage_prob=outage_prob or None,
            tag="predictive_handover:" + (result.reasons[-1] if result.reasons else "hold"),
            horizon_s=float(self.controller.cfg.safety.horizon_s),
            info={"predictive_reasons": list(result.reasons), "diagnostics": result.diagnostics},
        )


POLICY_REGISTRY = {c.name: c for c in (
    StaticPrimaryPolicy, ReactivePolicy, BondingPolicy, CostPredictivePolicy, PredictiveHandoverPolicy
)}


def make_policy(spec) -> Policy:
    """spec: Policy instance | 'name' | 'module:Class' | {'name'|'class': ..., 'params': {...}}"""
    if isinstance(spec, Policy):
        return spec
    params = {}
    if isinstance(spec, dict):
        params = dict(spec.get("params", {}))
        spec = spec.get("class") or spec["name"]
    if spec in POLICY_REGISTRY:
        return POLICY_REGISTRY[spec](**params)
    if ":" in spec:
        mod, cls = spec.split(":", 1)
        return getattr(importlib.import_module(mod), cls)(**params)
    raise ValueError(f"unknown policy {spec!r}; built-ins: {sorted(POLICY_REGISTRY)}")


# =============================================================================
# 11. LOGGING
# =============================================================================
class _Tee:
    def __init__(self, f, h):
        self.f, self.h = f, h

    def write(self, s):
        self.h.update(s.encode())
        return self.f.write(s)


def _fmt(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


class RunLogger:
    """level: none (no files) | events (events+decision changes+metrics) | full (+ticks.csv.gz, all policy calls)"""

    def __init__(self, out_dir: str, cols: List[str], level: str):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.level = out_dir, level
        self.ev = open(os.path.join(out_dir, "events.jsonl"), "w")
        self.dec = open(os.path.join(out_dir, "decisions.jsonl"), "w")
        self.h = hashlib.sha256()
        self.tf = self.w = None
        if level == "full":
            self.tf = gzip.open(os.path.join(out_dir, "ticks.csv.gz"), "wt", newline="", compresslevel=4)
            self.w = csv.writer(_Tee(self.tf, self.h))
            self.w.writerow(cols)

    def event(self, rec):
        self.ev.write(cjson(rec) + "\n")

    def decision(self, rec):
        self.dec.write(cjson(rec) + "\n")

    def tick(self, row):
        if self.w:
            self.w.writerow([_fmt(v) for v in row])

    def close(self) -> str:
        for f in (self.ev, self.dec, self.tf):
            if f:
                f.close()
        return self.h.hexdigest()[:16] if self.w else ""


class _Roll:
    """Rolling 0/1 window with O(1) updates."""

    def __init__(self, n: int):
        self.n, self.q, self.s = max(1, n), deque(), 0

    def push(self, v: int):
        self.q.append(v)
        self.s += v
        if len(self.q) > self.n:
            self.s -= self.q.popleft()

    def ratio(self) -> float:
        return self.s / len(self.q) if self.q else 1.0


@dataclass
class RunResult:
    run_id: str
    metrics: Dict[str, Any]
    outcome: str
    out_dir: Optional[str]
    tick_hash: str
    crc: int


# =============================================================================
# 12. SIMULATION CONTROLLER
# =============================================================================
class SimulationController:
    """Closed-loop engine. Deterministic given (scenario, policy spec, seed).

        sim = SimulationController(scenario, policy, seed, out_dir, log_level)
        sim.step(n) / sim.run(until_s) / sim.run_until(pred)
        sim.inject_fault(kind, duration_s, link, **params)      # on demand
        sim.add_hook(fn, every_s=1.0 | at_s=30.0)               # scripted scenario events
        cp = sim.checkpoint(); sim.restore(cp); sim.fork()      # what-if branching
        sim.snapshot(); sim.result(); sim.close()
    """

    def __init__(self, scenario, policy="static_primary", seed: int = 0, out_dir: Optional[str] = None,
                 log_level: str = "events", run_id: Optional[str] = None):
        self.scn = load_scenario(scenario)
        self.seed = int(seed)
        self.policy_spec = policy
        self.out_dir_root = out_dir
        self.log_level = log_level if out_dir else "none"
        pname = policy if isinstance(policy, str) else (policy.get("name") if isinstance(policy, dict) else policy.name)
        self.run_id = run_id or f"{self.scn.name}__{pname}__s{self.seed}"
        self.hooks: List[dict] = []
        self.logger: Optional[RunLogger] = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        s = self.scn
        if self.logger:
            self.logger.close()
        self.dt, self.t, self.tick, self.done, self.outcome = s.dt, 0.0, 0, False, "running"
        self.rng = RngBank(self.seed, [l.name for l in s.links])
        self.env = Environment(s.env, self.seed)
        self.weather = Weather(s.weather, self.rng["weather"])
        self.drone = Drone(s.drone, s.mission, self.env)
        self.links = {l.name: LinkRuntime(l, self.env, self.rng, s) for l in s.links}
        self.interferers = [Interferer(c, self.seed, i) for i, c in enumerate(s.interference)]
        self.faults = FaultManager(s.faults, s.fault_processes, self.rng["faults"], s.duration_s, list(self.links))
        self.gps = GpsModel(s.drone, self.drone.pos)
        self.geom_div = max(1, int(round(s.geometry_period_s / s.dt)))
        self._prior_cache: Dict[tuple, float] = {}
        # policy
        self.policy = make_policy(self.policy_spec)
        self.ctx = self._policy_ctx()
        self.policy.reset(self.ctx, self.rng["policy"])
        self.decision = self._validate(self.policy.initial(self.ctx), first=True)
        for L in self._used_links(self.decision) | set(self.decision.warm_links):
            self.links[L].state = "warm"                          # mission starts with the initial links associated
        self.pending: deque = deque()
        self.next_policy_t = 0.0
        self.next_meas_t = 0.0
        self.next_ctrl_t = 0.0
        tc = s.traffic
        self.w_ctrl = _Roll(int(math.ceil(tc.ctrl_window_s * tc.ctrl_hz)))
        self.w_vid = _Roll(int(math.ceil(tc.vid_window_s / s.dt)))
        self.last_ctrl_ok_t = 0.0
        self.ctrl_gap_ms = 0.0
        self.ctrl_rel = self.vid_rel = True
        self.p_comm_w = sum(self.links[L].cfg.warm_w for L in self.links)
        self.p_link_prev = {L: 0.0 for L in self.links}
        self.vid_consec_fail = 0
        self.failsafe_active = False
        self.fs_t0 = 0.0
        self.fs_rel_t = 0.0
        self._in_blind = self._in_cunrel = self._frozen = False
        self._blind_t0 = self._cunrel_t0 = 0.0
        self.last_primary = self.decision.control_links[0]
        self.last_switch_t = -1e9
        self.prev_primary = None
        self.primary_hist: List[Tuple[float, str]] = [(0.0, self.last_primary)]
        self.pred_records: List[Tuple[float, str, float]] = []
        self.out_phy = {L: bytearray() for L in self.links}
        self.crc = 0
        self.m = dict(
            ctrl_sent=0, ctrl_ok=0, ctrl_late=0, ctrl_lost=0, ctrl_lat=[], lat_le={str(x): 0 for x in s.metrics.latency_limits_ms},
            vid_ok=0, vid_n=0, vq_sum=0.0, vbps_sum=0.0, freeze_t=0.0, freeze_n=0,
            blind_t=0.0, blind_eps=[], ctrl_unrel_t=0.0, ctrl_unrel_eps=[], vid_unrel_t=0.0,
            n_hard=0, n_soft=0, n_pingpong=0, n_video_switch=0, n_dup_ticks=0, n_handover=0, n_rlf=0,
            n_failsafe=0, failsafe_t=0.0, n_faults=0, e_comm_wh=0.0, over_budget_t=0.0, spec_sum=0.0,
            spec_over_t=0.0, e_link={L: 0.0 for L in self.links}, geofence_breaches=0, n_policy=0,
            policy_us=[], warm_ticks=0, events={}, switch_blackout=[],
        )
        self._geo_flag = False
        self._ctrl_loss_run = 0
        if self.out_dir_root:
            self.out_dir = os.path.join(self.out_dir_root, self.run_id)
            self.logger = RunLogger(self.out_dir, self._columns(), self.log_level)
            self._write_manifest(partial=True)
        else:
            self.out_dir = None
            self.logger = None
        self._ev("start", seed=self.seed, policy=self.policy.name, scenario=s.name)

    def _policy_ctx(self) -> PolicyContext:
        specs = {}
        for n, lr in self.links.items():
            c = lr.cfg
            specs[n] = LinkSpec(n, c.tech, c.freq_hz, c.bw_hz, c.carries_video, list(lr.effs), c.ctrl_eff, c.mac_eff,
                                lr.noise_dbm, lr.rssi_floor_dbm, c.base_latency_ms, c.jitter_ms, c.retx_delay_ms,
                                c.base_latency_ms + 1.5 * c.jitter_ms + 1.2 * c.mac_delay_ms + c.retx_delay_ms,
                                c.activation_delay_s, c.warm_w, c.active_w, c.ul_tx_dbm, c.pa_eff, list(lr.paths))
        s = self.scn
        return PolicyContext(specs, [l.name for l in s.links], s.traffic.ctrl_deadline_ms, s.traffic.ctrl_hz,
                             s.traffic.video_max_bps, s.traffic.q_r0_bps, s.budget.comm_power_w, s.budget.spectrum_hz,
                             1.0 / s.sensing.policy_hz, s.sensing.meas_period_s, list(s.metrics.latency_limits_ms))

    def _columns(self) -> List[str]:
        base = ["t", "x", "y", "z", "vx", "vy", "vz", "mode", "soc", "p_prop_w", "p_comm_w", "e_total_wh", "wind_mps",
                "rain_mm_h", "temp_c", "gps_err_m", "faults_active", "interf_on", "tag", "ctrl_links", "video_bps",
                "outage_prob", "ctrl_sent", "ctrl_ok", "ctrl_lat_ms", "ctrl_via", "ctrl_rel", "ctrl_gap_ms", "vid_ok",
                "vid_q", "vid_lat_ms", "vid_rel", "blind", "failsafe"]
        per = ["state", "serving", "rx_dl_dbm", "sinr_dl_db", "sinr_ul_db", "eff", "per_ctrl", "per_vid", "obst_db",
               "cap_bps", "phy_out", "p_w", "rssi_meas"]
        return base + [f"{L}.{c}" for L in self.links for c in per]

    # ------------------------------------------------------------------ helpers
    def _used_links(self, d: Decision) -> set:
        u = set(d.control_links)
        if d.video_bps > 0:
            u |= {L for L, sh in d.video_shares.items() if sh > 0}
        return u

    def _ev(self, _k: str, **kw):
        self.m["events"][_k] = self.m["events"].get(_k, 0) + 1
        if self.logger:
            self.logger.event({"t": round(self.t, 4), "kind": _k, **kw})

    def _validate(self, d: Decision, first: bool = False) -> Decision:
        names = self.links
        cl = []
        for L in d.control_links:
            if L in names and L not in cl:
                cl.append(L)
        if not cl:
            cl = list(self.decision.control_links) if not first else [next(iter(names))]
        vs = {L: max(0.0, float(v)) for L, v in d.video_shares.items() if L in names and names[L].cfg.carries_video}
        tot = sum(vs.values())
        vs = {L: v / tot for L, v in vs.items()} if tot > 0 else {}
        vb = clamp(float(d.video_bps), 0.0, self.scn.traffic.video_max_bps) if vs else 0.0
        warm = [L for L in d.warm_links if L in names]
        return dataclasses.replace(d, control_links=cl, video_shares=vs, video_bps=vb, warm_links=warm)

    def _radio_prior(self, L: str, p) -> float:
        key = (L, int(round(p[0] / 5.0)), int(round(p[1] / 5.0)), int(round(p[2] / 5.0)))
        v = self._prior_cache.get(key)
        if v is not None:
            return v
        lr = self.links[L]
        c = lr.cfg
        pv = np.asarray(p, float)
        best = -1e9
        for sid in lr.paths:
            st = self.env.stations[sid]
            d = max(float(np.linalg.norm(pv - st)), 1.0)
            ob = self.env.obstruction_db(st, pv, c.freq_hz) if self.scn.env.policy_knows_map else 0.0
            pl = lr.fspl + 10.0 * c.pl_exp * math.log10(d) + ob
            best = max(best, c.dl_tx_dbm + c.gs_gain_dbi + c.drone_gain_dbi - 2.0 - pl)
        if len(self._prior_cache) > 100000:
            self._prior_cache.clear()
        self._prior_cache[key] = best
        return best

    def _build_obs(self, t: float) -> Observation:
        links = {}
        used = self._used_links(self.decision)
        for L, lr in self.links.items():
            r = np.fromiter(lr.h_rssi, float, len(lr.h_rssi))
            sn = np.fromiter(lr.h_snr, float, len(lr.h_snr))
            tt = np.fromiter(lr.h_t, float, len(lr.h_t))
            fin = np.flatnonzero(np.isfinite(r))
            last_r = float(r[fin[-1]]) if fin.size and len(r) - fin[-1] <= 5 else NAN
            last_s = float(sn[fin[-1]]) if fin.size and len(r) - fin[-1] <= 5 else NAN
            ack = lr.ack if t - lr.last_ctrl_t <= 2.0 else NAN
            links[L] = LinkObs(L, lr.state if lr.state != "warm" else "warm", last_r, last_s, r, sn, tt, ack, L in used)
        return Observation(t, self.gps.pos_est.copy(), self.gps.vel_est.copy(), self.drone.mode, self.drone.soc(), links,
                           self.ctrl_gap_ms, self.w_ctrl.ratio(), self.w_vid.ratio(), self.decision, self.p_comm_w,
                           self.scn.budget.comm_power_w, self.scn.sensing.meas_period_s, self.drone.forecast,
                           self._radio_prior, self.env.home.copy())

    def _on_fault_start(self, af: ActiveFault, t: float):
        s = af.spec
        self.m["n_faults"] += 1
        self._ev("fault_start", id=af.id, fault=s.kind, link=s.link, dur=round(af.t1 - af.t0, 3), origin=af.origin,
                 params=s.params)
        if s.kind == "handover_failure":
            for n in self.faults._targets(s.link):
                self.links[n].force_rlf = True
        elif s.kind == "interference_burst":
            p = s.params
            fc = p.get("fc_hz", self.links[next(iter(self.links))].cfg.freq_hz)
            off = p.get("offset_m", 60.0)
            pos = self.drone.pos + np.array([off, 0.0, 0.0])
            cfg = InterferenceCfg(name=f"burst{af.id}", kind="jammer", t0=t, t1=t + s.duration_s,
                                  pos=[float(pos[0]), float(pos[1]), float(pos[2])], eirp_dbm=p.get("eirp_dbm", 33.0),
                                  fc_hz=fc, bw_hz=p.get("bw_hz", 40e6), duty=p.get("duty", 1.0))
            self.interferers.append(Interferer(cfg, self.seed, 1000 + af.id))

    # ------------------------------------------------------------------ public API
    def inject_fault(self, kind: str, duration_s: float = 5.0, link: str = "*", **params) -> int:
        """Inject a fault starting at the current simulation time (effective next tick)."""
        return self.faults.inject(FaultCfg(kind=kind, t0=self.t, duration_s=duration_s, link=link,
                                           params={k: float(v) for k, v in params.items()}), self.t)

    def add_hook(self, fn: Callable[["SimulationController"], None], every_s: Optional[float] = None,
                 at_s: Optional[float] = None):
        self.hooks.append({"fn": fn, "every": every_s, "next": at_s if at_s is not None else (0.0 if every_s else None)})

    def step(self, n: int = 1) -> "SimulationController":
        for _ in range(n):
            if self.done:
                break
            self._tick()
        return self

    def run(self, until_s: Optional[float] = None) -> "SimulationController":
        while not self.done and (until_s is None or self.t < until_s - 1e-9):
            self._tick()
        return self

    def run_until(self, pred: Callable[["SimulationController"], bool], max_s: Optional[float] = None):
        while not self.done and not pred(self) and (max_s is None or self.t < max_s):
            self._tick()
        return self

    def checkpoint(self) -> dict:
        lg, hk = self.logger, self.hooks
        self.logger, self.hooks = None, []
        try:
            return copy.deepcopy(self.__dict__)
        finally:
            self.logger, self.hooks = lg, hk

    def restore(self, cp: dict) -> None:
        lg, hk = self.logger, self.hooks
        self.__dict__.update(copy.deepcopy(cp))
        self.logger, self.hooks = lg, hk

    def fork(self) -> "SimulationController":
        """Independent copy (no logging) for what-if branches."""
        other = object.__new__(SimulationController)
        other.__dict__.update(self.checkpoint())
        other.logger, other.hooks, other.out_dir, other.log_level, other.out_dir_root = None, [], None, "none", None
        return other

    def snapshot(self) -> dict:
        d = self.drone
        return dict(t=self.t, pos=d.pos.tolist(), vel=d.vel.tolist(), mode=d.mode, soc=d.soc(), decision=self.decision,
                    links={L: dict(state=l.state, rx_dl=l.rx_dl, sinr_dl=l.sinr_dl, sinr_ul=l.sinr_ul, per_ctrl=l.per_ctrl)
                           for L, l in self.links.items()},
                    ctrl_reliable=self.ctrl_rel, video_reliable=self.vid_rel, blind=(not self.ctrl_rel and not self.vid_rel),
                    done=self.done, outcome=self.outcome)

    # ------------------------------------------------------------------ one tick
    def _tick(self):
        s, dt, t = self.scn, self.dt, self.t
        tc, m = s.traffic, self.m
        # 1) faults / weather / interference
        started, ended = self.faults.update(t)
        for af in started:
            self._on_fault_start(af, t)
        for af in ended:
            self._ev("fault_end", id=af.id, fault=af.spec.kind)
        gfx = self.faults.glob
        wx = self.weather.step(t, dt, self.faults.wind_extra)
        n_int = 0
        for itf in self.interferers:
            n_int += itf.step(t)
        act_int = [i for i in self.interferers if i.on]
        # 2) drone + gps
        dr = self.drone
        for kind, kw in dr.step(dt, wx.wind, wx.temp_c, self.p_comm_w, gfx.drain_mult):
            self._ev(kind, **kw)
        m["e_comm_wh"] += self.p_comm_w * dt / 3600.0
        for L, pw in self.p_link_prev.items():
            m["e_link"][L] += pw * dt / 3600.0
        self.gps.step(t, dt, dr.pos, dr.vel, gfx.gps_dropout, gfx.gps_bias, self.rng["sensors"].standard_normal(9))
        gps_err = float(np.linalg.norm(self.gps.pos_est - dr.pos))
        if s.env.geofence.radius_m > 0 and not self._geo_flag:
            if math.hypot(dr.pos[0] - dr.home[0], dr.pos[1] - dr.home[1]) > s.env.geofence.radius_m:
                self._geo_flag = True
                m["geofence_breaches"] += 1
                self._ev("geofence_breach")
                for k, kw in dr.enter_rtl("geofence"):
                    self._ev(k, **kw)
        # 3) RF
        pos = (float(dr.pos[0]), float(dr.pos[1]), float(dr.pos[2]))
        speed = float(np.linalg.norm(dr.vel))
        axis = (float(dr.axis[0]), float(dr.axis[1]), float(dr.axis[2]))
        dec = self.decision
        want = self._used_links(dec) | set(dec.warm_links)
        for L, lr in self.links.items():
            fx = self.faults.fx[L]
            for k, kw in lr.update_rf(self.tick, dt, t, pos, speed, axis, self.env, wx.rain_mm_h, fx, act_int, self.geom_div):
                if k == "handover":
                    m["n_handover"] += 1
                self._ev(k, **kw)
            for k, kw in lr.update_state(t, dt, L in want, fx):
                if k == "rlf":
                    m["n_rlf"] += 1
                self._ev(k, **kw)
            self.out_phy[L].append(1 if lr.phy_out else 0)
        # 4) measurements (fixed sampling grid & draws => independent of policy)
        if t >= self.next_meas_t - 1e-9:
            self.next_meas_t += s.sensing.meas_period_s
            nz = self.rng["sensors"].standard_normal((len(self.links), 2))
            for i, lr in enumerate(self.links.values()):
                lr.meas_emit(t, nz[i], s.sensing.meas_sigma_db, s.sensing.meas_delay_s + gfx.tel_delay_ms / 1000.0)
        for lr in self.links.values():
            lr.meas_flush(t)
        # 5) policy
        tag_now, pout_now = self.decision.tag, NAN
        if t >= self.next_policy_t - 1e-9:
            self.next_policy_t += 1.0 / s.sensing.policy_hz
            if not gfx.policy_stall:
                obs = self._build_obs(t)
                t0 = time.perf_counter()
                d = self._validate(self.policy.decide(obs))
                m["policy_us"].append((time.perf_counter() - t0) * 1e6)
                m["n_policy"] += 1
                self.pending.append((t + s.sensing.decision_delay_s, d))
                if d.outage_prob:
                    for L, pv in d.outage_prob.items():
                        self.pred_records.append((t, L, float(pv)))
                    pout_now = d.outage_prob.get(d.control_links[0], NAN)
                if self.logger and (self.log_level == "full" or d.tag not in ("hold", "")):
                    self.logger.decision({"t": round(t, 3), "tag": d.tag, "ctrl": d.control_links, "vid": d.video_shares,
                                          "vbps": round(d.video_bps), "warm": d.warm_links, "pout": d.outage_prob,
                                          "info": d.info})
            else:
                self._ev("policy_stalled") if m["events"].get("policy_stalled", 0) == 0 else None
        while self.pending and self.pending[0][0] <= t + 1e-9:
            self._apply_decision(self.pending.popleft()[1], t)
        dec = self.decision
        # 6) traffic
        tr = self._traffic(t, dec)
        # 7) failsafe
        self._failsafe(t, dt)
        # 8) energy / budgets
        used = self._used_links(dec)
        p_tot, spec_hz = 0.0, 0.0
        offered_ul = dec.video_bps * (1 + tc.fec)
        for L, lr in self.links.items():
            sh = dec.video_shares.get(L, 0.0) if dec.video_bps > 0 else 0.0
            pw = lr.power_w(L in used, offered_ul * sh + 5e3, self.faults.fx[L])
            self.p_link_prev[L] = pw
            p_tot += pw
            if lr.state != "cold":
                spec_hz += lr.cfg.bw_hz
        self.p_comm_w = p_tot
        if p_tot > s.budget.comm_power_w:
            m["over_budget_t"] += dt
        if spec_hz > s.budget.spectrum_hz:
            m["spec_over_t"] += dt
        m["spec_sum"] += spec_hz * dt
        if len(dec.control_links) > 1:
            m["n_dup_ticks"] += 1
        # 9) blind-time accounting
        blind = (not self.ctrl_rel) and (not self.vid_rel)
        if blind:
            m["blind_t"] += dt
            if not self._in_blind:
                self._in_blind, self._blind_t0 = True, t
                self._ev("blind_start")
        elif self._in_blind:
            self._in_blind = False
            m["blind_eps"].append(t - self._blind_t0)
            self._ev("blind_end", dur=round(t - self._blind_t0, 3))
        if not self.ctrl_rel:
            m["ctrl_unrel_t"] += dt
            if not self._in_cunrel:
                self._in_cunrel, self._cunrel_t0 = True, t
        elif self._in_cunrel:
            self._in_cunrel = False
            m["ctrl_unrel_eps"].append(t - self._cunrel_t0)
        if not self.vid_rel:
            m["vid_unrel_t"] += dt
        # 10) logging, digest, hooks
        l0 = next(iter(self.links.values()))
        self.crc = zlib.crc32(struct.pack("<6dbb", t, dr.pos[0], dr.pos[1], dr.pos[2], dr.soc(), l0.sinr_dl,
                                          tr["ctrl_ok"], 1 if blind else 0), self.crc)
        if self.logger and self.log_level == "full":
            self.logger.tick(self._row(t, dr, wx, gps_err, n_int, dec, pout_now, tr, blind))
        for h in self.hooks:
            if h["next"] is not None and t >= h["next"] - 1e-9:
                h["next"] = (h["next"] + h["every"]) if h["every"] else None
                h["fn"](self)
        # 11) advance & termination
        self.tick += 1
        self.t = self.tick * dt
        if dr.mode in ("LANDED", "CRASHED"):
            self._finish()
        elif self.t >= s.duration_s - 1e-9:
            self._finish()
        elif dr.mission_done and not s.mission.rtl_at_end:
            self._finish()

    # ------------------------------------------------------------------ decisions
    def _apply_decision(self, d: Decision, t: float):
        prev = self.decision
        new_p, old_p = d.control_links[0], prev.control_links[0]
        if new_p != old_p:
            lr = self.links[new_p]
            kind = "soft" if lr.state == "warm" else "hard"
            self.m["n_soft" if kind == "soft" else "n_hard"] += 1
            pp = (self.prev_primary == new_p and t - self.last_switch_t < self.scn.metrics.pingpong_s)
            if pp:
                self.m["n_pingpong"] += 1
            self._ev("switch", switch_type=kind, frm=old_p, to=new_p, link_state=lr.state, pingpong=pp, tag=d.tag)
            self.prev_primary, self.last_switch_t, self.last_primary = old_p, t, new_p
            self.primary_hist.append((t, new_p))
        vold = max(prev.video_shares, key=prev.video_shares.get) if prev.video_shares else None
        vnew = max(d.video_shares, key=d.video_shares.get) if d.video_shares else None
        if vold != vnew:
            self.m["n_video_switch"] += 1
            self._ev("video_switch", frm=vold, to=vnew)
        self.decision = d

    # ------------------------------------------------------------------ traffic
    def _traffic(self, t: float, dec: Decision) -> dict:
        s, tc, dt, m = self.scn, self.scn.traffic, self.dt, self.m
        c_sent = c_ok = 0
        c_lat, c_via = NAN, ""
        if t >= self.next_ctrl_t - 1e-9:
            self.next_ctrl_t += 1.0 / tc.ctrl_hz
            c_sent = 1
            best = float("inf")
            for L in dec.control_links:
                lat = self.links[L].ctrl_attempt(t, tc, self.faults.fx[L])
                self.links[L].note_ack(t, lat is not None and lat <= tc.ctrl_deadline_ms)
                if lat is not None and lat < best:
                    best, c_via = lat, L
            m["ctrl_sent"] += 1
            if best < float("inf"):
                c_lat = best
                m["ctrl_lat"].append(best)
                if best <= tc.ctrl_deadline_ms:
                    c_ok = 1
                else:
                    m["ctrl_late"] += 1
                for k in m["lat_le"]:
                    if best <= float(k):
                        m["lat_le"][k] += 1
            else:
                m["ctrl_lost"] += 1
            m["ctrl_ok"] += c_ok
            self.w_ctrl.push(c_ok)
            if c_ok:
                self.last_ctrl_ok_t = t
        self.ctrl_gap_ms = (t - self.last_ctrl_ok_t) * 1000.0
        self.ctrl_rel = (self.ctrl_gap_ms <= tc.ctrl_gap_max_ms) and (self.w_ctrl.ratio() >= tc.ctrl_ok_ratio)
        # video chunk
        v_ok, v_lat = 0, NAN
        n_tot = lost = 0
        lat_max = 0.0
        if dec.video_bps > 0 and dec.video_shares:
            for L, lr in self.links.items():
                sh = dec.video_shares.get(L, 0.0)
                if sh > 0:
                    n, l, la = lr.video_tx(t, dt, tc, dec.video_bps * sh, self.faults.fx[L])
                    n_tot, lost, lat_max = n_tot + n, lost + l, max(lat_max, la)
                else:
                    lr.video_idle()
            tol = int(tc.fec / (1.0 + tc.fec) * n_tot)
            v_lat = lat_max
            v_ok = 1 if (lost <= tol and lat_max <= tc.video_deadline_ms) else 0
        else:
            for lr in self.links.values():
                lr.video_idle()
        m["vid_n"] += 1
        m["vid_ok"] += v_ok
        q = video_quality(dec.video_bps, tc.q_r0_bps, tc.video_max_bps) if v_ok else 0.0
        m["vq_sum"] += q
        m["vbps_sum"] += dec.video_bps * v_ok
        self.w_vid.push(v_ok)
        self.vid_rel = self.w_vid.ratio() >= tc.vid_ok_ratio
        self.vid_consec_fail = 0 if v_ok else self.vid_consec_fail + 1
        frozen = self.vid_consec_fail * dt >= tc.freeze_s
        if frozen:
            m["freeze_t"] += dt
            if not self._frozen:
                m["freeze_n"] += 1
        self._frozen = frozen
        return dict(ctrl_sent=c_sent, ctrl_ok=c_ok, ctrl_lat=c_lat, ctrl_via=c_via, vid_ok=v_ok, vid_q=q, vid_lat=v_lat)

    # ------------------------------------------------------------------ failsafe
    def _failsafe(self, t: float, dt: float):
        fs, dr = self.scn.failsafe, self.drone
        if fs.action == "none" or dr.mode not in Drone.AIRBORNE:
            return
        if not self.failsafe_active:
            if self.ctrl_gap_ms / 1000.0 >= fs.timeout_s:
                self.failsafe_active, self.fs_t0, self.fs_rel_t = True, t, 0.0
                self.m["n_failsafe"] += 1
                self._ev("failsafe_triggered", action=fs.action)
                for k, kw in (dr.enter_hold("link_loss") if fs.action == "hold" else dr.enter_rtl("link_loss")):
                    self._ev(k, **kw)
        else:
            self.m["failsafe_t"] += dt
            self.fs_rel_t = self.fs_rel_t + dt if self.ctrl_rel else 0.0
            if fs.action == "hold" and dr.mode == "HOLD" and t - self.fs_t0 > fs.hold_to_rtl_s:
                for k, kw in dr.enter_rtl("link_loss_hold_timeout"):
                    self._ev(k, **kw)
            if fs.resume and self.fs_rel_t >= fs.resume_s and dr.mode in ("HOLD", "RTL") and dr.soc() > self.scn.drone.rtl_soc:
                self.failsafe_active = False
                self._ev("failsafe_cleared")
                for k, kw in dr.resume_mission():
                    self._ev(k, **kw)
            elif dr.mode in ("LAND", "LANDED"):
                self.failsafe_active = False

    # ------------------------------------------------------------------ logging row
    def _row(self, t, dr, wx, gps_err, n_int, dec, pout, tr, blind) -> list:
        row = [t, dr.pos[0], dr.pos[1], dr.pos[2], dr.vel[0], dr.vel[1], dr.vel[2], dr.mode, dr.soc(), dr.p_prop,
               self.p_comm_w, (self.scn.drone.battery_wh * self.scn.drone.start_soc - dr.energy_wh), wx.wind_speed,
               wx.rain_mm_h, wx.temp_c, gps_err, self.faults.n_active, n_int, dec.tag, "+".join(dec.control_links),
               dec.video_bps, pout, tr["ctrl_sent"], tr["ctrl_ok"], tr["ctrl_lat"], tr["ctrl_via"], int(self.ctrl_rel),
               self.ctrl_gap_ms, tr["vid_ok"], tr["vid_q"], tr["vid_lat"], int(self.vid_rel), int(blind), int(self.failsafe_active)]
        for L, lr in self.links.items():
            row += [lr.state, lr.serving, lr.rx_dl, lr.sinr_dl, lr.sinr_ul, lr.eff, lr.per_ctrl, lr.per_vid, lr.obst,
                    lr.cap_bps, int(lr.phy_out), self.p_link_prev.get(L, 0.0), lr.h_rssi[-1] if lr.h_rssi else NAN]
        return row

    # ------------------------------------------------------------------ finish / metrics
    def _finish(self):
        if self.done:
            return
        self.done = True
        dr = self.drone
        if dr.mode == "LANDED":
            self.outcome = "completed" if dr.mission_done else ("aborted_landed_failsafe" if self.m["n_failsafe"] else "aborted_landed")
        elif dr.mode == "CRASHED":
            self.outcome = "crashed_battery"
        elif dr.mission_done:
            self.outcome = "completed"
        else:
            self.outcome = "timeout"
        self._ev("end", outcome=self.outcome)

    def _episodes(self, arr: bytearray) -> List[Tuple[float, float]]:
        mn = int(math.ceil(self.scn.metrics.outage_min_ms / 1000.0 / self.dt))
        eps, i, n = [], 0, len(arr)
        while i < n:
            if arr[i]:
                j = i
                while j < n and arr[j]:
                    j += 1
                if j - i >= mn:
                    eps.append((i * self.dt, j * self.dt))
                i = j
            else:
                i += 1
        return eps

    def _pred_metrics(self) -> dict:
        out = dict(pred_n=len(self.pred_records), pred_brier=NAN, pred_auc=NAN, pred_ep_recall=NAN,
                   pred_lead_mean_s=NAN, pred_false_alarms_per_min=NAN)
        if not self.pred_records:
            return out
        thr = self.scn.metrics.pred_threshold
        H = max(1e-3, getattr(self.decision, "horizon_s", 3.0))
        eps = {L: self._episodes(a) for L, a in self.out_phy.items()}
        ys, ps = [], []
        for (t, L, p) in self.pred_records:
            ys.append(1.0 if any(e0 < t + H and e1 > t for e0, e1 in eps[L]) else 0.0)
            ps.append(p)
        y, p = np.array(ys), np.array(ps)
        out["pred_brier"] = float(np.mean((p - y) ** 2))
        if 0 < y.sum() < len(y):
            r = np.empty(len(p))
            r[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
            for v in np.unique(p):                             # average ranks for ties
                idx = p == v
                r[idx] = r[idx].mean()
            n1, n0 = y.sum(), len(y) - y.sum()
            out["pred_auc"] = float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        # episodes on the active control primary
        hit, tot, leads = 0, 0, []
        ph = self.primary_hist
        for L, el in eps.items():
            for (e0, _) in el:
                prim = [x for (tt, x) in ph if tt <= e0][-1]
                if prim != L:
                    continue
                tot += 1
                w = [tt for (tt, LL, pp) in self.pred_records if LL == L and e0 - H <= tt <= e0 and pp >= thr]
                if w:
                    hit += 1
                    leads.append(e0 - min(w))
        if tot:
            out["pred_ep_recall"], out["pred_lead_mean_s"] = hit / tot, (float(np.mean(leads)) if leads else NAN)
        fa = float(np.sum((p >= thr) & (y == 0))) / max(1.0, len(p)) * (self.policy_hz_total() * 60.0)
        out["pred_false_alarms_per_min"] = fa
        return out

    def policy_hz_total(self) -> float:
        return self.scn.sensing.policy_hz * len(self.links)

    def metrics(self) -> Dict[str, Any]:
        m, s, dr = self.m, self.scn, self.drone
        T = max(self.t, 1e-9)
        lat = np.array(m["ctrl_lat"]) if m["ctrl_lat"] else np.array([NAN])
        bep, cep = list(m["blind_eps"]), list(m["ctrl_unrel_eps"])
        if self._in_blind:
            bep.append(self.t - self._blind_t0)
        if self._in_cunrel:
            cep.append(self.t - self._cunrel_t0)
        ns = max(1, m["ctrl_sent"])
        pu = np.array(m["policy_us"]) if m["policy_us"] else np.array([NAN])
        out = dict(
            run_id=self.run_id, seed=self.seed, policy=self.policy.name, scenario=s.name, outcome=self.outcome,
            mission_completed=int(self.outcome == "completed"), sim_time_s=self.t,
            wp_progress=dr.wp_reached / max(1, len(s.mission.waypoints)), dist_m=dr.dist_m,
            blind_time_s=m["blind_t"], blind_fraction=m["blind_t"] / T, n_blind_episodes=len(bep),
            max_blind_episode_s=max(bep) if bep else 0.0,
            ctrl_unreliable_s=m["ctrl_unrel_t"], ctrl_max_unreliable_episode_s=max(cep) if cep else 0.0,
            video_unreliable_s=m["vid_unrel_t"],
            ctrl_sent=m["ctrl_sent"], ctrl_ontime_rate=m["ctrl_ok"] / ns, ctrl_late_rate=m["ctrl_late"] / ns,
            ctrl_lost_rate=m["ctrl_lost"] / ns,
            ctrl_lat_p50_ms=float(np.nanpercentile(lat, 50)), ctrl_lat_p95_ms=float(np.nanpercentile(lat, 95)),
            ctrl_lat_p99_ms=float(np.nanpercentile(lat, 99)), ctrl_lat_max_ms=float(np.nanmax(lat)),
            video_quality_mean=m["vq_sum"] / max(1, m["vid_n"]), video_decodable_rate=m["vid_ok"] / max(1, m["vid_n"]),
            video_freeze_s=m["freeze_t"], video_freeze_events=m["freeze_n"], video_delivered_mbps=m["vbps_sum"] / max(1, m["vid_n"]) / 1e6,
            energy_total_wh=s.drone.battery_wh * s.drone.start_soc - dr.energy_wh, energy_prop_wh=dr.e_prop_wh,
            energy_comm_wh=m["e_comm_wh"], comm_power_mean_w=m["e_comm_wh"] * 3600.0 / T,
            over_power_budget_s=m["over_budget_t"], spectrum_mean_mhz=m["spec_sum"] / T / 1e6,
            over_spectrum_budget_s=m["spec_over_t"], soc_end=dr.soc(),
            n_hard_switch=m["n_hard"], n_soft_switch=m["n_soft"], n_pingpong=m["n_pingpong"],
            n_video_switch=m["n_video_switch"], dup_time_s=m["n_dup_ticks"] * self.dt,
            n_handover=m["n_handover"], n_rlf=m["n_rlf"], n_failsafe=m["n_failsafe"], failsafe_s=m["failsafe_t"],
            n_faults=m["n_faults"], geofence_breaches=m["geofence_breaches"], clearance_violations=dr.clearance_viol,
            wall_policy_us_mean=float(np.nanmean(pu)), wall_policy_us_p99=float(np.nanpercentile(pu, 99)),
        )
        for k, v in m["lat_le"].items():
            out[f"ctrl_within_{int(float(k))}ms_rate"] = v / ns
        for L, e in m["e_link"].items():
            out[f"energy_{L}_wh"] = e
        out.update(self._pred_metrics())
        return out

    def result(self) -> RunResult:
        return RunResult(self.run_id, self.metrics(), self.outcome, self.out_dir, "", self.crc)

    def _write_manifest(self, partial=False, tick_hash=""):
        if not self.out_dir:
            return
        try:
            src = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
        except Exception:
            src = "?"
        man = dict(run_id=self.run_id, seed=self.seed, policy=self.policy_spec if not isinstance(self.policy_spec, Policy) else self.policy.name,
                   scenario_hash=chash(self.scn), module_version=__version__, module_sha=src, numpy=np.__version__,
                   python=sys.version.split()[0], platform=platform.platform(), argv=sys.argv, partial=partial,
                   tick_hash=tick_hash, state_crc=self.crc, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1, default=_jdefault)
        with open(os.path.join(self.out_dir, "scenario.json"), "w") as f:
            json.dump(dataclasses.asdict(self.scn), f, indent=1, default=_jdefault)

    def close(self) -> RunResult:
        if not self.done:
            self._finish()
        res = self.result()
        if self.logger:
            th = self.logger.close()
            self.logger = None
            res.tick_hash = th
            with open(os.path.join(self.out_dir, "metrics.json"), "w") as f:
                json.dump(res.metrics, f, indent=1, default=_jdefault)
            self._write_manifest(partial=False, tick_hash=th)
        return res


# =============================================================================
# 13. SCENARIO PRESETS
# =============================================================================
LINK_TEMPLATES = {
    "wifi_5g": dict(tech="wifi", freq_hz=5.8e9, bw_hz=20e6, dl_tx_dbm=24.0, ul_tx_dbm=20.0, gs_gain_dbi=12.0,
                    drone_gain_dbi=2.0, nf_db=6.0, pl_exp=2.2, rician_k_db=8.0, mac_eff=0.6, base_latency_ms=6.0,
                    jitter_ms=3.0, mac_delay_ms=2.0, retx_delay_ms=3.0, activation_delay_s=0.4, rlf_timeout_s=1.0,
                    active_w=2.5, warm_w=0.8, pa_eff=0.25, carries_video=True),
    "lte": dict(tech="lte", freq_hz=1.8e9, bw_hz=10e6, dl_tx_dbm=40.0, ul_tx_dbm=23.0, gs_gain_dbi=15.0,
                drone_gain_dbi=2.0, nf_db=5.0, pl_exp=2.5, rician_k_db=6.0, mcs_eff=[0.15, 0.38, 0.88, 1.48, 2.41, 3.32, 4.52, 5.55],
                ctrl_eff=0.38, mac_eff=0.7, base_latency_ms=35.0, jitter_ms=12.0, mac_delay_ms=8.0, retx_delay_ms=8.0,
                activation_delay_s=0.3, rlf_timeout_s=1.0, active_w=2.0, warm_w=0.6, pa_eff=0.3, carries_video=True,
                shadow_sigma_los_db=3.0, shadow_sigma_nlos_db=7.0),
    "sub1ghz": dict(tech="sub1ghz", freq_hz=0.9e9, bw_hz=500e3, dl_tx_dbm=30.0, ul_tx_dbm=27.0, gs_gain_dbi=5.0,
                    drone_gain_dbi=2.0, nf_db=6.0, pl_exp=2.0, rician_k_db=10.0, mcs_eff=[0.1, 0.2, 0.4, 0.8],
                    ctrl_eff=0.2, mac_eff=0.8, base_latency_ms=12.0, jitter_ms=4.0, mac_delay_ms=1.0, retx_delay_ms=10.0,
                    max_attempts=3, activation_delay_s=0.05, rlf_timeout_s=0.5, active_w=0.8, warm_w=0.3,
                    pa_eff=0.35, carries_video=False, shadow_sigma_los_db=1.5, shadow_sigma_nlos_db=5.0),
}


def _link(name, template=None, **kw):
    d = dict(LINK_TEMPLATES[template or name])
    d.update(name=name, **kw)
    return d


def preset_urban_canyon() -> dict:
    return dict(
        name="urban_canyon", duration_s=600.0, dt=0.02,
        env=dict(size_m=[3000, 3000], env_seed=None,
                 terrain=dict(kind="hills", hill_count=4, hill_height_m=[0, 15], hill_sigma_m=[200, 500]),
                 proc_buildings=dict(count=110, footprint_m=[20, 70], height_m=[15, 95], keepout_m=30),
                 stations=[dict(id="gcs", pos=[400, 400, 8], kind="gcs"),
                           dict(id="cell1", pos=[700, 2600, 35], kind="cell"),
                           dict(id="cell2", pos=[2300, 500, 35], kind="cell"),
                           dict(id="cell3", pos=[2700, 2800, 35], kind="cell")],
                 home=[420, 420], geofence=dict(radius_m=3600, max_alt_m=0)),
        drone=dict(battery_wh=95.0),
        mission=dict(takeoff_alt_m=50, default_speed_mps=12, rtl_alt_m=90, agl=True, waypoints=[
            dict(x=700, y=800, z=80), dict(x=1300, y=1000, z=85), dict(x=1900, y=1500, z=90),
            dict(x=2400, y=2200, z=90), dict(x=2500, y=2700, z=80, hold_s=3), dict(x=1700, y=2400, z=80),
            dict(x=1000, y=1600, z=75), dict(x=550, y=800, z=65)]),
        weather=dict(timeline=[dict(t=0, wind_mps=4, wind_dir_deg=250, gust_sigma_mps=1.2, rain_mm_h=0, temp_c=18),
                               dict(t=100, wind_mps=6, wind_dir_deg=260, gust_sigma_mps=1.8, rain_mm_h=0, temp_c=17),
                               dict(t=130, wind_mps=7, wind_dir_deg=270, gust_sigma_mps=2.2, rain_mm_h=12, temp_c=15),
                               dict(t=220, wind_mps=5, wind_dir_deg=270, gust_sigma_mps=1.5, rain_mm_h=0, temp_c=16)]),
        links=[_link("wifi_5g", station_ids=["gcs"]), _link("lte", station_ids=["cell1", "cell2", "cell3"]),
               _link("sub1ghz", station_ids=["gcs"])],
        interference=[dict(name="wifi_burst1", kind="wifi_burst", t0=60, t1=140, pos=[1500, 1200, 10], eirp_dbm=27,
                           fc_hz=5.8e9, bw_hz=20e6, duty=0.35),
                      dict(name="jammer1", kind="jammer", t0=200, t1=235, pos=[2300, 2300, 8], eirp_dbm=33,
                           fc_hz=5.8e9, bw_hz=40e6, duty=1.0),
                      dict(name="lte_cochan", kind="cochannel", t0=150, t1=210, pos=[1500, 1700, 20], eirp_dbm=30,
                           fc_hz=1.8e9, bw_hz=10e6, duty=0.5)],
        fault_processes=[dict(kind="link_blackout", rate_per_min=0.4, duration_mean_s=2.0, duration_max_s=6.0, link="lte"),
                         dict(kind="rssi_offset", rate_per_min=0.5, duration_mean_s=4.0, link="any", params_range=dict(db=[8, 20])),
                         dict(kind="latency_spike", rate_per_min=0.5, duration_mean_s=3.0, link="lte", params_range=dict(ms=[80, 250])),
                         dict(kind="gps_dropout", rate_per_min=0.3, duration_mean_s=4.0, link="*")],
        budget=dict(comm_power_w=8.0, spectrum_hz=40e6),
    )


def preset_rural_ridge() -> dict:
    return dict(
        name="rural_ridge", duration_s=420.0, dt=0.02,
        env=dict(size_m=[6000, 3500], terrain=dict(kind="hills", res_m=25, hill_count=14, hill_height_m=[60, 260],
                                                    hill_sigma_m=[300, 800]),
                 stations=[dict(id="gcs", pos=[300, 1750, 6]), dict(id="cell1", pos=[1200, 600, 45], kind="cell"),
                           dict(id="cell2", pos=[4800, 3000, 45], kind="cell")],
                 home=[320, 1760], geofence=dict(radius_m=6500)),
        mission=dict(takeoff_alt_m=60, default_speed_mps=14, rtl_alt_m=140, waypoints=[
            dict(x=1500, y=1700, z=80), dict(x=3000, y=1900, z=90), dict(x=4500, y=1600, z=100),
            dict(x=5400, y=1800, z=100, hold_s=4), dict(x=3800, y=2400, z=90), dict(x=2000, y=2000, z=80)]),
        weather=dict(timeline=[dict(t=0, wind_mps=8, wind_dir_deg=300, gust_sigma_mps=2.5, rain_mm_h=0, temp_c=5)]),
        links=[_link("wifi_5g", station_ids=["gcs"], gs_gain_dbi=14.0), _link("lte", station_ids=["cell1", "cell2"]),
               _link("sub1ghz", station_ids=["gcs"])],
        interference=[dict(name="wifi_gcs", kind="wifi_burst", t0=90, t1=200, pos=[500, 1700, 8], eirp_dbm=25,
                           fc_hz=5.8e9, bw_hz=20e6, duty=0.25)],
        fault_processes=[dict(kind="gps_dropout", rate_per_min=0.4, duration_mean_s=5.0, link="*"),
                         dict(kind="link_blackout", rate_per_min=0.25, duration_mean_s=2.5, link="any")],
    )


def preset_jammed_inspection() -> dict:
    return dict(
        name="jammed_inspection", duration_s=300.0, dt=0.02,
        env=dict(size_m=[2000, 2000], proc_buildings=dict(count=45, height_m=[10, 60], keepout_m=30),
                 terrain=dict(kind="flat"),
                 stations=[dict(id="gcs", pos=[200, 200, 5]), dict(id="cell1", pos=[1700, 300, 30], kind="cell"),
                           dict(id="cell2", pos=[900, 1800, 30], kind="cell")],
                 home=[220, 220]),
        mission=dict(takeoff_alt_m=40, default_speed_mps=8, rtl_alt_m=70, waypoints=[
            dict(x=500, y=500, z=40), dict(x=1000, y=400, z=50), dict(x=1500, y=900, z=50, hold_s=5),
            dict(x=1500, y=1500, z=45), dict(x=900, y=1200, z=40), dict(x=400, y=1000, z=40)]),
        weather=dict(timeline=[dict(t=0, wind_mps=5, wind_dir_deg=200, gust_sigma_mps=1.5, temp_c=12)]),
        links=[_link("wifi_5g", station_ids=["gcs"]), _link("lte", station_ids=["cell1", "cell2"]),
               _link("sub1ghz", station_ids=["gcs"])],
        interference=[dict(name="sweeper", kind="pulsed_jammer", t0=20, t1=260, pos=[200, 1800, 5], vel=[6, -6, 0],
                           eirp_dbm=36, fc_hz=5.8e9, bw_hz=80e6, duty=0.6),
                      dict(name="lte_jam", kind="jammer", t0=100, t1=140, pos=[1000, 1000, 10], eirp_dbm=32,
                           fc_hz=1.8e9, bw_hz=10e6, duty=1.0)],
        faults=[dict(kind="policy_stall", t0=40, duration_s=3),
                dict(kind="gps_dropout", t0=60, duration_s=15),
                dict(kind="antenna_failure", t0=80, duration_s=12, link="wifi_5g", params=dict(db=25)),
                dict(kind="latency_spike", t0=100, duration_s=10, link="lte", params=dict(ms=220)),
                dict(kind="link_blackout", t0=130, duration_s=8, link="lte"),
                dict(kind="radio_brownout", t0=150, duration_s=20, link="wifi_5g", params=dict(db=8)),
                dict(kind="battery_sag", t0=170, duration_s=30, params=dict(drain_mult=1.8)),
                dict(kind="telemetry_delay", t0=190, duration_s=10, params=dict(ms=400))],
    )


PRESETS: Dict[str, Callable[[], dict]] = {"urban_canyon": preset_urban_canyon, "rural_ridge": preset_rural_ridge,
                                          "jammed_inspection": preset_jammed_inspection}


# =============================================================================
# 14. MONTE-CARLO SCENARIO RANDOMISATION
# =============================================================================
def randomize_scenario(d: dict, spec: dict, seed: int) -> dict:
    """Applies a randomisation spec using a stream that depends ONLY on the trial seed
    (identical for every policy => paired design). Supported keys:
      wind_scale [lo,hi], rain_scale [lo,hi], temp_shift_c [lo,hi], waypoint_jitter_m, alt_jitter_m,
      speed_scale [lo,hi], start_soc [lo,hi], station_jitter_m, link_tx_jitter_db, buildings_count [lo,hi],
      env_reseed (bool), interference {count [lo,hi], kinds [...], eirp_dbm [lo,hi], t_start [lo,hi], dur [lo,hi],
      duty [lo,hi], near_path (bool), links [names] (band to hit)}, fault_processes [FaultProcessCfg dicts]."""
    d = copy.deepcopy(d)
    if not spec:
        return d
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(9,)))
    U = lambda r: float(rng.uniform(r[0], r[1]))
    tl = d.setdefault("weather", {}).setdefault("timeline", [dict(t=0)])
    if "wind_scale" in spec:
        k = U(spec["wind_scale"])
        for w in tl:
            w["wind_mps"] = w.get("wind_mps", 3.0) * k
            w["gust_sigma_mps"] = w.get("gust_sigma_mps", 1.0) * k
    if "rain_scale" in spec:
        k = U(spec["rain_scale"])
        for w in tl:
            w["rain_mm_h"] = w.get("rain_mm_h", 0.0) * k
    if "temp_shift_c" in spec:
        k = U(spec["temp_shift_c"])
        for w in tl:
            w["temp_c"] = w.get("temp_c", 18.0) + k
    wps = d["mission"]["waypoints"]
    if "waypoint_jitter_m" in spec:
        for w in wps:
            w["x"] += float(rng.normal(0, spec["waypoint_jitter_m"]))
            w["y"] += float(rng.normal(0, spec["waypoint_jitter_m"]))
    if "alt_jitter_m" in spec:
        for w in wps:
            w["z"] = max(15.0, w["z"] + float(rng.normal(0, spec["alt_jitter_m"])))
    if "speed_scale" in spec:
        k = U(spec["speed_scale"])
        d["mission"]["default_speed_mps"] = d["mission"].get("default_speed_mps", 10.0) * k
        for w in wps:
            if w.get("speed"):
                w["speed"] *= k
    if "start_soc" in spec:
        d.setdefault("drone", {})["start_soc"] = U(spec["start_soc"])
    if "station_jitter_m" in spec:
        for st in d["env"]["stations"]:
            st["pos"][0] += float(rng.normal(0, spec["station_jitter_m"]))
            st["pos"][1] += float(rng.normal(0, spec["station_jitter_m"]))
    if "link_tx_jitter_db" in spec:
        for l in d["links"]:
            j = float(rng.normal(0, spec["link_tx_jitter_db"]))
            l["ul_tx_dbm"] = l.get("ul_tx_dbm", 20.0) + j
    if "buildings_count" in spec:
        lo, hi = spec["buildings_count"]
        d["env"].setdefault("proc_buildings", {})["count"] = int(rng.integers(lo, hi + 1))
    if spec.get("env_reseed"):
        d["env"]["env_seed"] = None
    ispec = spec.get("interference")
    if ispec:
        n = int(rng.integers(ispec["count"][0], ispec["count"][1] + 1))
        kinds = ispec.get("kinds", ["jammer", "wifi_burst", "pulsed_jammer"])
        size = d["env"]["size_m"]
        d.setdefault("interference", [])
        for i in range(n):
            lk = d["links"][int(rng.integers(len(d["links"])))]
            if ispec.get("links"):
                lk = next(l for l in d["links"] if l["name"] in ispec["links"][int(rng.integers(len(ispec["links"])))])
            if ispec.get("near_path", True):
                w = wps[int(rng.integers(len(wps)))]
                pos = [w["x"] + float(rng.normal(0, 150)), w["y"] + float(rng.normal(0, 150)), 6.0]
            else:
                pos = [float(rng.uniform(0, size[0])), float(rng.uniform(0, size[1])), 6.0]
            t0 = U(ispec.get("t_start", [10, 100]))
            d["interference"].append(dict(
                name=f"rnd{i}", kind=kinds[int(rng.integers(len(kinds)))], t0=t0, t1=t0 + U(ispec.get("dur", [5, 30])),
                pos=pos, eirp_dbm=U(ispec.get("eirp_dbm", [20, 36])), fc_hz=lk.get("freq_hz", 5.8e9),
                bw_hz=lk.get("bw_hz", 20e6) * float(rng.choice([1, 2])), duty=U(ispec.get("duty", [0.3, 1.0]))))
    if spec.get("fault_processes"):
        d.setdefault("fault_processes", []).extend(copy.deepcopy(spec["fault_processes"]))
    return d


# =============================================================================
# 15. SINGLE RUN + BATCH ENGINE
# =============================================================================
def trial_seed(master: int, idx: int) -> int:
    return int(np.random.SeedSequence([int(master), int(idx)]).generate_state(1, dtype=np.uint64)[0] % (2 ** 62))


def run_job(job: dict) -> dict:
    """Top-level (picklable) worker. Deterministic function of the job dict."""
    t0 = time.perf_counter()
    row = dict(run_id=job["run_id"], policy_label=job["label"], trial=job["trial"], seed=job["seed"])
    try:
        sd = load_scenario_dict(job["scenario"])
        sd = deep_merge(sd, job.get("overrides") or {})
        sd = randomize_scenario(sd, job.get("randomize") or {}, job["seed"])
        sim = SimulationController(sd, job["policy"], job["seed"], out_dir=job.get("out_dir"),
                                   log_level=job.get("log_level", "none"), run_id=job["run_id"])
        sim.run()
        res = sim.close()
        row.update(res.metrics)
        row["run_id"], row["policy_label"] = job["run_id"], job["label"]
        row["crc"] = res.crc
        row["tick_hash"] = res.tick_hash
        row["error"] = ""
    except Exception as e:  # keep the batch alive; report failures in the summary
        row["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
    row["wall_s"] = time.perf_counter() - t0
    return row


def expand_policies(specs: List[dict]) -> List[dict]:
    """[{label, name|class, params, sweep:{param:[v..]}}] -> flat list of variants w/ unique labels."""
    out = []
    for sp in specs:
        base = dict(label=sp.get("label", sp.get("name") or sp.get("class")), name=sp.get("name"), cls=sp.get("class"),
                    params=dict(sp.get("params", {})))
        sweep = sp.get("sweep") or {}
        combos = [dict()]
        for k, vals in sweep.items():
            combos = [{**c, k: v} for c in combos for v in vals]
        for c in combos:
            lab = base["label"] + ("[" + ",".join(f"{k}={v}" for k, v in c.items()) + "]" if c else "")
            out.append(dict(label=lab, policy={"class": base["cls"], "params": {**base["params"], **c}} if base["cls"]
                            else {"name": base["name"], "params": {**base["params"], **c}}))
    return out


def make_jobs(cfg: dict, out_dir: Optional[str] = None) -> List[dict]:
    variants = expand_policies(cfg["policies"])
    jobs = []
    for i in range(int(cfg["n_trials"])):
        sd = trial_seed(cfg.get("master_seed", 0), i)
        for v in variants:
            rid = f"{v['label']}__t{i:05d}"
            save = cfg.get("save_ticks", "none")
            lvl = "none"
            if out_dir and (save == "all" or (save.startswith("first:") and i < int(save.split(":")[1]))):
                lvl = "full"
            jobs.append(dict(run_id=rid, label=v["label"], policy=v["policy"], trial=i, seed=sd,
                             scenario=cfg["scenario"], overrides=cfg.get("overrides"), randomize=cfg.get("randomize"),
                             log_level=lvl, out_dir=os.path.join(out_dir, "logs") if (out_dir and lvl != "none") else None))
    return jobs


HEADLINE = ["blind_time_s", "blind_fraction", "max_blind_episode_s", "ctrl_unreliable_s", "ctrl_within_100ms_rate",
            "ctrl_within_150ms_rate", "ctrl_lat_p99_ms", "video_quality_mean", "video_freeze_s", "energy_comm_wh",
            "comm_power_mean_w", "n_hard_switch", "n_soft_switch", "n_pingpong", "n_failsafe", "mission_completed",
            "pred_auc", "pred_ep_recall", "wall_policy_us_mean"]


def _boot_ci(x: np.ndarray, rng, B=2000, alpha=0.05):
    n = len(x)
    if n < 2:
        return (NAN, NAN)
    idx = rng.integers(0, n, (B, n))
    mu = x[idx].mean(axis=1)
    return (float(np.percentile(mu, 100 * alpha / 2)), float(np.percentile(mu, 100 * (1 - alpha / 2))))


def _sign_p(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = int(min((d < 0).sum(), (d > 0).sum()))
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n * 2.0
    return min(1.0, p)


def summarize(rows: List[dict], baseline: Optional[str], master_seed: int = 0) -> dict:
    ok = [r for r in rows if not r.get("error")]
    rng = np.random.default_rng(np.random.SeedSequence(int(master_seed), spawn_key=(11,)))
    labels = sorted({r["policy_label"] for r in ok})
    by = {L: {r["trial"]: r for r in ok if r["policy_label"] == L} for L in labels}
    out = dict(n_rows=len(rows), n_errors=len(rows) - len(ok), policies={}, paired={}, baseline=baseline)
    metrics = [k for k in HEADLINE if any(k in r for r in ok)]
    for L in labels:
        ent = {}
        for k in metrics:
            x = np.array([r.get(k, NAN) for r in by[L].values()], float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            lo, hi = _boot_ci(x, rng)
            ent[k] = dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                          median=float(np.median(x)), p05=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                          ci95=[lo, hi])
        ent["_outcomes"] = {o: sum(1 for r in by[L].values() if r["outcome"] == o) for o in sorted({r["outcome"] for r in by[L].values()})}
        out["policies"][L] = ent
    if baseline and baseline in by:
        for L in labels:
            if L == baseline:
                continue
            ent = {}
            common = sorted(set(by[L]) & set(by[baseline]))
            for k in metrics:
                a = np.array([by[L][i].get(k, NAN) for i in common], float)
                b = np.array([by[baseline][i].get(k, NAN) for i in common], float)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 2:
                    continue
                dlt = a[m] - b[m]
                lo, hi = _boot_ci(dlt, rng)
                bm = float(b[m].mean())
                ent[k] = dict(n=int(m.sum()), mean_diff=float(dlt.mean()), ci95=[lo, hi], rel_change_pct=(100 * dlt.mean() / bm if abs(bm) > 1e-12 else NAN),
                              wins=int((dlt < 0).sum()), losses=int((dlt > 0).sum()), ties=int((dlt == 0).sum()),
                              sign_test_p=_sign_p(dlt), cohen_dz=float(dlt.mean() / dlt.std(ddof=1)) if dlt.std(ddof=1) > 0 else NAN)
            out["paired"][L] = ent
    return out


def summary_markdown(S: dict) -> str:
    L = ["| policy | blind s (mean [95% CI]) | max blind ep s | ctrl <=100ms | ctrl <=150ms | p99 lat ms | video Q | freeze s | comm W | hard/soft sw | completed |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda e, k, f="{:.3f}": (f.format(e[k]["mean"]) if k in e else "-")
    for p, e in S["policies"].items():
        b = e.get("blind_time_s")
        bs = f"{b['mean']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]" if b else "-"
        L.append(f"| {p} | {bs} | {g(e, 'max_blind_episode_s', '{:.2f}')} | {g(e, 'ctrl_within_100ms_rate')} | {g(e, 'ctrl_within_150ms_rate')} | "
                 f"{g(e, 'ctrl_lat_p99_ms', '{:.0f}')} | {g(e, 'video_quality_mean')} | {g(e, 'video_freeze_s', '{:.2f}')} | "
                 f"{g(e, 'comm_power_mean_w', '{:.2f}')} | {g(e, 'n_hard_switch', '{:.1f}')}/{g(e, 'n_soft_switch', '{:.1f}')} | {g(e, 'mission_completed')} |")
    if S.get("paired"):
        L += ["", f"Paired vs **{S['baseline']}** (negative = better for blind time):", "",
              "| policy | d blind s [95% CI] | rel % | wins/losses | sign-test p | dz |", "|---|---|---|---|---|---|"]
        for p, e in S["paired"].items():
            b = e.get("blind_time_s")
            if b:
                L.append(f"| {p} | {b['mean_diff']:+.2f} [{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] | {b['rel_change_pct']:+.1f} | "
                         f"{b['wins']}/{b['losses']} | {b['sign_test_p']:.3g} | {b['cohen_dz']:.2f} |")
    return "\n".join(L)


def run_batch(cfg: dict, out_dir: str, workers: int = 1, resume: bool = False, quiet: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cpath, rpath = os.path.join(out_dir, "batch_config.json"), os.path.join(out_dir, "runs.jsonl")
    chsh = chash(cfg)
    if os.path.exists(cpath):
        old = json.load(open(cpath))
        if chash(old) != chsh and not resume:
            raise SystemExit(f"{out_dir} already holds a different batch (use a new --out)")
        if chash(old) != chsh:
            raise SystemExit("config changed since the batch started; cannot resume")
    json.dump(cfg, open(cpath, "w"), indent=1)
    jobs = make_jobs(cfg, out_dir)
    done_ids, rows = set(), []
    if os.path.exists(rpath):
        if not resume:
            raise SystemExit(f"{rpath} exists; use --resume or a new --out")
        for line in open(rpath):
            r = json.loads(line)
            rows.append(r)
            if not r.get("error"):
                done_ids.add(r["run_id"])
        rows = [r for r in rows if not r.get("error")]
    todo = [j for j in jobs if j["run_id"] not in done_ids]
    manifest = dict(config_hash=chsh, n_jobs=len(jobs), module_version=__version__, numpy=np.__version__,
                    python=sys.version.split()[0], platform=platform.platform(), master_seed=cfg.get("master_seed", 0),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    t0, n = time.perf_counter(), 0
    with open(rpath, "a") as f:
        def sink(r):
            nonlocal n
            rows.append(r)
            f.write(cjson(r) + "\n")
            f.flush()
            n += 1
            if not quiet and (n % max(1, len(todo) // 20) == 0 or n == len(todo)):
                el = time.perf_counter() - t0
                print(f"  [{n}/{len(todo)}] {el:6.0f}s elapsed, ETA {el / n * (len(todo) - n):6.0f}s", file=sys.stderr)
        if workers <= 1:
            for j in todo:
                sink(run_job(j))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(run_job, j) for j in todo]
                for fu in as_completed(futs):
                    sink(fu.result())
    rows.sort(key=lambda r: (r["trial"], r["policy_label"]))
    S = summarize(rows, cfg.get("baseline"), cfg.get("master_seed", 0))
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=_jdefault)
    md = summary_markdown(S)
    open(os.path.join(out_dir, "summary.md"), "w").write(md + "\n")
    cols = sorted({k for r in rows for k in r if k != "error"})
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) if not isinstance(v, (list, dict)) else cjson(v) for k, v in r.items()})
    errs = [r for r in rows if r.get("error")]
    if errs and not quiet:
        print(f"WARNING: {len(errs)} runs failed; first error:\n{errs[0]['error']}", file=sys.stderr)
    return S


# =============================================================================
# 16. PLOT / SELFTEST / CLI
# =============================================================================
def plot_run(run_dir: str, out_png: Optional[str] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with gzip.open(os.path.join(run_dir, "ticks.csv.gz"), "rt") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    num = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
    t = num("t")
    links = sorted({k.split(".")[0] for k in rows[0] if k.endswith(".rx_dl_dbm")})
    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for L in links:
        ax[0].plot(t, num(f"{L}.rx_dl_dbm"), lw=0.5, label=L)
        ax[1].plot(t, num(f"{L}.sinr_dl_db"), lw=0.5, label=L)
    ax[0].set_ylabel("DL rx (dBm)"); ax[0].legend(loc="upper right")
    ax[1].set_ylabel("DL SINR (dB)")
    lat = num("ctrl_lat_ms")
    ax[2].plot(t, lat, ".", ms=1)
    for lim in (100, 150):
        ax[2].axhline(lim, color="r", ls="--", lw=0.6)
    ax[2].set_ylabel("ctrl latency (ms)"); ax[2].set_ylim(0, 400)
    ax[3].plot(t, num("vid_q"), lw=0.5)
    ax[3].set_ylabel("video quality")
    ax[4].plot(t, num("p_comm_w"), lw=0.7); ax[4].set_ylabel("comm power (W)"); ax[4].set_xlabel("t (s)")
    blind = num("blind") > 0.5
    for a in ax:
        a.fill_between(t, *a.get_ylim(), where=blind, color="red", alpha=0.15, step="mid")
    for line in open(os.path.join(run_dir, "events.jsonl")):
        e = json.loads(line)
        if e["kind"] == "switch":
            for a in ax:
                a.axvline(e["t"], color="k", ls=":" if e.get("switch_type") == "soft" else "-", lw=0.6)
    out_png = out_png or os.path.join(run_dir, "timeline.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=110)
    return out_png


def selftest() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    sd = preset_jammed_inspection()
    sd.update(duration_s=40.0, dt=0.05)
    sd["traffic"] = dict(ctrl_hz=20.0)
    a = SimulationController(sd, "cost_predictive", 3).run().result()
    b = SimulationController(sd, "cost_predictive", 3).run().result()
    check("determinism (same seed => identical state CRC & metrics)", a.crc == b.crc and
          all(a.metrics[k] == b.metrics[k] for k in a.metrics if not k.startswith("wall_") and a.metrics[k] == a.metrics[k]))
    c = SimulationController(sd, "cost_predictive", 4).run().result()
    check("different seed => different trajectory/state", a.crc != c.crc)
    # common random numbers across policies
    traces = []
    for pol in ("static_primary", "bonding"):
        s = SimulationController(sd, pol, 3)
        tr = []
        s.add_hook(lambda sim, tr=tr: tr.append(round(sim.links["wifi_5g"].rx_dl, 6)) if sim.t < 8 else None, every_s=0.05)
        s.run(until_s=9)
        traces.append(tr)
    check("CRN: wifi channel realisation identical across policies", len(traces[0]) > 50 and traces[0] == traces[1])
    # checkpoint / restore
    s = SimulationController(sd, "reactive", 5)
    s.run(until_s=10)
    cp = s.checkpoint()
    s.run(until_s=25); crc1 = s.crc
    s.restore(cp); s.run(until_s=25)
    check("checkpoint/restore reproduces future exactly", crc1 == s.crc)
    # fault injection effect
    base = SimulationController(sd, "static_primary", 6, ).run(until_s=30)
    inj = SimulationController(sd, "static_primary", 6)
    inj.run(until_s=10); inj.inject_fault("link_blackout", 6.0, link="wifi_5g"); inj.run(until_s=30)
    check("on-demand blackout increases unreliable time", inj.m["ctrl_unrel_t"] > base.m["ctrl_unrel_t"] + 2.0)
    # soft vs hard switch semantics
    s = SimulationController(sd, "bonding", 7).run(until_s=10)
    check("bonding keeps all links associated (warm)", all(l.state == "warm" for l in s.links.values()))
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def _parse_inject(s: str) -> dict:
    kv = dict(x.split("=", 1) for x in s.split(","))
    d = dict(kind=kv.pop("kind"), t0=float(kv.pop("t")), duration_s=float(kv.pop("dur", 5)), link=kv.pop("link", "*"))
    d["params"] = {k: float(v) for k, v in kv.items()}
    return d


def _cli_policy(arg: str, params: List[str]):
    pr = {}
    for p in params or []:
        k, v = p.split("=", 1)
        try:
            pr[k] = json.loads(v)
        except Exception:
            pr[k] = v
    return {"class": arg, "params": pr} if ":" in arg else {"name": arg, "params": pr}


EXAMPLE_BATCH = dict(
    name="blind_time_study", master_seed=2026, n_trials=100, scenario="urban_canyon", baseline="reactive",
    overrides=dict(dt=0.05, duration_s=600.0, traffic=dict(ctrl_hz=20.0)),
    randomize=dict(wind_scale=[0.5, 1.8], rain_scale=[0.0, 2.0], waypoint_jitter_m=120, alt_jitter_m=10,
                   speed_scale=[0.8, 1.2], start_soc=[0.7, 1.0], station_jitter_m=20, link_tx_jitter_db=1.0,
                   env_reseed=True, buildings_count=[60, 160],
                   interference=dict(count=[0, 3], eirp_dbm=[22, 36], t_start=[20, 250], dur=[5, 40], near_path=True),
                   fault_processes=[dict(kind="link_blackout", rate_per_min=0.3, duration_mean_s=3.0, link="any")]),
    policies=[dict(label="static", name="static_primary", params=dict(control_link="wifi_5g", video_link="wifi_5g")),
              dict(label="reactive", name="reactive", params=dict(priority=["wifi_5g", "sub1ghz", "lte"])),
              dict(label="bonding", name="bonding"),
              dict(label="pred", name="cost_predictive", sweep=dict(p_hard=[0.3, 0.5, 0.7], p_dup=[0.2, 0.4]))],
    save_ticks="first:2",
)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="uavlink_sim", description=__doc__.split("PURPOSE")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-presets")
    p = sub.add_parser("make-example"); p.add_argument("--preset", default="urban_canyon"); p.add_argument("--out", default="scenario.json")
    p = sub.add_parser("make-batch-example"); p.add_argument("--out", default="batch.json")
    p = sub.add_parser("run", help="single run")
    p.add_argument("--scenario", default="urban_canyon"); p.add_argument("--policy", default="cost_predictive")
    p.add_argument("--param", action="append", help="policy param k=v (repeatable)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="runs")
    p.add_argument("--log", default="full", choices=["none", "events", "full"])
    p.add_argument("--inject", action="append", help='"t=40,kind=link_blackout,link=wifi_5g,dur=6,db=20"')
    p.add_argument("--set", action="append", help="override scenario value, e.g. duration_s=120 or drone.battery_wh=60")
    p = sub.add_parser("batch", help="Monte-Carlo batch")
    p.add_argument("--config", required=True); p.add_argument("--out", default="batches/batch")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("replay", help="re-run one batch run with full logs (bit-exact)")
    p.add_argument("--batch", required=True); p.add_argument("--run-id", required=True); p.add_argument("--out", default="replays")
    p = sub.add_parser("plot"); p.add_argument("--run", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)

    if a.cmd == "list-presets":
        for k in PRESETS:
            print(k)
        print("\nfault kinds:")
        for k, v in FAULT_KINDS.items():
            print(f"  {k:20s} {v}")
    elif a.cmd == "make-example":
        json.dump(PRESETS[a.preset](), open(a.out, "w"), indent=1); print("wrote", a.out)
    elif a.cmd == "make-batch-example":
        json.dump(EXAMPLE_BATCH, open(a.out, "w"), indent=1); print("wrote", a.out)
    elif a.cmd == "run":
        sd = load_scenario_dict(a.scenario)
        for kv in a.set or []:
            k, v = kv.split("=", 1)
            try:
                v = json.loads(v)
            except Exception:
                pass
            set_by_path(sd, k, v)
        for inj in a.inject or []:
            sd.setdefault("faults", []).append(_parse_inject(inj))
        sim = SimulationController(sd, _cli_policy(a.policy, a.param), a.seed, out_dir=a.out, log_level=a.log)
        t0 = time.perf_counter()
        sim.run()
        res = sim.close()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()}, indent=1))
        print(f"\nwall {time.perf_counter() - t0:.1f}s  ({sim.tick / (time.perf_counter() - t0):.0f} ticks/s)   logs: {sim.out_dir}")
    elif a.cmd == "batch":
        cfg = json.load(open(a.config))
        S = run_batch(cfg, a.out, a.workers, a.resume)
        print("\n" + summary_markdown(S) + f"\n\nwritten to {a.out}/ (runs.csv, summary.json, summary.md)")
    elif a.cmd == "replay":
        cfg = json.load(open(os.path.join(a.batch, "batch_config.json")))
        job = next((j for j in make_jobs(cfg, None) if j["run_id"] == a.run_id), None)
        if job is None:
            raise SystemExit("run id not found in batch config")
        job.update(log_level="full", out_dir=a.out)
        r = run_job(job)
        prev = next((json.loads(l) for l in open(os.path.join(a.batch, "runs.jsonl")) if json.loads(l)["run_id"] == a.run_id), None)
        print("replayed", a.run_id, "->", os.path.join(a.out, a.run_id))
        if prev:
            print("state CRC identical to original batch run:", prev.get("crc") == r.get("crc"))
    elif a.cmd == "plot":
        print("wrote", plot_run(a.run))
    elif a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)


if __name__ == "__main__":
    main()
