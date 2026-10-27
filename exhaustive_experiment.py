#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exhaustive_experiment.py - Exhaustive / near-exhaustive radio-technology x handover-policy
experiment runner built ON TOP of uavlink_sim.py (simulation) and uavlink_eval.py (metrics).

WHAT IT DOES
    1. TECHNOLOGY CATALOG (21 switchable technologies, see `list-techs`): analog FPV, DJI-style
       digital, Walksnail/HDZero/OpenHD-style digital, 2.4 GHz ISM (control / Wi-Fi), 1.2-1.3 GHz,
       900/868 MHz, 1.7 / 3.3 / 7.2 GHz, LTE, 5G, 5G multi-SIM, Starlink-style LEO, other LEO / GEO /
       L-S-Ku-Ka / Iridium-style satellite, mesh multi-hop RF, fiber tether, plus an AI-compressed
       video mode. Every entry is a plain dict of uavlink_sim.LinkCfg fields (+ role flags), so the
       simulator's radio model is reused unchanged; override any field per technology in the JSON.
    2. COMBINATION SEARCH: generates technology subsets under hard constraints (max simultaneous links,
       forced include / exclude, band-slot conflicts, requirements, power / mass / link-budget /
       latency pre-screens) with full-factorial, stratified, random, pairwise (t-way covering array),
       explicit or successive-halving strategies and a hard safety limit.
    3. POLICIES x COMBOS x SEEDS: every combo is run with each selected policy (Reactive, Bonding,
       Predictive-threshold, Predictive-cost, ... or any "module:Class") on the SAME trial seeds
       (common random numbers: same map, weather, faults, interference and per-technology fading).
    4. RECORDING: append-only ledger (runs.jsonl, full float precision, resumable), results.csv
       (one row per combo+policy+seed, configurable decimals), results.json, aggregates with
       CI/SEM, ranked summaries (blind time, power efficiency, Pareto), paired significance between
       neighbours in the ranking, technology marginal values, resolution/precision report.

USAGE
    python exhaustive_experiment.py list-techs [--budget]
    python exhaustive_experiment.py make-config --preset screening --out study.json
    python exhaustive_experiment.py run --config study.json --dry-run
    python exhaustive_experiment.py run --config study.json --workers 8 --resume
    python exhaustive_experiment.py report --out experiments/study1
    python exhaustive_experiment.py audit  --out experiments/study1 --n 5
    python exhaustive_experiment.py selftest
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import uavlink_integration as integration

__version__ = "1.0.0"
NAN = float("nan")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _import_part(name: str, env: str, required: bool = True):
    """Import uavlink_sim / uavlink_eval by name or from a path given in an env var (so spawned
    worker processes find the same file as the parent)."""
    if name in sys.modules:
        return sys.modules[name]
    p = os.environ.get(env)
    if p and os.path.exists(p):
        spec = importlib.util.spec_from_file_location(name, p)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    try:
        return importlib.import_module(name)
    except ImportError:
        if required:
            raise SystemExit(f"cannot import {name}.py (put it next to this file or set {env}=/path/{name}.py)")
        return None


sim = _import_part("uavlink_sim", "UAVLINK_SIM_PATH")
ev = _import_part("uavlink_eval", "UAVLINK_EVAL_PATH", required=False)


# =============================================================================
# 1. SMALL HELPERS
# =============================================================================
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return jsonable(dataclasses.asdict(o))
    return o


def chash(o, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(jsonable(o), sort_keys=True).encode()).hexdigest()[:n]


def strip_comments(d):
    """Allow '_comment' style keys anywhere in the JSON config."""
    if isinstance(d, dict):
        return {k: strip_comments(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [strip_comments(v) for v in d]
    return d


def nanf(x) -> float:
    return NAN if x is None else float(x)


class Fmt:
    """Number formatting for CSV / tables. fixed: 0.111000 | sci | sig | repr (round-trip exact)."""

    def __init__(self, decimals: int = 6, mode: str = "fixed"):
        self.d, self.mode = int(decimals), mode

    def __call__(self, v) -> str:
        if v is None:
            return "nan"
        if isinstance(v, (bool, np.bool_)):
            return str(int(v))
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            f = float(v)
            if math.isnan(f):
                return "nan"
            if math.isinf(f):
                return "inf" if f > 0 else "-inf"
            if self.mode == "repr":
                return repr(f)
            if self.mode == "sci":
                return f"{f:.{self.d}e}"
            if self.mode == "sig":
                return f"{f:.{self.d}g}"
            return f"{f:.{self.d}f}"
        return str(v)


def _L(*v):
    return field(default_factory=lambda: list(v))


def _D(**kw):
    return field(default_factory=lambda: dict(kw))


# =============================================================================
# 2. RADIO TECHNOLOGY CATALOG
#    Each entry = uavlink_sim.LinkCfg fields (physical model) + role flags + mass + fault profile.
#    Numbers are engineering approximations of typical products; every one can be overridden in the
#    study JSON ("techs.overrides": {tech_id: {LinkCfg field: value}}).  `list-techs --budget` prints
#    the resulting free-space link budgets so you can sanity-check them against your hardware.
# =============================================================================
GROUPS = ["fpv_58", "digital_58", "ism_24", "lr_12", "sub_ghz", "other_bands", "cellular", "satellite", "mesh", "fiber"]


@dataclass
class TechSpec:
    id: str
    label: str
    group: str                       # switch group (GROUPS)
    family: str                      # band-slot family (used by default exclusivity rules)
    stations: str                    # gcs | cell | relay | sat | tether
    control: bool                    # can carry the control stream
    video: bool                      # can carry video
    link: Dict[str, Any]             # uavlink_sim.LinkCfg fields (no name / station_ids / carries_video)
    mass_kg: float = 0.05            # carried mass -> extra hover power in the drone model
    options: Dict[str, Any] = field(default_factory=dict)
    faults: List[Dict[str, Any]] = field(default_factory=list)   # FaultProcessCfg templates (link filled in)
    notes: str = ""


def build_catalog() -> "OrderedDict[str, TechSpec]":
    T = sim.LINK_TEMPLATES
    cat: "OrderedDict[str, TechSpec]" = OrderedDict()

    def add(s: TechSpec):
        cat[s.id] = s

    add(TechSpec("analog_fpv_58", "5.8 GHz analog FPV (video only)", "fpv_58", "slot58", "gcs", False, True,
                 dict(tech="analog_fpv", freq_hz=5.8e9, bw_hz=18e6, dl_tx_dbm=27.0, ul_tx_dbm=27.0, gs_gain_dbi=10.0, drone_gain_dbi=2.0,
                      nf_db=8.0, ambient_noise_rise_db=2.0, pl_exp=2.2, rician_k_db=8.0,
                      mcs_eff=[0.8, 1.27, 1.7, 2.19, 2.7, 3.3, 3.9, 4.55], ctrl_eff=0.8, mac_eff=0.12, ra_margin_db=1.0,
                      base_latency_ms=3.0, jitter_ms=0.5, mac_delay_ms=0.2, retx_delay_ms=1.0, max_attempts=1,
                      activation_delay_s=0.1, rlf_timeout_s=0.5, active_w=3.0, warm_w=1.5, pa_eff=0.3),
                 mass_kg=0.05,
                 notes="No control uplink. Wide rate ladder with low mac_eff => capacity falls smoothly with SNR (graceful degradation), ~3 ms latency."))
    add(TechSpec("digital_58_dji", "5.8 GHz digital, DJI O3/O4-style", "digital_58", "slot58", "gcs", True, True,
                 dict(tech="ocusync", freq_hz=5.8e9, bw_hz=20e6, dl_tx_dbm=27.0, ul_tx_dbm=27.0, gs_gain_dbi=8.0, drone_gain_dbi=3.0,
                      nf_db=5.0, pl_exp=2.2, rician_k_db=8.0, mcs_eff=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], ctrl_eff=0.25,
                      mac_eff=0.65, ra_margin_db=1.5, base_latency_ms=8.0, jitter_ms=2.0, mac_delay_ms=1.5, retx_delay_ms=2.5,
                      max_attempts=4, activation_delay_s=0.5, rlf_timeout_s=1.5, active_w=4.5, warm_w=1.2, pa_eff=0.25),
                 mass_kg=0.08, notes="Very robust control MCS, ~30 ms glass-to-glass video."))
    o = dict(T["wifi_5g"])
    o.update(tech="openhd", dl_tx_dbm=24.0, ul_tx_dbm=24.0, gs_gain_dbi=9.0, mac_eff=0.55, base_latency_ms=10.0, activation_delay_s=0.2,
             ctrl_eff=0.5, active_w=3.5, warm_w=1.0)
    add(TechSpec("digital_58_openhd", "5.8 GHz digital, Walksnail/HDZero/OpenHD-style", "digital_58", "slot58", "gcs", True, True, o,
                 mass_kg=0.10, notes="Wi-Fi-broadcast style link: no association, telemetry/RC passthrough."))
    add(TechSpec("ism_24_ctrl", "2.4 GHz ISM control (ELRS-style, control only)", "ism_24", "slot24", "gcs", True, False,
                 dict(tech="ism24ctrl", freq_hz=2.44e9, bw_hz=800e3, dl_tx_dbm=20.0, ul_tx_dbm=20.0, gs_gain_dbi=3.0, drone_gain_dbi=2.0,
                      nf_db=6.0, ambient_noise_rise_db=4.0, pl_exp=2.1, rician_k_db=10.0, mcs_eff=[0.1, 0.2, 0.4, 0.8], ctrl_eff=0.2,
                      mac_eff=0.8, base_latency_ms=4.0, jitter_ms=1.5, mac_delay_ms=0.5, retx_delay_ms=4.0, max_attempts=2,
                      activation_delay_s=0.05, rlf_timeout_s=0.5, active_w=0.6, warm_w=0.2, pa_eff=0.3, shadow_sigma_los_db=1.5,
                      shadow_sigma_nlos_db=5.0), mass_kg=0.03,
                 faults=[dict(kind="packet_burst_loss", rate_per_min=2.0, duration_mean_s=0.3, duration_min_s=0.05, duration_max_s=1.0,
                              params_range={"p": [0.2, 0.7]})], notes="Crowded band: +4 dB ambient noise, burst losses."))
    o = dict(T["wifi_5g"])
    o.update(tech="wifi24", freq_hz=2.44e9, ul_tx_dbm=20.0, dl_tx_dbm=22.0, gs_gain_dbi=8.0, ambient_noise_rise_db=6.0, pl_exp=2.1,
             mac_eff=0.5, base_latency_ms=8.0, jitter_ms=4.0, activation_delay_s=0.4, active_w=2.5, warm_w=0.8)
    add(TechSpec("ism_24_wifi", "2.4 GHz ISM Wi-Fi-class (control + video)", "ism_24", "slot24", "gcs", True, True, o, mass_kg=0.07,
                 faults=[dict(kind="packet_burst_loss", rate_per_min=3.0, duration_mean_s=0.25, duration_min_s=0.05, duration_max_s=1.0,
                              params_range={"p": [0.3, 0.8]})], notes="Contention / co-channel bursts."))
    add(TechSpec("lr_12", "1.2-1.3 GHz long-range digital (Microhard-class)", "lr_12", "slot12", "gcs", True, True,
                 dict(tech="lr12", freq_hz=1.28e9, bw_hz=8e6, dl_tx_dbm=30.0, ul_tx_dbm=30.0, gs_gain_dbi=8.0, drone_gain_dbi=2.0, nf_db=4.0,
                      pl_exp=2.0, rician_k_db=8.0, shadow_sigma_los_db=2.0, shadow_sigma_nlos_db=5.0,
                      mcs_eff=[0.25, 0.5, 1.0, 2.0, 3.0, 4.0], ctrl_eff=0.5, mac_eff=0.6, base_latency_ms=14.0, jitter_ms=4.0,
                      mac_delay_ms=2.0, retx_delay_ms=4.0, activation_delay_s=1.0, rlf_timeout_s=1.5, active_w=6.5, warm_w=2.0,
                      pa_eff=0.3), mass_kg=0.25, notes="Narrow channel, good penetration and range."))
    o = dict(T["sub1ghz"])
    o.update(tech="sub900")
    add(TechSpec("sub_868_900", "900/868 MHz control (control only)", "sub_ghz", "slot9", "gcs", True, False, o, mass_kg=0.04,
                 notes="Reused sub-GHz template from uavlink_sim."))

    def band(id_, label, fc, bw, ul, dl, gs, pl, nf, act, warm, pa, lat, mass):
        return TechSpec(id_, label, "other_bands", "band_" + id_, "gcs", True, True,
                        dict(tech=id_, freq_hz=fc, bw_hz=bw, dl_tx_dbm=dl, ul_tx_dbm=ul, gs_gain_dbi=gs, drone_gain_dbi=2.0, nf_db=nf,
                             pl_exp=pl, rician_k_db=7.0, mcs_eff=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], ctrl_eff=0.5, mac_eff=0.6,
                             base_latency_ms=lat, jitter_ms=4.0, mac_delay_ms=2.0, retx_delay_ms=4.0, activation_delay_s=0.8,
                             rlf_timeout_s=1.2, active_w=act, warm_w=warm, pa_eff=pa), mass_kg=mass)
    add(band("band_17", "1.7 GHz broadband link", 1.7e9, 10e6, 30.0, 30.0, 8.0, 2.1, 5.0, 5.0, 1.5, 0.30, 12.0, 0.20))
    add(band("band_33", "3.3 GHz broadband link", 3.4e9, 20e6, 27.0, 30.0, 10.0, 2.4, 5.0, 5.0, 1.5, 0.28, 11.0, 0.20))
    add(band("band_72", "7.2 GHz broadband link", 7.2e9, 20e6, 27.0, 29.0, 12.0, 2.3, 6.0, 6.0, 1.8, 0.20, 11.0, 0.25))
    cat["band_72"].link["drone_gain_dbi"] = 3.0
    o = dict(T["lte"])
    o.update(tech="lte")
    add(TechSpec("cell_lte", "Cellular 4G LTE", "cellular", "lte", "cell", True, True, o, mass_kg=0.06,
                 notes="Reused LTE template from uavlink_sim; serving-cell handover with A3/TTT."))
    n5 = dict(tech="nr5g", freq_hz=3.5e9, bw_hz=40e6, dl_tx_dbm=40.0, ul_tx_dbm=23.0, gs_gain_dbi=17.0, drone_gain_dbi=2.0, nf_db=6.0,
              pl_exp=2.7, rician_k_db=5.0, shadow_sigma_los_db=3.0, shadow_sigma_nlos_db=8.0,
              mcs_eff=[0.15, 0.38, 0.88, 1.48, 2.41, 3.32, 4.52, 5.55, 6.5, 7.4], ctrl_eff=0.38, mac_eff=0.75, base_latency_ms=14.0,
              jitter_ms=6.0, mac_delay_ms=4.0, retx_delay_ms=4.0, max_attempts=4, activation_delay_s=0.3, rlf_timeout_s=1.0,
              ho_hyst_db=3.0, ho_ttt_s=0.16, ho_interrupt_s=0.03, ho_fail_p=0.02, active_w=3.5, warm_w=0.9, pa_eff=0.3)
    add(TechSpec("cell_5g", "Cellular 5G NR (n78), single SIM", "cellular", "nr", "cell", True, True, dict(n5), mass_kg=0.08,
                 faults=[dict(kind="latency_spike", rate_per_min=1.5, duration_mean_s=0.5, duration_min_s=0.1, duration_max_s=2.0,
                              params_range={"ms": [30.0, 120.0]})], notes="Higher path loss than LTE, lower latency."))
    add(TechSpec("cell_5g_multisim", "Cellular 5G, multi-SIM bonding (N modems / operators)", "cellular", "nr", "cell", True, True, dict(n5),
                 mass_kg=0.10, options=dict(sims=2),
                 faults=[dict(kind="latency_spike", rate_per_min=1.5, duration_mean_s=0.5, duration_min_s=0.1, duration_max_s=2.0,
                              params_range={"ms": [30.0, 120.0]})],
                 notes="Expands into `sims` independent links (own fading/shadowing, alternate cells = operators): diversity + bonding."))
    add(TechSpec("sat_leo_starlink", "LEO satellite, Starlink-style (Ku, flat-panel)", "satellite", "sat_leo", "sat", True, True,
                 dict(tech="sat_leo", freq_hz=12.0e9, bw_hz=60e6, dl_tx_dbm=43.0, ul_tx_dbm=33.0, gs_gain_dbi=35.0, drone_gain_dbi=24.0,
                      ant_pattern="omni", nf_db=3.0, pl_exp=2.0, rician_k_db=10.0, shadow_sigma_los_db=1.5, shadow_sigma_nlos_db=10.0,
                      shadow_decorr_m=100.0, mcs_eff=[0.3, 0.6, 1.2, 2.0, 3.0, 4.0], ctrl_eff=0.6, mac_eff=0.6, base_latency_ms=25.0,
                      jitter_ms=10.0, mac_delay_ms=5.0, retx_delay_ms=15.0, max_attempts=3, activation_delay_s=45.0, rlf_timeout_s=4.0,
                      active_w=25.0, warm_w=10.0, pa_eff=0.15), mass_kg=1.1,
                 options=dict(orbit_km=550.0, elevation_deg=45.0, azimuth_deg=180.0),
                 faults=[dict(kind="packet_burst_loss", rate_per_min=4.0, duration_mean_s=0.15, duration_min_s=0.05, duration_max_s=0.4,
                              params_range={"p": [0.5, 0.95]})],
                 notes="Needs sky view (buildings obstruct), ~45 s cold acquisition => pre-warming matters, heavy and power hungry."))
    add(TechSpec("sat_kuka_leo", "LEO satellite, other Ku/Ka constellation", "satellite", "sat_leo", "sat", True, True,
                 dict(tech="sat_kuka_leo", freq_hz=12.0e9, bw_hz=40e6, dl_tx_dbm=45.0, ul_tx_dbm=33.0, gs_gain_dbi=38.0, drone_gain_dbi=28.0,
                      ant_pattern="omni", nf_db=3.5, pl_exp=2.0, rician_k_db=10.0, shadow_sigma_nlos_db=10.0, shadow_decorr_m=100.0,
                      mcs_eff=[0.3, 0.6, 1.2, 2.0, 3.0, 4.0], ctrl_eff=0.6, mac_eff=0.55, base_latency_ms=32.0, jitter_ms=12.0,
                      mac_delay_ms=6.0, retx_delay_ms=15.0, max_attempts=3, activation_delay_s=40.0, rlf_timeout_s=4.0, active_w=35.0,
                      warm_w=12.0, pa_eff=0.15), mass_kg=1.5, options=dict(orbit_km=1200.0, elevation_deg=40.0, azimuth_deg=170.0),
                 faults=[dict(kind="packet_burst_loss", rate_per_min=2.0, duration_mean_s=0.2, duration_min_s=0.05, duration_max_s=0.5,
                              params_range={"p": [0.5, 0.9]})]))
    add(TechSpec("sat_iridium", "LEO satellite, Iridium-style L-band (narrowband)", "satellite", "sat_iridium", "sat", True, False,
                 dict(tech="sat_iridium", freq_hz=1.62e9, bw_hz=50e3, dl_tx_dbm=21.0, ul_tx_dbm=32.0, gs_gain_dbi=24.0, drone_gain_dbi=3.0,
                      ant_pattern="omni", nf_db=4.0, pl_exp=2.0, rician_k_db=8.0, mcs_eff=[0.5, 1.0, 1.8], ctrl_eff=1.0, mac_eff=0.8,
                      base_latency_ms=110.0, jitter_ms=25.0, mac_delay_ms=10.0, retx_delay_ms=60.0, max_attempts=2,
                      activation_delay_s=20.0, rlf_timeout_s=3.0, active_w=3.5, warm_w=0.8, pa_eff=0.3), mass_kg=0.15,
                 options=dict(orbit_km=780.0, elevation_deg=40.0, azimuth_deg=200.0),
                 faults=[dict(kind="link_blackout", rate_per_min=1.0, duration_mean_s=0.8, duration_min_s=0.2, duration_max_s=3.0)],
                 notes="Global coverage, tiny capacity, ~100 ms: a last-resort control path."))
    add(TechSpec("sat_sband_leo", "LEO satellite, S-band (Globalstar/Orbcomm-style)", "satellite", "sat_sband", "sat", True, False,
                 dict(tech="sat_sband", freq_hz=2.2e9, bw_hz=1.25e6, dl_tx_dbm=35.0, ul_tx_dbm=33.0, gs_gain_dbi=30.0, drone_gain_dbi=3.0,
                      ant_pattern="omni", nf_db=4.0, pl_exp=2.0, rician_k_db=8.0, mcs_eff=[0.2, 0.5, 1.0, 2.0], ctrl_eff=0.5, mac_eff=0.7,
                      base_latency_ms=50.0, jitter_ms=15.0, mac_delay_ms=8.0, retx_delay_ms=30.0, max_attempts=3,
                      activation_delay_s=15.0, rlf_timeout_s=3.0, active_w=4.0, warm_w=1.0, pa_eff=0.3), mass_kg=0.2,
                 options=dict(orbit_km=1414.0, elevation_deg=30.0, azimuth_deg=190.0),
                 faults=[dict(kind="packet_burst_loss", rate_per_min=1.5, duration_mean_s=0.3, duration_min_s=0.1, duration_max_s=1.0,
                              params_range={"p": [0.4, 0.9]})]))
    add(TechSpec("sat_lband_geo", "GEO satellite, L-band (BGAN-style)", "satellite", "sat_geo", "sat", True, False,
                 dict(tech="sat_geo_l", freq_hz=1.6e9, bw_hz=200e3, dl_tx_dbm=35.0, ul_tx_dbm=33.0, gs_gain_dbi=33.0, drone_gain_dbi=10.0,
                      ant_pattern="omni", nf_db=3.0, pl_exp=2.0, rician_k_db=8.0, mcs_eff=[0.3, 0.6, 1.2, 2.0], ctrl_eff=0.6, mac_eff=0.7,
                      base_latency_ms=290.0, jitter_ms=30.0, mac_delay_ms=15.0, retx_delay_ms=550.0, max_attempts=2,
                      activation_delay_s=60.0, rlf_timeout_s=5.0, active_w=10.0, warm_w=3.0, pa_eff=0.3), mass_kg=1.0,
                 options=dict(orbit_km=35786.0, elevation_deg=70.0, azimuth_deg=180.0),
                 notes="~290 ms one way: cannot meet a 150 ms control deadline (kept for completeness / telemetry)."))
    add(TechSpec("sat_kuka_geo", "GEO satellite, Ku/Ka VSAT-style", "satellite", "sat_geo", "sat", True, True,
                 dict(tech="sat_geo_ku", freq_hz=12.0e9, bw_hz=10e6, dl_tx_dbm=38.0, ul_tx_dbm=40.0, gs_gain_dbi=42.0, drone_gain_dbi=30.0,
                      ant_pattern="omni", nf_db=2.0, pl_exp=2.0, rician_k_db=8.0, mcs_eff=[0.3, 0.6, 1.2, 2.0, 3.0], ctrl_eff=0.6,
                      mac_eff=0.6, base_latency_ms=280.0, jitter_ms=30.0, mac_delay_ms=15.0, retx_delay_ms=550.0, max_attempts=2,
                      activation_delay_s=90.0, rlf_timeout_s=6.0, active_w=60.0, warm_w=20.0, pa_eff=0.12), mass_kg=2.5,
                 options=dict(orbit_km=35786.0, elevation_deg=70.0, azimuth_deg=180.0),
                 notes="High throughput but ~280 ms and a gimballed dish: unrealistic for a small drone; included for completeness."))
    add(TechSpec("mesh_rf", "Mesh / multi-hop RF (relay nodes along the corridor)", "mesh", "mesh", "relay", True, True,
                 dict(tech="mesh", freq_hz=2.3e9, bw_hz=10e6, dl_tx_dbm=30.0, ul_tx_dbm=30.0, gs_gain_dbi=5.0, drone_gain_dbi=2.0, nf_db=5.0,
                      pl_exp=2.3, rician_k_db=7.0, mcs_eff=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], ctrl_eff=0.5, mac_eff=0.35,
                      base_latency_ms=14.0, jitter_ms=6.0, mac_delay_ms=6.0, retx_delay_ms=5.0, activation_delay_s=2.0, rlf_timeout_s=1.5,
                      ho_hyst_db=3.0, ho_ttt_s=0.2, ho_interrupt_s=0.15, ho_fail_p=0.05, active_w=6.0, warm_w=2.0, pa_eff=0.3),
                 mass_kg=0.2, options=dict(relay_spacing_m=800.0, max_relays=12, relay_height_m=12.0),
                 faults=[dict(kind="link_blackout", rate_per_min=1.0, duration_mean_s=0.7, duration_min_s=0.2, duration_max_s=2.5)],
                 notes="Relays on the ground along the corridor; route changes appear as handovers; capacity halves per hop (mac_eff 0.35)."))
    add(TechSpec("fiber_tether", "Fiber-optic tether (near-zero latency, jam-proof, range-limited)", "fiber", "fiber", "tether", True, True,
                 dict(tech="fiber", freq_hz=0.1e9, bw_hz=200e6, dl_tx_dbm=30.0, ul_tx_dbm=30.0, gs_gain_dbi=14.4, drone_gain_dbi=0.0,
                      ant_pattern="omni", nf_db=5.0, pl_exp=2.0, rician_k_db=40.0, shadow_sigma_los_db=0.05, shadow_sigma_nlos_db=0.05,
                      mcs_eff=[0.5, 2.0, 5.0], ctrl_eff=0.5, mac_eff=0.95, ra_margin_db=1.0, base_latency_ms=0.8, jitter_ms=0.1,
                      mac_delay_ms=0.05, retx_delay_ms=0.3, max_attempts=4, activation_delay_s=0.05, rlf_timeout_s=0.2, active_w=2.0,
                      warm_w=1.0, pa_eff=0.5), mass_kg=0.05,
                 options=dict(length_m=1500.0, kg_per_km=0.15, anchor_height_m=2000.0),
                 faults=[dict(kind="link_blackout", rate_per_min=0.03, duration_mean_s=15.0, duration_min_s=3.0, duration_max_s=60.0)],
                 notes="Virtual anchor above home keeps the sim's obstruction model from blocking the cable; range limit is enforced by a "
                       "pre-screen (mission must fit inside length_m) and a geofence at length_m; tether mass adds hover power."))
    dflt = dataclasses.asdict(sim.LinkCfg())
    for k in ("name", "station_ids", "carries_video"):
        dflt.pop(k, None)
    for sp in cat.values():
        sp.link = {**copy.deepcopy(dflt), **{k: v for k, v in sp.link.items() if k not in ("name", "station_ids", "carries_video")}}
    legacy = cat
    native: "OrderedDict[str, TechSpec]" = OrderedDict()
    for old_id, radio_name in integration.LEGACY_TECH_TO_RADIO.items():
        if old_id not in legacy:
            continue
        original = legacy[old_id]
        native[radio_name] = dataclasses.replace(original, id=radio_name)
    for radio in integration.drone.radio_catalog():
        if radio.name in native:
            continue
        profile = integration.generic_fast_profile(radio, sim)
        native[radio.name] = TechSpec(
            radio.name, f"{radio.name} ({radio.family or radio.kind})", profile["group"], profile["family"],
            profile["stations"], profile["control"], profile["video"], profile["link"],
            mass_kg=profile["mass_kg"], options=profile["options"], notes=profile["notes"])
    return native


def link_names_for(spec: TechSpec, opts: Dict[str, Any]) -> List[str]:
    n = int(opts.get("sims", 1)) if spec.id in ("cell_5g_multisim",) else 1
    return [spec.id] if n <= 1 else [f"{spec.id}_sim{k + 1}" for k in range(n)]


def exp_latency_ms(link: Dict[str, Any]) -> float:
    return link["base_latency_ms"] + 1.5 * link["jitter_ms"] + 1.2 * link["mac_delay_ms"] + link["retx_delay_ms"]


def peak_radio_w(link: Dict[str, Any]) -> float:
    """Static worst-case radio draw: active electronics + full-duty TX at the PA efficiency."""
    return link["active_w"] + 10.0 ** (link["ul_tx_dbm"] / 10.0) / 1000.0 / link["pa_eff"]


def design_snr_db(link: Dict[str, Any], d_m: float, direction: str = "dl") -> float:
    """Free-space (no obstruction/shadow/rain) design SNR at distance d_m, as the simulator defines it."""
    g_dr = link["drone_gain_dbi"] - (0.0 if link.get("ant_pattern", "dipole") == "omni" else 2.0)
    tx = link["dl_tx_dbm"] if direction == "dl" else link["ul_tx_dbm"]
    pl = sim.fspl_1m_db(link["freq_hz"]) + 10.0 * link["pl_exp"] * math.log10(max(d_m, 1.0))
    noise = sim.noise_dbm(link["bw_hz"], link["nf_db"]) + link.get("ambient_noise_rise_db", 0.0)
    return tx + link["gs_gain_dbi"] + g_dr - pl - noise


# =============================================================================
# 3. STUDY CONFIGURATION (everything controllable from JSON; CLI flags override)
# =============================================================================
DEFAULT_EXCLUSIVE = [["analog_fpv_58", "digital_58_dji", "digital_58_openhd",
                      "openhd58", "walksnail", "hdzero", "openipc58"],
                     ["ism_24_ctrl", "ism_24_wifi"],
                     ["cell_5g", "cell_5g_multisim"],
                     ["sat_leo_starlink", "sat_kuka_leo"],
                     ["sat_lband_geo", "sat_kuka_geo"]]


@dataclass
class TechSel:
    enabled: List[str] = _L("all")                         # tech ids and/or group names ("satellite", ...) or "all"
    groups_enabled: Dict[str, bool] = _D()                 # e.g. {"fiber": false, "mesh": true, "satellite": false, "ai_video": true}
    max_links: int = 3                                     # max simultaneous radio links (multi-SIM modems count individually)
    min_links: int = 1
    force_include_all: List[str] = _L()                    # every combo must contain all of these
    force_include_any: List[List[str]] = field(default_factory=list)   # each inner list: at least one must be present
    force_exclude: List[str] = _L()
    exclusive_groups: List[List[str]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_EXCLUSIVE))
    conflicts: List[List[str]] = field(default_factory=list)           # forbidden pairs
    requires: Dict[str, List[str]] = _D()                  # tech -> at least one of these must also be present
    overrides: Dict[str, Dict[str, Any]] = _D()            # tech -> LinkCfg field overrides
    options: Dict[str, Dict[str, Any]] = _D()              # tech -> options (sims, length_m, relay_spacing_m, orbit_km, ...)
    aliases: Dict[str, str] = field(default_factory=lambda: dict(integration.RADIO_ALIASES))
    radio_overrides: Dict[str, Dict[str, Any]] = _D()


@dataclass
class FilterCfg:
    max_radio_power_w: Optional[float] = None              # pre-run: static peak radio draw of the combo
    max_payload_kg: Optional[float] = 1.5                  # pre-run: radio + tether mass
    max_mean_power_w: Optional[float] = None               # post-run feasibility: measured mean communication power
    min_control_reliability: float = 0.0                   # post-run feasibility: mean on-time control rate
    min_completion_rate: float = 0.0                       # post-run feasibility: fraction of missions completed
    require_control_link: bool = True
    require_video_link: bool = True
    drop_nonrealtime_links: bool = True                    # skip combos holding a link that can meet neither deadline
    max_control_latency_ms: Optional[float] = None         # None -> scenario control deadline
    max_video_latency_ms: Optional[float] = None
    min_control_margin_db: float = 6.0                     # free-space design margin vs control threshold
    min_control_coverage: float = 1.0                      # fraction of waypoints with a control link above margin (0 disables)
    min_video_margin_db: float = 3.0
    min_video_coverage: float = 0.0
    min_video_bps: float = 0.3e6
    tether_margin: float = 1.1                             # mission radius * this must fit inside the tether


@dataclass
class PolicySel:
    policies: List[Any] = _L("reactive", "bonding", "pred_threshold", "pred_cost")   # presets, or dicts {label,name|class,params,sweep}
    mode: str = "cross"                                    # cross | random_subset | single
    per_combo: int = 2                                     # random_subset: policies per combo (reference always included)
    reference: str = "reactive"
    wrap_roles: bool = True                                # enforce control/video capability of each technology
    reactive_priority: str = "robust_latency"              # latency | robust_latency | listed


@dataclass
class AiVideoCfg:
    apply: str = "constrained_only"                        # all | constrained_only (only combos without a wideband video link)
    constrained_mbps: float = 3.0
    q_r0_bps: float = 0.06e6                               # quality knee: neural codec reaches good quality at low rate
    max_bps: float = 2.0e6
    fec: float = 0.25
    encode_latency_ms: float = 40.0
    compute_w: float = 6.0
    mass_kg: float = 0.05


@dataclass
class ScaleCfg:
    seeds: int = 5
    master_seed: int = 2026
    duration_s: Optional[float] = 120.0
    dt: Optional[float] = 0.05
    ctrl_hz: Optional[float] = 20.0
    scenarios: List[str] = _L("jammed_inspection")
    sim_overrides: Dict[str, Any] = _D()
    strategy: str = "pairwise"                             # full | stratified | random | pairwise | explicit | halving
    strength: int = 2                                      # pairwise: t-way coverage strength
    random_n: int = 100
    per_size: Dict[str, Any] = _D(**{"1": "all", "2": "all", "3": 50})
    explicit: List[List[str]] = field(default_factory=list)
    max_combinations: int = 300
    max_runs: int = 50000
    exceed_action: str = "error"                           # error | sample
    enumeration_limit: int = 3_000_000
    sampling_seed: int = 1
    halving: Dict[str, Any] = field(default_factory=lambda: {
        "base_strategy": "pairwise", "rank_by": "blind_time_s", "score_policy": "best", "min_survivors": 3,
        "stages": [{"seeds": 2, "keep": 0.3, "policies": "reference"}, {"seeds": 5, "keep": 0.5, "policies": "all"},
                   {"seeds": 12, "keep": 1.0, "policies": "all"}]})
    video_modes: List[str] = _L("standard")                # "standard", "ai"
    ai_video: AiVideoCfg = field(default_factory=AiVideoCfg)
    payload_hover_w_per_kg: float = 88.0                   # extra hover power per kg of radio payload


DEFAULT_RECORD = [
    "blind_time_s", "blind_fraction", "blind_any", "n_blind_episodes", "max_blind_episode_s", "ctrl_unreliable_s", "ctrl_only_loss_s",
    "video_unusable_s", "ctrl_ontime_rate", "ctrl_lost_rate", "ctrl_late_rate", "ctrl_within_100ms_rate", "ctrl_within_150ms_rate",
    "ctrl_lat_p50_ms", "ctrl_lat_p95_ms", "ctrl_lat_p99_ms", "ctrl_lat_p999_ms", "ctrl_lat_max_ms", "ctrl_max_gap_ms",
    "video_quality_mean", "video_usable_fraction", "video_freeze_s", "video_freeze_events", "video_mean_mbps",
    "energy_comm_wh", "comm_power_mean_w", "comm_power_p99_w", "over_power_budget_s", "spectrum_mean_mhz", "energy_per_video_mbit_j",
    "n_hard_switch", "n_soft_switch", "n_pingpong", "handover_cost", "switch_per_min", "proactive_switch_frac", "dup_time_s",
    "n_rlf", "n_failsafe", "utility_cost", "mission_completed", "mission_completion_s", "wp_progress", "soc_end", "flight_s", "sim_time_s",
    "pred_auc", "pred_brier", "pred_ep_recall", "pred_lead_mean_s", "pred_false_alarms_per_min", "outage_to_blind_rate"]


@dataclass
class MetricsSel:
    record: List[str] = field(default_factory=lambda: list(DEFAULT_RECORD))
    source: str = "eval"                                   # eval (recompute from tick logs) | sim (simulator's own) | both
    decimals: int = 6
    number_format: str = "fixed"                           # fixed | sci | sig | repr (round-trip exact)
    md_decimals: int = 4
    save_ticks: str = "none"                               # none | first:N | all   (kept under <out>/logs)
    include_runs_in_json: bool = False
    eval_config: Dict[str, Any] = _D()                     # uavlink_eval.EvalConfig overrides (definitions, thresholds)
    aggregate: List[str] = _L("blind_time_s", "blind_fraction", "blind_any", "max_blind_episode_s", "ctrl_within_150ms_rate",
                              "ctrl_lat_p99_ms", "video_quality_mean", "video_freeze_s", "comm_power_mean_w", "energy_comm_wh",
                              "n_hard_switch", "n_soft_switch", "handover_cost", "quality_per_watt", "utility_cost", "mission_completed")


@dataclass
class ExecCfg:
    backend: str = "fast"                                 # fast | high_fidelity
    physics_dt_s: float = 0.01
    radio_dt_s: Optional[float] = None
    workers: int = 0                                       # 0 = cpu_count-1
    resume: bool = False
    retry_errors: bool = True
    progress_interval_s: float = 10.0
    inflight_factor: int = 3
    fsync_every: int = 50
    calibrate: bool = True                                 # time a short sim to estimate the run time in --dry-run
    calibrate_sim_s: float = 12.0
    dry_run: bool = False
    force_resume: bool = False


@dataclass
class OutCfg:
    dir: str = "experiments/study"
    write_results_json: bool = True
    plan_files: bool = True


@dataclass
class RankCfg:
    primary: str = "blind_time_s"
    efficiency: str = "quality_per_watt"                   # quality_per_watt | energy_per_video_mbit_j | any recorded metric
    efficiency_higher_is_better: bool = True
    top_k: int = 20
    alpha: float = 0.05
    scope: str = "pooled"                                  # pooled | per_scenario
    min_runs: int = 0                                      # 0 = use the most common run count
    pareto: List[str] = _L("blind_time_s", "comm_power_mean_w", "video_quality_mean", "ctrl_lat_p99_ms")


@dataclass
class InterfCfg:
    keep_scenario: bool = True                             # keep the interferers defined in the scenario
    random: Dict[str, Any] = _D()                          # {"count":[0,2],"eirp_dbm":[22,36],"t_start":[10,200],"dur":[5,40],"duty":[0.3,1.0],"near_path":true}
    jam_groups: List[str] = _L("fpv_58", "digital_58", "ism_24", "lr_12", "sub_ghz", "other_bands", "cellular", "mesh")


@dataclass
class StudyCfg:
    name: str = "study"
    techs: TechSel = field(default_factory=TechSel)
    filters: FilterCfg = field(default_factory=FilterCfg)
    policies: PolicySel = field(default_factory=PolicySel)
    scale: ScaleCfg = field(default_factory=ScaleCfg)
    metrics: MetricsSel = field(default_factory=MetricsSel)
    execution: ExecCfg = field(default_factory=ExecCfg)
    output: OutCfg = field(default_factory=OutCfg)
    ranking: RankCfg = field(default_factory=RankCfg)
    interference: InterfCfg = field(default_factory=InterfCfg)
    randomize: Dict[str, Any] = _D(wind_scale=[0.7, 1.5], waypoint_jitter_m=60.0, alt_jitter_m=5.0, speed_scale=[0.9, 1.1])
    sim_module: Optional[str] = None


PRESET_POLICIES: Dict[str, dict] = {
    "reactive": {"label": "reactive", "name": "reactive"},
    "bonding": {"label": "bonding", "name": "bonding"},
    "static": {"label": "static", "name": "static_primary"},
    "pred_threshold": {"label": "pred_threshold", "name": "cost_predictive",
                       "params": {"p_warm": 0.15, "p_dup": 0.35, "p_hard": 0.6, "min_gain": 0.0, "switch_pen": 0.0}},
    "pred_cost": {"label": "pred_cost", "name": "cost_predictive",
                  "params": {"p_warm": 0.10, "p_dup": 0.35, "p_hard": 0.05, "min_gain": 0.1, "switch_pen": 0.3}},
    "pred_aggressive": {"label": "pred_aggressive", "name": "cost_predictive",
                        "params": {"p_warm": 0.08, "p_dup": 0.15, "p_hard": 0.3, "dwell_s": 2.0}},
    "pred_conservative": {"label": "pred_conservative", "name": "cost_predictive",
                          "params": {"p_warm": 0.3, "p_dup": 0.6, "p_hard": 0.85, "dwell_s": 4.0}},
    "pred_latency_guard": {"label": "pred_latency_guard", "name": "cost_predictive",
                           "params": {"lat_limit_ms": 100.0, "w_lat": 8.0, "p_hard": 0.5}},
}


def load_study(path: Optional[str] = None, overrides: Optional[dict] = None) -> StudyCfg:
    d: dict = {}
    if path:
        with open(path) as f:
            d = json.load(f)
    d = strip_comments(d)
    if overrides:
        d = sim.deep_merge(d, overrides)
    st = sim.from_dict(StudyCfg, d)
    return st


class PlanError(Exception):
    pass


# =============================================================================
# 4. TECHNOLOGY UNIVERSE: overrides, switches, static (scenario-independent) rules
# =============================================================================
class Universe:
    def __init__(self, st: StudyCfg):
        self.st = st
        aliases = dict(integration.RADIO_ALIASES)
        aliases.update(st.techs.aliases)
        st.techs.aliases = aliases

        def canon(name: str) -> str:
            if name == "all":
                return name
            try:
                return integration.canonical_radio_name(name, aliases)
            except ValueError:
                if name in GROUPS:
                    return name
                raise

        cat = build_catalog()
        lc_fields = {f.name for f in dataclasses.fields(sim.LinkCfg)}
        overrides = {}
        for tid, ov in st.techs.overrides.items():
            tid = canon(tid)
            if tid not in cat:
                raise PlanError(f"techs.overrides: unknown technology {tid!r}; known: {list(cat)}")
            bad = set(ov) - lc_fields
            if bad:
                raise PlanError(f"techs.overrides[{tid}]: unknown LinkCfg field(s) {sorted(bad)}")
            overrides.setdefault(tid, {}).update(copy.deepcopy(ov))
        for tid, ov in overrides.items():
            cat[tid].link.update(ov)
        options = {}
        for tid, op in st.techs.options.items():
            tid = canon(tid)
            if tid not in cat:
                raise PlanError(f"techs.options: unknown technology {tid!r}")
            options.setdefault(tid, {}).update(copy.deepcopy(op))
        for tid, op in options.items():
            cat[tid].options.update(op)
        st.techs.enabled = [canon(s) for s in st.techs.enabled]
        st.techs.force_include_all = [canon(s) for s in st.techs.force_include_all]
        st.techs.force_include_any = [[canon(s) for s in group] for group in st.techs.force_include_any]
        st.techs.force_exclude = [canon(s) for s in st.techs.force_exclude]
        st.techs.exclusive_groups = [[canon(s) for s in group] for group in st.techs.exclusive_groups]
        st.techs.conflicts = [[canon(s) for s in group] for group in st.techs.conflicts]
        st.techs.requires = {canon(k): [canon(x) for x in v] for k, v in st.techs.requires.items()}
        st.techs.radio_overrides = {canon(k): v for k, v in st.techs.radio_overrides.items()}
        sel = st.techs.enabled
        names = set(cat) if ("all" in sel) else set()
        for s in sel:
            if s == "all":
                continue
            if s in cat:
                names.add(s)
            elif s in GROUPS:
                names |= {t for t, sp in cat.items() if sp.group == s}
            else:
                raise PlanError(f"techs.enabled: {s!r} is neither a technology nor a group; groups: {GROUPS}")
        for g, on in st.techs.groups_enabled.items():
            if g == "ai_video":
                continue
            if g not in GROUPS:
                raise PlanError(f"groups_enabled: unknown group {g!r}; groups: {GROUPS}")
            if not on:
                names -= {t for t, sp in cat.items() if sp.group == g}
        for t in st.techs.force_exclude:
            if t not in cat:
                raise PlanError(f"force_exclude: unknown technology {t!r}")
            names.discard(t)
        self.catalog = cat
        self.specs: "OrderedDict[str, TechSpec]" = OrderedDict((t, cat[t]) for t in cat if t in names)
        self.ai_enabled = st.techs.groups_enabled.get("ai_video", True)
        for t in st.techs.force_include_all:
            if t not in self.specs:
                raise PlanError(f"force_include_all: {t!r} is not enabled (excluded by group/force_exclude?)")
        self.pair_conflicts = set()
        for grp in st.techs.exclusive_groups:
            for a, b in itertools.combinations([g for g in grp if g in self.specs], 2):
                self.pair_conflicts.add(frozenset((a, b)))
        for a, b in st.techs.conflicts:
            self.pair_conflicts.add(frozenset((a, b)))
        self.ids = list(self.specs)

    def n_links(self, ids: Sequence[str]) -> int:
        return sum(len(link_names_for(self.specs[t], self.specs[t].options)) for t in ids)

    def link_dicts(self, tid: str) -> List[Tuple[str, dict]]:
        sp = self.specs[tid]
        out = []
        for nm in link_names_for(sp, sp.options):
            d = copy.deepcopy(sp.link)
            d["carries_video"] = sp.video
            out.append((nm, d))
        return out

    def mass_kg(self, ids: Sequence[str]) -> float:
        m = 0.0
        for t in ids:
            sp = self.specs[t]
            m += sp.mass_kg * len(link_names_for(sp, sp.options))
            if t == "fiber_tether":
                m += sp.options.get("kg_per_km", 0.15) * sp.options.get("length_m", 1500.0) / 1000.0
        return m

    def peak_w(self, ids: Sequence[str]) -> float:
        return sum(peak_radio_w(d) for t in ids for _, d in self.link_dicts(t))

    def warm_w(self, ids: Sequence[str]) -> float:
        return sum(d["warm_w"] for t in ids for _, d in self.link_dicts(t))

    def static_reason(self, ids: Sequence[str]) -> Optional[str]:
        st, F = self.st, self.st.filters
        nl = self.n_links(ids)
        if nl > st.techs.max_links:
            return "too_many_links"
        if nl < st.techs.min_links:
            return "too_few_links"
        for t in st.techs.force_include_all:
            if t not in ids:
                return "missing_forced_include"
        for grp in st.techs.force_include_any:
            if not any(t in ids for t in grp):
                return "missing_forced_include_any"
        for a, b in itertools.combinations(ids, 2):
            if frozenset((a, b)) in self.pair_conflicts:
                return "exclusive_or_conflict"
        for t, alts in st.techs.requires.items():
            if t in ids and not any(a in ids for a in alts):
                return "requirement_unmet"
        cl = F.max_control_latency_ms or 150.0
        vl = F.max_video_latency_ms or 250.0
        lat = {t: exp_latency_ms(self.specs[t].link) for t in ids}
        if F.require_control_link and not any(self.specs[t].control and lat[t] <= cl for t in ids):
            return "no_realtime_control_link"
        if F.require_video_link and not any(self.specs[t].video and lat[t] <= vl for t in ids):
            return "no_realtime_video_link"
        if F.drop_nonrealtime_links and any(not ((self.specs[t].control and lat[t] <= cl) or (self.specs[t].video and lat[t] <= vl)) for t in ids):
            return "contains_nonrealtime_link"
        if F.max_radio_power_w is not None and self.peak_w(ids) > F.max_radio_power_w:
            return "radio_power_limit"
        if F.max_payload_kg is not None and self.mass_kg(ids) > F.max_payload_kg:
            return "payload_mass_limit"
        return None


# =============================================================================
# 5. SCENARIO GEOMETRY: stations, relays, satellites, tether; design-margin pre-screens
# =============================================================================
@dataclass
class Geom:
    name: str
    base: dict
    home: np.ndarray
    wps: np.ndarray
    stations: List[dict]
    ids_by_kind: Dict[str, List[str]]
    mission_radius_m: float
    ctrl_deadline_ms: float
    video_deadline_ms: float
    ctrl_hz: float
    tech_cover: Dict[str, dict] = field(default_factory=dict)


def scenario_base(st: StudyCfg, scn_name: str) -> dict:
    d = sim.load_scenario_dict(scn_name)
    d = sim.deep_merge(d, st.scale.sim_overrides or {})
    if st.scale.duration_s:
        d["duration_s"] = float(st.scale.duration_s)
    if st.scale.dt:
        d["dt"] = float(st.scale.dt)
    if st.scale.ctrl_hz:
        d.setdefault("traffic", {})["ctrl_hz"] = float(st.scale.ctrl_hz)
    return d


def build_geom(st: StudyCfg, uni: Universe, scn_name: str) -> Geom:
    base = scenario_base(st, scn_name)
    env = base["env"]
    hx, hy = env["home"][0], env["home"][1]
    wps = np.array([[w["x"], w["y"], w["z"]] for w in base["mission"]["waypoints"]], float)
    stations = [dict(id=s["id"], pos=list(s["pos"]), kind=s.get("kind", "gcs")) for s in env["stations"]]
    ids = defaultdict(list)
    for s in stations:
        ids[s["kind"]].append(s["id"])
    sx, sy = env.get("size_m", [2000, 2000])
    if not ids["gcs"] and stations:
        ids["gcs"].append(stations[0]["id"])
    if any(sp.stations == "cell" for sp in uni.specs.values()) and not ids["cell"]:
        for i, (fx, fy) in enumerate([(0.2, 0.8), (0.8, 0.2), (0.8, 0.8)]):
            s = dict(id=f"cell_auto{i + 1}", pos=[fx * sx, fy * sy, 35.0], kind="cell")
            stations.append(s)
            ids["cell"].append(s["id"])
    mesh_specs = [sp for sp in uni.specs.values() if sp.group == "mesh"]
    if mesh_specs:
        o = mesh_specs[0].options
        poly = np.vstack([[hx, hy], wps[:, :2]])
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        sp_ = float(o.get("relay_spacing_m", 800.0))
        n_rel = min(int(o.get("max_relays", 12)), int(cum[-1] // sp_))
        for k in range(1, n_rel + 1):
            dist = k * sp_
            i = min(int(np.searchsorted(cum, dist, side="right")) - 1, len(seg) - 1)
            f = (dist - cum[i]) / max(seg[i], 1e-9)
            p = poly[i] + f * (poly[i + 1] - poly[i])
            s = dict(id=f"relay{k}", pos=[float(p[0]), float(p[1]), float(o.get("relay_height_m", 12.0))], kind="relay")
            stations.append(s)
            ids["relay"].append(s["id"])
    for tid, sp in uni.specs.items():
        if sp.stations == "sat":
            o = sp.options
            h = o["orbit_km"] * 1000.0
            R = h / math.tan(math.radians(o["elevation_deg"]))
            az = math.radians(o["azimuth_deg"])
            s = dict(id=f"sat_{tid}", pos=[float(hx + R * math.sin(az)), float(hy + R * math.cos(az)), float(h)], kind="sat")
            stations.append(s)
            ids[f"sat:{tid}"].append(s["id"])
        elif sp.stations == "tether":
            s = dict(id="tether_anchor", pos=[float(hx), float(hy), float(sp.options.get("anchor_height_m", 2000.0))], kind="tether")
            stations.append(s)
            ids["tether"].append(s["id"])
    tr = base.get("traffic", {})
    radius = float(np.max(np.hypot(wps[:, 0] - hx, wps[:, 1] - hy))) if len(wps) else 0.0
    jit = float(st.randomize.get("waypoint_jitter_m", 0.0)) * 3.0
    return Geom(scn_name, base, np.array([hx, hy, 0.0]), wps, stations, dict(ids), radius + jit,
                float(tr.get("ctrl_deadline_ms", 150.0)), float(tr.get("video_deadline_ms", 250.0)), float(tr.get("ctrl_hz", 50.0)))


def station_ids_for(spec: TechSpec, geom: Geom, k: int = 0, n: int = 1) -> List[str]:
    kind = spec.stations
    if kind == "gcs":
        return list(geom.ids_by_kind.get("gcs", []))
    if kind == "cell":
        cells = list(geom.ids_by_kind.get("cell", []))
        if n > 1 and len(cells) >= n:
            return cells[k::n]
        return cells
    if kind == "relay":
        return list(geom.ids_by_kind.get("relay", [])) + list(geom.ids_by_kind.get("gcs", []))
    if kind == "sat":
        return list(geom.ids_by_kind.get(f"sat:{spec.id}", []))
    return list(geom.ids_by_kind.get("tether", []))


def tech_design(uni: Universe, geom: Geom, tid: str, st: StudyCfg) -> dict:
    """Per-waypoint free-space design margins of one technology (cached in geom.tech_cover)."""
    if tid in geom.tech_cover:
        return geom.tech_cover[tid]
    F, sp = st.filters, uni.specs[tid]
    name, link = uni.link_dicts(tid)[0]
    stn = {s["id"]: s for s in geom.stations}
    sids = station_ids_for(sp, geom)
    n = len(geom.wps)
    ctrl_margin, vid_ok = np.full(n, -99.0), np.zeros(n, bool)
    thr_c = sim.mcs_threshold_db(link["ctrl_eff"])
    offered = geom.ctrl_hz * (st_traffic_bytes(geom) * 8)
    ctrl_cap_ok = link["bw_hz"] * link["ctrl_eff"] * link["mac_eff"] >= offered
    for i, w in enumerate(geom.wps):
        if not sids:
            continue
        d = min(math.dist((w[0], w[1], w[2]), tuple(stn[s]["pos"])) for s in sids)
        if tid == "fiber_tether":
            reach = sp.options.get("length_m", 1500.0) >= F.tether_margin * math.hypot(w[0] - geom.home[0], w[1] - geom.home[1])
            ctrl_margin[i], vid_ok[i] = (60.0, True) if reach else (-99.0, False)
            continue
        ctrl_margin[i] = design_snr_db(link, d, "dl") - thr_c
        snr_ul = design_snr_db(link, d, "ul")
        vid_ok[i] = sp.video and any(link["bw_hz"] * e * link["mac_eff"] >= F.min_video_bps and sim.mcs_threshold_db(e) + F.min_video_margin_db <= snr_ul
                                    for e in link["mcs_eff"])
    lat = exp_latency_ms(link)
    c_ok = sp.control and lat <= (F.max_control_latency_ms or geom.ctrl_deadline_ms) and ctrl_cap_ok
    v_ok = sp.video and lat <= (F.max_video_latency_ms or geom.video_deadline_ms)
    out = dict(ctrl_cover=(ctrl_margin >= F.min_control_margin_db) & c_ok, vid_cover=vid_ok & v_ok, margin=ctrl_margin, exp_lat=lat,
               covers_all=bool(np.all((ctrl_margin >= F.min_control_margin_db) & c_ok)), ctrl_eligible=bool(c_ok), vid_eligible=bool(v_ok))
    geom.tech_cover[tid] = out
    return out


def st_traffic_bytes(geom: Geom) -> int:
    tr = geom.base.get("traffic", {})
    return int(tr.get("ctrl_bytes", 64)) + int(tr.get("hdr_bytes", 40))


def scenario_reason(uni: Universe, geom: Geom, ids: Sequence[str], st: StudyCfg) -> Optional[str]:
    F = st.filters
    cov_c = np.zeros(len(geom.wps), bool)
    cov_v = np.zeros(len(geom.wps), bool)
    for t in ids:
        d = tech_design(uni, geom, t, st)
        cov_c |= d["ctrl_cover"]
        cov_v |= d["vid_cover"]
    if F.require_control_link and F.min_control_coverage > 0 and cov_c.mean() + 1e-9 < F.min_control_coverage:
        if "fiber_tether" in ids and not any(tech_design(uni, geom, t, st)["covers_all"] for t in ids):
            return "tether_too_short_or_no_control_coverage"
        return "control_link_budget"
    if F.require_video_link and F.min_video_coverage > 0 and cov_v.mean() + 1e-9 < F.min_video_coverage:
        return "video_link_budget"
    return None


# =============================================================================
# 6. COMBINATION GENERATION
# =============================================================================
@dataclass
class Combo:
    ids: Tuple[str, ...]
    id: str
    name: str
    n_links: int
    peak_w: float
    warm_w: float
    mass_kg: float
    scn_ok: Dict[str, Optional[str]] = field(default_factory=dict)      # scenario -> reason or None
    constrained: bool = False                                            # no wideband video link (AI-video target)


def combo_id(ids: Sequence[str]) -> str:
    return hashlib.sha1("+".join(ids).encode()).hexdigest()[:8]


class Generator:
    def __init__(self, uni: Universe, st: StudyCfg):
        self.uni, self.st = uni, st
        self.rng = random.Random(st.scale.sampling_seed)
        self.rejects: Dict[str, int] = defaultdict(int)
        self.forced = tuple(t for t in uni.ids if t in st.techs.force_include_all)
        self.free = [t for t in uni.ids if t not in self.forced]
        self._memo: Dict[Tuple[str, ...], Optional[str]] = {}

    def reason(self, ids) -> Optional[str]:
        k = tuple(ids)
        if k not in self._memo:
            self._memo[k] = self.uni.static_reason(k)
        return self._memo[k]

    def _canon(self, ids) -> Tuple[str, ...]:
        s = set(ids)
        return tuple(t for t in self.uni.ids if t in s)

    def raw_count(self) -> int:
        maxk = self.st.techs.max_links
        return sum(math.comb(len(self.free), r - len(self.forced)) for r in range(max(1, len(self.forced)), maxk + 1)
                   if r - len(self.forced) >= 0)

    def enumerate_valid(self, sizes: Optional[Sequence[int]] = None):
        maxk = self.st.techs.max_links
        for r in range(max(1, len(self.forced)), maxk + 1):
            if sizes is not None and r not in sizes:
                continue
            for extra in itertools.combinations(self.free, r - len(self.forced)):
                ids = self._canon(self.forced + extra)
                why = self.reason(ids)
                if why:
                    self.rejects[why] += 1
                else:
                    yield ids

    def random_valid(self, size: Optional[int] = None, tries: int = 400) -> Optional[Tuple[str, ...]]:
        maxk = self.st.techs.max_links
        for _ in range(tries):
            r = size or self.rng.randint(max(1, len(self.forced), self.st.techs.min_links), maxk)
            need = r - len(self.forced)
            if need < 0 or need > len(self.free):
                continue
            ids = self._canon(self.forced + tuple(self.rng.sample(self.free, need)))
            why = self.reason(ids)
            if not why:
                return ids
            self.rejects[why] += 1
        return None

    def sample_n(self, n: int, size: Optional[int] = None) -> List[Tuple[str, ...]]:
        out, seen, fails = [], set(), 0
        while len(out) < n and fails < 50 * max(n, 10):
            c = self.random_valid(size)
            if c is None or c in seen:
                fails += 1
                continue
            seen.add(c)
            out.append(c)
        return out

    def pairwise(self, t: int, limit: int) -> List[Tuple[str, ...]]:
        """Greedy t-way covering array over technology presence (every coverable t-subset appears together
        in at least one valid combo, and every single technology appears)."""
        maxk = self.st.techs.max_links
        pool: Dict[Tuple[str, ...], set] = {}
        for _ in range(max(1500, 40 * len(self.uni.ids))):
            c = self.random_valid(self.rng.randint(max(t, 1, self.st.techs.min_links), maxk) if maxk >= t else None)
            if c is None:
                continue
            cover = set()
            for r in range(1, min(t, len(c)) + 1):
                cover |= {frozenset(x) for x in itertools.combinations(c, r)}
            pool[c] = cover
        # also try to realise every t-subset explicitly (so coverable ones are not missed by chance)
        targets = {frozenset(x) for r in range(1, t + 1) for x in itertools.combinations(self.uni.ids, r)}
        chosen: List[Tuple[str, ...]] = []
        covered: set = set()
        while len(chosen) < limit:
            best, gain = None, 0
            for c, cov in pool.items():
                g = len((cov & targets) - covered)
                if g > gain:
                    best, gain = c, g
            if best is None:
                miss = [s for s in targets - covered]
                self.rng.shuffle(miss)
                made = None
                for s in miss[:60]:
                    for _ in range(25):
                        cand = set(s)
                        while len(cand) < self.rng.randint(len(s), maxk) and len(cand) < maxk:
                            cand.add(self.rng.choice(self.free + list(self.forced)))
                        c = self._canon(cand | set(self.forced))
                        if not self.reason(c):
                            made = c
                            break
                    if made:
                        break
                if not made:
                    break
                cov = set()
                for r in range(1, min(t, len(made)) + 1):
                    cov |= {frozenset(x) for x in itertools.combinations(made, r)}
                pool[made] = cov
                continue
            chosen.append(best)
            covered |= pool.pop(best)
        self.pairwise_uncovered = len(targets - covered)
        self.pairwise_targets = len(targets)
        return chosen

    def generate(self, strategy: Optional[str] = None) -> List[Tuple[str, ...]]:
        sc = self.st.scale
        strategy = strategy or sc.strategy
        lim = sc.max_combinations
        if strategy == "explicit":
            out = []
            for ids in sc.explicit:
                c = self._canon(ids)
                if any(t not in self.uni.specs for t in ids):
                    raise PlanError(f"explicit combo {ids} uses a technology that is not enabled")
                why = self.reason(c)
                if why:
                    self.rejects[why] += 1
                else:
                    out.append(c)
            return self._cap(out, lim, "explicit")
        if strategy == "pairwise":
            return self.pairwise(max(1, sc.strength), lim)
        if strategy == "random":
            return self.sample_n(min(sc.random_n, lim))
        if strategy == "stratified":
            out: List[Tuple[str, ...]] = []
            for r in range(max(1, len(self.forced)), self.st.techs.max_links + 1):
                cap = sc.per_size.get(str(r), sc.per_size.get("rest", "all"))
                nraw = math.comb(len(self.free), r - len(self.forced)) if r - len(self.forced) >= 0 else 0
                if cap == "all":
                    if nraw > sc.enumeration_limit:
                        raise PlanError(f"stratified size {r}: {nraw:,} raw subsets > enumeration_limit; give a numeric per_size cap")
                    out += list(self.enumerate_valid([r]))
                else:
                    out += self.sample_n(int(cap), r)
            return self._cap(out, lim, "stratified")
        if strategy in ("full", "halving"):
            if strategy == "halving":
                return self.generate(sc.halving.get("base_strategy", "pairwise"))
            nraw = self.raw_count()
            if nraw > sc.enumeration_limit:
                raise PlanError(f"full factorial needs enumerating {nraw:,} raw subsets (> enumeration_limit {sc.enumeration_limit:,}). "
                                "Use strategy pairwise / stratified / random / halving, lower max_links, or exclude technologies.")
            return self._cap(list(self.enumerate_valid()), lim, "full")
        raise PlanError(f"unknown strategy {strategy!r}")

    def _cap(self, out, lim, what):
        if len(out) <= lim:
            return out
        if self.st.scale.exceed_action == "sample":
            idx = sorted(random.Random(self.st.scale.sampling_seed).sample(range(len(out)), lim))
            return [out[i] for i in idx]
        raise PlanError(f"{what}: {len(out):,} valid combinations exceed max_combinations={lim:,}. Options: raise max_combinations, "
                        "set scale.exceed_action='sample', use strategy 'pairwise'/'halving'/'random', lower techs.max_links, or "
                        "exclude technologies / use force_include to shrink the space.")


# =============================================================================
# 7. PLAN
# =============================================================================
@dataclass
class Plan:
    st: StudyCfg
    uni: Universe
    geoms: Dict[str, Geom]
    combos: List[Combo]
    variants: List[dict]                     # expanded policy variants [{label, policy}]
    rejects: Dict[str, int]
    scn_rejects: Dict[str, int]
    notes: List[str]
    plan_hash: str = ""

    def video_modes_for(self, c: Combo) -> List[str]:
        out = []
        for vm in self.st.scale.video_modes:
            if vm == "standard":
                out.append(vm)
            elif vm == "ai" and self.uni.ai_enabled and (self.st.scale.ai_video.apply == "all" or c.constrained):
                out.append(vm)
        return out or ["standard"]

    def policies_for(self, c: Combo) -> List[dict]:
        P = self.st.policies
        if P.mode == "cross":
            return self.variants
        ref = [v for v in self.variants if v["label"] == P.reference or v["label"].startswith(P.reference + "[")] or self.variants[:1]
        if P.mode == "single":
            return ref[:1]
        rr = random.Random(f"{self.st.scale.sampling_seed}-{c.id}")
        others = [v for v in self.variants if v not in ref]
        return ref[:1] + rr.sample(others, min(max(P.per_combo - 1, 0), len(others)))

    def scn_list(self, c: Combo) -> List[str]:
        return [s for s, why in c.scn_ok.items() if why is None]

    def n_runs(self, seeds: Optional[int] = None) -> int:
        n = seeds if seeds is not None else self.st.scale.seeds
        return sum(len(self.scn_list(c)) * len(self.video_modes_for(c)) * len(self.policies_for(c)) * n for c in self.combos)


def make_plan(st: StudyCfg) -> Plan:
    if st.execution.backend not in ("fast", "high_fidelity"):
        raise PlanError("execution.backend must be 'fast' or 'high_fidelity'")
    if st.execution.physics_dt_s <= 0 or (st.execution.radio_dt_s is not None and st.execution.radio_dt_s <= 0):
        raise PlanError("execution.physics_dt_s and execution.radio_dt_s must be > 0")
    if st.scale.exceed_action not in ("error", "sample"):
        raise PlanError("scale.exceed_action must be 'error' or 'sample'")
    uni = Universe(st)
    if not uni.ids:
        raise PlanError("no technologies enabled")
    for vm in st.scale.video_modes:
        if vm not in ("standard", "ai"):
            raise PlanError(f"unknown video mode {vm!r} (use 'standard' / 'ai')")
    notes: List[str] = []
    geoms = {s: build_geom(st, uni, s) for s in st.scale.scenarios}
    gen = Generator(uni, st)
    raw = gen.generate()
    if st.scale.strategy == "pairwise" and getattr(gen, "pairwise_uncovered", 0):
        notes.append(f"pairwise: {gen.pairwise_uncovered} of {gen.pairwise_targets} {st.scale.strength}-subsets are not coverable "
                     "under the constraints or were cut by max_combinations")
    combos, scn_rej = [], defaultdict(int)
    for ids in sorted(raw, key=lambda c: (uni.n_links(c), c)):
        okmap = {}
        for sname, g in geoms.items():
            why = scenario_reason(uni, g, ids, st)
            okmap[sname] = why
            if why:
                scn_rej[why] += 1
        if all(v is not None for v in okmap.values()):
            continue
        wide = any(uni.specs[t].video and uni.link_dicts(t)[0][1]["bw_hz"] * max(uni.link_dicts(t)[0][1]["mcs_eff"]) *
                   uni.link_dicts(t)[0][1]["mac_eff"] >= st.scale.ai_video.constrained_mbps * 1e6 for t in ids)
        combos.append(Combo(ids, combo_id(ids), "+".join(ids), uni.n_links(ids), uni.peak_w(ids), uni.warm_w(ids), uni.mass_kg(ids), okmap, not wide))
    plist = []
    for p in st.policies.policies:
        if isinstance(p, str):
            if p not in PRESET_POLICIES:
                raise PlanError(f"unknown policy preset {p!r}; presets: {sorted(PRESET_POLICIES)}")
            plist.append(copy.deepcopy(PRESET_POLICIES[p]))
        else:
            q = copy.deepcopy(PRESET_POLICIES.get(p.get("preset"), {})) if p.get("preset") else {}
            q.update({k: v for k, v in p.items() if k != "preset"})
            if "label" not in q:
                q["label"] = q.get("name") or q.get("class")
            plist.append(q)
    variants = sim.expand_policies(plist)
    labels = [v["label"] for v in variants]
    if len(set(labels)) != len(labels):
        raise PlanError("duplicate policy labels: " + str(labels))
    if st.policies.mode != "cross" and not any(l == st.policies.reference or l.startswith(st.policies.reference + "[") for l in labels):
        notes.append(f"reference policy {st.policies.reference!r} not among the policies; first policy used as reference")
    plan = Plan(st, uni, geoms, combos, variants, dict(gen.rejects), dict(scn_rej), notes)
    # safety limit on the number of runs
    total = plan.n_runs() if st.scale.strategy != "halving" else plan.n_runs(max(s["seeds"] for s in st.scale.halving["stages"]))
    if st.scale.strategy == "halving":                                    # upper bound of stage costs
        total = _halving_upper_bound(plan)
    if total > st.scale.max_runs:
        if st.scale.exceed_action == "sample" and plan.combos:
            per = max(1, total // len(plan.combos))
            keep = max(1, st.scale.max_runs // per)
            rr = random.Random(st.scale.sampling_seed)
            plan.combos = sorted(rr.sample(plan.combos, min(keep, len(plan.combos))), key=lambda c: (c.n_links, c.ids))
            notes.append(f"max_runs={st.scale.max_runs:,}: combinations sampled down to {len(plan.combos)}")
        else:
            raise PlanError(f"planned {total:,} runs > max_runs={st.scale.max_runs:,}. Reduce seeds / policies / combinations "
                            "(strategy halving or pairwise), or raise max_runs.")
    plan.plan_hash = chash(identity_of(st, uni))
    return plan


def _halving_upper_bound(plan: Plan) -> int:
    stages = plan.st.scale.halving["stages"]
    alive = len(plan.combos)
    tot = 0
    avg_pol = max(1, int(np.mean([len(plan.policies_for(c)) for c in plan.combos]))) if plan.combos else 1
    avg_scn = max(1.0, float(np.mean([len(plan.scn_list(c)) for c in plan.combos]))) if plan.combos else 1.0
    vm = max(1, len(plan.st.scale.video_modes))
    prev_seeds = 0
    for s in stages:
        npol = 1 if s.get("policies") == "reference" else avg_pol
        tot += int(alive * avg_scn * vm * npol * max(0, s["seeds"] - (prev_seeds if npol == avg_pol and s is not stages[0] else 0)))
        alive = max(int(plan.st.scale.halving.get("min_survivors", 3)), int(math.ceil(alive * s.get("keep", 1.0))))
        prev_seeds = s["seeds"]
    return tot


def identity_of(st: StudyCfg, uni: Universe) -> dict:
    """Everything that changes what a given run_id MEANS (not how many seeds / which combos are scheduled)."""
    return dict(catalog={t: dict(link=sp.link, options=sp.options, control=sp.control, video=sp.video, mass=sp.mass_kg, faults=sp.faults)
                         for t, sp in uni.specs.items()},
                scenarios=st.scale.scenarios, duration=st.scale.duration_s, dt=st.scale.dt, ctrl_hz=st.scale.ctrl_hz,
                sim_overrides=st.scale.sim_overrides, master_seed=st.scale.master_seed, randomize=st.randomize, interference=st.interference,
                ai=st.scale.ai_video, hover=st.scale.payload_hover_w_per_kg, aliases=st.techs.aliases,
                backend=st.execution.backend, radio_overrides=st.techs.radio_overrides,
                physics_dt_s=st.execution.physics_dt_s, radio_dt_s=st.execution.radio_dt_s,
                policies=[p if isinstance(p, str) else p for p in st.policies.policies], wrap=st.policies.wrap_roles,
                prio=st.policies.reactive_priority, filters_prescreen={k: v for k, v in dataclasses.asdict(st.filters).items() if k not in ("max_mean_power_w", "min_control_reliability", "min_completion_rate")},
                eval=dict(source=st.metrics.source, cfg=st.metrics.eval_config), versions=[sim.__version__, getattr(ev, "__version__", None)])


# =============================================================================
# 8. SCENARIO BUILDER (combo -> uavlink_sim scenario dict)  and role-aware policy wrapper
# =============================================================================
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", s)


class WorkerCtx:
    """Everything a worker needs; built once per process from the study JSON. All construction is a pure
    function of (study, scenario, combo, video mode, seed) => any run can be rebuilt bit-exactly."""

    def __init__(self, st: StudyCfg):
        self.st = st
        self.uni = Universe(st)
        self._geoms: Dict[str, Geom] = {}
        self._scn: Dict[tuple, dict] = {}
        self._pol: Dict[tuple, dict] = {}
        self.univ_links = [nm for t in self.uni.ids for nm, _ in self.uni.link_dicts(t)]
        bands, seen = [], set()
        for t in self.uni.ids:
            sp = self.uni.specs[t]
            if sp.group in st.interference.jam_groups:
                k = (sp.link["freq_hz"], sp.link["bw_hz"])
                if k not in seen:
                    seen.add(k)
                    bands.append(k)
        self.jam_bands = bands
        self.randomize = {k: v for k, v in st.randomize.items() if k not in ("interference", "link_tx_jitter_db")}
        self.tx_jitter = float(st.randomize.get("link_tx_jitter_db", 0.0) or 0.0)
        self.eval_cfg = None
        if st.execution.backend == "fast" and ev is not None and st.metrics.source in ("eval", "both"):
            ov = sim.deep_merge({"spatial": {"enabled": False}, "plots": {"enabled": False}}, st.metrics.eval_config or {})
            self.eval_cfg = ev.from_dict(ev.EvalConfig, ov)

    def geom(self, scn: str) -> Geom:
        if scn not in self._geoms:
            self._geoms[scn] = build_geom(self.st, self.uni, scn)
        return self._geoms[scn]

    def alias(self, link: str) -> str:
        al = self.st.techs.aliases
        return ",".join(al.get(x.strip(), x.strip()) for x in link.split(",")) if link not in ("*", "any") else link

    def scenario(self, scn: str, ids: Tuple[str, ...], vm: str) -> dict:
        key = (scn, ids, vm)
        if key in self._scn:
            return self._scn[key]
        st, uni = self.st, self.uni
        g = self.geom(scn)
        d = copy.deepcopy(g.base)
        d["env"]["stations"] = [dict(s) for s in g.stations]
        links = []
        for tid in ids:
            spec = uni.specs[tid]
            lds = uni.link_dicts(tid)
            for k, (nm, ld) in enumerate(lds):
                ld["name"] = nm
                ld["station_ids"] = station_ids_for(spec, g, k, len(lds))
                if not ld["station_ids"]:
                    raise PlanError(f"{tid}: scenario {scn} has no stations of kind {spec.stations!r}")
                links.append(ld)
        d["links"] = links
        # fault processes: identical for every combo (CRN); 'any' is split over the whole technology universe
        procs = []
        for p in d.get("fault_processes", []):
            if p.get("link", "any") == "any":
                for nm in self.univ_links:
                    q = copy.deepcopy(p)
                    q["link"] = nm
                    q["rate_per_min"] = p.get("rate_per_min", 0.5) / len(self.univ_links)
                    procs.append(q)
            else:
                q = copy.deepcopy(p)
                q["link"] = self.alias(q["link"])
                procs.append(q)
        for tid in uni.ids:
            for prof in uni.specs[tid].faults:
                for nm in link_names_for(uni.specs[tid], uni.specs[tid].options):
                    q = copy.deepcopy(prof)
                    q["link"] = nm
                    procs.append(q)
        d["fault_processes"] = procs
        for f in d.get("faults", []):
            f["link"] = self.alias(f.get("link", "*"))
        # payload: radio mass costs hover power
        dr = d.setdefault("drone", {})
        base_dr = dataclasses.asdict(sim.DroneCfg())
        mass = uni.mass_kg(ids)
        comp_w = 0.0
        if vm == "ai":
            ai = st.scale.ai_video
            mass += ai.mass_kg
            comp_w = ai.compute_w
            tr = d.setdefault("traffic", {})
            tr["q_r0_bps"], tr["video_max_bps"], tr["fec"] = ai.q_r0_bps, ai.max_bps, ai.fec
            tr["video_deadline_ms"] = tr.get("video_deadline_ms", 250.0) - ai.encode_latency_ms
        dr["payload_power_w"] = dr.get("payload_power_w", base_dr["payload_power_w"]) + mass * st.scale.payload_hover_w_per_kg + comp_w
        dr["mass_kg"] = dr.get("mass_kg", base_dr["mass_kg"]) + mass
        if "fiber_tether" in ids:
            L = float(uni.specs["fiber_tether"].options.get("length_m", 1500.0))
            gf = d["env"].setdefault("geofence", {})
            cur = gf.get("radius_m", 0.0) or 0.0
            gf["radius_m"] = min(cur, L) if cur > 0 else L
        if not st.interference.keep_scenario:
            d["interference"] = []
        self._scn[key] = d
        return d

    def trial_scenario(self, scn: str, ids: Tuple[str, ...], vm: str, seed: int) -> dict:
        """Cached combo scenario + trial-level pieces that depend only on the trial seed (never on the combo)."""
        d = copy.deepcopy(self.scenario(scn, ids, vm))
        if self.tx_jitter > 0:
            for ld in d["links"]:
                r = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(78, sim.stable_hash(ld["name"]))))
                ld["ul_tx_dbm"] += float(r.normal(0.0, self.tx_jitter))
        spec = self.st.interference.random or {}
        if spec.get("count") and self.jam_bands:
            r = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(77,)))
            U = lambda rg: float(r.uniform(rg[0], rg[1]))
            n = int(r.integers(spec["count"][0], spec["count"][1] + 1))
            wps = d["mission"]["waypoints"]
            size = d["env"]["size_m"]
            kinds = spec.get("kinds", ["jammer", "wifi_burst", "pulsed_jammer"])
            for i in range(n):
                fc, bw = self.jam_bands[int(r.integers(len(self.jam_bands)))]
                w = wps[int(r.integers(len(wps)))]
                nx, ny = r.normal(0, 150, 2)
                ux, uy = r.uniform(0, size[0]), r.uniform(0, size[1])
                pos = [w["x"] + float(nx), w["y"] + float(ny), 6.0] if spec.get("near_path", True) else [float(ux), float(uy), 6.0]
                t0 = U(spec.get("t_start", [10, 100]))
                d.setdefault("interference", []).append(dict(
                    name=f"rnd{i}", kind=kinds[int(r.integers(len(kinds)))], t0=t0, t1=t0 + U(spec.get("dur", [5, 30])), pos=pos,
                    eirp_dbm=U(spec.get("eirp_dbm", [20, 36])), fc_hz=fc, bw_hz=bw * float(r.choice([1, 2])), duty=U(spec.get("duty", [0.3, 1.0]))))
        return d

    def roles(self, scn: str, ids: Tuple[str, ...]):
        g = self.geom(scn)
        info = []
        for tid in ids:
            sp = self.uni.specs[tid]
            des = tech_design(self.uni, g, tid, self.st)
            for nm, ld in self.uni.link_dicts(tid):
                info.append(dict(name=nm, tech=tid, control=sp.control, video=sp.video, c_ok=des["ctrl_eligible"], v_ok=des["vid_eligible"],
                                 lat=exp_latency_ms(ld), covers=des["covers_all"], vcap=ld["bw_hz"] * max(ld["mcs_eff"]) * ld["mac_eff"]))
        return info

    def policy_spec(self, scn: str, ids: Tuple[str, ...], label: str, variants: Dict[str, dict]) -> dict:
        key = (scn, ids, label)
        if key in self._pol:
            return self._pol[key]
        info = self.roles(scn, ids)
        P = self.st.policies
        ctrl_ok = [i for i in info if i["control"] and i["c_ok"]] or [i for i in info if i["control"]] or info
        if P.reactive_priority == "latency":
            ordered = sorted(ctrl_ok, key=lambda i: i["lat"])
        elif P.reactive_priority == "listed":
            ordered = list(ctrl_ok)
        else:
            ordered = sorted(ctrl_ok, key=lambda i: (not i["covers"], i["lat"]))
        prio = [i["name"] for i in ordered] + [i["name"] for i in info if i["name"] not in {o["name"] for o in ordered}]
        base = copy.deepcopy(variants[label])
        params = dict(base.get("params", {}))
        if base.get("name") == "reactive" and "priority" not in params:
            params["priority"] = prio
        if base.get("name") == "static_primary":
            vid = [i for i in info if i["video"] and i["v_ok"]]
            params.setdefault("control_link", prio[0])
            if vid:
                params.setdefault("video_link", max(vid, key=lambda i: i["vcap"])["name"])
        inner = {("class" if base.get("class") else "name"): base.get("class") or base["name"], "params": params}
        if not P.wrap_roles:
            spec = inner
        else:
            spec = {"class": "exhaustive_experiment:RolePolicy",
                    "params": {"inner": inner, "control_ok": [i["name"] for i in ctrl_ok],
                               "video_ok": [i["name"] for i in info if i["video"] and i["v_ok"]]}}
        self._pol[key] = spec
        return spec


class RolePolicy(sim.Policy):
    """Wraps ANY uavlink_sim policy and enforces technology roles the simulator itself does not model:
    a video-only link (analog FPV) never carries control, a control-only link never carries video, and links
    that cannot meet the deadline are not used for control. The inner policy sees inflated latency for
    non-control links (so cost-based policies avoid them); whatever it still returns is sanitised."""
    name = "role"

    def __init__(self, inner=None, control_ok=None, video_ok=None, **params):
        self.inner = sim.make_policy(inner)
        self.control_ok = list(control_ok or [])
        self.video_ok = list(video_ok or [])
        self.name = self.inner.name
        self.params = dict(params)
        self._fallback = None

    def reset(self, ctx, rng):
        self.ctx = ctx
        ok = set(self.control_ok) or set(ctx.order)
        self.control_set = ok
        links = {n: (s if n in ok else dataclasses.replace(s, exp_lat_ms=1e6)) for n, s in ctx.links.items()}
        self.ctx_inner = dataclasses.replace(ctx, links=links)
        self.inner.reset(self.ctx_inner, rng)
        self._fallback = None

    def initial(self, ctx):
        return self._sanitize(self.inner.initial(self.ctx_inner), None)

    def decide(self, obs):
        return self._sanitize(self.inner.decide(obs), obs)

    def _pick_fallback(self, obs):
        cands = [n for n in self.ctx.order if n in self.control_set]
        if self._fallback in cands:
            return self._fallback
        warm = [n for n in cands if obs is not None and obs.links[n].state == "warm"] or cands
        self._fallback = min(warm, key=lambda n: self.ctx.links[n].exp_lat_ms)
        return self._fallback

    def _sanitize(self, d, obs):
        ctrl = [n for n in d.control_links if n in self.control_set]
        if not ctrl:
            ctrl = [self._pick_fallback(obs)]
        vs = {n: v for n, v in d.video_shares.items() if (not self.video_ok) or n in self.video_ok}
        return dataclasses.replace(d, control_links=ctrl, video_shares=vs, video_bps=d.video_bps if vs else 0.0)


# =============================================================================
# 9. WORKER, LEDGER, EXECUTOR
# =============================================================================
_CTX: Dict[str, Any] = {}


def worker_init(st_json: str, sim_path: Optional[str], eval_path: Optional[str]):
    import multiprocessing
    if multiprocessing.current_process().name != "MainProcess":     # Ctrl-C is handled by the parent only
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sim_path:
        os.environ["UAVLINK_SIM_PATH"] = sim_path
    if eval_path:
        os.environ["UAVLINK_EVAL_PATH"] = eval_path
    st = sim.from_dict(StudyCfg, json.loads(st_json))
    _CTX["st"] = st
    _CTX["ctx"] = WorkerCtx(st)
    _CTX["variants"] = {v["label"]: v["policy"] for v in make_variants(st)}


def make_variants(st: StudyCfg) -> List[dict]:
    plist = []
    for p in st.policies.policies:
        if isinstance(p, str):
            plist.append(copy.deepcopy(PRESET_POLICIES[p]))
        else:
            q = copy.deepcopy(PRESET_POLICIES.get(p.get("preset"), {})) if p.get("preset") else {}
            q.update({k: v for k, v in p.items() if k != "preset"})
            q.setdefault("label", q.get("name") or q.get("class"))
            plist.append(q)
    return sim.expand_policies(plist)


def make_run_id(scn: str, cid: str, label: str, vm: str, trial: int) -> str:
    return f"{_safe(scn)}__{cid}__{_safe(label)}__{vm}__t{trial:05d}"


def run_exp_job(job: dict) -> dict:
    """Top-level (picklable) worker: one (scenario, combo, policy, video mode, trial)."""
    t0 = time.perf_counter()
    st: StudyCfg = _CTX["st"]
    ctx: WorkerCtx = _CTX["ctx"]
    ids = tuple(job["ids"])
    uni = ctx.uni
    row: Dict[str, Any] = dict(
        run_id=job["run_id"], scenario=job["scn"], combo_id=job["cid"], combo=job["combo"], techs="+".join(ids), n_techs=len(ids),
        n_links=uni.n_links(ids), policy=job["label"], video_mode=job["vm"], trial=job["trial"], seed=job["seed"], dt=st.scale.dt,
        duration_cfg_s=st.scale.duration_s, peak_radio_w=uni.peak_w(ids), warm_floor_w=uni.warm_w(ids), payload_kg=uni.mass_kg(ids),
        active_technologies="+".join(ids), backend=st.execution.backend,
        error="", job=dict(ids=list(ids), scn=job["scn"], vm=job["vm"], label=job["label"], trial=job["trial"], seed=job["seed"],
                           cid=job["cid"], combo=job["combo"], run_id=job["run_id"], keep_ticks=job.get("keep_ticks", False)))
    for t in uni.ids:
        row[f"tech_{t}"] = int(t in ids)
    tmp = None
    try:
        scn = ctx.trial_scenario(job["scn"], ids, job["vm"], job["seed"])
        if st.execution.backend == "high_fidelity":
            log_csv = None
            if job.get("keep_ticks"):
                log_dir = os.path.join(job["out"], "logs", job["run_id"])
                os.makedirs(log_dir, exist_ok=True)
                log_csv = os.path.join(log_dir, "ticks.csv")
            tech_specs = {t: dict(control=uni.specs[t].control, video=uni.specs[t].video, stations=uni.specs[t].stations) for t in ids}
            result = integration.run_high_fidelity(
                st, sim, scn, ids, _CTX["variants"][job["label"]], tech_specs,
                int(job["seed"]), video_mode=job["vm"], log_csv=log_csv, randomize=ctx.randomize)
            for metric in st.metrics.record:
                value = result["metrics"].get(metric)
                row[metric] = nanf(value) if isinstance(value, (int, float, np.floating, np.integer)) else NAN
            row.update(
                outcome=result["end_reason"], metrics_source="drone_multilink",
                active_technologies="+".join(result["active_technologies"]),
                radio_names="+".join(result["radio_names"]),
                unsupported_faults="+".join(result["unsupported_faults"]),
                sim_blind_time_s=nanf(result["metrics"].get("blind_time_s")),
                crc="", tick_hash="")
        else:
            pspec = ctx.policy_spec(job["scn"], ids, job["label"], _CTX["variants"])
            want_ticks = ctx.eval_cfg is not None or job.get("keep_ticks")
            out_dir = None
            if want_ticks:
                if job.get("keep_ticks"):
                    out_dir = os.path.join(job["out"], "logs")
                    os.makedirs(out_dir, exist_ok=True)
                else:
                    tmp = tempfile.mkdtemp(prefix="exp_")
                    out_dir = tmp
            r = sim.run_job(dict(run_id=job["run_id"], label=job["label"], trial=job["trial"], seed=job["seed"], scenario=scn, overrides=None,
                                 randomize=ctx.randomize, policy=pspec, log_level="full" if want_ticks else "none", out_dir=out_dir))
            if r.get("error"):
                raise RuntimeError("sim: " + r["error"])
            scal = {}
            if ctx.eval_cfg is not None:
                ld = os.path.join(out_dir, job["run_id"])
                tk = ev.load_ticks(os.path.join(ld, "ticks.csv.gz"))
                o = ev.evaluate_run(tk, ev.load_jsonl(os.path.join(ld, "events.jsonl")), ev.load_jsonl(os.path.join(ld, "decisions.jsonl")),
                                    json.load(open(os.path.join(ld, "scenario.json"))), ctx.eval_cfg, r)
                scal = o["scalars"]
            src = st.metrics.source if ctx.eval_cfg is not None else "sim"
            for m in st.metrics.record:
                ev_v, sim_v = scal.get(m), r.get(m)
                v = ev_v if (src in ("eval", "both") and ev_v is not None) else sim_v
                row[m] = nanf(v) if isinstance(v, (int, float, np.floating, np.integer)) else NAN
                if src == "both" and sim_v is not None and ev_v is not None:
                    row[m + "__sim"] = nanf(sim_v)
            row.update(outcome=r.get("outcome"), crc=r.get("crc"), tick_hash=r.get("tick_hash"), metrics_source=src,
                       sim_blind_time_s=nanf(r.get("blind_time_s")))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    row["wall_s"] = time.perf_counter() - t0
    return row


class Ledger:
    """Append-only JSONL: one line per finished run, full float precision. NaN/inf are stored as null."""

    def __init__(self, path: str, fsync_every: int = 50):
        self.path, self.fsync_every, self._n, self._f = path, fsync_every, 0, None

    def load(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:                   # torn last line after a crash
                        continue
                    out[r["run_id"]] = r
        return out

    def open(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = open(self.path, "a")

    def append(self, row: dict):
        self._f.write(json.dumps(jsonable(row), separators=(",", ":")) + "\n")
        self._f.flush()
        self._n += 1
        if self._n % self.fsync_every == 0:
            os.fsync(self._f.fileno())

    def close(self):
        if self._f:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            self._f = None


def iter_jobs(plan: Plan, done: set, seeds: int, trial_from: int = 0, only: Optional[Dict[str, set]] = None,
              policies: str = "all", out: str = ""):
    """Trial-major order so an interrupted study is balanced across combos. `only` = {combo_id: allowed}."""
    st = plan.st
    keep = st.metrics.save_ticks
    for trial in range(trial_from, seeds):
        seed = sim.trial_seed(st.scale.master_seed, trial)
        for c in plan.combos:
            if only is not None and c.id not in only:
                continue
            pols = plan.policies_for(c)
            if policies == "reference":
                ref = [v for v in plan.variants if v["label"] == st.policies.reference or v["label"].startswith(st.policies.reference + "[")]
                pols = ref[:1] or plan.variants[:1]
            for scn in plan.scn_list(c):
                for vm in plan.video_modes_for(c):
                    for v in pols:
                        rid = make_run_id(scn, c.id, v["label"], vm, trial)
                        if rid in done:
                            continue
                        kt = keep == "all" or (keep.startswith("first:") and trial < int(keep.split(":")[1]))
                        yield dict(run_id=rid, scn=scn, cid=c.id, combo=c.name, ids=list(c.ids), label=v["label"], vm=vm, trial=trial,
                                   seed=seed, keep_ticks=kt, out=out)


def count_jobs(plan: Plan, done: set, seeds: int, **kw) -> int:
    return sum(1 for _ in iter_jobs(plan, done, seeds, **kw))


def execute(plan: Plan, jobs, ledger: Ledger, workers: int, total: int, label: str = "", quiet: bool = False) -> dict:
    st = plan.st
    ex_cfg = st.execution
    stats = dict(done=0, errors=0, interrupted=False, wall=0.0)
    t0, last = time.perf_counter(), time.perf_counter()
    st_json = json.dumps(jsonable(dataclasses.asdict(st)))
    sim_path = os.environ.get("UAVLINK_SIM_PATH") or getattr(sim, "__file__", None)
    eval_path = os.environ.get("UAVLINK_EVAL_PATH") or (getattr(ev, "__file__", None) if ev else None)

    def sink(row):
        nonlocal last
        ledger.append(row)
        stats["done"] += 1
        stats["errors"] += bool(row.get("error"))
        now = time.perf_counter()
        if not quiet and (now - last >= ex_cfg.progress_interval_s or stats["done"] == total):
            last = now
            el = now - t0
            rate = stats["done"] / max(el, 1e-9)
            eta = (total - stats["done"]) / max(rate, 1e-9)
            print(f"  {label}[{stats['done']}/{total}] {100 * stats['done'] / max(total, 1):5.1f}%  {rate:6.2f} runs/s  "
                  f"elapsed {el:7.0f}s  ETA {eta:7.0f}s  errors {stats['errors']}", file=sys.stderr, flush=True)
    try:
        if workers <= 1:
            worker_init(st_json, sim_path, eval_path)
            for j in jobs:
                sink(run_exp_job(j))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(st_json, sim_path, eval_path)) as ex:
                pending, it, exhausted = set(), iter(jobs), False
                try:
                    while pending or not exhausted:
                        while not exhausted and len(pending) < workers * max(1, ex_cfg.inflight_factor):
                            try:
                                pending.add(ex.submit(run_exp_job, next(it)))
                            except StopIteration:
                                exhausted = True
                        if not pending:
                            break
                        dn, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in dn:
                            sink(f.result())
                except KeyboardInterrupt:
                    stats["interrupted"] = True
                    ex.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        stats["interrupted"] = True
    stats["wall"] = time.perf_counter() - t0
    return stats


# =============================================================================
# 10. HALVING (successive-halving screening over combinations)
# =============================================================================
def combo_scores(rows: Dict[str, dict], plan: Plan, rank_by: str, score_policy: str, min_seeds: int) -> Dict[str, float]:
    ref = plan.st.policies.reference
    per: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows.values():
        if r.get("error") or r.get("combo_id") is None:
            continue
        v = r.get(rank_by)
        if v is not None:
            per[(r["combo_id"], r["policy"])].append(float(v))
    out: Dict[str, float] = {}
    for c in plan.combos:
        cands = []
        for (cid, pol), vs in per.items():
            if cid != c.id or len(vs) < min_seeds:
                continue
            if score_policy == "reference" and not (pol == ref or pol.startswith(ref + "[")):
                continue
            cands.append(float(np.mean(vs)))
        if cands:
            out[c.id] = float(np.mean(cands)) if score_policy == "mean" else min(cands)
    return out


def run_halving(plan: Plan, ledger: Ledger, workers: int, quiet: bool = False) -> dict:
    H = plan.st.scale.halving
    done = ledger.load()
    ok_ids = {k for k, r in done.items() if not r.get("error")}
    alive = {c.id for c in plan.combos}
    info = dict(stages=[], interrupted=False)
    prev_seeds = 0
    for si, s in enumerate(H["stages"]):
        seeds, pol = int(s["seeds"]), s.get("policies", "all")
        jobs = list(iter_jobs(plan, ok_ids, seeds, 0, only=alive, policies=pol, out=plan.st.output.dir))
        if not quiet:
            print(f"[stage {si + 1}/{len(H['stages'])}] {len(alive)} combos, {seeds} seeds, policies={pol}: {len(jobs)} new runs", file=sys.stderr)
        stats = execute(plan, iter(jobs), ledger, workers, len(jobs), label=f"s{si + 1} ", quiet=quiet)
        info["stages"].append(dict(stage=si + 1, combos=len(alive), seeds=seeds, new_runs=len(jobs), **stats))
        if stats["interrupted"]:
            info["interrupted"] = True
            break
        done = ledger.load()
        ok_ids = {k for k, r in done.items() if not r.get("error")}
        sc = combo_scores(done, plan, H.get("rank_by", "blind_time_s"), H.get("score_policy", "best"), min(seeds, 2))
        ranked = sorted((c for c in alive if c in sc), key=lambda c: sc[c])
        keep = max(int(H.get("min_survivors", 3)), int(math.ceil(len(ranked) * float(s.get("keep", 1.0)))))
        alive = set(ranked[:keep]) if si < len(H["stages"]) - 1 else alive
        info["stages"][-1]["survivors"] = len(alive)
        prev_seeds = seeds
    info["final_combos"] = sorted(alive)
    return info


# =============================================================================
# 11. CALIBRATION (time a short simulation to estimate the run time in --dry-run)
# =============================================================================
def calibrate(plan: Plan) -> Optional[dict]:
    st = plan.st
    if not plan.combos:
        return None
    ctx = WorkerCtx(st)
    variants = {v["label"]: v["policy"] for v in make_variants(st)}
    pick = [max(plan.combos, key=lambda c: c.n_links), min(plan.combos, key=lambda c: c.n_links)]
    res = []
    if st.execution.backend == "high_fidelity":
        for c in pick:
            scn = plan.scn_list(c)[0]
            d = ctx.trial_scenario(scn, c.ids, "standard", sim.trial_seed(st.scale.master_seed, 0))
            lab = plan.policies_for(c)[0]["label"]
            specs = {t: dict(control=plan.uni.specs[t].control, video=plan.uni.specs[t].video,
                             stations=plan.uni.specs[t].stations) for t in c.ids}
            t0 = time.perf_counter()
            integration.run_high_fidelity(st, sim, d, c.ids, variants[lab], specs,
                                          sim.trial_seed(st.scale.master_seed, 0),
                                          video_mode="standard", duration_s=st.execution.calibrate_sim_s)
            duration = max(st.execution.calibrate_sim_s, 1e-9)
            res.append((time.perf_counter() - t0) / duration)
        per_tick_max, per_tick_min = max(res), min(res)
        dur = float(st.scale.duration_s or 120.0)
        return dict(per_tick_us_max=per_tick_max * 1e6, per_tick_us_min=per_tick_min * 1e6,
                    run_s_max=per_tick_max * dur, run_s_mean=0.5 * (per_tick_max + per_tick_min) * dur)
    for c in pick:
        scn = plan.scn_list(c)[0]
        d = ctx.trial_scenario(scn, c.ids, "standard", sim.trial_seed(st.scale.master_seed, 0))
        d = copy.deepcopy(d)
        d["duration_s"] = st.execution.calibrate_sim_s
        lab = plan.policies_for(c)[0]["label"]
        sc = sim.SimulationController(d, ctx.policy_spec(scn, c.ids, lab, variants), 0, log_level="none")
        t0 = time.perf_counter()
        sc.run()
        res.append((time.perf_counter() - t0) / max(sc.tick, 1))
    per_tick_max, per_tick_min = res
    dur, dt = float(st.scale.duration_s or 120.0), float(st.scale.dt or 0.05)
    ticks = dur / dt
    overhead = 1.0 + (0.35 if (st.metrics.source in ("eval", "both") and ev is not None) else 0.0)
    return dict(per_tick_us_max=per_tick_max * 1e6, per_tick_us_min=per_tick_min * 1e6, run_s_max=per_tick_max * ticks * overhead,
                run_s_mean=0.5 * (per_tick_max + per_tick_min) * ticks * overhead)


# =============================================================================
# 12. AGGREGATION, RANKING, STATISTICS
# =============================================================================
from statistics import NormalDist

_T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def tcrit(df: int, p: float = 0.975) -> float:
    if df <= 0:
        return NAN
    if p == 0.975 and df in _T_TABLE:
        return _T_TABLE[df]
    z = NormalDist().inv_cdf(p)
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2) + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3)


def load_rows(path: str) -> List[dict]:
    rows = [r for r in Ledger(path).load().values() if not r.get("error")]
    for r in rows:
        for k, v in list(r.items()):
            if v is None and k not in ("error", "crc", "tick_hash", "outcome"):
                r[k] = NAN
        add_derived(r)
    return rows


def add_derived(r: dict):
    q, b, p = r.get("video_quality_mean", NAN), r.get("blind_fraction", NAN), r.get("comm_power_mean_w", NAN)
    r["quality_per_watt"] = (q * (1.0 - b) / p) if (p == p and p > 0 and q == q and b == b) else NAN


def mstats(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return dict(n=0)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else NAN
    h = tcrit(n - 1) * sem if n > 1 else NAN
    return dict(n=n, mean=float(v.mean()), sd=sd, sem=sem, ci_lo=float(v.mean() - h) if n > 1 else NAN,
                ci_hi=float(v.mean() + h) if n > 1 else NAN, median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def build_entries(rows: List[dict], st: StudyCfg) -> List[dict]:
    per = st.ranking.scope == "per_scenario"
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["combo_id"], r["policy"], r["video_mode"]) + ((r["scenario"],) if per else ())].append(r)
    F = st.filters
    out = []
    for key, rs in groups.items():
        e = dict(combo_id=key[0], policy=key[1], video_mode=key[2], scenario=key[3] if per else "pooled", combo=rs[0]["combo"],
                 techs=rs[0]["techs"], n_links=rs[0]["n_links"], peak_radio_w=rs[0]["peak_radio_w"], payload_kg=rs[0]["payload_kg"],
                 n=len(rs), rows=rs, stats={})
        for m in dict.fromkeys(st.metrics.aggregate + [st.ranking.primary, st.ranking.efficiency, "ctrl_ontime_rate"]):
            e["stats"][m] = mstats(np.array([r.get(m, NAN) for r in rs], float))
        S = e["stats"]
        why = []
        if F.max_mean_power_w is not None and S["comm_power_mean_w"].get("mean", NAN) > F.max_mean_power_w:
            why.append("mean_power")
        if F.min_control_reliability > 0 and not S.get("ctrl_ontime_rate", {}).get("mean", 0) >= F.min_control_reliability:
            why.append("control_reliability")
        if F.min_completion_rate > 0 and not S.get("mission_completed", {}).get("mean", 0) >= F.min_completion_rate:
            why.append("completion")
        e["feasible"], e["infeasible_reasons"] = (not why), "+".join(why)
        out.append(e)
    return out


def _keyvals(e: dict, metric: str) -> Dict[tuple, float]:
    return {(r["scenario"], r["trial"]): r.get(metric, NAN) for r in e["rows"]}


def paired(a: dict, b: dict, metric: str) -> dict:
    ka, kb = _keyvals(a, metric), _keyvals(b, metric)
    common = sorted(set(ka) & set(kb))
    d = np.array([ka[k] - kb[k] for k in common], float)
    d = d[np.isfinite(d)]
    out = dict(n=len(d), mean_diff=float(d.mean()) if len(d) else NAN, p=NAN)
    if len(d) >= 3 and ev is not None:
        out["p"] = float(ev.wilcoxon_signed_rank(d)["p"])
        out["sd_diff"] = float(d.std(ddof=1))
    return out


def pareto_front(P: np.ndarray, senses: Sequence[int]) -> np.ndarray:
    X = np.asarray(P, float) * np.asarray(senses, float)
    keep = np.ones(len(X), bool)
    for i in range(len(X)):
        if not np.all(np.isfinite(X[i])):
            keep[i] = False
            continue
        dom = np.all(X <= X[i], axis=1) & np.any(X < X[i], axis=1)
        keep[i] = not dom.any()
    return keep


def rank_entries(entries: List[dict], st: StudyCfg) -> dict:
    R = st.ranking
    ns = [e["n"] for e in entries]
    min_runs = R.min_runs or (int(np.bincount(ns).argmax()) if ns else 0)
    pool = [e for e in entries if e["n"] >= min_runs and e["stats"][R.primary].get("n", 0) > 0]
    for e in pool:
        e["tie_key"] = e["stats"].get("comm_power_mean_w", {}).get("mean", 0.0)
    feas = sorted([e for e in pool if e["feasible"]], key=lambda e: (e["stats"][R.primary]["mean"], e["tie_key"], e["combo"], e["policy"]))
    infeas = sorted([e for e in pool if not e["feasible"]], key=lambda e: (e["stats"][R.primary]["mean"], e["combo"], e["policy"]))
    for i, e in enumerate(feas):
        e["rank_blind"] = i + 1
        e["vs_next"] = paired(e, feas[i + 1], R.primary) if (i + 1 < len(feas) and i < R.top_k) else {}
    # significance of top-K vs best (Holm)
    if feas and ev is not None:
        best = feas[0]
        cmp = [(e, paired(e, best, R.primary)) for e in feas[1:R.top_k + 1]]
        ps = [c[1]["p"] for c in cmp if c[1]["p"] == c[1]["p"]]
        adj = ev.adjust_p(ps, "holm") if ps else []
        it = iter(adj)
        for e, c in cmp:
            e["vs_best"] = dict(c, p_holm=next(it) if c["p"] == c["p"] else NAN)
            e["tied_with_best"] = bool(c["p"] == c["p"] and e["vs_best"]["p_holm"] >= R.alpha)
        best["vs_best"], best["tied_with_best"] = {}, True
    eff_hi = R.efficiency_higher_is_better
    effv = lambda e: e["stats"].get(R.efficiency, {}).get("mean", NAN)
    ef = sorted([e for e in feas if effv(e) == effv(e)], key=lambda e: (-effv(e) if eff_hi else effv(e), e["stats"][R.primary]["mean"]))
    for i, e in enumerate(ef):
        e["rank_eff"] = i + 1
    pm = R.pareto
    sen = [-1 if m in ("video_quality_mean", "ctrl_within_150ms_rate", "quality_per_watt", "mission_completed") else 1 for m in pm]
    P = np.array([[e["stats"].get(m, {}).get("mean", NAN) for m in pm] for e in feas]) if feas else np.zeros((0, len(pm)))
    mask = pareto_front(P, sen) if len(P) else np.array([], bool)
    for e, m in zip(feas, mask):
        e["pareto"] = bool(m)
    return dict(min_runs=min_runs, feasible=feas, infeasible=infeas, efficiency=ef, pareto=[e for e, m in zip(feas, mask) if m],
                excluded_few_runs=[e for e in entries if e["n"] < min_runs])


def tech_marginals(entries: List[dict], st: StudyCfg, metric: str, order: List[str]) -> List[dict]:
    """Average change of `metric` when one technology is added to an otherwise identical, already evaluated combination."""
    idx: Dict[tuple, float] = {}
    for e in entries:
        m = e["stats"][metric].get("mean", NAN)
        if m == m:
            idx[(e["policy"], e["video_mode"], e["scenario"], tuple(e["techs"].split("+")))] = m
    acc: Dict[tuple, List[float]] = defaultdict(list)
    for (pol, vm, scn, ids), m in idx.items():
        if len(ids) < 2:
            continue
        for t in ids:
            sub = tuple(x for x in ids if x != t)
            if (pol, vm, scn, sub) in idx:
                acc[(t, pol)].append(m - idx[(pol, vm, scn, sub)])
                acc[(t, "ALL")].append(m - idx[(pol, vm, scn, sub)])
    low = better_is_lower(metric)
    rows = []
    for (t, pol), ds in sorted(acc.items()):
        d = np.array(ds)
        rows.append(dict(tech=t, policy=pol, n_pairs=len(d), mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                         frac_improving=float(np.mean(d < 0) if low else np.mean(d > 0)), metric=metric))
    return rows


def better_is_lower(metric: str) -> bool:
    return metric not in ("video_quality_mean", "quality_per_watt", "ctrl_ontime_rate", "mission_completed", "ctrl_within_100ms_rate",
                          "ctrl_within_150ms_rate", "video_usable_fraction", "pred_auc", "pred_ep_recall", "pred_lead_mean_s")


# =============================================================================
# 13. OUTPUT FILES
# =============================================================================
META_COLS = ["run_id", "scenario", "combo_id", "combo", "n_techs", "n_links", "policy", "video_mode", "trial", "seed", "dt", "duration_cfg_s",
             "peak_radio_w", "warm_floor_w", "payload_kg", "active_technologies", "backend", "radio_names", "unsupported_faults", "outcome"]


def md_tab(head, rows) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def write_csv(path: str, head: List[str], rows: List[List[Any]], fmt: Fmt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in rows:
            w.writerow([fmt(v) if not isinstance(v, str) else v for v in r])


def write_outputs(plan: Optional[Plan], st: StudyCfg, out: str, info: Optional[dict] = None) -> dict:
    fmt = Fmt(st.metrics.decimals, st.metrics.number_format)
    mdf = Fmt(st.metrics.md_decimals, "fixed")
    rows = load_rows(os.path.join(out, "runs.jsonl"))
    rows.sort(key=lambda r: (r["scenario"], r["combo_id"], r["policy"], r["video_mode"], r["trial"]))
    uni = plan.uni if plan else Universe(st)
    tech_cols = [f"tech_{t}" for t in uni.ids]
    mcols = list(dict.fromkeys(st.metrics.record + ["quality_per_watt"]))
    extra = [c for c in rows[0] if c.endswith("__sim")] if rows else []
    head = META_COLS + ["techs"] + tech_cols + mcols + extra + ["metrics_source", "sim_blind_time_s", "crc", "tick_hash", "wall_s"]
    write_csv(os.path.join(out, "results.csv"), head, [[r.get(c, NAN if c in mcols or c in extra else "") for c in head] for r in rows], fmt)
    entries = build_entries(rows, st)
    rk = rank_entries(entries, st)
    # aggregates.csv
    am = list(dict.fromkeys(st.metrics.aggregate))
    ah = ["scenario", "combo_id", "combo", "n_links", "policy", "video_mode", "n_runs", "feasible", "infeasible_reasons"]
    for m in am:
        ah += [f"{m}_{s}" for s in ("mean", "sd", "sem", "ci_lo", "ci_hi", "median", "min", "max")]
    write_csv(os.path.join(out, "aggregates.csv"), ah,
              [[e["scenario"], e["combo_id"], e["combo"], e["n_links"], e["policy"], e["video_mode"], e["n"], int(e["feasible"]), e["infeasible_reasons"]] +
               [e["stats"][m].get(s, NAN) for m in am for s in ("mean", "sd", "sem", "ci_lo", "ci_hi", "median", "min", "max")] for e in entries], fmt)
    R = st.ranking
    S = lambda e, m, k="mean": e["stats"].get(m, {}).get(k, NAN)

    def rank_rows(lst, rk_key, f):
        return [[e.get(rk_key, "-"), e["combo"], e["policy"], e["video_mode"], e["n_links"], e["n"], f(S(e, R.primary)), f(S(e, R.primary, "sem")),
                 f"[{f(S(e, R.primary, 'ci_lo'))}, {f(S(e, R.primary, 'ci_hi'))}]", f(S(e, "blind_any")), f(S(e, "ctrl_ontime_rate")),
                 f(S(e, "ctrl_lat_p99_ms")), f(S(e, "video_quality_mean")), f(S(e, "comm_power_mean_w")), f(S(e, "energy_comm_wh")),
                 f(S(e, "quality_per_watt")), f(S(e, "n_hard_switch")), f(S(e, "n_soft_switch")), f(S(e, "mission_completed")),
                 int(e.get("pareto", False))] for e in lst]
    rh = ["rank", "combo", "policy", "video_mode", "links", "n", f"{R.primary} mean", "sem", "95% CI", "P(any blind)", "ctrl on-time", "ctrl p99 ms",
          "video Q", "comm W", "comm Wh", "Q/W", "hard sw", "soft sw", "completed", "pareto"]
    rb = rank_rows(rk["feasible"], "rank_blind", fmt)
    rb2 = [r + [fmt(e.get("vs_next", {}).get("p", NAN)), fmt(e.get("vs_best", {}).get("p_holm", NAN)), int(e.get("tied_with_best", False))]
           for r, e in zip(rb, rk["feasible"])]
    rh2 = rh + ["p vs next (Wilcoxon)", "p vs best (Holm)", "tied with best"]
    write_csv(os.path.join(out, "ranking_blind.csv"), rh2, rb2, fmt)
    ef_rows = rank_rows(rk["efficiency"], "rank_eff", fmt)
    write_csv(os.path.join(out, "ranking_efficiency.csv"), rh, ef_rows, fmt)
    write_csv(os.path.join(out, "pareto.csv"), rh, rank_rows(rk["pareto"], "rank_blind", fmt), fmt)
    write_csv(os.path.join(out, "infeasible.csv"), rh + ["reasons"], [r + [e["infeasible_reasons"]] for r, e in zip(rank_rows(rk["infeasible"], "rank_blind", fmt), rk["infeasible"])], fmt)
    # technology marginal values + policy summary
    tm = tech_marginals([e for e in entries if e["n"] >= rk["min_runs"]], st, R.primary, uni.ids)
    write_csv(os.path.join(out, "tech_marginal_value.csv"), ["tech", "policy", "metric", "n_pairs", "mean_delta", "median_delta", "frac_improving"],
              [[r["tech"], r["policy"], r["metric"], r["n_pairs"], r["mean_delta"], r["median_delta"], r["frac_improving"]] for r in tm], fmt)
    ps = []
    ref = st.policies.reference
    by_pol: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for e in entries:
        if e["n"] >= rk["min_runs"]:
            by_pol[e["policy"]][(e["combo_id"], e["video_mode"], e["scenario"])] = e
    refkey = next((p for p in by_pol if p == ref or p.startswith(ref + "[")), None)
    for pol, d in sorted(by_pol.items()):
        vals = np.array([S(e, R.primary) for e in d.values()], float)
        rec = dict(policy=pol, n_combos=len(d), mean_blind=float(np.nanmean(vals)) if len(vals) else NAN, best_blind=float(np.nanmin(vals)) if len(vals) else NAN,
                   wins_vs_ref=NAN, p_vs_ref=NAN, mean_diff_vs_ref=NAN)
        if refkey and pol != refkey:
            com = sorted(set(d) & set(by_pol[refkey]))
            diff = np.array([S(d[k], R.primary) - S(by_pol[refkey][k], R.primary) for k in com], float)
            diff = diff[np.isfinite(diff)]
            if len(diff):
                rec.update(mean_diff_vs_ref=float(diff.mean()), wins_vs_ref=float(np.mean(diff < 0)),
                           p_vs_ref=float(ev.wilcoxon_signed_rank(diff)["p"]) if (ev is not None and len(diff) >= 3) else NAN)
        ps.append(rec)
    write_csv(os.path.join(out, "policy_summary.csv"), ["policy", "n_combos", "mean_blind", "best_blind", "mean_diff_vs_ref(per combo)", "frac_combos_better_than_ref", "p_vs_ref"],
              [[r["policy"], r["n_combos"], r["mean_blind"], r["best_blind"], r["mean_diff_vs_ref"], r["wins_vs_ref"], r["p_vs_ref"]] for r in ps], fmt)
    # combos table
    if plan:
        write_csv(os.path.join(out, "combos.csv"), ["combo_id", "combo", "n_techs", "n_links", "peak_radio_w", "warm_floor_w", "payload_kg", "ai_target"] + [f"valid_{s}" for s in plan.geoms],
                  [[c.id, c.name, len(c.ids), c.n_links, c.peak_w, c.warm_w, c.mass_kg, int(c.constrained)] + [int(c.scn_ok[s] is None) for s in plan.geoms] for c in plan.combos], fmt)
    # json
    if st.output.write_results_json:
        js = dict(study=jsonable(dataclasses.asdict(st)), plan_hash=plan.plan_hash if plan else None, versions=dict(exhaustive=__version__, sim=sim.__version__,
                  eval=getattr(ev, "__version__", None), numpy=np.__version__, python=sys.version.split()[0]),
                  catalog={t: dict(label=sp.label, group=sp.group, control=sp.control, video=sp.video, link=sp.link, mass_kg=sp.mass_kg, options=sp.options, notes=sp.notes)
                           for t, sp in uni.specs.items()},
                  n_runs=len(rows), min_runs_for_ranking=rk["min_runs"],
                  rankings=dict(blind=[_entry_json(e) for e in rk["feasible"]], efficiency=[_entry_json(e) for e in rk["efficiency"]],
                                pareto=[_entry_json(e) for e in rk["pareto"]], infeasible=[_entry_json(e) for e in rk["infeasible"]]),
                  aggregates=[_entry_json(e) for e in entries], tech_marginal_value=tm, policy_summary=ps, info=info)
        if st.metrics.include_runs_in_json:
            js["runs"] = rows
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump(jsonable(js), f, indent=1)
    # summary.md
    md = ["# Exhaustive radio-technology x handover-policy experiment", "",
          f"Study `{st.name}` | plan hash `{plan.plan_hash if plan else '?'}` | {len(rows)} finished runs | {len(entries)} (combo, policy, mode) entries | "
          f"ranked entries need >= {rk['min_runs']} runs | primary metric `{R.primary}` (lower is better)", ""]
    if info and info.get("notes"):
        md += ["Notes:"] + [f"- {n}" for n in info["notes"]] + [""]
    comp = [r.get("mission_completed", NAN) for r in rows]
    comp = [c for c in comp if c == c]
    if comp and np.mean(comp) < 0.5:
        md += [f"> Only {100 * np.mean(comp):.0f}% of runs completed the mission within duration_s={st.scale.duration_s} s (outcome `timeout`). Blind time is then "
               "compared over equal flight windows (blind_fraction / wp_progress are recorded); raise scale.duration_s to compare complete missions.", ""]
    nerr = sum(1 for r in Ledger(os.path.join(out, "runs.jsonl")).load().values() if r.get("error"))
    if nerr:
        md += [f"> **{nerr} runs failed** (see runs.jsonl `error`); they are excluded from every table.", ""]
    ff = lambda v: mdf(v)
    k = R.top_k
    md += [f"## 1. Ranking by {R.primary} (feasible entries only, top {k})", "",
           md_tab(rh2, [r + [ff(e.get('vs_next', {}).get('p', NAN)), ff(e.get('vs_best', {}).get('p_holm', NAN)), int(e.get('tied_with_best', False))]
                        for r, e in zip(rank_rows(rk["feasible"][:k], "rank_blind", ff), rk["feasible"][:k])]), "",
           "`p vs next`: paired Wilcoxon over trials against the next-ranked entry (small p => the ordering of these two is real). "
           "`tied with best`: not distinguishable from rank 1 after Holm correction over the top-K.", ""]
    md += [f"## 2. Ranking by power efficiency ({R.efficiency}, {'higher' if R.efficiency_higher_is_better else 'lower'} is better), top {k}", "",
           md_tab(rh, rank_rows(rk["efficiency"][:k], "rank_eff", ff)), ""]
    md += ["## 3. Pareto-optimal entries " + str(R.pareto), "", md_tab(rh, rank_rows(rk["pareto"][:60], "rank_blind", ff)), ""]
    best_pol = defaultdict(list)
    for e in rk["feasible"]:
        best_pol[e["policy"]].append(e)
    md += ["## 4. Best combination per policy", "", md_tab(["policy", "best combo", "n links", f"{R.primary}", "comm W", "ctrl p99 ms", "video Q"],
           [[p, es[0]["combo"], es[0]["n_links"], ff(S(es[0], R.primary)), ff(S(es[0], "comm_power_mean_w")), ff(S(es[0], "ctrl_lat_p99_ms")), ff(S(es[0], "video_quality_mean"))]
            for p, es in sorted(best_pol.items())]), ""]
    bysize = defaultdict(list)
    for e in rk["feasible"]:
        bysize[e["n_links"]].append(e)
    md += ["## 5. Does adding links help? (best feasible entry per number of links)", "",
           md_tab(["links", "best combo", "policy", R.primary, "comm W", "n entries"], [[n, es[0]["combo"], es[0]["policy"], ff(S(es[0], R.primary)), ff(S(es[0], "comm_power_mean_w")), len(es)]
                                                                                        for n, es in sorted(bysize.items())]), ""]
    allm = [r for r in tm if r["policy"] == "ALL"]
    if allm:
        allm.sort(key=lambda r: r["mean_delta"] if better_is_lower(R.primary) else -r["mean_delta"])
        md += [f"## 6. Marginal value of each technology on `{R.primary}` (adding it to an already evaluated combo; negative = fewer blind seconds)", "",
               md_tab(["technology", "pairs", "mean delta", "median delta", "share improving"], [[r["tech"], r["n_pairs"], ff(r["mean_delta"]), ff(r["median_delta"]), ff(r["frac_improving"])] for r in allm]), ""]
    md += ["## 7. Policies compared across combinations (paired by combination)", "",
           md_tab(["policy", "combos", "mean blind", "best blind", "mean diff vs ref", "share of combos better", "p (Wilcoxon)"],
                  [[r["policy"], r["n_combos"], ff(r["mean_blind"]), ff(r["best_blind"]), ff(r["mean_diff_vs_ref"]), ff(r["wins_vs_ref"]), ff(r["p_vs_ref"])] for r in ps]), ""]
    dts = st.scale.dt or 0.05
    sems = [S(e, R.primary, "sem") for e in rk["feasible"][:10] if S(e, R.primary, "sem") == S(e, R.primary, "sem")]
    sds = [e.get("vs_next", {}).get("sd_diff", NAN) for e in rk["feasible"][:10]]
    sds = [x for x in sds if x == x]
    n_ = rk["min_runs"]
    md += ["## 8. How far can you trust small differences?", "",
           f"- Per-run blind time is a multiple of the time step **dt = {dts:g} s**; a mean over n = {n_} runs therefore has a granularity of dt/n = {dts / max(n_, 1):.6f} s.",
           f"- Typical standard error of the top-10 means: {np.median(sems) if sems else float('nan'):.6f} s. Two entries whose CIs overlap are not separable without more seeds.",
           (f"- Because every entry uses the same trial seeds (common random numbers), neighbours are compared on paired differences; the median SD of those differences in the top-10 is "
            f"{np.median(sds):.6f} s, so a paired difference of about {1.96 * np.median(sds) / math.sqrt(max(n_, 1)):.6f} s is the smallest effect resolvable with n = {n_} (alpha 0.05, 50% power)." if sds else
            "- Paired resolution needs >= 3 common trials per neighbour pair."),
           f"- Values are stored with {st.metrics.decimals} decimals ({st.metrics.number_format}); runs.jsonl keeps full float precision. Recording precision is not statistical precision: use the `p vs next` column.",
           "- Re-running any recorded run reproduces it bit-exactly (`audit` command checks the state CRC and every recorded metric).", ""]
    if plan:
        md += ["## 9. Plan, filters and rejections", "", f"- Technologies in the universe ({len(plan.uni.ids)}): {', '.join(plan.uni.ids)}",
               f"- Strategy `{st.scale.strategy}`, max_links {st.techs.max_links}, {len(plan.combos)} combinations, policies {[v['label'] for v in plan.variants]}, "
               f"scenarios {st.scale.scenarios}, video modes {st.scale.video_modes}",
               f"- Rejected while generating (static rules): {dict(plan.rejects)}", f"- Rejected by link-budget / tether / coverage pre-screen: {dict(plan.scn_rejects)}"]
        md += [f"- {n}" for n in plan.notes] + [""]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    return dict(n_rows=len(rows), entries=len(entries), best=rk["feasible"][0]["combo"] if rk["feasible"] else None, ranking=rk)


def _entry_json(e: dict) -> dict:
    return {k: v for k, v in e.items() if k not in ("rows",)}


# =============================================================================
# 14. COMMANDS
# =============================================================================
def apply_cli(st_over: dict, a) -> dict:
    d = st_over
    def put(path, v):
        if v is None:
            return
        cur = d
        ks = path.split(".")
        for k in ks[:-1]:
            cur = cur.setdefault(k, {})
        cur[ks[-1]] = v
    put("output.dir", a.out)
    put("scale.seeds", a.seeds)
    put("scale.duration_s", a.duration)
    put("scale.dt", a.dt)
    put("scale.scenarios", a.scenario or None)
    put("scale.strategy", a.strategy)
    put("scale.max_combinations", a.max_combinations)
    put("scale.max_runs", a.max_runs)
    put("scale.exceed_action", a.exceed_action)
    put("scale.master_seed", a.master_seed)
    put("scale.video_modes", a.video_modes.split(",") if a.video_modes else None)
    put("techs.max_links", a.max_links)
    put("techs.min_links", a.min_links)
    put("techs.force_include_all", a.include or None)
    put("techs.force_exclude", a.exclude or None)
    put("techs.enabled", a.only.split(",") if a.only else None)
    put("policies.policies", a.policies.split(",") if a.policies else None)
    put("metrics.save_ticks", a.save_ticks)
    put("metrics.decimals", a.decimals)
    put("metrics.source", a.eval_mode)
    put("metrics.record", a.metrics.split(",") if a.metrics else None)
    put("execution.workers", a.workers)
    put("execution.backend", getattr(a, "backend", None))
    put("execution.progress_interval_s", a.progress_interval)
    put("filters.max_radio_power_w", a.max_radio_power)
    put("filters.min_control_reliability", a.min_control_reliability)
    put("filters.max_mean_power_w", a.max_mean_power)
    put("ranking.top_k", a.top_k)
    put("sim_module", a.sim_module)
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:
            pass
        put(k, v)
    if a.dry_run:
        put("execution.dry_run", True)
    if a.resume:
        put("execution.resume", True)
    if a.no_calibrate:
        put("execution.calibrate", False)
    if a.no_retry_errors:
        put("execution.retry_errors", False)
    if a.force_resume:
        put("execution.force_resume", True)
    return d


def describe_plan(plan: Plan, est: Optional[dict], workers: int) -> str:
    st = plan.st
    L = [f"Study: {st.name}", f"Technologies enabled ({len(plan.uni.ids)}): {', '.join(plan.uni.ids)}",
         f"Strategy: {st.scale.strategy}   max simultaneous links: {st.techs.max_links}   forced include: {st.techs.force_include_all or '-'}   "
         f"forced include any-of: {st.techs.force_include_any or '-'}   excluded: {st.techs.force_exclude or '-'}",
         f"Scenarios: {st.scale.scenarios}   duration {st.scale.duration_s}s   dt {st.scale.dt}s   seeds/combo {st.scale.seeds}   master seed {st.scale.master_seed}",
         f"Policies ({len(plan.variants)}, mode {st.policies.mode}): {[v['label'] for v in plan.variants]}   video modes: {st.scale.video_modes}"]
    try:
        raw = Generator(plan.uni, st).raw_count()
        L.append(f"Raw subset space (<= max_links): {raw:,}")
    except Exception:
        pass
    L.append(f"Valid combinations after static rules + scenario pre-screen: {len(plan.combos)}")
    L.append(f"  rejected (static): {dict(plan.rejects) or '-'}")
    L.append(f"  rejected (per scenario budget/tether/coverage): {dict(plan.scn_rejects) or '-'}")
    by = defaultdict(int)
    for c in plan.combos:
        by[c.n_links] += 1
    L.append(f"  combinations by number of links: {dict(sorted(by.items()))}")
    runs = _halving_upper_bound(plan) if st.scale.strategy == "halving" else plan.n_runs()
    L.append(f"Planned runs{' (upper bound)' if st.scale.strategy == 'halving' else ''}: {runs:,}   (safety limit max_runs = {st.scale.max_runs:,})")
    for n in plan.notes:
        L.append(f"  note: {n}")
    if est:
        tot = runs * est["run_s_mean"]
        L.append(f"Calibration: {est['per_tick_us_min']:.0f}-{est['per_tick_us_max']:.0f} us/tick  =>  ~{est['run_s_mean']:.1f} s per run (max {est['run_s_max']:.1f} s)")
        L.append(f"Estimated CPU time {tot / 3600:.2f} h  =>  wall time ~{tot / max(workers, 1) / 3600:.2f} h on {workers} worker(s)")
    return "\n".join(L)


def cmd_run(a) -> int:
    over = apply_cli({}, a)
    st = load_study(a.config, over)
    if st.sim_module:
        os.environ["UAVLINK_SIM_PATH"] = st.sim_module
    workers = st.execution.workers or max(1, (os.cpu_count() or 2) - 1)
    try:
        plan = make_plan(st)
    except PlanError as e:
        print(f"PLAN ERROR: {e}", file=sys.stderr)
        return 2
    out = st.output.dir
    est = calibrate(plan) if (st.execution.dry_run and st.execution.calibrate) else None
    print(describe_plan(plan, est, workers))
    os.makedirs(out, exist_ok=True)
    if st.output.plan_files:
        json.dump(jsonable(dict(plan_hash=plan.plan_hash, rejects=plan.rejects, scn_rejects=plan.scn_rejects, notes=plan.notes,
                                combos=[dict(id=c.id, name=c.name, n_links=c.n_links, peak_w=c.peak_w, mass_kg=c.mass_kg, scn_ok=c.scn_ok) for c in plan.combos])),
                  open(os.path.join(out, "plan.json"), "w"), indent=1)
    if st.execution.dry_run:
        write_csv(os.path.join(out, "combos.csv"), ["combo_id", "combo", "n_links", "peak_radio_w", "warm_floor_w", "payload_kg"] + [f"valid_{s}" for s in plan.geoms],
                  [[c.id, c.name, c.n_links, c.peak_w, c.warm_w, c.mass_kg] + [int(c.scn_ok[s] is None) for s in plan.geoms] for c in plan.combos], Fmt(6))
        print(f"\nDRY RUN: nothing executed. Plan written to {out}/plan.json and combos.csv")
        return 0
    ledger = Ledger(os.path.join(out, "runs.jsonl"), st.execution.fsync_every)
    existing = ledger.load()
    mpath = os.path.join(out, "manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        if existing and not st.execution.resume:
            print(f"ERROR: {out} already holds {len(existing)} runs; use --resume to continue or a new --out.", file=sys.stderr)
            return 2
        if old.get("plan_hash") != plan.plan_hash and existing and not st.execution.force_resume:
            print("ERROR: the study definition changed since these runs were recorded (plan hash differs); "
                  "results would not be comparable. Use a new --out, or --force-resume if you are sure.", file=sys.stderr)
            return 2
    json.dump(jsonable(dict(plan_hash=plan.plan_hash, study=dataclasses.asdict(st), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            versions=dict(exhaustive=__version__, sim=sim.__version__, eval=getattr(ev, "__version__", None),
                                          drone=getattr(integration.drone, "__version__", "unknown"),
                                          predictive_handover=getattr(integration.handover, "__version__", "unknown"),
                                          numpy=np.__version__, python=sys.version.split()[0]))),
              open(mpath, "w"), indent=1)
    json.dump(jsonable(dataclasses.asdict(st)), open(os.path.join(out, "study_resolved.json"), "w"), indent=1)
    done_ok = {k for k, r in existing.items() if not r.get("error") or not st.execution.retry_errors}
    ledger.open()
    try:
        if st.scale.strategy == "halving":
            info = run_halving(plan, ledger, workers)
        else:
            total = count_jobs(plan, done_ok, st.scale.seeds, out=out)
            print(f"\n{len(done_ok):,} runs already recorded; {total:,} to run on {workers} worker(s) ...", file=sys.stderr)
            stats = execute(plan, iter_jobs(plan, done_ok, st.scale.seeds, out=out), ledger, workers, total)
            info = dict(stages=[dict(stage=1, new_runs=total, **stats)], interrupted=stats["interrupted"])
    finally:
        ledger.close()
    info["notes"] = plan.notes
    res = write_outputs(plan, st, out, info)
    print(f"\nwrote {out}/results.csv ({res['n_rows']} rows), results.json, ranking_blind.csv, ranking_efficiency.csv, pareto.csv, summary.md")
    if info.get("interrupted"):
        print(f"INTERRUPTED: partial results written. Continue with:  python exhaustive_experiment.py run --config {a.config or '<config>'} --out {out} --resume")
        return 130
    r0 = res["ranking"]["feasible"][:5]
    if r0:
        print("\nTop by blind time:")
        for e in r0:
            print(f"  #{e['rank_blind']}  {e['combo']:<60s} {e['policy']:<16s} blind {e['stats'][st.ranking.primary]['mean']:.6f} s  comm {e['stats']['comm_power_mean_w'].get('mean', NAN):.3f} W")
    return 0


def cmd_report(a) -> int:
    out = a.out
    mp = os.path.join(out, "study_resolved.json")
    if not os.path.exists(mp):
        print("no study_resolved.json in --out", file=sys.stderr)
        return 2
    st = sim.from_dict(StudyCfg, json.load(open(mp)))
    if a.decimals is not None:
        st.metrics.decimals = a.decimals
    if st.sim_module:
        os.environ["UAVLINK_SIM_PATH"] = st.sim_module
    try:
        plan = make_plan(dataclasses.replace(st, execution=dataclasses.replace(st.execution, dry_run=True)))
    except PlanError:
        plan = None
    res = write_outputs(plan, st, out, {"notes": ["report regenerated from the ledger"]})
    print(f"regenerated reports from {res['n_rows']} runs in {out}")
    return 0


def cmd_audit(a) -> int:
    out = a.out
    st = sim.from_dict(StudyCfg, json.load(open(os.path.join(out, "study_resolved.json"))))
    rows = [r for r in Ledger(os.path.join(out, "runs.jsonl")).load().values() if not r.get("error")]
    rr = random.Random(a.seed)
    pick = rr.sample(rows, min(a.n, len(rows)))
    st_json = json.dumps(jsonable(dataclasses.asdict(st)))
    worker_init(st_json, os.environ.get("UAVLINK_SIM_PATH") or sim.__file__, os.environ.get("UAVLINK_EVAL_PATH") or (ev.__file__ if ev else None))
    bad = 0
    for r in pick:
        j = r["job"]
        job = dict(run_id=j["run_id"], scn=j["scn"], cid=j["cid"], combo=j["combo"], ids=j["ids"], label=j["label"], vm=j["vm"], trial=j["trial"],
                   seed=j["seed"], keep_ticks=False, out=out)
        new = run_exp_job(job)
        diffs = []
        for k in st.metrics.record + ["crc", "tick_hash"]:
            a_, b_ = r.get(k), jsonable(new.get(k))
            if a_ != b_ and not (a_ is None and b_ is None):
                if isinstance(a_, float) and isinstance(b_, float) and a_ == b_:
                    continue
                diffs.append((k, a_, b_))
        diffs = [d for d in diffs if d[0] != "tick_hash" or d[1]]
        print(f"{'OK  ' if not diffs else 'DIFF'} {r['run_id']}" + ("" if not diffs else f"  {diffs[:3]}"))
        bad += bool(diffs)
    print(f"audit: {len(pick) - bad}/{len(pick)} runs reproduced bit-exactly")
    return 1 if bad else 0


def cmd_list_techs(a) -> int:
    uni = Universe(load_study(None))
    print(f"{'id':<20s}{'group':<12s}{'ctrl':<6s}{'vid':<5s}{'fc GHz':>8s}{'bw MHz':>9s}{'peak W':>8s}{'warm W':>8s}{'lat ms':>8s}{'act s':>7s}{'kg':>6s}  label")
    for t, sp in uni.specs.items():
        l = sp.link
        print(f"{t:<20s}{sp.group:<12s}{'Y' if sp.control else '-':<6s}{'Y' if sp.video else '-':<5s}{l['freq_hz'] / 1e9:>8.3f}{l['bw_hz'] / 1e6:>9.3f}"
              f"{peak_radio_w(l):>8.1f}{l['warm_w']:>8.1f}{exp_latency_ms(l):>8.1f}{l['activation_delay_s']:>7.1f}{sp.mass_kg:>6.2f}  {sp.label}")
    print("\nTechnology aliases (legacy name -> canonical radio):")
    for alias, canonical in sorted(integration.RADIO_ALIASES.items()):
        print(f"  {alias} -> {canonical}")
    if a.budget:
        print("\nFree-space design SNR (dB) downlink/uplink at 0.5 / 2 / 5 / 10 km (satellite/fiber: fixed geometry shown at their own distance):")
        for t, sp in uni.specs.items():
            l = sp.link
            if sp.stations == "sat":
                o = sp.options
                dist = [math.hypot(o["orbit_km"] * 1000.0 / math.tan(math.radians(o["elevation_deg"])), o["orbit_km"] * 1000.0)]
                lab = [f"slant {dist[0] / 1000:.0f} km"]
            elif sp.stations == "tether":
                dist, lab = [2000.0, 2500.0], ["2.0 km", "2.5 km"]
            else:
                dist, lab = [500.0, 2000.0, 5000.0, 10000.0], ["0.5", "2", "5", "10"]
            cells = [f"{lb}: {design_snr_db(l, d, 'dl'):6.1f}/{design_snr_db(l, d, 'ul'):6.1f}" for lb, d in zip(lab, dist)]
            thr = sim.mcs_threshold_db(l["ctrl_eff"])
            print(f"  {t:<20s} ctrl thr {thr:5.1f} dB | " + " | ".join(cells))
    return 0


def preset_config(name: str) -> dict:
    base = json.loads(json.dumps(jsonable(dataclasses.asdict(StudyCfg()))))
    base["_readme"] = ("Every key is optional. techs.* = which radios and how many at once; filters.* = invalid-combination / feasibility rules; "
                       "policies.* = which handover policies; scale.* = seeds, flight duration, dt, scenarios, search strategy and safety limits; "
                       "metrics.* = what is recorded and how precisely; execution.* = workers / resume / progress / dry-run. CLI flags override this file.")
    if name == "demo":
        base["name"] = "demo"
        base["techs"].update(enabled=["analog_fpv_58", "digital_58_dji", "digital_58_openhd", "sub_868_900", "ism_24_ctrl", "cell_lte"], max_links=2)
        base["scale"].update(strategy="full", seeds=3, duration_s=60.0, max_combinations=100, max_runs=2000)
        base["output"]["dir"] = "experiments/demo"
    elif name == "screening":
        base["name"] = "screening"
        base["scale"].update(strategy="halving", seeds=12, duration_s=120.0, max_combinations=400, max_runs=40000,
                             video_modes=["standard", "ai"])
        base["techs"]["max_links"] = 3
        base["output"]["dir"] = "experiments/screening"
        base["metrics"]["save_ticks"] = "first:1"
    elif name == "full":
        base["name"] = "full"
        base["scale"].update(strategy="stratified", seeds=20, max_combinations=2000, max_runs=500000, per_size={"1": "all", "2": "all", "3": 400, "4": 400},
                             exceed_action="sample", scenarios=["urban_canyon", "rural_ridge", "jammed_inspection"], duration_s=300.0)
        base["techs"]["max_links"] = 4
        base["policies"]["policies"] = ["reactive", "bonding", "pred_threshold", "pred_cost", "pred_aggressive", "pred_conservative"]
        base["output"]["dir"] = "experiments/full"
    return base


# =============================================================================
# 15. SELFTEST
# =============================================================================
def selftest() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)
    cat = build_catalog()
    bad = []
    for t, sp in cat.items():
        try:
            sim.from_dict(sim.LinkCfg, dict(name=t, carries_video=sp.video, **sp.link))
        except Exception as e:
            bad.append((t, str(e)))
    check(f"all {len(cat)} catalog entries are valid uavlink_sim.LinkCfg {bad[:1]}", not bad)
    ranges_ok = True
    for t, sp in cat.items():
        l = sp.link
        if sp.stations == "sat":
            o = sp.options
            d = math.hypot(o["orbit_km"] * 1e3 / math.tan(math.radians(o["elevation_deg"])), o["orbit_km"] * 1e3)
            s_dl, s_ul = design_snr_db(l, d, "dl"), design_snr_db(l, d, "ul")
            if t == "geo_ka":
                ranges_ok &= math.isfinite(s_dl) and math.isfinite(s_ul)
            elif not (2.0 <= s_dl <= 40 and 2.0 <= s_ul <= 40):
                ranges_ok = False
                print("     link budget out of range:", t, round(s_dl, 1), round(s_ul, 1))
        elif sp.stations == "tether":
            ranges_ok &= design_snr_db(l, 2500.0, "dl") > 40
        elif sp.stations == "gcs" and sp.video:
            ranges_ok &= design_snr_db(l, 1000.0, "ul") > 8
        elif sp.stations == "gcs":
            ranges_ok &= design_snr_db(l, 2000.0, "dl") > 8
    check("design link budgets are plausible (sat 2-40 dB at slant range, fiber >40 dB, RF >8 dB at 1-2 km)", ranges_ok)
    # combination generation
    st = load_study(None, {"techs": {"enabled": ["fpv58_analog", "ocusync", "rf900", "elrs24", "cell", "lr12g"], "max_links": 3},
                           "scale": {"strategy": "full", "max_combinations": 500, "scenarios": ["jammed_inspection"], "seeds": 1}})
    plan = make_plan(st)
    names = {c.name for c in plan.combos}
    check("exclusive slot: analog FPV and DJI digital never together", not any("fpv58_analog" in n and "ocusync" in n for n in names))
    check("every combo has a control link and a video link", all(any(plan.uni.specs[t].control for t in c.ids) and any(plan.uni.specs[t].video for t in c.ids) for c in plan.combos))
    check("video-only analog FPV alone is rejected (no control), analog+sub900 is kept", "fpv58_analog" not in names and "fpv58_analog+rf900" in names)
    check("max_links respected", max(c.n_links for c in plan.combos) <= 3)
    st2 = load_study(None, {"techs": {"enabled": ["ocusync", "rf900", "cell", "lr12g"], "max_links": 3, "force_include_all": ["rf900"], "force_exclude": ["lr12g"]},
                            "scale": {"strategy": "full", "max_combinations": 50, "scenarios": ["jammed_inspection"]}})
    p2 = make_plan(st2)
    check("force_include / force_exclude honoured", all("rf900" in c.ids and "lr12g" not in c.ids for c in p2.combos) and len(p2.combos) == 3)
    try:
        make_plan(load_study(None, {"techs": {"max_links": 3}, "scale": {"strategy": "full", "max_combinations": 10, "scenarios": ["jammed_inspection"]}}))
        check("max_combinations safety limit raises", False)
    except PlanError:
        check("max_combinations safety limit raises", True)
    stp = load_study(None, {"techs": {"enabled": ["fpv58_analog", "ocusync", "openhd58", "rf900", "elrs24", "cell", "lr12g", "lr33g"], "max_links": 3},
                            "scale": {"strategy": "pairwise", "strength": 2, "max_combinations": 200, "scenarios": ["jammed_inspection"]}})
    pp = make_plan(stp)
    cov = {frozenset(x) for c in pp.combos for r in (1, 2) for x in itertools.combinations(c.ids, r)}
    gen = Generator(pp.uni, stp)
    allv = list(gen.enumerate_valid())
    coverable = {frozenset(x) for c in allv for r in (1, 2) for x in itertools.combinations(c, r)}
    unc = [x for x in coverable if x not in cov]
    check(f"pairwise covering array: {len(pp.combos)} of {len(allv)} valid combos cover all {len(coverable)} coverable singles/pairs (missing {len(unc)})", not unc)
    # CRN: same trial seed => identical environment for different combos
    wc = WorkerCtx(st)
    s1 = wc.trial_scenario("jammed_inspection", ("ocusync",), "standard", 99)
    s2 = wc.trial_scenario("jammed_inspection", ("rf900", "cell"), "standard", 99)
    e1, e2 = sim.Environment(sim.from_dict(sim.EnvCfg, s1["env"]), 99), sim.Environment(sim.from_dict(sim.EnvCfg, s2["env"]), 99)
    check("common random numbers: identical map/buildings for different combos", np.array_equal(e1.b_lo, e2.b_lo) and np.array_equal(e1.H, e2.H) and e1.nb > 0)
    check("fault processes identical for every combo (same list)", s1["fault_processes"] == s2["fault_processes"])
    # role enforcement
    scn = wc.trial_scenario("jammed_inspection", ("fpv58_analog", "rf900"), "standard", 5)
    scn = copy.deepcopy(scn)
    scn["duration_s"] = 20.0
    variants = {v["label"]: v["policy"] for v in make_variants(st)}
    bad_ctrl = 0
    for lab in ("reactive", "bonding", "pred_cost"):
        sc = sim.SimulationController(scn, wc.policy_spec("jammed_inspection", ("fpv58_analog", "rf900"), lab, variants), 5)
        while not sc.done and sc.t < 15:
            sc.step()
            bad_ctrl += "fpv58_analog" in sc.decision.control_links
    check("role wrapper: video-only analog FPV is never used for control (reactive, bonding, predictive)", bad_ctrl == 0)
    # end to end tiny study, resume, dry-run
    tmp = tempfile.mkdtemp(prefix="exp_selftest_")
    try:
        cfgd = {"name": "selftest", "output": {"dir": os.path.join(tmp, "o")},
                "techs": {"max_links": 2, "options": {"fiber_tether": {"length_m": 3000.0}}},
                "scale": {"strategy": "explicit", "explicit": [["ocusync"], ["fpv58_analog", "rf900"], ["fiber_tether"], ["leo", "rf900"]],
                          "seeds": 2, "duration_s": 30.0, "dt": 0.05, "ctrl_hz": 20.0, "scenarios": ["jammed_inspection"], "video_modes": ["standard", "ai"],
                          "ai_video": {"apply": "all"}},
                "policies": {"policies": ["reactive", "pred_cost"]}, "execution": {"workers": 1, "progress_interval_s": 999, "calibrate": False},
                "metrics": {"decimals": 6}}
        cp = os.path.join(tmp, "s.json")
        json.dump(cfgd, open(cp, "w"))
        ns = types_ns(config=cp, dry_run=True)
        check("dry-run returns without running", cmd_run(ns) == 0 and not os.path.exists(os.path.join(tmp, "o", "runs.jsonl")))
        t0 = time.time()
        rc = cmd_run(types_ns(config=cp))
        el = time.time() - t0
        rows = list(csv.DictReader(open(os.path.join(tmp, "o", "results.csv"))))
        exp_n = 4 * 2 * 2 * 2
        check(f"tiny study ran {len(rows)}/{exp_n} runs in {el:.0f}s", rc == 0 and len(rows) == exp_n)
        bl = [float(r["blind_time_s"]) for r in rows]
        check("results.csv has fine-grained numbers (6 decimals) and per-technology flags", all(len(r["blind_time_s"].split(".")[-1]) == 6 for r in rows) and "tech_fiber_tether" in rows[0])
        check("ranking / pareto / summary / json files written", all(os.path.exists(os.path.join(tmp, "o", f)) for f in ("ranking_blind.csv", "ranking_efficiency.csv", "pareto.csv", "summary.md", "results.json", "aggregates.csv")))
        n_before = len(Ledger(os.path.join(tmp, "o", "runs.jsonl")).load())
        rc = cmd_run(types_ns(config=cp, resume=True))
        check("resume with nothing left runs 0 new jobs", rc == 0 and len(Ledger(os.path.join(tmp, "o", "runs.jsonl")).load()) == n_before)
        aud = cmd_audit(types_ns(out=os.path.join(tmp, "o"), n=3, seed=1))
        check("audit: re-running recorded runs reproduces them bit-exactly", aud == 0)
        ai = [r for r in rows if r["video_mode"] == "ai"]
        check("AI-video mode is a separate recorded factor", len(ai) == len(rows) // 2)
        sim_vs_eval = max(abs(float(r["blind_time_s"]) - float(r["sim_blind_time_s"])) for r in rows)
        check(f"eval-recomputed blind time agrees with the simulator's (max diff {sim_vs_eval:.3f}s)", sim_vs_eval < 0.5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def types_ns(**kw):
    d = dict(config=None, out=None, seeds=None, duration=None, dt=None, scenario=None, strategy=None, max_combinations=None, max_runs=None,
             exceed_action=None, master_seed=None, video_modes=None, max_links=None, min_links=None, include=None, exclude=None, only=None,
             policies=None, save_ticks=None, decimals=None, eval_mode=None, metrics=None, workers=None, progress_interval=None, max_radio_power=None,
             min_control_reliability=None, max_mean_power=None, top_k=None, sim_module=None, set=None, dry_run=False, resume=False, no_calibrate=True,
             no_retry_errors=False, force_resume=False, n=5, seed=0, budget=False, preset="demo")
    d.update(kw)
    return argparse.Namespace(**d)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    if "--list-techs" in argv:
        argv = ["list-techs"] + [x for x in argv if x != "--list-techs"]
    elif argv and argv[0].startswith("-"):
        argv = ["run"] + argv
    ap = argparse.ArgumentParser(prog="exhaustive_experiment", description="Exhaustive radio-technology x handover-policy experiment runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list-techs"); p.add_argument("--budget", action="store_true", help="print free-space design SNRs")
    p = sub.add_parser("make-config"); p.add_argument("--preset", default="demo", choices=["demo", "screening", "full"]); p.add_argument("--out", default="study.json")
    p = sub.add_parser("run", help="plan + execute (use --dry-run to only plan)")
    p.add_argument("--config"); p.add_argument("--out"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--workers", type=int); p.add_argument("--seeds", type=int); p.add_argument("--duration", type=float); p.add_argument("--dt", type=float)
    p.add_argument("--backend", choices=["fast", "high_fidelity"], help="fast uavlink_sim or native drone_multilink physics")
    p.add_argument("--scenario", action="append"); p.add_argument("--strategy", choices=["full", "stratified", "random", "pairwise", "explicit", "halving"])
    p.add_argument("--max-links", type=int); p.add_argument("--min-links", type=int); p.add_argument("--include", action="append"); p.add_argument("--exclude", action="append")
    p.add_argument("--only", help="comma list of technology ids / groups to enable"); p.add_argument("--policies", help="comma list of policy presets")
    p.add_argument("--max-combinations", type=int); p.add_argument("--max-runs", type=int); p.add_argument("--exceed-action", choices=["error", "sample"])
    p.add_argument("--master-seed", type=int); p.add_argument("--video-modes"); p.add_argument("--save-ticks"); p.add_argument("--decimals", type=int)
    p.add_argument("--eval-mode", choices=["eval", "sim", "both"]); p.add_argument("--metrics", help="comma list of metrics to record")
    p.add_argument("--progress-interval", type=float); p.add_argument("--max-radio-power", type=float); p.add_argument("--min-control-reliability", type=float)
    p.add_argument("--max-mean-power", type=float); p.add_argument("--top-k", type=int); p.add_argument("--sim-module"); p.add_argument("--set", action="append")
    p.add_argument("--no-calibrate", action="store_true"); p.add_argument("--no-retry-errors", action="store_true"); p.add_argument("--force-resume", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--out", required=True); p.add_argument("--decimals", type=int)
    p = sub.add_parser("audit"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "list-techs":
        return cmd_list_techs(a)
    if a.cmd == "make-config":
        json.dump(preset_config(a.preset), open(a.out, "w"), indent=1)
        print("wrote", a.out)
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "report":
        return cmd_report(a)
    if a.cmd == "audit":
        return cmd_audit(a)
    if a.cmd == "selftest":
        return 0 if selftest() else 1


if __name__ == "__main__":
    sys.exit(main())
