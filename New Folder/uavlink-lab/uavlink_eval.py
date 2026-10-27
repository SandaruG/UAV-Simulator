#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uavlink_eval.py - Evaluation, Metrics & Visualisation for multi-link UAV handover experiments.

PURPOSE
    Turns raw simulation logs (from uavlink_sim.py, or any simulator mapped through a
    LogSchema) into quantified scientific evidence: blind time, control-latency tails,
    video quality/freezes, handover count AND cost, radio energy/spectrum, prediction
    quality; paired statistics against several baselines; goal scorecards with
    configurable accuracy targets; tables (md/csv/tex) and figures (timelines, heatmaps...).

KEY DESIGN POINTS
    * Metrics are RECOMPUTED from per-tick logs under configurable definitions
      (reliability windows, ratios, gaps, freeze threshold, latency limits ...), so you can
      re-ask "what if 'reliable control' meant <=100 ms?" without re-simulating.
      Runs without tick logs can be RE-SIMULATED bit-exactly (--resim, CRC-verified) and are
      evaluated on the fly, so a 4000-run batch never has to store 4000 tick logs.
      Runs that can neither be read nor re-simulated fall back to summary metrics (tier B).
    * Unit of analysis = trial (not packet/tick): paired by (scenario, trial) because the
      simulator uses common random numbers. Statistics: paired bootstrap CIs, Wilcoxon
      signed-rank (exact when possible), sign test, sign-flip permutation test, McNemar for
      "any blind episode", Friedman across policies, Holm / Benjamini-Hochberg correction,
      effect sizes (Cohen dz, Cliff delta, P(superiority), Hodges-Lehmann), non-inferiority,
      power / minimum detectable effect.
    * Threshold-tuning honesty: optional hold-out selection (choose the sweep variant on the
      first part of the trials, report it on the held-out rest) and multiplicity correction
      across all variants otherwise.
    * Pure numpy (+ matplotlib for figures, optional). No scipy needed.

USAGE
    python uavlink_eval.py make-config --out eval_config.json
    python uavlink_eval.py evaluate --batch batches/study1 --config eval_config.json \
                                    --out eval/study1 --resim --workers 8 [--holdout 0.5]
    python uavlink_eval.py timeline --run runs/urban_canyon__cost_predictive__s7
    python uavlink_eval.py selftest
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import dataclasses
import fnmatch
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import types
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "1.0.0"
NAN = float("nan")


# =============================================================================
# 1. CONFIGURATION (every threshold and target is here)
# =============================================================================
def _L(*v):
    return field(default_factory=lambda: list(v))


@dataclass
class LogSchema:
    """Logical column -> actual column name in ticks.csv(.gz). Defaults match uavlink_sim."""
    t: str = "t"
    ctrl_sent: str = "ctrl_sent"
    ctrl_ok: str = "ctrl_ok"
    ctrl_lat_ms: str = "ctrl_lat_ms"
    vid_ok: str = "vid_ok"
    vid_q: str = "vid_q"
    video_bps: str = "video_bps"
    mode: str = "mode"
    x: str = "x"
    y: str = "y"
    z: str = "z"
    p_comm_w: str = "p_comm_w"
    ctrl_links: str = "ctrl_links"                 # "primary+backup"
    ctrl_rel: str = "ctrl_rel"                     # logged reliability flags (blind_mode=logged)
    vid_rel: str = "vid_rel"
    blind: str = "blind"
    link_discover_suffix: str = ".rx_dl_dbm"       # used to discover link names
    link_state: str = ".state"
    link_power: str = ".p_w"
    link_outage: str = ".phy_out"
    link_rx: str = ".rx_dl_dbm"
    link_sinr: str = ".sinr_dl_db"


@dataclass
class Definitions:
    blind_mode: str = "recompute"                  # recompute | logged
    ctrl_window_s: float = 0.3
    ctrl_ok_ratio: float = 0.8                     # control reliable if on-time ratio in window >= this ...
    ctrl_gap_max_ms: float = 200.0                 # ... and no gap longer than this since last on-time packet
    deadline_ms: float = 150.0                     # packet on time if latency <= this
    vid_window_s: float = 0.5
    vid_ok_ratio: float = 0.7                      # video usable if decodable ratio in window >= this
    vid_min_quality: float = 0.0                   # chunk counts as usable only if quality >= this
    freeze_s: float = 0.2                          # stall >= this is a freeze
    latency_limits_ms: List[float] = _L(100.0, 150.0)
    lost_penalty_ms: float = 1000.0                # lost packet counts as this latency in "effective" percentiles
    critical_blind_s: List[float] = _L(1.0, 3.0)   # count blind episodes at least this long
    merge_gap_s: float = 0.0                       # merge blind episodes separated by <= this
    airborne_modes: List[str] = _L("TAKEOFF", "MISSION", "HOLD", "RTL", "LAND")
    min_flight_s: float = 5.0                      # runs with less airborne time are excluded


@dataclass
class HandoverCostCfg:
    hard: float = 1.0                              # cost units per hard switch (cold target)
    soft: float = 0.2                              # per soft switch (target already warm)
    pingpong: float = 2.0                          # extra per ping-pong
    disruption_per_s: float = 1.0                  # per s of unreliable control in window after switch
    lost_packet: float = 0.005                     # per control packet lost in window
    blind_per_s: float = 2.0                       # per s of blind time in window
    window_s: float = 2.0
    pingpong_s: float = 3.0


@dataclass
class UtilityCfg:                                  # composite cost J (lower is better) for ranking variants
    w_blind: float = 10.0                          # * blind fraction
    w_late: float = 3.0                            # * control deadline-miss rate
    w_quality: float = 1.0                         # * (1 - video quality)
    w_energy: float = 0.3                          # * mean comm power / power budget


@dataclass
class BudgetCfg:
    power_w: Optional[float] = None                # None -> scenario.json budget, else 8 W
    spectrum_hz: Optional[float] = None            # None -> scenario.json budget, else 40 MHz
    battery_wh: Optional[float] = None


@dataclass
class SpatialCfg:
    enabled: bool = True
    cell_m: float = 100.0
    n_phase_bins: int = 20                         # normalised flight-progress bins
    time_bin_s: float = 20.0                       # absolute-time bins


@dataclass
class PredictionCfg:
    enabled: bool = True
    horizon_s: float = 3.0
    threshold: float = 0.5                         # operating point for recall / false alarms
    thresholds: List[float] = _L(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    outage_min_ms: float = 200.0                   # physical outage must last this long to count
    label_mode: str = "onset"                      # onset | any_overlap
    primary_only: bool = True                      # score only the link that is primary at call time
    alert_merge_s: float = 1.0
    calib_bins: int = 10
    hist_bins: int = 100
    conv_window_s: float = 2.0                     # primary outage -> blind "conversion" window


@dataclass
class StatsCfg:
    alpha: float = 0.05
    bootstrap_B: int = 5000
    permutation_B: int = 20000
    seed: int = 12345
    correction: str = "holm"                       # holm | bh | none
    primary_metric: str = "blind_time_s"
    primary_test: str = "wilcoxon"                 # wilcoxon | permutation | sign
    power: float = 0.8
    min_pairs: int = 5
    family_metrics: List[str] = _L("blind_time_s", "blind_any", "ctrl_lat_p99_ms", "video_quality_mean",
                                   "video_freeze_s", "comm_power_mean_w", "handover_cost")


@dataclass
class GroupCfg:
    proposed: List[str] = _L("pred*", "cost_predictive*")      # fnmatch patterns on policy label
    baselines: List[str] = _L()                                  # empty = every non-proposed policy
    by_scenario: bool = False


@dataclass
class GoalCfg:
    """type: absolute_min | absolute_max | relative_reduction | relative_increase | noninferior | superior
    relative_*: target is a fraction (0.5 = 50 %) vs `vs`;  noninferior: margin in metric units
    (lower-better metric: proposed - baseline <= margin; higher-better: baseline - proposed <= margin)."""
    name: str = "goal"
    metric: str = "blind_time_s"
    type: str = "absolute_max"
    target: float = 0.0
    margin: float = 0.0
    vs: Optional[str] = None                       # baseline pattern
    policy: Optional[str] = None                   # None = proposed patterns


def default_goals() -> List[GoalCfg]:
    return [
        GoalCfg("Blind time cut >=50% vs reactive", "blind_time_s", "relative_reduction", target=0.5, vs="reactive*"),
        GoalCfg("Blind time not worse than bonding by >1 s", "blind_time_s", "noninferior", margin=1.0, vs="bonding*"),
        GoalCfg("Blind time significantly lower than static", "blind_time_s", "superior", vs="static*"),
        GoalCfg("Control p99 latency <= 150 ms", "ctrl_lat_p99_ms", "absolute_max", target=150.0),
        GoalCfg("Control within 150 ms >= 99 %", "ctrl_within_150ms_rate", "absolute_min", target=0.99),
        GoalCfg("Video quality not worse than reactive by >0.03", "video_quality_mean", "noninferior", margin=0.03, vs="reactive*"),
        GoalCfg("Video freeze time cut >=30% vs reactive", "video_freeze_s", "relative_reduction", target=0.3, vs="reactive*"),
        GoalCfg("Mean comm power <= 8 W budget", "comm_power_mean_w", "absolute_max", target=8.0),
        GoalCfg("Comm energy <=70% of bonding", "energy_comm_wh", "relative_reduction", target=0.3, vs="bonding*"),
        GoalCfg("Outage-prediction AUC >= 0.90", "pred_auc", "absolute_min", target=0.90),
        GoalCfg("Outage episode recall >= 0.90", "pred_ep_recall", "absolute_min", target=0.90),
        GoalCfg("Mean warning lead time >= 1.0 s", "pred_lead_mean_s", "absolute_min", target=1.0),
        GoalCfg("False alarms <= 2 per minute", "pred_false_alarms_per_min", "absolute_max", target=2.0),
        GoalCfg("Brier score <= 0.10", "pred_brier", "absolute_max", target=0.10),
    ]


@dataclass
class SelectionCfg:
    holdout_frac: float = 0.0                      # 0 = off; 0.5 = choose on first half of trials, report on rest
    objective: str = "utility_cost"                # metric minimised among feasible variants
    constraints: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"metric": "ctrl_within_150ms_rate", "op": ">=", "value": 0.99},
        {"metric": "comm_power_mean_w", "op": "<=", "value": 8.0}])


@dataclass
class PlotCfg:
    enabled: bool = True
    dpi: int = 130
    fmt: str = "png"
    n_timelines: int = 2                           # comparison timelines for most informative trials
    max_policies: int = 14
    sweep_metrics: List[str] = _L("blind_time_s", "video_quality_mean", "ctrl_lat_p99_ms", "comm_power_mean_w")
    pareto_x: str = "comm_power_mean_w"
    pareto_y: str = "blind_time_s"
    pareto_size: str = "video_quality_mean"


@dataclass
class EvalConfig:
    schema: LogSchema = field(default_factory=LogSchema)
    definitions: Definitions = field(default_factory=Definitions)
    handover: HandoverCostCfg = field(default_factory=HandoverCostCfg)
    utility: UtilityCfg = field(default_factory=UtilityCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    spatial: SpatialCfg = field(default_factory=SpatialCfg)
    prediction: PredictionCfg = field(default_factory=PredictionCfg)
    stats: StatsCfg = field(default_factory=StatsCfg)
    groups: GroupCfg = field(default_factory=GroupCfg)
    goals: List[GoalCfg] = field(default_factory=default_goals)
    selection: SelectionCfg = field(default_factory=SelectionCfg)
    plots: PlotCfg = field(default_factory=PlotCfg)


def from_dict(cls, d: Optional[dict]):
    d = d or {}
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    bad = set(d) - known
    if bad:
        raise ValueError(f"unknown key(s) {sorted(bad)} for {cls.__name__}; valid: {sorted(known)}")
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
        return dict(v)
    if tp is float and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    return v


def load_config(path: Optional[str], overrides: Optional[dict] = None) -> EvalConfig:
    d = {}
    if path:
        with open(path) as f:
            d = json.load(f)
    if overrides:
        d = _deep_merge(d, overrides)
    return from_dict(EvalConfig, d)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    raise TypeError(type(o))


def chash(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=_jdefault).encode()).hexdigest()[:16]


# =============================================================================
# 2. METRIC REGISTRY (direction used for 'better', tables and heatmap colouring)
# =============================================================================
#  name: (label, better: min|max, unit)
METRICS: Dict[str, Tuple[str, str, str]] = {
    "blind_time_s": ("Blind time", "min", "s"),
    "blind_fraction": ("Blind fraction", "min", ""),
    "blind_per_min": ("Blind time / flight min", "min", "s/min"),
    "blind_any": ("P(any blind episode)", "min", ""),
    "n_blind_episodes": ("Blind episodes", "min", ""),
    "max_blind_episode_s": ("Longest blind episode", "min", "s"),
    "mean_blind_episode_s": ("Mean blind episode", "min", "s"),
    "first_blind_s": ("Time to first blind", "max", "s"),
    "ctrl_unreliable_s": ("Control unreliable", "min", "s"),
    "video_unusable_s": ("Video unusable", "min", "s"),
    "ctrl_only_loss_s": ("Control lost, video OK", "min", "s"),
    "ctrl_ontime_rate": ("Control on-time rate", "max", ""),
    "ctrl_lost_rate": ("Control lost rate", "min", ""),
    "ctrl_late_rate": ("Control late rate", "min", ""),
    "ctrl_within_100ms_rate": ("Control <=100 ms", "max", ""),
    "ctrl_within_150ms_rate": ("Control <=150 ms", "max", ""),
    "ctrl_lat_mean_ms": ("Control latency mean", "min", "ms"),
    "ctrl_lat_p50_ms": ("Control latency p50", "min", "ms"),
    "ctrl_lat_p90_ms": ("Control latency p90", "min", "ms"),
    "ctrl_lat_p95_ms": ("Control latency p95", "min", "ms"),
    "ctrl_lat_p99_ms": ("Control latency p99", "min", "ms"),
    "ctrl_lat_p999_ms": ("Control latency p99.9", "min", "ms"),
    "ctrl_lat_max_ms": ("Control latency max", "min", "ms"),
    "ctrl_lat_eff_p99_ms": ("Control p99 (loss=penalty)", "min", "ms"),
    "ctrl_jitter_ms": ("Control jitter (std)", "min", "ms"),
    "ctrl_max_gap_ms": ("Longest control gap", "min", "ms"),
    "video_quality_mean": ("Video quality", "max", ""),
    "video_usable_fraction": ("Video usable fraction", "max", ""),
    "video_decodable_rate": ("Video decodable rate", "max", ""),
    "video_freeze_s": ("Video freeze time", "min", "s"),
    "video_freeze_events": ("Video freezes", "min", ""),
    "video_freeze_max_s": ("Longest freeze", "min", "s"),
    "video_mean_mbps": ("Video delivered rate", "max", "Mbps"),
    "n_hard_switch": ("Hard switches", "min", ""),
    "n_soft_switch": ("Soft switches", "min", ""),
    "n_switch": ("Switches", "min", ""),
    "n_pingpong": ("Ping-pongs", "min", ""),
    "switch_per_min": ("Switches / min", "min", "1/min"),
    "handover_cost": ("Handover cost", "min", "units"),
    "handover_disruption_s": ("Post-switch disruption", "min", "s"),
    "proactive_switch_frac": ("Proactive switch share", "max", ""),
    "hard_time_to_usable_s": ("Hard-switch time to usable", "min", "s"),
    "dup_time_s": ("Control duplication time", "min", "s"),
    "n_rlf": ("Radio link failures", "min", ""),
    "n_failsafe": ("Failsafe triggers", "min", ""),
    "energy_comm_wh": ("Radio energy", "min", "Wh"),
    "comm_power_mean_w": ("Radio power mean", "min", "W"),
    "comm_power_p99_w": ("Radio power p99", "min", "W"),
    "over_power_budget_s": ("Over power budget", "min", "s"),
    "spectrum_mean_mhz": ("Active spectrum mean", "min", "MHz"),
    "over_spectrum_budget_s": ("Over spectrum budget", "min", "s"),
    "energy_per_video_mbit_j": ("Energy per delivered video Mbit", "min", "J/Mbit"),
    "energy_comm_frac_battery": ("Radio share of battery", "min", ""),
    "utility_cost": ("Composite cost J", "min", ""),
    "mission_completed": ("Mission completed", "max", ""),
    "pred_auc": ("Outage AUC", "max", ""),
    "pred_ap": ("Outage average precision", "max", ""),
    "pred_brier": ("Brier score", "min", ""),
    "pred_ece": ("Expected calib. error", "min", ""),
    "pred_ep_recall": ("Outage episode recall", "max", ""),
    "pred_precision": ("Alert precision", "max", ""),
    "pred_lead_mean_s": ("Warning lead time", "max", "s"),
    "pred_false_alarms_per_min": ("False alarms / min", "min", "1/min"),
    "pred_actioned_rate": ("Outages acted on before onset", "max", ""),
    "outage_to_blind_rate": ("Primary outage -> blind rate", "min", ""),
    "n_primary_outages": ("Primary outage episodes", "min", ""),
}
BINARY_METRICS = {"blind_any", "mission_completed"}

HEADLINE = ["blind_time_s", "blind_any", "max_blind_episode_s", "ctrl_within_100ms_rate", "ctrl_within_150ms_rate",
            "ctrl_lat_p99_ms", "video_quality_mean", "video_freeze_s", "handover_cost", "n_hard_switch",
            "n_soft_switch", "energy_comm_wh", "comm_power_mean_w", "utility_cost", "mission_completed"]


def better_is_lower(metric: str) -> bool:
    return METRICS.get(metric, ("", "min", ""))[1] == "min"


def mlabel(metric: str) -> str:
    m = METRICS.get(metric)
    return f"{m[0]} [{m[2]}]" if m and m[2] else (m[0] if m else metric)


# =============================================================================
# 3. LOG IO
# =============================================================================
def load_ticks(path: str) -> Dict[str, np.ndarray]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        cols = [[] for _ in head]
        for row in rd:
            for i, v in enumerate(row):
                cols[i].append(v)
    out = {}
    for h, c in zip(head, cols):
        try:
            out[h] = np.array([float(x) if x != "" else NAN for x in c], float)
        except ValueError:
            out[h] = np.array(c, dtype=object)
    return out


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def discover_links(tk: Dict[str, np.ndarray], S: LogSchema) -> List[str]:
    return sorted({k[: -len(S.link_discover_suffix)] for k in tk if k.endswith(S.link_discover_suffix)})


# =============================================================================
# 4. SIGNAL HELPERS
# =============================================================================
def runs_of(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return start/end (exclusive) indices of True-runs."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def merge_runs(s: np.ndarray, e: np.ndarray, gap: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(s) < 2 or gap <= 0:
        return s, e
    ms, me = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a - me[-1] <= gap:
            me[-1] = b
        else:
            ms.append(a)
            me.append(b)
    return np.array(ms), np.array(me)


def rolling_ratio(t: np.ndarray, num: np.ndarray, den: np.ndarray, W: float) -> np.ndarray:
    cn = np.concatenate(([0.0], np.cumsum(num.astype(float))))
    cd = np.concatenate(([0.0], np.cumsum(den.astype(float))))
    j = np.searchsorted(t, t - W, side="right")
    i1 = np.arange(len(t)) + 1
    nn, dd = cn[i1] - cn[j], cd[i1] - cd[j]
    return np.where(dd > 0, nn / np.maximum(dd, 1e-12), 1.0)


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else NAN


LAT_EDGES = np.geomspace(1.0, 2000.0, 401)       # 0.9 % relative resolution for pooled percentiles


# =============================================================================
# 5. PER-RUN EVALUATION
# =============================================================================
def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auc_rank(p: np.ndarray, y: np.ndarray) -> float:
    n1 = float((y == 1).sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return NAN
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def avg_precision(p: np.ndarray, y: np.ndarray) -> float:
    if (y == 1).sum() == 0:
        return NAN
    o = np.argsort(-p, kind="mergesort")
    ys = y[o]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(prec[ys == 1].mean())


def _ep_arrays(mask: np.ndarray, t: np.ndarray, minn: int):
    s, e = runs_of(mask)
    k = (e - s) >= minn
    s, e = s[k], e[k]
    kept = np.zeros(len(mask), bool)
    for a, b in zip(s, e):
        kept[a:b] = True
    return s, e, kept


def evaluate_run(tk: Dict[str, np.ndarray], events: list, decisions: list, scn: dict, cfg: EvalConfig, row: dict) -> dict:
    """Compute all scalar metrics + compact arrays (histograms, grids) for ONE run."""
    S, D, H = cfg.schema, cfg.definitions, cfg.handover
    sc: Dict[str, float] = {}
    ar: Dict[str, Any] = {}
    out = {"scalars": sc, "arrays": ar}
    t = tk[S.t]
    n = len(t)
    if n < 3:
        sc["excluded"] = 1
        return out
    dt = float(np.median(np.diff(t)))
    mode = tk.get(S.mode)
    A = np.isin(mode, D.airborne_modes) if mode is not None else np.ones(n, bool)
    flight_s = float(A.sum() * dt)
    sc["flight_s"] = flight_s
    sc["excluded"] = 0 if flight_s >= D.min_flight_s else 1
    if sc["excluded"]:
        return out
    fmin = flight_s / 60.0

    # ---------------- control stream & reliability ----------------
    sent = tk[S.ctrl_sent] > 0.5
    lat = tk[S.ctrl_lat_ms]
    ontime = sent & np.isfinite(lat) & (lat <= D.deadline_ms)
    ratio = rolling_ratio(t, ontime, sent, D.ctrl_window_s)
    last_ok = np.maximum.accumulate(np.where(ontime, t, t[0]))
    gap_ms = (t - last_ok) * 1000.0
    vok = tk[S.vid_ok] > 0.5
    if D.vid_min_quality > 0 and S.vid_q in tk:
        vok = vok & (tk[S.vid_q] >= D.vid_min_quality)
    if D.blind_mode == "logged" and S.ctrl_rel in tk and S.vid_rel in tk:
        ctrl_rel, vid_use = tk[S.ctrl_rel] > 0.5, tk[S.vid_rel] > 0.5
    else:
        ctrl_rel = (gap_ms <= D.ctrl_gap_max_ms) & (ratio >= D.ctrl_ok_ratio)
        vid_use = rolling_ratio(t, vok, np.ones(n), D.vid_window_s) >= D.vid_ok_ratio
    blind = A & ~ctrl_rel & ~vid_use

    # ---------------- blind time ----------------
    s, e = merge_runs(*runs_of(blind), int(round(D.merge_gap_s / dt)))
    dur = (e - s) * dt
    sc["blind_time_s"] = float(blind.sum() * dt)
    sc["blind_fraction"] = sc["blind_time_s"] / flight_s
    sc["blind_per_min"] = sc["blind_time_s"] / fmin
    sc["blind_any"] = float(len(s) > 0)
    sc["n_blind_episodes"] = float(len(s))
    sc["max_blind_episode_s"] = float(dur.max()) if len(dur) else 0.0
    sc["mean_blind_episode_s"] = float(dur.mean()) if len(dur) else 0.0
    sc["first_blind_s"] = float(t[s[0]] - t[A][0]) if len(s) else NAN
    for c in D.critical_blind_s:
        sc[f"n_blind_ge_{c:g}s"] = float((dur >= c).sum())
    sc["ctrl_unreliable_s"] = float((A & ~ctrl_rel).sum() * dt)
    sc["video_unusable_s"] = float((A & ~vid_use).sum() * dt)
    sc["ctrl_only_loss_s"] = float((A & ~ctrl_rel & vid_use).sum() * dt)
    ar["blind_durations"] = dur

    # ---------------- control latency ----------------
    sa = sent & A
    ns = max(int(sa.sum()), 1)
    dl = np.isfinite(lat) & sa
    Lv = lat[dl]
    sc["ctrl_sent"] = float(sa.sum())
    sc["ctrl_lost_rate"] = 1.0 - float(dl.sum()) / ns
    sc["ctrl_ontime_rate"] = float((ontime & A).sum()) / ns
    sc["ctrl_late_rate"] = float((dl & ~ontime).sum()) / ns
    for lim in D.latency_limits_ms:
        sc[f"ctrl_within_{lim:g}ms_rate"] = float((dl & (lat <= lim)).sum()) / ns
    for name, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99), ("p999", 99.9)):
        sc[f"ctrl_lat_{name}_ms"] = pct(Lv, q)
    sc["ctrl_lat_mean_ms"] = float(Lv.mean()) if len(Lv) else NAN
    sc["ctrl_lat_max_ms"] = float(Lv.max()) if len(Lv) else NAN
    sc["ctrl_jitter_ms"] = float(Lv.std()) if len(Lv) else NAN
    eff = np.where(dl, lat, D.lost_penalty_ms)[sa]
    sc["ctrl_lat_eff_p99_ms"] = pct(eff, 99)
    sc["ctrl_max_gap_ms"] = float(gap_ms[A].max())
    ar["lat_hist"] = np.histogram(np.clip(Lv, 1.0, 1999.9), LAT_EDGES)[0] if len(Lv) else np.zeros(len(LAT_EDGES) - 1, int)
    ar["lat_lost"] = int(sa.sum() - dl.sum())

    # ---------------- video ----------------
    vq = tk[S.vid_q] if S.vid_q in tk else np.where(vok, 1.0, 0.0)
    sc["video_quality_mean"] = float(vq[A].mean())
    sc["video_usable_fraction"] = float(vid_use[A].mean())
    sc["video_decodable_rate"] = float(vok[A].mean())
    fs, fe = runs_of(~vok & A)
    fd = (fe - fs) * dt
    fz = fd >= D.freeze_s
    sc["video_freeze_s"] = float(fd[fz].sum())
    sc["video_freeze_events"] = float(fz.sum())
    sc["video_freeze_max_s"] = float(fd.max()) if len(fd) and fz.any() else 0.0
    ar["freeze_durations"] = fd[fz]
    vbps = tk.get(S.video_bps)
    if vbps is not None:
        sc["video_mean_mbps"] = float((vok * vbps)[A].mean() / 1e6)

    # ---------------- handovers (switches) and their cost ----------------
    links = discover_links(tk, S)
    prim = np.array([str(c).split("+")[0] for c in tk[S.ctrl_links]]) if S.ctrl_links in tk else np.array([""] * n)
    chg = [int(i) for i in (np.flatnonzero(prim[1:] != prim[:-1]) + 1) if A[i]]
    Wn = max(1, int(round(H.window_s / dt)))
    sw, last_t, last_frm = [], -1e18, None
    lost_pkt = sent & ~ontime
    for i in chg:
        frm, to = prim[i - 1], prim[i]
        st = tk.get(f"{to}{S.link_state}")
        kind = "soft" if (st is not None and str(st[i - 1]) == "warm") else "hard"
        ttu = 0.0
        if kind == "hard" and st is not None:
            w = np.flatnonzero(st[i:i + 3 * Wn] == "warm")
            ttu = float(t[i + w[0]] - t[i]) if len(w) else 3 * Wn * dt
        pp = (to == last_frm and (t[i] - last_t) < H.pingpong_s)
        dis = float((~ctrl_rel[i:i + Wn]).sum() * dt)
        lp = float(lost_pkt[i:i + Wn].sum())
        bw = float(blind[i:i + Wn].sum() * dt)
        cost = (H.hard if kind == "hard" else H.soft) + (H.pingpong if pp else 0.0) + H.disruption_per_s * dis \
            + H.lost_packet * lp + H.blind_per_s * bw
        sw.append(dict(t=float(t[i]), frm=frm, to=to, kind=kind, proactive=bool(ctrl_rel[i - 1]), pingpong=pp,
                       disruption_s=dis, lost_pkts=lp, blind_s=bw, time_to_usable_s=ttu, cost=cost))
        last_t, last_frm = float(t[i]), frm
    ar["switches"] = sw
    nh = sum(1 for x in sw if x["kind"] == "hard")
    sc["n_hard_switch"], sc["n_soft_switch"], sc["n_switch"] = float(nh), float(len(sw) - nh), float(len(sw))
    sc["n_pingpong"] = float(sum(x["pingpong"] for x in sw))
    sc["switch_per_min"] = len(sw) / fmin
    sc["handover_cost"] = float(sum(x["cost"] for x in sw))
    sc["handover_disruption_s"] = float(sum(x["disruption_s"] for x in sw))
    sc["proactive_switch_frac"] = float(np.mean([x["proactive"] for x in sw])) if sw else NAN
    hh = [x["time_to_usable_s"] for x in sw if x["kind"] == "hard"]
    sc["hard_time_to_usable_s"] = float(np.mean(hh)) if hh else NAN
    sc["dup_time_s"] = float(sum(1 for c in tk[S.ctrl_links][A] if "+" in str(c)) * dt) if S.ctrl_links in tk else NAN
    ev_ct: Dict[str, int] = {}
    for ev in events:
        ev_ct[ev["kind"]] = ev_ct.get(ev["kind"], 0) + 1
    sc["n_rlf"] = float(ev_ct.get("rlf", row.get("n_rlf", NAN) if not events else 0))
    sc["n_failsafe"] = float(ev_ct.get("failsafe_triggered", row.get("n_failsafe", NAN) if not events else 0))
    sc["n_cell_handover"] = float(ev_ct.get("handover", 0)) if events else NAN
    sc["n_faults"] = float(ev_ct.get("fault_start", 0)) if events else NAN

    # ---------------- energy / spectrum ----------------
    pc = tk[S.p_comm_w]
    e_wh = float((pc * dt)[A].sum() / 3600.0)
    sc["energy_comm_wh"] = e_wh
    sc["comm_power_mean_w"] = e_wh * 3600.0 / flight_s
    sc["comm_power_p99_w"] = pct(pc[A], 99)
    pbud = cfg.budget.power_w or (scn.get("budget") or {}).get("comm_power_w") or 8.0
    sbud = cfg.budget.spectrum_hz or (scn.get("budget") or {}).get("spectrum_hz") or 40e6
    batt = cfg.budget.battery_wh or (scn.get("drone") or {}).get("battery_wh") or 90.0
    sc["over_power_budget_s"] = float(((pc > pbud) & A).sum() * dt)
    bw = {l["name"]: l.get("bw_hz", 20e6) for l in (scn.get("links") or [])}
    spec = np.zeros(n)
    for L in links:
        pw = tk.get(f"{L}{S.link_power}")
        st = tk.get(f"{L}{S.link_state}")
        if pw is not None:
            sc[f"energy_{L}_wh"] = float((pw * dt)[A].sum() / 3600.0)
        if st is not None:
            spec += bw.get(L, 20e6) * (st != "cold")
            if pw is not None:
                for sname in ("cold", "warming", "warm"):
                    key = f"energy_state_{sname}_wh"
                    sc[key] = sc.get(key, 0.0) + float((pw * dt)[A & (st == sname)].sum() / 3600.0)
    if links and any(f"{L}{S.link_state}" in tk for L in links):
        sc["spectrum_mean_mhz"] = float(spec[A].mean() / 1e6)
        sc["over_spectrum_budget_s"] = float(((spec > sbud) & A).sum() * dt)
    if vbps is not None:
        mbit = float((vok * vbps * dt)[A].sum() / 1e6)
        sc["energy_per_video_mbit_j"] = e_wh * 3600.0 / mbit if mbit > 0 else NAN
    sc["energy_comm_frac_battery"] = e_wh / batt
    U = cfg.utility
    sc["utility_cost"] = (U.w_blind * sc["blind_fraction"] + U.w_late * (1 - sc["ctrl_ontime_rate"])
                          + U.w_quality * (1 - sc["video_quality_mean"]) + U.w_energy * sc["comm_power_mean_w"] / pbud)

    # ---------------- mission ----------------
    outcome = row.get("outcome")
    if outcome is None:
        outcome = next((e.get("outcome") for e in events if e["kind"] == "end"), "unknown")
    sc["mission_completed"] = float(outcome == "completed")
    ar["outcome"] = outcome

    # ---------------- spatial / temporal heat arrays ----------------
    if cfg.spatial.enabled:
        if S.x in tk and S.y in tk:
            cs = cfg.spatial.cell_m
            ix = np.clip((tk[S.x][A] // cs).astype(np.int64), 0, 65535)
            iy = np.clip((tk[S.y][A] // cs).astype(np.int64), 0, 65535)
            key = ix * 65536 + iy
            uk, inv = np.unique(key, return_inverse=True)
            ar["sp_keys"] = uk
            ar["sp_dwell"] = np.bincount(inv, minlength=len(uk)) * dt
            ar["sp_blind"] = np.bincount(inv, weights=blind[A].astype(float), minlength=len(uk)) * dt
        nb = cfg.spatial.n_phase_bins
        cum = np.cumsum(A) - 1
        pb = np.clip((cum[A] * nb // max(int(A.sum()), 1)).astype(int), 0, nb - 1)
        ar["ph_blind"] = np.bincount(pb, weights=blind[A].astype(float), minlength=nb) * dt
        ar["ph_dwell"] = np.bincount(pb, minlength=nb) * dt
        nt = int(math.ceil(t[-1] / cfg.spatial.time_bin_s)) + 1
        tb = np.clip((t[A] // cfg.spatial.time_bin_s).astype(int), 0, nt - 1)
        ar["tb_blind"] = np.bincount(tb, weights=blind[A].astype(float), minlength=nt) * dt
        ar["tb_dwell"] = np.bincount(tb, minlength=nt) * dt

    # ---------------- outage episodes / prediction quality ----------------
    if cfg.prediction.enabled and links:
        _eval_prediction(tk, links, prim, decisions, A, t, dt, blind, chg, sc, ar, cfg, flight_s)
    return out


def _eval_prediction(tk, links, prim, decisions, A, t, dt, blind, chg, sc, ar, cfg, flight_s):
    P, S = cfg.prediction, cfg.schema
    n = len(t)
    H = P.horizon_s
    minn = max(1, int(math.ceil(P.outage_min_ms / 1000.0 / dt)))
    ep: Dict[str, tuple] = {}
    for L in links:
        col = tk.get(f"{L}{S.link_outage}")
        if col is not None:
            s, e, kept = _ep_arrays((col > 0.5) & A, t, minn)
            ep[L] = (t[s], kept)
    if not ep:
        return
    onsets = []                                   # primary-link outage episodes
    for L, (on, _) in ep.items():
        for o in on:
            if prim[min(int(np.searchsorted(t, o)), n - 1)] == L:
                onsets.append((float(o), L))
    onsets.sort()
    Wc = max(1, int(round(P.conv_window_s / dt)))
    conv = sum(1 for o, _ in onsets if blind[int(np.searchsorted(t, o)):int(np.searchsorted(t, o)) + Wc].any())
    sc["n_primary_outages"] = float(len(onsets))
    sc["n_outage_blind"] = float(conv)
    sc["outage_to_blind_rate"] = conv / len(onsets) if onsets else NAN
    recs = [(d["t"], d["pout"]) for d in decisions if d.get("pout")]
    if not recs:
        sc["pred_available"] = 0.0
        return
    sc["pred_available"] = 1.0
    ts, ps, ys, ln = [], [], [], []
    for tt, po in recs:
        idx = int(np.searchsorted(t, tt))
        if idx >= n or not A[idx]:
            continue
        for L in ([prim[idx]] if P.primary_only else list(po)):
            if L not in po or L not in ep:
                continue
            on, kept = ep[L]
            k = int(np.searchsorted(on, tt, side="right"))
            nxt = on[k] if k < len(on) else np.inf
            if P.label_mode == "onset":
                if kept[idx]:
                    continue
                y = float(nxt <= tt + H)
            else:
                y = float(kept[idx] or nxt <= tt + H)
            ts.append(tt); ps.append(float(po[L])); ys.append(y); ln.append(L)
    if not ts:
        return
    ts, ps, ys, ln = np.array(ts), np.array(ps), np.array(ys), np.array(ln)
    sc["pred_n"] = float(len(ts))
    sc["pred_base_rate"] = float(ys.mean())
    sc["pred_auc"] = auc_rank(ps, ys)
    sc["pred_ap"] = avg_precision(ps, ys)
    sc["pred_brier"] = float(np.mean((ps - ys) ** 2))
    cb = np.clip((ps * P.calib_bins).astype(int), 0, P.calib_bins - 1)
    ece = 0.0
    for b in range(P.calib_bins):
        m = cb == b
        if m.any():
            ece += abs(ys[m].mean() - ps[m].mean()) * m.mean()
    sc["pred_ece"] = float(ece)
    ar["pred_hist_pos"] = np.histogram(ps[ys == 1], bins=P.hist_bins, range=(0, 1))[0]
    ar["pred_hist_neg"] = np.histogram(ps[ys == 0], bins=P.hist_bins, range=(0, 1))[0]
    thr = sorted(set(P.thresholds) | {P.threshold})
    act_t = [d["t"] for d in decisions if (len(d.get("ctrl", [])) > 1 or d.get("warm") or d.get("tag") in ("soft_switch", "hard_switch"))]
    act_t += [float(t[i]) for i in chg]
    act_t = np.array(sorted(act_t))
    K = len(thr)
    a_ep, a_det, a_lead, a_alerts, a_false = (np.zeros(K) for _ in range(5))
    for ki, th in enumerate(thr):
        sel = ps >= th
        a_t, a_L = ts[sel], ln[sel]
        det, leads = 0, []
        for o, L in onsets:
            m = (a_L == L) & (a_t >= o - H) & (a_t < o)
            if m.any():
                det += 1
                leads.append(o - a_t[m].min())
        starts, last = [], {}
        for tt, L in zip(a_t, a_L):
            if L not in last or tt - last[L] > P.alert_merge_s:
                starts.append((tt, L))
            last[L] = tt
        false = sum(1 for tt, L in starts if not any(L == oL and tt < o <= tt + H for o, oL in onsets))
        a_ep[ki], a_det[ki], a_lead[ki] = len(onsets), det, float(np.sum(leads))
        a_alerts[ki], a_false[ki] = len(starts), false
        if abs(th - P.threshold) < 1e-12:
            sc["pred_ep_recall"] = det / len(onsets) if onsets else NAN
            sc["pred_lead_mean_s"] = float(np.mean(leads)) if leads else NAN
            sc["pred_false_alarms_per_min"] = false / (flight_s / 60.0)
            sc["pred_precision"] = (len(starts) - false) / len(starts) if starts else NAN
    ar["thr"] = np.array(thr)
    ar["thr_ep"], ar["thr_det"], ar["thr_lead"], ar["thr_alerts"], ar["thr_false"] = a_ep, a_det, a_lead, a_alerts, a_false
    if onsets and len(act_t):
        ok = sum(1 for o, _ in onsets if ((act_t >= o - H) & (act_t < o)).any())
        sc["pred_actioned_rate"] = ok / len(onsets)


# =============================================================================
# 6. STATISTICS (numpy only)
# =============================================================================
def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def gammaincc(a: float, x: float) -> float:
    if x <= 0:
        return 1.0
    if x < a + 1:
        ap, s = a, 1.0 / a
        dl = s
        for _ in range(1000):
            ap += 1
            dl *= x / ap
            s += dl
            if abs(dl) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > 1e-300 else 1e-300
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    return gammaincc(df / 2.0, x / 2.0)


def wilcoxon_signed_rank(d, alternative: str = "two-sided", exact_max: int = 40) -> dict:
    """alternative: 'two-sided' | 'less' (diffs tend negative) | 'greater'. Exact (DP) if no ties and n<=exact_max."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return dict(W=0.0, p=1.0, n=0, method="empty")
    ad = np.abs(d)
    ranks = rankdata(ad)
    Wp = float(ranks[d > 0].sum())
    ties = len(np.unique(ad)) < n
    if n <= exact_max and not ties:
        total = n * (n + 1) // 2
        cnt = np.zeros(total + 1)
        cnt[0] = 1.0
        for r in range(1, n + 1):
            new = cnt.copy()
            new[r:] += cnt[:total + 1 - r]
            cnt = new
        cdf = np.cumsum(cnt / cnt.sum())
        w = int(round(Wp))
        p_le = float(cdf[w])
        p_ge = float(1.0 - (cdf[w - 1] if w > 0 else 0.0))
        method = "exact"
    else:
        mu = n * (n + 1) / 4.0
        _, tc = np.unique(ad, return_counts=True)
        var = n * (n + 1) * (2 * n + 1) / 24.0 - float(np.sum(tc ** 3 - tc)) / 48.0
        sd = math.sqrt(max(var, 1e-12))
        p_le = norm_cdf((Wp - mu + 0.5) / sd)
        p_ge = 1.0 - norm_cdf((Wp - mu - 0.5) / sd)
        method = "normal"
    p = min(1.0, 2 * min(p_le, p_ge)) if alternative == "two-sided" else (p_le if alternative == "less" else p_ge)
    return dict(W=Wp, p=float(p), n=n, method=method)


def sign_test(d, alternative: str = "two-sided") -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return dict(p=1.0, n=0, pos=0, neg=0)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())

    def cdf(k):
        return sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    p = min(1.0, 2 * cdf(min(pos, neg))) if alternative == "two-sided" else (cdf(neg) if alternative == "less" else cdf(pos))
    return dict(p=float(min(1.0, p)), n=n, pos=pos, neg=neg)


def perm_test_paired(d, B: int, rng, alternative: str = "two-sided") -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return dict(p=1.0, n=0)
    obs = d.mean()
    if n <= 14:
        signs = np.array(list(itertools.product((-1, 1), repeat=n)), float)
        m = (signs * d).mean(axis=1)
        exact = True
    else:
        m = np.empty(B)
        chunk = max(1, int(4e6 // n))
        i = 0
        while i < B:
            k = min(chunk, B - i)
            m[i:i + k] = (rng.choice(np.array([-1.0, 1.0]), size=(k, n)) * d).mean(axis=1)
            i += k
        exact = False
    add = 0 if exact else 1
    den = len(m) + add
    if alternative == "two-sided":
        p = (np.sum(np.abs(m) >= abs(obs) - 1e-12) + add) / den
    elif alternative == "less":
        p = (np.sum(m <= obs + 1e-12) + add) / den
    else:
        p = (np.sum(m >= obs - 1e-12) + add) / den
    return dict(p=float(min(1.0, p)), n=n, exact=exact)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())      # a=0,b=1 (a better if lower-is-better) / a=1,b=0
    n = n01 + n10
    if n == 0:
        return dict(p=1.0, a_only=n10, b_only=n01)
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return dict(p=float(p), a_only=n10, b_only=n01)


def friedman_test(M: np.ndarray) -> dict:
    """M: n blocks x k treatments. Returns chi2, df, p (tie-corrected)."""
    M = np.asarray(M, float)
    n, k = M.shape
    R = np.apply_along_axis(rankdata, 1, M)
    Rj = R.sum(axis=0)
    chi = 12.0 / (n * k * (k + 1)) * np.sum(Rj ** 2) - 3.0 * n * (k + 1)
    corr = 1.0
    for row in M:
        _, tc = np.unique(row, return_counts=True)
        corr -= np.sum(tc ** 3 - tc) / (n * k * (k * k - 1))
    chi = chi / corr if corr > 1e-12 else chi
    return dict(chi2=float(chi), df=k - 1, p=float(chi2_sf(chi, k - 1)), mean_ranks=(Rj / n).tolist(), n=n)


def hodges_lehmann(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return NAN
    if len(d) > 1500:
        d = np.random.default_rng(0).choice(d, 1500, replace=False)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > 2500:
        r = np.random.default_rng(0)
        a, b = r.choice(a, 2500, replace=False), r.choice(b, 2500, replace=False)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def adjust_p(p: Sequence[float], method: str) -> List[float]:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0 or method == "none":
        return p.tolist()
    o = np.argsort(p)
    ps = p[o]
    if method == "holm":
        adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * ps))
    elif method == "bh":
        adj = np.minimum.accumulate(np.minimum(1.0, ps * m / (np.arange(m) + 1))[::-1])[::-1]
    else:
        raise ValueError("correction must be holm|bh|none")
    out = np.empty(m)
    out[o] = adj
    return out.tolist()


def boot_stat(x: np.ndarray, stat, B: int, rng) -> np.ndarray:
    n = len(x)
    out = np.empty(B)
    chunk = max(1, int(3e6 // max(n, 1)))
    i = 0
    while i < B:
        k = min(chunk, B - i)
        idx = rng.integers(0, n, (k, n))
        out[i:i + k] = stat(x[idx])
        i += k
    return out


def ci_mean(x, B, rng, alpha=0.05) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (NAN, NAN)
    m = boot_stat(x, lambda z: z.mean(axis=1), B, rng)
    return float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def ci_rel_reduction(a, b, B, rng, alpha=0.05) -> Tuple[float, float, float]:
    """Paired bootstrap of 1 - mean(a)/mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2 or abs(b.mean()) < 1e-12:
        return (NAN, NAN, NAN)
    ab = np.stack([a, b], axis=1)
    r = boot_stat(ab, lambda z: 1.0 - z[:, :, 0].mean(axis=1) / np.where(np.abs(z[:, :, 1].mean(axis=1)) < 1e-12, np.nan, z[:, :, 1].mean(axis=1)), B, rng)
    r = r[np.isfinite(r)]
    point = 1.0 - a.mean() / b.mean()
    if len(r) < 10:
        return (point, NAN, NAN)
    return point, float(np.percentile(r, 100 * alpha / 2)), float(np.percentile(r, 100 * (1 - alpha / 2)))


def paired_compare(a, b, metric: str, cfg: EvalConfig, rng) -> dict:
    """Proposed `a` vs baseline `b`, same trials. 'improvement' is direction-aware (positive = a better)."""
    st = cfg.stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    out = dict(metric=metric, n=n)
    if n < 2:
        return out
    d = a - b
    low = better_is_lower(metric)
    imp = -d if low else d
    alt_better = "less" if low else "greater"
    lo, hi = ci_mean(d, st.bootstrap_B, rng, st.alpha)
    sd = float(d.std(ddof=1))
    out.update(mean_a=float(a.mean()), mean_b=float(b.mean()), median_a=float(np.median(a)), median_b=float(np.median(b)),
               diff=float(d.mean()), diff_lo=lo, diff_hi=hi, diff_median=float(np.median(d)),
               improvement=float(imp.mean()), wins=int((imp > 0).sum()), losses=int((imp < 0).sum()), ties=int((imp == 0).sum()))
    if abs(b.mean()) > 1e-12:
        pt, rlo, rhi = ci_rel_reduction(a, b, st.bootstrap_B, rng, st.alpha)
        out.update(rel_change=float(d.mean() / b.mean()), rel_reduction=pt, rel_reduction_lo=rlo, rel_reduction_hi=rhi)
    w2, wb = wilcoxon_signed_rank(d, "two-sided"), wilcoxon_signed_rank(d, alt_better)
    out.update(p_wilcoxon=w2["p"], p_wilcoxon_better=wb["p"], wilcoxon_method=w2["method"])
    out["p_sign"] = sign_test(d)["p"]
    out["p_perm"] = perm_test_paired(d, st.permutation_B, rng)["p"]
    out["p"] = out[{"wilcoxon": "p_wilcoxon", "permutation": "p_perm", "sign": "p_sign"}[st.primary_test]]
    out["cohen_dz"] = float(d.mean() / sd) if sd > 0 else NAN
    out["cliffs_delta"] = cliffs_delta(a, b)
    out["p_superiority"] = float(((imp > 0).sum() + 0.5 * (imp == 0).sum()) / n)
    out["hodges_lehmann"] = hodges_lehmann(d)
    za, zb = norm_ppf(1 - st.alpha / 2), norm_ppf(st.power)
    dz = abs(out["cohen_dz"]) if sd > 0 else NAN
    out["n_required_for_power"] = float(math.ceil(((za + zb) / dz) ** 2)) if dz and dz > 1e-9 else NAN
    out["mde_dz"] = float((za + zb) / math.sqrt(n))
    out["mde_units"] = float(out["mde_dz"] * sd)
    if metric in BINARY_METRICS or set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}:
        out["mcnemar_p"] = mcnemar_exact(a, b)["p"]
    return out


def noninferiority(a, b, metric: str, margin: float, B: int, rng, alpha: float = 0.05) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    loss = (a - b) if better_is_lower(metric) else (b - a)     # positive = proposed worse
    if len(loss) < 2:
        return dict(n=len(loss))
    ci_lo, ci_hi = ci_mean(loss, B, rng, 2 * alpha)             # one-sided alpha/2 each side of a 2*alpha interval
    return dict(n=len(loss), loss=float(loss.mean()), lo=ci_lo, hi=ci_hi,
                verdict="PASS" if ci_hi <= margin else ("FAIL" if ci_lo > margin else "INCONCLUSIVE"))


# =============================================================================
# 7. RESULT SET, SELECTION, GOALS, PARETO
# =============================================================================
def match(label: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(label, p) for p in patterns)


def parse_label(label: str) -> Tuple[str, Dict[str, float]]:
    m = re.match(r"^(.*?)\[(.*)\]$", label)
    if not m:
        return label, {}
    params = {}
    for kv in m.group(2).split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                params[k.strip()] = float(v)
            except ValueError:
                params[k.strip()] = v
    return m.group(1), params


class ResultSet:
    def __init__(self, runs: List[dict]):
        self.runs = [r for r in runs if not r.get("error")]
        self.excluded = [r for r in self.runs if r.get("scalars", {}).get("excluded", 0)]
        self.by: Dict[str, Dict[tuple, dict]] = {}
        for r in self.runs:
            if r.get("scalars", {}).get("excluded", 0):
                continue
            self.by.setdefault(r["label"], {})[(r.get("scenario", ""), r["trial"])] = r
        self.labels = sorted(self.by)

    def keys(self, label=None) -> List[tuple]:
        if label:
            return sorted(self.by.get(label, {}))
        return sorted({k for d in self.by.values() for k in d})

    def common_keys(self, labels: Sequence[str]) -> List[tuple]:
        ks = [set(self.by[l]) for l in labels if l in self.by]
        return sorted(set.intersection(*ks)) if ks else []

    def vec(self, label: str, metric: str, keys: Optional[Sequence[tuple]] = None) -> np.ndarray:
        d = self.by.get(label, {})
        ks = keys if keys is not None else sorted(d)
        return np.array([d[k]["scalars"].get(metric, NAN) if k in d else NAN for k in ks], float)

    def mean(self, label: str, metric: str, keys=None) -> float:
        v = self.vec(label, metric, keys)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else NAN

    def metrics_available(self) -> List[str]:
        s = set()
        for r in self.runs:
            s.update(r.get("scalars", {}))
        return sorted(s)

    def pool(self, label: str, keys=None) -> dict:
        """Pooled arrays across runs (latency histogram, blind durations, heat grids, prediction histograms)."""
        P: Dict[str, Any] = {"lat_hist": 0, "lat_lost": 0, "dur": [], "frz": [], "sp": {}, "ph_b": 0, "ph_d": 0,
                             "tb_b": [], "tb_d": [], "ph_runs": [], "ph_d_runs": [], "ph_hp": 0, "ph_hn": 0,
                             "thr": None, "sw": [], "n": 0}
        for k in (keys if keys is not None else self.keys(label)):
            r = self.by.get(label, {}).get(k)
            if not r or "arrays" not in r or not r["arrays"]:
                continue
            a = r["arrays"]
            P["n"] += 1
            if "lat_hist" in a:
                P["lat_hist"] = P["lat_hist"] + np.asarray(a["lat_hist"])
                P["lat_lost"] += a["lat_lost"]
            P["dur"].append(np.asarray(a.get("blind_durations", [])))
            P["frz"].append(np.asarray(a.get("freeze_durations", [])))
            P["sw"] += a.get("switches", [])
            if "sp_keys" in a:
                for kk, dw, bl in zip(a["sp_keys"], a["sp_dwell"], a["sp_blind"]):
                    c = P["sp"].setdefault(int(kk), [0.0, 0.0])
                    c[0] += dw
                    c[1] += bl
            if "ph_blind" in a:
                P["ph_b"] = P["ph_b"] + np.asarray(a["ph_blind"])
                P["ph_d"] = P["ph_d"] + np.asarray(a["ph_dwell"])
            if "tb_blind" in a:
                P["tb_b"].append(np.asarray(a["tb_blind"]))
                P["tb_d"].append(np.asarray(a["tb_dwell"]))
            if "pred_hist_pos" in a:
                P["ph_hp"] = P["ph_hp"] + np.asarray(a["pred_hist_pos"])
                P["ph_hn"] = P["ph_hn"] + np.asarray(a["pred_hist_neg"])
            if "thr" in a:
                t = {k2: np.asarray(a[k2]) for k2 in ("thr_ep", "thr_det", "thr_lead", "thr_alerts", "thr_false")}
                if P["thr"] is None:
                    P["thr"] = {"thr": np.asarray(a["thr"]), **t}
                else:
                    for k2 in t:
                        P["thr"][k2] = P["thr"][k2] + t[k2]
        P["dur"] = np.concatenate(P["dur"]) if P["dur"] else np.array([])
        P["frz"] = np.concatenate(P["frz"]) if P["frz"] else np.array([])
        return P


def lat_hist_percentile(hist: np.ndarray, lost: int, q: float, penalty_ms: float = 1000.0) -> float:
    """Pooled percentile from histogram; lost packets count as +inf (returns inf if beyond)."""
    tot = float(np.sum(hist) + lost)
    if tot <= 0:
        return NAN
    target = q / 100.0 * tot
    c = np.cumsum(hist)
    if target > c[-1]:
        return math.inf
    i = int(np.searchsorted(c, target))
    return float(np.sqrt(LAT_EDGES[i] * LAT_EDGES[i + 1]))


def check_op(v: float, op: str, thr: float) -> bool:
    return {">=": v >= thr, "<=": v <= thr, ">": v > thr, "<": v < thr}[op]


def summarize_metric(x: np.ndarray, B: int, rng, alpha: float) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    lo, hi = ci_mean(x, B, rng, alpha)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                p95=float(np.percentile(x, 95)), max=float(x.max()), ci_lo=lo, ci_hi=hi, frac_nonzero=float((x != 0).mean()))


def pareto_mask(P: np.ndarray, senses: Sequence[int]) -> np.ndarray:
    """senses: +1 minimise, -1 maximise. True = non-dominated."""
    X = np.asarray(P, float) * np.asarray(senses, float)
    n = len(X)
    keep = np.ones(n, bool)
    for i in range(n):
        if not np.all(np.isfinite(X[i])):
            keep[i] = False
            continue
        dom = np.all(X <= X[i], axis=1) & np.any(X < X[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep


def select_variants(rs: ResultSet, cfg: EvalConfig, variants: List[str], keys: List[tuple]) -> dict:
    sel = cfg.selection
    need = {sel.objective} | {c["metric"] for c in sel.constraints}
    rows = []
    for L in variants:
        means = {m: rs.mean(L, m, keys) for m in need}
        feas = all(np.isfinite(means[c["metric"]]) and check_op(means[c["metric"]], c["op"], c["value"]) for c in sel.constraints)
        rows.append(dict(label=L, feasible=feas, objective=means[sel.objective], means=means))
    ok = [r for r in rows if r["feasible"] and np.isfinite(r["objective"])]
    pool = ok or [r for r in rows if np.isfinite(r["objective"])]
    best = min(pool, key=lambda r: r["objective"]) if pool else None
    return dict(rows=rows, best=best["label"] if best else None, any_feasible=bool(ok), objective=sel.objective)


def eval_goals(rs: ResultSet, cfg: EvalConfig, proposed: List[str], baselines: List[str], keys: List[tuple], rng) -> List[dict]:
    st = cfg.stats
    out = []
    for g in cfg.goals:
        pols = [p for p in (rs.labels if g.policy else proposed) if (not g.policy or fnmatch.fnmatch(p, g.policy))]
        for p in pols:
            if g.type in ("absolute_min", "absolute_max"):
                v = rs.vec(p, g.metric, keys)
                v = v[np.isfinite(v)]
                if len(v) < 2:
                    out.append(dict(goal=g.name, policy=p, vs="", metric=g.metric, type=g.type, target=g.target, verdict="NO DATA"))
                    continue
                lo, hi = ci_mean(v, st.bootstrap_B, rng, st.alpha)
                est = float(v.mean())
                if g.type == "absolute_min":
                    vd = "PASS" if lo >= g.target else ("FAIL" if hi < g.target else "INCONCLUSIVE")
                else:
                    vd = "PASS" if hi <= g.target else ("FAIL" if lo > g.target else "INCONCLUSIVE")
                out.append(dict(goal=g.name, policy=p, vs="", metric=g.metric, type=g.type, target=g.target, estimate=est,
                                ci_lo=lo, ci_hi=hi, verdict=vd, n=len(v)))
                continue
            for b in [x for x in baselines if g.vs and fnmatch.fnmatch(x, g.vs) and x != p]:
                ks = rs.common_keys([p, b]) if not keys else [k for k in keys if k in rs.by.get(p, {}) and k in rs.by.get(b, {})]
                a, bb = rs.vec(p, g.metric, ks), rs.vec(b, g.metric, ks)
                m = np.isfinite(a) & np.isfinite(bb)
                a, bb = a[m], bb[m]
                row = dict(goal=g.name, policy=p, vs=b, metric=g.metric, type=g.type, target=g.target if g.type.startswith("rel") else g.margin, n=len(a))
                if len(a) < 2:
                    row["verdict"] = "NO DATA"
                elif g.type in ("relative_reduction", "relative_increase"):
                    pt, lo, hi = ci_rel_reduction(a, bb, st.bootstrap_B, rng, st.alpha)
                    if g.type == "relative_increase":
                        pt, lo, hi = -pt, -hi, -lo
                    row.update(estimate=pt, ci_lo=lo, ci_hi=hi, verdict="PASS" if lo >= g.target else ("FAIL" if hi < g.target else "INCONCLUSIVE"))
                elif g.type == "noninferior":
                    r = noninferiority(a, bb, g.metric, g.margin, st.bootstrap_B, rng, st.alpha)
                    row.update(estimate=r.get("loss", NAN), ci_lo=r.get("lo", NAN), ci_hi=r.get("hi", NAN), verdict=r.get("verdict", "NO DATA"))
                elif g.type == "superior":
                    c = paired_compare(a, bb, g.metric, cfg, rng)
                    sig = c.get("p", 1.0) < st.alpha
                    row.update(estimate=c.get("improvement", NAN), ci_lo=c.get("diff_lo", NAN), ci_hi=c.get("diff_hi", NAN), p=c.get("p"),
                               verdict="PASS" if (sig and c["improvement"] > 0) else ("FAIL" if (sig or c["improvement"] <= 0) else "INCONCLUSIVE"))
                else:
                    row["verdict"] = "BAD GOAL TYPE"
                out.append(row)
    return out


# =============================================================================
# 8. ANALYSIS ORCHESTRATION
# =============================================================================
def analyze(rs: ResultSet, cfg: EvalConfig, holdout: Optional[float] = None) -> dict:
    st = cfg.stats
    rng = np.random.default_rng(np.random.SeedSequence(st.seed))
    A: Dict[str, Any] = {"notes": []}
    labels = rs.labels
    prop_all = [l for l in labels if match(l, cfg.groups.proposed)]
    base = [l for l in labels if (match(l, cfg.groups.baselines) if cfg.groups.baselines else l not in prop_all)]
    if not prop_all:
        A["notes"].append("No policy label matched groups.proposed; comparisons skipped.")
    allkeys = rs.keys()
    frac = cfg.selection.holdout_frac if holdout is None else holdout
    keys_test, keys_train = allkeys, []
    proposed = prop_all
    A["selection"] = None
    if frac > 0 and len(allkeys) >= 6 and len(prop_all) > 1:
        ntr = max(2, int(round(frac * len(allkeys))))
        keys_train, keys_test = allkeys[:ntr], allkeys[ntr:]
        sel = select_variants(rs, cfg, prop_all, keys_train)
        A["selection"] = dict(sel, n_train=len(keys_train), n_test=len(keys_test))
        if sel["best"]:
            proposed = [sel["best"]]
        A["notes"].append(f"Hold-out: variant chosen on first {len(keys_train)} trials, evaluated on the remaining {len(keys_test)}.")
    elif len(prop_all) > 1:
        A["notes"].append("Several sweep variants analysed on the SAME trials: selecting the best-looking one is optimistically biased; "
                          "use --holdout for an unbiased estimate. Multiplicity correction spans all variants.")
    A.update(labels=labels, proposed=proposed, baselines=base, proposed_all=prop_all, keys_train=keys_train, keys_test=keys_test)
    mets = [m for m in dict.fromkeys(HEADLINE + st.family_metrics + [c.metric for c in cfg.goals] + ["blind_fraction", "ctrl_lat_p95_ms",
            "ctrl_lat_p999_ms", "video_usable_fraction", "n_pingpong", "proactive_switch_frac", "energy_per_video_mbit_j",
            "spectrum_mean_mhz", "n_primary_outages", "outage_to_blind_rate", "pred_auc", "pred_brier", "pred_ep_recall",
            "pred_lead_mean_s", "pred_false_alarms_per_min", "pred_precision", "pred_ece", "pred_ap", cfg.selection.objective]
            + [c["metric"] for c in cfg.selection.constraints]) if m in rs.metrics_available() or m in HEADLINE]
    A["metrics"] = mets
    # per-policy descriptive summary (all runs of the policy, on test keys when hold-out)
    A["summary"] = {L: {m: summarize_metric(rs.vec(L, m, keys_test if frac > 0 else None), st.bootstrap_B, rng, st.alpha) for m in mets}
                    for L in labels}
    A["n_runs"] = {L: len(rs.by[L]) for L in labels}
    A["outcomes"] = {L: {o: sum(1 for r in rs.by[L].values() if r.get("arrays", {}).get("outcome", r.get("outcome")) == o)
                         for o in sorted({r.get("arrays", {}).get("outcome", r.get("outcome")) for r in rs.by[L].values()}, key=str)} for L in labels}
    # paired comparisons
    comps = []
    for p in proposed if frac > 0 else prop_all:
        for b in base:
            ks = [k for k in keys_test if k in rs.by.get(p, {}) and k in rs.by.get(b, {})]
            if len(ks) < st.min_pairs:
                continue
            for m in mets:
                c = paired_compare(rs.vec(p, m, ks), rs.vec(b, m, ks), m, cfg, rng)
                if c.get("n", 0) >= st.min_pairs:
                    c.update(policy=p, baseline=b, family=m in st.family_metrics)
                    comps.append(c)
    fam = [i for i, c in enumerate(comps) if c["family"] and np.isfinite(c.get("p", NAN))]
    adj = adjust_p([comps[i]["p"] for i in fam], st.correction)
    for i, pa in zip(fam, adj):
        comps[i]["p_adj"] = pa
    A["comparisons"] = comps
    # Friedman across all policies on the primary metric
    common = rs.common_keys(labels) if len(labels) >= 3 else []
    if len(common) >= 5:
        M = np.stack([rs.vec(L, st.primary_metric, common) for L in labels], axis=1)
        M = M[np.all(np.isfinite(M), axis=1)]
        if len(M) >= 5:
            A["friedman"] = dict(friedman_test(M), labels=labels, metric=st.primary_metric)
    A["goals"] = eval_goals(rs, cfg, proposed if frac > 0 else prop_all, base, keys_test if frac > 0 else [], rng)
    # Pareto over policy means
    px, py, pz = cfg.plots.pareto_x, cfg.plots.pareto_y, cfg.plots.pareto_size
    pts = np.array([[A["summary"][L].get(py, {}).get("mean", NAN), A["summary"][L].get(px, {}).get("mean", NAN)] for L in labels])
    A["pareto"] = dict(labels=labels, mask=pareto_mask(pts, [1, 1]).tolist() if len(pts) else [], x=px, y=py, size=pz)
    # sweep table
    sweep = {L: parse_label(L)[1] for L in labels if parse_label(L)[1]}
    A["sweep"] = {L: dict(params=p, **{m: A["summary"][L].get(m, {}).get("mean", NAN) for m in
                                      ("blind_time_s", "video_quality_mean", "ctrl_lat_p99_ms", "ctrl_within_150ms_rate",
                                       "comm_power_mean_w", "handover_cost", "utility_cost", "n_hard_switch", "n_soft_switch")})
                  for L, p in sweep.items()}
    # data quality notes
    tiers = {}
    for r in rs.runs:
        tiers[r.get("tier", "?")] = tiers.get(r.get("tier", "?"), 0) + 1
    A["tiers"] = tiers
    if tiers.get("summary"):
        A["notes"].append(f"{tiers['summary']} runs had no tick logs and were evaluated from summary metrics only (fixed definitions, no latency/video/handover detail).")
    nonc = sum(1 for r in rs.runs if not r.get("scalars", {}).get("mission_completed", 1))
    if nonc:
        A["notes"].append(f"{nonc} runs did not complete the mission (failsafe landing, battery, timeout). Absolute blind time is censored by shorter flights; "
                          "read blind_fraction / blind_per_min alongside it.")
    zs = [L for L in labels if rs.vec(L, "blind_time_s").size and np.nanmean(rs.vec(L, "blind_time_s") == 0) > 0.3]
    if zs:
        A["notes"].append("Blind time is zero-inflated for: " + ", ".join(zs) + ". Means can be dominated by few runs; rank-based tests, the P(any blind) McNemar test and medians are reported too.")
    if len(rs.excluded):
        A["notes"].append(f"{len(rs.excluded)} runs excluded (airborne time < min_flight_s).")
    return A


# =============================================================================
# 9. BATCH PIPELINE: read logs / optional bit-exact re-simulation / parallel evaluate
# =============================================================================
def _load_sim(sim_module: Optional[str]):
    if sim_module and os.path.exists(sim_module):
        spec = importlib.util.spec_from_file_location("uavlink_sim", sim_module)
        m = importlib.util.module_from_spec(spec)
        sys.modules["uavlink_sim"] = m
        spec.loader.exec_module(m)
        return m
    import uavlink_sim
    return uavlink_sim


def _numeric_row(row: dict) -> dict:
    sc = {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    sc["blind_any"] = float(row.get("n_blind_episodes", 0) > 0)
    sc["excluded"] = 0.0
    sc["mission_completed"] = float(row.get("outcome") == "completed")
    return sc


def _eval_job(job: dict) -> dict:
    res = dict(run_id=job["run_id"], label=job["label"], trial=job["trial"], scenario=job.get("scenario", ""),
               tier="none", error="", crc_ok=None, outcome=job["row"].get("outcome"))
    tmp = None
    try:
        cfg = from_dict(EvalConfig, job["cfg"])
        ld = job.get("log_dir")
        have = bool(ld) and os.path.exists(os.path.join(ld, "ticks.csv.gz"))
        if not have and job.get("sim_job"):
            sim = _load_sim(job.get("sim_module"))
            tmp = tempfile.mkdtemp(prefix="uleval_")
            sj = dict(job["sim_job"], log_level="full", out_dir=tmp)
            rr = sim.run_job(sj)
            if rr.get("error"):
                raise RuntimeError("resim failed: " + rr["error"])
            ld, have = os.path.join(tmp, job["run_id"]), True
            if job["row"].get("crc") is not None:
                res["crc_ok"] = bool(rr.get("crc") == job["row"].get("crc"))
            job["row"] = dict(job["row"], **{k: rr[k] for k in ("outcome", "n_rlf", "n_failsafe") if k in rr})
            res["outcome"] = job["row"].get("outcome")
        if have:
            tk = load_ticks(os.path.join(ld, "ticks.csv.gz"))
            ev = load_jsonl(os.path.join(ld, "events.jsonl"))
            dc = load_jsonl(os.path.join(ld, "decisions.jsonl"))
            sp = os.path.join(ld, "scenario.json")
            scn = json.load(open(sp)) if os.path.exists(sp) else {}
            o = evaluate_run(tk, ev, dc, scn, cfg, job["row"])
            res.update(tier="ticks", scalars=o["scalars"], arrays=o["arrays"])
        else:
            res.update(tier="summary", scalars=_numeric_row(job["row"]), arrays={})
        res["scalars"]["tier_ticks"] = 1.0 if res["tier"] == "ticks" else 0.0
    except Exception:
        res["error"] = traceback.format_exc(limit=5)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return res


def collect_jobs(batch_dir: Optional[str], runs_dir: Optional[str], cfg: EvalConfig, resim: bool, sim_module: Optional[str]) -> List[dict]:
    jobs = []
    cdict = json.loads(json.dumps(dataclasses.asdict(cfg), default=_jdefault))
    if batch_dir:
        rows = load_jsonl(os.path.join(batch_dir, "runs.jsonl"))
        good = {}
        for r in rows:                                     # last good row wins (resume may append retries)
            if not r.get("error"):
                good[r["run_id"]] = r
        sim_jobs = {}
        cp = os.path.join(batch_dir, "batch_config.json")
        if resim and os.path.exists(cp):
            sim = _load_sim(sim_module)
            sim_jobs = {j["run_id"]: j for j in sim.make_jobs(json.load(open(cp)), None)}
        for rid, r in sorted(good.items()):
            ld = os.path.join(batch_dir, "logs", rid)
            jobs.append(dict(run_id=rid, label=r["policy_label"], trial=r["trial"], scenario=r.get("scenario", ""), row=r,
                             log_dir=ld, sim_job=sim_jobs.get(rid) if resim else None, sim_module=sim_module, cfg=cdict))
    elif runs_dir:
        for i, d in enumerate(sorted(glob_dirs(runs_dir))):
            man = json.load(open(os.path.join(d, "manifest.json"))) if os.path.exists(os.path.join(d, "manifest.json")) else {}
            met = json.load(open(os.path.join(d, "metrics.json"))) if os.path.exists(os.path.join(d, "metrics.json")) else {}
            pol = man.get("policy")
            lab = pol.get("name") if isinstance(pol, dict) else (pol or met.get("policy") or os.path.basename(d))
            jobs.append(dict(run_id=os.path.basename(d), label=str(lab), trial=int(man.get("seed", i)), scenario=met.get("scenario", ""),
                             row=dict(met, crc=None), log_dir=d, sim_job=None, sim_module=None, cfg=cdict))
    return jobs


def glob_dirs(root: str) -> List[str]:
    return [os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "ticks.csv.gz"))]


def build_results(batch_dir=None, runs_dir=None, cfg: Optional[EvalConfig] = None, resim=False, workers=1,
                  sim_module=None, quiet=False) -> Tuple[ResultSet, dict]:
    cfg = cfg or EvalConfig()
    jobs = collect_jobs(batch_dir, runs_dir, cfg, resim, sim_module)
    if not jobs:
        raise SystemExit("no runs found (need runs.jsonl in --batch, or run folders with ticks.csv.gz in --runs-dir)")
    out, t0 = [], time.perf_counter()

    def sink(r):
        out.append(r)
        if not quiet and (len(out) % max(1, len(jobs) // 10) == 0 or len(out) == len(jobs)):
            print(f"  evaluated {len(out)}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    if workers <= 1:
        for j in jobs:
            sink(_eval_job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(_eval_job, j) for j in jobs]):
                sink(f.result())
    info = dict(n_jobs=len(jobs), n_errors=sum(1 for r in out if r["error"]), errors=[r["error"] for r in out if r["error"]][:3],
                crc_mismatch=[r["run_id"] for r in out if r.get("crc_ok") is False],
                config_hash=chash(dataclasses.asdict(cfg)), module_version=__version__, numpy=np.__version__,
                python=sys.version.split()[0], source=batch_dir or runs_dir, resim=resim)
    return ResultSet(out), info


# =============================================================================
# 10. FIGURES
# =============================================================================
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def _save(fig, outdir: str, name: str, cfg: EvalConfig) -> str:
    fn = f"{name}.{cfg.plots.fmt}"
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fn), dpi=cfg.plots.dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fn


def _order(A: dict, cfg: EvalConfig) -> List[str]:
    labs = [l for l in A["baselines"]] + [l for l in A["proposed_all"]]
    labs += [l for l in A["labels"] if l not in labs]
    return labs[: cfg.plots.max_policies]


def _colors(labels: List[str]):
    plt = _plt()
    cm = plt.get_cmap("tab20")
    return {l: cm(i % 20) for i, l in enumerate(labels)}


def reliability(tk: Dict[str, np.ndarray], cfg: EvalConfig) -> dict:
    """Same definitions as evaluate_run (used by timelines)."""
    S, D = cfg.schema, cfg.definitions
    t = tk[S.t]
    n = len(t)
    dt = float(np.median(np.diff(t)))
    A = np.isin(tk[S.mode], D.airborne_modes) if S.mode in tk else np.ones(n, bool)
    sent, lat = tk[S.ctrl_sent] > 0.5, tk[S.ctrl_lat_ms]
    ontime = sent & np.isfinite(lat) & (lat <= D.deadline_ms)
    ratio = rolling_ratio(t, ontime, sent, D.ctrl_window_s)
    gap = (t - np.maximum.accumulate(np.where(ontime, t, t[0]))) * 1000.0
    vok = tk[S.vid_ok] > 0.5
    if D.vid_min_quality > 0 and S.vid_q in tk:
        vok = vok & (tk[S.vid_q] >= D.vid_min_quality)
    crel = (gap <= D.ctrl_gap_max_ms) & (ratio >= D.ctrl_ok_ratio)
    vuse = rolling_ratio(t, vok, np.ones(n), D.vid_window_s) >= D.vid_ok_ratio
    return dict(t=t, dt=dt, A=A, ctrl_rel=crel, vid_use=vuse, vok=vok, blind=A & ~crel & ~vuse, ontime=ontime, gap=gap)


def _shade(ax, t, mask, color, alpha=0.18):
    s, e = runs_of(mask)
    for a, b in zip(s, e):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=alpha, lw=0)


STATE_COLOR = {"cold": "#bbbbbb", "warming": "#f5a623", "warm": "#4caf50"}


def fig_timeline_run(tk, cfg: EvalConfig, outdir: str, name: str, title: str = "") -> Optional[str]:
    plt = _plt()
    if plt is None:
        return None
    S, D = cfg.schema, cfg.definitions
    R = reliability(tk, cfg)
    t, links = R["t"], discover_links(tk, S)
    fig, ax = plt.subplots(6, 1, figsize=(13, 13), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1, 1.2, 1, 1.3, 1]))
    for L in links:
        ax[0].plot(t, tk[f"{L}{S.link_rx}"], lw=0.6, label=L)
        if f"{L}{S.link_sinr}" in tk:
            ax[1].plot(t, tk[f"{L}{S.link_sinr}"], lw=0.6, label=L)
    ax[0].set_ylabel("DL rx (dBm)"); ax[0].legend(ncol=len(links), fontsize=8, loc="upper right")
    ax[1].set_ylabel("DL SINR (dB)")
    lat, sent = tk[S.ctrl_lat_ms], tk[S.ctrl_sent] > 0.5
    ax[2].plot(t[sent], lat[sent], ".", ms=1.5, color="#1f77b4")
    lost = sent & ~np.isfinite(lat)
    ax[2].plot(t[lost], np.full(lost.sum(), 5.0), "|", ms=6, color="red", label="lost")
    for lim in D.latency_limits_ms:
        ax[2].axhline(lim, color="r", ls="--", lw=0.7)
    ax[2].set_ylim(0, max(300, 2 * max(D.latency_limits_ms))); ax[2].set_ylabel("ctrl latency (ms)")
    ax[3].plot(t, tk[S.vid_q], lw=0.7, color="#9467bd"); ax[3].set_ylabel("video quality")
    _shade(ax[3], t, ~R["vok"] & R["A"], "#9467bd", 0.25)
    # link states + primary strip
    ax[4].set_ylabel("link state"); ax[4].set_yticks(range(len(links) + 1))
    for i, L in enumerate(links):
        st = tk.get(f"{L}{S.link_state}")
        if st is None:
            continue
        for sname, col in STATE_COLOR.items():
            s, e = runs_of(st == sname)
            for a, b in zip(s, e):
                ax[4].fill_between([t[a], t[min(b, len(t) - 1)]], i + 0.1, i + 0.9, color=col, lw=0)
    ax[4].set_yticklabels(links + ["primary"])
    prim = np.array([str(c).split("+")[0] for c in tk[S.ctrl_links]])
    cm = plt.get_cmap("tab10")
    for j, L in enumerate(links):
        s, e = runs_of(prim == L)
        for a, b in zip(s, e):
            ax[4].fill_between([t[a], t[min(b, len(t) - 1)]], len(links) + 0.1, len(links) + 0.9, color=cm(j), lw=0)
    dup = np.array(["+" in str(c) for c in tk[S.ctrl_links]])
    s, e = runs_of(dup)
    for a, b in zip(s, e):
        ax[4].fill_between([t[a], t[min(b, len(t) - 1)]], len(links) + 0.9, len(links) + 1.0, color="k", lw=0)
    for i in (np.flatnonzero(prim[1:] != prim[:-1]) + 1):
        st = tk.get(f"{prim[i]}{S.link_state}")
        hard = not (st is not None and str(st[i - 1]) == "warm")
        ax[4].plot(t[i], len(links) + 0.5, "v" if hard else "o", color="red" if hard else "blue", ms=7, mec="k")
    ax[5].plot(t, tk[S.p_comm_w], lw=0.8, color="#d62728"); ax[5].set_ylabel("radio power (W)")
    if "outage_prob" in tk:
        a2 = ax[5].twinx(); a2.plot(t, tk["outage_prob"], lw=0.8, color="k", alpha=0.6); a2.set_ylim(0, 1.05); a2.set_ylabel("P(outage)")
    ax[5].set_xlabel("t (s)")
    for a in ax:
        _shade(a, t, R["blind"], "red", 0.16)
    _shade(ax[2], t, R["A"] & ~R["ctrl_rel"] & ~R["blind"], "orange", 0.12)
    ax[0].set_title(title or "run timeline (red = blind, orange = control unreliable, red/blue markers = hard/soft switch)")
    return _save(fig, outdir, name, cfg)


def fig_timeline_compare(tks: Dict[str, dict], cfg: EvalConfig, outdir: str, name: str, title: str = "") -> Optional[str]:
    """Same trial, several policies: stacked event strips + shared channel (common random numbers)."""
    plt = _plt()
    if plt is None or not tks:
        return None
    S = cfg.schema
    labs = list(tks)
    first = tks[labs[0]]
    links = discover_links(first, S)
    fig, ax = plt.subplots(2, 1, figsize=(13, 3 + 0.9 * len(labs)), sharex=True, gridspec_kw=dict(height_ratios=[1.3, max(2, len(labs) * 0.9)]))
    for L in links:
        ax[0].plot(first[S.t], first[f"{L}{S.link_rx}"], lw=0.6, label=L)
    ax[0].set_ylabel("DL rx (dBm)"); ax[0].legend(ncol=len(links), fontsize=8, loc="upper right")
    for i, L in enumerate(labs):
        tk = tks[L]
        R = reliability(tk, cfg)
        t = R["t"]
        y0 = len(labs) - 1 - i
        rows = [(~R["ctrl_rel"] & R["A"], "#ff9800", 0.0, 0.3), (~R["vid_use"] & R["A"], "#2196f3", 0.3, 0.6), (R["blind"], "#d32f2f", 0.6, 0.95)]
        for m, col, lo, hi in rows:
            s, e = runs_of(m)
            for a, b in zip(s, e):
                ax[1].fill_between([t[a], t[min(b, len(t) - 1)]], y0 + lo, y0 + hi, color=col, lw=0)
        prim = np.array([str(c).split("+")[0] for c in tk[S.ctrl_links]])
        for k in (np.flatnonzero(prim[1:] != prim[:-1]) + 1):
            st = tk.get(f"{prim[k]}{S.link_state}")
            hard = not (st is not None and str(st[k - 1]) == "warm")
            ax[1].plot(t[k], y0 + 0.5, "v" if hard else "o", color="k" if hard else "w", mec="k", ms=5)
        ax[1].text(t[0], y0 + 0.96, f"{L}: blind {R['blind'].sum() * R['dt']:.1f}s", fontsize=8, va="bottom")
    ax[1].set_yticks([]); ax[1].set_xlabel("t (s)")
    ax[0].set_title(title or "same trial, all policies  (orange: control unreliable, blue: video unusable, red: BLIND; v hard / o soft switch)")
    return _save(fig, outdir, name, cfg)


def fig_blind_box(rs, A, cfg, outdir):
    plt = _plt()
    labs = _order(A, cfg)
    col = _colors(labs)
    keys = A["keys_test"] if A["keys_train"] else None
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, m in zip(axs, ("blind_time_s", "video_freeze_s", "ctrl_lat_p99_ms")):
        data = [rs.vec(L, m, keys) for L in labs]
        data = [d[np.isfinite(d)] for d in data]
        ax.boxplot(data, showfliers=False, widths=0.55)
        rng = np.random.default_rng(1)
        for i, d in enumerate(data):
            ax.plot(i + 1 + rng.uniform(-0.18, 0.18, len(d)), d, ".", ms=3, color=col[labs[i]], alpha=0.6)
            if len(d):
                ax.plot(i + 1, d.mean(), "D", color="k", ms=6)
        ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(mlabel(m)); ax.grid(alpha=0.3)
        if m == "ctrl_lat_p99_ms":
            for lim in cfg.definitions.latency_limits_ms:
                ax.axhline(lim, color="r", ls="--", lw=0.7)
    axs[0].set_title("per-trial distributions (diamond = mean)")
    return _save(fig, outdir, "fig_distributions", cfg)


def fig_forest(A, cfg, outdir):
    plt = _plt()
    m = cfg.stats.primary_metric
    rows = [c for c in A["comparisons"] if c["metric"] == m and "rel_reduction" in c and np.isfinite(c.get("rel_reduction_lo", NAN))]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.8))
    for i, c in enumerate(rows):
        y = len(rows) - 1 - i
        ax.errorbar(100 * c["rel_reduction"], y, xerr=[[100 * (c["rel_reduction"] - c["rel_reduction_lo"])], [100 * (c["rel_reduction_hi"] - c["rel_reduction"])]],
                    fmt="o", color="#1565c0", capsize=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{c['policy']}  vs  {c['baseline']}" for c in rows][::-1], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    for g in cfg.goals:
        if g.metric == m and g.type == "relative_reduction":
            ax.axvline(100 * g.target, color="g", ls="--", lw=1, label=f"goal {100 * g.target:.0f}%")
    ax.set_xlabel(f"reduction in {mlabel(m)} vs baseline (%), paired bootstrap {100 * (1 - cfg.stats.alpha):.0f}% CI")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    return _save(fig, outdir, "fig_forest", cfg)


def fig_latency_ccdf(rs, A, cfg, outdir):
    plt = _plt()
    labs = _order(A, cfg)
    col = _colors(labs)
    keys = A["keys_test"] if A["keys_train"] else None
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    any_ = False
    for L in labs:
        P = rs.pool(L, keys)
        h = np.asarray(P["lat_hist"]) if not np.isscalar(P["lat_hist"]) else None
        if h is None:
            continue
        tot = h.sum() + P["lat_lost"]
        cc = (tot - np.cumsum(h)) / tot
        ax.step(LAT_EDGES[1:], np.maximum(cc, 1e-6), where="post", color=col[L], label=L)
        any_ = True
    if not any_:
        plt.close(fig)
        return None
    for lim in cfg.definitions.latency_limits_ms:
        ax.axvline(lim, color="r", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 1.2)
    ax.set_xlabel("control latency (ms)"); ax.set_ylabel("P(latency > x or lost)")
    ax.set_title("pooled control-latency CCDF (lost packets stay in the tail)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    return _save(fig, outdir, "fig_latency_ccdf", cfg)


def fig_blind_ccdf(rs, A, cfg, outdir):
    plt = _plt()
    labs = _order(A, cfg)
    col = _colors(labs)
    keys = A["keys_test"] if A["keys_train"] else None
    fig, ax = plt.subplots(figsize=(8, 5))
    drawn = False
    for L in labs:
        d = np.sort(rs.pool(L, keys)["dur"])
        if len(d) < 2:
            continue
        ax.step(d, 1 - np.arange(len(d)) / len(d), where="post", color=col[L], label=f"{L} (n={len(d)})")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    for c in cfg.definitions.critical_blind_s:
        ax.axvline(c, color="r", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("blind episode duration (s)"); ax.set_ylabel("P(duration > x)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    return _save(fig, outdir, "fig_blind_episode_ccdf", cfg)


def fig_pareto(rs, A, cfg, outdir):
    plt = _plt()
    labs = A["labels"]
    px, py, pz = cfg.plots.pareto_x, cfg.plots.pareto_y, cfg.plots.pareto_size
    g = lambda L, m: A["summary"][L].get(m, {}).get("mean", NAN)
    X, Y, Z = (np.array([g(L, m) for L in labs]) for m in (px, py, pz))
    if not np.isfinite(X).any():
        return None
    mask = np.array(A["pareto"]["mask"], bool)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axs[0]
    Zs = np.where(np.isfinite(Z), Z, 0.3)
    ax.scatter(X, Y, s=60 + 400 * (Zs - np.nanmin(Zs)) / (np.ptp(Zs) + 1e-9), c=["#d32f2f" if m else "#90a4ae" for m in mask], edgecolor="k", alpha=0.85)
    for L, x, y in zip(labs, X, Y):
        ax.annotate(L, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(mlabel(px)); ax.set_ylabel(mlabel(py)); ax.grid(alpha=0.3)
    for g_ in cfg.goals:
        if g_.metric == px and g_.type == "absolute_max":
            ax.axvline(g_.target, color="g", ls="--", lw=0.8)
    ax.set_title(f"Pareto (red = non-dominated); bubble = {mlabel(pz)}")
    ax = axs[1]
    Q = np.array([g(L, "ctrl_within_150ms_rate") for L in labs])
    V = np.array([g(L, "video_quality_mean") for L in labs])
    ax.scatter(Q, V, c=["#d32f2f" if L in A["proposed_all"] else "#1565c0" for L in labs], edgecolor="k")
    for L, x, y in zip(labs, Q, V):
        ax.annotate(L, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("control within 150 ms (rate)"); ax.set_ylabel("video quality"); ax.grid(alpha=0.3)
    ax.set_title("latency-constraint vs video quality (red = proposed variants)")
    return _save(fig, outdir, "fig_pareto", cfg)


def fig_sweep(rs, A, cfg, outdir):
    plt = _plt()
    sw = A["sweep"]
    if not sw:
        return None
    from collections import defaultdict
    groups = defaultdict(list)
    for L in sw:
        groups[parse_label(L)[0]].append(L)
    base, labs = max(groups.items(), key=lambda kv: len(kv[1]))
    vals: Dict[str, set] = {}
    for L in labs:
        for k, v in sw[L]["params"].items():
            vals.setdefault(k, set()).add(v)
    var = [k for k, v in vals.items() if len(v) > 1 and all(isinstance(x, float) for x in v)]
    if not var:
        return None
    S = lambda L, m: A["summary"][L].get(m, {}).get("mean", NAN)
    metrics = [m for m in cfg.plots.sweep_metrics if any(np.isfinite(S(L, m)) for L in labs)]
    cons = cfg.selection.constraints
    feas = lambda L: all(np.isfinite(S(L, c["metric"])) and check_op(S(L, c["metric"]), c["op"], c["value"]) for c in cons)
    if len(var) == 1:
        k = var[0]
        fig, axs = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
        axs = np.atleast_1d(axs)
        for ax, m in zip(axs, metrics):
            xs = sorted(vals[k])
            ax.plot(xs, [np.nanmean([S(L, m) for L in labs if sw[L]["params"].get(k) == x]) for x in xs], "o-")
            ax.set_xlabel(k); ax.set_ylabel(mlabel(m)); ax.grid(alpha=0.3)
        return _save(fig, outdir, "fig_sweep", cfg)
    xk, yk = var[0], var[1]
    xs, ys = sorted(vals[xk]), sorted(vals[yk])
    fig, axs = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
    axs = np.atleast_1d(axs)
    best = min((L for L in labs if feas(L) and np.isfinite(S(L, cfg.selection.objective))), key=lambda L: S(L, cfg.selection.objective), default=None)
    for ax, m in zip(axs, metrics):
        Z = np.full((len(ys), len(xs)), np.nan)
        F = np.zeros((len(ys), len(xs)), bool)
        for yi, yv in enumerate(ys):
            for xi, xv in enumerate(xs):
                cell = [L for L in labs if sw[L]["params"].get(xk) == xv and sw[L]["params"].get(yk) == yv]
                if cell:
                    Z[yi, xi] = np.nanmean([S(L, m) for L in cell])
                    F[yi, xi] = all(feas(L) for L in cell)
        im = ax.imshow(Z, origin="lower", cmap="viridis_r" if better_is_lower(m) else "viridis", aspect="auto")
        for yi in range(len(ys)):
            for xi in range(len(xs)):
                if np.isfinite(Z[yi, xi]):
                    ax.text(xi, yi, f"{Z[yi, xi]:.3g}", ha="center", va="center", fontsize=8, color="w", fontweight="bold")
                    if not F[yi, xi]:
                        ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#ff5252", lw=0))
        if best is not None:
            bx, by = xs.index(sw[best]["params"][xk]), ys.index(sw[best]["params"][yk])
            ax.plot(bx, by, "*", color="gold", ms=16, mec="k")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{y:g}" for y in ys])
        ax.set_xlabel(xk); ax.set_ylabel(yk); ax.set_title(mlabel(m), fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{base}: threshold sweep (hatched = violates constraints "
                 + ", ".join(f"{c['metric']}{c['op']}{c['value']}" for c in cons) + "; star = best feasible)", fontsize=9)
    return _save(fig, outdir, "fig_sweep_heatmaps", cfg)


def fig_spatial(rs, A, cfg, outdir):
    plt = _plt()
    if not cfg.spatial.enabled:
        return None
    labs = _order(A, cfg)[:6]
    keys = A["keys_test"] if A["keys_train"] else None
    pools = {L: rs.pool(L, keys)["sp"] for L in labs}
    allk = [k for p in pools.values() for k in p]
    if not allk:
        return None
    nx, ny = max(k // 65536 for k in allk) + 1, max(k % 65536 for k in allk) + 1
    G = {}
    for L, p in pools.items():
        B, Dw = np.zeros((ny, nx)), np.zeros((ny, nx))
        for k, (dw, bl) in p.items():
            B[k % 65536, k // 65536], Dw[k % 65536, k // 65536] = bl, dw
        G[L] = (B, Dw)
    mind = 2.0
    ext = [0, nx * cfg.spatial.cell_m, 0, ny * cfg.spatial.cell_m]
    ncol = min(3, len(labs))
    nrow = int(math.ceil(len(labs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.3 * nrow), squeeze=False)
    vmax = max(np.nanmax(np.where(G[L][1] >= mind, G[L][0] / np.maximum(G[L][1], 1e-9), np.nan)) if (G[L][1] >= mind).any() else 0 for L in labs)
    vmax = max(vmax, 1e-3)
    for ax, L in zip(axs.ravel(), labs):
        B, Dw = G[L]
        F = np.where(Dw >= mind, B / np.maximum(Dw, 1e-9), np.nan)
        im = ax.imshow(F, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="magma_r")
        ax.set_title(L, fontsize=9); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="blind fraction")
    for ax in axs.ravel()[len(labs):]:
        ax.axis("off")
    fig.suptitle("where blind time happens (dwell-normalised; meaningful when the scenario geometry is fixed across trials)", fontsize=9)
    return _save(fig, outdir, "fig_spatial_blind", cfg)


def fig_time_heat(rs, A, cfg, outdir):
    plt = _plt()
    labs = _order(A, cfg)
    keys = A["keys_test"] if A["keys_train"] else None
    ph, tb = [], []
    for L in labs:
        P = rs.pool(L, keys)
        ph.append(np.asarray(P["ph_b"]) / np.maximum(np.asarray(P["ph_d"]), 1e-9) if not np.isscalar(P["ph_b"]) else None)
        if P["tb_b"]:
            w = max(len(x) for x in P["tb_b"])
            b = sum(np.pad(x, (0, w - len(x))) for x in P["tb_b"])
            d = sum(np.pad(x, (0, w - len(x))) for x in P["tb_d"])
            tb.append(b / np.maximum(d, 1e-9))
        else:
            tb.append(None)
    if all(x is None for x in ph):
        return None
    fig, axs = plt.subplots(1, 2, figsize=(15, 0.45 * len(labs) + 2.2))
    for ax, rows, title, xl in ((axs[0], ph, "blind fraction by flight progress", "flight progress (bins)"),
                                (axs[1], tb, "blind fraction by absolute time", f"time (bins of {cfg.spatial.time_bin_s:g} s)")):
        w = max(len(r) for r in rows if r is not None)
        Z = np.full((len(labs), w), np.nan)
        for i, r in enumerate(rows):
            if r is not None:
                Z[i, :len(r)] = r
        im = ax.imshow(Z, aspect="auto", cmap="magma_r", vmin=0)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8); ax.set_xlabel(xl); ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03)
    return _save(fig, outdir, "fig_time_heatmaps", cfg)


def fig_scorecard(rs, A, cfg, outdir):
    plt = _plt()
    labs = A["labels"]
    ref = next((b for b in A["baselines"] if fnmatch.fnmatch(b, "reactive*")), A["baselines"][0] if A["baselines"] else None)
    if ref is None:
        return None
    cols = [L for L in _order(A, cfg) if L != ref]
    mets = [m for m in ("blind_time_s", "max_blind_episode_s", "ctrl_lat_p99_ms", "ctrl_within_150ms_rate", "video_quality_mean",
                        "video_freeze_s", "handover_cost", "n_hard_switch", "energy_comm_wh", "comm_power_mean_w", "utility_cost")
            if m in A["metrics"]]
    Z, T, Sg = np.full((len(mets), len(cols)), np.nan), [[""] * len(cols) for _ in mets], [[""] * len(cols) for _ in mets]
    keys = A["keys_test"] if A["keys_train"] else None
    for j, L in enumerate(cols):
        ks = [k for k in (keys or rs.common_keys([L, ref])) if k in rs.by.get(L, {}) and k in rs.by.get(ref, {})]
        for i, m in enumerate(mets):
            a, b = rs.vec(L, m, ks), rs.vec(ref, m, ks)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3 or abs(b[ok].mean()) < 1e-12:
                continue
            pc = 100 * (a[ok].mean() - b[ok].mean()) / abs(b[ok].mean())
            Z[i, j] = -pc if better_is_lower(m) else pc
            T[i][j] = f"{pc:+.0f}%"
            p = wilcoxon_signed_rank(a[ok] - b[ok])["p"]
            Sg[i][j] = "**" if p < cfg.stats.alpha / len(mets) else ("*" if p < cfg.stats.alpha else "")
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 4, 0.5 * len(mets) + 2))
    im = ax.imshow(np.clip(Z, -100, 100), cmap="RdYlGn", vmin=-100, vmax=100, aspect="auto")
    for i in range(len(mets)):
        for j in range(len(cols)):
            if T[i][j]:
                ax.text(j, i, T[i][j] + Sg[i][j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mets))); ax.set_yticklabels([mlabel(m) for m in mets], fontsize=8)
    ax.set_title(f"change vs {ref} (green = better; * p<{cfg.stats.alpha:g}, ** Bonferroni over rows; paired Wilcoxon)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, label="improvement %")
    return _save(fig, outdir, "fig_scorecard_heatmap", cfg)


def fig_prediction(rs, A, cfg, outdir):
    plt = _plt()
    keys = A["keys_test"] if A["keys_train"] else None
    labs = [L for L in A["labels"] if not np.isscalar(rs.pool(L, keys)["ph_hp"])][:6]
    if not labs:
        return None
    col = _colors(labs)
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for L in labs:
        P = rs.pool(L, keys)
        hp, hn = np.asarray(P["ph_hp"], float), np.asarray(P["ph_hn"], float)
        if hp.sum() == 0 or hn.sum() == 0:
            continue
        tp, fp = np.cumsum(hp[::-1]) / hp.sum(), np.cumsum(hn[::-1]) / hn.sum()
        auc = float(np.trapz(np.concatenate(([0], tp)), np.concatenate(([0], fp))))
        axs[0, 0].plot(np.concatenate(([0], fp)), np.concatenate(([0], tp)), color=col[L], label=f"{L} AUC={auc:.3f}")
        prec = np.cumsum(hp[::-1]) / np.maximum(np.cumsum(hp[::-1]) + np.cumsum(hn[::-1]), 1e-9)
        axs[0, 1].plot(tp, prec, color=col[L], label=L)
        nb = cfg.prediction.calib_bins
        g = len(hp) // nb
        cp, cn = hp[:g * nb].reshape(nb, g).sum(1), hn[:g * nb].reshape(nb, g).sum(1)
        ok = (cp + cn) > 0
        mp = np.array([np.average(np.arange(i * g, (i + 1) * g) / len(hp) + 0.5 / len(hp), weights=(hp + hn)[i * g:(i + 1) * g] + 1e-12) for i in range(nb)])
        axs[0, 2].plot(mp[ok], (cp / np.maximum(cp + cn, 1))[ok], "o-", color=col[L], label=L)
        T = P["thr"]
        if T is not None:
            rec = T["thr_det"] / np.maximum(T["thr_ep"], 1)
            mins = max(float(np.nansum(rs.vec(L, "flight_s", keys))) / 60.0, 1e-9)
            axs[1, 0].plot(T["thr_false"] / mins, rec, "o-", color=col[L], label=L)
            axs[1, 1].plot(T["thr"], T["thr_lead"] / np.maximum(T["thr_det"], 1), "o-", color=col[L], label=L)
            axs[1, 2].plot(T["thr"], T["thr_false"] / mins, "o-", color=col[L], label=L)
    axs[0, 0].plot([0, 1], [0, 1], "k:"); axs[0, 0].set_xlabel("false-positive rate"); axs[0, 0].set_ylabel("true-positive rate"); axs[0, 0].set_title("ROC (window-level, pooled)")
    axs[0, 1].set_xlabel("recall"); axs[0, 1].set_ylabel("precision"); axs[0, 1].set_title("precision-recall")
    axs[0, 2].plot([0, 1], [0, 1], "k:"); axs[0, 2].set_xlabel("predicted P(outage)"); axs[0, 2].set_ylabel("observed frequency"); axs[0, 2].set_title("calibration")
    axs[1, 0].set_xlabel("false alarms / min"); axs[1, 0].set_ylabel("episode recall"); axs[1, 0].set_title("operating curve (threshold sweep)")
    axs[1, 1].set_xlabel("alert threshold"); axs[1, 1].set_ylabel("mean lead (s)"); axs[1, 1].set_title("warning lead time")
    axs[1, 2].set_xlabel("alert threshold"); axs[1, 2].set_ylabel("false alarms / min"); axs[1, 2].set_title("false-alarm rate")
    for g_ in cfg.goals:
        if g_.metric == "pred_ep_recall":
            axs[1, 0].axhline(g_.target, color="g", ls="--", lw=0.8)
        if g_.metric == "pred_false_alarms_per_min":
            axs[1, 0].axvline(g_.target, color="g", ls="--", lw=0.8)
            axs[1, 2].axhline(g_.target, color="g", ls="--", lw=0.8)
        if g_.metric == "pred_lead_mean_s":
            axs[1, 1].axhline(g_.target, color="g", ls="--", lw=0.8)
    for a in axs.ravel():
        a.grid(alpha=0.3); a.legend(fontsize=7)
    return _save(fig, outdir, "fig_prediction", cfg)


def fig_energy(rs, A, cfg, outdir):
    plt = _plt()
    labs = _order(A, cfg)
    keys = A["keys_test"] if A["keys_train"] else None
    names = sorted({m[len("energy_"):-3] for m in rs.metrics_available() if m.startswith("energy_") and m.endswith("_wh")
                    and not m.startswith("energy_state_") and m not in ("energy_comm_wh", "energy_comm_frac_battery")})
    if not names:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    bot = np.zeros(len(labs))
    for L_ in names:
        v = np.array([rs.mean(L, f"energy_{L_}_wh", keys) for L in labs])
        v = np.nan_to_num(v)
        axs[0].bar(range(len(labs)), v, bottom=bot, label=L_)
        bot += v
    axs[0].set_xticks(range(len(labs))); axs[0].set_xticklabels(labs, rotation=60, ha="right", fontsize=8); axs[0].set_ylabel("radio energy per flight (Wh)")
    axs[0].legend(); axs[0].set_title("energy by link")
    bot = np.zeros(len(labs))
    for sname in ("cold", "warming", "warm"):
        v = np.nan_to_num(np.array([rs.mean(L, f"energy_state_{sname}_wh", keys) for L in labs]))
        axs[1].bar(range(len(labs)), v, bottom=bot, label=sname, color=STATE_COLOR[sname])
        bot += v
    axs[1].set_xticks(range(len(labs))); axs[1].set_xticklabels(labs, rotation=60, ha="right", fontsize=8); axs[1].legend(); axs[1].set_title("energy by link state (warm = active + pre-warmed)")
    return _save(fig, outdir, "fig_energy", cfg)


def handover_components(sw: list, n_runs: int, H: HandoverCostCfg) -> Dict[str, float]:
    n = max(n_runs, 1)
    return {"hard": sum(1 for s in sw if s["kind"] == "hard") * H.hard / n, "soft": sum(1 for s in sw if s["kind"] == "soft") * H.soft / n,
            "ping-pong": sum(1 for s in sw if s["pingpong"]) * H.pingpong / n,
            "disruption": sum(s["disruption_s"] for s in sw) * H.disruption_per_s / n,
            "lost packets": sum(s["lost_pkts"] for s in sw) * H.lost_packet / n, "blind in window": sum(s["blind_s"] for s in sw) * H.blind_per_s / n}


def fig_handover(rs, A, cfg, outdir):
    plt = _plt()
    labs = _order(A, cfg)
    keys = A["keys_test"] if A["keys_train"] else None
    comp = {L: handover_components(rs.pool(L, keys)["sw"], rs.pool(L, keys)["n"], cfg.handover) for L in labs}
    if all(sum(c.values()) == 0 for c in comp.values()):
        return None
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bot = np.zeros(len(labs))
    for k in next(iter(comp.values())):
        v = np.array([comp[L][k] for L in labs])
        ax.bar(range(len(labs)), v, bottom=bot, label=k)
        bot += v
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8); ax.set_ylabel("handover cost per flight (units)")
    ax.legend(fontsize=8); ax.set_title("handover cost decomposition")
    return _save(fig, outdir, "fig_handover_cost", cfg)


def fig_paired_scatter(rs, A, cfg, outdir):
    plt = _plt()
    prop = A["proposed"][:1] or A["proposed_all"][:1]
    if not prop or not A["baselines"]:
        return None
    p, bs = prop[0], A["baselines"][:4]
    m = cfg.stats.primary_metric
    fig, axs = plt.subplots(1, len(bs), figsize=(4.4 * len(bs), 4.2), squeeze=False)
    for ax, b in zip(axs[0], bs):
        ks = [k for k in (A["keys_test"] or rs.keys()) if k in rs.by.get(p, {}) and k in rs.by.get(b, {})]
        x, y = rs.vec(b, m, ks), rs.vec(p, m, ks)
        ax.scatter(x, y, s=14, alpha=0.7)
        hi = max(np.nanmax(x) if len(x) else 1, np.nanmax(y) if len(y) else 1, 1e-6)
        ax.plot([0, hi], [0, hi], "k:"); ax.set_xlabel(f"{b}: {mlabel(m)}"); ax.set_ylabel(f"{p}"); ax.set_title(f"below diagonal = {p} better", fontsize=9); ax.grid(alpha=0.3)
    return _save(fig, outdir, "fig_paired_scatter", cfg)


# =============================================================================
# 11. TABLES & REPORT
# =============================================================================
def num(x, d=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "inf" if x == math.inf else "-"
    if isinstance(x, (int, np.integer)):
        return str(x)
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{d}g}" if ax < 100 else f"{x:.0f}"


def mci(S: dict, m: str, d=3) -> str:
    e = S.get(m)
    if not e or not e.get("n"):
        return "-"
    return f"{num(e['mean'], d)} [{num(e['ci_lo'], d)}, {num(e['ci_hi'], d)}]"


def md_table(head: List[str], rows: List[List[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def tex_table(head: List[str], rows: List[List[str]], caption: str) -> str:
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    return ("\\begin{table}[t]\\centering\\small\n\\begin{tabular}{" + "l" + "r" * (len(head) - 1) + "}\\toprule\n" + " & ".join(esc(h) for h in head)
            + " \\\\\\midrule\n" + "\n".join(" & ".join(esc(c) for c in r) + " \\\\" for r in rows) + f"\n\\bottomrule\\end{{tabular}}\\caption{{{esc(caption)}}}\\end{{table}}\n")


def build_tables(rs: ResultSet, A: dict, cfg: EvalConfig) -> Dict[str, Tuple[List[str], List[List[str]]]]:
    T: Dict[str, Tuple[List[str], List[List[str]]]] = {}
    labs = _order(A, cfg) if A["labels"] else []
    S = A["summary"]
    T["headline"] = (["policy", "n", "blind s", "P(any blind)", "longest blind s", "ctrl<=100ms", "ctrl<=150ms", "ctrl p99 ms", "video Q", "freeze s", "J cost", "done"],
                     [[L, str(A["n_runs"][L]), mci(S[L], "blind_time_s", 3), num(S[L].get("blind_any", {}).get("mean", NAN), 2),
                       num(S[L].get("max_blind_episode_s", {}).get("mean", NAN), 3), num(S[L].get("ctrl_within_100ms_rate", {}).get("mean", NAN), 4),
                       num(S[L].get("ctrl_within_150ms_rate", {}).get("mean", NAN), 4), mci(S[L], "ctrl_lat_p99_ms", 3), mci(S[L], "video_quality_mean", 3),
                       mci(S[L], "video_freeze_s", 3), mci(S[L], "utility_cost", 3), num(S[L].get("mission_completed", {}).get("mean", NAN), 2)] for L in labs])
    # paired comparisons on primary metric(s)
    pm = cfg.stats.primary_metric
    rows = []
    for c in A["comparisons"]:
        if c["metric"] != pm:
            continue
        rows.append([c["policy"], c["baseline"], str(c["n"]), num(c["mean_a"]), num(c["mean_b"]), f"{num(c['diff'])} [{num(c['diff_lo'])}, {num(c['diff_hi'])}]",
                     f"{100 * c.get('rel_reduction', NAN):.1f}% [{100 * c.get('rel_reduction_lo', NAN):.1f}, {100 * c.get('rel_reduction_hi', NAN):.1f}]",
                     f"{c['wins']}/{c['losses']}/{c['ties']}", num(c["p"], 3), num(c.get("p_adj", NAN), 3), num(c.get("mcnemar_p", NAN), 3),
                     num(c["cohen_dz"], 2), num(c["cliffs_delta"], 2), num(c["hodges_lehmann"], 3), num(c["n_required_for_power"], 3), num(c["mde_units"], 3)])
    T["paired_primary"] = (["proposed", "baseline", "n", "mean prop", "mean base", "diff [CI]", "reduction [CI]", "win/loss/tie",
                            f"p ({cfg.stats.primary_test})", f"p adj ({cfg.stats.correction})", "McNemar p (any blind)", "dz", "Cliff d", "HL shift",
                            f"n for {int(100 * cfg.stats.power)}% power", "MDE (units)"], rows)
    rows = []
    for c in A["comparisons"]:
        if c["family"] and c["metric"] != pm:
            rows.append([c["policy"], c["baseline"], mlabel(c["metric"]), num(c["mean_a"]), num(c["mean_b"]), f"{num(c['diff'])} [{num(c['diff_lo'])}, {num(c['diff_hi'])}]",
                         num(c["p"], 3), num(c.get("p_adj", NAN), 3), num(c["cohen_dz"], 2), f"{c['wins']}/{c['losses']}"])
    T["paired_secondary"] = (["proposed", "baseline", "metric", "mean prop", "mean base", "diff [CI]", "p", "p adj", "dz", "win/loss"], rows)
    T["goals"] = (["goal", "policy", "vs", "estimate", "95% CI", "target", "verdict"],
                  [[g["goal"], g["policy"], g.get("vs", ""), num(g.get("estimate", NAN)), f"[{num(g.get('ci_lo', NAN))}, {num(g.get('ci_hi', NAN))}]", num(g["target"]), g["verdict"]] for g in A["goals"]])
    keys = A["keys_test"] if A["keys_train"] else None
    lat_rows = []
    for L in labs:
        P = rs.pool(L, keys)
        pooled = ["-"] * 4
        if not np.isscalar(P["lat_hist"]):
            pooled = [num(lat_hist_percentile(np.asarray(P["lat_hist"]), P["lat_lost"], q)) for q in (50, 95, 99, 99.9)]
        lat_rows.append([L] + [mci(S[L], m, 3) for m in ("ctrl_lat_p50_ms", "ctrl_lat_p95_ms", "ctrl_lat_p99_ms", "ctrl_lat_p999_ms")] + pooled +
                        [num(S[L].get("ctrl_lost_rate", {}).get("mean", NAN), 3), num(S[L].get("ctrl_max_gap_ms", {}).get("mean", NAN), 4)])
    T["latency"] = (["policy", "p50 ms", "p95 ms", "p99 ms", "p99.9 ms", "pooled p50", "pooled p95", "pooled p99", "pooled p99.9", "lost rate", "mean longest gap ms"], lat_rows)
    T["video"] = (["policy", "quality", "usable fraction", "decodable", "freeze s", "freezes", "longest freeze s", "delivered Mbps"],
                  [[L, mci(S[L], "video_quality_mean"), num(S[L].get("video_usable_fraction", {}).get("mean", NAN)), num(S[L].get("video_decodable_rate", {}).get("mean", NAN)),
                    mci(S[L], "video_freeze_s"), num(S[L].get("video_freeze_events", {}).get("mean", NAN)), num(S[L].get("video_freeze_max_s", {}).get("mean", NAN)),
                    num(S[L].get("video_mean_mbps", {}).get("mean", NAN))] for L in labs])
    T["handover"] = (["policy", "hard", "soft", "ping-pong", "switch/min", "cost [CI]", "disruption s", "proactive share", "hard time-to-usable s", "RLF", "dup time s"],
                     [[L, num(S[L].get("n_hard_switch", {}).get("mean", NAN), 3), num(S[L].get("n_soft_switch", {}).get("mean", NAN), 3), num(S[L].get("n_pingpong", {}).get("mean", NAN), 3),
                       num(S[L].get("switch_per_min", {}).get("mean", NAN), 3), mci(S[L], "handover_cost"), num(S[L].get("handover_disruption_s", {}).get("mean", NAN)),
                       num(S[L].get("proactive_switch_frac", {}).get("mean", NAN), 2), num(S[L].get("hard_time_to_usable_s", {}).get("mean", NAN)),
                       num(S[L].get("n_rlf", {}).get("mean", NAN), 3), num(S[L].get("dup_time_s", {}).get("mean", NAN))] for L in labs])
    T["energy"] = (["policy", "radio Wh [CI]", "mean W", "p99 W", "over power budget s", "spectrum MHz", "over spectrum s", "J / delivered Mbit", "% of battery"],
                   [[L, mci(S[L], "energy_comm_wh"), num(S[L].get("comm_power_mean_w", {}).get("mean", NAN)), num(S[L].get("comm_power_p99_w", {}).get("mean", NAN)),
                     num(S[L].get("over_power_budget_s", {}).get("mean", NAN)), num(S[L].get("spectrum_mean_mhz", {}).get("mean", NAN)),
                     num(S[L].get("over_spectrum_budget_s", {}).get("mean", NAN)), num(S[L].get("energy_per_video_mbit_j", {}).get("mean", NAN)),
                     num(100 * S[L].get("energy_comm_frac_battery", {}).get("mean", NAN), 3)] for L in labs])
    T["prediction"] = (["policy", "AUC", "AP", "Brier", "ECE", "episode recall", "precision", "lead s", "false alarms/min", "outages acted on", "primary outages/run", "outage->blind"],
                       [[L] + [num(S[L].get(m, {}).get("mean", NAN), 3) for m in ("pred_auc", "pred_ap", "pred_brier", "pred_ece", "pred_ep_recall", "pred_precision",
                                                                                "pred_lead_mean_s", "pred_false_alarms_per_min", "pred_actioned_rate", "n_primary_outages", "outage_to_blind_rate")]
                        for L in labs if S[L].get("pred_auc", {}).get("n") or S[L].get("n_primary_outages", {}).get("n")])
    if A["sweep"]:
        T["sweep"] = (["variant", "blind s", "video Q", "ctrl<=150ms", "ctrl p99 ms", "power W", "handover cost", "hard", "soft", "J cost"],
                      [[L, num(v["blind_time_s"]), num(v["video_quality_mean"]), num(v["ctrl_within_150ms_rate"], 4), num(v["ctrl_lat_p99_ms"]), num(v["comm_power_mean_w"]),
                        num(v["handover_cost"]), num(v["n_hard_switch"], 3), num(v["n_soft_switch"], 3), num(v["utility_cost"])] for L, v in A["sweep"].items()])
    return T


def md_to_html(md: str, outdir: str) -> str:
    import html as H
    out, lines, i = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1250px;margin:2em auto}table{border-collapse:collapse;font-size:12px;margin:1em 0}"
                     "td,th{border:1px solid #ccc;padding:3px 7px}th{background:#eee}img{max-width:100%}code{background:#f3f3f3}</style></head><body>"], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            blk = []
            while i < len(lines) and lines[i].startswith("|"):
                blk.append(lines[i]); i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in blk if not set(r.replace("|", "").strip()) <= set("-: ")]
            t = "<table>" + "".join(("<tr>" + "".join(f"<{'th' if k == 0 else 'td'}>{H.escape(c)}</{'th' if k == 0 else 'td'}>" for c in r) + "</tr>") for k, r in enumerate(rows)) + "</table>"
            out.append(t); continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", ln)
        if m:
            p = os.path.join(outdir, m.group(2))
            if os.path.exists(p):
                out.append(f"<p><img src='data:image/png;base64,{base64.b64encode(open(p, 'rb').read()).decode()}' alt='{H.escape(m.group(1))}'></p>")
        elif ln.startswith("#"):
            lv = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{lv}>{H.escape(ln.lstrip('# '))}</h{lv}>")
        elif ln.startswith("- "):
            out.append(f"<li>{H.escape(ln[2:])}</li>")
        elif ln.strip():
            out.append(f"<p>{re.sub(r'[*][*](.*?)[*][*]', r'<b>\\1</b>', H.escape(ln))}</p>")
        i += 1
    return "\n".join(out) + "</body></html>"


def write_report(rs, A, cfg, info, T, figs, outdir) -> str:
    st, D = cfg.stats, cfg.definitions
    md = ["# Multi-link UAV handover - evaluation report", ""]
    md += [f"*Source*: `{info.get('source')}` | runs: {sum(A['n_runs'].values())} over {len(A['labels'])} policies | tiers: {A['tiers']} | "
           f"config hash `{info['config_hash']}` | evaluator v{__version__}", ""]
    md += ["## Definitions in force", "",
           f"- **Control reliable**: on-time (<= {D.deadline_ms:g} ms) ratio >= {D.ctrl_ok_ratio:g} over {D.ctrl_window_s:g} s AND no gap > {D.ctrl_gap_max_ms:g} ms.",
           f"- **Video usable**: decodable ratio >= {D.vid_ok_ratio:g} over {D.vid_window_s:g} s. Freeze = stall >= {D.freeze_s:g} s.",
           f"- **Blind time**: airborne time with control NOT reliable AND video NOT usable (mode `{D.blind_mode}`).",
           f"- Latency limits reported: {D.latency_limits_ms}. Statistics: alpha={st.alpha:g}, {st.correction} correction, primary test {st.primary_test}, paired by trial.", ""]
    if info.get("crc_mismatch"):
        md += [f"> **WARNING**: {len(info['crc_mismatch'])} re-simulated runs did not reproduce the original state CRC (module version / numpy differences?): {info['crc_mismatch'][:5]}", ""]
    if info.get("n_errors"):
        md += [f"> **WARNING**: {info['n_errors']} runs failed to evaluate. First error:\n```\n{info['errors'][0]}\n```", ""]
    if A["notes"]:
        md += ["## Caveats detected automatically", ""] + [f"- {n}" for n in A["notes"]] + [""]
    sec = [("1. Headline results", "headline", "Mean per policy with bootstrap CI over trials."),
           ("2. Primary hypothesis: blind-time reduction (paired)", "paired_primary", f"Negative diff = less blind time. p adjusted over the family {cfg.stats.family_metrics}."),
           ("3. Goal scorecard (configurable targets)", "goals", "PASS only if the CI clears the target; INCONCLUSIVE = point estimate and CI straddle it."),
           ("4. Control latency distribution", "latency", "Per-run percentiles (mean [CI] over runs) and pooled percentiles from the packet histogram (lost packets count as +inf)."),
           ("5. Video quality and freezes", "video", ""), ("6. Handover count and cost", "handover", "Cost model weights: " + json.dumps(dataclasses.asdict(cfg.handover))),
           ("7. Radio energy, power and spectrum", "energy", ""), ("8. Outage prediction quality", "prediction", f"Horizon {cfg.prediction.horizon_s:g} s, threshold {cfg.prediction.threshold:g}, label mode {cfg.prediction.label_mode}."),
           ("9. Secondary metrics, paired (family)", "paired_secondary", "")]
    fig_after = {"headline": ["fig_distributions", "fig_scorecard_heatmap"], "paired_primary": ["fig_forest", "fig_paired_scatter"], "latency": ["fig_latency_ccdf"],
                 "handover": ["fig_handover_cost", "fig_blind_episode_ccdf"], "energy": ["fig_energy", "fig_pareto"], "prediction": ["fig_prediction"],
                 "paired_secondary": ["fig_time_heatmaps", "fig_spatial_blind"]}
    for title, key, cap in sec:
        if key in T and T[key][1]:
            md += [f"## {title}", "", cap, "", md_table(*T[key]), ""]
        for f in fig_after.get(key, []):
            fn = next((x for x in figs if x.startswith(f)), None)
            if fn:
                md += [f"![{f}]({fn})", ""]
    if A.get("friedman"):
        F = A["friedman"]
        md += ["## Omnibus test across all policies", "", f"Friedman on `{F['metric']}`: chi2={F['chi2']:.2f}, df={F['df']}, p={F['p']:.3g} (n={F['n']} complete trials). "
               "Mean ranks: " + ", ".join(f"{l}={r:.2f}" for l, r in zip(F['labels'], F['mean_ranks'])), ""]
    if A.get("selection"):
        s = A["selection"]
        md += ["## Threshold selection (hold-out)", "", f"Objective: minimise `{s['objective']}` subject to {cfg.selection.constraints}; chosen on {s['n_train']} train trials, "
               f"reported on {s['n_test']} held-out trials. **Chosen variant: `{s['best']}`** (feasible variants existed: {s['any_feasible']}).", "",
               md_table(["variant", "feasible", s["objective"]], [[r["label"], str(r["feasible"]), num(r["objective"])] for r in sorted(s["rows"], key=lambda r: (not r["feasible"], r["objective"] if np.isfinite(r["objective"]) else 1e9))]), ""]
    if "sweep" in T:
        md += ["## Threshold / cost-function sweep", "", md_table(*T["sweep"]), ""]
        for f in figs:
            if f.startswith("fig_sweep"):
                md += [f"![sweep]({f})", ""]
    tl = [f for f in figs if f.startswith("timeline")]
    if tl:
        md += ["## Timelines (same trial, all policies; best / worst case for the proposed method)", ""] + [f"![{f}]({f})" for f in tl] + [""]
    text = "\n".join(md)
    open(os.path.join(outdir, "report.md"), "w").write(text)
    open(os.path.join(outdir, "report.html"), "w").write(md_to_html(text, outdir))
    return text


# =============================================================================
# 12. ORCHESTRATION: evaluate a batch end to end
# =============================================================================
def obtain_ticks(meta: dict, run_id: str):
    ld = meta.get("log_dir")
    tmp = None
    if not (ld and os.path.exists(os.path.join(ld, "ticks.csv.gz"))):
        if not meta.get("sim_job"):
            return None, None
        sim = _load_sim(meta.get("sim_module"))
        tmp = tempfile.mkdtemp(prefix="uleval_tl_")
        sim.run_job(dict(meta["sim_job"], log_level="full", out_dir=tmp))
        ld = os.path.join(tmp, run_id)
    return load_ticks(os.path.join(ld, "ticks.csv.gz")), tmp


def pick_timeline_trials(rs, A, cfg) -> List[tuple]:
    if not A["proposed"] or not A["baselines"]:
        return []
    p = A["proposed"][0]
    ref = next((b for b in A["baselines"] if fnmatch.fnmatch(b, "reactive*")), A["baselines"][0])
    ks = [k for k in (A["keys_test"] or rs.keys()) if k in rs.by.get(p, {}) and k in rs.by.get(ref, {})]
    if not ks:
        return []
    d = np.array([rs.by[ref][k]["scalars"].get("blind_time_s", NAN) - rs.by[p][k]["scalars"].get("blind_time_s", NAN) for k in ks])
    ok = np.isfinite(d)
    if not ok.any():
        return []
    order = np.argsort(np.where(ok, d, np.nan))
    ordered = [ks[i] for i in order if ok[i]]
    picks = [ordered[-1]]
    if cfg.plots.n_timelines > 1 and len(ordered) > 1:
        picks.append(ordered[0])
    return picks[: cfg.plots.n_timelines]


def evaluate(batch=None, runs_dir=None, cfg: Optional[EvalConfig] = None, out="eval_out", resim=False, workers=1,
             sim_module=None, holdout=None, quiet=False) -> dict:
    cfg = cfg or EvalConfig()
    os.makedirs(out, exist_ok=True)
    rs, info = build_results(batch, runs_dir, cfg, resim, workers, sim_module, quiet)
    # remember how to get logs again (timelines)
    jobs = {j["run_id"]: j for j in collect_jobs(batch, runs_dir, cfg, resim, sim_module)}
    for r in rs.runs:
        j = jobs.get(r["run_id"])
        if j:
            r["meta"] = dict(log_dir=j["log_dir"], sim_job=j["sim_job"], sim_module=j["sim_module"])
    A = analyze(rs, cfg, holdout)
    T = build_tables(rs, A, cfg)
    figs: List[str] = []
    if cfg.plots.enabled and _plt() is not None:
        for fn in (fig_blind_box, fig_forest, fig_latency_ccdf, fig_blind_ccdf, fig_pareto, fig_sweep, fig_spatial, fig_time_heat,
                   fig_scorecard, fig_prediction, fig_energy, fig_handover, fig_paired_scatter):
            try:
                f = fn(rs, A, cfg, out) if fn not in (fig_forest,) else fn(A, cfg, out)
                if f:
                    figs.append(f)
            except Exception as e:                                   # a broken plot must never lose the statistics
                info.setdefault("plot_errors", []).append(f"{fn.__name__}: {e}")
        for n, key in enumerate(pick_timeline_trials(rs, A, cfg)):
            try:
                p = A["proposed"][0]
                refs = [b for b in A["baselines"]][:3]
                tks, tmps = {}, []
                for L in refs + [p]:
                    r = rs.by[L].get(key)
                    if r and r.get("meta"):
                        tk, tmp = obtain_ticks(r["meta"], r["run_id"])
                        if tk is not None:
                            tks[L] = tk
                        if tmp:
                            tmps.append(tmp)
                f = fig_timeline_compare(tks, cfg, out, f"timeline_compare_{n + 1}", f"scenario/trial {key}: " + ", ".join(tks))
                if f:
                    figs.append(f)
                if p in tks:
                    f = fig_timeline_run(tks[p], cfg, out, f"timeline_detail_{n + 1}", f"{p} - {key}")
                    if f:
                        figs.append(f)
                for tmp in tmps:
                    shutil.rmtree(tmp, ignore_errors=True)
            except Exception as e:
                info.setdefault("plot_errors", []).append(f"timeline: {e}")
    elif cfg.plots.enabled:
        info["plot_errors"] = ["matplotlib not available: figures skipped"]
    # exports
    with open(os.path.join(out, "runs_metrics.csv"), "w", newline="") as f:
        cols = sorted({k for r in rs.runs for k in r.get("scalars", {})})
        w = csv.writer(f)
        w.writerow(["run_id", "policy", "trial", "scenario", "tier", "outcome"] + cols)
        for r in rs.runs:
            w.writerow([r["run_id"], r["label"], r["trial"], r.get("scenario", ""), r["tier"], r.get("outcome", "")] + [r["scalars"].get(c, "") for c in cols])
    for name, (h, rows) in T.items():
        with open(os.path.join(out, f"table_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(h)
            w.writerows(rows)
        if name in ("headline", "paired_primary", "goals", "latency"):
            open(os.path.join(out, f"table_{name}.tex"), "w").write(tex_table(h, rows, name.replace("_", " ")))
    with open(os.path.join(out, "paired_comparisons_all.csv"), "w", newline="") as f:
        if A["comparisons"]:
            ks = sorted({k for c in A["comparisons"] for k in c})
            w = csv.DictWriter(f, fieldnames=ks)
            w.writeheader()
            w.writerows(A["comparisons"])
    slim = {k: v for k, v in A.items() if k not in ("keys_train", "keys_test")}
    slim["keys_train"], slim["keys_test"] = [list(k) for k in A["keys_train"]], [list(k) for k in A["keys_test"]]
    json.dump(slim, open(os.path.join(out, "analysis.json"), "w"), indent=1, default=_jdefault)
    json.dump(dataclasses.asdict(cfg), open(os.path.join(out, "eval_config_used.json"), "w"), indent=1, default=_jdefault)
    json.dump(info, open(os.path.join(out, "provenance.json"), "w"), indent=1, default=_jdefault)
    write_report(rs, A, cfg, info, T, figs, out)
    return dict(analysis=A, tables=T, info=info, figures=figs, results=rs)


# =============================================================================
# 13. SELFTEST + CLI
# =============================================================================
def _synthetic_ticks(dt=0.02, T=60.0, ctrl_fail=(10, 14), vid_fail=(12, 20)) -> Dict[str, np.ndarray]:
    t = np.arange(0, T, dt)
    n = len(t)
    cf = (t >= ctrl_fail[0]) & (t < ctrl_fail[1])
    vf = (t >= vid_fail[0]) & (t < vid_fail[1])
    tk = {"t": t, "ctrl_sent": np.ones(n), "ctrl_ok": (~cf).astype(float), "ctrl_lat_ms": np.where(cf, np.nan, 20.0),
          "vid_ok": (~vf).astype(float), "vid_q": np.where(vf, 0.0, 0.8), "video_bps": np.full(n, 4e6), "mode": np.array(["MISSION"] * n, dtype=object),
          "x": t * 5.0, "y": np.zeros(n), "z": np.full(n, 50.0), "p_comm_w": np.full(n, 3.0),
          "ctrl_links": np.array(["A"] * n, dtype=object), "A.rx_dl_dbm": np.full(n, -70.0), "A.state": np.array(["warm"] * n, dtype=object),
          "A.p_w": np.full(n, 3.0), "A.phy_out": np.zeros(n)}
    po = (t >= 30) & (t < 33)
    tk["A.phy_out"] = po.astype(float)
    return tk


def selftest(sim_path: Optional[str] = None) -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)
    check("norm_ppf(0.975)=1.95996", abs(norm_ppf(0.975) - 1.959964) < 1e-4)
    check("chi2_sf(3.841,1)=0.05 / chi2_sf(9.21,2)=0.01", abs(chi2_sf(3.841459, 1) - 0.05) < 1e-4 and abs(chi2_sf(9.21034, 2) - 0.01) < 1e-4)
    check("wilcoxon exact n=5 all positive -> p=0.0625", abs(wilcoxon_signed_rank([1, 2, 3, 4, 5])["p"] - 0.0625) < 1e-12)
    check("sign test 5+/1- -> p=0.21875", abs(sign_test([1, 1, 1, 1, 1, -1])["p"] - 0.21875) < 1e-12)
    check("Holm / BH adjustment", np.allclose(adjust_p([0.01, 0.04, 0.03], "holm"), [0.03, 0.06, 0.06]) and np.allclose(adjust_p([0.01, 0.04, 0.03], "bh"), [0.03, 0.04, 0.04]))
    try:
        from scipy import stats as sps
        r = np.random.default_rng(0)
        d = r.normal(0.3, 1, 25)
        check("wilcoxon matches scipy (exact)", abs(wilcoxon_signed_rank(d)["p"] - sps.wilcoxon(d, method="exact").pvalue) < 1e-9)
        M = r.normal(size=(30, 4)) + np.arange(4) * 0.3
        check("friedman matches scipy", abs(friedman_test(M)["p"] - sps.friedmanchisquare(*M.T).pvalue) < 1e-6)
    except ImportError:
        print("  [skip] scipy cross-checks (scipy not installed)")
    rng = np.random.default_rng(1)
    check("paired permutation detects shift / not null", perm_test_paired(rng.normal(1, 1, 30), 5000, rng)["p"] < 0.01 and perm_test_paired(rng.normal(0, 1, 30), 5000, rng)["p"] > 0.05)
    cfg = EvalConfig()
    tk = _synthetic_ticks()
    dec = [{"t": round(i * 0.1, 3), "tag": "hold", "ctrl": ["A"], "warm": [], "pout": {"A": 0.9 if 28 <= i * 0.1 < 30 else 0.05}} for i in range(600)]
    o = evaluate_run(tk, [], dec, {"links": [{"name": "A", "bw_hz": 20e6}]}, cfg, {"outcome": "completed"})
    s = o["scalars"]
    check(f"synthetic blind time ~2.1 s (got {s['blind_time_s']:.2f})", 1.8 <= s["blind_time_s"] <= 2.4)
    check("synthetic control unreliable ~4.3 s", 4.0 <= s["ctrl_unreliable_s"] <= 4.8)
    check("synthetic latency p99 = 20 ms, lost rate ~6.7%", abs(s["ctrl_lat_p99_ms"] - 20) < 1e-6 and abs(s["ctrl_lost_rate"] - 4 / 60) < 1e-3)
    check("synthetic video freeze ~8 s", abs(s["video_freeze_s"] - 8.0) < 0.1)
    check("synthetic energy = 3 W * 60 s / 3600", abs(s["energy_comm_wh"] - 3 * 60 / 3600) < 1e-4)
    check(f"prediction: recall=1, lead~2 s (got {s.get('pred_ep_recall')}, {s.get('pred_lead_mean_s')})", s.get("pred_ep_recall") == 1.0 and abs(s.get("pred_lead_mean_s", 0) - 2.0) < 0.15)
    check("prediction: AUC high", s.get("pred_auc", 0) > 0.9)
    R = reliability(tk, cfg)
    check("timeline helper agrees with evaluate_run", abs(R["blind"].sum() * R["dt"] - s["blind_time_s"]) < 1e-9)
    cfg2 = from_dict(EvalConfig, {"definitions": {"ctrl_gap_max_ms": 100, "deadline_ms": 10}})
    s2 = evaluate_run(tk, [], [], {}, cfg2, {})["scalars"]
    check("configurable thresholds change results (deadline 10 ms => everything late)", s2["ctrl_ontime_rate"] == 0.0 and s2["blind_time_s"] > s["blind_time_s"])
    # end-to-end against the real simulator if available
    sp = sim_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "uavlink_sim.py")
    if os.path.exists(sp):
        try:
            u = _load_sim(sp)
            d = tempfile.mkdtemp(prefix="uleval_e2e_")
            c = dict(name="e2e", master_seed=5, n_trials=3, scenario="jammed_inspection", baseline="reactive",
                     overrides=dict(dt=0.05, duration_s=90.0, traffic=dict(ctrl_hz=20.0)),
                     randomize=dict(wind_scale=[0.8, 1.5]),
                     policies=[dict(label="static", name="static_primary"), dict(label="reactive", name="reactive"), dict(label="bonding", name="bonding"),
                               dict(label="pred", name="cost_predictive", sweep=dict(p_hard=[0.3, 0.6]))], save_ticks="first:1")
            u.run_batch(c, os.path.join(d, "b"), workers=1, quiet=True)
            res = evaluate(batch=os.path.join(d, "b"), cfg=EvalConfig(plots=PlotCfg(n_timelines=1)), out=os.path.join(d, "ev"), resim=True, workers=1, sim_module=sp, quiet=True)
            info, A = res["info"], res["analysis"]
            check("e2e: no failed evaluations, CRC reproduced on re-simulation", info["n_errors"] == 0 and not info["crc_mismatch"])
            rr = res["results"]
            lg = [(r["scalars"]["blind_time_s"], r["scalars"].get("tier_ticks")) for r in rr.runs]
            rows = {r["run_id"]: r for r in u.load_jsonl(os.path.join(d, "b", "runs.jsonl"))} if hasattr(u, "load_jsonl") else {}
            diffs = [abs(r["scalars"]["blind_time_s"] - json.loads(l)["blind_time_s"]) for r in rr.runs for l in open(os.path.join(d, "b", "runs.jsonl")) if json.loads(l)["run_id"] == r["run_id"]]
            check(f"e2e: recomputed blind time agrees with simulator's own (max |diff| {max(diffs):.2f}s)", max(diffs) < 0.5)
            check("e2e: report.md/html and tables written", all(os.path.exists(os.path.join(d, "ev", f)) for f in ("report.md", "report.html", "table_headline.csv", "analysis.json")))
            check("e2e: figures produced" if _plt() else "e2e: (matplotlib absent)", (len(res["figures"]) >= 5) if _plt() else True)
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            traceback.print_exc()
            check("e2e run with uavlink_sim", False)
    else:
        print("  [skip] end-to-end test (uavlink_sim.py not next to this file; pass --sim-module)")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog="uavlink_eval", description="Evaluation, statistics and visualisation for multi-link UAV handover experiments.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config"); p.add_argument("--out", default="eval_config.json")
    p = sub.add_parser("evaluate")
    p.add_argument("--batch"); p.add_argument("--runs-dir"); p.add_argument("--config"); p.add_argument("--out", default="eval_out")
    p.add_argument("--resim", action="store_true", help="re-simulate (bit-exact) runs that have no tick logs")
    p.add_argument("--sim-module", help="path to uavlink_sim.py (default: import uavlink_sim)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--holdout", type=float, default=None, help="fraction of trials used to choose the sweep variant")
    p.add_argument("--set", action="append", default=[], help="override config, e.g. stats.alpha=0.01 or definitions.deadline_ms=100")
    p.add_argument("--no-plots", action="store_true")
    p = sub.add_parser("timeline"); p.add_argument("--run", required=True); p.add_argument("--config"); p.add_argument("--out")
    p = sub.add_parser("selftest"); p.add_argument("--sim-module")
    a = ap.parse_args(argv)
    if a.cmd == "make-config":
        json.dump(dataclasses.asdict(EvalConfig()), open(a.out, "w"), indent=1, default=_jdefault)
        print("wrote", a.out)
    elif a.cmd == "evaluate":
        ov: dict = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            try:
                v = json.loads(v)
            except Exception:
                pass
            d = ov
            ks = k.split(".")
            for x in ks[:-1]:
                d = d.setdefault(x, {})
            d[ks[-1]] = v
        if a.no_plots:
            ov.setdefault("plots", {})["enabled"] = False
        cfg = load_config(a.config, ov)
        if not (a.batch or a.runs_dir):
            raise SystemExit("give --batch DIR or --runs-dir DIR")
        res = evaluate(a.batch, a.runs_dir, cfg, a.out, a.resim, a.workers, a.sim_module, a.holdout)
        T = res["tables"]
        print("\n" + md_table(*T["headline"]))
        if T["paired_primary"][1]:
            print("\n" + md_table(*T["paired_primary"]))
        print("\n" + md_table(*T["goals"]) if T["goals"][1] else "")
        print(f"\nreport: {os.path.join(a.out, 'report.html')}  (+ report.md, tables, figures, analysis.json)")
    elif a.cmd == "timeline":
        cfg = load_config(a.config)
        out = a.out or a.run
        tk = load_ticks(os.path.join(a.run, "ticks.csv.gz"))
        print("wrote", os.path.join(out, fig_timeline_run(tk, cfg, out, "timeline_eval", os.path.basename(a.run.rstrip('/')))))
    elif a.cmd == "selftest":
        sys.exit(0 if selftest(a.sim_module) else 1)


if __name__ == "__main__":
    main()
