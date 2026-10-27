#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictive_handover.py  --  Predictive multi-link handover intelligence for a drone
====================================================================================

Standalone (stdlib only, Python >= 3.8).  Continuously ingests per-link RSSI/SNR,
drone state and recent history; forecasts the probability that each link becomes
unusable over a controllable horizon; and decides

  * which link carries high-priority control/telemetry (primary),
  * whether/where critical control packets are duplicated,
  * which link carries video, at what bitrate, plus a reduced-rate video fallback,
  * which standby links are kept warm (power vs. readiness),
  * the exact timing and mode (soft = make-before-break, hard = break-before-make)
    of every primary-link switch,
  * an advisory flight hint when every link is forecast to fail.

Layers (each replaceable)
-------------------------
  OutagePredictor   HybridPredictor: Kalman SNR trend + online path-loss model driven by the
                    forecast trajectory (+ optional shadow map) + online error corrector
  RiskModel         correlated (one-factor + band-group factor) joint-outage probabilities,
                    closed form for "all of these links are down" events
  DecisionPolicy    CostFunctionPolicy (main contribution), ThresholdPolicy, RuleBasedPolicy,
                    ReactivePolicy / BondingPolicy (baselines), CallablePolicy (plug anything)
  HandoverController  state machine: warm-up, dwell/hysteresis, TTO-based switch timing,
                    soft/hard execution, emergency fast path, flight advisory

All numbers in HandoverConfig/LinkSpec are *defaults and assumptions*, not measured values.
Calibrate them on your own radios before drawing conclusions.

Usage (library)
---------------
    ctrl = HandoverController(specs, HandoverConfig(), CostFunctionPolicy())
    ctrl.set_mission(waypoints_enu, speed_mps)             # optional
    ctrl.set_shadow_map("link5g8", fn(pos)->extra_loss_db) # optional
    ctrl.on_link_sample("link5g8", LinkSample(t, rssi_dbm=-80, snr_db=12))
    ctrl.on_drone_state(DroneState(t, pos, vel))
    decision = ctrl.step(t)          # call at 10-50 Hz; policy runs every cfg.decision_period_s

CLI:  python predictive_handover.py bench --seeds 3      (simulated blind-time benchmark)
      python predictive_handover.py dump-config           (all tunables as JSON)
"""
from __future__ import annotations

import abc
import collections
import dataclasses
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
INF = math.inf

# ====================================================================================
# 0. small math helpers
# ====================================================================================
_ND = statistics.NormalDist()


def phi(x: float) -> float:
    return 0.5 * math.erfc(-x / 1.4142135623730951)


def phi_inv(p: float) -> float:
    return _ND.inv_cdf(min(max(p, 1e-9), 1 - 1e-9))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _mul(a, s): return tuple(x * s for x in a)
def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _norm(a): return math.sqrt(_dot(a, a))


# Gauss-Hermite nodes/weights for the standard normal (probabilists')
GH5_N = (-2.8569700138728056, -1.3556261799742659, 0.0, 1.3556261799742659, 2.8569700138728056)
GH5_W = (0.011257411327720691, 0.2220759220056127, 0.5333333333333333,
         0.2220759220056127, 0.011257411327720691)
GH3_N = (-1.7320508075688772, 0.0, 1.7320508075688772)
GH3_W = (1 / 6, 2 / 3, 1 / 6)


def make_tau_grid(horizon_s: float, n: int) -> List[float]:
    """Prediction grid, denser near 'now'."""
    return [horizon_s * ((k / n) ** 1.6) for k in range(n + 1)]


# ====================================================================================
# 1. configuration (everything tunable lives here)
# ====================================================================================
@dataclass
class LinkSpec:
    """Static description of one radio link (all values are per-deployment assumptions)."""
    name: str
    band_group: str = "default"            # links in the same group share a fading/interference factor
    endpoint: Vec3 = (0.0, 0.0, 0.0)       # ground antenna / tower position, local ENU metres
    noise_floor_dbm: float = -95.0
    prior_rssi_1m_dbm: float = -20.0       # path-loss model prior: RSSI = P0 - 10 n log10(d)
    prior_path_loss_exp: float = 2.4
    # SNR -> rate table: ((min_snr_db, rate_mbps), ...) ascending; stepwise MCS capacity
    mcs_table: Tuple[Tuple[float, float], ...] = ((4.0, 1.5), (8.0, 6.0), (12.0, 14.0), (16.0, 24.0), (20.0, 35.0))
    ctrl_snr_min_db: float = 3.0           # control-grade usability threshold (robust mode)
    ctrl_per50_offset_db: float = 1.5      # control PER = 50 % at ctrl_snr_min - offset
    per_slope: float = 1.2                 # logistic slope of PER vs SNR (1/dB)
    base_latency_ms: float = 8.0           # one-way control latency, no retransmission
    retx_interval_ms: float = 10.0
    mtu_bytes: int = 1200                  # video packet size (head-of-line blocking of control)
    supports_video: bool = True
    active_power_w: float = 2.5            # radio power when carrying traffic
    power_w_per_mbps: float = 0.15
    idle_power_w: float = 0.5              # warm standby power
    acquire_time_s: float = 0.0            # cold -> ready time
    always_warm: bool = True
    airtime_budget: float = 0.7            # max utilisation this link may be loaded to
    spectrum_cost: float = 1.0             # relative scarcity of this band (cost weight)

    def capacity_mbps(self, snr_db: float) -> float:
        cap = 0.0
        for th, r in self.mcs_table:
            if snr_db >= th:
                cap = r
        return cap

    def snr_needed_db(self, rate_mbps: float) -> float:
        if rate_mbps <= 0:
            return -INF
        for th, r in self.mcs_table:
            if r >= rate_mbps:
                return th
        return INF

    def ctrl_per(self, snr_db: float) -> float:
        x = self.per_slope * (snr_db - (self.ctrl_snr_min_db - self.ctrl_per50_offset_db))
        return 1.0 / (1.0 + math.exp(clamp(x, -60.0, 60.0)))

    def ctrl_deadline_prob(self, snr_db: float, deadline_ms: float, util: float = 0.0) -> float:
        """P(control packet delivered within deadline) with ARQ and head-of-line blocking by video."""
        per = self.ctrl_per(snr_db)
        low_rate = self.mcs_table[0][1]
        hol = min(200.0, self.mtu_bytes * 8 / (max(self.capacity_mbps(snr_db), low_rate) * 1e3))
        lat0 = self.base_latency_ms + clamp(util, 0.0, 1.0) * hol
        if deadline_ms < lat0:
            return 0.0
        n = int((deadline_ms - lat0) // self.retx_interval_ms) + 1
        return 1.0 - per ** n


@dataclass
class TrafficProfile:
    ctrl_kbps: float = 20.0
    telemetry_kbps: float = 40.0
    dup_fraction: float = 1.0                       # share of control stream duplicated (<1 = critical classes only)
    duplicate_classes: Tuple[str, ...] = ("failsafe", "arm_disarm", "rth", "heartbeat")
    ctrl_deadline_hard_ms: float = 120.0            # hard limit L_max (e.g. 100-150 ms)
    ctrl_deadline_soft_ms: float = 60.0             # preferred latency
    ctrl_reliability_target: float = 0.99           # require P(latency <= hard limit) >= this
    video_ladder_kbps: Tuple[float, ...] = (0, 300, 800, 1500, 3000, 6000)   # index 0 = video off
    video_min_level: int = 1                        # reduced-video fallback level
    video_cap_margin: float = 1.3                   # capacity must exceed bitrate by this factor
    video_u0_kbps: float = 200.0                    # utility knee: u(b)=log(1+b/u0)/log(1+bmax/u0)

    def video_utility(self, level: int) -> float:
        if level <= 0:
            return 0.0
        b = self.video_ladder_kbps[level]
        bmax = self.video_ladder_kbps[-1]
        return math.log1p(b / self.video_u0_kbps) / math.log1p(bmax / self.video_u0_kbps)


@dataclass
class CostWeights:
    blind: float = 100.0            # horizon-integrated P(no control AND no video)
    ctrl_loss: float = 30.0         # horizon-integrated P(no control)
    ctrl_latency: float = 10.0      # P(miss soft latency target) at look-ahead
    video_quality: float = 1.0      # (1 - expected normalised video quality)
    power: float = 0.05             # per watt
    spectrum: float = 0.3           # per unit weighted airtime
    duplication: float = 0.3        # per duplicated control path
    switch_soft: float = 0.5
    switch_hard: float = 3.0
    video_change: float = 0.05
    chance_violation: float = 200.0  # penalty slope when max P(ctrl loss) exceeds ctrl_fail_max


@dataclass
class SafetyMargins:
    snr_margin_db: float = 3.0          # added to every usability threshold
    sigma_inflation: float = 1.2        # multiplies forecast std (conservatism)
    horizon_s: float = 8.0
    horizon_points: int = 7
    risk_discount_s: float = 5.0        # nearer risk weighted more
    ctrl_fail_max: float = 0.02         # chance constraint: max_k P(all control links down)
    min_dwell_s: float = 2.0            # min time between primary-link changes (non-urgent)
    switch_hysteresis: float = 0.5      # cost improvement needed to leave current assignment
    switch_risk_quantile: float = 0.05  # alpha: time-to-outage defined as first time P_fail >= alpha
    lead_margin_s: float = 0.3
    soft_overlap_s: float = 0.6         # duplicate-on-both window of a soft switch
    hard_blackout_s: float = 0.35       # control gap of a hard switch (re-association etc.)
    stale_timeout_s: float = 0.6        # no sample for this long => link considered dead
    emergency_detect_s: float = 0.25    # measured SNR below ctrl threshold this long => failover
    enable_emergency: bool = True
    max_dups: int = 2                   # max duplicate control paths besides primary
    warm_trigger_prob: float = 0.02
    warm_lead_s: float = 3.0
    cool_hold_s: float = 5.0
    power_budget_w: float = 12.0        # radio power budget (all links)
    video_up_hold_s: float = 3.0        # bitrate may rise only after this long at current level
    video_min_dwell_s: float = 3.0       # min time between video link changes while the current video link is healthy
    rho_global: float = 0.10            # common-factor correlation of link outages
    rho_group: float = 0.35             # extra correlation inside a band group
    latency_check_tau_s: float = 2.0
    latency_quantile_z: float = 1.64
    advisory_prob: float = 0.5          # all-links-fail probability that triggers flight advisory
    enable_flight_advisory: bool = True


@dataclass
class PredictorConfig:
    use_kalman: bool = True
    use_geometry: bool = True
    use_learning: bool = True
    history_len: int = 300
    kalman_q: float = 1.0               # slope process noise (dB/s)^2 per s
    kalman_r_db: float = 2.0
    slope_damping_s: float = 3.0
    rls_forgetting: float = 0.999
    rls_p0_var: float = 25.0
    rls_n_var: float = 0.5
    shadow_sigma_prior_db: float = 4.0
    shadow_corr_dist_m: float = 60.0
    resid_smooth_s: float = 0.7
    fast_sigma_prior_db: float = 2.0
    noise_decay_s: float = 3.0
    geo_blend_tau_s: float = 2.5
    geo_min_samples: int = 30
    sigma_floor_db: float = 0.7
    prior_sigma_db: float = 8.0
    map_uncertainty_frac: float = 0.25
    learn_rate: float = 0.05
    learn_bias_limit_db: float = 8.0
    learn_scale_range: Tuple[float, float] = (0.8, 2.5)
    traj_accel_decay_s: float = 1.5
    traj_sigma0_m: float = 3.0
    traj_sigma_acc: float = 1.5
    mission_weight: float = 0.7
    # step-change detector + learned hazard of abrupt (unforecastable) outages, e.g. interference bursts
    use_hazard: bool = True
    gate_sigma: float = 3.5             # innovation gate (in sqrt(S)) plus gate_floor_db
    gate_floor_db: float = 6.0
    abrupt_drop_db: float = 8.0         # a gated drop at least this large counts as an abrupt-outage event
    event_recover_db: float = 6.0       # event ends when fast SNR is back within this of pre-event level
    event_max_s: float = 20.0
    hazard_prior_rate_per_s: float = 1.0 / 120.0
    hazard_prior_weight_s: float = 120.0
    hazard_prior_dur_s: float = 2.0
    hazard_prior_dur_weight: float = 1.0
    hazard_max: float = 0.5


@dataclass
class HandoverConfig:
    traffic: TrafficProfile = field(default_factory=TrafficProfile)
    weights: CostWeights = field(default_factory=CostWeights)
    safety: SafetyMargins = field(default_factory=SafetyMargins)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    decision_period_s: float = 0.3

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HandoverConfig":
        def mk(klass, sub):
            sub = sub or {}
            kw = {}
            for f in dataclasses.fields(klass):
                if f.name in sub:
                    v = sub[f.name]
                    kw[f.name] = tuple(v) if isinstance(v, list) else v
            return klass(**kw)
        return cls(traffic=mk(TrafficProfile, d.get("traffic")), weights=mk(CostWeights, d.get("weights")),
                   safety=mk(SafetyMargins, d.get("safety")), predictor=mk(PredictorConfig, d.get("predictor")),
                   decision_period_s=d.get("decision_period_s", 0.3))

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "HandoverConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ====================================================================================
# 2. data types
# ====================================================================================
@dataclass
class LinkSample:
    t: float
    rssi_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    noise_dbm: Optional[float] = None
    loss_ratio: Optional[float] = None   # carried for custom predictors
    rtt_ms: Optional[float] = None       # carried for custom predictors
    connected: bool = True


@dataclass
class DroneState:
    t: float
    pos: Vec3
    vel: Vec3 = (0.0, 0.0, 0.0)
    acc: Vec3 = (0.0, 0.0, 0.0)
    battery_frac: float = 1.0


@dataclass(frozen=True)
class Assignment:
    control_primary: Optional[str] = None
    control_dups: Tuple[str, ...] = ()          # links carrying duplicate control stream
    video_primary: Optional[str] = None
    video_fallback: Optional[str] = None        # carries reduced-rate duplicate stream
    video_level: int = 0                        # index into video ladder (0 = off)
    video_bonded: Tuple[str, ...] = ()          # striped video (used by the bonding baseline)
    keep_warm: Tuple[str, ...] = ()

    def control_links(self) -> Tuple[str, ...]:
        return tuple(x for x in ((self.control_primary,) + self.control_dups) if x)

    def active_links(self) -> set:
        s = set(self.control_links()) | set(self.video_bonded)
        if self.video_level > 0 or self.video_fallback:
            for x in (self.video_primary, self.video_fallback):
                if x:
                    s.add(x)
        return s


@dataclass
class LinkForecast:
    link: str
    taus: List[float]
    mu: List[float]            # predicted SNR mean (dB)
    sigma: List[float]         # predicted SNR std (dB)
    alive: bool = True
    haz: float = 0.0           # stationary P(link inside an abrupt outage) not visible to the trend/geometry model

    def p_below(self, k: int, thr_db: float) -> float:
        if thr_db == INF:
            return 1.0
        if thr_db == -INF:
            return 0.0
        p = phi((thr_db - self.mu[k]) / self.sigma[k])
        if k > 0 and self.haz > 0.0:
            p = 1.0 - (1.0 - p) * (1.0 - self.haz)
        return p

    def interp(self, tau: float) -> Tuple[float, float]:
        ts = self.taus
        if tau <= ts[0]:
            return self.mu[0], self.sigma[0]
        for k in range(1, len(ts)):
            if tau <= ts[k]:
                f = (tau - ts[k - 1]) / max(ts[k] - ts[k - 1], 1e-9)
                return (self.mu[k - 1] + f * (self.mu[k] - self.mu[k - 1]),
                        self.sigma[k - 1] + f * (self.sigma[k] - self.sigma[k - 1]))
        return self.mu[-1], self.sigma[-1]


@dataclass
class Proposal:
    """What a policy returns. Timing fields left None are filled by the controller's scheduler."""
    target: Assignment
    mode: Optional[str] = None          # 'soft' | 'hard' | None (auto)
    execute_at: Optional[float] = None
    alpha: Optional[float] = None       # time-to-outage risk quantile override
    urgent: bool = False                # bypass dwell/hysteresis
    reason: str = ""
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledAction:
    execute_at: float
    mode: str
    target: Assignment
    reason: str = ""


@dataclass
class Decision:
    t: float
    assignment: Assignment               # effective NOW (includes soft-switch overlap duplicates)
    control_blackout_until: float
    scheduled: List[ScheduledAction]
    warm_links: Tuple[str, ...]
    flight_advisory: Optional[dict]
    reasons: List[str]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ====================================================================================
# 3. trajectory forecaster
# ====================================================================================
class TrajectoryForecaster:
    """Constant-velocity + decaying-acceleration forecast, optionally blended with mission path."""

    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.wps: Optional[List[Vec3]] = None

    def set_mission(self, waypoints: Sequence[Vec3]):
        self.wps = [tuple(w) for w in waypoints] if waypoints and len(waypoints) >= 2 else None

    def _project(self, pos):
        best = (1e18, 0, 0.0)
        w = self.wps
        for i in range(len(w) - 1):
            a, b = w[i], w[i + 1]
            ab = _sub(b, a)
            l2 = _dot(ab, ab)
            u = 0.0 if l2 == 0 else clamp(_dot(_sub(pos, a), ab) / l2, 0.0, 1.0)
            d = dist(pos, _add(a, _mul(ab, u)))
            if d < best[0]:
                best = (d, i, u)
        return best

    def _advance(self, i, u, s):
        w = self.wps
        while True:
            a, b = w[i], w[i + 1]
            L = dist(a, b)
            rem = L * (1 - u)
            if s <= rem or i >= len(w) - 2:
                uu = min(u + (s / L if L > 0 else 0.0), 1.0)
                return _add(a, _mul(_sub(b, a), uu))
            s -= rem
            i += 1
            u = 0.0

    def predict(self, st: DroneState, taus: Sequence[float]) -> List[Tuple[Vec3, float]]:
        c = self.cfg
        speed = _norm(st.vel)
        proj = self._project(st.pos) if self.wps else None
        out = []
        for tau in taus:
            ta = c.traj_accel_decay_s
            disp = _add(_mul(st.vel, tau), _mul(st.acc, ta * (tau - ta * (1 - math.exp(-tau / ta)))))
            p = _add(st.pos, disp)
            sig = c.traj_sigma0_m + 0.5 * c.traj_sigma_acc * tau * tau
            if proj is not None:
                off, i, u = proj
                pm = self._advance(i, u, speed * tau)
                wgt = c.mission_weight * math.exp(-off / 50.0)
                p = _add(_mul(p, 1 - wgt), _mul(pm, wgt))
                sig *= (1 - 0.5 * wgt)
            out.append((p, sig))
        return out


# ====================================================================================
# 4. predictor
# ====================================================================================
class OutagePredictor(abc.ABC):
    """Replace this to plug a different forecaster (e.g. a trained network)."""

    @abc.abstractmethod
    def update(self, link: str, sample: LinkSample, pos: Vec3) -> None: ...

    @abc.abstractmethod
    def forecast(self, link: str, t: float, traj: List[Tuple[Vec3, float]], taus: List[float],
                 speed: float, register: bool = True) -> LinkForecast: ...

    @abc.abstractmethod
    def measured(self, link: str) -> Dict[str, Any]: ...

    def set_shadow_map(self, link: str, fn: Optional[Callable[[Vec3], float]]) -> None:
        pass


class ErrorCorrector:
    """Online (NLMS) bias + variance calibration of the physics forecast, per horizon index."""
    NF = 4

    def __init__(self, K: int, cfg: PredictorConfig):
        self.cfg = cfg
        self.w = [[0.0] * self.NF for _ in range(K)]
        self.s2 = [1.0] * K
        self.pending: List[tuple] = []

    def bias(self, k, f):
        b = sum(wi * fi for wi, fi in zip(self.w[k], f))
        lim = self.cfg.learn_bias_limit_db
        return clamp(b, -lim, lim)

    def scale(self, k):
        lo, hi = self.cfg.learn_scale_range
        return clamp(math.sqrt(self.s2[k]), lo, hi)

    def register(self, due, k, f, mu, sg):
        self.pending.append((due, k, f, mu, sg))
        if len(self.pending) > 600:
            self.pending = self.pending[-400:]

    def resolve(self, t, snr):
        keep = []
        for rec in self.pending:
            due, k, f, mu, sg = rec
            if due > t:
                keep.append(rec)
                continue
            if t - due > 0.3:
                continue
            err = snr - mu
            pb = sum(wi * fi for wi, fi in zip(self.w[k], f))
            d = err - pb
            nrm = 1.0 + sum(fi * fi for fi in f)
            lr = self.cfg.learn_rate
            self.w[k] = [wi + lr * d * fi / nrm for wi, fi in zip(self.w[k], f)]
            z = (snr - (mu + clamp(pb, -20, 20))) / max(sg, 0.3)
            self.s2[k] = 0.98 * self.s2[k] + 0.02 * min(z * z, 25.0)
        self.pending = keep


class LinkTracker:
    def __init__(self, spec: LinkSpec, cfg: PredictorConfig, stale_s: float):
        self.spec, self.cfg, self.stale_s = spec, cfg, stale_s
        self.n = 0
        self.last_t: Optional[float] = None
        self.connected = True
        self.kx: Optional[List[float]] = None            # [snr, slope]
        self.kP = [4.0, 0.0, 1.0]                        # p00, p01, p11
        self.th = [spec.prior_rssi_1m_dbm, spec.prior_path_loss_exp]
        self.RP = [cfg.rls_p0_var, 0.0, cfg.rls_n_var]   # p00, p01, p11
        self.resid = 0.0
        self.shadow_var = cfg.shadow_sigma_prior_db ** 2
        self.fast_var = cfg.fast_sigma_prior_db ** 2
        self.noise = spec.noise_floor_dbm
        self.noise_base = spec.noise_floor_dbm
        self.raw_snr: Optional[float] = None
        self.last_d: Optional[float] = None
        self.hist: collections.deque = collections.deque(maxlen=cfg.history_len)
        self.recent: collections.deque = collections.deque(maxlen=3)
        self._gate = 0
        self._ev_start: Optional[float] = None
        self._ev_ref = 0.0
        self.h_n, self.h_T, self.h_dsum, self.h_k = 0.0, 0.0, 0.0, 0.0

    def fast_snr(self) -> Optional[float]:
        if not self.recent:
            return None
        v = sorted(self.recent)
        return v[len(v) // 2]

    def hazard_floor(self) -> float:
        c = self.cfg
        if not c.use_hazard:
            return 0.0
        a0 = c.hazard_prior_rate_per_s * c.hazard_prior_weight_s
        rate = (a0 + self.h_n) / (c.hazard_prior_weight_s + self.h_T)
        dbar = (c.hazard_prior_dur_s * c.hazard_prior_dur_weight + self.h_dsum) / (c.hazard_prior_dur_weight + self.h_k)
        x = rate * dbar
        return clamp(x / (1.0 + x), 0.0, c.hazard_max)

    # ---------------------------------------------------------------- update
    def update(self, s: LinkSample, pos: Vec3, smap) -> None:
        cfg, sp = self.cfg, self.spec
        if not s.connected:
            self.connected = False
            self.last_t = s.t
            return
        self.connected = True
        noise = s.noise_dbm
        if s.snr_db is not None and s.rssi_dbm is not None:
            snr, rssi = s.snr_db, s.rssi_dbm
            noise = noise if noise is not None else rssi - snr
        elif s.snr_db is not None:
            snr = s.snr_db
            noise = noise if noise is not None else self.noise
            rssi = snr + noise
        elif s.rssi_dbm is not None:
            rssi = s.rssi_dbm
            noise = noise if noise is not None else self.noise
            snr = rssi - noise
        else:
            return
        d = max(dist(pos, sp.endpoint), 1.0)
        t = s.t
        dt = clamp(t - self.last_t, 1e-3, 2.0) if self.last_t is not None else None
        self.raw_snr = snr
        self.recent.append(snr)
        if self._ev_start is not None:
            fs = self.fast_snr()
            if (fs is not None and fs >= self._ev_ref - cfg.event_recover_db and len(self.recent) >= 3) \
                    or t - self._ev_start > cfg.event_max_s:
                self.h_dsum += t - self._ev_start
                self.h_k += 1.0
                self._ev_start = None
        # noise floor: fast attack, slow release; baseline = slow minimum
        if noise > self.noise:
            self.noise += 0.6 * (noise - self.noise)
        else:
            self.noise += 0.05 * (noise - self.noise)
        self.noise_base = min(self.noise_base + 0.002 * (self.noise - self.noise_base), self.noise) \
            if self.noise > self.noise_base else self.noise
        # --- Kalman (state: snr, slope)
        R = cfg.kalman_r_db ** 2
        if self.kx is None or dt is None:
            self.kx = [snr, 0.0]
            self.kP = [R, 0.0, 1.0]
        else:
            dec = math.exp(-dt / cfg.slope_damping_s)
            x0 = self.kx[0] + self.kx[1] * dt
            x1 = self.kx[1] * dec
            p00, p01, p11 = self.kP
            q = cfg.kalman_q
            n00 = p00 + 2 * dt * p01 + dt * dt * p11 + q * dt ** 3 / 3
            n01 = (p01 + dt * p11) * dec + q * dt * dt / 2 * dec
            n11 = p11 * dec * dec + q * dt
            y = snr - x0
            S = n00 + R
            self.h_T += dt
            if abs(y) > cfg.gate_sigma * math.sqrt(S) + cfg.gate_floor_db:
                self._gate += 1
            else:
                self._gate = 0
            if self._gate >= 2:                              # step change: re-initialise the trend filter
                new0 = 0.5 * (snr + self.recent[-1]) if self.recent else snr
                if new0 < self.kx[0] - cfg.abrupt_drop_db and self._ev_start is None:
                    self._ev_start, self._ev_ref = t, self.kx[0]
                    self.h_n += 1.0
                self.kx = [new0, 0.0]
                self.kP = [R, 0.0, 1.0]
                self._gate = 0
            else:
                k0, k1 = n00 / S, n01 / S
                self.kx = [x0 + k0 * y, x1 + k1 * y]
                self.kP = [(1 - k0) * n00, (1 - k0) * n01, n11 - k1 * n01]
                a = 0.05
                self.fast_var = clamp((1 - a) * self.fast_var + a * y * y, 1.0, 36.0)
        # --- path-loss RLS (map-corrected observations)
        ph0, ph1 = 1.0, -10.0 * math.log10(d)
        yv = rssi + (smap(pos) if smap else 0.0)
        e = yv - (self.th[0] * ph0 + self.th[1] * ph1)
        lam = cfg.rls_forgetting
        P00, P01, P11 = self.RP
        Pp0, Pp1 = P00 * ph0 + P01 * ph1, P01 * ph0 + P11 * ph1
        den = lam + ph0 * Pp0 + ph1 * Pp1
        k0, k1 = Pp0 / den, Pp1 / den
        self.th[0] += k0 * e
        self.th[1] = clamp(self.th[1] + k1 * e, 1.5, 6.0)
        P00n, P01n, P11n = (P00 - k0 * Pp0) / lam, (P01 - k0 * Pp1) / lam, (P11 - k1 * Pp1) / lam
        self.RP = [min(P00n, cfg.rls_p0_var), clamp(P01n, -5.0, 5.0), min(P11n, cfg.rls_n_var)]
        e_post = yv - (self.th[0] * ph0 + self.th[1] * ph1)
        if dt is not None:
            a = 1 - math.exp(-dt / cfg.resid_smooth_s)
            self.resid += a * (e_post - self.resid)
            b = 1 - math.exp(-dt / 30.0)
            self.shadow_var = max(1.0, self.shadow_var + b * (self.resid ** 2 - self.shadow_var))
        self.last_t = t
        self.last_d = d
        self.n += 1
        self.hist.append((t, rssi, snr, d))

    # ---------------------------------------------------------------- forecast
    def forecast(self, t, taus, traj, smap, speed):
        cfg, sp = self.cfg, self.spec
        K = len(taus)
        aux = {"alive": True, "f": [1.0, 0.0, 0.0, 0.0]}
        if self.n == 0 and self.last_t is None:                      # never heard: prior geometry only
            mu, sg = [], []
            for pos, _ in traj:
                d = max(dist(pos, sp.endpoint), 1.0)
                rssi = self.th[0] - self.th[1] * 10 * math.log10(d) - (smap(pos) if smap else 0.0)
                mu.append(rssi - self.noise)
                sg.append(cfg.prior_sigma_db)
            aux["alive"] = False
            return mu, sg, aux
        if (not self.connected) or (t - self.last_t > self.stale_s) or self.kx is None:
            aux["alive"] = False
            return [-30.0] * K, [0.5] * K, aux
        tc = max(1.0, cfg.shadow_corr_dist_m / max(speed, 1.0))
        slope = self.kx[1] if cfg.use_kalman else 0.0
        geo_conf = clamp(self.n / cfg.geo_min_samples, 0.0, 1.0) if cfg.use_geometry else 0.0
        pos0 = traj[0][0]
        d0 = max(dist(pos0, sp.endpoint), 1.0)
        map0 = smap(pos0) if smap else 0.0
        P00, P01, P11 = self.kP
        q_eff = cfg.kalman_q * 0.3
        mu, sg = [], []
        d_far = d0
        for k, tau in enumerate(taus):
            pos, spos = traj[k]
            g = cfg.slope_damping_s * (1 - math.exp(-tau / cfg.slope_damping_s)) if cfg.use_kalman else 0.0
            kal = self.kx[0] + slope * g
            kvar = P00 + 2 * g * P01 + g * g * P11 + q_eff * tau + self.fast_var
            if geo_conf > 0 and tau > 0:
                d = max(dist(pos, sp.endpoint), 1.0)
                if tau >= 1.0 and d_far == d0:
                    d_far = d
                mapk = smap(pos) if smap else 0.0
                rssi_g = self.th[0] - self.th[1] * 10 * math.log10(d) - mapk + self.resid * math.exp(-tau / tc)
                nk = self.noise_base + (self.noise - self.noise_base) * math.exp(-tau / cfg.noise_decay_s)
                geo = rssi_g - nk
                ph = (1.0, -10 * math.log10(d))
                par = self.RP[0] * ph[0] ** 2 + 2 * self.RP[1] * ph[0] * ph[1] + self.RP[2] * ph[1] ** 2
                dslope = 10 * self.th[1] / (d * math.log(10))
                gvar = (self.shadow_var * (1 - math.exp(-2 * tau / tc)) + (dslope * spos) ** 2
                        + self.fast_var + par + (cfg.map_uncertainty_frac * (mapk - map0)) ** 2)
                w = geo_conf * (1 - math.exp(-tau / cfg.geo_blend_tau_s))
                m = (1 - w) * kal + w * geo
                v = (1 - w) * kvar + w * gvar + w * (1 - w) * (kal - geo) ** 2
            else:
                m, v = kal, kvar
            mu.append(m)
            sg.append(max(math.sqrt(max(0.0, v)), cfg.sigma_floor_db))
        ddot = clamp((d_far - d0) / 1.0, -30.0, 30.0)
        aux["f"] = [1.0, clamp(slope, -5, 5) / 5.0, clamp(self.resid, -15, 15) / 15.0, ddot / 30.0]
        return mu, sg, aux


class HybridPredictor(OutagePredictor):
    """Kalman trend (short term) + geometry/path-loss model (long term) + online error corrector.
    Ablations: set PredictorConfig.use_kalman / use_geometry / use_learning to False."""

    def __init__(self, specs: Dict[str, LinkSpec], cfg: PredictorConfig, safety: SafetyMargins):
        self.cfg, self.safety = cfg, safety
        self.trackers = {n: LinkTracker(s, cfg, safety.stale_timeout_s) for n, s in specs.items()}
        self.maps: Dict[str, Callable] = {}
        self.correctors: Dict[str, ErrorCorrector] = {}

    def set_shadow_map(self, link, fn):
        if fn is None:
            self.maps.pop(link, None)
        else:
            self.maps[link] = fn

    def update(self, link, sample, pos):
        tr = self.trackers[link]
        tr.update(sample, pos, self.maps.get(link))
        c = self.correctors.get(link)
        if c is not None and sample.connected and tr.raw_snr is not None:
            c.resolve(sample.t, tr.raw_snr)

    def forecast(self, link, t, traj, taus, speed, register=True):
        tr = self.trackers[link]
        mu, sg, aux = tr.forecast(t, taus, traj, self.maps.get(link), speed)
        K = len(taus)
        if self.cfg.use_learning and aux["alive"]:
            cor = self.correctors.get(link)
            if cor is None or len(cor.w) != K:
                cor = self.correctors[link] = ErrorCorrector(K, self.cfg)
            f = aux["f"]
            for k in range(1, K):
                if register:
                    cor.register(t + taus[k], k, f, mu[k], sg[k])
                mu[k] += cor.bias(k, f)
                sg[k] *= cor.scale(k)
        infl = self.safety.sigma_inflation
        sg = [max(s * infl, self.cfg.sigma_floor_db) for s in sg]
        return LinkForecast(link, list(taus), mu, sg, aux["alive"], tr.hazard_floor() if aux["alive"] else 0.0)

    def measured(self, link):
        tr = self.trackers[link]
        return {"snr": tr.kx[0] if tr.kx else None, "raw": tr.raw_snr, "fast": tr.fast_snr(), "last_t": tr.last_t,
                "hazard": tr.hazard_floor(), "in_event": tr._ev_start is not None,
                "has_data": tr.n > 0, "connected": tr.connected, "slope": tr.kx[1] if tr.kx else 0.0}


# ====================================================================================
# 5. correlated risk model
# ====================================================================================
class RiskModel:
    """P(link i unusable) from forecasts + joint 'all-of-these-down' probabilities with correlation.

    z_i = sqrt(r0) Z0 + sqrt(rg) Zgroup(i) + sqrt(1-r0-rg) e_i ; link i is down iff z_i < Phi^-1(p_i).
    'All links in S down' is a conjunction, so it is E_Z[ prod_i p_i(Z) ] (closed form per factor value).
    Control lost = all control links down; video lost = primary AND fallback down; blind = both =
    conjunction over the union of those links with per-link threshold = the lowest one that applies."""

    def __init__(self, specs, forecasts: Dict[str, LinkForecast], taus, safety: SafetyMargins):
        self.specs, self.fc, self.taus = specs, forecasts, list(taus)
        r0, rg = safety.rho_global, safety.rho_group
        assert r0 + rg < 0.99
        self.a, self.b = math.sqrt(r0), math.sqrt(rg)
        self.s_marg, self.s_full = math.sqrt(1 - r0), math.sqrt(1 - r0 - rg)
        K = len(self.taus)
        raw = []
        for k in range(K):
            lo = self.taus[max(k - 1, 0)]
            hi = self.taus[min(k + 1, K - 1)]
            raw.append(max(hi - lo, 1e-3) * 0.5 * math.exp(-self.taus[k] / safety.risk_discount_s))
        s = sum(raw)
        self.wk = [x / s for x in raw]
        self._c0: Dict[tuple, tuple] = {}
        self._cj: Dict[tuple, float] = {}

    def p(self, link, thr, k):
        return self.fc[link].p_below(k, thr)

    def _c(self, link, thr, k):
        return phi_inv(self.p(link, thr, k))

    def _cond0(self, link, thr, k):
        key = (link, thr, k)
        v = self._c0.get(key)
        if v is None:
            c = self._c(link, thr, k)
            v = tuple(phi((c - self.a * z) / self.s_marg) for z in GH5_N)
            self._c0[key] = v
        return v

    def conj(self, spec: tuple, k: int) -> float:
        key = (spec, k)
        r = self._cj.get(key)
        if r is not None:
            return r
        groups: Dict[str, list] = {}
        for l, thr in spec:
            groups.setdefault(self.specs[l].band_group, []).append((l, thr))
        vecs = []
        for items in groups.values():
            if len(items) == 1:
                vecs.append(self._cond0(items[0][0], items[0][1], k))
            else:
                cs = [self._c(l, thr, k) for l, thr in items]
                v = []
                for z in GH5_N:
                    acc = 0.0
                    for zg, wg in zip(GH3_N, GH3_W):
                        pr = 1.0
                        for c in cs:
                            pr *= phi((c - self.a * z - self.b * zg) / self.s_full)
                        acc += wg * pr
                    v.append(acc)
                vecs.append(v)
        res = 0.0
        for i, w in enumerate(GH5_W):
            pr = w
            for v in vecs:
                pr *= v[i]
            res += pr
        self._cj[key] = res
        return res

    def integral(self, spec: tuple) -> float:
        return sum(self.wk[k] * self.conj(spec, k) for k in range(len(self.taus)))

    def maximum(self, spec: tuple) -> float:
        return max(self.conj(spec, k) for k in range(len(self.taus)))

    def time_to_risk(self, link: str, thr: float, alpha: float) -> Optional[float]:
        for k, tau in enumerate(self.taus):
            p = self.p(link, thr, k)
            if p >= alpha:
                if k == 0:
                    return 0.0
                p0, t0 = self.p(link, thr, k - 1), self.taus[k - 1]
                return t0 + (alpha - p0) / max(p - p0, 1e-12) * (tau - t0)
        return None


# ====================================================================================
# 6. decision context & policies
# ====================================================================================
@dataclass
class DecisionContext:
    t: float
    cfg: HandoverConfig
    specs: Dict[str, LinkSpec]
    forecasts: Dict[str, LinkForecast]
    risk: RiskModel
    taus: List[float]
    current: Assignment
    eligible: List[str]
    ready_in: Dict[str, float]
    measured: Dict[str, Dict[str, Any]]
    drone: Optional[DroneState]

    def thr_ctrl(self, l: str) -> float:
        return self.specs[l].ctrl_snr_min_db + self.cfg.safety.snr_margin_db

    def thr_video(self, l: str, level: int) -> float:
        T = self.cfg.traffic
        need = self.specs[l].snr_needed_db(T.video_ladder_kbps[level] / 1000.0 * T.video_cap_margin)
        if need == INF:
            return INF
        return max(self.thr_ctrl(l), need + self.cfg.safety.snr_margin_db)

    def feasible_levels(self, l: str) -> List[int]:
        T = self.cfg.traffic
        if not self.specs[l].supports_video:
            return []
        out = []
        for lv in range(T.video_min_level, len(T.video_ladder_kbps)):
            thr = self.thr_video(l, lv)
            if thr < INF and self.risk.p(l, thr, 0) < 0.5:
                out.append(lv)
        return out

    def p_at(self, link: str, thr: float, tau: float) -> float:
        mu, sg = self.forecasts[link].interp(tau)
        return phi((thr - mu) / sg) if thr != INF else 1.0

    def snr_now(self, l: str) -> float:
        m = self.measured[l]["snr"]
        return m if m is not None else self.forecasts[l].mu[0]


class DecisionPolicy(abc.ABC):
    """Replace this class to change the decision logic (rule-based, threshold, learned ...)."""
    name = "base"

    @abc.abstractmethod
    def decide(self, ctx: DecisionContext) -> Proposal: ...


class CallablePolicy(DecisionPolicy):
    """Wrap any function ctx -> Proposal (e.g. a learned model)."""

    def __init__(self, fn: Callable[[DecisionContext], Proposal], name: str = "callable"):
        self.fn, self.name = fn, name

    def decide(self, ctx):
        return self.fn(ctx)


# ---------------------------------------------------------------- helpers shared by policies
def _best_video(ctx: DecisionContext, exclude: Sequence[str] = (), prefer: Optional[str] = None,
                p_max: float = 0.05, tau_chk: float = 2.0) -> Tuple[Optional[str], int]:
    """Highest bitrate level whose forecast outage prob. at tau_chk is <= p_max."""
    best = (None, 0, -1.0)
    T = ctx.cfg.traffic
    for l in ctx.eligible:
        if l in exclude or not ctx.specs[l].supports_video:
            continue
        for lv in range(len(T.video_ladder_kbps) - 1, T.video_min_level - 1, -1):
            thr = ctx.thr_video(l, lv)
            if thr == INF:
                continue
            if ctx.p_at(l, thr, tau_chk) <= p_max and ctx.p_at(l, thr, 0.0) <= p_max:
                score = T.video_utility(lv) + (0.01 if l == prefer else 0.0)
                if score > best[2]:
                    best = (l, lv, score)
                break
    return best[0], best[1]


class ReactivePolicy(DecisionPolicy):
    """Baseline: classic reactive switching. Preference order by latency; never moves the control
    primary proactively (failover is done by the controller's emergency path); switches back to
    a preferred link only after it has been healthy for `hold_s`. Video follows measured capacity."""
    name = "reactive"

    def __init__(self, hysteresis_db: float = 3.0, hold_s: float = 3.0):
        self.h, self.hold = hysteresis_db, hold_s
        self.good_since: Dict[str, float] = {}

    def decide(self, ctx):
        sp = ctx.specs
        pref = sorted(ctx.eligible, key=lambda l: sp[l].base_latency_ms)
        for l in pref:
            ok = ctx.snr_now(l) >= sp[l].ctrl_snr_min_db + self.h
            if ok:
                self.good_since.setdefault(l, ctx.t)
            else:
                self.good_since.pop(l, None)
        cur = ctx.current.control_primary
        tgt = cur
        if cur is None or cur not in ctx.eligible:
            cand = [l for l in pref if l in self.good_since] or sorted(pref, key=lambda l: -ctx.snr_now(l))
            tgt = cand[0] if cand else cur
        else:
            for l in pref:
                if l == cur:
                    break
                if l in self.good_since and ctx.t - self.good_since[l] >= self.hold:
                    tgt = l
                    break
        T = ctx.cfg.traffic
        vp, lv = None, 0
        for l in ctx.eligible:       # reactive rate adaptation on measured SNR
            if not sp[l].supports_video:
                continue
            for lev in range(len(T.video_ladder_kbps) - 1, T.video_min_level - 1, -1):
                if sp[l].capacity_mbps(ctx.snr_now(l) - 2.0) >= T.video_ladder_kbps[lev] / 1000 * T.video_cap_margin:
                    if lev > lv or (lev == lv and l == ctx.current.video_primary):
                        vp, lv = l, lev
                    break
        return Proposal(Assignment(tgt, (), vp, None, lv, (), tuple(ctx.specs)), mode="hard", execute_at=ctx.t,
                        reason="reactive")


class BondingPolicy(DecisionPolicy):
    """Baseline: simple bonding. Control duplicated on all links, video striped over all video links
    at the highest level the aggregate measured capacity supports. No prediction."""
    name = "bonding"

    def decide(self, ctx):
        sp, T = ctx.specs, ctx.cfg.traffic
        if not ctx.eligible:
            return Proposal(ctx.current, reason="none eligible")
        prim = max(ctx.eligible, key=lambda l: ctx.snr_now(l) - sp[l].ctrl_snr_min_db - 0.05 * sp[l].base_latency_ms)
        if ctx.current.control_primary in ctx.eligible:
            prim = ctx.current.control_primary
        dups = tuple(l for l in ctx.eligible if l != prim)
        vl = tuple(l for l in ctx.eligible if sp[l].supports_video)
        agg = sum(sp[l].capacity_mbps(ctx.snr_now(l) - 2.0) for l in vl)
        lv = 0
        for lev in range(len(T.video_ladder_kbps) - 1, T.video_min_level - 1, -1):
            if agg >= T.video_ladder_kbps[lev] / 1000 * T.video_cap_margin:
                lv = lev
                break
        return Proposal(Assignment(prim, dups, None, None, lv, vl if lv else (), tuple(sp)),
                        mode="hard", execute_at=ctx.t, reason="bonding")


class ThresholdPolicy(DecisionPolicy):
    """Single-number decision rule on the *predicted outage probability* of the current primary.

        P = P(primary below control threshold at tau* = lead time of the switch)
        P >= theta_soft            -> soft switch to the alternative with the lowest forecast risk
        theta_dup <= P < theta_soft -> pre-arm duplication on the best alternative
        P >= theta_hard (now)       -> hard switch immediately

    theta_soft defaults to the Bayes-style break-even  c_switch / (c_switch + c_loss)  derived from
    the cost weights (hypothesis to validate, not a proven optimum). All thresholds are overridable."""
    name = "threshold"

    def __init__(self, theta_soft: Optional[float] = None, theta_dup: Optional[float] = None,
                 theta_hard: float = 0.5, theta_video: float = 0.05, lead_extra_s: float = 0.5):
        self.ts, self.td, self.th, self.tv, self.lead_extra = theta_soft, theta_dup, theta_hard, theta_video, lead_extra_s

    @staticmethod
    def derive_theta(w: CostWeights) -> float:
        return w.switch_soft / (w.switch_soft + w.ctrl_loss)

    def decide(self, ctx):
        S, sp = ctx.cfg.safety, ctx.specs
        ts = self.ts if self.ts is not None else self.derive_theta(ctx.cfg.weights)
        td = self.td if self.td is not None else ts * 0.3
        cur = ctx.current.control_primary
        if not ctx.eligible:
            return Proposal(ctx.current, reason="none eligible")
        stay_tau = 3.0
        def risk(l, tau):
            return max(ctx.p_at(l, ctx.thr_ctrl(l), tau), ctx.p_at(l, ctx.thr_ctrl(l), tau + stay_tau) * 0.7)
        if cur is None or cur not in ctx.eligible:
            cur_new = min(ctx.eligible, key=lambda l: (risk(l, 0.0), sp[l].base_latency_ms))
            vp, lv = _best_video(ctx, exclude=(cur_new,), prefer=None, p_max=self.tv)
            vp, lv = (vp, lv) if vp else _best_video(ctx, p_max=self.tv)
            return Proposal(Assignment(cur_new, (), vp, None, lv, (), tuple(sp)), mode="hard", execute_at=ctx.t,
                            reason="init", urgent=True)
        lead = ctx.ready_in.get(cur, 0.0) + S.soft_overlap_s + S.lead_margin_s + self.lead_extra
        alts = [l for l in ctx.eligible if l != cur]
        best_alt = min(alts, key=lambda l: (risk(l, lead), sp[l].base_latency_ms)) if alts else None
        p_cur = risk(cur, lead)
        p_now = ctx.p_at(cur, ctx.thr_ctrl(cur), 0.0)
        tgt_p, dups, mode, urg, why = cur, (), None, False, "stay"
        if best_alt and p_cur >= ts and risk(best_alt, lead) < p_cur * 0.6:
            tgt_p, why = best_alt, f"soft: P={p_cur:.3f}>=theta={ts:.3f}"
            mode = "hard" if p_now >= self.th else None
            urg = p_now >= self.th
        elif best_alt and p_cur >= td:
            dups, why = (best_alt,), f"dup: P={p_cur:.3f}"
        vp, lv = _best_video(ctx, exclude=(tgt_p,), prefer=ctx.current.video_primary, p_max=self.tv)
        if vp is None:
            vp, lv = _best_video(ctx, prefer=ctx.current.video_primary, p_max=self.tv)
        vf = None
        if vp is not None:
            others = [l for l in ctx.eligible if l != vp and sp[l].supports_video
                      and ctx.thr_video(l, ctx.cfg.traffic.video_min_level) < INF]
            if others:
                vf = min(others, key=lambda l: ctx.p_at(l, ctx.thr_video(l, ctx.cfg.traffic.video_min_level), 2.0))
        warm = tuple(sp) if p_cur >= S.warm_trigger_prob else tuple(l for l in sp if sp[l].always_warm)
        return Proposal(Assignment(tgt_p, dups, vp, vf, lv, (), warm), mode=mode, alpha=ts, urgent=urg, reason=why)


class RuleBasedPolicy(DecisionPolicy):
    """Transparent dB-margin rules on the forecast *mean* SNR at the switch lead time:
         R1 margin_primary(lead) < m_switch and best alt has +h dB more  -> switch
         R2 margin_primary(lead) < m_dup                                  -> duplicate on best alt
         R3 video: highest level with margin >= m_video at lead; fallback = best other link."""
    name = "rule"

    def __init__(self, m_switch_db: float = 4.0, m_dup_db: float = 8.0, hyst_db: float = 3.0, m_video_db: float = 2.0):
        self.ms, self.md, self.h, self.mv = m_switch_db, m_dup_db, hyst_db, m_video_db

    def decide(self, ctx):
        S, sp, T = ctx.cfg.safety, ctx.specs, ctx.cfg.traffic
        if not ctx.eligible:
            return Proposal(ctx.current, reason="none eligible")
        cur = ctx.current.control_primary
        lead = S.soft_overlap_s + S.lead_margin_s + 1.0
        def margin(l, tau, thr=None):
            mu, _ = ctx.forecasts[l].interp(tau)
            return mu - (sp[l].ctrl_snr_min_db if thr is None else thr)
        if cur is None or cur not in ctx.eligible:
            cur_new = max(ctx.eligible, key=lambda l: margin(l, lead))
            return Proposal(Assignment(cur_new, (), None, None, 0, (), tuple(sp)), mode="hard", execute_at=ctx.t,
                            urgent=True, reason="init")
        alts = [l for l in ctx.eligible if l != cur]
        alt = max(alts, key=lambda l: min(margin(l, lead), margin(l, lead + 3.0))) if alts else None
        mc = min(margin(cur, lead), margin(cur, lead + 3.0))
        tgt, dups, why = cur, (), "stay"
        if alt and mc < self.ms and min(margin(alt, lead), margin(alt, lead + 3.0)) > mc + self.h:
            tgt, why = alt, f"R1 margin {mc:.1f}dB"
        elif alt and mc < self.md:
            dups, why = (alt,), f"R2 margin {mc:.1f}dB"
        vp, lv, best_u = None, 0, -1.0
        for l in ctx.eligible:
            for lev in range(len(T.video_ladder_kbps) - 1, T.video_min_level - 1, -1):
                need = sp[l].snr_needed_db(T.video_ladder_kbps[lev] / 1000 * T.video_cap_margin)
                if need < INF and min(ctx.forecasts[l].interp(lead)[0], ctx.forecasts[l].interp(lead + 3)[0]) - need >= self.mv + 2 \
                        and sp[l].supports_video:
                    if T.video_utility(lev) > best_u:
                        vp, lv, best_u = l, lev, T.video_utility(lev)
                    break
        vf = None
        if vp:
            o = [l for l in ctx.eligible if l != vp and sp[l].supports_video]
            if o:
                vf = max(o, key=lambda l: margin(l, lead))
        warm = tuple(sp) if mc < self.md else tuple(l for l in sp if sp[l].always_warm)
        return Proposal(Assignment(tgt, dups, vp, vf, lv, (), warm), reason=why)


@dataclass
class CandEval:
    a: Assignment
    cost: float
    viol: float
    terms: Dict[str, float]


class CostFunctionPolicy(DecisionPolicy):
    """Main contribution: enumerate (control primary, duplicates, video primary/fallback, bitrate level),
    score each by a weighted cost over the forecast horizon and pick the minimum.

    cost = w.blind * E[P(no control & no video)] + w.ctrl_loss * E[P(no control)] + w.ctrl_latency * P(miss soft deadline)
         + w.video_quality * (1 - E[quality]) + w.power * W + w.spectrum * airtime + w.duplication * n_dups
         + switching costs + w.chance_violation * max(0, max_k P(ctrl loss) - eps)
    Hard constraints (lexicographically first): P(latency <= L_max) >= target at the current SNR,
    airtime <= per-link budget, total radio power <= budget. If nothing satisfies them, the candidate with
    the smallest violation wins ('survival mode')."""
    name = "cost"

    def __init__(self, allow_dups: bool = True, allow_video_fallback: bool = True):
        self.allow_dups, self.allow_vf = allow_dups, allow_video_fallback

    # ------------------------------------------------------------ candidate generation
    def _candidates(self, ctx):
        S = ctx.cfg.safety
        L = ctx.eligible
        vlinks = [l for l in L if ctx.specs[l].supports_video]
        lv_cache = {l: ctx.feasible_levels(l) for l in vlinks}
        for p in L:
            dopts = [()]
            if self.allow_dups and S.max_dups >= 1:
                others = [x for x in L if x != p]
                dopts += [(x,) for x in others]
                if S.max_dups >= 2 and len(others) >= 2:
                    dopts += [tuple(others[:2])]
            for dups in dopts:
                yield Assignment(p, dups, None, None, 0)
                for vp in vlinks:
                    for lv in lv_cache[vp]:
                        yield Assignment(p, dups, vp, None, lv)
                        if self.allow_vf:
                            for vf in vlinks:
                                if vf != vp and ctx.thr_video(vf, ctx.cfg.traffic.video_min_level) < INF:
                                    yield Assignment(p, dups, vp, vf, lv)

    # ------------------------------------------------------------ evaluation
    def _eval(self, ctx, a: Assignment, cache) -> Optional[CandEval]:
        cfg, W, S, T, sp, risk = ctx.cfg, ctx.cfg.weights, ctx.cfg.safety, ctx.cfg.traffic, ctx.specs, ctx.risk
        C = a.control_links()
        for l in a.active_links():
            if l not in ctx.eligible:
                return None
        thr_c = {l: round(ctx.thr_ctrl(l), 2) for l in C}
        ck = ("c", tuple(sorted(thr_c.items())))
        if ck not in cache:
            cache[ck] = (risk.integral(ck[1]), risk.maximum(ck[1]))
        ctrl_int, ctrl_max = cache[ck]
        lv_min = T.video_min_level
        vp, vf, lv = a.video_primary, a.video_fallback, a.video_level
        bs = dict(thr_c)
        thr_vp = thr_vf = None
        if vp is not None and lv > 0:
            thr_vp = round(ctx.thr_video(vp, lv), 2)
            bs.setdefault(vp, thr_vp)
        if vf is not None:
            thr_vf = round(ctx.thr_video(vf, lv_min), 2)
            bs.setdefault(vf, thr_vf)
        if thr_vp is None and thr_vf is None:
            blind_int, quality = ctrl_int, 0.0
        else:
            bk = ("b", tuple(sorted(bs.items())))
            if bk not in cache:
                cache[bk] = risk.integral(bk[1])
            blind_int = cache[bk]
            qk = ("q", vp, lv, vf)
            if qk not in cache:
                if thr_vp is not None:
                    s1 = (("%s" % vp, thr_vp),)
                    pv_int = risk.integral(s1)
                    if thr_vf is not None:
                        s2 = tuple(sorted(((vp, thr_vp), (vf, thr_vf))))
                        pb_int = risk.integral(s2)
                    else:
                        pb_int = pv_int
                    cache[qk] = (1 - pv_int) * T.video_utility(lv) + (pv_int - pb_int) * T.video_utility(lv_min)
                else:
                    pf = risk.integral(((vf, thr_vf),))
                    cache[qk] = (1 - pf) * T.video_utility(lv_min)
            quality = cache[qk]
        # ----- loads, latency, power, spectrum
        rate_vp = T.video_ladder_kbps[lv] if (vp and lv > 0) else 0.0
        rate_vf = T.video_ladder_kbps[lv_min] if vf else 0.0
        kL = min(range(len(ctx.taus)), key=lambda k: abs(ctx.taus[k] - S.latency_check_tau_s))
        viol = 0.0
        power = 0.0
        spec_cost = 0.0
        util: Dict[str, float] = {}
        for l in a.active_links():
            kb = (rate_vp if l == vp else 0.0) + (rate_vf if l == vf else 0.0)
            if l == a.control_primary:
                kb += T.ctrl_kbps + T.telemetry_kbps
            elif l in a.control_dups:
                kb += T.ctrl_kbps * T.dup_fraction
            cap = sp[l].capacity_mbps(ctx.forecasts[l].mu[0]) * 1000.0
            u = kb / cap if cap > 0 else (0.0 if kb <= 0 else 9.9)
            util[l] = u
            if u > sp[l].airtime_budget + 1e-9:
                viol += u - sp[l].airtime_budget
            power += sp[l].active_power_w + sp[l].power_w_per_mbps * kb / 1000.0
            spec_cost += sp[l].spectrum_cost * min(u, 1.0)
        for l, s in sp.items():
            if l not in a.active_links() and s.always_warm:
                power += s.idle_power_w
        if power > S.power_budget_w:
            viol += (power - S.power_budget_w) / S.power_budget_w
        vid_util = {l: util[l] for l in util}
        F_now = 1.0
        for l in C:
            F_now *= 1 - sp[l].ctrl_deadline_prob(ctx.forecasts[l].mu[0], T.ctrl_deadline_hard_ms, 0.0 if False else
                                                 vid_util[l] if l not in (a.control_primary,) else
                                                 max(0.0, vid_util[l] - (T.ctrl_kbps + T.telemetry_kbps) / max(1.0, sp[l].capacity_mbps(ctx.forecasts[l].mu[0]) * 1000.0)))
        F_now = 1 - F_now
        if F_now < T.ctrl_reliability_target:
            viol += T.ctrl_reliability_target - F_now
        p = a.control_primary
        fc = ctx.forecasts[p]
        snr_pess = fc.mu[kL] - S.latency_quantile_z * fc.sigma[kL]
        F_soft = sp[p].ctrl_deadline_prob(snr_pess, T.ctrl_deadline_soft_ms, min(util.get(p, 0.0), 0.95))
        lat_miss = 1 - F_soft
        # ----- switching cost
        cur = ctx.current
        sw = 0.0
        if cur.control_primary is not None and a.control_primary != cur.control_primary:
            p_cur = risk.p(cur.control_primary, ctx.thr_ctrl(cur.control_primary), 0) if cur.control_primary in ctx.forecasts else 1.0
            if a.control_primary in cur.control_dups:
                sw += 0.1 * W.switch_soft            # seamless: stream already flowing on the target
            else:
                sw += W.switch_hard if p_cur > 0.5 else W.switch_soft
        if a.video_primary != cur.video_primary or a.video_level != cur.video_level:
            sw += W.video_change * (1 + abs(a.video_level - cur.video_level))
        chance = max(0.0, ctrl_max - S.ctrl_fail_max)
        terms = {"blind": W.blind * blind_int, "ctrl_loss": W.ctrl_loss * ctrl_int, "latency": W.ctrl_latency * lat_miss,
                 "video": W.video_quality * (1 - quality), "power": W.power * power, "spectrum": W.spectrum * spec_cost,
                 "dup": W.duplication * len(a.control_dups), "switch": sw, "chance": W.chance_violation * chance}
        cost = sum(terms.values())
        terms.update({"P_blind_int": blind_int, "P_ctrl_loss_max": ctrl_max, "F_ctrl_now": F_now,
                      "E_quality": quality, "power_w": power})
        return CandEval(a, cost, viol, terms)

    # ------------------------------------------------------------ warm set
    def _warm(self, ctx, a: Assignment) -> Tuple[str, ...]:
        S, sp = ctx.cfg.safety, ctx.specs
        warm = {l for l in sp if sp[l].always_warm} | a.active_links()
        p = a.control_primary
        if p:
            cold = [sp[l].acquire_time_s for l in sp if not sp[l].always_warm]
            hz = (max(cold) if cold else 0.0) + S.warm_lead_s
            kmax = max(1, max((k for k, t in enumerate(ctx.taus) if t <= hz), default=0) + 1)
            pm = max(ctx.risk.p(p, ctx.thr_ctrl(p), k) for k in range(min(kmax, len(ctx.taus))))
            if pm >= S.warm_trigger_prob:
                warm |= set(sp)
        return tuple(sorted(warm))

    def decide(self, ctx):
        if not ctx.eligible:
            return Proposal(ctx.current, reason="no eligible links")
        cache: Dict[tuple, Any] = {}
        evals = [e for e in (self._eval(ctx, a, cache) for a in self._candidates(ctx)) if e is not None]
        if not evals:
            return Proposal(ctx.current, reason="no candidates")
        best = min(evals, key=lambda e: (e.viol > 1e-9, e.viol if e.viol > 1e-9 else 0.0, e.cost))
        cur = replace(ctx.current, keep_warm=())
        cur_eval = self._eval(ctx, cur, cache) if cur.control_primary else None
        chosen, kept = best, False
        if cur_eval and cur_eval.viol <= 1e-9 and best.viol <= 1e-9 \
                and cur_eval.cost - best.cost < ctx.cfg.safety.switch_hysteresis:
            chosen, kept = cur_eval, True
        warm = self._warm(ctx, chosen.a)
        if best.viol > 1e-9:
            warm = tuple(sorted(ctx.specs))          # survival mode: bring every link up
        tgt = replace(chosen.a, keep_warm=warm)
        top = sorted(evals, key=lambda e: (e.viol > 1e-9, e.cost))[:3]
        info = {"chosen_terms": {k: round(v, 4) for k, v in chosen.terms.items()}, "n_candidates": len(evals),
                "survival_mode": best.viol > 1e-9, "kept_current": kept,
                "top3_cost": [round(e.cost, 3) for e in top]}
        return Proposal(tgt, reason="keep" if kept else "cost-min", info=info)


# ====================================================================================
# 7. controller (state machine, scheduler)
# ====================================================================================
class LinkRuntime:
    def __init__(self, spec: LinkSpec):
        self.spec = spec
        self.warm = spec.always_warm
        self.warming_until = -1e9
        self.cool_since: Optional[float] = None

    def ready_in(self, t: float) -> float:
        if self.warm:
            return max(0.0, self.warming_until - t)
        return self.spec.acquire_time_s

    def state(self, t) -> str:
        if not self.warm:
            return "cold"
        return "warming" if t < self.warming_until else "ready"


class HandoverController:
    def __init__(self, specs: Sequence[LinkSpec], cfg: Optional[HandoverConfig] = None,
                 policy: Optional[DecisionPolicy] = None, predictor: Optional[OutagePredictor] = None):
        self.cfg = cfg or HandoverConfig()
        self.specs: Dict[str, LinkSpec] = {s.name: s for s in specs}
        self.policy = policy or CostFunctionPolicy()
        self.predictor = predictor or HybridPredictor(self.specs, self.cfg.predictor, self.cfg.safety)
        self.traj = TrajectoryForecaster(self.cfg.predictor)
        self.taus = make_tau_grid(self.cfg.safety.horizon_s, self.cfg.safety.horizon_points)
        self.runtime = {n: LinkRuntime(s) for n, s in self.specs.items()}
        self.drone: Optional[DroneState] = None
        self.current = Assignment()
        self.pending: Optional[dict] = None
        self.switch: Optional[dict] = None
        self.blackout_until = -1e9
        self.last_decide = -1e9
        self.last_primary_change = -1e9
        self.last_video_change = -1e9
        self.counters: collections.Counter = collections.Counter()
        self._below_since: Dict[str, float] = {}
        self._reasons: List[str] = []
        self._diag: Dict[str, Any] = {}
        self._advisory: Optional[dict] = None
        self._last_risk: Optional[RiskModel] = None
        self._last_fc: Dict[str, LinkForecast] = {}

    # ------------------------------------------------------------------ inputs
    def set_mission(self, waypoints: Sequence[Vec3], speed_mps: Optional[float] = None):
        self.traj.set_mission(waypoints)

    def set_shadow_map(self, link: str, fn: Optional[Callable[[Vec3], float]]):
        self.predictor.set_shadow_map(link, fn)

    def on_drone_state(self, st: DroneState):
        self.drone = st

    def on_link_sample(self, link: str, s: LinkSample, pos: Optional[Vec3] = None):
        p = pos if pos is not None else (self.drone.pos if self.drone else (0.0, 0.0, 0.0))
        self.predictor.update(link, s, p)

    # ------------------------------------------------------------------ main step
    def step(self, t: float) -> Decision:
        S = self.cfg.safety
        self._reasons = []
        self._update_warm(t, set(self.current.keep_warm))
        self._advance(t)
        if S.enable_emergency and self.drone is not None:
            self._emergency(t)
        if self.drone is not None and t - self.last_decide >= self.cfg.decision_period_s:
            self.last_decide = t
            self._decide(t)
        return self._make_decision(t)

    # ------------------------------------------------------------------ internals
    def _eligible(self, t) -> List[str]:
        out = []
        for l in self.specs:
            m = self.predictor.measured(l)
            if m["has_data"] and m["connected"] and m["last_t"] is not None \
                    and t - m["last_t"] <= self.cfg.safety.stale_timeout_s and self.runtime[l].ready_in(t) <= 1e-9:
                out.append(l)
        return out

    def _update_warm(self, t, wanted: set):
        for l, rt in self.runtime.items():
            need = l in wanted or rt.spec.always_warm or l in self.current.active_links()
            if need:
                rt.cool_since = None
                if not rt.warm:
                    rt.warm, rt.warming_until = True, t + rt.spec.acquire_time_s
            elif rt.warm:
                if rt.cool_since is None:
                    rt.cool_since = t
                elif t - rt.cool_since >= self.cfg.safety.cool_hold_s:
                    rt.warm = False

    def _effective(self) -> Assignment:
        a = self.current
        if self.switch:
            to = self.switch["to"]
            return replace(a, control_dups=tuple(x for x in dict.fromkeys(a.control_dups + (to,)) if x != a.control_primary))
        return a

    def _advance(self, t):
        if self.switch and t >= self.switch["end"]:
            tgt = self.switch["target"]
            self.current = replace(tgt, keep_warm=self.current.keep_warm, video_primary=self.current.video_primary,
                                   video_fallback=self.current.video_fallback, video_level=self.current.video_level,
                                   video_bonded=self.current.video_bonded)
            self.switch = None
        if self.pending and t >= self.pending["at"] and not self.switch:
            pend, self.pending = self.pending, None
            self._begin_switch(t, pend["target"], pend["mode"], pend["reason"])

    def _begin_switch(self, t, tgt: Assignment, mode: str, reason: str):
        S, cur = self.cfg.safety, self.current
        Q = tgt.control_primary
        ready = self.runtime[Q].ready_in(t)
        self._merge_noncontrol(t, tgt, keep_primary=False)
        if Q in cur.control_dups and ready <= 0.05:
            # target already receives the duplicated control stream -> relabelling the primary is seamless
            self.current = replace(self.current, control_primary=Q,
                                   control_dups=tuple(x for x in tgt.control_dups if x != Q))
            self.counters["seamless_switch"] += 1
            mode = "seamless"
        elif mode == "soft" and cur.control_primary is not None and ready <= 0.05:
            self.switch = {"mode": "soft", "end": t + S.soft_overlap_s, "to": Q, "target": replace(tgt)}
            self.counters["soft_switch"] += 1
        else:
            self.current = replace(self.current, control_primary=Q, control_dups=tgt.control_dups)
            self.blackout_until = t + S.hard_blackout_s + ready
            self.counters["hard_switch"] += 1
            mode = "hard"
        self.last_primary_change = t
        self._reasons.append(f"{mode} switch -> {Q} ({reason})")

    def _merge_noncontrol(self, t, tgt: Assignment, keep_primary: bool = True):
        cur, S = self.current, self.cfg.safety
        lvl = tgt.video_level
        if tgt.video_primary == cur.video_primary and lvl > cur.video_level \
                and t - self.last_video_change < S.video_up_hold_s and cur.video_level > 0:
            lvl = cur.video_level
        vp_new, vf_new = tgt.video_primary, tgt.video_fallback
        if vp_new != cur.video_primary and cur.video_primary is not None and cur.video_level > 0 \
                and t - self.last_video_change < S.video_min_dwell_s and not cur.video_bonded:
            m = self.predictor.measured(cur.video_primary)
            sp = self.specs[cur.video_primary]
            if m["snr"] is not None and m["snr"] >= sp.ctrl_snr_min_db + S.snr_margin_db + 3.0:
                vp_new, vf_new, lvl = cur.video_primary, cur.video_fallback, min(lvl, cur.video_level)
        tgt = replace(tgt, video_primary=vp_new, video_fallback=vf_new)
        changed = (tgt.video_primary, tgt.video_fallback, lvl, tgt.video_bonded) != \
                  (cur.video_primary, cur.video_fallback, cur.video_level, cur.video_bonded)
        new = replace(cur, video_primary=tgt.video_primary, video_fallback=tgt.video_fallback, video_level=lvl,
                      video_bonded=tgt.video_bonded, keep_warm=tgt.keep_warm)
        if keep_primary:
            new = replace(new, control_dups=tuple(x for x in tgt.control_dups if x != new.control_primary))
        if changed:
            self.last_video_change = t
            self.counters["video_change"] += 1
        self.current = new

    def _emergency(self, t):
        S, cur = self.cfg.safety, self.current
        p = cur.control_primary
        if p is None or self.switch is not None:
            return
        m = self.predictor.measured(p)
        sp = self.specs[p]
        stale = (m["last_t"] is None) or (t - m["last_t"] > S.stale_timeout_s) or (not m["connected"])
        fs = m["fast"] if m["fast"] is not None else m["snr"]
        low = m["has_data"] and fs is not None and min(fs, m["snr"]) < sp.ctrl_snr_min_db
        if low or stale:
            self._below_since.setdefault(p, t)
        else:
            self._below_since.pop(p, None)
            return
        if not stale and t - self._below_since[p] < S.emergency_detect_s:
            return
        alts = [l for l in self._eligible(t) if l != p]
        def _q(l):
            mm = self.predictor.measured(l)
            f = mm["fast"] if mm["fast"] is not None else mm["snr"]
            return min(f if f is not None else -99, mm["snr"] if mm["snr"] is not None else -99) \
                - self.specs[l].ctrl_snr_min_db
        usable = [l for l in alts if _q(l) >= 0.0]
        if not usable:
            return
        q = max(usable, key=_q)
        tgt = replace(cur, control_primary=q, control_dups=tuple(x for x in cur.control_dups if x != q))
        if cur.video_primary == p:
            tgt = replace(tgt, video_primary=q if self.specs[q].supports_video else None,
                          video_level=min(cur.video_level, self.cfg.traffic.video_min_level) if self.specs[q].supports_video else 0)
        self.pending = None
        self.counters["emergency"] += 1
        self._begin_switch(t, tgt, "hard", "EMERGENCY failover")
        self._below_since.pop(p, None)

    def _decide(self, t):
        S = self.cfg.safety
        fcs: Dict[str, LinkForecast] = {}
        traj = self.traj.predict(self.drone, self.taus)
        speed = _norm(self.drone.vel)
        for l in self.specs:
            fcs[l] = self.predictor.forecast(l, t, traj, self.taus, speed, register=True)
        risk = RiskModel(self.specs, fcs, self.taus, S)
        self._last_risk, self._last_fc = risk, fcs
        elig = self._eligible(t)
        ctx = DecisionContext(t, self.cfg, self.specs, fcs, risk, self.taus, self.current, elig,
                              {l: self.runtime[l].ready_in(t) for l in self.specs},
                              {l: self.predictor.measured(l) for l in self.specs}, self.drone)
        prop = self.policy.decide(ctx)
        self._diag = {"policy": self.policy.name, "info": prop.info, "reason": prop.reason, "eligible": elig}
        self._apply(t, prop, ctx)
        self._advisory = self._flight_advisory(ctx) if S.enable_flight_advisory else None

    def _apply(self, t, prop: Proposal, ctx: DecisionContext):
        S, cur, tgt = self.cfg.safety, self.current, prop.target
        if tgt.control_primary is None:
            return
        if self.switch is not None:
            return
        if tgt.control_primary == cur.control_primary or cur.control_primary is None:
            if cur.control_primary is None:
                self.current = replace(cur, control_primary=tgt.control_primary, control_dups=tgt.control_dups)
                self.last_primary_change = t
                self._reasons.append(f"init primary {tgt.control_primary}")
            self.pending = None
            self._merge_noncontrol(t, tgt, keep_primary=True)
            return
        # ---- primary change requested
        self._merge_noncontrol(t, replace(tgt, control_primary=cur.control_primary), keep_primary=True)
        if not prop.urgent and t - self.last_primary_change < S.min_dwell_s:
            self.pending = None
            self._reasons.append("switch suppressed by dwell")
            return
        mode, at = prop.mode, prop.execute_at
        if mode is None or at is None:
            mode, at = self._schedule(t, tgt, prop, ctx)
        if at <= t + 1e-9:
            self.pending = None
            self._begin_switch(t, tgt, mode, prop.reason or "policy")
        else:
            self.pending = {"at": at, "mode": mode, "target": tgt, "reason": prop.reason or "policy"}

    def _schedule(self, t, tgt: Assignment, prop: Proposal, ctx: DecisionContext) -> Tuple[str, float]:
        """Time-to-outage based timing: switch as late as possible while still finishing before risk >= alpha."""
        S = self.cfg.safety
        cur = self.current.control_primary
        Q = tgt.control_primary
        alpha = prop.alpha if prop.alpha is not None else S.switch_risk_quantile
        tto = ctx.risk.time_to_risk(cur, ctx.thr_ctrl(cur), alpha)
        ready = ctx.ready_in.get(Q, 0.0)
        lead_soft = ready + S.soft_overlap_s + S.lead_margin_s
        m = ctx.measured[cur]
        cur_ok = m["snr"] is not None and m["snr"] >= self.specs[cur].ctrl_snr_min_db
        if not cur_ok:
            return "hard", t
        if tto is None:
            return "soft", t            # opportunistic improvement, no risk deadline
        if tto <= lead_soft:
            return "soft", t
        return "soft", t + tto - lead_soft

    def _flight_advisory(self, ctx: DecisionContext) -> Optional[dict]:
        S = self.cfg.safety
        if not ctx.eligible:
            return {"action": "LOITER_OR_RTL", "reason": "no usable link", "heading_enu": None}
        k = min(len(self.taus) - 1, max(1, len(self.taus) // 2))
        pall = ctx.risk.conj(tuple(sorted((l, round(ctx.thr_ctrl(l), 2)) for l in self.specs)), k)
        if pall < S.advisory_prob:
            return None
        best = min(self.specs, key=lambda l: ctx.risk.p(l, ctx.thr_ctrl(l), 0))
        vec = _sub(self.specs[best].endpoint, self.drone.pos)
        n = _norm(vec[:2]) or 1.0
        return {"action": "SLOW_AND_TURN_TOWARD_LINK", "link": best, "p_all_down": round(pall, 3),
                "heading_enu": (vec[0] / n, vec[1] / n)}

    def _make_decision(self, t) -> Decision:
        sched = []
        if self.pending:
            sched.append(ScheduledAction(self.pending["at"], self.pending["mode"], self.pending["target"],
                                         self.pending["reason"]))
        diag = dict(self._diag)
        diag["counters"] = dict(self.counters)
        diag["link_state"] = {l: self.runtime[l].state(t) for l in self.specs}
        if self._last_risk is not None:
            diag["p_ctrl_fail"] = {l: [round(self._last_risk.p(l, self.specs[l].ctrl_snr_min_db + self.cfg.safety.snr_margin_db, k), 3)
                                       for k in range(len(self.taus))] for l in self.specs}
            diag["taus"] = [round(x, 2) for x in self.taus]
        warm = tuple(l for l, r in self.runtime.items() if r.warm)
        return Decision(t, self._effective(), self.blackout_until, sched, warm, self._advisory, list(self._reasons), diag)


# ====================================================================================
# 8. simulation + benchmark (synthetic channel; use to compare policies, NOT as a field result)
# ====================================================================================
def default_link_specs() -> List[LinkSpec]:
    gcs = (0.0, 0.0, 2.0)
    return [
        LinkSpec("r58", "5g8", gcs, -92.0, -18.0, 2.3,
                 ((4, 1.5), (8, 6), (12, 14), (16, 24), (20, 35)), 3.0, base_latency_ms=8, retx_interval_ms=10,
                 active_power_w=3.5, power_w_per_mbps=0.12, idle_power_w=0.7),
        LinkSpec("l900", "900m", gcs, -105.0, 2.0, 2.4,
                 ((3, 0.35), (6, 0.8), (10, 1.5)), 1.0, base_latency_ms=14, retx_interval_ms=15, mtu_bytes=400,
                 active_power_w=3.0, power_w_per_mbps=1.0, idle_power_w=0.8),
        LinkSpec("lte", "cell", (1200.0, -500.0, 40.0), -97.0, 0.0, 2.7,
                 ((2, 1.0), (6, 4.0), (10, 9.0), (14, 15.0), (18, 25.0)), 1.0, base_latency_ms=45, retx_interval_ms=30,
                 active_power_w=1.5, power_w_per_mbps=0.2, idle_power_w=0.4, acquire_time_s=1.5,
                 always_warm=False, spectrum_cost=0.5),
    ]


class _Truth:
    """Ground-truth channel for one link."""

    def __init__(self, spec, p0, n, obstacles, shadow_sigma, rng, events):
        self.spec, self.p0, self.n, self.obs, self.rng, self.events = spec, p0, n, obstacles, rng, events
        self.sig, self.sh = shadow_sigma, 0.0
        self.rho = math.exp(-0.1 * 20.0 / 60.0)

    def obstacle_loss(self, pos):
        tot = 0.0
        for cx, cy, r, loss, edge in self.obs:
            d = math.hypot(pos[0] - cx, pos[1] - cy)
            tot += loss / (1 + math.exp(clamp((d - r) / edge, -50, 50)))
        return tot

    def slow(self, t, pos):
        d = max(dist(pos, self.spec.endpoint), 1.0)
        self.sh = self.rho * self.sh + math.sqrt(1 - self.rho ** 2) * self.sig * self.rng.gauss(0, 1)
        rssi = self.p0 - self.n * 10 * math.log10(d) - self.obstacle_loss(pos) + self.sh
        noise = self.spec.noise_floor_dbm
        for (t0, t1, kind, mag) in self.events:
            if t0 - 0.3 <= t <= t1 + 0.3:
                f = min(1.0, (t - (t0 - 0.3)) / 0.3, ((t1 + 0.3) - t) / 0.3)
                if kind == "noise":
                    noise += mag * f
                else:
                    rssi -= mag * f
        return rssi, noise


class Scenario:
    WAYPOINTS = [(0, 0, 80), (1500, 0, 80), (3000, 600, 80), (2400, -600, 80), (0, 0, 80)]

    def __init__(self, seed: int, speed: float = 24.0, map_quality: str = "rough"):
        self.rng = random.Random(seed)
        self.speed = speed
        self.specs = default_link_specs()
        self.wps = [tuple(map(float, w)) for w in self.WAYPOINTS]
        self.seglen = [dist(self.wps[i], self.wps[i + 1]) for i in range(len(self.wps) - 1)]
        self.total = sum(self.seglen)
        self.duration = self.total / speed
        r = self.rng
        obs = {"r58": [(800, 0, 70, 22, 15), (1400, 150, 80, 18, 15)],
               "l900": [(850, 0, 180, 30, 40), (2150, 300, 300, 14, 120)],
               "lte": [(1900, 150, 350, 25, 60)]}
        self.obs = obs
        ev = {"r58": self._poisson(1 / 20, 1, 3, "noise", 14, 24), "l900": self._poisson(1 / 25, 1, 4, "noise", 15, 28),
              "lte": self._poisson(1 / 40, 0.8, 1.5, "signal", 25, 30)}
        sig = {"r58": 3.0, "l900": 3.0, "lte": 4.0}
        offs = {"r58": (2.0, 0.1), "l900": (-2.0, 0.1), "lte": (3.0, -0.15)}
        self.truth = {}
        for s in self.specs:
            self.truth[s.name] = _Truth(s, s.prior_rssi_1m_dbm + offs[s.name][0], s.prior_path_loss_exp + offs[s.name][1],
                                        obs[s.name], sig[s.name], random.Random(seed * 100 + sum(map(ord, s.name)) % 97), ev[s.name])
        self.map_quality = map_quality
        mr = random.Random(seed + 7)
        self.maps = {}
        if map_quality != "none":
            for name, lst in obs.items():
                est = [(cx + mr.uniform(-30, 30), cy + mr.uniform(-30, 30), r_ * mr.uniform(0.85, 1.15),
                        l * (0.75 if map_quality == "rough" else 1.0), e * 1.3) for cx, cy, r_, l, e in lst]
                self.maps[name] = (lambda est_: (lambda pos: sum(
                    l / (1 + math.exp(clamp((math.hypot(pos[0] - cx, pos[1] - cy) - r_) / e, -50, 50)))
                    for cx, cy, r_, l, e in est_)))(est)

    def _poisson(self, rate, dmin, dmax, kind, mmin, mmax):
        out, t = [], self.rng.expovariate(rate)
        while t < 400:
            d = self.rng.uniform(dmin, dmax)
            out.append((t, t + d, kind, self.rng.uniform(mmin, mmax)))
            t += d + self.rng.expovariate(rate)
        return out

    def drone_at(self, t):
        s = min(t * self.speed, self.total - 1e-6)
        i = 0
        while s > self.seglen[i]:
            s -= self.seglen[i]
            i += 1
        a, b = self.wps[i], self.wps[i + 1]
        u = s / self.seglen[i]
        pos = _add(a, _mul(_sub(b, a), u))
        d = _sub(b, a)
        n = _norm(d)
        return pos, _mul(d, self.speed / n)


def make_policy(name: str):
    return {"reactive": ReactivePolicy, "bonding": BondingPolicy, "rule": RuleBasedPolicy,
            "threshold": ThresholdPolicy, "cost": CostFunctionPolicy, "cost_nomap": CostFunctionPolicy,
            "cost_trend": CostFunctionPolicy, "cost_nodup": lambda: CostFunctionPolicy(allow_dups=False)}[name]() \
        if name != "cost_nodup" else CostFunctionPolicy(allow_dups=False)


def run_episode(policy_name: str, seed: int, dt: float = 0.1, cfg: Optional[HandoverConfig] = None,
                verbose: bool = False, trace: Optional[list] = None) -> Dict[str, Any]:
    sc = Scenario(seed, map_quality="none" if policy_name == "cost_nomap" else "rough")
    cfg = cfg or HandoverConfig()
    cfg = HandoverConfig.from_dict(cfg.to_dict())
    if policy_name == "cost_trend":
        cfg.predictor.use_geometry = False
    ctrl = HandoverController(sc.specs, cfg, make_policy(policy_name))
    ctrl.set_mission(sc.wps, sc.speed)
    for name, fn in sc.maps.items():
        ctrl.set_shadow_map(name, fn)
    T, sp = cfg.traffic, {s.name: s for s in sc.specs}
    rng = random.Random(seed + 999)
    n = int(sc.duration / dt)
    m = collections.Counter()
    power_sum = quality_sum = 0.0
    lat_miss_sum = 0.0
    for i in range(n):
        t = i * dt
        pos, vel = sc.drone_at(t)
        ctrl.on_drone_state(DroneState(t, pos, vel))
        snr_true = {}
        for name, tr in sc.truth.items():
            rssi, noise = tr.slow(t, pos)
            snr_true[name] = rssi - noise
            fast = rng.gauss(0, 2.0)
            mn = rng.gauss(0, 0.5)
            r_m, n_m = rssi + fast + mn, noise + rng.gauss(0, 0.5)
            ctrl.on_link_sample(name, LinkSample(t, rssi_dbm=r_m, snr_db=r_m - n_m, noise_dbm=n_m), pos)
        d = ctrl.step(t)
        a = d.assignment
        # ---------- ground-truth scoring (slow SNR, i.e. without per-packet fast fading)
        ready = {l: ctrl.runtime[l].ready_in(t) <= 1e-9 for l in sp}
        kb = collections.defaultdict(float)
        vp, vf, lv = a.video_primary, a.video_fallback, a.video_level
        if a.video_bonded:
            share = None
        for l in a.control_links():
            kb[l] += T.ctrl_kbps * (1.0 if l != a.control_primary else 1.0) + (T.telemetry_kbps if l == a.control_primary else 0)
        rate = T.video_ladder_kbps[lv]
        vid_ok, q = False, 0.0
        if a.video_bonded and lv > 0:
            agg = sum(sp[l].capacity_mbps(snr_true[l]) for l in a.video_bonded if ready[l])
            if agg * 1000 >= rate * T.video_cap_margin:
                vid_ok, q = True, T.video_utility(lv)
            elif agg * 1000 >= T.video_ladder_kbps[T.video_min_level] * T.video_cap_margin:
                vid_ok, q = True, T.video_utility(T.video_min_level)
            for l in a.video_bonded:
                kb[l] += rate / max(1, len(a.video_bonded))
        else:
            if vp and lv > 0:
                kb[vp] += rate
                if ready[vp] and sp[vp].capacity_mbps(snr_true[vp]) * 1000 >= rate * T.video_cap_margin:
                    vid_ok, q = True, T.video_utility(lv)
            if vf:
                kb[vf] += T.video_ladder_kbps[T.video_min_level]
                if not vid_ok and ready[vf] and sp[vf].capacity_mbps(snr_true[vf]) * 1000 >= \
                        T.video_ladder_kbps[T.video_min_level] * T.video_cap_margin:
                    vid_ok, q = True, T.video_utility(T.video_min_level)
        miss = 1.0
        for l in a.control_links():
            if not ready[l]:
                continue
            cap = sp[l].capacity_mbps(snr_true[l]) * 1000
            u = min(0.95, (kb[l] - (T.ctrl_kbps + (T.telemetry_kbps if l == a.control_primary else 0))) / cap) if cap > 0 else 0.0
            miss *= 1 - sp[l].ctrl_deadline_prob(snr_true[l], T.ctrl_deadline_hard_ms, max(0.0, u))
        F = 1 - miss
        in_black = t < d.control_blackout_until
        ctrl_ok = (F >= T.ctrl_reliability_target) and not in_black
        # oracle: could ANY link have given control or reduced video right now?
        any_ctrl = any(sp[l].ctrl_deadline_prob(snr_true[l], T.ctrl_deadline_hard_ms) >= T.ctrl_reliability_target for l in sp)
        any_vid = any(sp[l].capacity_mbps(snr_true[l]) * 1000 >= T.video_ladder_kbps[T.video_min_level] * T.video_cap_margin
                      for l in sp if sp[l].supports_video)
        if trace is not None and ((not ctrl_ok) and (not vid_ok)):
            trace.append((round(t, 1), a.control_primary, a.control_dups, a.video_primary, a.video_level, in_black,
                          {k: round(v, 1) for k, v in snr_true.items()}))
        m["ticks"] += 1
        m["ctrl_lost"] += (not ctrl_ok)
        m["blind"] += (not ctrl_ok) and (not vid_ok)
        m["unavoidable_blind"] += (not any_ctrl) and (not any_vid)
        m["ctrl_lat_miss"] += 1 - F if not in_black else 1.0
        quality_sum += q
        pw = 0.0
        for l in sp:
            if l in a.active_links() or l in kb:
                pw += sp[l].active_power_w + sp[l].power_w_per_mbps * kb.get(l, 0) / 1000
            elif ctrl.runtime[l].warm:
                pw += sp[l].idle_power_w
        power_sum += pw
        if verbose and d.reasons:
            print(f"  t={t:6.1f} {d.reasons}")
    k = m["ticks"]
    return {"policy": policy_name, "seed": seed, "duration_s": k * dt,
            "blind_s": m["blind"] * dt, "unavoidable_blind_s": m["unavoidable_blind"] * dt,
            "ctrl_lost_s": m["ctrl_lost"] * dt, "ctrl_deadline_miss_pct": 100 * m["ctrl_lat_miss"] / k,
            "video_quality": quality_sum / k, "avg_power_w": power_sum / k,
            "soft": ctrl.counters["soft_switch"], "hard": ctrl.counters["hard_switch"],
            "emergency": ctrl.counters["emergency"], "seamless": ctrl.counters["seamless_switch"]}


def _run_job(args):
    return run_episode(*args)


def _sweep_job(j):
    return run_episode("cost", j[2], cfg=j[3])


def bench(policies: Sequence[str], seeds: int, workers: int = 0) -> None:
    import concurrent.futures as cf
    jobs = [(p, s) for p in policies for s in range(seeds)]
    if workers != 1:
        with cf.ProcessPoolExecutor(max_workers=workers or None) as ex:
            res = list(ex.map(_run_job, jobs))
    else:
        res = [_run_job(j) for j in jobs]
    keys = ["blind_s", "unavoidable_blind_s", "ctrl_lost_s", "ctrl_deadline_miss_pct", "video_quality",
            "avg_power_w", "soft", "hard", "emergency", "seamless"]
    print(f"\nmean over {seeds} seeds, run length ~{res[0]['duration_s']:.0f}s "
          f"(blind = neither reliable control nor usable video; unavoidable = no link could provide either)")
    print(f"{'policy':<12}" + "".join(f"{k:>24}" for k in keys))
    for p in policies:
        rs = [r for r in res if r["policy"] == p]
        print(f"{p:<12}" + "".join(f"{statistics.mean(r[k] for r in rs):>24.3f}" for k in keys))
    print("\nper-seed blind_s:")
    for p in policies:
        print(f"  {p:<12}", [round(r["blind_s"], 1) for r in res if r["policy"] == p])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "bench"
    if cmd == "dump-config":
        print(json.dumps(HandoverConfig().to_dict(), indent=2))
        return
    seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else 3
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 0
    pols = argv[argv.index("--policies") + 1].split(",") if "--policies" in argv else \
        ["reactive", "bonding", "rule", "threshold", "cost_trend", "cost_nomap", "cost"]
    if cmd == "sweep":
        # Pareto sweep of the blind-time weight and the safety margin: blind time vs power vs video quality
        import concurrent.futures as cf
        grid = [(wb, mg) for wb in (30.0, 100.0, 300.0, 1000.0) for mg in (1.0, 3.0, 5.0)]
        jobs = []
        for wb, mg in grid:
            c = HandoverConfig()
            c.weights.blind, c.safety.snr_margin_db = wb, mg
            for sd in range(seeds):
                jobs.append((wb, mg, sd, c))
        with cf.ProcessPoolExecutor(max_workers=workers or None) as ex:
            res = list(ex.map(_sweep_job, jobs))
        print(f"{'w_blind':>8}{'margin_dB':>10}{'blind_s':>9}{'ctrl_lost_s':>12}{'ctrl_miss%':>11}{'video_q':>9}{'power_W':>9}")
        for wb, mg in grid:
            rs = [r for j, r in zip(jobs, res) if j[0] == wb and j[1] == mg]
            f = lambda k: statistics.mean(r[k] for r in rs)
            print(f"{wb:>8.0f}{mg:>10.1f}{f('blind_s'):>9.2f}{f('ctrl_lost_s'):>12.2f}"
                  f"{f('ctrl_deadline_miss_pct'):>11.3f}{f('video_quality'):>9.3f}{f('avg_power_w'):>9.2f}")
        return
    if cmd == "demo":
        r = run_episode(pols[-1], 0, verbose=True)
        print(json.dumps(r, indent=2))
        return
    bench(pols, seeds, workers)


if __name__ == "__main__":
    main()
