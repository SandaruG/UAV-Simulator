#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone_multilink.py  --  Part 2: Drone + Multi-Link Radio Model
=================================================================
Standalone (numpy only) simulator of a quadrotor with a heterogeneous radio
suite: direct RF (2.4 GHz / optional 900 MHz), cellular (LTE/5G-like), LEO
satellite (Starlink-like Ku-band phased array) and optional Wi-Fi / mesh.

WHAT IS MODELLED
----------------
Drone      6-DoF rigid body (RK4), quaternion attitude, 4 rotors with first-order
           motor lag, quadratic body drag against wind, gyroscopic term, cascaded
           position->velocity->attitude->rate controller, X-mixer, waypoint
           follower + trajectory forecast, optional link-loss failsafe (RTH).
Battery    OCV(SoC) curve, SoC/temperature dependent R0, 1-RC polarisation,
           Peukert-like loss, current limit, thermal model, voltage sag ->
           thrust loss and radio PA derating/brown-out. Radio power is part of
           the electrical load, so radios that are on shorten endurance.
Radios     Every radio has: OFF/BOOTING/STANDBY/ACQUIRING/CONNECTED state
           machine (boot, search, wake times and power per state), link budget
           from environment-supplied propagation (path loss, correlated
           shadowing, Rician fading, obstruction, rain, interference/jamming),
           per-antenna 3-D patterns + polarisation + airframe blockage in the
           drone body frame, separate UL (drone->GS: video+telemetry) and DL
           (GS->drone: control) SNR, lagging rate adaptation (MCS table) and
           logistic PER, Gilbert-Elliott burst loss, HARQ/ARQ, two-class fluid
           queue (control strict priority over video), latency model (access,
           queue, serialisation, propagation, backhaul, retransmissions,
           handover interruptions), PA efficiency vs back-off, thermal
           throttling, voltage derating, radio-link-failure detection.
           Cellular: tower sectors with down-tilt, A3 handover, UL power
           control, altitude interference. LEO: Walker constellation, serving
           satellite selection/handover, phased-array scan loss, slant-range
           delay. Wi-Fi: roaming, CSMA contention, multi-hop mesh capacity.
Traffic    control (GS->drone, small packets, high priority), telemetry
           (drone->GS), adaptive video (frames, FEC, GOP / keyframe
           dependency, playout deadline, encoder lag + feedback delay).
Metrics    control_lost / video_lost / BLIND time (neither), latency
           percentiles, deadline violations, video quality, energy per radio.

EXTENDED LINK CATALOG (all selectable/switchable; new ones are enabled=False by default -> --enable name,... | all)
  5.8 GHz   fpv58_analog (FM, snow), ocusync (OFDM, 2.4/5.8 adaptive hopping), walksnail, hdzero (fixed mode), openhd58, openipc58
  2.4 GHz   rf24 (legacy), elrs24 (LoRa/FLRC control, FHSS)     sub-GHz: rf900 (legacy), crossfire868 (duty-cycle/ERP limits)
  L/S/C     fpv12g_analog, lr12g (1.2-1.3), lr17g, lr33g, lr72g (7.2 GHz emerging)   SDR: sdr_ofdm (retune(), custom MCS)
  cellular  cell (legacy LTE), lte_a/lte_b (two SIMs), nr_a/nr_b (5G n78 on NR sites), sim_bond (multi-SIM bonding)
  satellite leo (Starlink-like, legacy), oneweb_ku, iridium_certus (L), globalstar_s (S), inmarsat_l (GEO L), geo_ku, geo_ka
  mesh_manet (multi-hop relays incl. loitering relay UAV)   fiber_tether (spool mass, length limit, jam-proof)
  Codec layer (suite.set_codec): standard | ladder | neural | event | key_object | auto (capacity-driven)
  Legacy aliases: wifi_5g->wifi, lte->cell, sub1ghz->rf900.   python drone_multilink.py --catalog | --range-sweep

PART 3 HOOKS: suite.observation(), suite.history(), sim.trajectory_forecast(),
suite.set_routing(), radio.request('on'|'standby'|'off'), radio.set_tx_power().

Run:  python drone_multilink.py --duration 240 --policy reactive --csv out.csv
      python drone_multilink.py --compare          (reactive vs bonding)
      python drone_multilink.py --dump-config cfg.json / --config cfg.json
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time as _time
import typing
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# 0. Constants and helpers
# =============================================================================
C0 = 299_792_458.0
G0 = 9.80665
RHO_AIR = 1.225
INF_T = 1e12


def db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-300))


def undb(x: float) -> float:
    return 10.0 ** (x / 10.0)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def band_class(f_hz: float) -> str:
    """Interference/noise-rise class of a frequency (used when interference_class='auto')."""
    if f_hz < 1.0e9: return "sub_ghz"
    if f_hz < 1.45e9: return "l_band_1g2"
    if f_hz < 2.0e9: return "l_band_1g7"
    if f_hz < 2.7e9: return "ism24"
    if f_hz < 4.2e9: return "c_band_3g3"
    if f_hz < 6.5e9: return "ism58"
    return "c_band_7g2"


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def fspl_db(d_m: float, f_hz: float) -> float:
    return 20 * math.log10(max(d_m, 1.0)) + 20 * math.log10(f_hz) - 147.55


def thermal_noise_dbm(bw_hz: float, nf_db: float) -> float:
    return -174.0 + 10 * math.log10(bw_hz) + nf_db


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def q_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def q_from_euler(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r/2), math.sin(r/2), math.cos(p/2),
                              math.sin(p/2), math.cos(y/2), math.sin(y/2))
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def R_to_euler(R):
    return (math.atan2(R[2, 1], R[2, 2]), -math.asin(clamp(R[2, 0], -1, 1)),
            math.atan2(R[1, 0], R[0, 0]))


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


# ---- generic dataclass <-> dict (all parameters are controllable via JSON) ---
def _coerce(ty, v):
    if v is None:
        return None
    origin, args = typing.get_origin(ty), typing.get_args(ty)
    if dataclasses.is_dataclass(ty) and isinstance(v, dict):
        return dc_from_dict(ty, v)
    if origin is typing.Union:
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, v)
            except Exception:
                pass
        return v
    if origin is list:
        return [_coerce(args[0], x) for x in v] if args else list(v)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], x) for x in v)
        return tuple(_coerce(a, x) for a, x in zip(args, v))
    if origin is dict:
        return {k: _coerce(args[1], x) for k, x in v.items()} if args else dict(v)
    return v


def dc_from_dict(cls, d):
    hints = typing.get_type_hints(cls)
    kw = {f.name: _coerce(hints[f.name], d[f.name])
          for f in dataclasses.fields(cls) if f.name in d}
    return cls(**kw)


def apply_overrides(obj, ov: dict):
    """Recursive in-place update. For SimConfig, ov['radios'] is {name: {...}}."""
    hints = typing.get_type_hints(type(obj))
    for k, v in ov.items():
        if isinstance(obj, SimConfig) and k == "radios" and isinstance(v, dict):
            for r in obj.radios:
                if r.name in v:
                    apply_overrides(r, v[r.name])
            continue
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            apply_overrides(cur, v)
        else:
            setattr(obj, k, _coerce(hints[k], v))


# =============================================================================
# 1. Drone: parameters, state, dynamics, controller, trajectory
# =============================================================================
@dataclass
class DroneParams:
    mass_kg: float = 1.8
    inertia: Tuple[float, float, float] = (0.0123, 0.0123, 0.0224)
    arm_m: float = 0.225
    max_thrust_per_motor_n: float = 12.0
    max_motor_w: float = 1100.0          # rad/s
    yaw_coeff_m: float = 0.016           # kQ/kT
    motor_tau_s: float = 0.03
    rotor_inertia: float = 1.0e-5
    cda_m2: Tuple[float, float, float] = (0.04, 0.04, 0.08)   # body-axis Cd*A
    rotor_radius_m: float = 0.127
    figure_of_merit: float = 0.65
    motor_esc_eff: float = 0.80
    motor_idle_w: float = 1.5
    avionics_w: float = 12.0             # FC + companion computer
    payload_w: float = 6.0               # camera, gimbal (encoder power is in traffic)


@dataclass
class ControllerParams:
    kp_pos: float = 0.9
    kp_vel: float = 2.5
    ki_vel: float = 0.6
    kp_att: float = 7.0
    kp_rate: float = 18.0
    max_h_speed: float = 20.0
    max_up_speed: float = 4.0
    max_down_speed: float = 3.0
    max_tilt_deg: float = 35.0
    int_limit: float = 2.0


@dataclass
class DroneState:
    t: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_w: np.ndarray = field(default_factory=lambda: np.zeros(4))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @property
    def euler(self):
        return R_to_euler(self.R)


class Quadrotor:
    def __init__(self, p: DroneParams, c: ControllerParams, pos0):
        self.p, self.c = p, c
        self.kT = p.max_thrust_per_motor_n / p.max_motor_w ** 2
        self.J = np.array(p.inertia, float)
        self.cda = np.array(p.cda_m2, float)
        ang = np.deg2rad([45, 135, 225, 315])
        self.mxy = p.arm_m * np.stack([np.cos(ang), np.sin(ang)], 1)
        self.spin = np.array([1.0, -1.0, 1.0, -1.0])
        A = np.zeros((4, 4))
        A[0, :] = 1
        A[1, :] = self.mxy[:, 1]
        A[2, :] = -self.mxy[:, 0]
        A[3, :] = self.spin * p.yaw_coeff_m
        self.A_inv = np.linalg.inv(A)
        self.disk_area = math.pi * p.rotor_radius_m ** 2
        self.state = DroneState(pos=np.array(pos0, float))
        self.int_v = np.zeros(3)
        self.thrust_scale = 1.0           # set by battery voltage sag
        self.mass = p.mass_kg             # airframe + carried spools (updated by the sim)
        self.motor_power_w = 0.0
        self.crashed = False

    # ---- controller ---------------------------------------------------------
    def control(self, dt, sp_pos, sp_vel, sp_yaw):
        c, p, s = self.c, self.p, self.state
        v_sp = sp_vel + c.kp_pos * (sp_pos - s.pos)
        h = np.linalg.norm(v_sp[:2])
        if h > c.max_h_speed:
            v_sp[:2] *= c.max_h_speed / h
        v_sp[2] = clamp(v_sp[2], -c.max_down_speed, c.max_up_speed)
        ev = v_sp - s.vel
        self.int_v = np.clip(self.int_v + ev * dt, -c.int_limit, c.int_limit)
        a = c.kp_vel * ev + c.ki_vel * self.int_v
        amax = G0 * math.tan(math.radians(c.max_tilt_deg))
        ah = np.linalg.norm(a[:2])
        if ah > amax:
            a[:2] *= amax / ah
        F = self.mass * (a + np.array([0, 0, G0]))
        R = s.R
        T = max(0.0, float(F @ R[:, 2]))
        zd = unit(F)
        xc = np.array([math.cos(sp_yaw), math.sin(sp_yaw), 0.0])
        yd = unit(np.cross(zd, xc))
        xd = np.cross(yd, zd)
        Rd = np.stack([xd, yd, zd], 1)
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        w_des = -c.kp_att * eR
        tau = self.J * (c.kp_rate * (w_des - s.omega)) + np.cross(s.omega, self.J * s.omega)
        Ti = self.A_inv @ np.array([T, tau[0], tau[1], tau[2]])
        return np.clip(Ti, 0.0, p.max_thrust_per_motor_n)

    # ---- dynamics -----------------------------------------------------------
    def _deriv(self, x, Tt, tau, h, wind):
        v, q, w = x[3:6], x[6:10], x[10:13]
        R = q_to_R(q)
        vrb = R.T @ (v - wind)
        Fd = R @ (-0.5 * RHO_AIR * self.cda * np.abs(vrb) * vrb)
        acc = (R[:, 2] * Tt + Fd) / self.mass - np.array([0, 0, G0])
        qd = 0.5 * q_mul(q, np.array([0.0, w[0], w[1], w[2]]))
        gyro = np.cross(w, np.array([0, 0, h]))
        wd = (tau - np.cross(w, self.J * w) - gyro) / self.J
        return np.concatenate([v, acc, qd, wd])

    def step(self, dt, thrust_cmd, wind):
        p, s = self.p, self.state
        tmax = p.max_thrust_per_motor_n * self.thrust_scale
        w_cmd = np.sqrt(np.clip(thrust_cmd, 0.0, tmax) / self.kT)
        s.motor_w += (w_cmd - s.motor_w) * (1 - math.exp(-dt / p.motor_tau_s))
        T = self.kT * s.motor_w ** 2
        Tt = float(T.sum())
        tau = np.array([(self.mxy[:, 1] * T).sum(), -(self.mxy[:, 0] * T).sum(),
                        (self.spin * p.yaw_coeff_m * T).sum()])
        h = p.rotor_inertia * float((self.spin * s.motor_w).sum())
        p_ind = (T ** 1.5) / (math.sqrt(2 * RHO_AIR * self.disk_area) * p.figure_of_merit * p.motor_esc_eff)
        self.motor_power_w = float(p_ind.sum() + 4 * p.motor_idle_w)
        x = np.concatenate([s.pos, s.vel, s.quat, s.omega])
        f = lambda xx: self._deriv(xx, Tt, tau, h, wind)
        k1 = f(x); k2 = f(x + 0.5*dt*k1); k3 = f(x + 0.5*dt*k2); k4 = f(x + dt*k3)
        x = x + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        s.acc = k1[3:6].copy()
        s.pos, s.vel, s.quat, s.omega = x[0:3].copy(), x[3:6].copy(), x[6:10] / np.linalg.norm(x[6:10]), x[10:13].copy()
        if s.pos[2] <= 0.0:                         # ground contact
            s.pos[2] = 0.0
            s.vel[2] = max(0.0, s.vel[2])
            if Tt < self.mass * G0 * 0.97:
                s.vel[:] = 0.0
                s.omega[:] = 0.0
                _, _, yaw = R_to_euler(q_to_R(s.quat))
                s.quat = q_from_euler(0, 0, yaw)
        s.R = q_to_R(s.quat)
        s.t += dt
        roll, pitch, _ = R_to_euler(s.R)
        if s.pos[2] > 1.0 and (abs(roll) > 1.9 or abs(pitch) > 1.9):
            self.crashed = True


@dataclass
class Waypoint:
    pos: Tuple[float, float, float] = (0.0, 0.0, 30.0)
    speed_mps: Optional[float] = None
    hold_s: float = 0.0
    yaw_deg: Optional[float] = None


class WaypointFollower:
    """Moves a 'carrot' along the polyline; the position controller chases it."""

    def __init__(self, wps: List[Waypoint], default_speed=12.0, acc=2.5, max_climb=3.0,
                 max_descent=2.0, lead_max=6.0, loop=False):
        self.wps, self.default_speed, self.acc = list(wps), default_speed, acc
        self.max_climb, self.max_descent, self.lead_max, self.loop = max_climb, max_descent, lead_max, loop
        self.idx, self.c, self.v, self.hold, self.yaw, self.done = 0, None, 0.0, 0.0, 0.0, False

    def step(self, dt, pos):
        if self.c is None:
            self.c = np.array(pos, float)
        if self.done or not self.wps:
            return self.c.copy(), np.zeros(3), self.yaw
        wp = self.wps[self.idx]
        tgt = np.array(wp.pos, float)
        vec = tgt - self.c
        dist = float(np.linalg.norm(vec))
        if dist < 0.3:
            self.c, self.v = tgt.copy(), 0.0
            self.hold += dt
            if self.hold >= wp.hold_s:
                self.hold = 0.0
                self.idx += 1
                if self.idx >= len(self.wps):
                    if self.loop:
                        self.idx = 0
                    else:
                        self.idx, self.done = len(self.wps) - 1, True
            return self.c.copy(), np.zeros(3), self.yaw
        d = vec / dist
        speed = wp.speed_mps or self.default_speed
        if abs(d[2]) > 1e-3:
            speed = min(speed, (self.max_climb if d[2] > 0 else self.max_descent) / abs(d[2]))
        v_des = min(speed, math.sqrt(2 * self.acc * dist))
        self.v += clamp(v_des - self.v, -2 * self.acc * dt, self.acc * dt)
        if np.linalg.norm(self.c - pos) > self.lead_max:
            self.v = min(self.v, 0.3)
        self.c = self.c + d * min(self.v * dt, dist)
        if wp.yaw_deg is not None:
            self.yaw = math.radians(wp.yaw_deg)
        elif math.hypot(d[0], d[1]) > 0.3:
            self.yaw = math.atan2(d[1], d[0])
        return self.c.copy(), d * self.v, self.yaw

    def forecast(self, horizon_s: float, step_s: float) -> np.ndarray:
        """Predicted positions (N,3) assuming remaining polyline at cruise speed."""
        if self.c is None:
            return np.zeros((0, 3))
        n = int(horizon_s / step_s)
        out, p, i = [], self.c.copy(), self.idx
        for _ in range(n):
            rem = (self.wps[i].speed_mps or self.default_speed) * step_s if not self.done else 0.0
            while rem > 1e-6 and not self.done:
                tgt = np.array(self.wps[i].pos, float)
                dd = float(np.linalg.norm(tgt - p))
                if dd <= rem:
                    p, rem = tgt, rem - dd
                    i += 1
                    if i >= len(self.wps):
                        i = 0 if self.loop else len(self.wps) - 1
                        if not self.loop:
                            break
                else:
                    p = p + (tgt - p) / dd * rem
                    rem = 0.0
            out.append(p.copy())
        return np.array(out)


# =============================================================================
# 2. Battery
# =============================================================================
@dataclass
class BatteryParams:
    cells: int = 6
    capacity_ah: float = 5.2
    soc0: float = 1.0
    health: float = 1.0                  # state of health (capacity & resistance)
    r_cell_mohm: float = 7.0
    r1_cell_mohm: float = 4.0
    tau1_s: float = 25.0
    max_current_a: float = 120.0
    cutoff_v_cell: float = 3.30
    peukert_k: float = 1.04
    rated_current_a: float = 5.2
    cth_j_per_k: float = 350.0
    h_w_per_k: float = 1.2
    h_airspeed: float = 0.12             # extra convective cooling per m/s
    ocv_soc: Tuple[float, ...] = (0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)
    ocv_v: Tuple[float, ...] = (3.0, 3.4, 3.55, 3.66, 3.72, 3.77, 3.81, 3.86, 3.92, 3.99, 4.08, 4.2)


class Battery:
    def __init__(self, p: BatteryParams, ambient_c: float):
        self.p = p
        self.soc, self.v1, self.temp = p.soc0, 0.0, ambient_c
        self.v, self.i, self.power_limited, self.depleted = p.cells * 4.0, 0.0, False, False
        self.energy_wh = 0.0

    def ocv(self):
        return self.p.cells * float(np.interp(self.soc, self.p.ocv_soc, self.p.ocv_v))

    def r0(self):
        p = self.p
        r = p.cells * p.r_cell_mohm * 1e-3 * (1 + 2.0 * (1 - self.soc) ** 3)
        return r * (1 + 0.025 * (25 - self.temp)) / max(p.health, 0.3)

    def step(self, dt, P_w, ambient_c, airspeed=0.0):
        p = self.p
        ocv, R = self.ocv(), self.r0()
        Vo = ocv - self.v1
        disc = Vo * Vo - 4 * R * P_w
        self.power_limited = False
        if disc < 0:
            I = Vo / (2 * R)
            self.power_limited = True
        else:
            I = (Vo - math.sqrt(disc)) / (2 * R)
        if I > p.max_current_a:
            I, self.power_limited = p.max_current_a, True
        self.v = Vo - I * R
        self.i = I
        R1 = p.cells * p.r1_cell_mohm * 1e-3
        a = math.exp(-dt / p.tau1_s)
        self.v1 = self.v1 * a + I * R1 * (1 - a)
        dAh = I * dt / 3600.0 * max(I / p.rated_current_a, 0.2) ** (p.peukert_k - 1)
        self.soc = max(0.0, self.soc - dAh / (p.capacity_ah * p.health))
        self.energy_wh += self.v * I * dt / 3600.0
        heat = I * I * R + self.v1 ** 2 / max(R1, 1e-6)
        h = p.h_w_per_k * (1 + p.h_airspeed * airspeed)
        self.temp += dt * (heat - h * (self.temp - ambient_c)) / p.cth_j_per_k
        self.depleted = self.soc <= 0.0 or self.v < p.cutoff_v_cell * p.cells * 0.98
        return self.v

    def thrust_scale(self):
        return clamp((self.v / (self.p.cells * 3.7)) ** 2, 0.4, 1.15)


# =============================================================================
# 3. Environment (propagation, wind, jammers, rain, assets, constellation)
# =============================================================================
@dataclass
class Building:
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 50.0
    y1: float = 50.0
    height: float = 40.0
    wall_loss_db: float = 10.0
    loss_db_per_m: float = 0.5


@dataclass
class Hill:
    x: float = 0.0
    y: float = 0.0
    height: float = 50.0
    sigma: float = 200.0


@dataclass
class Jammer:
    name: str = "jammer"
    pos: Tuple[float, float, float] = (0.0, 0.0, 2.0)
    power_dbm: float = 30.0
    gain_dbi: float = 0.0
    f_hz: float = 2.4e9
    bw_hz: float = 40e6
    t_on: float = 0.0
    t_off: float = INF_T
    pulse_period_s: float = 0.0
    duty: float = 1.0
    mode: str = "noise"              # noise (barrage/tone within f_hz+-bw/2) | sweep
    sweep_lo_hz: float = 0.0
    sweep_hi_hz: float = 0.0
    sweep_period_s: float = 1.0


@dataclass
class RainEvent:
    t0: float = 0.0
    t1: float = 60.0
    rate_mm_h: float = 20.0


@dataclass
class CellTower:
    id: str = "T0"
    pos: Tuple[float, float, float] = (0.0, 0.0, 30.0)
    tx_power_dbm: float = 43.0
    freq_hz: float = 2.1e9
    azimuths_deg: Tuple[float, ...] = (0.0, 120.0, 240.0)
    downtilt_deg: float = 6.0
    peak_gain_dbi: float = 17.0
    hbw_deg: float = 65.0
    vbw_deg: float = 10.0
    load: float = 0.5
    operators: Tuple[str, ...] = ("A", "B")      # operators sharing the site
    rats: Tuple[str, ...] = ("lte",)           # radio access technologies deployed (lte, nr)

    def gain_toward(self, p) -> float:
        dx, dy, dz = p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2]
        hd = math.hypot(dx, dy)
        az = math.degrees(math.atan2(dx, dy))
        el = math.degrees(math.atan2(dz, max(hd, 1.0)))
        best = -1e9
        for a in self.azimuths_deg:
            da = ((az - a + 180) % 360) - 180
            A = min(12 * (da / self.hbw_deg) ** 2, 25.0) + min(12 * ((el + self.downtilt_deg) / self.vbw_deg) ** 2, 20.0)
            best = max(best, self.peak_gain_dbi - min(A, 25.0))
        return best


@dataclass
class WifiNode:
    id: str = "AP0"
    pos: Tuple[float, float, float] = (30.0, 0.0, 3.0)
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hops: int = 1
    channel_busy: float = 0.2
    backhaul_latency_ms: float = 2.0
    tx_power_dbm: float = 20.0
    gain_dbi: float = 5.0


@dataclass
class GroundStation:
    pos: Tuple[float, float, float] = (0.0, 0.0, 2.0)


@dataclass
class ConstellationParams:
    n_planes: int = 72
    sats_per_plane: int = 22
    alt_km: float = 550.0
    incl_deg: float = 53.0
    phasing_f: int = 39
    raan_span_deg: float = 360.0          # 360 = Walker-delta, 180 = polar Walker-star
    fixed_lons_deg: Tuple[float, ...] = ()   # non-empty -> geostationary sats at these longitudes


def _default_towers() -> List[CellTower]:
    out = []
    for i in range(-1, 3):
        for j in range(-2, 3):
            if (i + j) % 5 == 3:          # coverage holes
                continue
            x, y = i * 2500 + (j % 2) * 1250 + 120 * math.sin(7 * i + 3 * j), j * 2165 + 150 * math.cos(5 * i + 2 * j)
            out.append(CellTower(id=f"T{i}_{j}", pos=(x, y, 30.0),
                                 rats=("lte", "nr") if (i * 5 + j) % 3 != 0 else ("lte",)))
    return out


@dataclass
class MeshNode:
    id: str = "R0"
    pos: Tuple[float, float, float] = (0.0, 0.0, 10.0)
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orbit_radius_m: float = 0.0           # >0: relay UAV loitering in a circle around pos
    orbit_period_s: float = 60.0
    tx_power_dbm: float = 27.0
    gain_dbi: float = 3.0
    is_gateway: bool = False              # wired/backhauled to the ground station


def _default_mesh_nodes() -> List[MeshNode]:
    return [MeshNode("GW", (40, 10, 8), is_gateway=True, gain_dbi=6.0),
            MeshNode("R1", (1100, 250, 25)), MeshNode("R2", (2200, 300, 30)), MeshNode("R3", (3000, -100, 20)),
            MeshNode("UAVR", (1700, 500, 110), orbit_radius_m=120, orbit_period_s=45, tx_power_dbm=30)]


def _default_extra_constellations() -> Dict[str, ConstellationParams]:
    return {"oneweb": ConstellationParams(12, 49, 1200.0, 87.9, 1, 180.0),
            "iridium": ConstellationParams(6, 11, 780.0, 86.4, 0, 180.0),
            "globalstar": ConstellationParams(8, 6, 1414.0, 52.0, 1, 360.0),
            "geo": ConstellationParams(1, 3, 35786.0, 0.0, 0, 360.0, (95.0, 125.0, 155.0))}


@dataclass
class EnvironmentParams:
    lat_deg: float = 45.0
    lon_deg: float = 126.0
    ambient_c: float = 25.0
    lapse_c_per_m: float = 0.0065
    ground_station: GroundStation = field(default_factory=GroundStation)
    towers: List[CellTower] = field(default_factory=_default_towers)
    wifi_nodes: List[WifiNode] = field(default_factory=lambda: [
        WifiNode(id="AP0", pos=(30, 0, 3), hops=1, channel_busy=0.25),
        WifiNode(id="MESH1", pos=(900, 150, 6), hops=2, channel_busy=0.15, backhaul_latency_ms=6)])
    buildings: List[Building] = field(default_factory=lambda: [
        Building(480, 60, 560, 140, 45), Building(600, 110, 660, 170, 30), Building(300, -180, 360, -120, 25)])
    hills: List[Hill] = field(default_factory=lambda: [Hill(1800, -250, 80, 260)])
    jammers: List[Jammer] = field(default_factory=lambda: [
        Jammer(name="ism_interferer", pos=(2400, 500, 3), power_dbm=33, f_hz=2.44e9, bw_hz=30e6,
               t_on=110.0, t_off=150.0, pulse_period_s=0.4, duty=0.6)])
    rain: List[RainEvent] = field(default_factory=lambda: [RainEvent(70, 130, 25.0)])
    constellation: ConstellationParams = field(default_factory=ConstellationParams)
    noise_rise_db: Dict[str, float] = field(default_factory=lambda: {
        "direct": 3.0, "wifi": 8.0, "cellular": 0.0, "sat": 0.0,
        # band classes (used by radios with interference_class="auto")
        "sub_ghz": 4.0, "l_band_1g2": 1.5, "l_band_1g7": 1.0, "ism24": 6.0, "c_band_3g3": 2.0, "ism58": 5.0,
        "c_band_7g2": 0.5, "fiber": 0.0})
    wind_mean_mps: Tuple[float, float, float] = (3.0, 1.0, 0.0)
    turb_sigma_mps: float = 1.0
    turb_tau_s: float = 3.0
    gust_rate_hz: float = 0.02
    gust_amp_mps: float = 4.0
    gust_dur_s: float = 4.0
    shadow_sigma_los_db: float = 3.0
    shadow_sigma_nlos_db: float = 7.0
    shadow_corr_dist_m: float = 30.0
    k_los_db: float = 8.0
    ground_reflection_coeff: float = 0.6
    clutter_db_max: float = 6.0
    rain_height_m: float = 3000.0
    gas_zenith_db_12ghz: float = 0.25
    extra_constellations: Dict[str, ConstellationParams] = field(default_factory=_default_extra_constellations)
    mesh_nodes: List[MeshNode] = field(default_factory=_default_mesh_nodes)
    wall_loss_exp: float = 0.6          # building penetration loss ~ (f/2.4GHz)^exp
    clutter_exp: float = 0.4            # foliage/ground clutter ~ (f/2.4GHz)^exp
    iono_scint_db: float = 1.0          # extra L/S-band scintillation sigma (sat links < 4 GHz)


@dataclass
class PropSample:
    path_loss_db: float
    shadow_db: float
    fading_db: float
    los: bool
    distance_m: float
    elev_deg: float
    rain_db: float = 0.0

    @property
    def total_loss_db(self) -> float:
        return self.path_loss_db + self.shadow_db - self.fading_db


_RAIN_F = np.array([1, 2, 4, 6, 8, 10, 12, 15, 20, 30.0])
_RAIN_K = np.array([3.87e-5, 1.54e-4, 6.5e-4, 1.75e-3, 4.54e-3, 1.01e-2, 1.88e-2, 3.67e-2, 7.51e-2, 0.187])
_RAIN_A = np.array([.912, .963, 1.121, 1.308, 1.327, 1.276, 1.217, 1.154, 1.099, 1.021])


class LEOConstellation:
    MU, RE, WE = 3.986004418e14, 6371e3, 7.2921159e-5

    def __init__(self, cp: ConstellationParams, lat_deg, lon_deg):
        P, S = cp.n_planes, cp.sats_per_plane
        if cp.fixed_lons_deg:                                   # geostationary
            self.raan = np.zeros(len(cp.fixed_lons_deg))
            self.u0 = np.radians(np.array(cp.fixed_lons_deg, float))
        else:
            k, j = np.repeat(np.arange(P), S), np.tile(np.arange(S), P)
            self.raan = np.radians(cp.raan_span_deg) * k / P
            self.u0 = 2 * np.pi * (j / S + k * cp.phasing_f / (P * S))
        self.inc = math.radians(cp.incl_deg)
        self.r = self.RE + cp.alt_km * 1e3
        self.n = math.sqrt(self.MU / self.r ** 3)
        la, lo = math.radians(lat_deg), math.radians(lon_deg)
        self.E = np.array([-math.sin(lo), math.cos(lo), 0.0])
        self.N = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
        self.U = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
        self.obs0 = self.RE * self.U
        self._ck = None

    def _ecef(self, t, idx):
        u = self.u0[idx] + self.n * t
        xo, yo = self.r * np.cos(u), self.r * np.sin(u)
        ci, si = math.cos(self.inc), math.sin(self.inc)
        cO, sO = np.cos(self.raan[idx]), np.sin(self.raan[idx])
        x, y, z = cO * xo - sO * ci * yo, sO * xo + cO * ci * yo, si * yo
        th = -self.WE * t
        c, s = math.cos(th), math.sin(th)
        return np.stack([c * x - s * y, s * x + c * y, z], -1)

    def _rel(self, t, pos, idx):
        obs = self.obs0 + pos[0] * self.E + pos[1] * self.N + pos[2] * self.U
        d = self._ecef(t, idx) - obs
        enu = np.stack([d @ self.E, d @ self.N, d @ self.U], -1)
        rng = np.linalg.norm(d, axis=-1)
        return enu, rng, np.degrees(np.arcsin(np.clip(enu[..., 2] / rng, -1, 1)))

    def visible(self, t, pos, min_elev):
        key = (int(t * 5), int(pos[0] // 200), int(pos[1] // 200))
        if self._ck != key:
            _, rng, el = self._rel(t, pos, slice(None))
            self._cv, self._ck = (rng, el), key
        rng, el = self._cv
        idx = np.nonzero(el >= min_elev)[0]
        return idx, el[idx], rng[idx]

    def single(self, idx, t, pos):
        enu, rng, el = self._rel(t, pos, np.array([idx]))
        return enu[0], float(rng[0]), float(el[0])


class Environment:
    def __init__(self, p: EnvironmentParams, seed: int = 1):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.gs = np.array(p.ground_station.pos, float)
        self.towers, self.wifi_nodes, self.jammers = p.towers, p.wifi_nodes, p.jammers
        self.constellation = LEOConstellation(p.constellation, p.lat_deg, p.lon_deg)
        self.constellations = {"": self.constellation}
        for nm, cp in p.extra_constellations.items():
            self.constellations[nm] = LEOConstellation(cp, p.lat_deg, p.lon_deg)
        self.mesh_nodes = p.mesh_nodes
        self._hx, self._hy = np.array([h.x for h in p.hills]), np.array([h.y for h in p.hills])
        self._hh, self._hs = np.array([h.height for h in p.hills]), np.array([h.sigma for h in p.hills])
        self._turb, self._gust, self._gust_t, self._gust_d = np.zeros(3), np.zeros(3), 0.0, 0.0
        self.wind_now = np.array(p.wind_mean_mps, float)
        self._ls: Dict[tuple, dict] = {}

    # ---- terrain / wind / weather --------------------------------------------
    def terrain_height(self, x, y):
        if len(self._hh) == 0:
            return np.zeros_like(np.asarray(x, float))
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (self._hh * np.exp(-((x[..., None] - self._hx) ** 2 + (y[..., None] - self._hy) ** 2)
                                  / (2 * self._hs ** 2))).sum(-1)

    def temperature_c(self, alt):
        return self.p.ambient_c - self.p.lapse_c_per_m * alt

    def step(self, t, dt):
        p = self.p
        self._turb += -self._turb / p.turb_tau_s * dt + p.turb_sigma_mps * math.sqrt(2 * dt / p.turb_tau_s) * self.rng.standard_normal(3)
        if self._gust_d <= 0 and self.rng.random() < p.gust_rate_hz * dt:
            self._gust = self.rng.rayleigh(p.gust_amp_mps / 1.25, 3) * self.rng.choice([-1, 1], 3) * np.array([1, 1, 0.3])
            self._gust_t, self._gust_d = 0.0, p.gust_dur_s
        g = np.zeros(3)
        if self._gust_d > 0:
            self._gust_t += dt
            g = self._gust * 0.5 * (1 - math.cos(2 * math.pi * self._gust_t / self._gust_d))
            if self._gust_t >= self._gust_d:
                self._gust_d = 0.0
        self.wind_now = np.array(p.wind_mean_mps) + self._turb + g

    def rain_rate(self, t):
        r = 0.0
        for e in self.p.rain:
            if e.t0 - 15 <= t <= e.t1 + 15:
                r += e.rate_mm_h * clamp(min(t - e.t0 + 15, e.t1 + 15 - t) / 15.0, 0, 1)
        return r

    def mesh_pos(self, n: MeshNode, t: float):
        pos = np.array(n.pos, float) + np.array(n.vel, float) * t
        if n.orbit_radius_m > 0:
            a = 2 * math.pi * t / n.orbit_period_s
            pos = pos + n.orbit_radius_m * np.array([math.cos(a), math.sin(a), 0.0])
        return pos

    def add_jammer(self, j: Jammer):
        self.jammers.append(j)

    # ---- propagation ----------------------------------------------------------
    def _rain_db(self, f_hz, path_km, R):
        if R <= 0 or path_km <= 0:
            return 0.0
        fg = f_hz / 1e9
        gam = float(np.interp(fg, _RAIN_F, _RAIN_K)) * R ** float(np.interp(fg, _RAIN_F, _RAIN_A))
        return gam * path_km / (1 + path_km / (35 * math.exp(-0.015 * R)))

    def _obstruction(self, pa, pb, f_hz):
        loss, los = 0.0, True
        d = pb - pa
        fscale = clamp((f_hz / 2.4e9) ** self.p.wall_loss_exp, 0.4, 3.0)
        for b in self.p.buildings:
            tmin, tmax, ok = 0.0, 1.0, True
            for ax, lo, hi in ((0, b.x0, b.x1), (1, b.y0, b.y1), (2, 0.0, b.height)):
                if abs(d[ax]) < 1e-9:
                    if pa[ax] < lo or pa[ax] > hi:
                        ok = False
                        break
                else:
                    t1, t2 = (lo - pa[ax]) / d[ax], (hi - pa[ax]) / d[ax]
                    if t1 > t2:
                        t1, t2 = t2, t1
                    tmin, tmax = max(tmin, t1), min(tmax, t2)
                    if tmin > tmax:
                        ok = False
                        break
            if ok:
                loss += fscale * (b.wall_loss_db + b.loss_db_per_m * (tmax - tmin) * float(np.linalg.norm(d)))
                los = False
        loss = min(loss, 55.0)
        if len(self._hh):
            dist = float(np.linalg.norm(d))
            s = np.linspace(0, 1, 26)[1:-1]
            pts = pa + np.outer(s, d)
            clear = pts[:, 2] - self.terrain_height(pts[:, 0], pts[:, 1])
            lam = C0 / f_hz
            nu = (-clear * np.sqrt(2.0 / (lam * s * (1 - s) * dist))).max() if dist > 1 else -9
            if nu > -0.78:
                loss += 6.9 + 20 * math.log10(math.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
            if nu > -0.2:
                los = False
        return loss, los

    def _two_ray_db(self, pa, pb, f_hz, circ=False):
        dh = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        if dh < 5:
            return 0.0
        ht = max(float(pa[2] - self.terrain_height(pa[0], pa[1])), 0.5)
        hr = max(float(pb[2] - self.terrain_height(pb[0], pb[1])), 0.5)
        d1, d2 = math.hypot(dh, ht - hr), math.hypot(dh, ht + hr)
        phi = 2 * math.pi * (d2 - d1) / (C0 / f_hz)
        g = -self.p.ground_reflection_coeff * (0.15 if circ else 1.0)   # CP antennas reject the flipped-hand ground bounce
        mag2 = (1 + g * math.cos(phi)) ** 2 + (g * math.sin(phi)) ** 2
        return clamp(10 * math.log10(max(mag2, 1e-3)), -25.0, 6.0)

    def sample_link(self, key, kind, f_hz, pa, pb, speed, t, elev_deg=None, circ=False, fade_tag="") -> PropSample:
        """Stateful propagation sample. Shadowing is per link (frequency independent); small-scale
        fading is per (link, fade_tag) so a hopping radio sees independent fading on each channel."""
        p = self.p
        st = self._ls.get(key)
        if st is None:
            z0 = self.rng.standard_normal()
            g0 = complex(*self.rng.standard_normal(2)) / math.sqrt(2)
            st = self._ls[key] = dict(t=t, z=z0, gd={"": (g0, t)}, cache=None, ctag=None)
        if st["cache"] is not None and st["t"] == t and st["ctag"] == fade_tag:
            return st["cache"]
        dt = max(t - st["t"], 0.0)
        d = float(np.linalg.norm(pb - pa))
        horiz = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        el = elev_deg if elev_deg is not None else math.degrees(math.atan2(pa[2] - pb[2], max(horiz, 1.0)))
        lam = C0 / f_hz
        R = self.rain_rate(t)
        rain_db = 0.0
        if kind == "sat":
            path = max(p.rain_height_m - pa[2], 0.0) / max(math.sin(math.radians(el)), 0.1) / 1000.0
            rain_db = self._rain_db(f_hz, path, R)
            gas = p.gas_zenith_db_12ghz * clamp((f_hz / 12e9) ** 1.2, 0.05, 3) / max(math.sin(math.radians(el)), 0.1)
            pl = fspl_db(d, f_hz) + gas + rain_db
            los, sig, K = True, 0.5 + (p.iono_scint_db if f_hz < 4e9 else 0.0), undb(15.0)
        elif kind == "cellular":
            pl_los = 1.0 / (1.0 + 9.61 * math.exp(-0.16 * (el - 9.61)))
            obs, _ = self._obstruction(pa, pb, f_hz) if len(self._hh) else (0.0, True)
            pl = fspl_db(d, f_hz) + 1.0 * pl_los + 20.0 * (f_hz / 2.1e9) ** 0.3 * (1 - pl_los) + obs
            rain_db = self._rain_db(f_hz, d / 1000.0, R)
            pl += rain_db
            los, sig, K = pl_los > 0.5, 3.0 + 5.0 * (1 - pl_los), undb(6.0 * pl_los) if pl_los > 0.3 else 0.0
        else:                                    # direct / wifi / mesh: terrestrial geometry, frequency-scaled
            obs, los = self._obstruction(pa, pb, f_hz)
            pl = fspl_db(d, f_hz) + obs
            if los:
                pl -= self._two_ray_db(pa, pb, f_hz, circ)
            pl += p.clutter_db_max * (f_hz / 2.4e9) ** p.clutter_exp * math.exp(-max(el, 0.0) / 8.0)
            rain_db = self._rain_db(f_hz, d / 1000.0, R)
            pl += rain_db
            sig = p.shadow_sigma_los_db if los else p.shadow_sigma_nlos_db
            K = undb(p.k_los_db) if los else 0.0
        # correlated shadowing (Gudmundson) and Rician fading (Gaussian-Jakes AR(1))
        rho_s = math.exp(-max(speed * dt, 1e-3) / p.shadow_corr_dist_m) if kind != "sat" else math.exp(-dt / 20.0)
        st["z"] = rho_s * st["z"] + math.sqrt(1 - rho_s ** 2) * self.rng.standard_normal()
        if fade_tag not in st["gd"]:
            st["gd"][fade_tag] = (complex(*self.rng.standard_normal(2)) / math.sqrt(2), t)
        g, tl = st["gd"][fade_tag]
        fd = max(speed / lam, 1.0)
        rho_f = math.exp(-(math.pi * fd * max(t - tl, 0.0)) ** 2)
        n = complex(*self.rng.standard_normal(2)) / math.sqrt(2)
        g = rho_f * g + math.sqrt(1 - rho_f ** 2) * n
        st["gd"][fade_tag] = (g, t)
        h = math.sqrt(K / (K + 1)) + math.sqrt(1 / (K + 1)) * g
        fade_db = clamp(20 * math.log10(max(abs(h), 1e-3)), -35.0, 8.0)
        out = PropSample(pl, st["z"] * sig, fade_db, bool(los), d, el, rain_db)
        st["t"], st["cache"], st["ctag"] = t, out, fade_tag
        return out

    def interference_mw(self, kind, f_hz, bw_hz, pos, t, jam_rej_db=0.0) -> float:
        """Noise-rise (band class) + jammers overlapping [f-bw/2, f+bw/2]; jam_rej_db models
        spreading/hopping/blanking that suppresses jammer energy (never the band noise rise)."""
        thermal_mw = undb(-174.0 + 10 * math.log10(bw_hz))
        I = thermal_mw * (undb(self.p.noise_rise_db.get(kind, 0.0)) - 1.0)
        rej = undb(-jam_rej_db)
        for j in self.jammers:
            if not (j.t_on <= t < j.t_off):
                continue
            if j.pulse_period_s > 0 and ((t - j.t_on) % j.pulse_period_s) > j.duty * j.pulse_period_s:
                continue
            fj = j.f_hz
            if j.mode == "sweep" and j.sweep_hi_hz > j.sweep_lo_hz:
                fj = j.sweep_lo_hz + (j.sweep_hi_hz - j.sweep_lo_hz) * ((t / max(j.sweep_period_s, 1e-3)) % 1.0)
            ov = min(f_hz + bw_hz / 2, fj + j.bw_hz / 2) - max(f_hz - bw_hz / 2, fj - j.bw_hz / 2)
            if ov <= 0:
                continue
            dd = float(np.linalg.norm(np.array(j.pos) - pos))
            prx_dbm = j.power_dbm + j.gain_dbi - fspl_db(dd, f_hz)
            I += undb(prx_dbm) * ov / j.bw_hz * rej
        return I


# =============================================================================
# 4. Antennas
# =============================================================================
@dataclass
class AntennaSpec:
    name: str = "ant"
    pattern: str = "dipole"          # isotropic | dipole | patch | sector | phased_array
    peak_gain_dbi: float = 2.15
    axis_body: Tuple[float, float, float] = (0.0, 0.0, 1.0)   # dipole axis / boresight (FLU frame)
    beamwidth_deg: float = 80.0
    polarization: str = "linear"      # linear | circular
    cable_loss_db: float = 0.5
    back_lobe_db: float = 20.0
    max_scan_deg: float = 60.0
    scan_exponent: float = 1.3
    block_axis_body: Optional[Tuple[float, float, float]] = None   # airframe/battery shadow cone
    block_halfangle_deg: float = 55.0
    block_loss_db: float = 8.0
    ripple_db: float = 1.0           # deterministic pattern ripple from airframe scattering
    handedness: str = ""             # rhcp | lhcp (circular only); opposite hands -> ~20 dB cross-pol loss


def antenna_gain_dbi(a: AntennaSpec, ub: np.ndarray) -> float:
    ax = unit(np.array(a.axis_body, float))
    c = clamp(float(ub @ ax), -1.0, 1.0)
    if a.pattern == "isotropic":
        g = a.peak_gain_dbi
    elif a.pattern == "dipole":
        g = a.peak_gain_dbi + 20 * math.log10(max(math.sqrt(max(0.0, 1 - c * c)), 0.06))
    elif a.pattern == "patch":
        hp = math.radians(a.beamwidth_deg / 2)
        n = math.log(0.5) / math.log(max(math.cos(hp), 1e-3))
        g = a.peak_gain_dbi + (10 * n * math.log10(c) if c > 1e-3 else -99)
        g = max(g, a.peak_gain_dbi - a.back_lobe_db)
    elif a.pattern == "sector":
        th = math.degrees(math.acos(c))
        g = a.peak_gain_dbi - min(12 * (th / a.beamwidth_deg) ** 2, a.back_lobe_db)
    else:                                  # phased_array
        th = math.degrees(math.acos(c))
        g = a.peak_gain_dbi - 40.0 if th > a.max_scan_deg else a.peak_gain_dbi + 10 * a.scan_exponent * math.log10(max(c, 1e-3))
    if a.ripple_db:
        g += a.ripple_db * math.sin(3 * math.atan2(ub[1], ub[0])) * math.cos(2 * math.asin(clamp(ub[2], -1, 1)))
    if a.block_axis_body is not None:
        cb = float(ub @ unit(np.array(a.block_axis_body, float)))
        edge = math.cos(math.radians(a.block_halfangle_deg))
        if cb > edge:
            g -= a.block_loss_db * clamp((cb - edge) / 0.15, 0.0, 1.0)
    return g - a.cable_loss_db


def polarization_loss_db(a: AntennaSpec, R, u_w, peer_pol: str, peer_axis, peer_hand: str = "") -> float:
    if a.polarization == "circular" and peer_pol == "circular":
        if a.handedness and peer_hand and a.handedness != peer_hand:
            return 20.0
        return 0.5
    if a.polarization == "circular" or peer_pol == "circular":
        return 3.0
    aw = R @ unit(np.array(a.axis_body, float))
    pa_, pg = aw - (aw @ u_w) * u_w, np.array(peer_axis, float) - (np.array(peer_axis, float) @ u_w) * u_w
    na, ng = np.linalg.norm(pa_), np.linalg.norm(pg)
    if na < 1e-6 or ng < 1e-6:
        return 3.0
    return -20 * math.log10(max(abs(float(pa_ @ pg)) / (na * ng), 0.03))


# =============================================================================
# 5. Radio parameters and presets
# =============================================================================
OFDM_TABLE = [(2, .35), (5, .7), (8, 1.05), (11, 1.4), (14, 2.1), (18, 2.8), (22, 3.5), (25, 4.0)]
LTE_TABLE = [(-6.5, .15), (-4, .23), (-2, .38), (0, .6), (2, .88), (4, 1.18), (6, 1.48), (8, 1.91), (10, 2.41),
             (12, 2.73), (14, 3.32), (16, 3.9), (18, 4.52), (20, 5.12), (22, 5.55)]
DVBS2X_TABLE = [(-2.4, .5), (1, .99), (4, 1.32), (6, 1.65), (8, 1.98), (10, 2.32), (12, 2.64), (14, 2.97), (16, 3.3),
                (18, 3.62), (20, 3.95)]
WIFI_TABLE = [(2, .325), (5, .65), (9, .975), (11, 1.3), (15, 1.95), (18, 2.6), (20, 2.925), (25, 3.25), (29, 3.9), (31, 4.33)]
LORA_TABLE = [(-20, .012), (-15, .05), (-10, .15), (-5, .4), (0, .9), (5, 1.6)]


@dataclass
class RadioParams:
    name: str = "radio"
    kind: str = "direct"                 # direct | cellular | sat | wifi
    enabled: bool = True
    initial_mode: str = "on"             # on | standby | off  (warm start)
    freq_hz: float = 2.4e9
    bw_hz: float = 20e6
    bw_ul_hz: float = 0.0                # 0 -> same as bw_hz
    tx_power_dbm: float = 27.0
    tx_power_min_dbm: float = 10.0
    tx_power_max_dbm: float = 30.0
    noise_figure_db: float = 5.0
    antennas: List[AntennaSpec] = field(default_factory=list)
    # peer side (ground station / tower / satellite / AP)
    peer_tx_power_dbm: float = 30.0
    peer_gain_dbi: float = 8.0
    peer_noise_figure_db: float = 4.0
    peer_polarization: str = "linear"
    peer_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    peer_tracker_hpbw_deg: float = 0.0   # >0: GS tracking antenna with pointing lag
    peer_tracker_lag_s: float = 0.3
    # PHY / MAC
    mcs_table: List[Tuple[float, float]] = field(default_factory=lambda: list(OFDM_TABLE))
    per_slope_db: float = 2.0
    snr_ceiling_db: float = 30.0
    mac_efficiency: float = 0.7
    ul_share: float = 0.75
    dl_share: float = 0.25
    rate_cap_ul_bps: float = 1e12
    rate_cap_dl_bps: float = 1e12
    ra_tau_s: float = 0.2
    ra_margin_db: float = 1.0
    ra_up_period_s: float = 0.5
    ge_burst_s: float = 0.03
    harq_gain: float = 0.35              # per extra attempt loss multiplier (1.0 = plain ARQ)
    max_tx: int = 3
    retx_delay_ms: float = 5.0
    mtu_bytes: int = 1200
    buffer_bytes: int = 300_000
    max_queue_delay_s: float = 1.5
    # latency components
    lat_ul_ms: float = 4.0
    lat_dl_ms: float = 4.0
    lat_jitter_ms: float = 1.5
    backhaul_ms: float = 0.0
    backhaul_jitter_ms: float = 0.0
    # state machine
    boot_time_s: float = 8.0
    search_min_s: float = 1.0
    search_max_s: float = 3.0
    wake_time_s: float = 0.6
    reacq_min_s: float = 0.5
    reacq_max_s: float = 2.0
    snr_acq_db: float = 3.0
    snr_rlf_db: float = 0.0
    rlf_timer_s: float = 0.5
    measure_in_standby: bool = True
    # power
    p_off_w: float = 0.0
    p_boot_w: float = 4.0
    p_standby_w: float = 0.8
    p_idle_w: float = 3.0
    p_search_extra_w: float = 0.5
    p_rx_w: float = 1.5
    tx_overhead_w: float = 1.5
    pa_eff_max: float = 0.3
    min_tx_duty: float = 0.02
    pa_v_knee_v: float = 0.0
    v_min_operating_v: float = 9.0
    heat_fraction: float = 0.7
    rth_k_per_w: float = 6.0
    cth_j_per_k: float = 40.0
    throttle_c: float = 70.0
    shutdown_c: float = 85.0
    # cellular specific
    ul_p0_dbm: float = -90.0
    ul_alpha: float = 0.8
    ho_hyst_db: float = 3.0
    ho_ttt_s: float = 0.16
    ho_interrupt_min_s: float = 0.03
    ho_interrupt_max_s: float = 0.06
    ho_fail_sinr_db: float = -3.0
    ho_fail_prob: float = 0.4
    meas_period_s: float = 0.1
    n_meas_cells: int = 6
    ul_rise_db_per_m: float = 0.04
    # satellite specific
    min_elev_deg: float = 25.0
    resel_period_s: float = 15.0
    sat_eirp_dbw: float = 38.0
    sat_gain_dbi: float = 30.0
    gw_factor: float = 0.8
    sat_sched_ms: float = 8.0
    sat_load: float = 0.4
    # wifi specific
    roam_hyst_db: float = 6.0
    roam_ttt_s: float = 0.3
    roam_interrupt_s: float = 0.05
    hop_ms: float = 1.5
    # ---- extended technology fields (all default to legacy behaviour) -----------------------
    family: str = ""                       # catalog label, e.g. "ocusync", "iridium_certus"
    interference_class: str = ""           # env noise-rise class; "" -> kind, "auto" -> band_class(freq)
    peer_handedness: str = ""              # rhcp | lhcp for circular GS antennas
    peer_diversity_branches: int = 1       # GS receive diversity (selection) -> shallower fades
    jam_rejection_db: float = 0.0          # spreading / FHSS / blanking gain against jammers
    eirp_limit_dbm: float = 99.0           # regulatory EIRP cap (clips PA power)
    duty_cycle_limit: float = 1.0          # e.g. 0.1 for EU 868 MHz sub-bands
    analog_video: bool = False             # FM analog video: quality follows SNR, no packets
    analog_snr_mid_db: float = 10.0
    analog_snr_slope: float = 0.5
    analog_sync_snr_db: float = 3.0
    analog_min_quality: float = 0.3
    fixed_mcs: int = -1                    # >=0 disables rate adaptation (HDZero-style fixed mode)
    max_video_bps: float = 1e12            # system video bitrate ceiling
    fixed_video_bps: float = 0.0           # >0: encoder forced to this rate (no bitrate adaptation)
    video_fec_overhead: float = -1.0       # <0 -> traffic default; app-level FEC (OpenHD/OpenIPC)
    lat_dist: str = "exp"                  # exp | fixed | lognormal | uniform
    lat_sigma: float = 0.5
    lat_spike_prob: float = 0.0
    lat_spike_ms: float = 0.0
    hop_channels: List[float] = field(default_factory=list)   # centre frequencies for adaptive hopping
    hop_period_s: float = 1.0
    hop_hyst_db: float = 3.0
    hop_interrupt_s: float = 0.005
    tunable: bool = False                  # SDR: radio.retune(f, bw) at runtime
    f_min_hz: float = 0.0
    f_max_hz: float = 0.0
    retune_time_s: float = 0.05
    rat: str = ""                          # cellular: "" any | lte | nr
    operator: str = ""                     # cellular SIM/operator: "" any | A | B
    peer_extra_gain_db: float = 0.0        # cellular: carrier-BW power scaling + massive-MIMO gain
    constellation: str = ""                # satellite: "" default | oneweb | iridium | globalstar | geo
    mesh_convergence_s: float = 1.5
    mesh_hop_ms: float = 6.0
    mesh_min_hop_snr_db: float = 5.0
    tether_len_m: float = 0.0              # fiber: spool length
    tether_mass_kg_per_m: float = 0.0001
    tether_break_prob_per_km: float = 0.0
    members: List[str] = field(default_factory=list)   # bonded: member radio names
    bond_mode: str = "aggregate"           # aggregate | duplicate
    bond_reorder_ms: float = 25.0


def _dipole_pair() -> List[AntennaSpec]:
    return [AntennaSpec("belly", "dipole", 2.15, (0, 0, 1), block_axis_body=(0, 0, 1), block_loss_db=8),
            AntennaSpec("top", "dipole", 2.15, (0, 0, 1), block_axis_body=(0, 0, -1), block_loss_db=6)]


def make_direct_rf(name="rf24") -> RadioParams:
    return RadioParams(name=name, kind="direct", antennas=_dipole_pair(), peer_gain_dbi=8.0)


def make_direct_rf_900(name="rf900") -> RadioParams:
    return RadioParams(name=name, kind="direct", enabled=False, freq_hz=915e6, bw_hz=500e3, tx_power_dbm=30,
                       tx_power_max_dbm=30, tx_power_min_dbm=10, noise_figure_db=4.0, mcs_table=list(LORA_TABLE),
                       antennas=_dipole_pair(), peer_gain_dbi=3.0, peer_tx_power_dbm=30, snr_ceiling_db=20,
                       ul_share=0.5, dl_share=0.5, lat_ul_ms=10, lat_dl_ms=10, boot_time_s=3, p_idle_w=1.2,
                       p_boot_w=1.5, p_standby_w=0.3, tx_overhead_w=0.6, pa_eff_max=0.4, snr_rlf_db=-22, snr_acq_db=-17)


def make_cellular(name="cell") -> RadioParams:
    return RadioParams(
        name=name, kind="cellular", freq_hz=2.1e9, bw_hz=20e6, tx_power_dbm=23, tx_power_max_dbm=23, tx_power_min_dbm=-40,
        noise_figure_db=7.0, antennas=_dipole_pair(), peer_noise_figure_db=3.5, mcs_table=list(LTE_TABLE),
        per_slope_db=2.2, snr_ceiling_db=25, mac_efficiency=0.75, ul_share=0.3, dl_share=0.7, lat_ul_ms=8, lat_dl_ms=4,
        lat_jitter_ms=3, backhaul_ms=18, backhaul_jitter_ms=4, retx_delay_ms=8, max_tx=4, buffer_bytes=500_000,
        boot_time_s=10, search_min_s=2, search_max_s=8, wake_time_s=0.3, reacq_min_s=0.5, reacq_max_s=2.0,
        snr_acq_db=-6, snr_rlf_db=-8, rlf_timer_s=1.0, p_boot_w=2.5, p_standby_w=0.4, p_idle_w=1.2, p_search_extra_w=0.6,
        p_rx_w=1.2, tx_overhead_w=0.8, pa_eff_max=0.35, min_tx_duty=0.05, rth_k_per_w=10, cth_j_per_k=25,
        rate_cap_ul_bps=60e6, rate_cap_dl_bps=150e6, v_min_operating_v=8.0)


def make_leo_sat(name="leo") -> RadioParams:
    return RadioParams(
        name=name, kind="sat", freq_hz=12e9, bw_hz=240e6, bw_ul_hz=60e6, tx_power_dbm=33, tx_power_max_dbm=36,
        tx_power_min_dbm=20, noise_figure_db=3.5, peer_noise_figure_db=3.0, peer_polarization="circular",
        antennas=[AntennaSpec("array", "phased_array", 28.0, (0, 0, 1), polarization="circular", max_scan_deg=55,
                              ripple_db=0.5, cable_loss_db=0.5)],
        mcs_table=list(DVBS2X_TABLE), per_slope_db=2.5, snr_ceiling_db=18, mac_efficiency=0.85, ul_share=1.0, dl_share=1.0,
        rate_cap_ul_bps=30e6, rate_cap_dl_bps=100e6, lat_ul_ms=10, lat_dl_ms=8, lat_jitter_ms=4, backhaul_ms=6,
        backhaul_jitter_ms=3, retx_delay_ms=45, max_tx=2, buffer_bytes=1_500_000, max_queue_delay_s=2.0, boot_time_s=25,
        search_min_s=5, search_max_s=35, wake_time_s=4, reacq_min_s=2, reacq_max_s=10, snr_acq_db=-1, snr_rlf_db=-3,
        rlf_timer_s=2.0, measure_in_standby=False, p_boot_w=20, p_standby_w=5, p_idle_w=24, p_search_extra_w=6, p_rx_w=4,
        tx_overhead_w=6, pa_eff_max=0.25, min_tx_duty=0.1, rth_k_per_w=0.8, cth_j_per_k=300, throttle_c=75,
        shutdown_c=90, v_min_operating_v=10.0, heat_fraction=0.8)


def make_wifi_mesh(name="wifi") -> RadioParams:
    return RadioParams(
        name=name, kind="wifi", freq_hz=5.745e9, bw_hz=20e6, tx_power_dbm=20, tx_power_max_dbm=23, tx_power_min_dbm=5,
        noise_figure_db=6.0, antennas=_dipole_pair(), peer_gain_dbi=5.0, peer_noise_figure_db=6.0, mcs_table=list(WIFI_TABLE),
        snr_ceiling_db=35, mac_efficiency=0.6, ul_share=0.5, dl_share=0.5, lat_ul_ms=2, lat_dl_ms=2, harq_gain=1.0,
        max_tx=6, retx_delay_ms=1.5, boot_time_s=6, search_min_s=1, search_max_s=4, wake_time_s=0.3, snr_acq_db=5,
        snr_rlf_db=2, rlf_timer_s=0.4, p_boot_w=2, p_standby_w=0.3, p_idle_w=1.0, p_search_extra_w=0.4, p_rx_w=1.0,
        tx_overhead_w=0.6, pa_eff_max=0.15, min_tx_duty=0.02, rth_k_per_w=8, cth_j_per_k=20, throttle_c=80,
        initial_mode="off")


NR_TABLE = LTE_TABLE + [(24, 6.0), (26, 6.6)]
LORA_2G4_TABLE = [(-17, .02), (-13, .06), (-9, .15), (-5, .35), (0, .8), (5, 1.4)]
FIBER_TABLE = [(1.0, 8.0)]


def _cp_pair(peak=1.5, hand="rhcp") -> List[AntennaSpec]:
    """Cloverleaf/skew-planar circularly polarised omni pair (top + belly) as used on FPV drones."""
    return [AntennaSpec("belly_cp", "dipole", peak, (0, 0, 1), polarization="circular", handedness=hand,
                        block_axis_body=(0, 0, 1), block_loss_db=8, cable_loss_db=0.3),
            AntennaSpec("top_cp", "dipole", peak, (0, 0, 1), polarization="circular", handedness=hand,
                        block_axis_body=(0, 0, -1), block_loss_db=6, cable_loss_db=0.3)]


def _dipole_gain(peak, block=True) -> List[AntennaSpec]:
    return [AntennaSpec("belly", "dipole", peak, (0, 0, 1), block_axis_body=(0, 0, 1) if block else None, block_loss_db=8),
            AntennaSpec("top", "dipole", peak, (0, 0, 1), block_axis_body=(0, 0, -1) if block else None, block_loss_db=6)]


def make_ofdm_table(modulations=(2, 4, 16, 64, 256), code_rates=(0.5, 0.75, 0.83), overhead=0.25, gap_db=3.0):
    """Build an MCS table for a software-defined OFDM link: eff = bits/sym * rate * (1-overhead);
    SNR threshold from Shannon with an implementation gap. Fully configurable (custom SDR PHY)."""
    out = []
    for M in modulations:
        for r in code_rates:
            raw = math.log2(M) * r
            out.append((round(10 * math.log10(2 ** raw - 1) + gap_db, 2), round(raw * (1 - overhead), 3)))
    out.sort()
    keep, last = [], -1
    for thr, eff in out:
        if eff > last:
            keep.append((thr, eff))
            last = eff
    return keep


# ---- 5.8 GHz ISM ------------------------------------------------------------------------
def make_fpv_analog(name="fpv58_analog", f_hz=5.8e9, bw_hz=18e6, tx_dbm=27.0, family="analog_fpv_5g8", cls="auto",
                    peer_gain=12.0) -> RadioParams:
    """Analog FM FPV video (video-only downlink). 1-25 ms, graceful 'snow' degradation, no packets."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family=family, interference_class=cls,
        freq_hz=f_hz, bw_hz=bw_hz, tx_power_dbm=tx_dbm, tx_power_min_dbm=14, tx_power_max_dbm=30, noise_figure_db=5.0,
        antennas=_cp_pair(), peer_polarization="circular", peer_handedness="rhcp", peer_gain_dbi=peer_gain,
        peer_noise_figure_db=4.0, peer_diversity_branches=2, peer_tracker_hpbw_deg=40.0, peer_tracker_lag_s=0.3,
        analog_video=True, analog_snr_mid_db=11.0, analog_snr_slope=0.45, analog_sync_snr_db=4.0, analog_min_quality=0.3,
        mcs_table=[(0.0, 1.0)], snr_ceiling_db=38, ul_share=0.0, dl_share=0.0, lat_ul_ms=12, lat_dl_ms=12, lat_jitter_ms=2.0,
        lat_dist="fixed", eirp_limit_dbm=36, boot_time_s=3, search_min_s=0.1, search_max_s=0.4, wake_time_s=0.2,
        reacq_min_s=0.05, reacq_max_s=0.2, snr_acq_db=5, snr_rlf_db=3.5, rlf_timer_s=0.15, p_boot_w=1.5, p_standby_w=0.2,
        p_idle_w=1.0, tx_overhead_w=1.2, pa_eff_max=0.2, min_tx_duty=1.0, rth_k_per_w=10, cth_j_per_k=15, throttle_c=70,
        shutdown_c=85, v_min_operating_v=7.0, jam_rejection_db=0.0)


def make_ocusync(name="ocusync") -> RadioParams:
    """DJI OcuSync/O3/O4 style: OFDM, dual-band 2.4/5.8, adaptive frequency hopping, ~22-40 ms video, 12 ms control."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family="ocusync_o3_o4", interference_class="auto",
        freq_hz=5.785e9, bw_hz=40e6, tx_power_dbm=28, tx_power_min_dbm=10, tx_power_max_dbm=30, noise_figure_db=5.5,
        antennas=_dipole_gain(2.0), peer_gain_dbi=5.0, peer_noise_figure_db=5.0, peer_diversity_branches=2,
        hop_channels=[2.4155e9, 2.4375e9, 2.4575e9, 5.745e9, 5.785e9, 5.825e9], hop_period_s=0.5, hop_hyst_db=3.0,
        hop_interrupt_s=0.005, jam_rejection_db=3.0, mcs_table=list(OFDM_TABLE), per_slope_db=2.4, snr_ceiling_db=32,
        mac_efficiency=0.8, ul_share=0.85, dl_share=0.15, rate_cap_ul_bps=60e6, rate_cap_dl_bps=10e6, max_video_bps=50e6,
        ra_tau_s=0.15, ra_up_period_s=0.3, ge_burst_s=0.02, harq_gain=0.4, max_tx=3, retx_delay_ms=4, lat_ul_ms=24,
        lat_dl_ms=12, lat_jitter_ms=3.0, lat_dist="lognormal", lat_sigma=0.4, lat_spike_prob=0.002, lat_spike_ms=40,
        eirp_limit_dbm=33, boot_time_s=6, search_min_s=0.5, search_max_s=1.5, wake_time_s=0.3, reacq_min_s=0.3,
        reacq_max_s=1.2, snr_acq_db=4, snr_rlf_db=1.5, rlf_timer_s=0.4, p_boot_w=4, p_standby_w=0.8, p_idle_w=4.0,
        p_rx_w=1.0, tx_overhead_w=2.5, pa_eff_max=0.25, min_tx_duty=0.1, rth_k_per_w=8, cth_j_per_k=20)


def make_walksnail(name="walksnail") -> RadioParams:
    """Walksnail Avatar style: 5.8 GHz digital HD, ~22 ms, adaptive bitrate (<=25 Mb/s), manual channel."""
    r = make_ocusync(name)
    r.family, r.freq_hz, r.bw_hz, r.hop_channels = "walksnail_avatar", 5.8e9, 20e6, []
    r.tx_power_dbm, r.tx_power_max_dbm, r.jam_rejection_db = 27, 30, 0.0
    r.antennas, r.peer_gain_dbi, r.peer_diversity_branches = _cp_pair(1.5, "rhcp"), 6.0, 2
    r.peer_polarization, r.peer_handedness = "circular", "rhcp"
    r.rate_cap_ul_bps, r.max_video_bps, r.lat_ul_ms, r.lat_dl_ms, r.lat_jitter_ms = 30e6, 25e6, 22, 15, 2.0
    r.lat_spike_prob, r.ul_share, r.dl_share, r.p_idle_w, r.tx_overhead_w = 0.001, 0.9, 0.1, 3.5, 2.0
    return r


def make_hdzero(name="hdzero") -> RadioParams:
    """HDZero style: fixed-mode digital (no rate adaptation, no ARQ), very low fixed latency, video-only."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family="hdzero", interference_class="auto",
        freq_hz=5.8e9, bw_hz=27e6, tx_power_dbm=27, tx_power_min_dbm=14, tx_power_max_dbm=29, noise_figure_db=5.0,
        antennas=_cp_pair(), peer_polarization="circular", peer_handedness="rhcp", peer_gain_dbi=10.0,
        peer_diversity_branches=2, mcs_table=[(6.0, 0.8)], fixed_mcs=0, per_slope_db=3.0, snr_ceiling_db=30,
        mac_efficiency=1.0, ul_share=1.0, dl_share=0.0, max_video_bps=21e6, fixed_video_bps=21e6, video_fec_overhead=0.1,
        harq_gain=1.0, max_tx=1, lat_ul_ms=4, lat_dl_ms=4, lat_jitter_ms=0.3, lat_dist="fixed", eirp_limit_dbm=36,
        boot_time_s=4, search_min_s=0.1, search_max_s=0.5, wake_time_s=0.2, reacq_min_s=0.1, reacq_max_s=0.5,
        snr_acq_db=7, snr_rlf_db=4, rlf_timer_s=0.15, p_boot_w=2, p_standby_w=0.3, p_idle_w=2.5, tx_overhead_w=1.5,
        pa_eff_max=0.22, min_tx_duty=1.0, rth_k_per_w=9, cth_j_per_k=15)


def make_openhd(name="openhd58", openipc=False) -> RadioParams:
    """OpenHD / OpenIPC style: Wi-Fi hardware in injection mode (no association, no L2 ARQ), app-level FEC."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family="openipc" if openipc else "openhd",
        interference_class="auto", freq_hz=5.805e9, bw_hz=20e6, tx_power_dbm=27, tx_power_min_dbm=5, tx_power_max_dbm=30,
        noise_figure_db=6.0, antennas=_dipole_gain(2.0), peer_gain_dbi=5.0, peer_noise_figure_db=6.0,
        peer_diversity_branches=2, mcs_table=list(WIFI_TABLE), per_slope_db=2.0, snr_ceiling_db=35, mac_efficiency=0.65,
        ul_share=0.9, dl_share=0.1, ra_tau_s=0.5, ra_up_period_s=1.0, harq_gain=1.0, max_tx=1,
        video_fec_overhead=0.4 if openipc else 0.5, max_video_bps=25e6, lat_ul_ms=28 if openipc else 45,
        lat_dl_ms=15, lat_jitter_ms=6 if openipc else 12, lat_dist="lognormal", lat_sigma=0.5, lat_spike_prob=0.005,
        lat_spike_ms=60, eirp_limit_dbm=36, boot_time_s=15, search_min_s=0.5, search_max_s=2, wake_time_s=0.5,
        snr_acq_db=5, snr_rlf_db=2, rlf_timer_s=0.4, p_boot_w=2.5, p_standby_w=0.6, p_idle_w=3.0, p_rx_w=1.0,
        tx_overhead_w=1.5, pa_eff_max=0.2, min_tx_duty=0.1, rth_k_per_w=8, cth_j_per_k=20,
        tunable=True, f_min_hz=5.17e9, f_max_hz=5.835e9)


# ---- 2.4 GHz control + lower bands -------------------------------------------------------
def make_elrs24(name="elrs24") -> RadioParams:
    """2.4 GHz ISM LoRa/FLRC control link (ExpressLRS-style): tiny packets, FHSS spreading, 2-4 ms latency."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family="elrs_2g4", interference_class="auto",
        freq_hz=2.44e9, bw_hz=0.8e6, tx_power_dbm=27, tx_power_min_dbm=10, tx_power_max_dbm=30, noise_figure_db=6.0,
        antennas=_dipole_gain(2.0), peer_tx_power_dbm=27, peer_gain_dbi=3.0, peer_noise_figure_db=5.0,
        mcs_table=list(LORA_2G4_TABLE), per_slope_db=2.5, snr_ceiling_db=20, mac_efficiency=0.6, ul_share=0.3, dl_share=0.7,
        jam_rejection_db=12.0, harq_gain=1.0, max_tx=1, lat_ul_ms=4, lat_dl_ms=3, lat_jitter_ms=0.6, lat_dist="fixed",
        eirp_limit_dbm=36, boot_time_s=3, search_min_s=0.3, search_max_s=1.2, wake_time_s=0.2, snr_acq_db=-12,
        snr_rlf_db=-16, rlf_timer_s=0.5, p_boot_w=1.0, p_standby_w=0.2, p_idle_w=0.5, p_rx_w=0.3, tx_overhead_w=0.4,
        pa_eff_max=0.3, min_tx_duty=0.1, rth_k_per_w=12, cth_j_per_k=10)


def make_crossfire868(name="crossfire868") -> RadioParams:
    """868/915 MHz long-range control (Crossfire/ELRS-900 style): LoRa/FSK, EU duty-cycle & ERP limits."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family="crossfire_868", interference_class="auto",
        freq_hz=868.3e6, bw_hz=500e3, tx_power_dbm=27, tx_power_min_dbm=10, tx_power_max_dbm=33, noise_figure_db=4.5,
        antennas=_dipole_gain(2.0), peer_tx_power_dbm=27, peer_gain_dbi=3.0, peer_noise_figure_db=4.0,
        mcs_table=list(LORA_TABLE), per_slope_db=2.5, snr_ceiling_db=20, mac_efficiency=0.6, ul_share=0.3, dl_share=0.7,
        jam_rejection_db=8.0, duty_cycle_limit=0.5, harq_gain=1.0, max_tx=1, lat_ul_ms=8, lat_dl_ms=6, lat_jitter_ms=1.0,
        lat_dist="fixed", eirp_limit_dbm=27, boot_time_s=3, search_min_s=0.3, search_max_s=1.2, wake_time_s=0.2,
        snr_acq_db=-17, snr_rlf_db=-21, rlf_timer_s=0.6, p_boot_w=1.0, p_standby_w=0.2, p_idle_w=0.6, p_rx_w=0.3,
        tx_overhead_w=0.5, pa_eff_max=0.35, min_tx_duty=0.1, rth_k_per_w=12, cth_j_per_k=10)


def make_lr_digital(name, f_hz, bw_hz, tx_dbm=30.0, family="", ant_gain=2.15, peer_gain=9.0, lat_ms=15.0) -> RadioParams:
    """Long-range digital OFDM on 1.2-1.3 / 1.7 / 3.3 / 7.2 GHz (path loss, diffraction, penetration from env by band)."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family=family or f"lr_{f_hz/1e9:.1f}ghz",
        interference_class="auto", freq_hz=f_hz, bw_hz=bw_hz, tx_power_dbm=tx_dbm, tx_power_min_dbm=10, tx_power_max_dbm=33,
        noise_figure_db=4.5, antennas=_dipole_gain(ant_gain), peer_tx_power_dbm=tx_dbm, peer_gain_dbi=peer_gain,
        peer_noise_figure_db=4.0, peer_diversity_branches=2, mcs_table=list(OFDM_TABLE), per_slope_db=2.3,
        snr_ceiling_db=30, mac_efficiency=0.75, ul_share=0.75, dl_share=0.25, jam_rejection_db=2.0, harq_gain=0.4,
        max_tx=3, retx_delay_ms=6, lat_ul_ms=lat_ms, lat_dl_ms=lat_ms * 0.6, lat_jitter_ms=3.0, lat_dist="lognormal",
        lat_sigma=0.4, boot_time_s=8, search_min_s=1, search_max_s=3, wake_time_s=0.5, snr_acq_db=3, snr_rlf_db=0.5,
        rlf_timer_s=0.5, p_boot_w=4, p_standby_w=0.8, p_idle_w=3.0, p_rx_w=1.5, tx_overhead_w=1.8, pa_eff_max=0.3,
        min_tx_duty=0.05, rth_k_per_w=7, cth_j_per_k=30, eirp_limit_dbm=40)


# ---- cellular ------------------------------------------------------------------------------
def make_lte(name, operator) -> RadioParams:
    r = make_cellular(name)
    r.family, r.operator, r.rat, r.enabled, r.initial_mode = f"lte_{operator}", operator, "lte", False, "off"
    return r


def make_nr(name, operator) -> RadioParams:
    """5G NR sub-6 (n78 3.5 GHz, 100 MHz TDD): more capacity, lower latency, shorter reach, only on NR sites."""
    r = make_cellular(name)
    r.family, r.operator, r.rat, r.enabled, r.initial_mode = f"nr_{operator}", operator, "nr", False, "off"
    r.freq_hz, r.bw_hz, r.mcs_table, r.snr_ceiling_db = 3.5e9, 100e6, list(NR_TABLE), 28
    r.peer_extra_gain_db = 13.0                  # +7 dB carrier-BW power scaling, +6 dB massive-MIMO beamforming
    r.ul_share, r.dl_share, r.rate_cap_ul_bps, r.rate_cap_dl_bps = 0.25, 0.7, 120e6, 400e6
    r.lat_ul_ms, r.lat_dl_ms, r.backhaul_ms, r.retx_delay_ms = 5, 2, 12, 4
    r.p_idle_w, r.p_rx_w, r.tx_overhead_w, r.p_boot_w = 1.8, 2.0, 1.2, 3.5
    r.snr_acq_db, r.snr_rlf_db = -4, -6
    return r


def make_sim_bond(name="sim_bond", members=("lte_a", "lte_b", "nr_a", "nr_b"), mode="aggregate") -> RadioParams:
    return RadioParams(name=name, kind="bond", enabled=False, initial_mode="off", family="multi_sim_bond",
                       members=list(members), bond_mode=mode, bond_reorder_ms=25.0, p_idle_w=1.5, p_standby_w=0.5,
                       antennas=[], mcs_table=[(0.0, 1.0)], ul_share=1.0, dl_share=1.0)


# ---- satellite ---------------------------------------------------------------------------------
SAT_PRESETS = {
    "oneweb_ku": dict(family="oneweb_ku", constellation="oneweb", freq_hz=12.5e9, bw_hz=250e6, bw_ul_hz=50e6,
                      sat_eirp_dbw=43.0, sat_gain_dbi=30.0, min_elev_deg=30.0, rate_cap_ul_bps=15e6, rate_cap_dl_bps=60e6,
                      lat_ul_ms=14, lat_dl_ms=12, backhaul_ms=8, p_idle_w=22, tx_overhead_w=5, resel_period_s=30.0),
    "iridium_certus": dict(family="iridium_certus_l", constellation="iridium", freq_hz=1.62e9, bw_hz=700e3, bw_ul_hz=240e3,
                           tx_power_dbm=35, tx_power_max_dbm=36, tx_power_min_dbm=20, noise_figure_db=2.5,
                           peer_noise_figure_db=2.5, sat_eirp_dbw=28.0, sat_gain_dbi=24.0, min_elev_deg=8.2,
                           antennas=[AntennaSpec("l_patch", "patch", 4.0, (0, 0, 1), beamwidth_deg=110.0, polarization="circular",
                                                 ripple_db=0.5, cable_loss_db=0.5, back_lobe_db=25)],
                           mcs_table=[(-3, .7), (0, 1.0), (3, 1.3), (7, 1.6)], rate_cap_ul_bps=352e3, rate_cap_dl_bps=704e3,
                           per_slope_db=2.2, snr_ceiling_db=22, lat_ul_ms=60, lat_dl_ms=60, lat_jitter_ms=25,
                           lat_dist="lognormal", lat_sigma=0.5, backhaul_ms=60, sat_sched_ms=120.0, retx_delay_ms=300,
                           max_tx=2, buffer_bytes=200_000, resel_period_s=240.0, boot_time_s=40, search_min_s=10,
                           search_max_s=40, wake_time_s=8, snr_acq_db=-2, snr_rlf_db=-4, rlf_timer_s=3, reacq_min_s=5,
                           reacq_max_s=20, p_boot_w=6, p_standby_w=1.5, p_idle_w=4.0, p_search_extra_w=2, p_rx_w=1.5,
                           tx_overhead_w=3.0, pa_eff_max=0.2, rth_k_per_w=2.5, cth_j_per_k=60, sat_load=0.3),
    "globalstar_s": dict(family="globalstar_s", constellation="globalstar", freq_hz=2.49e9, bw_hz=1.23e6, bw_ul_hz=1.23e6,
                         tx_power_dbm=33, tx_power_max_dbm=34, tx_power_min_dbm=15, noise_figure_db=3.0,
                         peer_noise_figure_db=2.5, sat_eirp_dbw=33.0, sat_gain_dbi=24.0, min_elev_deg=10.0,
                         antennas=[AntennaSpec("s_patch", "patch", 5.0, (0, 0, 1), beamwidth_deg=100.0, polarization="circular",
                                               ripple_db=0.5, cable_loss_db=0.5, back_lobe_db=25)],
                         mcs_table=[(-3, .1), (0, .2), (3, .4), (6, .6)], rate_cap_ul_bps=150e3, rate_cap_dl_bps=300e3,
                         snr_ceiling_db=20, lat_ul_ms=50, lat_dl_ms=50, lat_jitter_ms=20, backhaul_ms=40,
                         sat_sched_ms=60.0, retx_delay_ms=250, buffer_bytes=100_000, resel_period_s=120.0, boot_time_s=30,
                         search_min_s=8, search_max_s=30, wake_time_s=6, snr_acq_db=-2, snr_rlf_db=-4, rlf_timer_s=3,
                         p_boot_w=5, p_standby_w=1.2, p_idle_w=3.0, tx_overhead_w=2.5, pa_eff_max=0.2, sat_load=0.3),
    "inmarsat_l": dict(family="inmarsat_l_geo", constellation="geo", freq_hz=1.6e9, bw_hz=400e3, bw_ul_hz=400e3,
                       tx_power_dbm=33, tx_power_max_dbm=34, tx_power_min_dbm=20, noise_figure_db=2.5,
                       peer_noise_figure_db=2.5, sat_eirp_dbw=47.0, sat_gain_dbi=44.0, min_elev_deg=10.0,
                       antennas=[AntennaSpec("l_patch", "patch", 4.0, (0, 0, 1), beamwidth_deg=100.0, polarization="circular",
                                             ripple_db=0.5, cable_loss_db=0.5, back_lobe_db=25)],
                       mcs_table=[(-3, .7), (0, 1.0), (3, 1.3), (7, 1.6)], rate_cap_ul_bps=240e3, rate_cap_dl_bps=450e3,
                       snr_ceiling_db=22, lat_ul_ms=40, lat_dl_ms=40, lat_jitter_ms=20, backhaul_ms=40, sat_sched_ms=80.0,
                       retx_delay_ms=600, resel_period_s=1e9, boot_time_s=45, search_min_s=15, search_max_s=45,
                       wake_time_s=10, snr_acq_db=-2, snr_rlf_db=-4, rlf_timer_s=4, p_boot_w=6, p_standby_w=1.5,
                       p_idle_w=5.0, tx_overhead_w=3.0, pa_eff_max=0.2, sat_load=0.3),
    "geo_ku": dict(family="geo_ku", constellation="geo", freq_hz=12.0e9, bw_hz=36e6, bw_ul_hz=10e6, tx_power_dbm=36,
                   tx_power_max_dbm=37, tx_power_min_dbm=20, sat_eirp_dbw=56.0, sat_gain_dbi=42.0, min_elev_deg=10.0,
                   antennas=[AntennaSpec("ku_array", "phased_array", 30.0, (0, 0, 1), polarization="circular", max_scan_deg=60,
                                         ripple_db=0.5, cable_loss_db=0.5)],
                   rate_cap_ul_bps=8e6, rate_cap_dl_bps=40e6, lat_ul_ms=40, lat_dl_ms=40, lat_jitter_ms=15, backhaul_ms=20,
                   sat_sched_ms=25.0, retx_delay_ms=600, resel_period_s=1e9, boot_time_s=60, search_min_s=15,
                   search_max_s=60, wake_time_s=10, p_idle_w=30, tx_overhead_w=8, sat_load=0.35),
    "geo_ka": dict(family="geo_ka", constellation="geo", freq_hz=20.0e9, bw_hz=36e6, bw_ul_hz=10e6, tx_power_dbm=36,
                   tx_power_max_dbm=38, tx_power_min_dbm=20, sat_eirp_dbw=62.0, sat_gain_dbi=50.0, min_elev_deg=15.0,
                   antennas=[AntennaSpec("ka_array", "phased_array", 33.0, (0, 0, 1), polarization="circular", max_scan_deg=55,
                                         ripple_db=0.5, cable_loss_db=0.5)],
                   rate_cap_ul_bps=8e6, rate_cap_dl_bps=40e6, lat_ul_ms=40, lat_dl_ms=40, lat_jitter_ms=15, backhaul_ms=20,
                   sat_sched_ms=25.0, retx_delay_ms=600, resel_period_s=1e9, boot_time_s=60, search_min_s=15,
                   search_max_s=60, wake_time_s=10, p_idle_w=34, tx_overhead_w=10, sat_load=0.35),
}


def make_sat(name: str, preset: str) -> RadioParams:
    """LEO/MEO/GEO satellite terminals: starlink_ku (=make_leo_sat), oneweb_ku, iridium_certus, globalstar_s,
    inmarsat_l, geo_ku, geo_ka. Rain fade scales with the env ITU table, L/S-band gets scintillation."""
    r = make_leo_sat(name)
    r.enabled, r.initial_mode = False, "off"
    for k, v in SAT_PRESETS[preset].items():
        setattr(r, k, v)
    return r


# ---- mesh, fiber, SDR ---------------------------------------------------------------------------
def make_mesh_manet(name="mesh_manet", f_hz=1.4e9, bw_hz=20e6) -> RadioParams:
    return RadioParams(
        name=name, kind="mesh", enabled=False, initial_mode="off", family="mesh_manet", interference_class="auto",
        freq_hz=f_hz, bw_hz=bw_hz, tx_power_dbm=30, tx_power_min_dbm=10, tx_power_max_dbm=33, noise_figure_db=5.0,
        antennas=_dipole_gain(2.15), mcs_table=list(OFDM_TABLE), per_slope_db=2.2, snr_ceiling_db=30, mac_efficiency=0.65,
        ul_share=0.5, dl_share=0.5, jam_rejection_db=3.0, harq_gain=0.5, max_tx=3, retx_delay_ms=6, lat_ul_ms=8, lat_dl_ms=8,
        lat_jitter_ms=4.0, lat_dist="lognormal", lat_sigma=0.5, mesh_hop_ms=6.0, mesh_convergence_s=1.5,
        boot_time_s=12, search_min_s=2, search_max_s=6, wake_time_s=0.6, snr_acq_db=4, snr_rlf_db=1, rlf_timer_s=0.6,
        p_boot_w=4, p_standby_w=1.0, p_idle_w=4.0, p_rx_w=1.5, tx_overhead_w=2.0, pa_eff_max=0.3, min_tx_duty=0.1,
        rth_k_per_w=7, cth_j_per_k=30, eirp_limit_dbm=40)


def make_fiber_tether(name="fiber_tether", length_m=3000.0) -> RadioParams:
    return RadioParams(
        name=name, kind="fiber", enabled=False, initial_mode="off", family="fiber_tether", interference_class="fiber",
        freq_hz=1.31e14, bw_hz=125e6, tx_power_dbm=0, tx_power_min_dbm=-5, tx_power_max_dbm=3, noise_figure_db=0.0,
        antennas=[], mcs_table=list(FIBER_TABLE), per_slope_db=3.0, snr_ceiling_db=40, mac_efficiency=0.95, ul_share=1.0,
        dl_share=1.0, harq_gain=1.0, max_tx=1, lat_ul_ms=0.4, lat_dl_ms=0.4, lat_jitter_ms=0.05, lat_dist="fixed",
        boot_time_s=3, search_min_s=0.2, search_max_s=0.5, wake_time_s=0.2, snr_acq_db=5, snr_rlf_db=5, rlf_timer_s=0.05,
        p_boot_w=2, p_standby_w=0.5, p_idle_w=1.8, p_rx_w=0.3, tx_overhead_w=0.8, pa_eff_max=0.5, min_tx_duty=1.0,
        buffer_bytes=2_000_000, tether_len_m=length_m, tether_mass_kg_per_m=0.0001, tether_break_prob_per_km=0.0,
        v_min_operating_v=7.0)


def make_sdr_ofdm(name="sdr_ofdm", f_hz=2.3e9, bw_hz=10e6) -> RadioParams:
    """Custom software-defined OFDM link: PHY (modulations, code rates, overhead, gap) and RF (frequency, bandwidth,
    power, latency) are all parameters; retune() at runtime within [f_min, f_max]."""
    return RadioParams(
        name=name, kind="direct", enabled=False, initial_mode="off", family="sdr_ofdm", interference_class="auto",
        freq_hz=f_hz, bw_hz=bw_hz, tx_power_dbm=27, tx_power_min_dbm=0, tx_power_max_dbm=33, noise_figure_db=5.0,
        antennas=_dipole_gain(2.15), peer_gain_dbi=6.0, peer_diversity_branches=2, mcs_table=make_ofdm_table(),
        per_slope_db=2.2, snr_ceiling_db=32, mac_efficiency=0.8, ul_share=0.7, dl_share=0.3, jam_rejection_db=2.0,
        harq_gain=0.4, max_tx=3, retx_delay_ms=5, lat_ul_ms=10, lat_dl_ms=8, lat_jitter_ms=2.0, lat_dist="lognormal",
        lat_sigma=0.4, boot_time_s=10, search_min_s=0.5, search_max_s=2, wake_time_s=0.4, snr_acq_db=3, snr_rlf_db=0.5,
        rlf_timer_s=0.4, p_boot_w=5, p_standby_w=1.2, p_idle_w=5.0, p_rx_w=2.0, tx_overhead_w=3.0, pa_eff_max=0.3,
        min_tx_duty=0.1, rth_k_per_w=6, cth_j_per_k=40, tunable=True, f_min_hz=0.4e9, f_max_hz=6.0e9, eirp_limit_dbm=40)


def radio_catalog() -> List[RadioParams]:
    """Every selectable link technology (legacy five first, then all new ones, disabled by default)."""
    return [
        make_direct_rf("rf24"), make_cellular("cell"), make_leo_sat("leo"), make_wifi_mesh("wifi"), make_direct_rf_900("rf900"),
        make_fpv_analog("fpv58_analog"), make_ocusync("ocusync"), make_walksnail("walksnail"), make_hdzero("hdzero"),
        make_openhd("openhd58"), make_openhd("openipc58", openipc=True),
        make_elrs24("elrs24"), make_crossfire868("crossfire868"),
        make_fpv_analog("fpv12g_analog", 1.28e9, 18e6, 30.0, "analog_fpv_1g2", "auto", 9.0),
        make_lr_digital("lr12g", 1.28e9, 10e6, 30.0, "lr_digital_1g2", 2.15, 9.0, 18),
        make_lr_digital("lr17g", 1.7e9, 20e6, 30.0, "lr_digital_1g7", 2.15, 9.0, 15),
        make_lr_digital("lr33g", 3.3e9, 20e6, 30.0, "lr_digital_3g3", 2.5, 10.0, 15),
        make_lr_digital("lr72g", 7.2e9, 20e6, 27.0, "lr_digital_7g2_emerging", 3.0, 12.0, 15),
        make_lte("lte_a", "A"), make_lte("lte_b", "B"), make_nr("nr_a", "A"), make_nr("nr_b", "B"), make_sim_bond(),
        make_sat("oneweb_ku", "oneweb_ku"), make_sat("iridium_certus", "iridium_certus"), make_sat("globalstar_s", "globalstar_s"),
        make_sat("inmarsat_l", "inmarsat_l"), make_sat("geo_ku", "geo_ku"), make_sat("geo_ka", "geo_ka"),
        make_mesh_manet(), make_fiber_tether(), make_sdr_ofdm(),
    ]


# =============================================================================
# 6. Radio base class and concrete radios
# =============================================================================
S_OFF, S_BOOT, S_STANDBY, S_ACQ, S_CONN = "off", "booting", "standby", "acquiring", "connected"


@dataclass
class PeerInfo:
    pos: np.ndarray
    key: tuple
    eirp_dbm: float                    # peer transmit EIRP (DL)
    rx_gain_dbi: float                 # peer receive gain (UL)
    extra_loss_db: float = 0.0
    pointing_loss_db: float = 0.0
    elev_deg: Optional[float] = None
    extra_dl_i_mw: float = 0.0
    extra_ul_i_mw: float = 0.0
    range_m: float = 0.0


@dataclass
class LinkStatus:
    t: float
    name: str
    kind: str
    state: str
    measurable: bool
    rssi_dbm: float
    snr_ul_db: float
    snr_dl_db: float
    rate_ul_bps: float
    rate_dl_bps: float
    cap_ul_bps: float
    cap_dl_bps: float
    per_ul: float
    per_dl: float
    ctrl_loss_prob: float
    lat_ctrl_ms: float
    lat_video_ms: float
    queue_ul_ms: float
    power_w: float
    tx_power_dbm: float
    temp_c: float
    los: bool
    outage: bool
    extra: dict


class _Dir:
    def __init__(self):
        self.snr, self.snr_est, self.mcs, self.up_t = -30.0, None, 0, 0.0
        self.rate = self.cap = 0.0
        self.per, self.p_state, self.bad = 1.0, 1.0, False
        self.q_hp = self.q_lp = 0.0
        self.off_hp = self.off_lp = 0.0          # offered bps
        self.del_hp = self.del_lp = 0.0          # delivered bps (this step)
        self.drop_bytes = 0.0


class Radio:
    kind_label = "direct"

    def __init__(self, p: RadioParams, rng: np.random.Generator):
        self.p, self.rng = p, rng
        self.state, self.remain, self.want = S_OFF, 0.0, "off"
        self.tx_power_dbm = p.tx_power_dbm
        self.dirs = {"ul": _Dir(), "dl": _Dir()}
        self.temp_c, self.derate_db, self.thermal_lock = 25.0, 0.0, False
        self.interrupt_until, self.rlf_acc, self.t = -1.0, 0.0, 0.0
        self.rssi, self.snr_ul, self.snr_dl = -150.0, -60.0, -60.0
        self.peer: Optional[PeerInfo] = None
        self.prop: Optional[PropSample] = None
        self.power_w = self.energy_j = self.last_ptx_dbm = 0.0
        self.events: List[Tuple[float, str]] = []
        self.hist: deque = deque(maxlen=1200)
        self._hist_t = -1.0
        self.measurable = False
        self.n_brownouts = self.n_rlf = self.n_handover = 0
        self._u_prev = None
        self.ptx_eff = p.tx_power_dbm
        self.freq_hz, self.bw_hz = p.freq_hz, p.bw_hz            # runtime-tunable (SDR / hopping)
        self.bw_ul_hz = p.bw_ul_hz or p.bw_hz
        self.next_hop, self.n_hops = 0.0, 0
        self.fault = False

    # ---- control API ----------------------------------------------------------
    def warm_start(self):
        self.want = self.p.initial_mode
        self.state = {"on": S_CONN, "standby": S_STANDBY, "off": S_OFF}[self.want]

    def set_tx_power(self, dbm: float):
        self.tx_power_dbm = clamp(dbm, self.p.tx_power_min_dbm, self.p.tx_power_max_dbm)

    def retune(self, f_hz: Optional[float] = None, bw_hz: Optional[float] = None):
        """SDR-style runtime retune (needs tunable=True). Causes a short interruption."""
        p = self.p
        if not p.tunable:
            raise ValueError(f"{p.name} is not tunable")
        if f_hz is not None:
            if p.f_max_hz and not (p.f_min_hz <= f_hz <= p.f_max_hz):
                raise ValueError("frequency outside tuning range")
            self.freq_hz = f_hz
        if bw_hz:
            ratio = bw_hz / p.bw_hz
            self.bw_hz, self.bw_ul_hz = bw_hz, (p.bw_ul_hz * ratio if p.bw_ul_hz else bw_hz)
        self.interrupt_until = max(self.interrupt_until, self.t + p.retune_time_s)
        self.events.append((self.t, f"retune {self.freq_hz/1e6:.1f}MHz bw={self.bw_hz/1e6:.2f}MHz"))

    def inject_fault(self, kind: str = "break"):
        self.fault = True
        self.events.append((self.t, f"fault:{kind}"))

    @property
    def extra_mass_kg(self) -> float:
        return 0.0

    def _icls(self) -> str:
        c = self.p.interference_class
        return band_class(self.freq_hz) if c == "auto" else (c or self.p.kind)

    def analog_quality(self) -> float:
        """FM analog picture quality 0..1 vs UL SNR (snow). 0 when sync is lost."""
        p = self.p
        if not (p.analog_video and self.link_up()) or self.snr_ul < p.analog_sync_snr_db or self.t < self.interrupt_until:
            return 0.0
        return 1.0 / (1.0 + math.exp(-p.analog_snr_slope * (self.snr_ul - p.analog_snr_mid_db)))

    def _jitter_ms(self) -> float:
        p = self.p
        if p.lat_dist == "fixed":
            j = abs(self.rng.normal(0.0, p.lat_jitter_ms))
        elif p.lat_dist == "lognormal":
            j = p.lat_jitter_ms * self.rng.lognormal(0.0, p.lat_sigma)
        elif p.lat_dist == "uniform":
            j = self.rng.uniform(0.0, 2 * p.lat_jitter_ms)
        else:
            j = self.rng.exponential(p.lat_jitter_ms)
        j += abs(self.rng.normal(0, p.backhaul_jitter_ms))
        if p.lat_spike_prob > 0 and self.rng.random() < p.lat_spike_prob:
            j += p.lat_spike_ms * (0.5 + self.rng.random())
        return j

    def _choose_channel(self, t, drone, env):
        """Adaptive frequency hopping / dual-band selection: sense both ends, move to the quietest
        channel (path-loss penalised ~20log f) when it is better than the current one by hop_hyst_db."""
        p = self.p
        self.next_hop = t + p.hop_period_s
        icls = self._icls()

        def score(f):
            cls = band_class(f) if p.interference_class == "auto" else icls
            acc = 0.0
            for k in range(8):                       # sensing history over the last ~0.5 s (pulsed jammers!)
                tt = max(t - 0.06 * k, 0.0)
                acc += env.interference_mw(cls, f, self.bw_hz, drone.pos, tt, p.jam_rejection_db)
                acc += env.interference_mw(cls, f, self.bw_hz, env.gs, tt, p.jam_rejection_db)
            return db(acc / 8.0) + 20 * math.log10(f / self.freq_hz)
        cur = score(self.freq_hz)
        best = min(p.hop_channels, key=score)
        if best != self.freq_hz and score(best) < cur - p.hop_hyst_db:
            self.events.append((t, f"hop {self.freq_hz/1e6:.0f}->{best/1e6:.0f}MHz"))
            self.freq_hz = best
            self.n_hops += 1
            self.interrupt_until = max(self.interrupt_until, t + p.hop_interrupt_s)

    def _cap_limit(self, name: str) -> float:
        return 1e18

    def _extra_per(self, name: str) -> float:
        return 0.0

    def _outage(self, ctrl_loss, ul) -> bool:
        if self.p.analog_video:
            return (not self.link_up()) or self.analog_quality() < self.p.analog_min_quality
        return (not self.link_up()) or (ctrl_loss > 0.5 and ul.cap < 3e5)

    def request(self, mode: str):
        self.want = mode
        if mode == "off":
            self._to(S_OFF)
        elif mode == "standby":
            if self.state in (S_CONN, S_ACQ):
                self._to(S_STANDBY)
            elif self.state == S_OFF:
                self._to(S_BOOT, self.p.boot_time_s)
        elif mode == "on":
            if self.state == S_OFF:
                self._to(S_BOOT, self.p.boot_time_s)
            elif self.state == S_STANDBY:
                self._to(S_ACQ, self.p.wake_time_s)

    def _to(self, st, remain=0.0):
        if st != self.state:
            self.events.append((self.t, f"{self.state}->{st}"))
        self.state, self.remain = st, remain
        if st != S_CONN:
            for d in self.dirs.values():
                d.q_hp = d.q_lp = 0.0
        self._on_state(st)

    def _on_state(self, st):
        pass

    def link_up(self) -> bool:
        return self.state == S_CONN

    # ---- link budget -----------------------------------------------------------
    def _peer(self, t, dt, drone, env) -> Optional[PeerInfo]:
        raise NotImplementedError

    def _best_antenna(self, ub, u, R):
        best = (-1e9, 0.0, 0)
        for i, a in enumerate(self.p.antennas):
            g = antenna_gain_dbi(a, ub)
            pl = polarization_loss_db(a, R, u, self.p.peer_polarization, self.p.peer_axis, self.p.peer_handedness)
            if g - pl > best[0] - best[1]:
                best = (g, pl, i)
        return best

    def _tx_eff_dbm(self, v_bus):
        p = self.p
        x = clamp(self.tx_power_dbm, p.tx_power_min_dbm, p.tx_power_max_dbm) - self.derate_db
        if p.eirp_limit_dbm < 90 and p.antennas:
            x = min(x, p.eirp_limit_dbm - max(a.peak_gain_dbi - a.cable_loss_db for a in p.antennas))
        if p.pa_v_knee_v > 0 and v_bus < p.pa_v_knee_v:
            x -= 20 * math.log10(p.pa_v_knee_v / max(v_bus, 1.0))
        return x

    def _ul_tx_power_dbm(self, prop, peer, v_bus):
        return self._tx_eff_dbm(v_bus)

    def _ceil(self, snr):
        return -10 * math.log10(10 ** (-snr / 10) + 10 ** (-self.p.snr_ceiling_db / 10))

    def _compute_link(self, t, dt, drone, env, v_bus):
        p = self.p
        peer = self._peer(t, dt, drone, env)
        self.peer = peer
        if peer is None:
            self.snr_ul = self.snr_dl = -60.0
            self.rssi = -150.0
            return
        circ = bool(p.antennas) and all(a.polarization == "circular" for a in p.antennas)
        tag = str(int(self.freq_hz / 1e5)) if p.hop_channels else ""
        prop = env.sample_link(peer.key, p.kind, self.freq_hz, drone.pos, peer.pos, drone.speed, t, peer.elev_deg, circ, tag)
        self.prop = prop
        u = unit(peer.pos - drone.pos)
        gd, pol, _ = self._best_antenna(drone.R.T @ u, u, drone.R)
        fade = prop.fading_db
        fade_ul = fade / p.peer_diversity_branches if (p.peer_diversity_branches > 1 and fade < 0) else fade
        base = prop.path_loss_db + prop.shadow_db + peer.extra_loss_db
        L_dl, L_ul = base - fade, base - fade_ul
        ptx = self._ul_tx_power_dbm(prop, peer, v_bus)
        self.last_ptx_dbm = ptx
        pr_ul = ptx + gd - pol + peer.rx_gain_dbi - peer.pointing_loss_db - L_ul
        pr_dl = peer.eirp_dbm - peer.pointing_loss_db + gd - pol - L_dl
        icls = self._icls()
        n_ul = undb(thermal_noise_dbm(self.bw_ul_hz, p.peer_noise_figure_db)) + env.interference_mw(icls, self.freq_hz, self.bw_ul_hz, peer.pos, t, p.jam_rejection_db) + peer.extra_ul_i_mw
        n_dl = undb(thermal_noise_dbm(self.bw_hz, p.noise_figure_db)) + env.interference_mw(icls, self.freq_hz, self.bw_hz, drone.pos, t, p.jam_rejection_db) + peer.extra_dl_i_mw
        self.snr_ul = self._ceil(pr_ul - db(n_ul))
        self.snr_dl = self._ceil(pr_dl - db(n_dl))
        self.rssi = pr_dl

    # ---- per-step update ---------------------------------------------------------
    def step(self, t, dt, drone, env, v_bus, ambient_c) -> LinkStatus:
        p = self.p
        self.t = t
        if p.v_min_operating_v > 0 and v_bus < p.v_min_operating_v and self.state != S_OFF and self.state != S_BOOT:
            self.n_brownouts += 1
            self._to(S_BOOT, p.boot_time_s)
        if self.state == S_BOOT:
            self.remain -= dt
            if self.remain <= 0:
                if self.want == "standby":
                    self._to(S_STANDBY)
                elif self.want == "on":
                    self._to(S_ACQ, self.rng.uniform(p.search_min_s, p.search_max_s))
                else:
                    self._to(S_OFF)
        if self.thermal_lock and self.temp_c < p.throttle_c - 5:
            self.thermal_lock = False
            if self.want != "off":
                self.request(self.want)
        if p.hop_channels and self.state not in (S_OFF, S_BOOT) and t >= self.next_hop:
            self._choose_channel(t, drone, env)
        self.measurable = self.state in (S_CONN, S_ACQ) or (self.state == S_STANDBY and p.measure_in_standby)
        if self.measurable:
            self._compute_link(t, dt, drone, env, v_bus)
        else:
            self.snr_ul = self.snr_dl = -60.0
            self.rssi, self.peer = -150.0, None
        snr_min = min(self.snr_ul, self.snr_dl)
        if self.state == S_ACQ:
            if self.peer is not None and snr_min >= p.snr_acq_db:
                self.remain -= dt
                if self.remain <= 0:
                    self._to(S_CONN)
        elif self.state == S_CONN:
            if snr_min < p.snr_rlf_db or self.peer is None:
                self.rlf_acc += dt
            else:
                self.rlf_acc = max(0.0, self.rlf_acc - 2 * dt)
            if self.rlf_acc >= p.rlf_timer_s:
                self.n_rlf += 1
                self.rlf_acc = 0.0
                self._to(S_ACQ, self.rng.uniform(p.reacq_min_s, p.reacq_max_s))
        self._update_phy(t, dt)
        self._power_thermal(dt, v_bus, ambient_c, drone)
        if t - self._hist_t >= 0.099:
            self._hist_t = t
            self.hist.append((t, self.rssi, self.snr_dl, self.snr_ul))
        return self.status(t)

    def _avail(self) -> float:
        return 1.0

    def _per(self, snr, thr, size_scale=1.0):
        x = self.p.per_slope_db * (snr - thr) + math.log(9.0)
        per = 1.0 / (1.0 + math.exp(clamp(x, -50, 50)))
        return per

    def _update_phy(self, t, dt):
        p = self.p
        conn = self.state == S_CONN
        interrupted = t < self.interrupt_until
        for name, d in self.dirs.items():
            snr = self.snr_ul if name == "ul" else self.snr_dl
            share = p.ul_share if name == "ul" else p.dl_share
            bw = self.bw_ul_hz if name == "ul" else self.bw_hz
            cap_limit = p.rate_cap_ul_bps if name == "ul" else p.rate_cap_dl_bps
            d.snr = snr
            if d.snr_est is None:
                d.snr_est = snr
            d.snr_est += (1 - math.exp(-dt / p.ra_tau_s)) * (snr - d.snr_est)
            target = 0
            for i, (thr, _) in enumerate(p.mcs_table):
                if d.snr_est - p.ra_margin_db >= thr:
                    target = i
            if target < d.mcs:
                d.mcs = target
            elif target > d.mcs and t - d.up_t >= p.ra_up_period_s:
                d.mcs, d.up_t = d.mcs + 1, t
            if p.fixed_mcs >= 0:
                d.mcs = min(p.fixed_mcs, len(p.mcs_table) - 1)
            thr, eff = p.mcs_table[d.mcs]
            d.rate = min(eff * bw, cap_limit)
            d.per = self._per(snr, thr) if conn else 1.0
            if conn:
                d.per = 1.0 - (1.0 - d.per) * (1.0 - self._extra_per(name))
            avail = self._avail()
            d.cap = d.rate * p.mac_efficiency * share * avail * (1 - d.per) * min(1.0, p.duty_cycle_limit / max(share, 1e-9)) if (conn and not interrupted) else 0.0
            d.cap = min(d.cap, self._cap_limit(name))
            # Gilbert-Elliott burst state with stationary loss = per
            pb, pbar = 0.8, d.per
            pg = 0.05 * pbar
            if pbar >= pb:
                d.bad, d.p_state = True, pbar
            elif pbar < 1e-6:
                d.bad, d.p_state = False, 0.0
            else:
                pi_b = clamp((pbar - pg) / (pb - pg), 0.0, 0.999)
                p_bg = clamp(dt / p.ge_burst_s, 0.0, 1.0)
                p_gb = p_bg * pi_b / (1 - pi_b)
                if d.bad and self.rng.random() < p_bg:
                    d.bad = False
                elif (not d.bad) and self.rng.random() < p_gb:
                    d.bad = True
                d.p_state = pb if d.bad else pg
            # two-class fluid queue (HP strict priority)
            if not conn:
                d.del_hp = d.del_lp = 0.0
                continue
            served = d.cap / 8.0 * dt
            hp = d.q_hp + d.off_hp / 8.0 * dt
            s_hp = min(hp, served)
            d.q_hp = hp - s_hp
            lp = d.q_lp + d.off_lp / 8.0 * dt
            s_lp = min(lp, served - s_hp)
            d.q_lp = lp - s_lp
            over = d.q_hp + d.q_lp - p.buffer_bytes
            if over > 0:
                cut = min(over, d.q_lp)
                d.q_lp -= cut
                d.drop_bytes += over
                d.q_hp -= (over - cut)
            d.del_hp, d.del_lp = s_hp * 8 / dt, s_lp * 8 / dt

    # ---- packet-level API ---------------------------------------------------------
    def residual_loss(self, direction: str, size_bytes: int, mean: bool = False) -> float:
        d = self.dirs[direction]
        if not self.link_up() or (self.p.ul_share if direction == "ul" else self.p.dl_share) <= 0:
            return 1.0
        pk = 1 - (1 - (d.per if mean else d.p_state)) ** (size_bytes / self.p.mtu_bytes)
        r = 1.0
        for k in range(self.p.max_tx):
            r *= clamp(pk * self.p.harq_gain ** k, 0.0, 1.0)
        return r

    def _extra_latency_mean_s(self, direction: str) -> float:
        return 0.0

    def _queue_delay_s(self, d: _Dir, klass: str) -> float:
        cap = max(d.cap / 8.0, 1.0)
        if klass == "hp":
            hol = (self.p.mtu_bytes / cap) * 0.5 if (d.off_lp > 0 or d.q_lp > 0) else 0.0
            return d.q_hp / cap + hol
        return (d.q_hp + d.q_lp) / cap

    def sample_latency(self, direction: str, klass: str, size_bytes: int) -> Tuple[float, bool]:
        """One-way latency sample (s) incl. queue; dropped=True if hopeless."""
        p, d = self.p, self.dirs[direction]
        wait = max(0.0, self.interrupt_until - self.t)
        qd = self._queue_delay_s(d, klass)
        if (not self.link_up()) or qd + wait > p.max_queue_delay_s or (p.ul_share if direction == "ul" else p.dl_share) <= 0:
            return qd + wait, True
        base = (p.lat_ul_ms if direction == "ul" else p.lat_dl_ms) + p.backhaul_ms
        jit = self._jitter_ms()
        lat = (base + jit) / 1000.0 + self._extra_latency_mean_s(direction) + qd + wait + size_bytes * 8 / max(d.cap, 1e3)
        return lat, False

    def send_packet(self, direction: str, size_bytes: int, klass: str = "hp") -> Tuple[bool, float]:
        lat, dropped = self.sample_latency(direction, klass, size_bytes)
        if dropped:
            return False, lat
        d, p = self.dirs[direction], self.p
        pk = 1 - (1 - d.p_state) ** (size_bytes / p.mtu_bytes)
        tries, ok = 0, False
        for k in range(p.max_tx):
            tries += 1
            if self.rng.random() > clamp(pk * p.harq_gain ** k, 0, 1):
                ok = True
                break
        return ok, lat + (tries - 1) * p.retx_delay_ms / 1000.0

    # ---- power / thermal ----------------------------------------------------------
    def _power_thermal(self, dt, v_bus, ambient_c, drone):
        p, s = self.p, self.state
        if s == S_OFF:
            P = p.p_off_w
        elif s == S_BOOT:
            P = p.p_boot_w
        elif s == S_STANDBY:
            P = p.p_standby_w
        else:
            P = p.p_idle_w
            if s == S_ACQ:
                P += p.p_search_extra_w
            else:
                ul, dl = self.dirs["ul"], self.dirs["dl"]
                ul_air = max(min(1.0, (ul.off_hp + ul.off_lp) / max(ul.cap, 1.0)) * p.ul_share, p.min_tx_duty)
                dl_air = min(1.0, (dl.off_hp + dl.off_lp) / max(dl.cap, 1.0)) * p.dl_share
                ptx_dbm = self.last_ptx_dbm if self.last_ptx_dbm else self._tx_eff_dbm(v_bus)
                ptx_w, ptx_max = undb(ptx_dbm) / 1000.0, undb(p.tx_power_max_dbm) / 1000.0
                eff = max(0.03, p.pa_eff_max * math.sqrt(min(1.0, ptx_w / ptx_max)))
                P += ul_air * (p.tx_overhead_w + ptx_w / eff) + dl_air * p.p_rx_w
        self.power_w = P
        self.energy_j += P * dt
        rth = p.rth_k_per_w / (1 + drone.speed / 8.0)
        self.temp_c = min(150.0, self.temp_c + dt * (P * p.heat_fraction - (self.temp_c - ambient_c) / rth) / p.cth_j_per_k)
        self.derate_db = clamp((self.temp_c - p.throttle_c) * 0.5, 0.0, 12.0)
        if self.temp_c > p.shutdown_c and self.state != S_OFF:
            self.thermal_lock = True
            self._to(S_OFF)

    # ---- status ----------------------------------------------------------------------
    def status(self, t) -> LinkStatus:
        ul, dl = self.dirs["ul"], self.dirs["dl"]
        ctrl_loss = self.residual_loss("dl", 64)
        lat_c = (self.p.lat_dl_ms + self.p.backhaul_ms) / 1000 + self._extra_latency_mean_s("dl") + self._queue_delay_s(dl, "hp") + 64 * 8 / max(dl.cap, 1e3)
        lat_v = (self.p.lat_ul_ms + self.p.backhaul_ms) / 1000 + self._extra_latency_mean_s("ul") + self._queue_delay_s(ul, "lp")
        if not self.link_up():
            lat_c = lat_v = 9.999
        if self.p.dl_share <= 0:
            lat_c = 9.999
        if self.p.ul_share <= 0:
            lat_v = 9.999
        outage = self._outage(ctrl_loss, ul)
        return LinkStatus(
            t, self.p.name, self.p.kind, self.state, self.measurable, self.rssi, self.snr_ul, self.snr_dl,
            ul.rate, dl.rate, ul.cap, dl.cap, ul.per, dl.per, ctrl_loss, lat_c * 1000, lat_v * 1000,
            self._queue_delay_s(ul, "lp") * 1000, self.power_w, self.last_ptx_dbm or self.tx_power_dbm, self.temp_c,
            bool(self.prop.los) if self.prop else False, outage,
            {"family": self.p.family, "freq_mhz": self.freq_hz / 1e6, "bw_mhz": self.bw_hz / 1e6,
             "analog_q": self.analog_quality() if self.p.analog_video else None, **self._extra()})

    def _extra(self) -> dict:
        return {}


class DirectRFRadio(Radio):
    def _peer(self, t, dt, drone, env):
        p, gs = self.p, env.gs
        u = unit(gs - drone.pos)
        loss = 0.0
        if p.peer_tracker_hpbw_deg > 0:
            rate = 0.0
            if self._u_prev is not None and dt > 0:
                rate = math.acos(clamp(float(u @ self._u_prev), -1, 1)) / dt
            err_deg = math.degrees(rate) * p.peer_tracker_lag_s + 0.5
            loss = min(12 * (err_deg / p.peer_tracker_hpbw_deg) ** 2, 20.0)
        self._u_prev = u
        return PeerInfo(pos=gs, key=(p.name, "gs"), eirp_dbm=p.peer_tx_power_dbm + p.peer_gain_dbi,
                        rx_gain_dbi=p.peer_gain_dbi, pointing_loss_db=loss)


class CellularRadio(Radio):
    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.serving: Optional[CellTower] = None
        self.meas: Dict[str, Tuple[float, CellTower]] = {}
        self.next_meas, self.ho_start, self.load_noise = 0.0, None, 0.0
        self.sinr_dl = -30.0

    def _tower_ok(self, tw: CellTower) -> bool:
        return (not self.p.rat or self.p.rat in tw.rats) and (not self.p.operator or self.p.operator in tw.operators)

    def _rx_from(self, t, drone, env, tw: CellTower):
        prop = env.sample_link((self.p.name, tw.id), "cellular", self.freq_hz, drone.pos, np.array(tw.pos), drone.speed, t)
        u = unit(np.array(tw.pos) - drone.pos)
        gd, pol, _ = self._best_antenna(drone.R.T @ u, u, drone.R)
        gt = tw.gain_toward(drone.pos)
        return tw.tx_power_dbm + self.p.peer_extra_gain_db + gt + gd - pol - prop.total_loss_db, gt

    def _peer(self, t, dt, drone, env):
        p = self.p
        self.load_noise += -self.load_noise / 5.0 * dt + 0.3 * math.sqrt(dt) * self.rng.standard_normal()
        if self.serving is None or t >= self.next_meas:
            self.next_meas = t + p.meas_period_s
            near = sorted([tw for tw in env.towers if self._tower_ok(tw)], key=lambda tw: np.linalg.norm(np.array(tw.pos) - drone.pos))[:p.n_meas_cells]
            self.meas = {tw.id: (self._rx_from(t, drone, env, tw)[0], tw) for tw in near}
            if not self.meas:
                return None
            best_id = max(self.meas, key=lambda k: self.meas[k][0])
            if self.serving is None:
                self.serving = self.meas[best_id][1]
            elif best_id != self.serving.id and self.serving.id in self.meas and \
                    self.meas[best_id][0] > self.meas[self.serving.id][0] + p.ho_hyst_db:
                self.ho_start = self.ho_start if self.ho_start is not None else t
                if t - self.ho_start >= p.ho_ttt_s:
                    self._handover(t, self.meas[best_id][1])
            else:
                self.ho_start = None
                if self.serving.id not in self.meas:
                    self.serving = self.meas[best_id][1]
        tw = self.serving
        rx, gt = self._rx_from(t, drone, env, tw)
        self.meas[tw.id] = (rx, tw)
        i_dl = sum(t2.load * undb(r) for k, (r, t2) in self.meas.items() if k != tw.id)
        rise = min(12.0, p.ul_rise_db_per_m * max(0.0, drone.pos[2] - 20.0)) + 2.0 * tw.load
        n_tw = undb(thermal_noise_dbm(self.bw_ul_hz, p.peer_noise_figure_db))
        self.sinr_dl = self.snr_dl
        return PeerInfo(pos=np.array(tw.pos), key=(p.name, tw.id), eirp_dbm=tw.tx_power_dbm + p.peer_extra_gain_db + gt, rx_gain_dbi=gt + 3.0 + p.peer_extra_gain_db,
                        extra_dl_i_mw=i_dl, extra_ul_i_mw=n_tw * (undb(rise) - 1.0))

    def _handover(self, t, new: CellTower):
        p = self.p
        self.ho_start = None
        self.n_handover += 1
        self.events.append((t, f"handover {self.serving.id}->{new.id}"))
        self.serving = new
        self.interrupt_until = t + self.rng.uniform(p.ho_interrupt_min_s, p.ho_interrupt_max_s)
        if self.state == S_CONN and self.sinr_dl < p.ho_fail_sinr_db and self.rng.random() < p.ho_fail_prob:
            self.events.append((t, "handover failure"))
            self._to(S_ACQ, self.rng.uniform(p.reacq_min_s, p.reacq_max_s))

    def _ul_tx_power_dbm(self, prop, peer, v_bus):
        p = self.p
        pc = p.ul_p0_dbm + 10 * math.log10(self.bw_ul_hz / 180e3) + p.ul_alpha * (prop.total_loss_db + peer.extra_loss_db)
        return min(pc, self._tx_eff_dbm(v_bus))

    def _avail(self):
        tw = self.serving
        return clamp(1.0 - (tw.load if tw else 0.5) * 0.6 * (1 + 0.5 * self.load_noise), 0.25, 1.0)

    def _extra(self):
        return {"serving": self.serving.id if self.serving else None, "handovers": self.n_handover}


class LEOSatRadio(Radio):
    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.serving: Optional[int] = None
        self.last_resel = -1e9
        self.range_m, self.elev, self.load_noise, self._prev_range = 1e6, 0.0, 0.0, None
        self.doppler_hz = 0.0
        self.n_resel = 0

    def _peer(self, t, dt, drone, env):
        p = self.p
        cons = env.constellations.get(p.constellation, env.constellation)
        idx, el, rng = cons.visible(t, drone.pos, p.min_elev_deg)
        if len(idx) == 0:
            self.serving = None
            return None
        self.load_noise += -self.load_noise / 8.0 * dt + 0.3 * math.sqrt(dt) * self.rng.standard_normal()
        need = self.serving is None or self.serving not in set(idx.tolist())
        if need or t - self.last_resel >= p.resel_period_s:
            new = int(idx[int(np.argmax(el))])
            if new != self.serving:
                if self.serving is not None and self.state == S_CONN:
                    self.interrupt_until = t + self.rng.uniform(0.02, 0.06)
                    self.n_resel += 1
                    self.events.append((t, f"sat handover {self.serving}->{new}"))
                self.serving = new
                self._prev_range = None
            self.last_resel = t
        enu, rng_m, elev = cons.single(self.serving, t, drone.pos)
        if elev < p.min_elev_deg - 3:
            self.serving = None
            return None
        if self._prev_range is not None and dt > 0:
            self.doppler_hz = -p.freq_hz * (rng_m - self._prev_range) / dt / C0
        self._prev_range, self.range_m, self.elev = rng_m, rng_m, elev
        return PeerInfo(pos=drone.pos + enu, key=(p.name, "sat"), eirp_dbm=p.sat_eirp_dbw + 30.0, rx_gain_dbi=p.sat_gain_dbi,
                        elev_deg=elev, range_m=rng_m)

    def _avail(self):
        return clamp(1.0 - self.p.sat_load * (1 + 0.6 * self.load_noise), 0.15, 1.0)

    def _extra_latency_mean_s(self, direction):
        return self.range_m * (1 + self.p.gw_factor) / C0 + self.p.sat_sched_ms / 1000.0

    def _extra(self):
        return {"sat": self.serving, "elev_deg": self.elev, "range_km": self.range_m / 1e3,
                "doppler_khz": self.doppler_hz / 1e3, "resel": self.n_resel}


class WifiMeshRadio(Radio):
    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.node: Optional[WifiNode] = None
        self.next_meas, self.roam_start = 0.0, None
        self.meas: Dict[str, Tuple[float, WifiNode]] = {}

    def _npos(self, n: WifiNode, t):
        return np.array(n.pos, float) + np.array(n.vel, float) * t

    def _rx_from(self, t, drone, env, n: WifiNode):
        pos = self._npos(n, t)
        prop = env.sample_link((self.p.name, n.id), "wifi", self.p.freq_hz, drone.pos, pos, drone.speed, t)
        u = unit(pos - drone.pos)
        gd, pol, _ = self._best_antenna(drone.R.T @ u, u, drone.R)
        return n.tx_power_dbm + n.gain_dbi + gd - pol - prop.total_loss_db

    def _peer(self, t, dt, drone, env):
        p = self.p
        if not env.wifi_nodes:
            return None
        if self.node is None or t >= self.next_meas:
            self.next_meas = t + 0.2
            near = sorted(env.wifi_nodes, key=lambda n: np.linalg.norm(self._npos(n, t) - drone.pos))[:4]
            self.meas = {n.id: (self._rx_from(t, drone, env, n), n) for n in near}
            best = max(self.meas, key=lambda k: self.meas[k][0])
            if self.node is None or self.node.id not in self.meas:
                self.node = self.meas[best][1]
            elif best != self.node.id and self.meas[best][0] > self.meas[self.node.id][0] + p.roam_hyst_db:
                self.roam_start = self.roam_start if self.roam_start is not None else t
                if t - self.roam_start >= p.roam_ttt_s:
                    self.events.append((t, f"roam {self.node.id}->{best}"))
                    self.node, self.roam_start = self.meas[best][1], None
                    self.interrupt_until = t + p.roam_interrupt_s
            else:
                self.roam_start = None
        n = self.node
        pos = self._npos(n, t)
        return PeerInfo(pos=pos, key=(p.name, n.id), eirp_dbm=n.tx_power_dbm + n.gain_dbi, rx_gain_dbi=n.gain_dbi,
                        range_m=float(np.linalg.norm(pos - drone.pos)))

    def _avail(self):
        n = self.node
        if n is None:
            return 0.0
        rng = self.peer.range_m if self.peer else 0.0
        return (1 - n.channel_busy) / max(n.hops, 1) / (1 + (rng / 3000.0) ** 2)

    def _extra_latency_mean_s(self, direction):
        n = self.node
        if n is None:
            return 0.0
        return (0.4e-3 / max(1 - n.channel_busy, 0.05) ** 2 + n.hops * self.p.hop_ms * 1e-3 + n.backhaul_latency_ms * 1e-3)

    def _extra(self):
        return {"node": self.node.id if self.node else None}


class MeshRFRadio(Radio):
    """Multi-hop MANET (Doodle Labs / Silvus style). Routes over env.mesh_nodes (static relays, relay UAVs on
    loiter orbits) to a gateway with an ETX-like metric. End-to-end capacity ~ min(hop rate)/hops (single
    channel, half-duplex), per-hop latency, compounded hop loss and route re-convergence outages."""

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.route: List[str] = []
        self.first: Optional[MeshNode] = None
        self.next_plan, self.next_edges, self.n_route_changes = -1.0, -1.0, 0
        self.edges: Dict[Tuple[str, str], float] = {}
        self.hop_rates: List[float] = []
        self.hop_pers: List[float] = []

    def _hop_phy(self, snr):
        p, m = self.p, -1
        for i, (thr, _) in enumerate(p.mcs_table):
            if snr - p.ra_margin_db >= thr:
                m = i
        if m < 0:
            return 0.0, 1.0
        thr, eff = p.mcs_table[m]
        return eff * self.bw_hz * p.mac_efficiency, self._per(snr, thr)

    def _node_snr(self, env, t, a: MeshNode, b: MeshNode):
        pa, pb = env.mesh_pos(a, t), env.mesh_pos(b, t)
        prop = env.sample_link((self.p.name, *sorted((a.id, b.id))), "direct", self.freq_hz, pa, pb, 0.0, t)
        pr = min(a.tx_power_dbm, b.tx_power_dbm) + a.gain_dbi + b.gain_dbi - prop.total_loss_db
        n = undb(thermal_noise_dbm(self.bw_hz, 6.0)) + env.interference_mw(self._icls(), self.freq_hz, self.bw_hz, pb, t, self.p.jam_rejection_db)
        return pr - db(n)

    def _drone_snr(self, env, t, drone, n: MeshNode):
        pos = env.mesh_pos(n, t)
        prop = env.sample_link((self.p.name, n.id), "direct", self.freq_hz, drone.pos, pos, drone.speed, t)
        u = unit(pos - drone.pos)
        gd, pol, _ = self._best_antenna(drone.R.T @ u, u, drone.R)
        pr = n.tx_power_dbm + n.gain_dbi + gd - pol - prop.total_loss_db
        nn = undb(thermal_noise_dbm(self.bw_hz, self.p.noise_figure_db)) + env.interference_mw(self._icls(), self.freq_hz, self.bw_hz, drone.pos, t, self.p.jam_rejection_db)
        return pr - db(nn)

    def _cost(self, snr):
        rate, per = self._hop_phy(snr)
        return 1e9 if rate <= 0 else 1e6 / (rate * max(1 - per, 0.05)) + 0.02

    def _plan(self, t, drone, env):
        p = self.p
        nodes = {n.id: n for n in env.mesh_nodes}
        if t >= self.next_edges:
            self.next_edges = t + 1.0
            ids = list(nodes)
            self.edges = {}
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    self.edges[(ids[a], ids[b])] = self.edges[(ids[b], ids[a])] = self._node_snr(env, t, nodes[ids[a]], nodes[ids[b]])
        dsnr = {i: self._drone_snr(env, t, drone, n) for i, n in nodes.items()}
        dist, prev = {}, {}
        for i, sn in dsnr.items():
            if sn >= p.mesh_min_hop_snr_db:
                dist[i], prev[i] = self._cost(sn), None
        done = set()
        while True:
            cand = [(d, i) for i, d in dist.items() if i not in done]
            if not cand:
                break
            d0, u = min(cand)
            done.add(u)
            for v in nodes:
                sn = self.edges.get((u, v))
                if v in done or sn is None or sn < p.mesh_min_hop_snr_db:
                    continue
                nd = d0 + self._cost(sn)
                if nd < dist.get(v, 1e18):
                    dist[v], prev[v] = nd, u
        gws = [(dist[i], i) for i in dist if nodes[i].is_gateway]
        best, best_cost = [], 1e18
        if gws:
            best_cost, g = min(gws)
            while g is not None:
                best.append(g)
                g = prev[g]
            best.reverse()

        def route_cost(r):
            if not r or dsnr.get(r[0], -99) < p.mesh_min_hop_snr_db - 3:
                return None
            c = self._cost(dsnr[r[0]])
            for a, b in zip(r, r[1:]):
                sn = self.edges.get((a, b), -99)
                if sn < p.mesh_min_hop_snr_db - 3:
                    return None
                c += self._cost(sn)
            return c
        cur = route_cost(self.route)
        if not best:
            if self.route:
                self.events.append((t, "mesh route lost"))
            self.route, self.first = [], None
        elif cur is None:
            self._switch(t, best, nodes, True)
        elif best != self.route and best_cost < 0.75 * cur:
            self._switch(t, best, nodes, False)
        self.hop_rates = [self._hop_phy(self.edges[(a, b)])[0] for a, b in zip(self.route, self.route[1:])]
        self.hop_pers = [self._hop_phy(self.edges[(a, b)])[1] for a, b in zip(self.route, self.route[1:])]

    def _switch(self, t, route, nodes, full):
        self.n_route_changes += 1
        self.events.append((t, f"mesh route {'>'.join(route)}" + ("" if full else " (proactive)")))
        self.route, self.first = route, nodes[route[0]]
        self.interrupt_until = max(self.interrupt_until, t + self.p.mesh_convergence_s * (1.0 if full else 0.2))

    def _peer(self, t, dt, drone, env):
        if t >= self.next_plan:
            self.next_plan = t + 0.25
            self._plan(t, drone, env)
        n = self.first
        if n is None:
            return None
        pos = env.mesh_pos(n, t)
        return PeerInfo(pos=pos, key=(self.p.name, n.id), eirp_dbm=n.tx_power_dbm + n.gain_dbi, rx_gain_dbi=n.gain_dbi,
                        range_m=float(np.linalg.norm(pos - drone.pos)))

    def _cap_limit(self, name):
        return min(self.hop_rates) / max(len(self.route), 1) if self.hop_rates else 1e18

    def _extra_per(self, name):
        keep = 1.0
        for pr in self.hop_pers:
            keep *= (1 - pr)
        return 1 - keep

    def _extra_latency_mean_s(self, direction):
        return max(len(self.route) - 1, 0) * self.p.mesh_hop_ms / 1000.0

    def _extra(self):
        return {"route": ">".join(self.route), "hops": len(self.route), "route_changes": self.n_route_changes}


class FiberTetherRadio(Radio):
    """Fiber-optic tether (spool carried on the drone). Near-zero latency, immune to RF jamming/weather,
    but range-limited by spool length; carried spool mass falls as it pays out; optional break hazard."""

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.paid_out, self.broken, self._pp = 0.0, False, None

    @property
    def extra_mass_kg(self) -> float:
        return 0.0 if self.broken else self.p.tether_mass_kg_per_m * max(self.p.tether_len_m - self.paid_out, 0.0)

    def step(self, t, dt, drone, env, v_bus, ambient_c):
        p = self.p
        if self._pp is not None and not self.broken:
            ds = float(np.linalg.norm(drone.pos - self._pp))
            self.paid_out += ds
            if self.fault or self.paid_out >= p.tether_len_m or (p.tether_break_prob_per_km > 0 and self.rng.random() < p.tether_break_prob_per_km * ds / 1000.0):
                self.broken = True
                self.events.append((t, "tether broken" if self.paid_out < p.tether_len_m else "tether exhausted"))
        self._pp = drone.pos.copy()
        return super().step(t, dt, drone, env, v_bus, ambient_c)

    def _compute_link(self, t, dt, drone, env, v_bus):
        if self.broken:
            self.peer, self.snr_ul, self.snr_dl, self.rssi = None, -60.0, -60.0, -150.0
            return
        self.peer = PeerInfo(pos=env.gs, key=("fiber",), eirp_dbm=0.0, rx_gain_dbi=0.0)
        self.snr_ul = self.snr_dl = self.p.snr_ceiling_db
        self.rssi = -8.0 - 0.35 * self.paid_out / 1000.0       # optical power at the receiver [dBm]
        self.last_ptx_dbm = 0.0

    def _extra(self):
        return {"paid_out_m": self.paid_out, "remaining_m": max(self.p.tether_len_m - self.paid_out, 0.0), "broken": self.broken}


class BondedRadio(Radio):
    """Virtual aggregation link (multi-SIM / multi-modem bonding, Peplink/Speedify style) over member radios.
    aggregate: capacity = sum, packets spread by capacity, reorder-buffer latency. duplicate: first copy wins."""

    def __init__(self, p, rng):
        super().__init__(p, rng)
        self.members: List[Radio] = []

    def bind(self, radios: Dict[str, "Radio"]):
        self.members = [radios[n] for n in self.p.members if n in radios]

    def request(self, mode):
        self.want = mode
        for m in self.members:
            m.request(mode)

    def link_up(self):
        return any(m.link_up() for m in self.members)

    def _weights(self, dn):
        ups = [m for m in self.members if m.link_up()]
        tot = sum(m.dirs[dn].cap for m in ups)
        return [(m, (m.dirs[dn].cap / tot) if tot > 0 else 1.0 / len(ups)) for m in ups]

    def distribute(self):
        for dn in ("ul", "dl"):
            d = self.dirs[dn]
            if self.p.bond_mode == "duplicate":
                for m in self.members:
                    if m.link_up():
                        m.dirs[dn].off_hp += d.off_hp
                        m.dirs[dn].off_lp += d.off_lp
            else:
                for m, w in self._weights(dn):
                    m.dirs[dn].off_hp += d.off_hp * w
                    m.dirs[dn].off_lp += d.off_lp * w

    def step(self, t, dt, drone, env, v_bus, ambient_c):
        self.t = t
        ms = self.members
        ups = [m for m in ms if m.link_up()]
        states = [m.state for m in ms]
        self.state = S_CONN if ups else (S_ACQ if S_ACQ in states else (S_STANDBY if S_STANDBY in states else (S_BOOT if S_BOOT in states else S_OFF)))
        self.measurable = any(m.measurable for m in ms)
        if ms:
            self.rssi, self.snr_ul, self.snr_dl = max(m.rssi for m in ms), max(m.snr_ul for m in ms), max(m.snr_dl for m in ms)
            self.last_ptx_dbm = max(m.last_ptx_dbm for m in ms)
        dup = self.p.bond_mode == "duplicate"
        for dn in ("ul", "dl"):
            d = self.dirs[dn]
            caps = [m.dirs[dn].cap for m in ups]
            d.cap = (max(caps) if dup else sum(caps)) if caps else 0.0
            d.rate = (max(m.dirs[dn].rate for m in ups) if dup else sum(m.dirs[dn].rate for m in ups)) if ups else 0.0
            if ups:
                if dup:
                    d.per = min(m.dirs[dn].per for m in ups)
                    d.p_state = min(m.dirs[dn].p_state for m in ups)
                else:
                    tc = sum(caps) or 1.0
                    d.per = sum(m.dirs[dn].cap * m.dirs[dn].per for m in ups) / tc
                    d.p_state = sum(m.dirs[dn].cap * m.dirs[dn].p_state for m in ups) / tc
            else:
                d.per = d.p_state = 1.0
            d.q_hp, d.q_lp = sum(m.dirs[dn].q_hp for m in ms), sum(m.dirs[dn].q_lp for m in ms)
        self.power_w = self.p.p_idle_w if ups else self.p.p_standby_w
        self.energy_j += self.power_w * dt
        self.temp_c = ambient_c
        if t - self._hist_t >= 0.099:
            self._hist_t = t
            self.hist.append((t, self.rssi, self.snr_dl, self.snr_ul))
        return self.status(t)

    def sample_latency(self, direction, klass, size_bytes):
        ws = self._weights(direction)
        if not ws:
            return 0.0, True
        if self.p.bond_mode == "duplicate":
            res = [m.sample_latency(direction, klass, size_bytes) for m, _ in ws]
            ok = [r[0] for r in res if not r[1]]
            return (min(ok), False) if ok else (min(r[0] for r in res), True)
        m = ws[int(self.rng.choice(len(ws), p=[w for _, w in ws]))][0]
        lat, dropped = m.sample_latency(direction, klass, size_bytes)
        return lat + self.p.bond_reorder_ms / 1000.0, dropped

    def send_packet(self, direction, size_bytes, klass="hp"):
        ws = self._weights(direction)
        if not ws:
            return False, 0.0
        if self.p.bond_mode == "duplicate":
            res = [m.send_packet(direction, size_bytes, klass) for m, _ in ws]
            ok = [lat for o, lat in res if o]
            return (True, min(ok)) if ok else (False, max(lat for _, lat in res))
        m = ws[int(self.rng.choice(len(ws), p=[w for _, w in ws]))][0]
        ok, lat = m.send_packet(direction, size_bytes, klass)
        return ok, lat + self.p.bond_reorder_ms / 1000.0

    def residual_loss(self, direction, size_bytes, mean=False):
        ws = self._weights(direction)
        if not ws:
            return 1.0
        rl = [(m.residual_loss(direction, size_bytes, mean), w) for m, w in ws]
        if self.p.bond_mode == "duplicate":
            r = 1.0
            for x, _ in rl:
                r *= x
            return r
        return sum(x * w for x, w in rl)

    def _extra(self):
        return {"members_up": [m.p.name for m in self.members if m.link_up()], "mode": self.p.bond_mode}


RADIO_CLASSES = {"direct": DirectRFRadio, "cellular": CellularRadio, "sat": LEOSatRadio, "wifi": WifiMeshRadio,
                 "mesh": MeshRFRadio, "fiber": FiberTetherRadio, "bond": BondedRadio}
SatRadio = LEOSatRadio


# =============================================================================
# 7. Traffic, routing and the multi-link suite (+ metrics)
# =============================================================================
@dataclass
class CodecParams:
    """Video representation over constrained links. standard | ladder | neural | event | key_object | auto."""
    mode: str = "standard"
    ladder: List[Tuple[float, float, float]] = field(default_factory=lambda: [      # (bps, fps, quality cap)
        (0.15e6, 8.0, 0.30), (0.4e6, 12.0, 0.45), (1.0e6, 20.0, 0.62), (2.5e6, 30.0, 0.80), (5.0e6, 30.0, 0.92), (8.0e6, 30.0, 1.0)])
    ladder_up_margin: float = 1.15
    neural_gain: float = 2.5               # learned codec: quality at bps*gain (bits saved vs H.264/5)
    neural_power_w: float = 9.0            # onboard NPU/GPU encoder
    neural_latency_ms: float = 25.0        # extra encode/decode latency
    semantic_hz: float = 4.0               # key-object / semantic updates per second
    semantic_bps: float = 24e3             # detections + crops/keypoints
    semantic_fec: float = 0.5
    semantic_min_hz: float = 1.0
    semantic_quality: float = 0.4          # situational-awareness value of a semantic-only picture
    event_idle_bps: float = 0.12e6         # event-based: low-rate thumbnails between events
    event_idle_hz: float = 2.0
    event_rate_hz: float = 0.03            # events per second (Poisson)
    event_dur_s: float = 6.0
    detector_power_w: float = 6.0          # onboard detector (key_object / event)
    auto_dwell_s: float = 3.0
    semantic_deadline_s: float = 1.5       # semantic updates tolerate far more latency than live video
    auto_thresholds_bps: Tuple[float, float, float] = (0.15e6, 0.8e6, 2.5e6)   # key_object|neural|ladder|standard


@dataclass
class TrafficParams:
    control_hz: float = 50.0
    control_bytes: int = 64
    telemetry_hz: float = 10.0
    telemetry_bytes: int = 200
    video_fps: float = 30.0
    video_max_bps: float = 8e6
    video_min_bps: float = 0.4e6
    video_start_bps: float = 4e6
    video_fec_overhead: float = 0.15
    video_mtu: int = 1200
    gop_frames: int = 30
    keyframe_factor: float = 4.0
    encoder_headroom: float = 0.8
    encoder_tau_down_s: float = 0.5
    encoder_tau_up_s: float = 3.0
    feedback_delay_s: float = 0.3
    video_adapt: bool = True
    keyframe_request_delay_s: float = 0.3   # decoder PLI/FIR round trip; <=0 disables (wait for next GOP)
    playout_deadline_s: float = 0.25
    quality_r0_bps: float = 2e6
    encoder_w_base: float = 2.5
    encoder_w_per_mbps: float = 0.6
    # reliability definitions used for blind-time accounting
    control_hard_limit_s: float = 0.15
    control_gap_s: float = 0.25
    video_min_fps: float = 15.0
    video_window_s: float = 1.0
    codec: CodecParams = field(default_factory=CodecParams)


@dataclass
class RoutingPolicy:
    control_links: List[str] = field(default_factory=lambda: ["rf24"])
    control_mode: str = "primary"        # primary (first link) | duplicate
    video_links: List[str] = field(default_factory=lambda: ["rf24"])
    video_mode: str = "split"            # split | duplicate
    video_weights: Optional[Dict[str, float]] = None   # None -> proportional to capacity
    video_bps_override: Optional[float] = None


def frame_success_prob(n: int, m: int, p: float) -> float:
    """P(at most m of n packets lost) with i.i.d. loss p (FEC tolerates m)."""
    if p <= 0:
        return 1.0
    if m >= n:
        return 1.0
    if p >= 1:
        return 0.0
    if n <= 400:
        return min(1.0, sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(m + 1)))
    mu, sd = n * p, math.sqrt(n * p * (1 - p))
    return 0.5 * (1 + math.erf((m + 0.5 - mu) / (sd * math.sqrt(2) + 1e-9)))


class MultiLinkRadioSuite:
    def __init__(self, radios: List[Radio], traffic: TrafficParams, routing: RoutingPolicy, rng):
        self.radios = {r.p.name: r for r in radios}
        self.tp, self.routing, self.rng = traffic, routing, rng
        self.status: Dict[str, LinkStatus] = {}
        self.t = 0.0
        self.video_bps = traffic.video_start_bps
        self._fb: deque = deque()
        self._acc = {"ctrl": 0.0, "tele": 0.0, "video": 0.0}
        self._vacc: Dict[str, float] = {}
        self.frame_idx, self.gop_broken, self._force_key_t = 0, False, None
        self.last_ctrl_ok_t, self.last_tele_t = 0.0, 0.0
        self._decoded: deque = deque()
        self.video_quality = 1.0
        self.ctrl_lat: List[float] = []
        self.counts = dict(ctrl_sent=0, ctrl_ontime=0, ctrl_late=0, ctrl_lost=0, frames=0, frames_ok=0, frames_late=0,
                           frames_lost=0, frames_broken=0)
        self.m = dict(blind_s=0.0, ctrl_out_s=0.0, vid_out_s=0.0, blind_episodes=0, longest_blind_s=0.0, vq_int=0.0, T=0.0)
        self._blind_run, self._was_blind = 0.0, False
        self.ctrl_ok = self.video_ok = True
        self.p_encoder_w = traffic.encoder_w_base
        self.switch_count = 0
        self.link_frames: Dict[str, dict] = {}
        self.aliases: Dict[str, str] = {}
        self.rate_state = traffic.video_start_bps
        self.codec_request, self.codec_mode, self._mode_t, self._rung = traffic.codec.mode, None, -1e9, 0
        self.cur_fps, self.cur_min_fps, self.q_gain, self.q_cap, self.q_fixed = traffic.video_fps, traffic.video_min_fps, 1.0, 1.0, None
        self.codec_power_w, self.codec_lat_s, self.cur_fec, self.indep_frames = 0.0, 0.0, -1.0, False
        self.comm_energy_j = 0.0
        self.event_active, self._event_end = False, None
        self._ana_t, self._ana_q, self._dig_t = -1e9, 0.0, 0.0
        self.m.update(ev_s=0.0, ev_cov_s=0.0)

    # ---- control API (used by Part 3) ----------------------------------------------
    def radio(self, name: str) -> Radio:
        return self.radios[self.aliases.get(name, name)]

    def set_codec(self, mode: str):
        assert mode in ("standard", "ladder", "neural", "event", "key_object", "auto"), mode
        self.codec_request = mode

    def set_routing(self, **kw):
        for k, v in kw.items():
            setattr(self.routing, k, v)

    def warm_start(self, t0=0.0):
        for r in self.radios.values():
            r.warm_start()
        self.last_ctrl_ok_t = self.last_tele_t = t0
        self._dig_t = t0
        for i in range(int(self.tp.video_window_s * self.tp.video_fps)):
            self._decoded.append(t0 - i / self.tp.video_fps)

    def total_power_w(self) -> float:
        return sum(r.power_w for r in self.radios.values()) + self.p_encoder_w

    # ---- features / observation -------------------------------------------------------
    def history(self, name: str, window_s: float = 5.0) -> np.ndarray:
        h = [x for x in self.radios[name].hist if x[0] >= self.t - window_s]
        return np.array(h) if h else np.zeros((0, 4))      # cols: t, rssi, snr_dl, snr_ul

    def control_lost_duration(self) -> float:
        return max(0.0, self.t - self.last_ctrl_ok_t)

    def observation(self) -> dict:
        return {"t": self.t, "ctrl_ok": self.ctrl_ok, "video_ok": self.video_ok, "video_bps": self.video_bps,
                "codec": self.codec_mode or self.codec_request, "event_active": self.event_active,
                "links": {n: dataclasses.asdict(s) for n, s in self.status.items()}}

    # ---- step ---------------------------------------------------------------------------
    def _res(self, n):
        return self.aliases.get(n, n)

    def _vcap(self, n) -> float:
        """Usable video capacity of a link [bps]. Analog: the picture, not bits, is carried."""
        r = self.radios[n]
        if not r.link_up():
            return 0.0
        if r.p.analog_video:
            return self.tp.video_max_bps if r.analog_quality() >= r.p.analog_min_quality else 0.0
        return r.dirs["ul"].cap * (1 - r.residual_loss("ul", self.tp.video_mtu, mean=True))

    def _targets(self):
        R = self.routing
        ctrl_all = [self._res(n) for n in R.control_links]
        ctrl = ctrl_all[:1] if R.control_mode == "primary" else ctrl_all
        vids = [self._res(n) for n in R.video_links]
        vw = {self._res(k): v for k, v in R.video_weights.items()} if R.video_weights else None
        if R.video_mode == "duplicate" or len(vids) <= 1:
            w = {n: 1.0 for n in vids}
        elif vw:
            tot = sum(vw.get(n, 0.0) for n in vids) or 1.0
            w = {n: vw.get(n, 0.0) / tot for n in vids}
        else:
            caps = {n: self._vcap(n) for n in vids}
            tot = sum(caps.values())
            w = {n: (caps[n] / tot if tot > 0 else 1.0 / len(vids)) for n in vids}
        return ctrl, vids, w

    def step(self, t, dt, drone, env, v_bus, ambient_c):
        tp = self.tp
        self.t = t
        ctrl, vids, w = self._targets()
        for r in self.radios.values():
            for d in r.dirs.values():
                d.off_hp = d.off_lp = 0.0
        for n in ctrl:
            self.radios[n].dirs["dl"].off_hp += tp.control_hz * tp.control_bytes * 8
        for n in (ctrl[:1] if self.routing.control_mode == "primary" else ctrl):
            self.radios[n].dirs["ul"].off_hp += tp.telemetry_hz * tp.telemetry_bytes * 8
        for n in vids:
            if not self.radios[n].p.analog_video:        # analog FM carries the picture itself, not bits
                self.radios[n].dirs["ul"].off_lp += self.video_bps * (w[n] if self.routing.video_mode == "split" else 1.0)
        bonded = [r for r in self.radios.values() if isinstance(r, BondedRadio)]
        for b in bonded:
            b.distribute()
        for r in [r for r in self.radios.values() if not isinstance(r, BondedRadio)] + bonded:
            if r.p.enabled:
                self.status[r.p.name] = r.step(t, dt, drone, env, v_bus, ambient_c)
        cap_est = self._encoder(dt, vids)
        self._codec_step(t, dt, cap_est)
        self._control(dt, ctrl)
        self._telemetry(dt, ctrl)
        self._video(dt, vids, w)
        self._metrics(dt)
        self.p_encoder_w = tp.encoder_w_base + tp.encoder_w_per_mbps * self.video_bps / 1e6 + self.codec_power_w
        self.comm_energy_j += self.total_power_w() * dt

    def _encoder(self, dt, vids):
        """Rate control (capacity feedback with delay + lag). Returns the instantaneous capacity-based target."""
        tp, R = self.tp, self.routing
        up = [n for n in vids if self.radios[n].link_up()]
        if R.video_bps_override is not None:
            target = R.video_bps_override
        elif tp.video_adapt:
            caps = [self._vcap(n) for n in up]
            cap = (min(caps) if R.video_mode == "duplicate" else sum(caps)) if caps else 0.0
            ceil = [self.radios[n].p.max_video_bps for n in up]
            ceiling = min(tp.video_max_bps, (min(ceil) if R.video_mode == "duplicate" else sum(ceil)) if ceil else tp.video_max_bps)
            raw = tp.encoder_headroom * cap - tp.telemetry_hz * tp.telemetry_bytes * 8
            floor = tp.video_min_bps if self.codec_request == "standard" else 0.05e6
            target = clamp(raw, floor, max(ceiling, floor))
            self._raw_cap = max(raw, 0.0)
        else:
            target = tp.video_max_bps
        fixed = [self.radios[n].p.fixed_video_bps for n in up if self.radios[n].p.fixed_video_bps > 0]
        if fixed:
            target = min(fixed)
        cap_est = getattr(self, "_raw_cap", target) if R.video_bps_override is None and tp.video_adapt else target
        self._fb.append((self.t, target))
        while len(self._fb) > 1 and self._fb[1][0] <= self.t - tp.feedback_delay_s:
            self._fb.popleft()
        tgt = self._fb[0][1]
        tau = tp.encoder_tau_down_s if tgt < self.rate_state else tp.encoder_tau_up_s
        self.rate_state += (1 - math.exp(-dt / tau)) * (tgt - self.rate_state)
        if fixed:
            self.rate_state = min(fixed)
        return cap_est

    def _codec_step(self, t, dt, cap_est):
        """Video representation layer: standard | ladder | neural | event | key_object | auto (capacity-driven)."""
        tp, cd = self.tp, self.tp.codec
        if self._event_end is None or t >= self._event_end:
            self.event_active = False
            if self.rng.random() < cd.event_rate_hz * dt:
                self.event_active, self._event_end = True, t + cd.event_dur_s
        mode = self.codec_request
        if mode == "auto":
            th = cd.auto_thresholds_bps
            if self.codec_mode is None or t - self._mode_t >= cd.auto_dwell_s:
                new = "key_object" if cap_est < th[0] else "neural" if cap_est < th[1] else "ladder" if cap_est < th[2] else "standard"
                if new != self.codec_mode:
                    self.codec_mode, self._mode_t = new, t
            mode = self.codec_mode
        else:
            self.codec_mode = mode
        fps, min_fps, gain, power, lat, qf, qcap, fec, indep = tp.video_fps, tp.video_min_fps, 1.0, 0.0, 0.0, None, 1.0, -1.0, False
        bps = self.rate_state
        if mode == "ladder":
            lad = cd.ladder
            idx = max([i for i, r in enumerate(lad) if r[0] <= bps] or [0])
            if idx > self._rung and bps < lad[idx][0] * cd.ladder_up_margin:
                idx = self._rung
            self._rung = idx
            bps, fps, qcap = lad[idx]
            min_fps = max(5.0, 0.5 * fps)
        elif mode == "neural":
            gain, power, lat = cd.neural_gain, cd.neural_power_w, cd.neural_latency_ms / 1000.0
        elif mode == "event":
            power = cd.detector_power_w
            if not self.event_active:
                bps, fps, min_fps, qf, indep = cd.event_idle_bps, cd.event_idle_hz, cd.semantic_min_hz, 0.5, True
        elif mode == "key_object":
            bps, fps, min_fps, qf, power, fec, indep = cd.semantic_bps, cd.semantic_hz, cd.semantic_min_hz, cd.semantic_quality, cd.detector_power_w, cd.semantic_fec, True
        self.video_bps = bps
        self.cur_fps, self.cur_min_fps, self.q_gain, self.codec_power_w = fps, min_fps, gain, power
        self.codec_lat_s, self.q_fixed, self.q_cap, self.cur_fec, self.indep_frames = lat, qf, qcap, fec, indep

    def _control(self, dt, ctrl):
        tp = self.tp
        self._acc["ctrl"] += tp.control_hz * dt
        while self._acc["ctrl"] >= 1.0:
            self._acc["ctrl"] -= 1.0
            best = None
            for n in ctrl:
                ok, lat = self.radios[n].send_packet("dl", tp.control_bytes, "hp")
                if ok and (best is None or lat < best):
                    best = lat
            self.counts["ctrl_sent"] += 1
            if best is None:
                self.counts["ctrl_lost"] += 1
            elif best <= tp.control_hard_limit_s:
                self.counts["ctrl_ontime"] += 1
                self.last_ctrl_ok_t = self.t
                self.ctrl_lat.append(best)
            else:
                self.counts["ctrl_late"] += 1
                self.ctrl_lat.append(best)

    def _telemetry(self, dt, ctrl):
        self._acc["tele"] += self.tp.telemetry_hz * dt
        while self._acc["tele"] >= 1.0:
            self._acc["tele"] -= 1.0
            for n in (ctrl[:1] if self.routing.control_mode == "primary" else ctrl):
                ok, _ = self.radios[n].send_packet("ul", self.tp.telemetry_bytes, "hp")
                if ok:
                    self.last_tele_t = self.t

    def _video(self, dt, vids, w):
        tp, R = self.tp, self.routing
        fps = self.cur_fps
        self._acc["video"] += fps * dt
        while self._acc["video"] >= 1.0:
            self._acc["video"] -= 1.0
            gop = max(1, int(round(fps)))
            key = self.indep_frames or self.frame_idx % gop == 0
            if self._force_key_t is not None and self.t >= self._force_key_t:
                key, self._force_key_t = True, None
                self.frame_idx = 0
            kf = 1.0 if self.indep_frames else tp.keyframe_factor
            norm = (gop - 1 + kf) / gop
            size = self.video_bps / fps / norm * (kf if key else 1.0) / 8.0
            if R.video_mode == "duplicate":
                chosen = list(vids)
            else:
                for n in vids:
                    self._vacc[n] = self._vacc.get(n, 0.0) + w[n]
                pick = max(vids, key=lambda n: self._vacc[n]) if vids else None
                if pick:
                    self._vacc[pick] -= 1.0
                chosen = [pick] if pick else []
            ok_dig = ok_ana = late = False
            ana_q = 0.0
            for n in chosen:
                r = self.radios[n]
                lf = self.link_frames.setdefault(n, dict(sent=0, drop=0, loss=0, late=0))
                lf["sent"] += 1
                if r.p.analog_video:
                    q = r.analog_quality()
                    lat = (r.p.lat_ul_ms + r._jitter_ms()) / 1000.0
                    if q >= r.p.analog_min_quality and lat <= tp.playout_deadline_s:
                        ok_ana, ana_q = True, max(ana_q, q)
                    else:
                        lf["drop"] += 1
                    continue
                fec = r.p.video_fec_overhead if r.p.video_fec_overhead >= 0 else (self.cur_fec if self.cur_fec >= 0 else tp.video_fec_overhead)
                n_pk = max(1, math.ceil(size * (1 + fec) / tp.video_mtu))
                m = int(fec / (1 + fec) * n_pk)
                lat, dropped = r.sample_latency("ul", "lp", int(size * (1 + fec)))
                lat += self.codec_lat_s
                if dropped:
                    lf["drop"] += 1
                    continue
                if self.rng.random() < frame_success_prob(n_pk, m, self._pkt_resid(r)):
                    if lat <= (self.tp.codec.semantic_deadline_s if self.indep_frames else tp.playout_deadline_s):
                        ok_dig = True
                    else:
                        late = True
                        lf["late"] += 1
                else:
                    lf["loss"] += 1
            self.frame_idx += 1
            self.counts["frames"] += 1
            decoded = False
            if ok_ana:
                decoded = True
                self._ana_t, self._ana_q = self.t, ana_q
            elif ok_dig:
                self.gop_broken = self.gop_broken and not key
                decoded = not self.gop_broken
                if decoded:
                    self._dig_t = self.t
                else:
                    self.counts["frames_broken"] += 1
            else:
                self.gop_broken = True
                if tp.keyframe_request_delay_s > 0 and self._force_key_t is None:
                    self._force_key_t = self.t + tp.keyframe_request_delay_s
                self.counts["frames_late" if late else "frames_lost"] += 1
            if decoded:
                self.counts["frames_ok"] += 1
                self._decoded.append(self.t)
        while self._decoded and self._decoded[0] < self.t - tp.video_window_s:
            self._decoded.popleft()

    def _pkt_resid(self, r: Radio) -> float:
        return r.residual_loss("ul", self.tp.video_mtu)

    def _metrics(self, dt):
        tp, m = self.tp, self.m
        self.ctrl_ok = (self.t - self.last_ctrl_ok_t) <= tp.control_gap_s
        self.video_ok = len(self._decoded) / tp.video_window_s >= self.cur_min_fps
        q_dig = self.q_fixed if self.q_fixed is not None else min(self.q_cap, 1 - math.exp(-self.video_bps * self.q_gain / tp.quality_r0_bps))
        q = max(self._ana_q if self.t - self._ana_t < 0.3 else 0.0, q_dig if self.t - self._dig_t < 0.5 else 0.0)
        self.video_quality = q if self.video_ok else 0.0
        m["T"] += dt
        m["vq_int"] += self.video_quality * dt
        m["ctrl_out_s"] += dt * (not self.ctrl_ok)
        m["vid_out_s"] += dt * (not self.video_ok)
        if self.event_active:
            m["ev_s"] += dt
            m["ev_cov_s"] += dt * (self.video_ok and self.video_quality >= 0.35)
        blind = (not self.ctrl_ok) and (not self.video_ok)
        if blind:
            m["blind_s"] += dt
            self._blind_run += dt
            m["longest_blind_s"] = max(m["longest_blind_s"], self._blind_run)
            if not self._was_blind:
                m["blind_episodes"] += 1
        else:
            self._blind_run = 0.0
        self._was_blind = blind

    def summary(self) -> dict:
        m, c = self.m, self.counts
        lat = np.array(self.ctrl_lat) * 1000 if self.ctrl_lat else np.zeros(1)
        T = max(m["T"], 1e-9)
        return {
            "duration_s": T, "blind_time_s": m["blind_s"], "blind_pct": 100 * m["blind_s"] / T,
            "blind_episodes": m["blind_episodes"], "longest_blind_s": m["longest_blind_s"],
            "control_outage_s": m["ctrl_out_s"], "video_outage_s": m["vid_out_s"],
            "ctrl_ontime_pct": 100 * c["ctrl_ontime"] / max(c["ctrl_sent"], 1),
            "ctrl_lat_ms_p50": float(np.percentile(lat, 50)), "ctrl_lat_ms_p95": float(np.percentile(lat, 95)),
            "ctrl_lat_ms_p99": float(np.percentile(lat, 99)),
            "ctrl_deadline_violation_pct": 100 * (c["ctrl_late"] + c["ctrl_lost"]) / max(c["ctrl_sent"], 1),
            "frames_decoded_pct": 100 * c["frames_ok"] / max(c["frames"], 1),
            "video_quality_mean": m["vq_int"] / T,
            "radio_energy_wh": {n: r.energy_j / 3600.0 for n, r in self.radios.items()},
            "comm_energy_wh": self.comm_energy_j / 3600.0,
            "comm_power_mean_w": self.comm_energy_j / T,
            "radio_events": {n: len(r.events) for n, r in self.radios.items()},
            "video_frames_by_link": self.link_frames,
            "codec_mode": self.codec_mode or self.codec_request,
            "event_awareness_pct": 100 * m["ev_cov_s"] / m["ev_s"] if m["ev_s"] > 0 else None,
        }


# =============================================================================
# 8. Simulation orchestrator
# =============================================================================
@dataclass
class SimConfig:
    seed: int = 1
    physics_dt: float = 0.005
    radio_dt: float = 0.01
    log_hz: float = 10.0
    start_pos: Tuple[float, float, float] = (5.0, 0.0, 0.0)
    cruise_speed_mps: float = 16.0
    drone: DroneParams = field(default_factory=DroneParams)
    controller: ControllerParams = field(default_factory=ControllerParams)
    battery: BatteryParams = field(default_factory=BatteryParams)
    env: EnvironmentParams = field(default_factory=EnvironmentParams)
    traffic: TrafficParams = field(default_factory=TrafficParams)
    routing: RoutingPolicy = field(default_factory=RoutingPolicy)
    radios: List[RadioParams] = field(default_factory=radio_catalog)   # legacy five enabled as before, new ones disabled
    initial_modes: Dict[str, str] = field(default_factory=lambda: {"rf24": "on", "cell": "on", "leo": "standby", "wifi": "off", "rf900": "off"})
    radio_aliases: Dict[str, str] = field(default_factory=lambda: {"wifi_5g": "wifi", "lte": "cell", "sub1ghz": "rf900", "rf_24": "rf24"})
    waypoints: List[Waypoint] = field(default_factory=lambda: [
        Waypoint((5, 0, 60), hold_s=1.0), Waypoint((3000, 0, 70)), Waypoint((3000, 800, 70)), Waypoint((0, 0, 50))])
    failsafe_rth_after_s: float = 0.0    # 0 disables link-loss RTH
    rtl_altitude_m: float = 80.0

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SimConfig":
        return dc_from_dict(SimConfig, d)


class DroneMultiLinkSim:
    def __init__(self, cfg: Optional[SimConfig] = None):
        self.cfg = cfg or SimConfig()
        self.reset()

    def reset(self, seed: Optional[int] = None):
        c = self.cfg
        if seed is not None:
            c.seed = seed
        self.rng = np.random.default_rng(c.seed)
        self.env = Environment(c.env, c.seed)
        self.quad = Quadrotor(c.drone, c.controller, c.start_pos)
        self.batt = Battery(c.battery, c.env.ambient_c)
        radios = []
        by_name = {rp.name: rp for rp in c.radios}
        for rp in c.radios:                      # a bonded link needs its member modems
            if rp.kind == "bond" and rp.enabled:
                for mn in rp.members:
                    if mn in by_name and not by_name[mn].enabled:
                        by_name[mn].enabled = True
                        c.initial_modes.setdefault(mn, "standby")
        for rp in c.radios:
            if not rp.enabled:
                continue
            rp.initial_mode = c.initial_modes.get(rp.name, rp.initial_mode)
            radios.append(RADIO_CLASSES[rp.kind](rp, np.random.default_rng(self.rng.integers(1 << 31))))
        self.suite = MultiLinkRadioSuite(radios, c.traffic, c.routing, np.random.default_rng(self.rng.integers(1 << 31)))
        self.suite.aliases = dict(c.radio_aliases)
        for r in self.suite.radios.values():
            if isinstance(r, BondedRadio):
                r.bind(self.suite.radios)
        self.suite.warm_start(0.0)
        self.follower = WaypointFollower(c.waypoints, c.cruise_speed_mps)
        self.t, self._radio_acc, self._log_acc = 0.0, 0.0, 0.0
        self.log: List[dict] = []
        self.p_radio_w, self.failsafe_active, self.end_reason = self.suite.total_power_w(), False, "running"
        self.hooks: List[Callable] = []

    @property
    def drone(self) -> DroneState:
        return self.quad.state

    def trajectory_forecast(self, horizon_s=10.0, step_s=0.5) -> np.ndarray:
        return self.follower.forecast(horizon_s, step_s)

    def _failsafe(self):
        c = self.cfg
        if c.failsafe_rth_after_s > 0 and not self.failsafe_active and self.suite.control_lost_duration() > c.failsafe_rth_after_s:
            p = self.drone.pos
            self.follower = WaypointFollower([Waypoint((p[0], p[1], c.rtl_altitude_m)), Waypoint((c.start_pos[0], c.start_pos[1], c.rtl_altitude_m)),
                                              Waypoint((c.start_pos[0], c.start_pos[1], 0.0))], c.cruise_speed_mps)
            self.failsafe_active = True
            self.suite.radios[next(iter(self.suite.radios))].events.append((self.t, "FAILSAFE RTH"))

    def step(self):
        c, s = self.cfg, self.drone
        dt = c.physics_dt
        sp, sv, yaw = self.follower.step(dt, s.pos)
        cmd = self.quad.control(dt, sp, sv, yaw)
        self.quad.thrust_scale = self.batt.thrust_scale()
        self.quad.step(dt, cmd, self.env.wind_now)
        self._radio_acc += dt
        if self._radio_acc >= c.radio_dt - 1e-9:
            rdt, self._radio_acc = self._radio_acc, 0.0
            self.env.step(self.t, rdt)
            self.suite.step(self.t, rdt, s, self.env, self.batt.v, self.env.temperature_c(s.pos[2]))
            self.p_radio_w = self.suite.total_power_w()
            self.quad.mass = c.drone.mass_kg + sum(r.extra_mass_kg for r in self.suite.radios.values())
            self._failsafe()
            for h in self.hooks:
                h(self)
        P = self.quad.motor_power_w + c.drone.avionics_w + c.drone.payload_w + self.p_radio_w
        self.batt.step(dt, P, self.env.temperature_c(s.pos[2]), s.speed)
        self.t += dt
        self._log_acc += dt
        if self._log_acc >= 1.0 / c.log_hz - 1e-9:
            self._log_acc = 0.0
            self._record(P)
        if self.batt.depleted:
            self.end_reason = "battery depleted"
        elif self.quad.crashed:
            self.end_reason = "crashed"
        return self.end_reason == "running"

    def _record(self, P):
        s, b, su = self.drone, self.batt, self.suite
        r, p_, y = s.euler
        row = dict(t=round(self.t, 3), x=s.pos[0], y=s.pos[1], z=s.pos[2], speed=s.speed, roll=r, pitch=p_, yaw=y,
                   batt_v=b.v, batt_i=b.i, soc=b.soc, batt_temp=b.temp, p_total_w=P, p_motor_w=self.quad.motor_power_w,
                   p_radio_w=self.p_radio_w, ctrl_ok=int(su.ctrl_ok), video_ok=int(su.video_ok),
                   blind=int(not su.ctrl_ok and not su.video_ok), video_bps=su.video_bps, video_q=su.video_quality,
                   rain_mm_h=self.env.rain_rate(self.t))
        for n, st in su.status.items():
            for k in ("state", "rssi_dbm", "snr_ul_db", "snr_dl_db", "cap_ul_bps", "cap_dl_bps", "per_dl", "ctrl_loss_prob",
                      "lat_ctrl_ms", "power_w", "tx_power_dbm", "temp_c", "outage"):
                row[f"{n}_{k}"] = getattr(st, k)
        self.log.append(row)

    def run(self, duration_s: float, progress=False) -> dict:
        n = int(duration_s / self.cfg.physics_dt)
        for i in range(n):
            if not self.step():
                break
            if progress and i % int(20 / self.cfg.physics_dt) == 0:
                print(f"  t={self.t:6.1f}s  pos=({self.drone.pos[0]:.0f},{self.drone.pos[1]:.0f},{self.drone.pos[2]:.0f}) "
                      f"soc={self.batt.soc*100:.1f}% blind={self.suite.m['blind_s']:.2f}s", file=sys.stderr)
        return self.summary()

    def summary(self) -> dict:
        s = self.suite.summary()
        s.update(end_reason=self.end_reason, soc_end=self.batt.soc, batt_wh_used=self.batt.energy_wh,
                 mean_radio_w=float(np.mean([r["p_radio_w"] for r in self.log])) if self.log else self.p_radio_w)
        return s

    def save_csv(self, path):
        if not self.log:
            return
        keys = list(self.log[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.log)


# =============================================================================
# 9. Baseline policies (examples; Part 3 replaces these with the predictor)
# =============================================================================
class ReactivePolicy:
    """Stay on primary; hard-switch only after control has been lost for detect_s."""

    def __init__(self, order=("rf24", "cell", "leo", "wifi"), detect_s=0.5, standby=("cell", "leo")):
        self.order, self.detect_s, self.standby, self.cur, self.last = list(order), detect_s, standby, 0, -1e9

    def __call__(self, sim: DroneMultiLinkSim):
        su, t = sim.suite, sim.t
        if t < 0.02:
            for n in self.standby:
                if n in su.radios and su.radios[n].state == S_OFF:
                    su.radios[n].request("standby")
            self._route(su)
        if su.control_lost_duration() > self.detect_s and t - self.last > 2.0:
            self.last = t
            self.cur = (self.cur + 1) % len(self.order)
            su.radios[self.order[self.cur]].request("on")
            su.switch_count += 1
            self._route(su)

    def _route(self, su):
        n = self.order[self.cur]
        su.set_routing(control_links=[n], control_mode="primary", video_links=[n], video_mode="split")


class BondingPolicy:
    """Everything on, control duplicated on all links, video split across all."""

    def __init__(self, order=("rf24", "cell", "leo", "wifi")):
        self.order, self.done = order, False

    def __call__(self, sim: DroneMultiLinkSim):
        if not self.done:
            for n in self.order:
                if n in sim.suite.radios:
                    sim.suite.radios[n].request("on")
            names = [n for n in self.order if n in sim.suite.radios]
            sim.suite.set_routing(control_links=names, control_mode="duplicate", video_links=names, video_mode="split")
            self.done = True


def _print_summary(tag, s):
    print(f"\n=== {tag} ===")
    print(f"  end: {s['end_reason']}  soc_end={s['soc_end']*100:.1f}%  battery used={s['batt_wh_used']:.1f} Wh  mean radio power={s['mean_radio_w']:.1f} W")
    print(f"  BLIND time: {s['blind_time_s']:.2f} s ({s['blind_pct']:.2f}%)  episodes={s['blind_episodes']}  longest={s['longest_blind_s']:.2f} s")
    print(f"  control outage {s['control_outage_s']:.2f} s | video outage {s['video_outage_s']:.2f} s")
    print(f"  control on-time {s['ctrl_ontime_pct']:.2f}%  lat p50/p95/p99 = {s['ctrl_lat_ms_p50']:.0f}/{s['ctrl_lat_ms_p95']:.0f}/{s['ctrl_lat_ms_p99']:.0f} ms")
    print(f"  frames decoded {s['frames_decoded_pct']:.1f}%  mean video quality {s['video_quality_mean']:.3f}")
    print("  radio energy [Wh]: " + ", ".join(f"{k}={v:.2f}" for k, v in s["radio_energy_wh"].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--policy", choices=["none", "reactive", "bonding"], default="reactive")
    ap.add_argument("--config", help="JSON file with (partial) SimConfig overrides")
    ap.add_argument("--dump-config", help="write the full default config JSON and exit")
    ap.add_argument("--csv", help="write time-series log")
    ap.add_argument("--compare", action="store_true", help="run reactive and bonding with the same seed")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--catalog", action="store_true", help="list every selectable radio technology")
    ap.add_argument("--range-sweep", nargs="?", const="all", help="link-budget table (optionally comma-separated radio names)")
    ap.add_argument("--enable", help="comma-separated radios to enable (standby at start) or 'all'")
    ap.add_argument("--codec", choices=["standard", "ladder", "neural", "event", "key_object", "auto"], default=None)
    a = ap.parse_args(argv)
    cfg = SimConfig()
    if a.dump_config:
        json.dump(cfg.to_dict(), open(a.dump_config, "w"), indent=1)
        print("wrote", a.dump_config)
        return
    if a.catalog:
        return print_catalog()
    if a.range_sweep:
        return range_sweep(None if a.range_sweep == "all" else a.range_sweep.split(","))
    if a.selftest:
        return selftest()
    if a.config:
        apply_overrides(cfg, json.load(open(a.config)))
    cfg.seed = a.seed
    if a.enable:
        names = [r.name for r in cfg.radios] if a.enable == "all" else a.enable.split(",")
        for r in cfg.radios:
            if r.name in names and not r.enabled:
                r.enabled = True
                cfg.initial_modes.setdefault(r.name, "standby")
    if a.codec:
        cfg.traffic.codec.mode = a.codec
    pols = ["reactive", "bonding"] if a.compare else [a.policy]
    for pol in pols:
        sim = DroneMultiLinkSim(SimConfig.from_dict(cfg.to_dict()))
        if pol == "reactive":
            sim.hooks.append(ReactivePolicy())
        elif pol == "bonding":
            sim.hooks.append(BondingPolicy())
        t0 = _time.time()
        s = sim.run(a.duration, progress=True)
        _print_summary(f"{pol} ({_time.time()-t0:.1f}s wall)", s)
        if a.csv:
            sim.save_csv(a.csv if not a.compare else a.csv.replace(".csv", f"_{pol}.csv"))


def print_catalog():
    print(f"{'name':15s} {'family':24s} {'kind':8s} {'f[MHz]':>9s} {'bw[MHz]':>8s} {'Ptx[dBm]':>8s} {'lat ul/dl[ms]':>14s} {'P idle/standby[W]':>18s} en")
    for r in radio_catalog():
        print(f"{r.name:15s} {r.family or r.kind:24s} {r.kind:8s} {r.freq_hz/1e6:9.1f} {r.bw_hz/1e6:8.2f} {r.tx_power_dbm:8.1f} "
              f"{r.lat_ul_ms+r.backhaul_ms:6.1f}/{r.lat_dl_ms+r.backhaul_ms:<6.1f} {r.p_idle_w:8.1f}/{r.p_standby_w:<8.1f} {'Y' if r.enabled else '-'}")


def range_sweep(names=None, distances=(250, 1000, 3000, 8000), alt=60.0, seed=1, seconds=3.0):
    """Static link-budget table for every radio technology on a clean (no hill/building/jammer/rain) environment."""
    import copy
    base = SimConfig()
    base.env.jammers, base.env.rain, base.env.buildings, base.env.hills = [], [], [], []
    print(f"{'radio':15s} {'dist[m]':>8s} {'state':>10s} {'SNRul':>6s} {'SNRdl':>6s} {'UL Mb/s':>8s} {'DL Mb/s':>8s} {'ctrl lat ms':>11s} {'extra':s}")
    for rp0 in base.radios:
        if rp0.kind == "bond" or (names and rp0.name not in names):
            continue
        for d in distances:
            rp = copy.deepcopy(rp0)
            rp.enabled, rp.initial_mode = True, "on"
            env = Environment(copy.deepcopy(base.env), seed)
            r = RADIO_CLASSES[rp.kind](rp, np.random.default_rng(seed))
            r.warm_start()
            ds = DroneState(pos=np.array([float(d), 0.0, alt]))
            acc = []
            n = int(seconds / 0.01)
            for i in range(n):
                st = r.step(i * 0.01, 0.01, ds, env, 22.2, 25.0)
                if i >= n * 2 // 3:
                    acc.append((st.snr_ul_db, st.snr_dl_db, st.cap_ul_bps, st.cap_dl_bps, st.lat_ctrl_ms))
            a = np.mean(np.array(acc), 0)
            ex = ""
            if rp.analog_video:
                ex = f"analog_q={st.extra['analog_q']:.2f}"
            elif rp.kind in ("sat", "mesh", "fiber", "cellular"):
                ex = str({k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.extra.items() if k in ("elev_deg", "range_km", "route", "serving", "remaining_m")})
            print(f"{rp.name:15s} {d:8d} {st.state:>10s} {a[0]:6.1f} {a[1]:6.1f} {a[2]/1e6:8.2f} {a[3]/1e6:8.2f} {a[4]:11.1f} {ex}")
            if rp.kind in ("sat", "cellular", "mesh", "fiber"):
                break                                    # distance-independent: one row


def selftest():
    cfg = SimConfig()
    cfg.env.jammers, cfg.env.rain, cfg.env.wind_mean_mps = [], [], (0.0, 0.0, 0.0)
    cfg.waypoints = [Waypoint((5, 0, 30), hold_s=100)]
    sim = DroneMultiLinkSim(cfg)
    sim.run(25)
    z = sim.drone.pos[2]
    print(f"hover altitude {z:.2f} m (target 30), speed {sim.drone.speed:.3f} m/s, motor P {sim.quad.motor_power_w:.0f} W")
    assert abs(z - 30) < 0.5, "hover failed"
    st = sim.suite.status
    for n, s in st.items():
        print(f"{n:6s} state={s.state:10s} rssi={s.rssi_dbm:7.1f} snr_ul={s.snr_ul_db:6.1f} snr_dl={s.snr_dl_db:6.1f} "
              f"cap_ul={s.cap_ul_bps/1e6:7.2f} Mb/s lat_ctrl={s.lat_ctrl_ms:6.1f} ms P={s.power_w:5.1f} W")
    assert st["rf24"].state == S_CONN and st["rf24"].snr_ul_db > 15
    print("selftest OK")


if __name__ == "__main__":
    main()
