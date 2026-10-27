"""LEO constellation geometry: Walker-delta shells (analytic Keplerian + J2 nodal drift) or real
TLEs (if `sgp4` is installed), look angles, serving-satellite selection with 15 s slot logic and
horizon/obstruction screening, Doppler.
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

import numpy as np

from .config import SatelliteConfig
from .geo import LocalFrame
from .util import OMEGA_EARTH, as_points, gmst_deg, julian_date

def surface_range(surface):
    return getattr(surface, 'obstruction_range_m', 1500.0)


MU = 3.986004418e14
RE = 6378137.0
J2 = 1.08263e-3


class Constellation:
    def __init__(self, cfg: SatelliteConfig, epoch: _dt.datetime, frame: LocalFrame):
        self.cfg, self.epoch, self.frame = cfg, epoch, frame
        self.gmst0 = np.radians(gmst_deg(julian_date(epoch)))
        shells = [cfg.shell] + list(cfg.extra_shells)
        a, inc, raan, u0 = [], [], [], []
        for sh in shells:
            P, S, F = sh.planes, sh.sats_per_plane, sh.phasing
            ai = RE + sh.alt_km * 1e3
            for p in range(P):
                for s in range(S):
                    a.append(ai)
                    inc.append(np.radians(sh.inc_deg))
                    raan.append(2 * np.pi * p / P)
                    u0.append(2 * np.pi * s / S + 2 * np.pi * F * p / (P * S))
        self.a, self.inc = np.array(a), np.array(inc)
        self.raan0, self.u0 = np.array(raan), np.array(u0)
        self.n = np.sqrt(MU / self.a**3)
        self.raan_dot = (-1.5 * J2 * (RE / self.a) ** 2 * self.n * np.cos(self.inc)) if cfg.j2 else np.zeros_like(self.a)
        self.cosi, self.sini = np.cos(self.inc), np.sin(self.inc)
        self.count = len(self.a)
        self._tle = None
        if cfg.tle_path:
            self._load_tle(cfg.tle_path)

    # ---------------------------------------------------------------- optional real TLEs
    def _load_tle(self, path):  # pragma: no cover - needs sgp4
        try:
            from sgp4.api import Satrec, jday
        except ImportError as e:
            raise ImportError("TLE propagation needs `pip install sgp4`") from e
        lines = [l.strip() for l in open(path) if l.strip()]
        sats = []
        for i in range(len(lines) - 2):
            if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
                sats.append(Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
        self._tle = sats
        self.count = len(sats)

    def _ecef_tle(self, t, idx):  # pragma: no cover
        from sgp4.api import jday
        out = np.zeros((len(t), self.count if idx is None else 1, 3))
        for i, ti in enumerate(t):
            dt = self.epoch + _dt.timedelta(seconds=float(ti))
            jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
            th = np.radians(gmst_deg(jd + fr))
            sel = range(self.count) if idx is None else [int(idx[i])]
            for j, k in enumerate(sel):
                e, r, v = self._tle[k].sgp4(jd, fr)
                x, y, z = np.array(r) * 1e3
                out[i, j] = [x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th), z]
        return out if idx is None else out[:, 0, :]

    # ---------------------------------------------------------------- propagation
    def ecef(self, t, idx=None):
        """ECEF positions: (n_t, S, 3) for all satellites, or (n_t, 3) for one satellite per time."""
        t = np.asarray(t, dtype=float).reshape(-1)
        if self._tle is not None:
            return self._ecef_tle(t, idx)
        if idx is None:
            tt, sel = t[:, None], slice(None)
        else:
            tt, sel = t, np.asarray(idx)
        u = self.u0[sel] + self.n[sel] * tt
        om = self.raan0[sel] + self.raan_dot[sel] * tt
        r = self.a[sel]
        xo, yo = r * np.cos(u), r * np.sin(u)
        X = xo * np.cos(om) - yo * self.cosi[sel] * np.sin(om)
        Y = xo * np.sin(om) + yo * self.cosi[sel] * np.cos(om)
        Z = yo * self.sini[sel]
        th = self.gmst0 + OMEGA_EARTH * tt
        c, s = np.cos(th), np.sin(th)
        x, y = X * c + Y * s, -X * s + Y * c
        return np.stack([x, y, Z], -1)

    def look(self, obs_ecef, sat_ecef):
        """Elevation/azimuth (deg) and range (m). obs (n,3); sat (n,S,3) or (n,3)."""
        obs = np.asarray(obs_ecef)
        rel = sat_ecef - (obs[:, None, :] if sat_ecef.ndim == 3 else obs)
        rng = np.linalg.norm(rel, axis=-1)
        E, Nn, U = rel @ self.frame.east, rel @ self.frame.north, rel @ self.frame.up
        el = np.degrees(np.arcsin(np.clip(U / np.maximum(rng, 1.0), -1, 1)))
        az = np.degrees(np.arctan2(E, Nn)) % 360.0
        return el, az, rng

    def count_visible(self, obs_ecef, t, min_elev, chunk=64):
        obs = np.asarray(obs_ecef)
        t = np.asarray(t, dtype=float)
        out = np.zeros(len(t), dtype=int)
        for a in range(0, len(t), chunk):
            sl = slice(a, a + chunk)
            el, _, _ = self.look(obs[sl], self.ecef(t[sl]))
            out[sl] = (el >= min_elev).sum(1)
        return out

    def select(self, obs_enu, t, surface, lam, chunk=64):
        """Serving satellite per query. At every `slot_s` boundary the terminal picks, among the
        `n_candidates` highest satellites (evaluated at mid-slot so it stays visible), the first one
        whose line of sight is not blocked by terrain/buildings. Returns idx (-1 = none), and the
        number of satellites above the mask at slot mid-time."""
        cfg = self.cfg
        obs_enu = as_points(obs_enu)
        N = len(obs_enu)
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,))
        obs = self.frame.enu_to_ecef(obs_enu)
        slot = np.floor(t / cfg.slot_s)
        t_mid = (slot + 0.5) * cfg.slot_s
        chosen = np.full(N, -1, dtype=int)
        n_vis = np.zeros(N, dtype=int)
        K = min(cfg.n_candidates, self.count)
        for a in range(0, N, chunk):
            sl = slice(a, min(N, a + chunk))
            el, az, rng = self.look(obs[sl], self.ecef(t_mid[sl]))
            n_vis[sl] = (el >= cfg.min_elev_deg).sum(1)
            top = np.argsort(-el, axis=1)[:, :K]
            rows = np.arange(el.shape[0])
            done = np.zeros(el.shape[0], dtype=bool)
            pick = np.full(el.shape[0], -1, dtype=int)
            obs_chunk = obs_enu[sl]
            for j in range(K):
                cand = top[:, j]
                ej, aj = el[rows, cand], az[rows, cand]
                todo = (~done) & (ej >= cfg.min_elev_deg)
                if not todo.any():
                    break
                ii = np.nonzero(todo)[0]
                _, _, blocked = surface.ray_obstruction(obs_chunk[ii], aj[ii], ej[ii], lam,
                                                        max_d=surface_range(surface))
                good = ii[~blocked]
                pick[good] = cand[good]
                done[good] = True
            chosen[sl] = pick
        return chosen, n_vis

    def doppler_hz(self, obs_enu, idx, t, f_hz, vel_enu=None, dt=0.5):
        """Doppler shift (Hz) of the link to satellite idx (per query) at time t."""
        obs_enu = as_points(obs_enu)
        N = len(obs_enu)
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,))
        ok = idx >= 0
        out = np.zeros(N)
        if not ok.any():
            return out
        i = np.nonzero(ok)[0]
        obs = self.frame.enu_to_ecef(obs_enu[i])
        p0 = self.ecef(t[i] - dt, idx[i])
        p1 = self.ecef(t[i] + dt, idx[i])
        v_sat = (p1 - p0) / (2 * dt)
        v_obs = np.zeros_like(obs) if vel_enu is None else self.frame.vec_enu_to_ecef(np.broadcast_to(np.asarray(vel_enu, dtype=float), (N, 3))[i])
        pm = 0.5 * (p0 + p1)
        u = pm - obs
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        rr = np.sum((v_sat - v_obs) * u, axis=1)
        out[i] = -rr / 299792458.0 * f_hz
        return out
