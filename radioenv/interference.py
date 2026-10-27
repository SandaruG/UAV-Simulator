"""Dynamic interference: ambient Wi-Fi (bursty, diurnal), other drones' video/C2 transmitters
flying through the area, and scripted jammers (noise / narrowband / swept / pulsed), plus the
ITU-R P.372 man-made noise floor.

Each source is a deterministic function of time, so `power_dbm(rx, t, f, bw)` can be queried at
any (rx position, t). Interference paths go through the same terrain/building engine as the
wanted signal (fast, coarser profile).
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

import numpy as np

from .config import InterferenceConfig, PropagationConfig
from .propagation import Surface, terrestrial_path
from .terrain import OPEN, SUBURBAN, URBAN, VEGETATION, WATER
from .util import (as_points, db, fspl_db, hash_uniform, local_hour, thermal_noise_dbm, undb)

KIND_WIFI, KIND_UAV, KIND_JAM = 0, 1, 2
J_NOISE, J_NARROW, J_SWEPT, J_PULSED = 0, 1, 2, 3
_JMAP = {"noise": J_NOISE, "narrowband": J_NARROW, "swept": J_SWEPT, "pulsed": J_PULSED}

# ITU-R P.372 man-made noise: Fam = c - d*log10(f_MHz)  (dB above kT0B)
_FAM = {"city": (76.8, 27.7), "residential": (72.5, 27.7), "rural": (67.2, 27.7), "quiet_rural": (53.6, 28.6)}


def overlap_factor(fv, bv, fi, bi):
    """Fraction of an interferer's power that falls in the victim band, including simple
    adjacent-channel leakage (-35 dB within one bandwidth, -50 dB within three)."""
    fv, bv, fi, bi = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in (fv, bv, fi, bi)])
    lo = np.maximum(fv - bv / 2, fi - bi / 2)
    hi = np.minimum(fv + bv / 2, fi + bi / 2)
    ov = np.maximum(hi - lo, 0.0) / np.maximum(bi, 1.0)
    gap = np.abs(fv - fi) - (bv + bi) / 2
    ref = np.maximum(np.maximum(bv, bi), 1.0)
    leak = np.where(gap < ref, 10 ** (-3.5), np.where(gap < 3 * ref, 1e-5, 0.0))
    return np.where(ov > 0, ov, np.where(gap > 0, leak, 0.0))


class InterferenceField:
    def __init__(self, cfg: InterferenceConfig, surface: Surface, pc: PropagationConfig, seed: int,
                 start_utc: _dt.datetime, lon0: float, area_half_xy=(2000.0, 2000.0)):
        self.cfg, self.S, self.pc, self.seed = cfg, surface, pc, int(seed)
        self.start_utc, self.lon0 = start_utc, lon0
        rng = np.random.default_rng(self.seed + 4242)
        rows = []   # dict per source

        # ---- Wi-Fi access points --------------------------------------------------------
        w = cfg.wifi
        if w.enabled:
            ch24, ch58 = [2412e6, 2437e6, 2462e6], [5745e6, 5765e6, 5785e6, 5805e6, 5825e6]
            for i in range(w.n_sources):
                is58 = rng.random() < w.frac_58
                x = float(np.clip(rng.normal(0, w.region_radius_m / 2), -area_half_xy[0], area_half_xy[0]))
                y = float(np.clip(rng.normal(0, w.region_radius_m / 2), -area_half_xy[1], area_half_xy[1]))
                top = float(surface.top_at(x, y))
                bh = float(surface.bldg_at(x, y))
                hagl = rng.uniform(*w.height_agl_m)
                z = float(surface.terrain_at(x, y)) + (rng.uniform(2.0, max(2.5, 0.8 * bh)) if bh > 0 else min(hagl, 6.0))
                rows.append(dict(kind=KIND_WIFI, f0=(rng.choice(ch58) if is58 else rng.choice(ch24)), bw=w.bw_mhz * 1e6,
                                 eirp=w.eirp_dbm_58 if is58 else w.eirp_dbm_24, x0=x, y0=y, z0=z,
                                 duty=float(np.clip(w.duty_mean * np.exp(0.7 * rng.normal()), 0.01, 0.9)), name=f"wifi{i}"))
        # ---- other UAVs -----------------------------------------------------------------
        u = cfg.uav
        if u.enabled:
            for i in range(u.n_drones):
                cx, cy = rng.uniform(-1500, 1500, 2)
                R = rng.uniform(*u.radius_m)
                v = rng.uniform(*u.speed_ms)
                phi = rng.uniform(0, 2 * np.pi)
                zagl = rng.uniform(*u.alt_agl_m)
                ph = rng.uniform(0, u.period_s)
                common = dict(kind=KIND_UAV, cx=cx, cy=cy, R=R, omega=v / R * rng.choice([-1, 1]), phi=phi, zagl=zagl,
                              phase=ph, period=u.period_s, afrac=u.active_fraction)
                rows.append(dict(common, f0=rng.choice(u.freqs_mhz) * 1e6, bw=u.bw_mhz * 1e6, eirp=u.eirp_dbm, name=f"uav{i}_video"))
                rows.append(dict(common, f0=rng.choice(u.extra_c2_freqs_mhz) * 1e6, bw=u.c2_bw_mhz * 1e6, eirp=u.c2_eirp_dbm, name=f"uav{i}_c2"))
        # ---- jammers --------------------------------------------------------------------
        self.jam_windows = {}
        for j in cfg.jammers:
            top = float(surface.top_at(j.x, j.y)) + j.z_agl_m
            kind = _JMAP.get(j.kind, J_NOISE)
            bw = j.bw_mhz * 1e6
            rows.append(dict(kind=KIND_JAM, f0=j.freq_mhz * 1e6, bw=bw, eirp=j.eirp_dbm, x0=j.x, y0=j.y, z0=top,
                             vx=j.vx, vy=j.vy, jkind=kind, sweep=j.sweep_range_mhz * 1e6, sweep_T=j.sweep_period_s,
                             pulse_T=j.pulse_period_s, jduty=j.duty, name=j.id))
            self.jam_windows[len(rows) - 1] = np.array(j.windows, dtype=float).reshape(-1, 2)
        self.rows = rows
        n = len(rows)
        g = lambda k, d=0.0: np.array([r.get(k, d) for r in rows], dtype=float)
        self.n = n
        self.kind = np.array([r["kind"] for r in rows], dtype=int)
        self.f0, self.bw, self.eirp = g("f0"), g("bw", 1e6), g("eirp")
        self.x0, self.y0, self.z0 = g("x0"), g("y0"), g("z0")
        self.duty = g("duty", 1.0)
        self.cx, self.cy, self.R, self.omega, self.phi, self.zagl = g("cx"), g("cy"), g("R", 1.0), g("omega"), g("phi"), g("zagl")
        self.phase, self.period, self.afrac = g("phase"), g("period", 1.0), g("afrac", 1.0)
        self.vx, self.vy = g("vx"), g("vy")
        self.jkind = np.array([r.get("jkind", 0) for r in rows], dtype=int)
        self.sweep, self.sweep_T, self.pulse_T, self.jduty = g("sweep"), g("sweep_T", 1.0), g("pulse_T", 1.0), g("jduty", 1.0)
        self.names = [r["name"] for r in rows]
        self.ids = np.arange(n)
        # swept jammers faster than the typical sample interval are treated in "average" mode
        self.swept_avg = (self.kind == KIND_JAM) & (self.jkind == J_SWEPT) & (self.sweep_T < 0.01)
        # band extent over time (for cheap pre-culling)
        self.band_lo = self.f0 - np.where(self.jkind == J_SWEPT, self.sweep / 2, 0) - self.bw / 2
        self.band_hi = self.f0 + np.where(self.jkind == J_SWEPT, self.sweep / 2, 0) + self.bw / 2

    # ------------------------------------------------------------------ source state
    def _diurnal(self, t):
        h = local_hour(self.start_utc, t, self.lon0)
        return 0.7 + 0.3 * np.cos(2 * np.pi * (h - 21.0) / 24.0)

    def positions(self, t, idx):
        t = np.asarray(t, dtype=float)[:, None]
        i = idx[None, :]
        x = np.where(self.kind[i] == KIND_UAV, self.cx[i] + self.R[i] * np.cos(self.omega[i] * t + self.phi[i]), self.x0[i] + self.vx[i] * t)
        y = np.where(self.kind[i] == KIND_UAV, self.cy[i] + self.R[i] * np.sin(self.omega[i] * t + self.phi[i]), self.y0[i] + self.vy[i] * t)
        zg = self.S.terrain_at(x, y) + self.zagl[i]
        z = np.where(self.kind[i] == KIND_UAV, zg, self.z0[i] + 0 * x)
        return np.stack([x, y, z], -1)

    def freqs(self, t, idx):
        t = np.asarray(t, dtype=float)[:, None]
        i = idx[None, :]
        f = self.f0[i] + 0 * t
        bw = self.bw[i] + 0 * t
        swept = (self.kind[i] == KIND_JAM) & (self.jkind[i] == J_SWEPT)
        inst = swept & ~self.swept_avg[i]
        frac = (t / np.maximum(self.sweep_T[i], 1e-6)) % 1.0
        f = np.where(inst, self.f0[i] - self.sweep[i] / 2 + frac * self.sweep[i], f)
        bw = np.where(swept & self.swept_avg[i], self.sweep[i], bw)
        return f, bw

    def activity(self, t, idx):
        t = np.asarray(t, dtype=float)
        tt = t[:, None]
        i = idx[None, :]
        act = np.ones((len(t), len(idx)))
        # wifi: bursty on/off per time bin
        wm = self.kind[i] == KIND_WIFI
        if wm.any():
            b = np.floor(tt / self.cfg.wifi.busy_bin_s)
            p = self.duty[i] * (self._diurnal(t)[:, None] if self.cfg.wifi.diurnal else 1.0)
            u = hash_uniform(self.seed, 401, self.ids[i], b)
            act = np.where(wm, (u < p).astype(float), act)
        um = self.kind[i] == KIND_UAV
        if um.any():
            ph = (tt + self.phase[i]) % self.period[i]
            act = np.where(um, (ph < self.afrac[i] * self.period[i]).astype(float), act)
        for k in np.nonzero(self.kind[idx] == KIND_JAM)[0]:
            src = idx[k]
            win = self.jam_windows[src]
            on = np.zeros(len(t), dtype=bool)
            for (a, b) in win:
                on |= (t >= a) & (t < b)
            if self.jkind[src] == J_PULSED:
                on &= ((t % self.pulse_T[src]) / self.pulse_T[src]) < self.jduty[src]
            act[:, k] = on.astype(float)
        return act

    # ------------------------------------------------------------------ main query
    def power_dbm(self, rx, t, f_hz, bw_hz, rx_gain_dbi=0.0, return_top=False):
        """Aggregate in-band interference power (dBm) at receiver position(s) rx at time(s) t."""
        rx = as_points(rx)
        N = len(rx)
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,))
        floor = np.full(N, -200.0)
        if self.n == 0:
            return (floor, None) if return_top else floor
        lo, hi = f_hz - bw_hz / 2, f_hz + bw_hz / 2
        guard = 3.5 * np.maximum(self.bw, bw_hz)
        cand = np.nonzero((self.band_hi + guard > lo) & (self.band_lo - guard < hi))[0]
        if len(cand) == 0:
            return (floor, None) if return_top else floor
        total = np.zeros(N)
        top_val = np.full(N, -300.0)
        top_idx = np.full(N, -1)
        # chunk over receivers to bound memory
        step = max(1, int(30000 // max(len(cand), 1)))
        for a in range(0, N, step):
            sl = slice(a, min(N, a + step))
            tt, rr = t[sl], rx[sl]
            pos = self.positions(tt, cand)
            fi, bi = self.freqs(tt, cand)
            act = self.activity(tt, cand)
            ov = overlap_factor(f_hz, bw_hz, fi, bi) * act
            d = np.linalg.norm(pos - rr[:, None, :], axis=-1)
            keep = (ov > 0) & ((d < self.cfg.max_range_m) | (self.kind[cand][None, :] == KIND_JAM))
            ii, jj = np.nonzero(keep)
            if len(ii) == 0:
                continue
            path = terrestrial_path(self.S, pos[ii, jj], rr[ii], f_hz, self.pc, fast=True, two_ray=False)
            pl = fspl_db(path["d3d"], f_hz) + path["loss_diff"] + path["loss_veg"]
            pr = self.eirp[cand][jj] + rx_gain_dbi - pl + db(ov[ii, jj])
            np.add.at(total, ii + a, undb(pr))
            if return_top:
                for q in range(len(ii)):
                    g = ii[q] + a
                    if pr[q] > top_val[g]:
                        top_val[g], top_idx[g] = pr[q], cand[jj[q]]
        out = np.where(total > 0, db(total), floor)
        if return_top:
            return out, top_idx
        return out

    def noise_dbm(self, rx, f_hz, bw_hz, nf_db, clutter_at_rx=None, agl=None):
        """Thermal + receiver NF combined with the P.372 man-made noise floor."""
        rx = as_points(rx)
        env = self.cfg.noise_environment
        if env == "auto":
            cl = self.S.clutter_at(rx[:, 0], rx[:, 1]) if clutter_at_rx is None else np.asarray(clutter_at_rx)
            c = np.where(cl == URBAN, 76.8, np.where(cl == SUBURBAN, 72.5, np.where(cl == WATER, 53.6, 67.2)))
            d = np.where(cl == WATER, 28.6, 27.7)
        else:
            c, d = _FAM[env]
        fam = c - d * np.log10(f_hz / 1e6)
        if agl is not None:
            # man-made noise sources sit near the ground: up to 6 dB less for a receiver >130 m AGL (assumption)
            fam = fam - 6.0 * np.clip((np.asarray(agl, dtype=float) - 30.0) / 100.0, 0.0, 1.0)
        f_lin = undb(fam) + undb(nf_db) - 1.0
        return thermal_noise_dbm(bw_hz, 0.0) + db(f_lin)

    def table(self):
        return [dict(name=self.names[i], kind=["wifi", "uav", "jammer"][self.kind[i]], f_mhz=self.f0[i] / 1e6,
                     bw_mhz=self.bw[i] / 1e6, eirp_dbm=self.eirp[i]) for i in range(self.n)]
