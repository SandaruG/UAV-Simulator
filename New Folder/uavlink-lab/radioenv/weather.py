"""Weather layers: advecting rain cells (convective + stratiform), cloud liquid water, fog,
gaseous absorption, rain attenuation (ITU-R P.838 style k/alpha), cloud/fog attenuation
(ITU-R P.840 Rayleigh double-Debye) and tropospheric scintillation.

All fields are analytic functions of (x, y, z, t); no state is advanced, so any query order is
valid and fully reproducible for a given seed.
"""
from __future__ import annotations

import math

import numpy as np

from .config import WeatherConfig
from .util import as_points, fractal_field, hash_normal, hash_uniform, smooth_noise

# ITU-R P.838-3 specific attenuation coefficients (approximate table, log-log interpolated).
_F = np.array([1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60], dtype=float)
_KH = np.array([0.0000259, 0.0000847, 0.0001071, 0.0007056, 0.004115, 0.01217, 0.02386, 0.04481,
                0.09164, 0.1571, 0.2403, 0.3374, 0.4431, 0.6600, 0.8606])
_AH = np.array([0.9691, 1.0664, 1.6009, 1.5900, 1.3905, 1.2571, 1.1825, 1.1233, 1.0568, 0.9991,
                0.9485, 0.9047, 0.8673, 0.8084, 0.7656])
_KV = np.array([0.0000308, 0.0000998, 0.0002461, 0.0004878, 0.003450, 0.01129, 0.02455, 0.05008,
                0.09611, 0.1533, 0.2291, 0.3224, 0.4274, 0.6472, 0.8515])
_AV = np.array([0.8592, 0.9490, 1.2476, 1.5728, 1.3797, 1.2156, 1.1216, 1.0440, 0.9847, 0.9491,
                0.9129, 0.8761, 0.8421, 0.7871, 0.7486])

_SCENARIOS = {  # (background mm/h, convective cell peak median mm/h)
    "clear": (0.0, 0.0),
    "light_rain": (0.6, 8.0),
    "moderate_rain": (2.5, 25.0),
    "heavy_rain": (6.0, 70.0),
    "tropical_storm": (12.0, 130.0),
}


def rain_k_alpha(f_ghz: float, pol: str = "V"):
    lf = math.log(min(max(f_ghz, 1.0), 60.0))
    lg = np.log(_F)
    kh, kv = math.exp(np.interp(lf, lg, np.log(_KH))), math.exp(np.interp(lf, lg, np.log(_KV)))
    ah, av = float(np.interp(lf, lg, _AH)), float(np.interp(lf, lg, _AV))
    p = pol.upper()
    if p == "H":
        return kh, ah
    if p == "V":
        return kv, av
    k = (kh + kv) / 2
    return k, (kh * ah + kv * av) / (2 * k)


def rain_specific_db_per_km(f_ghz, rain_mmh, pol="V"):
    k, a = rain_k_alpha(f_ghz, pol)
    return k * np.maximum(rain_mmh, 0.0) ** a


def cloud_k_l(f_ghz: float, temp_k: float = 273.15):
    """Specific cloud attenuation coefficient ((dB/km)/(g/m^3)), ITU-R P.840 double-Debye."""
    th = 300.0 / temp_k
    e0 = 77.66 + 103.3 * (th - 1)
    e1 = 0.0671 * e0
    e2 = 3.52
    fp = 20.20 - 146 * (th - 1) + 316 * (th - 1) ** 2
    fs = 39.8 * fp
    f = f_ghz
    epp = f * (e0 - e1) / (fp * (1 + (f / fp) ** 2)) + f * (e1 - e2) / (fs * (1 + (f / fs) ** 2))
    ep = (e0 - e1) / (1 + (f / fp) ** 2) + (e1 - e2) / (1 + (f / fs) ** 2) + e2
    eta = (2 + ep) / epp
    return 0.819 * f / (epp * (1 + eta**2))


def gas_specific_db_per_km(f_ghz: float, rho_wv_gm3: float):
    """Sea-level oxygen and water-vapour specific attenuation (simplified ITU-R P.676, f < 57 GHz)."""
    f = f_ghz
    go = (7.19e-3 + 6.09 / (f**2 + 0.227) + 4.81 / ((f - 57.0) ** 2 + 1.50)) * f**2 * 1e-3
    gw = (0.050 + 0.0021 * rho_wv_gm3 + 3.6 / ((f - 22.2) ** 2 + 8.5) + 10.6 / ((f - 183.3) ** 2 + 9.0)
          + 8.9 / ((f - 325.4) ** 2 + 26.3)) * f**2 * rho_wv_gm3 * 1e-4
    return go, gw


class Weather:
    H_O2 = 6000.0   # oxygen scale height (m)
    H_WV = 2000.0   # water-vapour scale height (m)

    def __init__(self, cfg: WeatherConfig, seed: int, pol: str = "V"):
        self.cfg = cfg
        self.seed = int(seed)
        self.pol = pol
        bg, peak = _SCENARIOS.get(cfg.scenario, (cfg.bg_rain_mmh, cfg.cell_peak_mmh))
        self.bg = bg * cfg.rain_scale
        self.peak = peak * cfg.rain_scale
        b = math.radians(cfg.wind_to_deg)
        self.wind = np.array([math.sin(b), math.cos(b)]) * cfg.wind_speed_ms
        rng = np.random.default_rng(self.seed + 991)
        self.cell_phase = rng.uniform(0, cfg.cell_period_s, cfg.n_cells)
        self.cell_period = cfg.cell_period_s * rng.uniform(0.8, 1.25, cfg.n_cells)
        # cloud / fog texture (tiled, advected)
        self.cloud_tile_m = 24000.0
        self.cloud_field = fractal_field(128, 128, 0.7, np.random.default_rng(self.seed + 17))
        self.fog_field = fractal_field(128, 128, 0.6, np.random.default_rng(self.seed + 18))
        T = cfg.temperature_c
        es = 6.1121 * math.exp(17.502 * T / (240.97 + T))          # hPa
        self.rho_wv = 216.7 * (cfg.humidity_pct / 100.0 * es) / (T + 273.15)   # g/m^3 at the surface

    # ---------------------------------------------------------------- rain field
    def rain_rate(self, x, y, t):
        """Rain rate (mm/h) at map position(s) and time(s) (broadcast)."""
        x, y, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(t, dtype=float))
        c = self.cfg
        if self.bg <= 0 and self.peak <= 0:
            return np.zeros_like(x)
        r = self.bg * np.maximum(0.0, 1.0 + 0.3 * smooth_noise(t, self.seed + 5, 900.0)) if self.bg > 0 else np.zeros_like(x)
        life = c.cell_life_s
        lo_r, hi_r = c.cell_radius_km
        total = r.copy()
        for k in range(c.n_cells):
            cyc = np.floor((t - self.cell_phase[k]) / self.cell_period[k])
            age = t - self.cell_phase[k] - cyc * self.cell_period[k]
            active = age < life
            env = np.sin(np.pi * np.clip(age / life, 0, 1)) ** 2
            u1 = hash_uniform(self.seed, 301, k, cyc)
            u2 = hash_uniform(self.seed, 302, k, cyc)
            u3 = hash_uniform(self.seed, 303, k, cyc)
            peak = self.peak * np.exp(0.5 * hash_normal(self.seed, 304, k, cyc))
            sigma = 1000.0 * (lo_r + (hi_r - lo_r) * u3)
            cx0 = (u1 - 0.5) * 8000.0          # centre at mid-life, within +-4 km of the map centre
            cy0 = (u2 - 0.5) * 8000.0
            dtm = age - life / 2
            cx = cx0 + self.wind[0] * dtm
            cy = cy0 + self.wind[1] * dtm
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            total = total + np.where(active, env * peak * np.exp(-0.5 * r2 / sigma**2), 0.0)
        return total

    def rain_atten_path(self, A, B, f_hz, t, n=12):
        """Rain attenuation (dB) along terrestrial path(s) A->B at time(s) t."""
        f_ghz = f_hz / 1e9
        if self.bg <= 0 and self.peak <= 0:
            return np.zeros(max(len(as_points(A)), len(as_points(B))))
        A, B = as_points(A), as_points(B)
        N = max(len(A), len(B))
        A, B = np.broadcast_to(A, (N, 3)), np.broadcast_to(B, (N, 3))
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,))
        s = (np.arange(n) + 0.5) / n
        px = A[:, 0, None] + s * (B[:, 0] - A[:, 0])[:, None]
        py = A[:, 1, None] + s * (B[:, 1] - A[:, 1])[:, None]
        R = self.rain_rate(px, py, t[:, None])
        D = np.linalg.norm(B - A, axis=1)
        return rain_specific_db_per_km(f_ghz, R, self.pol).mean(1) * D / 1000.0

    def rain_atten_slant(self, P, az_deg, el_deg, f_hz, t, n=12):
        P = as_points(P)
        N = len(P)
        el = np.radians(np.maximum(np.broadcast_to(el_deg, (N,)), 5.0))
        az = np.radians(np.broadcast_to(az_deg, (N,)))
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,))
        hR = self.cfg.freezing_height_m
        dz = np.maximum(hR - P[:, 2], 0.0)
        if (self.bg <= 0 and self.peak <= 0) or not np.any(dz > 0):
            return np.zeros(N)
        Ls = dz / np.sin(el)
        horiz = dz / np.tan(el)
        s = (np.arange(n) + 0.5) / n
        px = P[:, 0, None] + s * (horiz * np.sin(az))[:, None]
        py = P[:, 1, None] + s * (horiz * np.cos(az))[:, None]
        R = self.rain_rate(px, py, t[:, None])
        return rain_specific_db_per_km(f_hz / 1e9, R, self.pol).mean(1) * Ls / 1000.0

    # ---------------------------------------------------------------- cloud + fog
    def _tile(self, field, x, y, t, speed_scale=1.0):
        u = ((np.asarray(x) - self.wind[0] * speed_scale * np.asarray(t)) / self.cloud_tile_m) % 1.0 * field.shape[1]
        v = ((np.asarray(y) - self.wind[1] * speed_scale * np.asarray(t)) / self.cloud_tile_m) % 1.0 * field.shape[0]
        i0, j0 = np.floor(u).astype(int) % field.shape[1], np.floor(v).astype(int) % field.shape[0]
        i1, j1 = (i0 + 1) % field.shape[1], (j0 + 1) % field.shape[0]
        tx, ty = u - np.floor(u), v - np.floor(v)
        return (field[j0, i0] * (1 - tx) * (1 - ty) + field[j0, i1] * tx * (1 - ty)
                + field[j1, i0] * (1 - tx) * ty + field[j1, i1] * tx * ty)

    def cloud_lwc(self, x, y, t):
        """Cloud liquid-water density (g/m^3) inside the cloud layer at map position(s)."""
        c = self.cfg
        if c.cloud_cover <= 0:
            return np.zeros(np.broadcast(x, y, t).shape)
        # cover -> threshold of a unit gaussian field
        from math import erf, sqrt
        thr = -sqrt(2) * _erfinv(2 * c.cloud_cover - 1) if 0 < c.cloud_cover < 1 else (-9 if c.cloud_cover >= 1 else 9)
        n = self._tile(self.cloud_field, x, y, t)
        return c.cloud_lwc_gm3 * np.clip((n - thr) / 0.8, 0.0, 1.5)

    def fog_lwc(self, x, y, t):
        c = self.cfg
        if c.fog_density_gm3 <= 0:
            return np.zeros(np.broadcast(x, y, t).shape)
        n = self._tile(self.fog_field, x, y, t, 0.2)
        return c.fog_density_gm3 * np.clip(0.8 + 0.25 * n, 0.2, 1.5)

    def cloud_atten_slant(self, P, az_deg, el_deg, f_hz, t):
        P = as_points(P)
        N = len(P)
        c = self.cfg
        el = np.radians(np.maximum(np.broadcast_to(el_deg, (N,)), 5.0))
        az = np.radians(np.broadcast_to(az_deg, (N,)))
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,))
        lower = np.maximum(P[:, 2], c.cloud_base_m)
        thick = np.maximum(c.cloud_top_m - lower, 0.0)
        hmid = lower + thick / 2
        off = (hmid - P[:, 2]) / np.tan(el)
        lwc = self.cloud_lwc(P[:, 0] + off * np.sin(az), P[:, 1] + off * np.cos(az), t)
        kl = cloud_k_l(f_hz / 1e9, 273.15)
        out = kl * lwc * (thick / np.sin(el)) / 1000.0
        # fog layer (vertical slice near the ground)
        if c.fog_density_gm3 > 0:
            fz = np.maximum(c.fog_height_m - np.maximum(P[:, 2] - 0.0, 0.0), 0.0)
            out = out + kl * self.fog_lwc(P[:, 0], P[:, 1], t) * (fz / np.sin(el)) / 1000.0
        return out

    def fog_atten_path(self, A, B, f_hz, t, ground_a, ground_b, n=8):
        """Fog attenuation for a near-horizontal terrestrial path (ground_* = terrain height at ends)."""
        if self.cfg.fog_density_gm3 <= 0:
            return np.zeros(max(len(as_points(A)), len(as_points(B))))
        A, B = as_points(A), as_points(B)
        N = max(len(A), len(B))
        A, B = np.broadcast_to(A, (N, 3)), np.broadcast_to(B, (N, 3))
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,))
        s = (np.arange(n) + 0.5) / n
        px = A[:, 0, None] + s * (B[:, 0] - A[:, 0])[:, None]
        py = A[:, 1, None] + s * (B[:, 1] - A[:, 1])[:, None]
        pz = A[:, 2, None] + s * (B[:, 2] - A[:, 2])[:, None]
        gnd = np.broadcast_to(ground_a, (N,))[:, None] * (1 - s) + np.broadcast_to(ground_b, (N,))[:, None] * s
        inside = (pz - gnd) < self.cfg.fog_height_m
        m = self.fog_lwc(px, py, t[:, None]) * inside
        D = np.linalg.norm(B - A, axis=1)
        return cloud_k_l(f_hz / 1e9, 273.15) * m.mean(1) * D / 1000.0

    # ---------------------------------------------------------------- gases
    def gas_atten_path(self, A, B, f_hz):
        A, B = as_points(A), as_points(B)
        N = max(len(A), len(B))
        A, B = np.broadcast_to(A, (N, 3)), np.broadcast_to(B, (N, 3))
        zbar = (A[:, 2] + B[:, 2]) / 2
        go, gw = gas_specific_db_per_km(f_hz / 1e9, self.rho_wv)
        D = np.linalg.norm(B - A, axis=1)
        return (go * np.exp(-zbar / self.H_O2) + gw * np.exp(-zbar / self.H_WV)) * D / 1000.0

    def gas_atten_slant(self, z_m, el_deg, f_hz):
        z = np.maximum(np.asarray(z_m, dtype=float), 0.0)
        el = np.radians(np.maximum(np.asarray(el_deg, dtype=float), 5.0))
        go, gw = gas_specific_db_per_km(f_hz / 1e9, self.rho_wv)
        a = go * self.H_O2 * np.exp(-z / self.H_O2) + gw * self.H_WV * np.exp(-z / self.H_WV)
        return a / 1000.0 / np.sin(el)

    # ---------------------------------------------------------------- scintillation
    def scintillation_db(self, f_hz, el_deg, t, seed_key: int = 0):
        el = np.radians(np.maximum(np.asarray(el_deg, dtype=float), 5.0))
        sref = 3.6e-3 + 1e-4 * self.cfg.n_wet
        sig = sref * (f_hz / 1e9) ** (7.0 / 12.0) * 0.7 / np.sin(el) ** 1.2
        return sig * smooth_noise(t, self.seed + 77 + seed_key, 3.0)


def _erfinv(y: float) -> float:
    """Winitzki approximation + two Newton steps (accurate enough for cloud thresholds)."""
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    x = math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)
    for _ in range(2):
        err = math.erf(x) - y
        x -= err / (2 / math.sqrt(math.pi) * math.exp(-x * x))
    return x
