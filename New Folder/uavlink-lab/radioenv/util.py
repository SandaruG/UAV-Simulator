"""Shared numerical helpers: constants, deterministic hash-based RNG, random fields,
astronomy helpers (Julian date, GMST, Sun position) and small RF utilities.

Everything in this module is stateless / deterministic so that the environment can be
queried at any (x, y, z, t) in any order and always return the same answer for the same
seed.
"""
from __future__ import annotations

import datetime as _dt
import functools
import math

import numpy as np

C_LIGHT = 299_792_458.0
BOLTZMANN = 1.380649e-23
T0_K = 290.0
R_EARTH = 6_371_000.0
OMEGA_EARTH = 7.2921159e-5  # rad/s


# ----------------------------------------------------------------------------- dB helpers
def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-30))


def undb(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def dbm_sum(*levels_dbm):
    """Power-sum of levels given in dBm (broadcasting)."""
    arrs = np.broadcast_arrays(*[np.asarray(l, dtype=float) for l in levels_dbm])
    return db(np.sum([undb(a) for a in arrs], axis=0))


def thermal_noise_dbm(bw_hz, nf_db=0.0):
    return -174.0 + 10.0 * np.log10(bw_hz) + nf_db


# ----------------------------------------------------------------------------- hashing RNG
_U64 = np.uint64


def _mix(x):
    with np.errstate(over="ignore"):
        x = x + _U64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)
        return x ^ (x >> _U64(31))


def hash_uniform(*keys):
    """Deterministic U[0,1) from integer keys (broadcasting, vectorised)."""
    h = np.zeros((), dtype=np.uint64) + _U64(0x243F6A8885A308D3)
    for k in keys:
        k = np.asarray(k)
        if k.dtype.kind == "f":
            k = np.floor(k)
        k = k.astype(np.int64).astype(np.uint64)
        h = _mix(h ^ k)
    return (h >> _U64(11)).astype(np.float64) / float(1 << 53)


def hash_normal(*keys):
    u1 = np.clip(hash_uniform(*keys, 7919), 1e-12, 1.0)
    u2 = hash_uniform(*keys, 104729)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def str_seed(s: str, base: int = 0) -> int:
    """Stable (process independent) integer seed from a string."""
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int((h + base * 7919) % (2**31 - 1))


# ----------------------------------------------------------------------------- smooth noise in time
@functools.lru_cache(maxsize=512)
def _sinus_params(seed: int, tau: float, n: int):
    rng = np.random.default_rng(seed)
    w = (2 * np.pi / tau) * rng.uniform(0.3, 1.7, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return w, ph


def smooth_noise(t, seed: int, tau: float, n: int = 10):
    """Deterministic ~N(0,1) process with correlation time of order `tau` seconds."""
    w, ph = _sinus_params(int(seed), float(tau), int(n))
    t = np.asarray(t, dtype=float)
    return np.sqrt(2.0 / n) * np.cos(t[..., None] * w + ph).sum(-1)


# ----------------------------------------------------------------------------- random fields
def fractal_field(ny: int, nx: int, hurst: float, rng: np.random.Generator):
    """Zero-mean, unit-variance fractal (power-law spectrum) field via FFT."""
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    amp = k ** (-(hurst + 1.0))
    amp[0, 0] = 0.0
    spec = np.fft.rfft2(rng.normal(size=(ny, nx))) * amp
    f = np.fft.irfft2(spec, s=(ny, nx))
    return (f - f.mean()) / (f.std() + 1e-12)


def correlated_field(ny: int, nx: int, res_m: float, d_corr_m: float, rng: np.random.Generator):
    """Unit-variance 2-D Gaussian field with ~exponential (Gudmundson) autocorrelation."""
    ky = np.fft.fftfreq(ny, d=res_m)[:, None]
    kx = np.fft.rfftfreq(nx, d=res_m)[None, :]
    w = 2 * np.pi * np.sqrt(kx**2 + ky**2)
    amp = (1.0 + (w * d_corr_m) ** 2) ** (-0.75)
    spec = np.fft.rfft2(rng.normal(size=(ny, nx))) * amp
    f = np.fft.irfft2(spec, s=(ny, nx))
    return (f - f.mean()) / (f.std() + 1e-12)


def box_mean(arr, r: int):
    """Mean over (2r+1)^2 window using an integral image (edge-clamped)."""
    if r <= 0:
        return arr.astype(float)
    a = np.pad(arr.astype(float), r, mode="edge")
    c = np.cumsum(np.cumsum(a, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


# ----------------------------------------------------------------------------- RF utilities
def fspl_db(d_m, f_hz):
    d = np.maximum(np.asarray(d_m, dtype=float), 1.0)
    return 20 * np.log10(d) + 20 * np.log10(f_hz) - 147.5522


def knife_edge_loss_db(v):
    """ITU-R P.526 single knife-edge diffraction loss J(v) (dB), >= 0."""
    v = np.clip(np.nan_to_num(np.asarray(v, dtype=float), nan=-10.0), -10.0, 1e5)
    x = v - 0.1
    out = 6.9 + 20 * np.log10(np.sqrt(x * x + 1.0) + x)
    return np.where(v > -0.78, np.maximum(out, 0.0), 0.0)


def rician_outage_prob(gamma_th_db, gamma_mean_db, k_lin):
    """P(instantaneous SINR < threshold) for Rician fading with mean SINR gamma_mean."""
    th = undb(gamma_th_db)
    mean = undb(gamma_mean_db)
    k = np.maximum(np.asarray(k_lin, dtype=float), 0.0)
    x = 2.0 * (k + 1.0) * th / np.maximum(mean, 1e-30)
    try:
        from scipy.stats import ncx2

        return np.clip(ncx2.cdf(x, 2, 2.0 * k), 0.0, 1.0)
    except Exception:  # Rayleigh fallback
        return np.clip(1.0 - np.exp(-th / np.maximum(mean, 1e-30)), 0.0, 1.0)


# ----------------------------------------------------------------------------- astronomy
def parse_utc(s) -> _dt.datetime:
    if isinstance(s, _dt.datetime):
        return s.replace(tzinfo=None)
    return _dt.datetime.fromisoformat(str(s).replace("Z", "")).replace(tzinfo=None)


def julian_date(dt: _dt.datetime) -> float:
    a = (14 - dt.month) // 12
    yy = dt.year + 4800 - a
    mm = dt.month + 12 * a - 3
    jdn = dt.day + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    frac = (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0 + dt.microsecond / 86400e6
    return jdn + frac


def gmst_deg(jd: float) -> float:
    return (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0


def sun_az_el(dt_utc: _dt.datetime, lat_deg: float, lon_deg: float):
    """Low-precision (~0.01 deg) solar azimuth (from north, cw) and elevation in degrees."""
    jd = julian_date(dt_utc)
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360.0
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    H = math.radians(gmst_deg(jd)) + math.radians(lon_deg) - ra
    la = math.radians(lat_deg)
    el = math.asin(math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(H))
    az = math.atan2(-math.cos(dec) * math.sin(H), math.sin(dec) * math.cos(la) - math.cos(dec) * math.sin(la) * math.cos(H))
    return math.degrees(az) % 360.0, math.degrees(el)


def local_hour(start_utc: _dt.datetime, t_s, lon_deg: float):
    """Local mean solar hour [0,24) at time offset t_s seconds after start."""
    h0 = start_utc.hour + start_utc.minute / 60 + start_utc.second / 3600
    return (h0 + np.asarray(t_s, dtype=float) / 3600.0 + lon_deg / 15.0) % 24.0


def as_points(p, n=None):
    """Coerce to float (N,3) array."""
    a = np.atleast_2d(np.asarray(p, dtype=float))
    if a.shape[-1] != 3:
        raise ValueError("positions must have 3 components (x, y, z)")
    if n is not None and a.shape[0] == 1 and n > 1:
        a = np.repeat(a, n, axis=0)
    return a
