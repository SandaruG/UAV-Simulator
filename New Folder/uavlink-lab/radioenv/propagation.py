"""Deterministic propagation engine on top of the geodata raster.

* `Surface`            - fast point/ray queries on terrain, buildings, canopy, clutter
* `terrestrial_path`   - profile-based path analysis: line-of-sight, Fresnel clearance,
                         Deygout multiple knife-edge diffraction (ITU-R P.526 J(v)),
                         building penetration, vegetation loss (Weissberger), two-ray
                         ground reflection with clutter-dependent specularity,
                         delay spread and Rician K-factor
* antenna patterns     - 3GPP sector, vertical dipole, flat-panel phased array
* 3GPP aerial path-loss models (UMa-AV / RMa-AV flavoured, see notes in docstrings)
"""
from __future__ import annotations

import numpy as np

from .config import PropagationConfig
from .terrain import GeoData, OPEN, SUBURBAN, URBAN, VEGETATION, WATER
from .util import C_LIGHT, R_EARTH, as_points, fspl_db, hash_uniform, knife_edge_loss_db

# ground electrical properties and specular-reflection weight per clutter class
#            eps_r  sigma(S/m)  weight
_GROUND = {
    OPEN: (15.0, 0.005, 0.70),
    WATER: (80.0, 5.0, 1.00),
    VEGETATION: (12.0, 0.010, 0.12),
    SUBURBAN: (8.0, 0.003, 0.25),
    URBAN: (5.0, 0.002, 0.20),
}
_G_EPS = np.array([_GROUND[k][0] for k in range(5)])
_G_SIG = np.array([_GROUND[k][1] for k in range(5)])
_G_W = np.array([_GROUND[k][2] for k in range(5)])


class Surface:
    """Point / ray access to the geodata with out-of-map clamping."""

    def __init__(self, gd: GeoData, leaf_factor: float = 1.0):
        self.gd = gd
        self.grid = gd.grid
        self.leaf_factor = leaf_factor

    def inside(self, x, y):
        return self.grid.inside(np.asarray(x), np.asarray(y))

    def terrain_at(self, x, y):
        return self.grid.bilinear(self.gd.terrain, x, y)

    def bldg_at(self, x, y):
        return np.where(self.inside(x, y), self.grid.nearest(self.gd.bldg, x, y), 0.0)

    def canopy_at(self, x, y):
        return np.where(self.inside(x, y), self.grid.nearest(self.gd.canopy, x, y), 0.0)

    def clutter_at(self, x, y):
        return self.grid.nearest(self.gd.clutter, x, y)

    def built_frac_at(self, x, y):
        return self.grid.bilinear(self.gd.built_frac, x, y)

    def top_at(self, x, y):
        return self.terrain_at(x, y) + self.bldg_at(x, y)

    def agl(self, p):
        p = as_points(p)
        return p[:, 2] - self.top_at(p[:, 0], p[:, 1])

    def collision(self, p):
        """True where the point is inside a building or underground."""
        p = as_points(p)
        return p[:, 2] < self.top_at(p[:, 0], p[:, 1])

    # ------------------------------------------------------------------ rays (satellite horizon)
    def ray_obstruction(self, P, az_deg, el_deg, lam, max_d=1500.0, n=64):
        """March from P along (az, el). Returns (knife-edge loss dB for a far target,
        path length inside canopy m, hard-blocked bool)."""
        P = as_points(P)
        az, el = np.radians(np.asarray(az_deg, dtype=float)), np.radians(np.asarray(el_deg, dtype=float))
        dirs = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)
        ds = np.geomspace(3.0, max_d, n)
        pts = P[:, None, :] + ds[None, :, None] * dirs[:, None, :]
        ter = self.terrain_at(pts[..., 0], pts[..., 1])
        bld = self.bldg_at(pts[..., 0], pts[..., 1])
        can = self.canopy_at(pts[..., 0], pts[..., 1])
        h = ter + bld - pts[..., 2]
        v = h * np.sqrt(2.0 / (lam * ds))[None, :]
        loss = knife_edge_loss_db(v.max(1))
        dl = np.gradient(ds)
        inside_c = (pts[..., 2] < ter + can) & (pts[..., 2] > ter) & (can > 0.5)
        len_c = (inside_c * dl[None, :]).sum(1)
        return loss, len_c, h.max(1) > 0


# ----------------------------------------------------------------------------- vegetation, walls
def foliage_loss_db(len_m, f_hz, leaf_factor=1.0):
    """Weissberger modified exponential decay model (230 MHz - 95 GHz)."""
    f = max(f_hz / 1e9, 0.23)
    d = np.minimum(np.asarray(len_m, dtype=float), 400.0)
    loss = np.where(d <= 14.0, 0.45 * f**0.284 * d, 1.33 * f**0.284 * np.maximum(d, 1e-9) ** 0.588)
    return loss * leaf_factor


def _deygout(hT, hR, obst, s, D, lam, cap):
    N, M = obst.shape
    Dc = D[:, None]
    line = hT[:, None] + s[None, :] * (hR - hT)[:, None]
    h = obst - line
    d1, d2 = s[None, :] * Dc, (1 - s[None, :]) * Dc
    with np.errstate(invalid="ignore", divide="ignore"):
        v = h * np.sqrt(2 * Dc / (lam * d1 * d2))
    rows = np.arange(N)
    m = v.argmax(1)
    vm = v[rows, m]
    sm = s[m]
    om = obst[rows, m]
    idx = np.arange(M)[None, :]
    smc = sm[:, None]
    # left sub-path: Tx -> main edge
    with np.errstate(invalid="ignore", divide="ignore"):
        lineL = hT[:, None] + (s[None, :] / smc) * (om - hT)[:, None]
        hL = obst - lineL
        d1L, d2L = s[None, :] * Dc, (smc - s[None, :]) * Dc
        vL = hL * np.sqrt(2 * (smc * Dc) / (lam * d1L * d2L))
        vL = np.where(idx < m[:, None], vL, -np.inf)
        lineR = om[:, None] + ((s[None, :] - smc) / (1 - smc)) * (hR[:, None] - om[:, None])
        hRr = obst - lineR
        d1R, d2R = (s[None, :] - smc) * Dc, (1 - s[None, :]) * Dc
        vR = hRr * np.sqrt(2 * ((1 - smc) * Dc) / (lam * d1R * d2R))
        vR = np.where(idx > m[:, None], vR, -np.inf)
    vL = np.nan_to_num(vL, nan=-np.inf)
    vR = np.nan_to_num(vR, nan=-np.inf)
    L = knife_edge_loss_db(vm) + knife_edge_loss_db(vL.max(1)) + knife_edge_loss_db(vR.max(1))
    L = np.where(vm > -0.78, L, 0.0)
    return np.minimum(L, cap)


def two_ray_gain_db(S: Surface, A, B, d2d, lam, pc: PropagationConfig, los):
    hA = np.maximum(A[:, 2] - S.terrain_at(A[:, 0], A[:, 1]), 0.3)
    hB = np.maximum(B[:, 2] - S.terrain_at(B[:, 0], B[:, 1]), 0.3)
    d = np.maximum(d2d, 1.0)
    rd = np.sqrt(d**2 + (hA - hB) ** 2)
    rr = np.sqrt(d**2 + (hA + hB) ** 2)
    psi = np.arctan2(hA + hB, d)
    cl = S.clutter_at((A[:, 0] + B[:, 0]) / 2, (A[:, 1] + B[:, 1]) / 2).astype(int)
    eps, sig, w = _G_EPS[cl], _G_SIG[cl], _G_W[cl]
    epsc = eps - 1j * 60.0 * sig * lam
    root = np.sqrt(epsc - np.cos(psi) ** 2)
    if pc.polarization.upper() == "H":
        G = (np.sin(psi) - root) / (np.sin(psi) + root)
    else:
        G = (epsc * np.sin(psi) - root) / (epsc * np.sin(psi) + root)
    E = 1.0 + w * G * (rd / rr) * np.exp(-2j * np.pi * (rr - rd) / lam)
    g = 20 * np.log10(np.maximum(np.abs(E), 10 ** (pc.ground_reflection_floor_db / 20)))
    return np.where(los, g, 0.0)


# ----------------------------------------------------------------------------- main path analysis
def terrestrial_path(S: Surface, A, B, f_hz, pc: PropagationConfig, fast: bool = False, two_ray: bool = True,
                     cfg_fading=None):
    """Analyse the straight path A->B (both (N,3) or (3,)). All outputs are (N,) arrays.
    Large batches are processed in row chunks to bound memory."""
    A, B = as_points(A), as_points(B)
    N = max(len(A), len(B))
    A, B = np.broadcast_to(A, (N, 3)), np.broadcast_to(B, (N, 3))
    Dmax = float(np.linalg.norm(B - A, axis=1).max())
    M = int(np.clip(np.ceil(Dmax / S.grid.res), 16, pc.profile_max_samples))
    if fast:
        M = min(M, pc.fast_samples)
    chunk = max(64, int(1.2e6 // M))
    if N <= chunk:
        return _path_core(S, A, B, f_hz, pc, M, two_ray, cfg_fading)
    parts = [_path_core(S, A[a:a + chunk], B[a:a + chunk], f_hz, pc, M, two_ray, cfg_fading) for a in range(0, N, chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _path_core(S, A, B, f_hz, pc, M, two_ray, cfg_fading):
    N = len(A)
    lam = C_LIGHT / f_hz
    dv = B - A
    D = np.maximum(np.linalg.norm(dv, axis=1), 1.0)
    d2d = np.hypot(dv[:, 0], dv[:, 1])
    s = (np.arange(M) + 1.0) / (M + 1.0)
    px = A[:, 0, None] + s * dv[:, 0, None]
    py = A[:, 1, None] + s * dv[:, 1, None]
    pz = A[:, 2, None] + s * dv[:, 2, None]
    ter = S.terrain_at(px, py)
    bld = S.bldg_at(px, py)
    can = S.canopy_at(px, py)
    Dc = D[:, None]
    d1, d2 = s * Dc, (1 - s) * Dc
    bulge = d1 * d2 / (2 * pc.k_factor * R_EARTH)
    obst = ter + bld + bulge
    h = obst - pz
    los = h.max(1) < 0
    r1 = np.sqrt(lam * d1 * d2 / Dc)
    fres = (-h / r1).min(1)
    # A raster profile samples one physical obstacle (a roof, a ridge crest) at several adjacent cells.
    # Deygout would count each cell as a separate knife edge, so keep only local maxima: one edge
    # per plateau / crest, placed at its highest (and, on ties, leading) sample.
    ninf = np.full((N, 1), -np.inf)
    left = np.concatenate([ninf, obst[:, :-1]], axis=1)
    right = np.concatenate([obst[:, 1:], ninf], axis=1)
    obst_pk = np.where((obst > left) & (obst >= right), obst, -1e9)
    L_diff = _deygout(A[:, 2], B[:, 2], obst_pk, s, D, lam, pc.max_diff_loss_db)

    inside_b = (pz < ter + bld) & (bld > 0.5)
    inside_c = (pz < ter + can) & (pz > ter) & (can > 0.5)
    len_b = inside_b.mean(1) * D
    len_c = inside_c.mean(1) * D
    f_ghz = f_hz / 1e9
    L_veg = foliage_loss_db(len_c, f_hz, S.leaf_factor)
    L_pen = np.minimum(pc.wall_base_db + pc.wall_db_per_m * len_b + pc.wall_db_per_ghz * f_ghz, 60.0)
    L_obst = np.where(len_b > 0, np.minimum(L_diff, L_pen), L_diff)

    tr = two_ray_gain_db(S, A, B, d2d, lam, pc, los) if (pc.use_two_ray and two_ray) else np.zeros(N)

    cl_a = S.clutter_at(A[:, 0], A[:, 1])
    cl_b = S.clutter_at(B[:, 0], B[:, 1])
    urban = np.maximum(cl_a, cl_b) >= SUBURBAN
    fc = max(f_ghz, 0.5)
    lg = np.where(urban,
                  np.where(los, -6.955 - 0.0963 * np.log10(fc), -6.28 - 0.204 * np.log10(fc)),
                  np.where(los, -7.49, -7.43))
    ds_ns = 10 ** lg * 1e9
    elev = np.degrees(np.arctan2(np.abs(dv[:, 2]), np.maximum(d2d, 1e-3)))
    k0 = getattr(cfg_fading, "k_los_ground_db", 6.0)
    k1 = getattr(cfg_fading, "k_los_slope_db", 9.0)
    k_db = np.where(los, np.minimum(k0 + k1 * np.sin(np.radians(elev)), 15.0), -30.0)

    return dict(d3d=D, d2d=d2d, los=los, fresnel_clear=fres, loss_diff=L_obst, loss_veg=L_veg,
                loss_pen=np.where(len_b > 0, L_pen, 0.0), two_ray_db=tr, len_canopy=len_c, len_bldg=len_b,
                clutter_a=cl_a, clutter_b=cl_b, delay_spread_ns=ds_ns, elev_deg=elev, k_factor_db=k_db)


# ----------------------------------------------------------------------------- antennas
def sector_gain_dbi(az_rel_deg, el_off_deg, gmax, hpbw_h=65.0, hpbw_v=10.0, sla_v=28.0, am=30.0):
    """3GPP TR 38.901 single-element sector pattern (az relative to boresight, el offset from the
    tilted boresight)."""
    az = (np.asarray(az_rel_deg) + 180.0) % 360.0 - 180.0
    ah = -np.minimum(12.0 * (az / hpbw_h) ** 2, am)
    av = -np.minimum(12.0 * (np.asarray(el_off_deg) / hpbw_v) ** 2, sla_v)
    return gmax - np.minimum(-(ah + av), am)


def dipole_gain_dbi(el_deg, peak=2.15):
    """Vertical half-wave dipole; null towards zenith / nadir (floored at -20 dB)."""
    th = np.radians(90.0 - np.abs(np.asarray(el_deg, dtype=float)))
    s = np.maximum(np.sin(th), 1e-3)
    pat = np.abs(np.cos(np.pi / 2 * np.cos(th)) / s)
    return peak + 20 * np.log10(np.maximum(pat, 0.1))


def panel_gain_dbi(off_boresight_deg, g0, max_scan_deg=60.0, exponent=1.3):
    """Flat phased-array / patch: cos^n scan loss, large penalty beyond max scan."""
    th = np.radians(np.minimum(np.abs(np.asarray(off_boresight_deg, dtype=float)), 89.0))
    g = g0 + 10 * exponent * np.log10(np.maximum(np.cos(th), 1e-3))
    return np.where(np.abs(off_boresight_deg) > max_scan_deg, g0 - 30.0, g)


# ----------------------------------------------------------------------------- 3GPP aerial PL
# Formulas follow TR 38.901 (UMa/RMa) and TR 36.777 (aerial UE extensions) as remembered by the
# author; verify coefficients against the specs before quoting absolute numbers in a paper.
def p_los_3gpp(env: str, d2d, h_ut):
    d2d = np.maximum(d2d, 1e-3)
    h_ut = np.maximum(h_ut, 1.5)
    if env == "RMa":
        low = np.where(d2d <= 10, 1.0, np.exp(-(d2d - 10) / 1000.0))
        d1 = np.maximum(1350 * np.log10(h_ut) - 1602, 18.0)
        p1 = np.maximum(15021 * np.log10(h_ut) - 16053, 1000.0)
        high = np.where(d2d <= d1, 1.0, d1 / d2d + np.exp(-d2d / p1) * (1 - d1 / d2d))
        return np.where(h_ut <= 10, low, high)
    base = np.where(d2d <= 18, 1.0, 18.0 / d2d + np.exp(-d2d / 63.0) * (1 - 18.0 / d2d))
    c = np.where(h_ut <= 13, 0.0, ((h_ut - 13) / 10.0) ** 1.5)
    mid = base * (1 + c * 1.25 * (d2d / 100.0) ** 3 * np.exp(-d2d / 150.0))
    d1 = np.maximum(460 * np.log10(h_ut) - 700, 18.0)
    p1 = np.maximum(4300 * np.log10(h_ut) - 3800, 1.0)
    high = np.where(d2d <= d1, 1.0, d1 / d2d + np.exp(-d2d / p1) * (1 - d1 / d2d))
    return np.clip(np.where(h_ut <= 22.5, mid, high), 0, 1)


def pl_3gpp_db(env: str, los, d3d, d2d, h_bs, h_ut, fc_ghz, w=20.0, h_bld=20.0):
    d3d = np.maximum(d3d, 10.0)
    h_ut = np.maximum(h_ut, 1.5)
    lf = 20 * np.log10(fc_ghz)
    if env == "RMa":
        h = 5.0
        def pl1(d):
            return (20 * np.log10(40 * np.pi * d * fc_ghz / 3.0) + np.minimum(0.03 * h**1.72, 10.0) * np.log10(d)
                    - np.minimum(0.044 * h**1.72, 14.77) + 0.002 * np.log10(h) * d)
        dbp = 2 * np.pi * h_bs * h_ut * fc_ghz * 1e9 / C_LIGHT
        pl_los = np.where(d2d <= dbp, pl1(d3d), pl1(dbp) + 40 * np.log10(d3d / np.maximum(dbp, 1.0)))
        pl_nlos = (161.04 - 7.1 * np.log10(w) + 7.5 * np.log10(h) - (24.37 - 3.7 * (h / h_bs) ** 2) * np.log10(h_bs)
                   + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d3d) - 3) + lf
                   - (3.2 * np.log10(11.75 * h_ut) ** 2 - 4.97))
        pl_nlos = np.maximum(pl_los, pl_nlos)
    else:
        dbp = 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc_ghz * 1e9 / C_LIGHT
        pl_a = 28.0 + 22 * np.log10(d3d) + lf
        pl_b = 28.0 + 40 * np.log10(d3d) + lf - 9 * np.log10(dbp**2 + (h_bs - h_ut) ** 2)
        pl_los = np.where(d2d <= dbp, pl_a, pl_b)
        nl_low = 13.54 + 39.08 * np.log10(d3d) + lf - 0.6 * (h_ut - 1.5)
        nl_high = -17.5 + (46 - 7 * np.log10(h_ut)) * np.log10(d3d) + 20 * np.log10(40 * np.pi * fc_ghz / 3.0)
        pl_nlos = np.maximum(pl_los, np.where(h_ut <= 22.5, nl_low, nl_high))
    return np.where(los, pl_los, pl_nlos)


def sigma_sf_3gpp(env: str, los, h_ut):
    h_ut = np.maximum(h_ut, 1.5)
    if env == "RMa":
        sl = np.where(h_ut <= 10, 4.0, 4.2 * np.exp(-0.0046 * h_ut))
        sn = np.where(h_ut <= 10, 8.0, 6.0)
    else:
        sl = np.where(h_ut <= 22.5, 4.0, 4.64 * np.exp(-0.0066 * h_ut))
        sn = 6.0 * np.ones_like(h_ut)
    return np.where(los, sl, sn)
