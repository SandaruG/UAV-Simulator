"""Spatially/temporally consistent fading generators.

* ShadowField : 2-layer correlated log-normal shadowing (Gudmundson-like spatial correlation,
                decorrelating with altitude), sampled at (x, y, z).
* FastFading  : Rician/Rayleigh small-scale fading as a deterministic sum of plane waves whose
                phases depend on position AND time, so a hovering drone still sees scatterer-
                induced fading while a moving one sees the physically correct fade spacing.
"""
from __future__ import annotations

import numpy as np

from .geo import Grid
from .util import C_LIGHT, correlated_field


class ShadowField:
    def __init__(self, grid: Grid, d_corr_m: float, seed: int, z_decorr_m: float = 60.0, pad_m: float = 1500.0):
        res = max(grid.res, d_corr_m / 5.0)
        w = (grid.x1 - grid.x0) + 2 * pad_m
        h = (grid.y1 - grid.y0) + 2 * pad_m
        nx, ny = int(min(1024, np.ceil(w / res))), int(min(1024, np.ceil(h / res)))
        res = max(w / nx, h / ny)
        self.grid = Grid(grid.x0 - pad_m, grid.y0 - pad_m, res, nx, ny)
        rng = np.random.default_rng(seed)
        self.f1 = correlated_field(ny, nx, res, d_corr_m, rng)
        self.f2 = correlated_field(ny, nx, res, d_corr_m, rng)
        self.z_decorr = z_decorr_m

    def sample(self, x, y, z):
        a = self.grid.bilinear(self.f1, x, y)
        b = self.grid.bilinear(self.f2, x, y)
        th = np.clip(np.asarray(z, dtype=float) / self.z_decorr, 0.0, 1.0) * (np.pi / 2)
        return np.cos(th) * a + np.sin(th) * b          # unit variance


class FastFading:
    def __init__(self, seed: int, f_hz: float, n: int = 20, doppler_hz: float = 3.0):
        rng = np.random.default_rng(seed)
        az = rng.uniform(0, 2 * np.pi, n)
        el = np.arcsin(rng.uniform(-0.9, 0.5, n))        # more energy arriving from below/horizon for aerial rx
        self.dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
        self.phi = rng.uniform(0, 2 * np.pi, n)
        self.fd = rng.uniform(-doppler_hz, doppler_hz, n)
        self.k = 2 * np.pi * f_hz / C_LIGHT
        self.n = n

    def gain_db(self, pos, t, k_db, d3d):
        """Small-scale power gain (dB, unit mean) for Rician K (dB) at positions `pos` (N,3)."""
        pos = np.asarray(pos, dtype=float)
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(pos),))
        ph = self.k * (pos @ self.dirs.T) + 2 * np.pi * self.fd[None, :] * t[:, None] + self.phi[None, :]
        diffuse = np.exp(1j * ph).sum(1) / np.sqrt(self.n)
        K = 10 ** (np.asarray(k_db, dtype=float) / 10.0)
        los = np.exp(1j * self.k * np.asarray(d3d, dtype=float))
        h = np.sqrt(K / (K + 1.0)) * los + np.sqrt(1.0 / (K + 1.0)) * diffuse
        return 10 * np.log10(np.maximum(np.abs(h) ** 2, 1e-4))
