"""Digital twin of the physical world: terrain, buildings, vegetation canopy and a clutter
class raster, either synthesised (statistically controllable) or loaded from real data
(SRTM/ASCII/GeoTIFF DEM, OSM GeoJSON / Overpass buildings and forest polygons).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RadioEnvConfig
from .geo import (Grid, LocalFrame, building_height_from_props, load_dem, load_vector_features,
                  rasterize_polygon)
from .util import box_mean, fractal_field, hash_uniform

OPEN, WATER, VEGETATION, SUBURBAN, URBAN = 0, 1, 2, 3, 4
CLUTTER_NAMES = {OPEN: "open", WATER: "water", VEGETATION: "vegetation", SUBURBAN: "suburban", URBAN: "urban"}


@dataclass
class GeoData:
    grid: Grid
    frame: LocalFrame
    terrain: np.ndarray        # m above MSL
    bldg: np.ndarray           # building height above terrain (m), 0 = none
    canopy: np.ndarray         # vegetation height above terrain (m), 0 = none
    clutter: np.ndarray        # uint8 class raster
    built_frac: np.ndarray     # local building plan-area fraction (0..1)
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        n = self.clutter.size
        return {
            "grid": f"{self.grid.nx}x{self.grid.ny} @ {self.grid.res} m",
            "terrain_min_max_m": (float(self.terrain.min()), float(self.terrain.max())),
            "tallest_building_m": float(self.bldg.max()),
            "building_cover_pct": float(100 * (self.bldg > 0).mean()),
            "canopy_cover_pct": float(100 * (self.canopy > 0).mean()),
            "clutter_pct": {CLUTTER_NAMES[k]: float(100 * (self.clutter == k).sum() / n) for k in CLUTTER_NAMES},
        }


# ----------------------------------------------------------------------------- terrain
def _terrain(cfg: RadioEnvConfig, grid: Grid, frame: LocalFrame, rng):
    tc = cfg.terrain
    if tc.source == "file":
        if not tc.path:
            raise ValueError("terrain.source='file' needs terrain.path")
        if tc.path.endswith(".npy"):
            arr = np.load(tc.path).astype(float)
            if arr.shape != (grid.ny, grid.nx):
                raise ValueError(f".npy DEM must already be on the map grid {(grid.ny, grid.nx)}")
            return arr, None
        dem = load_dem(tc.path)
        X, Y = np.meshgrid(grid.xs(), grid.ys())
        lat, lon, _ = frame.enu_to_ll(np.stack([X, Y, np.zeros_like(X)], -1))
        return dem.sample(lat, lon), None
    f = fractal_field(grid.ny, grid.nx, tc.hurst, rng)
    terr = tc.base_elevation_m + 0.25 * tc.relief_m * f
    water = None
    if tc.water_fraction > 0:
        level = np.quantile(terr, tc.water_fraction)
        water = terr < level
        terr = np.where(water, level, terr)
    return terr, water


# ----------------------------------------------------------------------------- buildings
def _synth_buildings(cfg: RadioEnvConfig, grid: Grid, rng):
    bc = cfg.buildings
    seed = cfg.area.seed
    X, Y = np.meshgrid(grid.xs(), grid.ys())
    pitch = bc.block_m + bc.street_m
    bi = np.floor(X / pitch).astype(np.int64)
    bj = np.floor(Y / pitch).astype(np.int64)
    ux, uy = X - bi * pitch, Y - bj * pitch
    in_block = (ux < bc.block_m) & (uy < bc.block_m)
    li = np.floor(ux / bc.lot_m).astype(np.int64)
    lj = np.floor(uy / bc.lot_m).astype(np.int64)
    cxb, cyb = (bi + 0.5) * pitch, (bj + 0.5) * pitch
    centers = [(0.0, 0.0, bc.urban_core_radius_m)]
    ext = 0.8 * min(grid.x1 - grid.x0, grid.y1 - grid.y0) / 2
    for _ in range(bc.n_subcenters):
        centers.append((rng.uniform(-ext, ext), rng.uniform(-ext, ext), bc.subcenter_radius_m * rng.uniform(0.6, 1.2)))
    d = np.zeros_like(X)
    for (cx, cy, r) in centers:
        d = np.maximum(d, np.exp(-(((cxb - cx) ** 2 + (cyb - cy) ** 2) / r**2)))
    dens = bc.rural_density + (1 - bc.rural_density) * d
    u_blk = hash_uniform(seed, 11, bi, bj)
    present = u_blk < np.clip(dens**0.7 * bc.fill, 0, 1)
    u_lot = hash_uniform(seed, 12, bi, bj, li, lj)
    lot_ok = u_lot < (0.5 + 0.45 * dens)
    hmean = bc.mean_height_m * (0.25 + 1.5 * dens**1.2)
    mult = np.exp((hash_uniform(seed, 13, bi, bj, li, lj) - 0.5) * 1.4)
    blk = np.exp((hash_uniform(seed, 14, bi, bj) - 0.5) * 0.8)
    h = np.clip(hmean * mult * blk, 3.5, bc.max_height_m)
    mask = in_block & present & lot_ok
    if grid.res <= 5.0:
        lx, ly = ux - li * bc.lot_m, uy - lj * bc.lot_m
        mask &= (lx > 1.5) & (lx < bc.lot_m - 1.5) & (ly > 1.5) & (ly < bc.lot_m - 1.5)
    return np.where(mask, h, 0.0)


def _real_buildings(cfg: RadioEnvConfig, grid: Grid, frame: LocalFrame):
    bc = cfg.buildings
    out = np.zeros((grid.ny, grid.nx))
    for ring, props in load_vector_features(bc.path):
        h = building_height_from_props(props, bc.default_height_m, bc.level_height_m)
        enu = frame.ll_to_enu(ring[:, 1], ring[:, 0], 0.0)
        rasterize_polygon(grid, enu[:, :2], out, h, "max")
    return out


# ----------------------------------------------------------------------------- vegetation
def _synth_vegetation(cfg: RadioEnvConfig, grid: Grid, bldg, water, built_frac, rng):
    vc = cfg.vegetation
    seed = cfg.area.seed
    f = fractal_field(grid.ny, grid.nx, vc.hurst, rng)
    thr = np.quantile(f, 1 - vc.coverage) if vc.coverage > 0 else np.inf
    norm = np.clip((f - thr) / (f.max() - thr + 1e-9), 0, 1)
    X, Y = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny))
    jitter = 0.85 + 0.3 * hash_uniform(seed, 21, X, Y)
    canopy = np.where(f > thr, vc.canopy_mean_m * (0.7 + 0.6 * norm) * jitter, 0.0)
    street_tree = (built_frac > 0.05) & (built_frac < 0.4) & (bldg == 0) & (hash_uniform(seed, 22, X, Y) < 0.12)
    canopy = np.where(street_tree & (canopy == 0), 8.0 * (0.8 + 0.4 * jitter), canopy)
    canopy[bldg > 0] = 0
    if water is not None:
        canopy[water] = 0
    return canopy


def _real_vegetation(cfg: RadioEnvConfig, grid: Grid, frame: LocalFrame):
    vc = cfg.vegetation
    out = np.zeros((grid.ny, grid.nx))
    for ring, props in load_vector_features(vc.path):
        tag = " ".join(str(v) for v in props.values()).lower()
        if any(k in tag for k in ("wood", "forest", "tree", "orchard", "scrub", "park")):
            h = building_height_from_props(props, vc.canopy_mean_m, 1.0)
            enu = frame.ll_to_enu(ring[:, 1], ring[:, 0], 0.0)
            rasterize_polygon(grid, enu[:, :2], out, h, "max")
    return out


# ----------------------------------------------------------------------------- top level
def build_geodata(cfg: RadioEnvConfig) -> GeoData:
    a = cfg.area
    nx, ny = int(round(a.size_x_m / a.resolution_m)), int(round(a.size_y_m / a.resolution_m))
    grid = Grid(-a.size_x_m / 2, -a.size_y_m / 2, a.resolution_m, nx, ny)
    frame = LocalFrame(a.origin_lat, a.origin_lon, 0.0)
    rng = np.random.default_rng(a.seed)

    terrain, water = _terrain(cfg, grid, frame, np.random.default_rng(a.seed + 1))

    if cfg.buildings.source == "synthetic":
        bldg = _synth_buildings(cfg, grid, np.random.default_rng(a.seed + 2))
    elif cfg.buildings.source == "geojson":
        bldg = _real_buildings(cfg, grid, frame)
    else:
        bldg = np.zeros((ny, nx))
    if water is not None:
        bldg = np.where(water, 0.0, bldg)
    built_frac = box_mean((bldg > 0).astype(float), max(1, int(round(100.0 / a.resolution_m))))

    if cfg.vegetation.source == "synthetic":
        canopy = _synth_vegetation(cfg, grid, bldg, water, built_frac, np.random.default_rng(a.seed + 3))
    elif cfg.vegetation.source == "geojson":
        canopy = _real_vegetation(cfg, grid, frame)
        canopy[bldg > 0] = 0
    else:
        canopy = np.zeros((ny, nx))

    clutter = np.full((ny, nx), OPEN, dtype=np.uint8)
    clutter[canopy > 2.0] = VEGETATION
    clutter[built_frac > 0.08] = SUBURBAN
    clutter[built_frac > 0.35] = URBAN
    if water is not None:
        clutter[water] = WATER

    gd = GeoData(grid, frame, terrain, bldg, canopy, clutter, built_frac,
                 meta={"seed": a.seed, "terrain_source": cfg.terrain.source,
                       "building_source": cfg.buildings.source, "vegetation_source": cfg.vegetation.source})
    return gd
