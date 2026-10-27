"""Ready-made scenarios spanning the regimes a predictive-handover study needs.

    cfg = presets.get("rural_sparse", seed=3, weather="light_rain")
    env = RadioEnvironment(cfg)

All presets are ordinary `RadioEnvConfig` objects: change anything afterwards with `cfg.replace(...)`.
Numbers are plausible engineering defaults, NOT calibrated to a particular real operator or city.
"""
from __future__ import annotations

from typing import Callable, Dict

from .config import RadioEnvConfig


def dense_urban(seed: int = 1) -> RadioEnvConfig:
    c = RadioEnvConfig()
    c.area.name, c.area.seed = "dense-urban", seed
    c.buildings.mean_height_m, c.buildings.max_height_m, c.buildings.urban_core_radius_m = 35.0, 160.0, 1400.0
    c.cellular.isd_m = 500.0
    c.cellular.site_height_agl_m = 25.0
    c.interference.wifi.n_sources = 120
    return c


def suburban(seed: int = 1) -> RadioEnvConfig:
    c = RadioEnvConfig()
    c.area.name, c.area.seed = "suburban", seed
    c.buildings.mean_height_m, c.buildings.max_height_m = 9.0, 30.0
    c.buildings.urban_core_radius_m, c.buildings.n_subcenters = 600.0, 4
    c.vegetation.coverage, c.vegetation.canopy_mean_m = 0.35, 11.0
    c.cellular.isd_m, c.cellular.n_rings = 1200.0, 2
    c.stations[0].x, c.stations[0].y = -1200.0, -900.0
    return c


def rural_sparse(seed: int = 1) -> RadioEnvConfig:
    """8 x 8 km, few cell sites 4.5 km apart (NR on a quarter of them), forests and rolling terrain.
    Cellular holes appear; LEO is the fallback; the direct links dominate near the GCS."""
    c = RadioEnvConfig()
    c.area.name, c.area.seed = "rural-sparse", seed
    c.area.size_x_m = c.area.size_y_m = 8000.0
    c.area.resolution_m = 20.0
    c.terrain.relief_m, c.terrain.hurst = 250.0, 0.9
    c.buildings.urban_core_radius_m, c.buildings.n_subcenters, c.buildings.subcenter_radius_m = 250.0, 3, 220.0
    c.buildings.mean_height_m, c.buildings.max_height_m, c.buildings.rural_density = 7.0, 18.0, 0.01
    c.vegetation.coverage, c.vegetation.canopy_mean_m = 0.45, 16.0
    c.cellular.layout, c.cellular.isd_m, c.cellular.n_rings, c.cellular.jitter_m = "hex", 4500.0, 1, 400.0
    c.cellular.site_height_agl_m = 40.0
    for b in c.cellular.bands:
        if b.id == "nr3500":
            b.coverage_fraction = 0.25
    c.interference.wifi.n_sources, c.interference.wifi.region_radius_m = 15, 1800.0
    c.interference.uav.n_drones = 1
    c.interference.noise_environment = "rural"
    c.stations[0].x, c.stations[0].y, c.stations[0].height_agl_m = -2500.0, -2000.0, 4.0
    return c


def mountain_valley(seed: int = 1) -> RadioEnvConfig:
    """Strong relief: ridges shadow the direct links and the satellite at low elevation."""
    c = rural_sparse(seed)
    c.area.name = "mountain-valley"
    c.terrain.relief_m, c.terrain.hurst, c.terrain.base_elevation_m = 700.0, 0.75, 400.0
    c.vegetation.coverage = 0.55
    c.cellular.isd_m = 6000.0
    c.satellite.min_elev_deg = 30.0
    return c


_PRESETS: Dict[str, Callable[[int], RadioEnvConfig]] = {
    "dense_urban": dense_urban, "suburban": suburban, "rural_sparse": rural_sparse, "mountain_valley": mountain_valley,
    "demo": lambda seed=1: RadioEnvConfig().replace(**{"area.seed": seed}),
}


def names():
    return list(_PRESETS)


def get(name: str, seed: int = 1, weather: str = "clear", start_time_utc: str = None) -> RadioEnvConfig:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset '{name}'. Available: {names()}")
    c = _PRESETS[name](seed)
    c.weather.scenario = weather
    if start_time_utc:
        c.area.start_time_utc = start_time_utc
    return c
