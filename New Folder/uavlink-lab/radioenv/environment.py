"""RadioEnvironment: the queryable digital twin.

    env = RadioEnvironment(RadioEnvConfig())
    q = env.query("video_58", (x, y, z), t)          # -> LinkQuality
    qs = env.query_all((x, y, z), t)                  # {link_id: LinkQuality}

Positions are local ENU metres relative to the map centre (z = metres above MSL, same datum as the
terrain). `t` is seconds since `area.start_time_utc`. Queries are stateless and deterministic: the
same (config, seed, x, y, z, t) always gives the same answer, in any order, from any process.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .cellular import CellularNetwork
from .config import GroundStationConfig, LinkSpec, RadioEnvConfig
from .fading import FastFading, ShadowField
from .interference import InterferenceField
from .links import eval_direct, eval_satellite
from .propagation import Surface
from .quality import LinkQuality, squeeze_result
from .satellite import Constellation
from .terrain import CLUTTER_NAMES, GeoData, build_geodata
from .util import as_points, parse_utc, str_seed
from .weather import Weather


def leaf_factor(month: int, evergreen_fraction: float, hemisphere: str = "N") -> float:
    """Foliage attenuation multiplier: 1 = full leaf, evergreen part always 1."""
    m = month if hemisphere.upper() == "N" else (month + 5) % 12 + 1
    season = {12: 0.35, 1: 0.3, 2: 0.3, 3: 0.5, 4: 0.8, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 0.95, 10: 0.7, 11: 0.45}[m]
    return evergreen_fraction * 1.0 + (1.0 - evergreen_fraction) * season


class RadioEnvironment:
    def __init__(self, cfg: Optional[RadioEnvConfig] = None, geodata: Optional[GeoData] = None):
        self.cfg = cfg or RadioEnvConfig()
        c = self.cfg
        self.start_utc = parse_utc(c.area.start_time_utc)
        self.gd = geodata or build_geodata(c)
        lf = leaf_factor(self.start_utc.month, c.vegetation.evergreen_fraction, c.vegetation.hemisphere)
        self.surface = Surface(self.gd, leaf_factor=lf)
        self.surface.obstruction_range_m = c.propagation.obstruction_range_m
        self.weather = Weather(c.weather, c.area.seed, c.propagation.polarization)
        half = (c.area.size_x_m / 2, c.area.size_y_m / 2)
        self.interf = InterferenceField(c.interference, self.surface, c.propagation, c.area.seed,
                                        self.start_utc, c.area.origin_lon, half)
        self.constellation = Constellation(c.satellite, self.start_utc, self.gd.frame) if c.satellite.enabled else None
        self._shadow: Dict[str, ShadowField] = {}
        self._fast: Dict[str, FastFading] = {}
        self._stations = {}
        for s in c.stations:
            z = float(self.surface.top_at(s.x, s.y)) + s.height_agl_m
            self._stations[s.id] = dict(id=s.id, xyz=np.array([s.x, s.y, z]), gain_dbi=s.gain_dbi,
                                        pattern=s.pattern, height_agl_m=s.height_agl_m)
        self.cellular = CellularNetwork(self) if c.cellular.enabled else None
        self._links: Dict[str, LinkSpec] = {}
        for l in c.links:
            self.add_link(l)

    # ------------------------------------------------------------------ context for sub-modules
    def station(self, sid):
        if sid not in self._stations:
            raise KeyError(f"unknown ground station '{sid}' (have {list(self._stations)})")
        return self._stations[sid]

    def get_shadow(self, key: str, d_corr_m: float) -> ShadowField:
        k = f"{key}|{d_corr_m:.1f}"
        if k not in self._shadow:
            self._shadow[k] = ShadowField(self.gd.grid, d_corr_m, str_seed(key, self.cfg.area.seed),
                                          self.cfg.fading.shadow_zdecorr_m)
        return self._shadow[k]

    def get_fast(self, key: str, f_hz: float) -> FastFading:
        k = f"{key}|{f_hz:.0f}"
        if k not in self._fast:
            fc = self.cfg.fading
            self._fast[k] = FastFading(str_seed(key, self.cfg.area.seed + 1), f_hz, fc.n_scatterers, fc.scatter_doppler_hz)
        return self._fast[k]

    # ------------------------------------------------------------------ link registry
    def add_link(self, spec: LinkSpec):
        if spec.kind == "cellular":
            if self.cellular is None:
                raise ValueError("cellular is disabled in the config")
            if spec.band not in self.cellular.bands:
                raise ValueError(f"link {spec.id}: unknown band '{spec.band}'")
        if spec.kind == "satellite" and self.constellation is None:
            raise ValueError("satellite is disabled in the config")
        if spec.kind == "direct":
            self.station(spec.station)
        self._links[spec.id] = spec

    @property
    def link_ids(self) -> List[str]:
        return list(self._links)

    def link(self, link_id: str) -> LinkSpec:
        return self._links[link_id]

    # ------------------------------------------------------------------ queries
    def query(self, link_id: str, pos, t=0.0, vel=None, fading: str = "full", **cellular_kw) -> LinkQuality:
        """Instantaneous quality of `link_id` at position(s) `pos` (3,) or (N,3) and time(s) `t`.
        fading: 'full' (shadowing + small-scale), 'shadow' (no small-scale) or 'none' (mean only)."""
        if link_id not in self._links:
            raise KeyError(f"unknown link '{link_id}'. Available: {self.link_ids}")
        spec = self._links[link_id]
        P = as_points(pos)
        single = P.shape[0] == 1 and np.ndim(t) == 0
        if spec.kind == "direct":
            lq = eval_direct(self, spec, P, t, fading)
        elif spec.kind == "cellular":
            lq = self.cellular.evaluate(spec, P, t, fading, **cellular_kw)
        elif spec.kind == "satellite":
            lq = eval_satellite(self, spec, P, t, vel, fading)
        else:
            raise ValueError(f"unknown link kind {spec.kind}")
        return squeeze_result(lq) if single else lq

    def query_xyz(self, link_id: str, x, y, z, t=0.0, **kw) -> LinkQuality:
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
        return self.query(link_id, pts if pts.shape[0] > 1 else pts[0], t, **kw)

    def query_all(self, pos, t=0.0, vel=None, fading: str = "full", links: Optional[Iterable[str]] = None) -> Dict[str, LinkQuality]:
        return {k: self.query(k, pos, t, vel=vel, fading=fading) for k in (links or self.link_ids)}

    def query_track(self, positions, times, vel=None, fading: str = "full", links=None) -> Dict[str, LinkQuality]:
        """Evaluate every link along a flight track (N,3) sampled at times (N,). Velocities are derived
        from the track when not given (used for satellite Doppler)."""
        P = as_points(positions)
        t = np.asarray(times, dtype=float)
        if vel is None and len(P) > 1:
            vel = np.gradient(P, t, axis=0)
        return self.query_all(P, t, vel=vel, fading=fading, links=links)

    # ------------------------------------------------------------------ geometry helpers
    def ground_height(self, x, y):
        return self.surface.terrain_at(x, y)

    def surface_height(self, x, y):
        """Terrain + building height (a drone must stay above this)."""
        return self.surface.top_at(x, y)

    def agl(self, pos):
        return self.surface.agl(pos)

    def collides(self, pos):
        return self.surface.collision(pos)

    def in_map(self, x, y):
        return self.gd.grid.inside(np.asarray(x), np.asarray(y))

    def station_xyz(self, sid="gcs0"):
        return self.station(sid)["xyz"].copy()

    def bounds(self):
        g = self.gd.grid
        return dict(x=(g.x0, g.x1), y=(g.y0, g.y1))

    def local_to_latlon(self, pos):
        lat, lon, alt = self.gd.frame.enu_to_ll(as_points(pos))
        return lat, lon, alt

    def latlon_to_local(self, lat, lon, alt=0.0):
        return self.gd.frame.ll_to_enu(lat, lon, alt)

    # ------------------------------------------------------------------ maps
    def coverage_map(self, link_id: str, z_agl: Optional[float] = None, z_msl: Optional[float] = None, t: float = 0.0,
                     nx: int = 100, ny: int = 100, fading: str = "shadow", fields: Sequence[str] = ("sinr_db", "throughput_mbps")):
        """Raster of link quality on a plane (constant AGL or constant MSL altitude)."""
        g = self.gd.grid
        xs = np.linspace(g.x0 + g.res / 2, g.x1 - g.res / 2, nx)
        ys = np.linspace(g.y0 + g.res / 2, g.y1 - g.res / 2, ny)
        X, Y = np.meshgrid(xs, ys)
        if z_msl is not None:
            Z = np.full_like(X, z_msl)
        else:
            Z = self.surface.top_at(X, Y) + (z_agl if z_agl is not None else 100.0)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
        q = self.query(link_id, pts, t, fading=fading)
        out = dict(x=xs, y=ys, z=Z)
        for f in fields:
            v = getattr(q, f, None)
            if v is None:
                v = q.extra[f]
            out[f] = np.asarray(v).reshape(ny, nx)
        return out

    def describe(self) -> dict:
        c = self.cfg
        d = dict(area=c.area.name, geodata=self.gd.summary(), links={k: (v.kind, v.role, v.direction) for k, v in self._links.items()},
                 weather=c.weather.scenario, leaf_factor=round(self.surface.leaf_factor, 2),
                 interference_sources=self.interf.n, satellites=0 if self.constellation is None else self.constellation.count)
        if self.cellular:
            d["cell_sites"] = len(self.cellular.sites)
            d["cells_per_band"] = {b: len(v["site"]) for b, v in self.cellular._cells.items()}
        return d
