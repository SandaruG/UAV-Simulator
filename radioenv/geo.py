"""Geodesy, raster grid, and real-world data loaders (DEM, OSM/GeoJSON buildings,
land cover, OpenCelliD towers). Everything works with numpy only; rasterio is used
opportunistically for GeoTIFF.
"""
from __future__ import annotations

import csv
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


# ----------------------------------------------------------------------------- geodesy
def geodetic_to_ecef(lat, lon, alt):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    alt = np.asarray(alt, dtype=float)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], -1)


def ecef_to_geodetic(xyz):
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


class LocalFrame:
    """East-North-Up tangent frame at (lat0, lon0). z is metres above the tangent plane
    (treated as height above MSL for terrain purposes; the curvature error is <8 m at 10 km)."""

    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0):
        self.lat0, self.lon0, self.alt0 = lat0, lon0, alt0
        la, lo = math.radians(lat0), math.radians(lon0)
        self.east = np.array([-math.sin(lo), math.cos(lo), 0.0])
        self.north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
        self.up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
        self.R = np.stack([self.east, self.north, self.up])      # rows = ENU axes in ECEF
        self.origin_ecef = geodetic_to_ecef(lat0, lon0, alt0)

    def enu_to_ecef(self, enu):
        return self.origin_ecef + np.asarray(enu, dtype=float) @ self.R

    def ecef_to_enu(self, ecef):
        return (np.asarray(ecef, dtype=float) - self.origin_ecef) @ self.R.T

    def ll_to_enu(self, lat, lon, alt=0.0):
        return self.ecef_to_enu(geodetic_to_ecef(lat, lon, alt))

    def enu_to_ll(self, enu):
        lat, lon, alt = ecef_to_geodetic(self.enu_to_ecef(enu))
        return lat, lon, alt

    def vec_enu_to_ecef(self, v):
        """Rotate a free vector (velocity) from ENU into ECEF."""
        return np.asarray(v, dtype=float) @ self.R


# ----------------------------------------------------------------------------- raster grid
@dataclass
class Grid:
    x0: float          # west edge (m)
    y0: float          # south edge (m)
    res: float
    nx: int
    ny: int

    @property
    def x1(self):
        return self.x0 + self.nx * self.res

    @property
    def y1(self):
        return self.y0 + self.ny * self.res

    def xs(self):
        return self.x0 + (np.arange(self.nx) + 0.5) * self.res

    def ys(self):
        return self.y0 + (np.arange(self.ny) + 0.5) * self.res

    def inside(self, x, y):
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)

    def bilinear(self, arr, x, y):
        fx = np.clip((np.asarray(x, dtype=float) - self.x0) / self.res - 0.5, 0, self.nx - 1.000001)
        fy = np.clip((np.asarray(y, dtype=float) - self.y0) / self.res - 0.5, 0, self.ny - 1.000001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        a = arr[iy, ix] * (1 - tx) + arr[iy, ix + 1] * tx
        b = arr[iy + 1, ix] * (1 - tx) + arr[iy + 1, ix + 1] * tx
        return a * (1 - ty) + b * ty

    def nearest(self, arr, x, y):
        ix = np.clip(((np.asarray(x, dtype=float) - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y, dtype=float) - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return arr[iy, ix]


def rasterize_polygon(grid: Grid, ring_xy: np.ndarray, out: np.ndarray, value: float, mode: str = "max"):
    """Even-odd scan fill of one polygon ring (Nx2, metres) into `out` (ny,nx)."""
    xs, ys = ring_xy[:, 0], ring_xy[:, 1]
    i0 = max(0, int(math.floor((xs.min() - grid.x0) / grid.res)))
    i1 = min(grid.nx - 1, int(math.ceil((xs.max() - grid.x0) / grid.res)))
    j0 = max(0, int(math.floor((ys.min() - grid.y0) / grid.res)))
    j1 = min(grid.ny - 1, int(math.ceil((ys.max() - grid.y0) / grid.res)))
    if i1 < i0 or j1 < j0:
        return
    cx = grid.x0 + (np.arange(i0, i1 + 1) + 0.5) * grid.res
    cy = grid.y0 + (np.arange(j0, j1 + 1) + 0.5) * grid.res
    X, Y = np.meshgrid(cx, cy)
    inside = np.zeros(X.shape, dtype=bool)
    n = len(xs)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(n):
            xa, ya, xb, yb = xs[k], ys[k], xs[(k + 1) % n], ys[(k + 1) % n]
            cond = (ya > Y) != (yb > Y)
            xint = (xb - xa) * (Y - ya) / (yb - ya + 1e-12) + xa
            inside ^= cond & (X < xint)
    if not inside.any():       # sub-cell polygon: mark the cell holding its centroid
        ci = int((xs.mean() - grid.x0) / grid.res)
        cj = int((ys.mean() - grid.y0) / grid.res)
        if 0 <= ci < grid.nx and 0 <= cj < grid.ny:
            out[cj, ci] = max(out[cj, ci], value) if mode == "max" else value
        return
    sub = out[j0:j1 + 1, i0:i1 + 1]
    if mode == "max":
        sub[inside] = np.maximum(sub[inside], value)
    else:
        sub[inside] = value


# ----------------------------------------------------------------------------- DEM loaders
@dataclass
class LatLonRaster:
    arr: np.ndarray          # row 0 = north
    lat_top: float
    lon_left: float
    dlat: float              # negative (north -> south)
    dlon: float

    def sample(self, lat, lon):
        fy = (np.asarray(lat) - self.lat_top) / self.dlat
        fx = (np.asarray(lon) - self.lon_left) / self.dlon
        fy = np.clip(fy, 0, self.arr.shape[0] - 1.000001)
        fx = np.clip(fx, 0, self.arr.shape[1] - 1.000001)
        iy, ix = fy.astype(int), fx.astype(int)
        ty, tx = fy - iy, fx - ix
        a = self.arr[iy, ix] * (1 - tx) + self.arr[iy, ix + 1] * tx
        b = self.arr[iy + 1, ix] * (1 - tx) + self.arr[iy + 1, ix + 1] * tx
        return a * (1 - ty) + b * ty


def read_hgt(path: str) -> LatLonRaster:
    """SRTM .hgt tile (1 or 3 arc-second); tile origin parsed from the file name (N45E126.hgt)."""
    name = os.path.basename(path).upper()
    lat0 = int(name[1:3]) * (1 if name[0] == "N" else -1)
    lon0 = int(name[4:7]) * (1 if name[3] == "E" else -1)
    n = int(round(math.sqrt(os.path.getsize(path) / 2)))
    arr = np.fromfile(path, dtype=">i2").reshape(n, n).astype(float)
    arr[arr < -1000] = np.nan
    if np.isnan(arr).any():
        arr[np.isnan(arr)] = np.nanmean(arr)
    return LatLonRaster(arr, lat0 + 1.0, float(lon0), -1.0 / (n - 1), 1.0 / (n - 1))


def read_esri_ascii(path: str) -> LatLonRaster:
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
    arr = np.loadtxt(path, skiprows=6)
    arr[arr == hdr.get("nodata_value", -9999)] = np.nan
    arr[np.isnan(arr)] = np.nanmean(arr)
    cs = hdr["cellsize"]
    lat_top = hdr["yllcorner"] + hdr["nrows"] * cs - cs / 2
    lon_left = hdr["xllcorner"] + cs / 2
    return LatLonRaster(arr, lat_top, lon_left, -cs, cs)


def read_geotiff(path: str) -> LatLonRaster:
    try:
        import rasterio
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading GeoTIFF needs `pip install rasterio`") from e
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(float)
        t = ds.transform
        arr[arr == ds.nodata] = np.nan
        arr[np.isnan(arr)] = np.nanmean(arr)
        return LatLonRaster(arr, t.f + t.e / 2, t.c + t.a / 2, t.e, t.a)


def load_dem(path: str) -> LatLonRaster:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hgt":
        return read_hgt(path)
    if ext in (".asc", ".txt"):
        return read_esri_ascii(path)
    if ext in (".tif", ".tiff"):
        return read_geotiff(path)
    raise ValueError(f"unsupported DEM format: {ext}")


# ----------------------------------------------------------------------------- vector loaders
def _rings_from_geometry(geom: dict) -> List[np.ndarray]:
    t = geom.get("type")
    if t == "Polygon":
        return [np.asarray(geom["coordinates"][0], dtype=float)[:, :2]]
    if t == "MultiPolygon":
        return [np.asarray(p[0], dtype=float)[:, :2] for p in geom["coordinates"]]
    return []


def load_vector_features(path_or_dict) -> List[Tuple[np.ndarray, dict]]:
    """Return [(ring[lon,lat], properties)] from GeoJSON or Overpass JSON (`out geom;`)."""
    if isinstance(path_or_dict, (str, os.PathLike)):
        with open(path_or_dict) as f:
            data = json.load(f)
    else:
        data = path_or_dict
    feats: List[Tuple[np.ndarray, dict]] = []
    if data.get("type") == "FeatureCollection":
        for ft in data["features"]:
            for ring in _rings_from_geometry(ft.get("geometry") or {}):
                feats.append((ring, ft.get("properties") or {}))
    elif "elements" in data:      # Overpass
        for el in data["elements"]:
            if el.get("type") == "way" and "geometry" in el:
                ring = np.array([[p["lon"], p["lat"]] for p in el["geometry"]], dtype=float)
                if len(ring) >= 3:
                    feats.append((ring, el.get("tags") or {}))
    return feats


def building_height_from_props(props: dict, default: float, level_h: float) -> float:
    for k in ("height", "building:height", "render_height"):
        if k in props:
            try:
                return float(str(props[k]).replace("m", "").strip())
            except ValueError:
                pass
    for k in ("building:levels", "levels"):
        if k in props:
            try:
                return float(props[k]) * level_h
            except ValueError:
                pass
    return default


def fetch_overpass(bbox, what: str = "building", timeout: int = 60, url: str = "https://overpass-api.de/api/interpreter"):
    """Download OSM ways with geometry for bbox=(south, west, north, east). Needs network access."""
    s, w, n, e = bbox
    q = f"[out:json][timeout:{timeout}];way[\"{what}\"]({s},{w},{n},{e});out geom;"
    req = urllib.request.Request(url, data=q.encode(), headers={"User-Agent": "radioenv/1.0"})
    with urllib.request.urlopen(req, timeout=timeout + 10) as r:
        return json.load(r)


def load_opencellid(path: str, bbox=None):
    """OpenCelliD CSV -> list of dict(lat, lon, radio, cell). bbox=(south,west,north,east)."""
    out = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        for row in rd:
            try:
                lon, lat = float(row[6]), float(row[7])
            except (ValueError, IndexError):
                continue
            if bbox and not (bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]):
                continue
            out.append(dict(lat=lat, lon=lon, radio=row[0], cell=row[4]))
    return out
