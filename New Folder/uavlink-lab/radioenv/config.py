"""Configuration schema. Every physical knob of the environment lives here and the whole
tree round-trips through JSON (`RadioEnvConfig.to_json / from_json`) so experiments are
fully reproducible: same config + same seed -> same radio world.
"""
from __future__ import annotations

import dataclasses as dc
import json
import typing
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- geography
@dc.dataclass
class AreaConfig:
    name: str = "demo-area"
    origin_lat: float = 45.75           # centre of the map (WGS-84)
    origin_lon: float = 126.63
    size_x_m: float = 4000.0            # East-West extent
    size_y_m: float = 4000.0            # North-South extent
    resolution_m: float = 10.0          # raster cell size
    start_time_utc: str = "2026-10-24T04:00:00"   # simulation t=0
    seed: int = 1


@dc.dataclass
class TerrainConfig:
    source: str = "synthetic"           # synthetic | file (.hgt/.asc/.npy/.tif)
    path: Optional[str] = None
    base_elevation_m: float = 150.0     # mean height above MSL
    relief_m: float = 60.0              # peak-to-peak-ish relief (synthetic)
    hurst: float = 0.85                 # roughness (higher = smoother)
    water_fraction: float = 0.03        # fraction of area flooded (lakes/rivers)


@dc.dataclass
class BuildingConfig:
    source: str = "synthetic"           # synthetic | geojson | none
    path: Optional[str] = None          # GeoJSON or Overpass JSON file
    block_m: float = 70.0
    street_m: float = 18.0
    lot_m: float = 25.0
    urban_core_radius_m: float = 1100.0
    n_subcenters: int = 3
    subcenter_radius_m: float = 450.0
    mean_height_m: float = 24.0
    max_height_m: float = 130.0
    rural_density: float = 0.015        # background building probability far from core
    fill: float = 0.9
    default_height_m: float = 10.0      # when real data has no height
    level_height_m: float = 3.2


@dc.dataclass
class VegetationConfig:
    source: str = "synthetic"           # synthetic | geojson | none
    path: Optional[str] = None
    coverage: float = 0.25
    canopy_mean_m: float = 12.0
    hurst: float = 0.6
    evergreen_fraction: float = 0.3     # share of canopy that is leaf-independent
    hemisphere: str = "N"


# --------------------------------------------------------------------------- propagation
@dc.dataclass
class PropagationConfig:
    k_factor: float = 4.0 / 3.0         # effective Earth radius factor
    profile_max_samples: int = 384
    fast_samples: int = 32              # profile samples for interferer paths
    max_diff_loss_db: float = 55.0
    wall_base_db: float = 12.0
    wall_db_per_m: float = 0.6
    wall_db_per_ghz: float = 2.0
    polarization: str = "V"             # V | H | C (rain + ground reflection)
    use_two_ray: bool = True
    weather_min_ghz: float = 3.0        # skip rain/cloud below this (negligible)
    ground_reflection_floor_db: float = -25.0
    obstruction_range_m: float = 1500.0  # ray length for satellite horizon checks


@dc.dataclass
class FadingConfig:
    shadow_sigma_los_db: float = 3.5
    shadow_sigma_nlos_db: float = 7.0
    shadow_dcorr_m: float = 45.0
    shadow_zdecorr_m: float = 60.0      # altitude over which shadowing decorrelates
    n_scatterers: int = 20
    scatter_doppler_hz: float = 3.0
    k_los_ground_db: float = 6.0        # Rician K at 0 deg elevation
    k_los_slope_db: float = 9.0         # extra K at 90 deg elevation


# --------------------------------------------------------------------------- weather
@dc.dataclass
class WeatherConfig:
    scenario: str = "clear"             # clear|light_rain|moderate_rain|heavy_rain|tropical_storm|custom
    rain_scale: float = 1.0
    bg_rain_mmh: float = 0.0            # custom: stratiform background
    cell_peak_mmh: float = 0.0          # custom: convective cell peak (median)
    n_cells: int = 8
    cell_radius_km: List[float] = dc.field(default_factory=lambda: [0.8, 3.5])
    cell_life_s: float = 1500.0
    cell_period_s: float = 5400.0
    wind_speed_ms: float = 8.0
    wind_to_deg: float = 90.0           # direction cells move TOWARD (0=N, 90=E)
    temperature_c: float = 8.0
    humidity_pct: float = 65.0
    freezing_height_m: float = 2500.0   # rain-top (0 C isotherm) above MSL
    cloud_cover: float = 0.3
    cloud_base_m: float = 1500.0
    cloud_top_m: float = 3200.0
    cloud_lwc_gm3: float = 0.25
    fog_density_gm3: float = 0.0        # 0.05 ~ 300 m visibility, 0.5 ~ 50 m
    fog_height_m: float = 80.0
    n_wet: float = 60.0                 # wet refractivity for scintillation


# --------------------------------------------------------------------------- cellular
@dc.dataclass
class BandConfig:
    id: str = "lte1800"
    tech: str = "LTE"
    freq_mhz: float = 1800.0
    bw_mhz: float = 20.0
    tx_power_dbm: float = 46.0          # per sector
    bs_gain_dbi: float = 17.0
    n_prb: int = 100
    ue_tx_dbm: float = 23.0
    ue_nf_db: float = 9.0
    bs_nf_db: float = 5.0
    hpbw_h_deg: float = 65.0
    hpbw_v_deg: float = 10.0
    sla_v_db: float = 28.0
    downtilt_deg: float = 6.0
    load_mean: float = 0.35
    coverage_fraction: float = 1.0      # share of sites that transmit this band
    ul_prb_alloc: int = 20
    ul_p0_dbm: float = -80.0
    ul_alpha: float = 0.8
    air_latency_ms: float = 12.0        # one-way scheduling+HARQ air interface
    core_latency_ms: float = 12.0       # backhaul + core to the GCS network
    harq_ms: float = 8.0
    max_se: float = 5.5
    se_alpha: float = 0.75


@dc.dataclass
class SiteConfig:
    id: str = "site0"
    x: float = 0.0
    y: float = 0.0
    height_agl_m: float = 30.0          # above local roof/ground
    azimuths_deg: List[float] = dc.field(default_factory=lambda: [0.0, 120.0, 240.0])
    lat: Optional[float] = None         # if given, overrides x/y
    lon: Optional[float] = None


@dc.dataclass
class CellularConfig:
    enabled: bool = True
    layout: str = "hex"                 # hex | list | opencellid
    isd_m: float = 750.0
    n_rings: int = 2
    jitter_m: float = 60.0
    site_height_agl_m: float = 30.0
    sites: List[SiteConfig] = dc.field(default_factory=list)
    opencellid_csv: Optional[str] = None
    model: str = "hybrid"               # hybrid | geometric | 3gpp
    environment: str = "auto"           # auto | UMa | RMa
    diurnal_load: bool = True
    k_neighbors: int = 6
    inter_site_shadow_corr: float = 0.5
    bands: List[BandConfig] = dc.field(
        default_factory=lambda: [
            BandConfig(id="lte1800"),
            BandConfig(id="nr3500", tech="NR", freq_mhz=3500.0, bw_mhz=60.0, tx_power_dbm=49.0,
                       bs_gain_dbi=20.0, n_prb=162, hpbw_v_deg=8.0, downtilt_deg=7.0,
                       coverage_fraction=0.7, load_mean=0.25, air_latency_ms=6.0, max_se=7.4),
        ]
    )


# --------------------------------------------------------------------------- satellite
@dc.dataclass
class ShellConfig:
    planes: int = 72
    sats_per_plane: int = 22
    inc_deg: float = 53.0
    alt_km: float = 550.0
    phasing: int = 1                    # Walker-delta F


@dc.dataclass
class SatelliteConfig:
    enabled: bool = True
    shell: ShellConfig = dc.field(default_factory=ShellConfig)
    extra_shells: List[ShellConfig] = dc.field(
        default_factory=lambda: [ShellConfig(planes=36, sats_per_plane=20, inc_deg=70.0, alt_km=570.0, phasing=1)]
    )
    tle_path: Optional[str] = None      # optional real TLEs (needs `sgp4`)
    j2: bool = True
    slot_s: float = 15.0                # serving-satellite re-selection slot
    guard_s: float = 0.1                # interruption at slot boundary
    min_elev_deg: float = 25.0
    n_candidates: int = 6
    dl_freq_ghz: float = 11.7
    ul_freq_ghz: float = 14.25
    dl_bw_mhz: float = 240.0
    ul_bw_mhz: float = 60.0
    sat_eirp_dbw: float = 38.0          # per-beam EIRP
    sat_gt_dbk: float = 9.0
    terminal_gain_dbi: float = 26.0     # boresight, small flat-panel on a drone
    terminal_tx_dbm: float = 33.0
    terminal_tsys_k: float = 300.0
    max_scan_deg: float = 60.0
    cci_dl_db: float = 18.0             # co-channel C/I from neighbouring beams
    cci_ul_db: float = 20.0
    impl_loss_db: float = 2.0
    pol_loss_db: float = 0.5
    sun_outage: bool = True
    gateway_latency_ms: float = 14.0    # extra gateway hop + terrestrial to GCS
    proc_latency_ms: float = 6.0
    sched_latency_ms: float = 10.0
    user_share: float = 0.25            # fraction of the beam capacity available to this terminal


# --------------------------------------------------------------------------- interference
@dc.dataclass
class WifiConfig:
    enabled: bool = True
    n_sources: int = 60
    region_radius_m: float = 1500.0
    eirp_dbm_24: float = 20.0
    eirp_dbm_58: float = 23.0
    frac_58: float = 0.4
    duty_mean: float = 0.18
    height_agl_m: List[float] = dc.field(default_factory=lambda: [3.0, 40.0])
    busy_bin_s: float = 2.0
    diurnal: bool = True
    bw_mhz: float = 20.0


@dc.dataclass
class UavTxConfig:
    """Other drones' video / C2 transmitters flying through the area."""
    enabled: bool = True
    n_drones: int = 4
    eirp_dbm: float = 27.0
    freqs_mhz: List[float] = dc.field(default_factory=lambda: [5658.0, 5695.0, 5732.0, 5769.0, 5806.0, 5843.0, 5880.0, 5917.0])
    bw_mhz: float = 18.0
    radius_m: List[float] = dc.field(default_factory=lambda: [200.0, 900.0])
    speed_ms: List[float] = dc.field(default_factory=lambda: [6.0, 18.0])
    alt_agl_m: List[float] = dc.field(default_factory=lambda: [20.0, 100.0])
    period_s: float = 420.0             # flight/charge cycle
    active_fraction: float = 0.65
    extra_c2_freqs_mhz: List[float] = dc.field(default_factory=lambda: [915.0, 2440.0])
    c2_bw_mhz: float = 1.0
    c2_eirp_dbm: float = 27.0


@dc.dataclass
class JammerConfig:
    id: str = "jam0"
    kind: str = "noise"                 # noise | narrowband | swept | pulsed
    x: float = 0.0
    y: float = 0.0
    z_agl_m: float = 3.0
    vx: float = 0.0
    vy: float = 0.0
    eirp_dbm: float = 33.0
    freq_mhz: float = 5800.0
    bw_mhz: float = 80.0
    sweep_range_mhz: float = 400.0
    sweep_period_s: float = 2.0
    pulse_period_s: float = 1.0
    duty: float = 0.5
    windows: List[List[float]] = dc.field(default_factory=lambda: [[0.0, 1e9]])  # active [t_on, t_off]


@dc.dataclass
class InterferenceConfig:
    wifi: WifiConfig = dc.field(default_factory=WifiConfig)
    uav: UavTxConfig = dc.field(default_factory=UavTxConfig)
    jammers: List[JammerConfig] = dc.field(default_factory=list)
    noise_environment: str = "auto"     # auto | city | residential | rural | quiet_rural
    max_range_m: float = 3000.0         # culling distance for ambient sources


# --------------------------------------------------------------------------- endpoints & links
@dc.dataclass
class GroundStationConfig:
    id: str = "gcs0"
    x: float = -1300.0
    y: float = -1000.0
    height_agl_m: float = 3.0           # above local roof/ground
    gain_dbi: float = 8.0
    pattern: str = "tracking"           # tracking (peak gain on drone) | omni (dipole)


@dc.dataclass
class LinkSpec:
    id: str = "link0"
    kind: str = "direct"                # direct | cellular | satellite
    role: str = "data"                  # control | video | data
    direction: str = "uplink"           # uplink: drone transmits; downlink: drone receives
    station: Optional[str] = "gcs0"     # direct links
    band: Optional[str] = None          # cellular links
    freq_mhz: float = 5800.0
    bw_mhz: float = 20.0
    tx_power_dbm: float = 25.0
    cable_loss_db: float = 1.0
    drone_gain_dbi: float = 2.0
    drone_pattern: str = "dipole"       # dipole | omni
    rx_nf_db: float = 6.0
    mode: str = "adaptive"              # fixed | adaptive
    rate_mbps: float = 8.0              # fixed-rate payload
    req_sinr_db: float = 2.0            # decode threshold (fixed) / minimum (adaptive)
    per_slope_db: float = 0.8
    se_alpha: float = 0.7
    se_max: float = 4.0
    interference_rx_gain_dbi: float = 0.0
    # service thresholds used for the convenience flags
    reliable_per: float = 0.01
    min_video_mbps: float = 3.0
    latency_limit_ms: float = 150.0
    base_latency_ms: float = 6.0
    retx_ms: float = 8.0
    margin_full_db: float = 20.0


def default_stations() -> List[GroundStationConfig]:
    return [GroundStationConfig()]


def default_links() -> List[LinkSpec]:
    return [
        LinkSpec(id="c2_915", kind="direct", role="control", direction="downlink", station="gcs0",
                 freq_mhz=915.0, bw_mhz=0.5, tx_power_dbm=30.0, drone_gain_dbi=2.0, rx_nf_db=5.0,
                 mode="fixed", rate_mbps=0.05, req_sinr_db=-3.0, per_slope_db=0.6, base_latency_ms=4.0,
                 retx_ms=5.0, interference_rx_gain_dbi=2.0, margin_full_db=15.0),
        LinkSpec(id="video_58", kind="direct", role="video", direction="uplink", station="gcs0",
                 freq_mhz=5800.0, bw_mhz=20.0, tx_power_dbm=27.0, drone_gain_dbi=2.0, rx_nf_db=6.0,
                 mode="adaptive", req_sinr_db=3.0, se_alpha=0.7, se_max=4.0, base_latency_ms=28.0,
                 retx_ms=10.0, min_video_mbps=4.0, interference_rx_gain_dbi=0.0),
        LinkSpec(id="cell_lte1800", kind="cellular", role="data", direction="uplink", band="lte1800",
                 freq_mhz=1800.0, bw_mhz=20.0, req_sinr_db=-5.0, mode="adaptive", min_video_mbps=3.0),
        LinkSpec(id="cell_nr3500", kind="cellular", role="data", direction="uplink", band="nr3500",
                 freq_mhz=3500.0, bw_mhz=60.0, req_sinr_db=-3.0, mode="adaptive", min_video_mbps=3.0),
        LinkSpec(id="cell_lte1800_dl", kind="cellular", role="data", direction="downlink", band="lte1800",
                 freq_mhz=1800.0, bw_mhz=20.0, req_sinr_db=-5.0, mode="adaptive", min_video_mbps=3.0),
        LinkSpec(id="cell_nr3500_dl", kind="cellular", role="data", direction="downlink", band="nr3500",
                 freq_mhz=3500.0, bw_mhz=60.0, req_sinr_db=-3.0, mode="adaptive", min_video_mbps=3.0),
        LinkSpec(id="sat_ku", kind="satellite", role="data", direction="uplink", freq_mhz=14250.0,
                 bw_mhz=60.0, req_sinr_db=2.0, mode="adaptive", se_alpha=0.75, se_max=4.5,
                 min_video_mbps=3.0, base_latency_ms=0.0),
    ]


@dc.dataclass
class RadioEnvConfig:
    area: AreaConfig = dc.field(default_factory=AreaConfig)
    terrain: TerrainConfig = dc.field(default_factory=TerrainConfig)
    buildings: BuildingConfig = dc.field(default_factory=BuildingConfig)
    vegetation: VegetationConfig = dc.field(default_factory=VegetationConfig)
    propagation: PropagationConfig = dc.field(default_factory=PropagationConfig)
    fading: FadingConfig = dc.field(default_factory=FadingConfig)
    weather: WeatherConfig = dc.field(default_factory=WeatherConfig)
    cellular: CellularConfig = dc.field(default_factory=CellularConfig)
    satellite: SatelliteConfig = dc.field(default_factory=SatelliteConfig)
    interference: InterferenceConfig = dc.field(default_factory=InterferenceConfig)
    stations: List[GroundStationConfig] = dc.field(default_factory=default_stations)
    links: List[LinkSpec] = dc.field(default_factory=default_links)

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return dc.asdict(self)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict) -> "RadioEnvConfig":
        return _build(cls, d)

    @classmethod
    def from_json(cls, path_or_str: str) -> "RadioEnvConfig":
        try:
            with open(path_or_str) as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = json.loads(path_or_str)
        return cls.from_dict(d)

    def replace(self, **dotted) -> "RadioEnvConfig":
        """Return a deep copy with dotted overrides: cfg.replace(**{"weather.scenario": "heavy_rain"})."""
        d = self.to_dict()
        for key, val in dotted.items():
            cur = d
            parts = key.split(".")
            for p in parts[:-1]:
                cur = cur[p]
            cur[parts[-1]] = val
        return RadioEnvConfig.from_dict(d)


def _coerce(t, v):
    origin = typing.get_origin(t)
    if dc.is_dataclass(t) and isinstance(v, dict):
        return _build(t, v)
    if origin in (list, List):
        (inner,) = typing.get_args(t) or (typing.Any,)
        return [_coerce(inner, x) for x in v]
    if origin is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]
        return None if v is None else _coerce(args[0], v)
    return v


def _build(cls, d):
    hints = typing.get_type_hints(cls)
    kw = {}
    for f in dc.fields(cls):
        if f.name in d:
            kw[f.name] = _coerce(hints[f.name], d[f.name])
    return cls(**kw)
