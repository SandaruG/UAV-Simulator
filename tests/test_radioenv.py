"""Physics + API verification for radioenv. Run:  python -m pytest -q tests   (or: python tests/test_radioenv.py)"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from radioenv import (JammerConfig, LinkSpec, RadioEnvConfig, RadioEnvironment)                     # noqa: E402
from radioenv.util import fspl_db, knife_edge_loss_db                                                # noqa: E402
from radioenv.weather import cloud_k_l, rain_specific_db_per_km                                      # noqa: E402


def stripped_cfg(**over):
    """Flat, empty, clear, quiet world: only free-space physics remains."""
    c = RadioEnvConfig()
    c.area.size_x_m = c.area.size_y_m = 3000.0
    c.terrain.relief_m = 0.0
    c.terrain.water_fraction = 0.0
    c.buildings.source = "none"
    c.vegetation.source = "none"
    c.weather.scenario = "clear"
    c.weather.cloud_cover = 0.0
    c.interference.wifi.enabled = False
    c.interference.uav.enabled = False
    c.propagation.use_two_ray = False
    c.interference.noise_environment = "quiet_rural"
    for k, v in over.items():
        c = c.replace(**{k: v})
    return c


_ENV = {}


def default_env():
    if "d" not in _ENV:
        _ENV["d"] = RadioEnvironment(RadioEnvConfig())
    return _ENV["d"]


# ------------------------------------------------------------------------------------ numerics
def test_fspl_and_knife_edge_reference_values():
    assert abs(float(fspl_db(1000.0, 5.8e9)) - 107.72) < 0.05
    assert abs(float(fspl_db(1.0, 1.0e9)) - 32.45) < 0.05              # classic 32.45 + 20log(f_MHz) + 20log(d_km)
    assert abs(float(knife_edge_loss_db(0.0)) - 6.0) < 0.1             # grazing incidence ~6 dB
    assert float(knife_edge_loss_db(-1.0)) == 0.0                      # well clear
    assert float(knife_edge_loss_db(2.4)) > 19.0


def test_itu_rain_and_cloud_magnitudes():
    # ITU-R P.838: 12 GHz @ 25 mm/h ~ 0.9-1.0 dB/km; 5.8 GHz is an order of magnitude lower
    assert 0.7 < float(rain_specific_db_per_km(12.0, 25.0, "V")) < 1.3
    assert float(rain_specific_db_per_km(14.25, 25.0)) > 8 * float(rain_specific_db_per_km(5.8, 25.0))
    assert 0.1 < cloud_k_l(12.0) < 0.2                                # (dB/km)/(g/m3)


# ------------------------------------------------------------------------------------ config
def test_config_roundtrip_and_overrides():
    c = RadioEnvConfig()
    c2 = RadioEnvConfig.from_json(c.to_json())
    assert c2.to_dict() == c.to_dict()
    c3 = c.replace(**{"weather.scenario": "heavy_rain", "area.seed": 9})
    assert c3.weather.scenario == "heavy_rain" and c3.area.seed == 9 and c.weather.scenario == "clear"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(c3.to_json())
    assert RadioEnvConfig.from_json(f.name).area.seed == 9


# ------------------------------------------------------------------------------------ free-space budget
def test_direct_link_matches_analytic_free_space_budget():
    env = RadioEnvironment(stripped_cfg())
    spec = env.link("video_58")
    gcs = env.station_xyz()
    for off, z in ((800.0, 160.0), (1200.0, 300.0)):
        p = np.array([gcs[0] + off, gcs[1] + 300.0, gcs[2] + z])
        q = env.query("video_58", p, 10.0, fading="none")
        d = np.linalg.norm(p - gcs)
        el = np.degrees(np.arctan2(gcs[2] - p[2], np.hypot(*(p - gcs)[:2])))
        from radioenv.propagation import dipole_gain_dbi
        g_dr = float(dipole_gain_dbi(el, peak=spec.drone_gain_dbi))
        gas = float(env.weather.gas_atten_path(gcs, p, spec.freq_mhz * 1e6)[0])
        exp = spec.tx_power_dbm - spec.cable_loss_db + g_dr + env.cfg.stations[0].gain_dbi - float(fspl_db(d, 5.8e9)) - gas
        assert abs(q.signal_dbm - exp) < 0.05, (q.signal_dbm, exp)
        assert q.los and abs(q.shadow_db) < 1e-9 and abs(q.fading_db) < 1e-9


def test_signal_falls_with_distance_in_free_space():
    env = RadioEnvironment(stripped_cfg())
    gcs = env.station_xyz()
    xs = np.linspace(100, 1400, 14)
    P = np.c_[gcs[0] + xs, np.full_like(xs, gcs[1]), np.full_like(xs, gcs[2] + 100)]
    d = np.hypot(xs, 100)
    env.link("video_58").drone_pattern = "omni"                         # isolate the propagation law
    q = env.query("video_58", P, 0.0, fading="none")
    assert np.all(np.diff(q.signal_dbm) < 0)
    assert abs(np.polyfit(np.log10(d), q.signal_dbm, 1)[0] + 20.0) < 0.1  # exactly -20 dB/decade in free space
    env.link("video_58").drone_pattern = "dipole"                       # dipole: gain rises as elevation falls
    qd = env.query("video_58", P, 0.0, fading="none")
    slope_d = np.polyfit(np.log10(d), qd.signal_dbm, 1)[0]
    assert -20.0 < slope_d < -14.0
    assert qd.extra["tx_gain_dbi"][-1] > qd.extra["tx_gain_dbi"][0] + 3.0


# ------------------------------------------------------------------------------------ determinism / API
def test_determinism_and_batch_consistency():
    cfg = RadioEnvConfig()
    a, b = RadioEnvironment(cfg), RadioEnvironment(cfg)
    rng = np.random.default_rng(3)
    P = np.c_[rng.uniform(-1500, 1500, (40, 2)), np.zeros(40)]
    P[:, 2] = a.surface_height(P[:, 0], P[:, 1]) + 90
    t = rng.uniform(0, 600, 40)
    for l in ("c2_915", "video_58", "cell_lte1800", "sat_ku"):
        qa = a.query(l, P, t)
        qb = b.query(l, P[::-1], t[::-1])                               # different order, other instance
        np.testing.assert_allclose(qa.sinr_db, qb.sinr_db[::-1], atol=1e-6, err_msg=l)
        i = 7                                                           # single-point query == element of batch
        qs = a.query(l, P[i], t[i])
        assert abs(qs.sinr_db - qa.sinr_db[i]) < 1e-6, l
    other = RadioEnvironment(cfg.replace(**{"area.seed": 2}))
    assert not np.allclose(other.gd.bldg, a.gd.bldg)


def test_scalar_and_vector_return_types():
    env = default_env()
    p = np.array([0.0, 0.0, 250.0])
    q = env.query("video_58", p, 1.0)
    assert np.ndim(q.sinr_db) == 0
    q2 = env.query_xyz("video_58", [0, 10, 20], [0, 0, 0], 250.0, 1.0)
    assert q2.sinr_db.shape == (3,)
    allq = env.query_all(p, 1.0)
    assert set(allq) == set(env.link_ids)
    d = q.to_dict()
    assert {"rssi_dbm", "snr_db", "sinr_db", "latency_ms", "throughput_mbps", "outage_prob"} <= set(d)


def test_track_query_and_maps():
    env = default_env()
    t = np.arange(0, 60.0, 1.0)
    P = np.c_[np.linspace(-800, 800, len(t)), np.linspace(-500, 500, len(t)), np.full(len(t), 260.0)]
    res = env.query_track(P, t)
    assert res["sat_ku"].sinr_db.shape == (len(t),)
    m = env.coverage_map("cell_lte1800", z_agl=100, nx=20, ny=20)
    assert m["sinr_db"].shape == (20, 20)


# ------------------------------------------------------------------------------------ weather
def test_rain_degrades_ku_satellite_not_much_at_900mhz():
    base = stripped_cfg()
    clear = RadioEnvironment(base)
    rain = RadioEnvironment(base.replace(**{"weather.scenario": "heavy_rain", "weather.cloud_cover": 0.0}))
    p = np.array([0.0, 0.0, 400.0])
    worst = 0.0
    for t in np.arange(0.0, 7200.0, 300.0):
        a, b = clear.query("sat_ku", p, t, fading="none"), rain.query("sat_ku", p, t, fading="none")
        worst = max(worst, a.sinr_db - b.sinr_db)
        assert b.extra["rain_db"] >= 0
    assert worst > 3.0
    l1, l2 = clear.query("c2_915", p, 100.0, fading="none"), rain.query("c2_915", p, 100.0, fading="none")
    assert abs(l1.signal_dbm - l2.signal_dbm) < 0.1


def test_rain_field_statistics_and_advection():
    env = RadioEnvironment(RadioEnvConfig().replace(**{"weather.scenario": "moderate_rain"}))
    w = env.weather
    xx, yy = np.meshgrid(np.linspace(-3000, 3000, 61), np.linspace(-3000, 3000, 61))
    r = w.rain_rate(xx, yy, np.full_like(xx, 1200.0))
    assert r.min() >= 0 and r.max() > r.mean() > 0
    assert np.allclose(w.rain_rate(xx, yy, np.full_like(xx, 1200.0)), r)          # deterministic
    r_later = w.rain_rate(xx, yy, np.full_like(xx, 1260.0))
    assert not np.allclose(r, r_later)                                             # evolving


# ------------------------------------------------------------------------------------ interference
def test_jammer_window_and_frequency_selectivity():
    cfg = stripped_cfg()
    gx, gy = cfg.stations[0].x, cfg.stations[0].y
    cfg.interference.jammers = [JammerConfig(id="j_in", kind="noise", x=gx + 150, y=gy, freq_mhz=5800, bw_mhz=80,
                                             eirp_dbm=30, windows=[[100.0, 200.0]]),
                                JammerConfig(id="j_out", kind="noise", x=gx + 150, y=gy, freq_mhz=2440, bw_mhz=80,
                                             eirp_dbm=30, windows=[[300.0, 400.0]])]
    env = RadioEnvironment(cfg)
    g = env.station_xyz()
    p = np.array([g[0] + 900, g[1], g[2] + 120])
    q = lambda t: env.query("video_58", p, t, fading="none")
    quiet, jam, after, offband = q(50.0), q(150.0), q(250.0), q(350.0)
    assert jam.interference_dbm - quiet.interference_dbm > 25 or quiet.interference_dbm < -150
    assert jam.sinr_db < quiet.sinr_db - 15
    assert abs(after.sinr_db - quiet.sinr_db) < 0.1
    assert abs(offband.sinr_db - quiet.sinr_db) < 0.5                    # 2.44 GHz jammer vs 5.8 GHz victim


def test_pulsed_jammer_duty_cycle():
    cfg = stripped_cfg()
    gx, gy = cfg.stations[0].x, cfg.stations[0].y
    cfg.interference.jammers = [JammerConfig(id="jp", kind="pulsed", x=gx + 100, y=gy, freq_mhz=5800, bw_mhz=40,
                                             eirp_dbm=27, pulse_period_s=1.0, duty=0.3)]
    env = RadioEnvironment(cfg)
    g = env.station_xyz()
    p = np.array([g[0] + 700, g[1], g[2] + 80])
    t = np.arange(0, 20.0, 0.05)
    q = env.query("video_58", np.tile(p, (len(t), 1)), t, fading="none")
    on = q.interference_dbm > -100
    assert 0.25 < on.mean() < 0.35


def test_wifi_and_uav_sources_are_bursty_and_in_band():
    env = default_env()
    tab = env.interf.table()
    assert any(s["kind"] == "wifi" for s in tab) and any(s["kind"] == "uav" for s in tab)
    g = env.station_xyz()
    P = np.tile(g, (200, 1))
    t = np.linspace(0, 400, 200)
    I = env.interf.power_dbm(P, t, 2.437e9, 20e6)
    assert I.std() > 3.0                                                 # time-varying, not constant


# ------------------------------------------------------------------------------------ satellite
def test_satellite_geometry_and_slots():
    env = default_env()
    C = env.constellation
    obs = env.gd.frame.enu_to_ecef(np.array([[0.0, 0.0, 200.0]]))
    ts = np.arange(0, 1800, 120.0)
    nv = C.count_visible(np.repeat(obs, len(ts), 0), ts, 25.0)
    assert 8 <= nv.min() and nv.max() <= 60
    r = np.linalg.norm(C.ecef(np.array([0.0])), axis=-1)
    assert abs(r.min() - 6928137.0) < 1.0                                # 550 km shell
    p = np.array([0.0, 0.0, 250.0])
    t = np.arange(0.0, 60.0, 0.1)
    q = env.query("sat_ku", np.tile(p, (len(t), 1)), t, fading="none")
    assert np.all(q.extra["sat_elev_deg"][q.extra["sat_index"] >= 0] >= 25.0 - 1e-6)
    interrupt = q.extra["handover_interruption"]
    # interruption only in the first guard_s of each slot; ~0.1 s / 15 s of the time
    assert 0.003 < interrupt.mean() < 0.012
    assert not np.any(q.available[interrupt])
    c = 299792458.0
    assert np.all(q.latency_ms[q.extra["sat_index"] >= 0] >= 2 * q.extra["range_km"][q.extra["sat_index"] >= 0] * 1e3 / c * 1e3)


def test_satellite_blocked_by_buildings_in_street_canyon():
    env = default_env()
    rng = np.random.default_rng(5)
    xy = rng.uniform(-900, 900, (300, 2))
    street = np.c_[xy, env.surface.top_at(xy[:, 0], xy[:, 1]) + 1.5]   # 1.5 m above the surface (urban canyon / roofs)
    air = np.c_[xy, env.surface.top_at(xy[:, 0], xy[:, 1]) + 150.0]
    qs, qa = env.query("sat_ku", street, 40.0), env.query("sat_ku", air, 40.0)
    assert np.mean(qa.available) > np.mean(qs.available)
    assert np.mean(qs.sinr_db) < np.mean(qa.sinr_db) - 1.0


# ------------------------------------------------------------------------------------ cellular
def test_cellular_serving_is_strongest_and_aerial_dl_sinr_degrades():
    env = default_env()
    rng = np.random.default_rng(11)
    xy = rng.uniform(-1200, 1200, (300, 2))
    meds = []
    for h in (2, 100):
        P = np.c_[xy, env.surface_height(xy[:, 0], xy[:, 1]) + h]
        q = env.query("cell_lte1800_dl", P, 300.0, fading="shadow")
        assert np.all(q.extra["serving_cell"] == q.extra["best_cell"])
        meds.append(np.median(q.sinr_db))
    assert meds[1] < meds[0]                                               # known aerial-UE interference problem


def test_cell_load_is_diurnal():
    env = default_env()
    t = np.arange(0, 86400, 3600.0)
    L = env.cellular.load("lte1800", t).mean(1)
    assert L.max() / L.min() > 1.5


# ------------------------------------------------------------------------------------ real-data loaders
def test_real_building_and_dem_loaders():
    c = RadioEnvConfig()
    c.area.size_x_m = c.area.size_y_m = 1000.0
    c.area.resolution_m = 5.0
    fr = RadioEnvironment(c).gd.frame
    lat, lon = fr.enu_to_ll(np.array([[100.0, 100.0, 0.0]]))[:2]
    lat2, lon2 = fr.enu_to_ll(np.array([[160.0, 150.0, 0.0]]))[:2]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"building:levels": "10"},
          "geometry": {"type": "Polygon", "coordinates": [[[lon[0], lat[0]], [lon2[0], lat[0]], [lon2[0], lat2[0]], [lon[0], lat2[0]], [lon[0], lat[0]]]]}}]}
    with tempfile.TemporaryDirectory() as d:
        bp = os.path.join(d, "b.geojson")
        json.dump(gj, open(bp, "w"))
        # ESRI ASCII DEM: 100 m plateau to 200 m
        n = 40
        cs = 0.0005
        lat0, lon0 = c.area.origin_lat - n * cs / 2, c.area.origin_lon - n * cs / 2
        with open(os.path.join(d, "dem.asc"), "w") as f:
            f.write(f"ncols {n}\nnrows {n}\nxllcorner {lon0}\nyllcorner {lat0}\ncellsize {cs}\nNODATA_value -9999\n")
            for r in range(n):
                f.write(" ".join(f"{100 + 5 * (n - 1 - r):.1f}" for _ in range(n)) + "\n")
        c.buildings.source, c.buildings.path = "geojson", bp
        c.terrain.source, c.terrain.path = "file", os.path.join(d, "dem.asc")
        env = RadioEnvironment(c)
        assert abs(env.surface.bldg_at(130.0, 125.0) - 32.0) < 1e-6        # 10 levels * 3.2 m
        assert env.surface.bldg_at(-300.0, -300.0) == 0
        assert env.ground_height(0, 400) > env.ground_height(0, -400) + 10   # north is higher in the DEM


def _wall_env(wall_x, H, st_h_agl, res=2.0):
    """Flat empty world with thin (8 m) N-S walls of height H at the given x positions; GCS at x=-1000."""
    c = stripped_cfg()
    c.area.resolution_m = res
    fr = RadioEnvironment(c).gd.frame
    feats = []
    for x in wall_x:
        (la0, lo0), (la1, lo1) = [(a, b) for a, b in zip(*fr.enu_to_ll(np.array([[x - 4, -200, 0.0], [x + 4, 200, 0.0]]))[:2])]
        feats.append({"type": "Feature", "properties": {"height": str(H)}, "geometry": {"type": "Polygon", "coordinates": [[
            [lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]]}})
    d = tempfile.mkdtemp()
    path = os.path.join(d, "walls.json")
    json.dump({"type": "FeatureCollection", "features": feats}, open(path, "w"))
    c = c.replace(**{"buildings.source": "geojson", "buildings.path": path})
    c.stations[0].x, c.stations[0].y, c.stations[0].height_agl_m = -1000.0, 0.0, st_h_agl
    return RadioEnvironment(c)


def test_single_knife_edge_matches_itu_p526():
    """Engine diffraction loss == J(v) for one thin screen, from clear through grazing to obstructed."""
    from radioenv.propagation import terrestrial_path
    from radioenv.util import C_LIGHT, R_EARTH
    f = 2.4e9
    lam = C_LIGHT / f
    H = 30.0
    for h_target in (-3.0, 0.0, 3.0, 6.0, 10.0):         # edge height above the direct line (m)
        env = _wall_env([0.0], H, H - h_target)
        g = env.station_xyz()
        P = np.array([[1000.0, 0.0, g[2]]])
        r = terrestrial_path(env.surface, g, P, f, env.cfg.propagation)
        h = env.surface.top_at(0.0, 0.0) + 1000.0 * 1000.0 / (2 * env.cfg.propagation.k_factor * R_EARTH) - g[2]
        exp = float(knife_edge_loss_db(h * np.sqrt(2 * 2000.0 / (lam * 1000.0 * 1000.0))))
        assert abs(r["loss_diff"][0] - exp) < 0.05, (h_target, r["loss_diff"][0], exp)


def test_two_edges_follow_deygout_not_double_counting_cells():
    from radioenv.propagation import terrestrial_path
    from radioenv.util import C_LIGHT
    f, H = 2.4e9, 30.0
    lam = C_LIGHT / f
    env = _wall_env([-300.0, 300.0], H, H - 8.0)           # both screens 8 m above the direct line
    g = env.station_xyz()
    P = np.array([[1000.0, 0.0, g[2]]])
    r = terrestrial_path(env.surface, g, P, f, env.cfg.propagation)
    top, z0, D = env.surface.top_at(-300.0, 0.0), g[2], 2000.0
    v1 = (top - z0) * np.sqrt(2 * D / (lam * 700.0 * 1300.0))
    # sub-path: first screen top -> Rx, second screen 600 m / 700 m along that 1300 m path
    z_line = top + 600.0 / 1300.0 * (z0 - top)
    v2 = (top - z_line) * np.sqrt(2 * 1300.0 / (lam * 600.0 * 700.0))
    exp = float(knife_edge_loss_db(v1) + knife_edge_loss_db(v2))
    single = _wall_env([0.0], H, H - 8.0)
    gs = single.station_xyz()
    r1 = terrestrial_path(single.surface, gs, np.array([[1000.0, 0.0, gs[2]]]), f, single.cfg.propagation)
    assert r["loss_diff"][0] > r1["loss_diff"][0] + 3.0
    assert abs(r["loss_diff"][0] - exp) < 0.6, (r["loss_diff"][0], exp)


def test_all_presets_build_and_answer_every_link():
    from radioenv import presets
    for name in presets.names():
        env = RadioEnvironment(presets.get(name, seed=2, weather="light_rain"))
        b = env.bounds()
        p = np.array([0.5 * b["x"][0], 0.5 * b["y"][0], 0.0])
        p[2] = float(env.surface_height(p[0], p[1])) + 80.0
        for lid, q in env.query_all(p, 30.0).items():
            assert np.isfinite(q.sinr_db) and np.isfinite(q.latency_ms) and 0.0 <= q.per <= 1.0, (name, lid)


def test_collision_and_agl_helpers():
    env = default_env()
    xs, ys = np.meshgrid(np.linspace(-500, 500, 50), np.linspace(-500, 500, 50))
    i = np.argmax(env.surface.bldg_at(xs, ys))
    x, y = xs.ravel()[i], ys.ravel()[i]
    top = env.surface_height(x, y)
    assert env.collides(np.array([x, y, top - 2.0]))[0] and not env.collides(np.array([x, y, top + 2.0]))[0]
    assert abs(env.agl(np.array([x, y, top + 10.0]))[0] - 10.0) < 1e-6


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback
                print("FAIL", name, "->", repr(e))
                traceback.print_exc(limit=3)
    sys.exit(1 if fails else 0)
