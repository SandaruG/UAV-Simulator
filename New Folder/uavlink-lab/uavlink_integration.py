"""Adapters between the selectable radio catalog, both simulators, and Part 3 policy."""
from __future__ import annotations

import copy
import dataclasses
import importlib
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import drone_multilink as drone
import predictive_handover as handover


LEGACY_TECH_TO_RADIO = {
    "analog_fpv_58": "fpv58_analog",
    "digital_58_dji": "ocusync",
    "digital_58_openhd": "wifi",
    "ism_24_ctrl": "elrs24",
    "ism_24_wifi": "rf24",
    "lr_12": "lr12g",
    "sub_868_900": "rf900",
    "band_17": "lr17g",
    "band_33": "lr33g",
    "band_72": "lr72g",
    "cell_lte": "cell",
    "cell_5g": "nr_a",
    "cell_5g_multisim": "sim_bond",
    "sat_leo_starlink": "leo",
    "sat_kuka_leo": "oneweb_ku",
    "sat_iridium": "iridium_certus",
    "sat_sband_leo": "globalstar_s",
    "sat_lband_geo": "inmarsat_l",
    "sat_kuka_geo": "geo_ku",
    "mesh_rf": "mesh_manet",
    "fiber_tether": "fiber_tether",
}

RADIO_ALIASES = {
    **{legacy: radio for legacy, radio in LEGACY_TECH_TO_RADIO.items()},
    "wifi_5g": "wifi",
    "lte": "cell",
    "sub1ghz": "rf900",
    "rf_24": "rf24",
    "sat_leo": "leo",
    "cell_lte": "cell",
    "cell_5g": "nr_a",
    "cell_5g_multisim": "sim_bond",
    "sat_oneweb_ku": "oneweb_ku",
    "sat_iridium": "iridium_certus",
    "sat_sband_leo": "globalstar_s",
    "sat_lband_geo": "inmarsat_l",
    "sat_kuka_geo": "geo_ku",
    "mesh_rf": "mesh_manet",
}


def canonical_radio_name(name: str, aliases: Optional[Dict[str, str]] = None) -> str:
    table = dict(RADIO_ALIASES)
    if aliases:
        table.update(aliases)
    seen = set()
    while name in table and table[name] != name:
        if name in seen:
            raise ValueError(f"radio alias cycle involving {name!r}")
        seen.add(name)
        name = table[name]
    valid = {r.name for r in drone.radio_catalog()}
    if name not in valid:
        raise ValueError(f"unknown radio technology {name!r}; valid names: {sorted(valid)}")
    return name


def reverse_radio_aliases() -> Dict[str, List[str]]:
    reverse: Dict[str, List[str]] = {}
    for alias, name in RADIO_ALIASES.items():
        reverse.setdefault(name, []).append(alias)
    return reverse


def radio_group(rp: drone.RadioParams) -> str:
    if rp.kind == "sat":
        return "satellite"
    if rp.kind == "cellular" or rp.kind == "bond":
        return "cellular"
    if rp.kind == "mesh":
        return "mesh"
    if rp.kind == "fiber":
        return "fiber"
    if rp.analog_video and rp.freq_hz > 4e9:
        return "fpv_58"
    if rp.freq_hz < 1e9:
        return "sub_ghz"
    if rp.freq_hz < 1.45e9:
        return "lr_12"
    if 2.0e9 <= rp.freq_hz < 2.7e9:
        return "ism_24"
    if rp.freq_hz >= 4.0e9 and rp.freq_hz < 6.5e9:
        return "digital_58"
    return "other_bands"


def radio_family(rp: drone.RadioParams) -> str:
    if rp.kind in ("sat", "cellular", "bond", "mesh", "fiber"):
        return rp.kind
    if 5.0e9 <= rp.freq_hz < 6.5e9:
        return "slot58"
    if 2.0e9 <= rp.freq_hz < 2.7e9:
        return "slot24"
    if 1.0e9 <= rp.freq_hz < 1.45e9:
        return "slot12"
    if rp.freq_hz < 1.0e9:
        return "slot9"
    return f"band_{round(rp.freq_hz / 1e8)}"


def generic_fast_profile(rp: drone.RadioParams, sim_module) -> Dict[str, Any]:
    """Approximate a drone radio as a LinkCfg for fast-mode screening only."""
    rates = [max(float(rate), 0.05) for _, rate in rp.mcs_table]
    if not rates:
        rates = [0.25, 0.5, 1.0]
    tx_w = 10.0 ** (rp.tx_power_dbm / 10.0) / 1000.0
    pa_eff = max(float(rp.pa_eff_max), 0.05)
    peak_active_w = float(rp.p_idle_w + rp.tx_overhead_w)
    link = dataclasses.asdict(sim_module.LinkCfg())
    link.update(
        tech=rp.family or rp.kind,
        freq_hz=float(rp.freq_hz),
        bw_hz=float(rp.bw_hz),
        dl_tx_dbm=float(rp.peer_tx_power_dbm),
        ul_tx_dbm=float(rp.tx_power_dbm),
        gs_gain_dbi=float(rp.peer_gain_dbi),
        drone_gain_dbi=max([float(a.peak_gain_dbi) for a in rp.antennas] or [2.0]),
        ant_pattern="omni" if rp.kind in ("sat", "fiber") else "dipole",
        nf_db=float(rp.noise_figure_db),
        pl_exp=2.2,
        mcs_eff=rates,
        ctrl_eff=min(rates),
        mac_eff=float(rp.mac_efficiency),
        base_latency_ms=float(rp.lat_dl_ms + rp.backhaul_ms),
        jitter_ms=float(rp.lat_jitter_ms + rp.backhaul_jitter_ms),
        mac_delay_ms=float(rp.hop_ms),
        retx_delay_ms=float(rp.retx_delay_ms),
        max_attempts=int(rp.max_tx),
        activation_delay_s=float(rp.boot_time_s + rp.search_min_s),
        active_w=peak_active_w,
        warm_w=float(rp.p_standby_w),
        pa_eff=pa_eff,
        carries_video=not rp.analog_video and rp.max_video_bps > 0.0,
    )
    if rp.analog_video:
        link.update(mcs_eff=[0.8, 1.2], ctrl_eff=0.8, carries_video=True)
    control_only = rp.name in ("elrs24", "crossfire868", "rf900")
    if control_only:
        link["carries_video"] = False
    if rp.kind == "fiber":
        link.update(tech="fiber", freq_hz=100e6, pl_exp=2.0, base_latency_ms=0.8, jitter_ms=0.1)
    options: Dict[str, Any] = {}
    if rp.kind == "sat":
        constellation = {
            "leo": (550.0, 45.0),
            "oneweb_ku": (1200.0, 40.0),
            "iridium": (780.0, 40.0),
            "globalstar": (1414.0, 30.0),
            "geo": (35786.0, 70.0),
        }.get(rp.constellation, (550.0, 45.0))
        options = dict(orbit_km=constellation[0], elevation_deg=constellation[1], azimuth_deg=180.0)
    elif rp.kind == "fiber":
        options = dict(length_m=rp.tether_len_m or 1500.0, kg_per_km=0.15, anchor_height_m=2000.0)
    elif rp.kind == "mesh":
        options = dict(relay_spacing_m=800.0, max_relays=12, relay_height_m=12.0)
    for key in ("name", "station_ids", "carries_video"):
        link.pop(key, None)
    return dict(
        link=link,
        group=radio_group(rp),
        family=radio_family(rp),
        stations={"direct": "gcs", "wifi": "gcs", "cellular": "cell", "sat": "sat", "mesh": "relay",
                  "fiber": "tether", "bond": "cell"}.get(rp.kind, "gcs"),
        control=not rp.analog_video,
        video=bool(rp.analog_video or not control_only),
        mass_kg={"sat": 0.4, "cellular": 0.08, "bond": 0.12, "mesh": 0.2, "fiber": 0.05}.get(rp.kind, 0.08),
        options=options,
        notes=f"Fast-mode approximation of drone_multilink radio family {rp.family or rp.kind}.",
    )


def _sat_constellation(rp: drone.RadioParams) -> drone.ConstellationParams:
    if rp.constellation in drone.SimConfig().env.extra_constellations:
        return copy.deepcopy(drone.SimConfig().env.extra_constellations[rp.constellation])
    if rp.constellation == "geo":
        return drone.ConstellationParams(1, 3, 35786.0, 0.0, 0, 360.0, (95.0, 125.0, 155.0))
    return copy.deepcopy(drone.SimConfig().env.constellation)


def _native_policy(name: str, params: Dict[str, Any]):
    policy_name = name.rsplit(":", 1)[-1].lower()
    if policy_name in ("reactive", "reactivepolicy"):
        return handover.ReactivePolicy(**{k: v for k, v in params.items() if k in ("hysteresis_db", "hold_s")})
    if policy_name in ("bonding", "bondingpolicy"):
        return handover.BondingPolicy()
    if policy_name in ("pred_threshold", "threshold", "thresholdpolicy"):
        allowed = {k: v for k, v in params.items() if k in ("theta_soft", "theta_dup", "theta_hard", "theta_video", "lead_extra_s")}
        return handover.ThresholdPolicy(**allowed)
    if policy_name in ("cost_predictive", "cost", "costfunctionpolicy", "pred_cost", "pred_aggressive",
                       "pred_conservative", "pred_latency_guard"):
        return handover.CostFunctionPolicy(
            allow_dups=bool(params.get("allow_dups", True)),
            allow_video_fallback=bool(params.get("allow_video_fallback", True)),
        )
    if policy_name in ("static", "static_primary", "staticprimarypolicy"):
        return None
    if ":" in name:
        module_name, class_name = name.rsplit(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        if not issubclass(cls, handover.DecisionPolicy):
            raise TypeError(f"{name} must subclass predictive_handover.DecisionPolicy")
        return cls(**params)
    raise ValueError(
        f"policy {name!r} is not supported in high_fidelity mode; use reactive, bonding, "
        "static_primary, pred_threshold, pred_cost, or a predictive_handover DecisionPolicy"
    )


class _RouteHook:
    def __init__(self, sim, specs: List[handover.LinkSpec], policy, cfg, mission, params):
        self.sim = sim
        self.controller = handover.HandoverController(specs, cfg, policy)
        self.controller.set_mission(mission, sim.cruise_speed_mps)
        self.static = policy is None
        self.static_control = params.get("control_link")
        self.static_video = params.get("video_link")
        self.last_assignment = None

    def __call__(self, sim):
        su = sim.suite
        obs = su.observation()
        now = sim.t
        pos = tuple(float(x) for x in sim.drone.pos)
        vel = tuple(float(x) for x in sim.drone.vel)
        state = handover.DroneState(now, pos, vel, battery_frac=float(sim.batt.soc))
        self.controller.on_drone_state(state)
        for name, status in su.status.items():
            if name not in self.controller.specs:
                continue
            self.controller.on_link_sample(name, handover.LinkSample(
                t=now,
                rssi_dbm=float(status.rssi_dbm),
                snr_db=float(status.snr_dl_db),
                noise_dbm=None,
                connected=status.state == drone.S_CONN,
            ), pos)
        if self.static:
            available = [n for n, st in su.status.items() if n in self.controller.specs and st.state == drone.S_CONN]
            control = self.static_control if self.static_control in available else next(
                (n for n in available if not su.radios[n].p.analog_video), None)
            video = self.static_video if self.static_video in available else next(iter(available), None)
            assignment = handover.Assignment(control_primary=control, video_primary=video, video_level=1)
        else:
            assignment = self.controller.step(now).assignment
        if assignment == self.last_assignment and now - su.t < 0.02:
            return
        control = [n for n in assignment.control_links() if n in su.radios]
        if not control:
            control = [n for n, st in su.status.items() if n in self.controller.specs and st.state == drone.S_CONN
                       and not su.radios[n].p.analog_video][:1]
        if assignment.video_bonded:
            video = [n for n in assignment.video_bonded if n in su.radios]
        else:
            video = [n for n in (assignment.video_primary, assignment.video_fallback) if n in su.radios]
        if not video and control:
            video = [n for n in control if su.radios[n].p.max_video_bps > 0 or su.radios[n].p.analog_video]
        su.set_routing(
            control_links=control,
            control_mode="duplicate" if len(control) > 1 else "primary",
            video_links=video,
            video_mode="duplicate" if assignment.video_fallback else "split",
            video_bps_override=(
                (0.3e6, 0.3e6, 0.8e6, 1.5e6, 3.0e6, 6.0e6)[
                    max(1, min(int(assignment.video_level), 5))
                ] if assignment.video_fallback else None
            ),
        )
        for name, radio in su.radios.items():
            if name in set(control) | set(video) | set(assignment.keep_warm):
                radio.request("on")
        self.last_assignment = assignment


def _env_to_drone(drone_module, sim_module, scenario: dict, seed: int):
    parsed = sim_module.from_dict(sim_module.ScenarioCfg, scenario)
    source_env = sim_module.Environment(parsed.env, seed)
    env = copy.deepcopy(drone_module.SimConfig().env)
    env.buildings = []
    for lo, hi in zip(source_env.b_lo, source_env.b_hi):
        env.buildings.append(drone_module.Building(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]), float(hi[2] - lo[2])))
    env.hills = []
    terrain = parsed.env.terrain
    if terrain.kind == "hills" and parsed.env.map_file is None:
        map_seed = parsed.env.env_seed if parsed.env.env_seed is not None else seed
        rng = np.random.default_rng(np.random.SeedSequence(int(map_seed), spawn_key=(7,)))
        low, high = terrain.hill_height_m
        for _ in range(terrain.hill_count):
            x = rng.uniform(0, parsed.env.size_m[0])
            y = rng.uniform(0, parsed.env.size_m[1])
            sigma = rng.uniform(*terrain.hill_sigma_m)
            height = rng.uniform(low, high)
            env.hills.append(drone_module.Hill(float(x), float(y), float(height), float(sigma)))
    env.jammers = []
    for jammer in parsed.interference:
        env.jammers.append(drone_module.Jammer(
            name=jammer.name, pos=tuple(jammer.pos), power_dbm=float(jammer.eirp_dbm),
            f_hz=float(jammer.fc_hz), bw_hz=float(jammer.bw_hz), t_on=float(jammer.t0),
            t_off=float(jammer.t1), duty=float(jammer.duty),
            mode="sweep" if jammer.kind == "pulsed_jammer" else "noise",
        ))
    env.rain = []
    timeline = sorted(parsed.weather.timeline, key=lambda k: k.t)
    for i, key in enumerate(timeline):
        if key.rain_mm_h <= 0:
            continue
        end = timeline[i + 1].t if i + 1 < len(timeline) else parsed.duration_s
        env.rain.append(drone_module.RainEvent(float(key.t), float(end), float(key.rain_mm_h)))
    if timeline:
        first = timeline[0]
        angle = math.radians(first.wind_dir_deg)
        env.wind_mean_mps = (-first.wind_mps * math.sin(angle), -first.wind_mps * math.cos(angle), 0.0)
        env.turb_sigma_mps = float(first.gust_sigma_mps)
        env.ambient_c = float(first.temp_c)
    def terrain_height(x, y):
        return sum(h.height * math.exp(-((x - h.x) ** 2 + (y - h.y) ** 2) / (2 * h.sigma ** 2)) for h in env.hills)

    stations = parsed.env.stations
    station_positions = {
        s.id: (float(s.pos[0]), float(s.pos[1]), float(s.pos[2]) + terrain_height(s.pos[0], s.pos[1]))
        for s in stations
    }
    gcs = [s for s in stations if s.kind == "gcs"] or stations[:1]
    env.ground_station = drone_module.GroundStation(station_positions[gcs[0].id]) if gcs else drone_module.GroundStation()
    env.wifi_nodes = [drone_module.WifiNode(id=s.id, pos=station_positions[s.id]) for s in gcs]
    env.towers = [drone_module.CellTower(id=s.id, pos=station_positions[s.id]) for s in stations if s.kind == "cell"]
    env.mesh_nodes = [drone_module.MeshNode(id=s.id, pos=station_positions[s.id]) for s in stations if s.kind == "relay"]
    if not env.mesh_nodes:
        env.mesh_nodes = []
    return parsed, env, station_positions


def _make_link_spec(rp: drone.RadioParams, spec, endpoint: Tuple[float, float, float]) -> handover.LinkSpec:
    rates = tuple((float(th), float(rate * rp.bw_hz * rp.mac_efficiency / 1e6)) for th, rate in rp.mcs_table)
    if not rates:
        rates = ((0.0, 0.1),)
    noise = -174.0 + 10.0 * math.log10(max(rp.bw_hz, 1.0)) + rp.noise_figure_db
    peak_gain = max((float(a.peak_gain_dbi) for a in rp.antennas), default=2.0)
    one_m = 20.0 * math.log10(4.0 * math.pi * rp.freq_hz / drone.C0)
    p0 = rp.peer_tx_power_dbm + rp.peer_gain_dbi + peak_gain - one_m
    return handover.LinkSpec(
        name=rp.name,
        band_group=rp.interference_class or rp.family or rp.kind,
        endpoint=endpoint,
        noise_floor_dbm=noise,
        prior_rssi_1m_dbm=p0,
        prior_path_loss_exp=2.2,
        mcs_table=rates,
        ctrl_snr_min_db=min(float(th) for th, _ in rp.mcs_table),
        base_latency_ms=float(rp.lat_dl_ms + rp.backhaul_ms),
        retx_interval_ms=max(float(rp.retx_delay_ms), 1.0),
        mtu_bytes=int(rp.mtu_bytes),
        supports_video=bool(spec["video"]),
        active_power_w=float(rp.p_idle_w + rp.tx_overhead_w + 10.0 ** (rp.tx_power_dbm / 10.0) / 1000.0 /
                             max(rp.pa_eff_max, 0.05)),
        power_w_per_mbps=0.15,
        idle_power_w=float(rp.p_standby_w),
        acquire_time_s=0.0,
        always_warm=True,
    )


def run_high_fidelity(study, sim_module, scenario: dict, technology_ids: Iterable[str], policy_variant: dict,
                      technology_specs: Dict[str, dict], seed: int, video_mode: str = "standard",
                      duration_s: Optional[float] = None,
                      log_csv: Optional[str] = None, randomize: Optional[dict] = None) -> Dict[str, Any]:
    """Execute one study trial with drone_multilink's native radio and flight models."""
    selected = {canonical_radio_name(name, study.techs.aliases) for name in technology_ids}
    if not selected:
        raise ValueError("high_fidelity requires at least one selected radio")
    if randomize:
        scenario = sim_module.randomize_scenario(scenario, randomize, int(seed))
    parsed, env, endpoints = _env_to_drone(drone, sim_module, scenario, seed)
    cfg = drone.SimConfig()
    cfg.seed = int(seed)
    cfg.physics_dt = float(study.execution.physics_dt_s)
    cfg.radio_dt = max(0.02, float(study.execution.radio_dt_s or study.scale.dt or parsed.dt))
    cfg.log_hz = min(10.0, 1.0 / max(cfg.radio_dt, 0.02))
    cfg.env = env
    home_z = sum(h.height * math.exp(-((parsed.env.home[0] - h.x) ** 2 + (parsed.env.home[1] - h.y) ** 2) / (2 * h.sigma ** 2))
                 for h in env.hills)
    cfg.start_pos = (float(parsed.env.home[0]), float(parsed.env.home[1]), home_z)
    cfg.cruise_speed_mps = float(parsed.mission.default_speed_mps)
    cfg.waypoints = []
    for w in parsed.mission.waypoints:
        x, y = float(w.x), float(w.y)
        z = float(w.z)
        if parsed.mission.agl:
            z += sum(h.height * math.exp(-((x - h.x) ** 2 + (y - h.y) ** 2) / (2 * h.sigma ** 2)) for h in env.hills)
        cfg.waypoints.append(drone.Waypoint((x, y, z), hold_s=float(w.hold_s or 0.0),
                                           speed_mps=float(w.speed) if w.speed else None))
    cfg.failsafe_rth_after_s = float(parsed.failsafe.timeout_s if parsed.failsafe.action == "rtl" else 0.0)
    cfg.traffic.control_hz = float(parsed.traffic.ctrl_hz)
    cfg.traffic.control_bytes = int(parsed.traffic.ctrl_bytes + parsed.traffic.hdr_bytes)
    cfg.traffic.video_max_bps = float(parsed.traffic.video_max_bps)
    cfg.traffic.video_min_bps = min(0.4e6, cfg.traffic.video_max_bps)
    cfg.traffic.video_start_bps = min(4.0e6, cfg.traffic.video_max_bps)
    cfg.traffic.video_fec_overhead = float(parsed.traffic.fec)
    cfg.traffic.quality_r0_bps = float(parsed.traffic.q_r0_bps)
    cfg.traffic.control_gap_s = float(parsed.traffic.ctrl_gap_max_ms / 1000.0)
    cfg.traffic.playout_deadline_s = float(parsed.traffic.video_deadline_ms / 1000.0)
    if video_mode == "ai":
        cfg.traffic.codec.mode = "neural"
        cfg.traffic.codec.neural_power_w = float(study.scale.ai_video.compute_w)
        cfg.traffic.codec.neural_latency_ms = float(study.scale.ai_video.encode_latency_ms)
    radios = {rp.name: rp for rp in cfg.radios}
    overrides = dict(study.techs.radio_overrides)
    for name, values in overrides.items():
        radio_name = canonical_radio_name(name, study.techs.aliases)
        if radio_name not in radios:
            raise ValueError(f"radio override references unavailable radio {radio_name!r}")
        valid_fields = {f.name for f in dataclasses.fields(radios[radio_name])}
        unknown = set(values) - valid_fields
        if unknown:
            raise ValueError(f"radio_overrides[{name!r}] contains unknown field(s): {sorted(unknown)}")
        for key, value in values.items():
            setattr(radios[radio_name], key, copy.deepcopy(value))
    for name, radio in radios.items():
        radio.enabled = name in selected
        if radio.enabled:
            radio.initial_mode = "on"
            cfg.initial_modes[name] = "on"
            jitter_db = float(study.randomize.get("link_tx_jitter_db", 0.0) or 0.0)
            if jitter_db:
                rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(78, sim_module.stable_hash(name))))
                radio.tx_power_dbm += float(rng.normal(0.0, jitter_db))
    if "sim_bond" in selected:
        for member in radios["sim_bond"].members:
            cfg.initial_modes[member] = "on"
    cfg.radio_aliases.update(study.techs.aliases)

    specs = []
    selected_specs = {tid: technology_specs[tid] for tid in selected}
    for name in selected:
        rp = radios[name]
        station_kind = selected_specs[name].get("stations", "gcs")
        matching = [s.id for s in parsed.env.stations if s.kind == station_kind]
        endpoint = endpoints[matching[0]] if matching else next(iter(endpoints.values()), (0.0, 0.0, 0.0))
        specs.append(_make_link_spec(rp, selected_specs[name], endpoint))

    names = selected_specs
    control_names = [n for n in names if names[n]["control"]]
    video_names = [n for n in names if names[n]["video"]]
    first_control = min(control_names, key=lambda n: radios[n].lat_dl_ms) if control_names else next(iter(selected))
    first_video = min(video_names, key=lambda n: radios[n].lat_ul_ms) if video_names else first_control
    cfg.routing.control_links = [first_control]
    cfg.routing.video_links = [first_video]

    policy_info = policy_variant.get("policy", policy_variant)
    name = policy_info.get("name") or policy_info.get("class") or "cost_predictive"
    label = str(policy_variant.get("label", ""))
    if label.startswith("pred_threshold"):
        name = "pred_threshold"
    params = dict(policy_info.get("params", {}))
    handover_config = handover.HandoverConfig.from_dict(params.pop("handover_config", {}))
    handover_config.traffic.ctrl_deadline_hard_ms = float(parsed.traffic.ctrl_deadline_ms)
    handover_config.traffic.video_ladder_kbps = (0.0, 300.0, 800.0, 1500.0, 3000.0, 6000.0)
    policy = _native_policy(name, params)
    hook = _RouteHook(cfg, specs, policy, handover_config,
                      [tuple(float(v) for v in w.pos) for w in cfg.waypoints], params)
    instance = drone.DroneMultiLinkSim(cfg)
    hook.sim = instance
    instance.hooks.append(hook)
    summary = instance.run(float(duration_s or study.scale.duration_s or parsed.duration_s))
    if log_csv:
        instance.save_csv(log_csv)
    comm_energy = float(summary["comm_energy_wh"])
    duration = float(summary["duration_s"])
    metrics = {
        "blind_time_s": float(summary["blind_time_s"]),
        "blind_fraction": float(summary["blind_time_s"] / max(duration, 1e-9)),
        "blind_any": int(summary["blind_time_s"] > 0.0),
        "n_blind_episodes": int(summary["blind_episodes"]),
        "max_blind_episode_s": float(summary["longest_blind_s"]),
        "ctrl_unreliable_s": float(summary["control_outage_s"]),
        "video_unusable_s": float(summary["video_outage_s"]),
        "ctrl_ontime_rate": float(summary["ctrl_ontime_pct"] / 100.0),
        "ctrl_lost_rate": float(summary["ctrl_deadline_violation_pct"] / 100.0),
        "ctrl_lat_p50_ms": float(summary["ctrl_lat_ms_p50"]),
        "ctrl_lat_p95_ms": float(summary["ctrl_lat_ms_p95"]),
        "ctrl_lat_p99_ms": float(summary["ctrl_lat_ms_p99"]),
        "video_quality_mean": float(summary["video_quality_mean"]),
        "video_usable_fraction": float(1.0 - summary["video_outage_s"] / max(duration, 1e-9)),
        "energy_comm_wh": comm_energy,
        "comm_power_mean_w": float(summary["comm_power_mean_w"]),
        "soc_end": float(summary["soc_end"]),
        "flight_s": duration,
        "sim_time_s": duration,
        "mission_completed": int(instance.follower.done),
    }
    metrics["quality_per_watt"] = metrics["video_quality_mean"] / max(metrics["comm_power_mean_w"], 1e-12)
    return {
        "metrics": metrics,
        "summary": summary,
        "active_technologies": sorted(selected),
        "backend": "high_fidelity",
        "radio_names": sorted(instance.suite.radios),
        "end_reason": summary["end_reason"],
        "unsupported_faults": sorted({f.kind for f in parsed.faults + parsed.fault_processes}),
    }
