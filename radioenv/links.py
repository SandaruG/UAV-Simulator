"""Link-budget evaluators for direct (GCS <-> drone) radio links and LEO satellite links."""
from __future__ import annotations

import numpy as np

from .config import LinkSpec
from .propagation import dipole_gain_dbi, foliage_loss_db, panel_gain_dbi, terrestrial_path
from .quality import LinkQuality, finalize, radio_abstraction, retx_latency
from .util import (BOLTZMANN, as_points, db, dbm_sum, fspl_db, local_hour, sun_az_el, thermal_noise_dbm, undb)


# ======================================================================================= direct
def eval_direct(ctx, spec: LinkSpec, pos, t, fading: str = "full") -> LinkQuality:
    S, cfg = ctx.surface, ctx.cfg
    pos = as_points(pos)
    N = len(pos)
    t = np.broadcast_to(np.asarray(t, dtype=float), (N,)).copy()
    st = ctx.station(spec.station)
    gcs = st["xyz"]
    f, bw = spec.freq_mhz * 1e6, spec.bw_mhz * 1e6
    fc = cfg.fading

    path = terrestrial_path(S, gcs, pos, f, cfg.propagation, cfg_fading=fc)
    d3, los = path["d3d"], path["los"]
    atm = ctx.weather.gas_atten_path(gcs, pos, f)
    if f >= cfg.propagation.weather_min_ghz * 1e9:
        atm = atm + ctx.weather.rain_atten_path(gcs, pos, f, t)
        atm = atm + ctx.weather.fog_atten_path(gcs, pos, f, t, S.terrain_at(gcs[0], gcs[1]), S.terrain_at(pos[:, 0], pos[:, 1]))
    pl = fspl_db(d3, f) + path["loss_diff"] + path["loss_veg"] - path["two_ray_db"]

    sig = np.where(los, fc.shadow_sigma_los_db, fc.shadow_sigma_nlos_db)
    if fading != "none":
        sh = sig * ctx.get_shadow(f"direct:{spec.station}", fc.shadow_dcorr_m).sample(pos[:, 0], pos[:, 1], pos[:, 2])
    else:
        sh = np.zeros(N)
    ff = np.zeros(N)
    if fading == "full":
        ff = ctx.get_fast(f"direct:{spec.id}", f).gain_db(pos, t, path["k_factor_db"], d3)

    d2 = path["d2d"]
    el_drone_sees_gcs = np.degrees(np.arctan2(gcs[2] - pos[:, 2], np.maximum(d2, 1.0)))
    el_gcs_sees_drone = -el_drone_sees_gcs
    if spec.drone_pattern == "dipole":
        g_dr = dipole_gain_dbi(el_drone_sees_gcs, peak=spec.drone_gain_dbi)
    else:
        g_dr = np.full(N, spec.drone_gain_dbi)
    if st["pattern"] == "tracking":
        g_gs = np.full(N, st["gain_dbi"])
    else:
        g_gs = dipole_gain_dbi(el_gcs_sees_drone, peak=st["gain_dbi"])

    S_dbm = spec.tx_power_dbm - spec.cable_loss_db + g_dr + g_gs - pl - sh - atm + ff
    if spec.direction == "uplink":      # drone transmits, GCS receives
        rx_pos = np.broadcast_to(gcs, (N, 3))
        agl_rx = np.full(N, st["height_agl_m"])
    else:
        rx_pos = pos
        agl_rx = S.agl(pos)
    interf = ctx.interf.power_dbm(rx_pos, t, f, bw, rx_gain_dbi=spec.interference_rx_gain_dbi)
    noise = ctx.interf.noise_dbm(rx_pos, f, bw, spec.rx_nf_db, agl=agl_rx)
    n_i = db(undb(noise) + undb(interf))
    sinr, snr = S_dbm - n_i, S_dbm - noise
    per, thr = radio_abstraction(spec, sinr, bw)
    lat, lat99 = retx_latency(spec.base_latency_ms, spec.retx_ms, per, jitter_ms=1.0)

    lq = LinkQuality(spec.id, "direct", t.copy())
    lq.rssi_dbm = dbm_sum(S_dbm, interf, noise)
    lq.signal_dbm, lq.noise_dbm, lq.interference_dbm = S_dbm, noise, interf
    lq.snr_db, lq.sinr_db = snr, sinr
    lq.pathloss_db, lq.los, lq.shadow_db, lq.fading_db = pl + atm, los, -sh, ff
    k_lin = undb(path["k_factor_db"])
    lq = finalize(spec, lq, per, thr, lat, lat99, sinr, snr, k_lin=k_lin, mean_sinr_db=sinr - ff)
    lq.extra.update(dist_m=d3, elev_deg=np.abs(el_gcs_sees_drone), fresnel_clearance=path["fresnel_clear"],
                    diffraction_db=path["loss_diff"], foliage_db=path["loss_veg"], penetration_db=path["loss_pen"],
                    atmos_db=atm, delay_spread_ns=path["delay_spread_ns"], k_factor_db=path["k_factor_db"],
                    tx_gain_dbi=g_dr if spec.direction == "uplink" else g_gs,
                    rx_gain_dbi=g_gs if spec.direction == "uplink" else g_dr, direction=spec.direction)
    return lq


# ======================================================================================= satellite
def _sun_separation_deg(ctx, t, az_sat, el_sat):
    """Angle between the satellite direction and the sun (deg) at time(s) t."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.full(len(t), 180.0)
    # the sun moves < 0.01 deg per 60 s: evaluate on a coarse grid
    tq = np.round(t / 60.0)
    for tk in np.unique(tq):
        sel = tq == tk
        dt = ctx.start_utc + __import__("datetime").timedelta(seconds=float(tk * 60.0))
        saz, sel_ = sun_az_el(dt, ctx.cfg.area.origin_lat, ctx.cfg.area.origin_lon)
        a1, e1 = np.radians(az_sat[sel]), np.radians(el_sat[sel])
        a2, e2 = np.radians(saz), np.radians(sel_)
        c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
        out[sel] = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def eval_satellite(ctx, spec: LinkSpec, pos, t, vel=None, fading: str = "full") -> LinkQuality:
    """LEO bent-pipe link. `spec.freq_mhz`/`spec.bw_mhz` set the carrier; direction picks the budget."""
    sc = ctx.cfg.satellite
    S, W = ctx.surface, ctx.weather
    pos = as_points(pos)
    N = len(pos)
    t = np.broadcast_to(np.asarray(t, dtype=float), (N,)).copy()
    f, bw = spec.freq_mhz * 1e6, spec.bw_mhz * 1e6
    lam = 299792458.0 / f
    C = ctx.constellation
    idx, n_vis = C.select(pos, t, S, lam)
    ok = idx >= 0
    safe = np.where(ok, idx, 0)
    obs = ctx.gd.frame.enu_to_ecef(pos)
    sat_xyz = C.ecef(t, safe)
    el, az, rng = C.look(obs, sat_xyz)
    el = np.where(ok, el, 0.0)

    # ---- losses
    obs_loss, len_can, blocked = S.ray_obstruction(pos, az, np.maximum(el, 1.0), lam, max_d=ctx.cfg.propagation.obstruction_range_m)
    obs_loss = np.where(ok, obs_loss, 0.0)
    veg = foliage_loss_db(len_can, f, S.leaf_factor) * ok
    fspl = fspl_db(rng, f)
    gas = W.gas_atten_slant(pos[:, 2], el, f)
    rain = W.rain_atten_slant(pos, az, el, f, t)
    cloud = W.cloud_atten_slant(pos, az, el, f, t)
    scint = W.scintillation_db(f, el, t, seed_key=7)
    atm = gas + rain + cloud
    # ---- terminal antenna: panel boresight is local zenith
    off = 90.0 - el
    g_term = panel_gain_dbi(off, sc.terminal_gain_dbi, sc.max_scan_deg)
    if spec.direction == "uplink":
        # terminal transmit power is a property of the satellite terminal (satellite.terminal_tx_dbm)
        eirp_dbw = sc.terminal_tx_dbm - 30.0 - spec.cable_loss_db + g_term
        gt = sc.sat_gt_dbk
        cci = sc.cci_ul_db
        t_ref = np.full(N, 500.0)                  # reference noise temperature at the satellite input
    else:
        eirp_dbw = sc.sat_eirp_dbw
        sun_sep = _sun_separation_deg(ctx, t, az, el) if sc.sun_outage else np.full(N, 180.0)
        # solar noise burst when the sun is within ~1 deg of the satellite direction (Ku-band, small dish)
        tsun = np.where(sun_sep < 1.0, 6000.0 * (1.0 - sun_sep) ** 2, 0.0)
        t_ref = sc.terminal_tsys_k + tsun
        gt = g_term - db(t_ref)
        cci = sc.cci_dl_db
    loss_total = fspl + atm + obs_loss + veg + sc.pol_loss_db
    cn0 = eirp_dbw + gt - loss_total - 10 * np.log10(BOLTZMANN) + scint
    ff = np.zeros(N)
    k_db = np.clip(14.0 - 0.5 * (obs_loss + veg), -5.0, 14.0)
    if fading == "full":
        ff = ctx.get_fast(f"sat:{spec.id}", f).gain_db(pos, t, k_db, rng)
    snr_ideal = cn0 - 10 * np.log10(bw) + ff
    c_over_i = undb(cci)
    sinr = -db(undb(-snr_ideal) + 1.0 / c_over_i) - sc.impl_loss_db
    # receiver-referred levels: noise = kT_ref*B, signal = noise + SNR, interference = co-channel (C/I)
    noise = thermal_noise_dbm(bw, 0.0) + db(t_ref / 290.0)
    S_dbm = noise + snr_ideal
    interf = S_dbm - cci
    per, thr = radio_abstraction(spec, sinr, bw, prb_share=sc.user_share)
    # ---- slot-boundary interruption
    guard = (t % sc.slot_s) < sc.guard_s
    per = np.where(guard, 1.0, per)
    thr = np.where(guard, 0.0, thr)
    per = np.where(ok, per, 1.0)
    thr = np.where(ok, thr, 0.0)
    prop_ms = 2.0 * rng / 299792458.0 * 1e3
    base = prop_ms + sc.gateway_latency_ms + sc.proc_latency_ms + sc.sched_latency_ms + spec.base_latency_ms
    lat, lat99 = retx_latency(base, 2.0 * prop_ms + sc.sched_latency_ms, per, jitter_ms=3.0)
    sinr = np.where(ok, sinr, -60.0)
    snr_out = np.where(ok, snr_ideal, -60.0)

    lq = LinkQuality(spec.id, "satellite", t.copy())
    lq.signal_dbm = np.where(ok, S_dbm, -200.0)
    lq.noise_dbm = noise
    lq.interference_dbm = np.where(ok, interf, -200.0)
    lq.rssi_dbm = lq.signal_dbm
    lq.snr_db, lq.sinr_db = snr_out, sinr
    lq.pathloss_db = np.where(ok, loss_total, 300.0)
    lq.los = ok & ~blocked
    lq.shadow_db, lq.fading_db = obs_loss + veg, ff
    lq = finalize(spec, lq, per, thr, lat, lat99, sinr, snr_out, k_lin=undb(k_db), avail=ok & ~guard & (per < 0.5),
                  mean_sinr_db=sinr - ff)
    dop = C.doppler_hz(pos, idx, t, f, vel_enu=vel) if ok.any() else np.zeros(N)
    lq.extra.update(sat_index=idx, sat_elev_deg=el, sat_az_deg=az, range_km=rng / 1e3, n_visible=n_vis,
                    doppler_khz=dop / 1e3, rain_db=rain, cloud_db=cloud, gas_db=gas, scint_db=scint,
                    obstruction_db=obs_loss, foliage_db=veg, terminal_gain_dbi=g_term, handover_interruption=guard,
                    slot_time_left_s=sc.slot_s - (t % sc.slot_s), direction=spec.direction)
    return lq
