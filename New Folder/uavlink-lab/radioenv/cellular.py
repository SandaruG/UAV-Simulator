"""Cellular network: hex / listed / OpenCelliD site layouts, sectorised antennas, multi-band
carriers with time-varying load, air-to-ground path loss (geometric, 3GPP-statistical or hybrid),
correlated shadowing, DL/UL SINR with co-channel inter-cell interference, RSRP/RSRQ, neighbour
lists, and a latency model with HARQ + load-dependent queueing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import BandConfig, LinkSpec, RadioEnvConfig
from .geo import load_opencellid
from .propagation import (dipole_gain_dbi, p_los_3gpp, pl_3gpp_db, sector_gain_dbi, sigma_sf_3gpp,
                          terrestrial_path)
from .quality import LinkQuality, finalize, radio_abstraction, retx_latency
from .terrain import SUBURBAN
from .util import (as_points, db, dbm_sum, fspl_db, hash_uniform, local_hour, str_seed, thermal_noise_dbm, undb)


@dataclass
class Site:
    id: str
    idx: int
    x: float
    y: float
    z: float
    h_agl: float
    azimuths: List[float]
    env: str


class CellularNetwork:
    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg: RadioEnvConfig = ctx.cfg
        self.cc = self.cfg.cellular
        self.S = ctx.surface
        self.bands: Dict[str, BandConfig] = {b.id: b for b in self.cc.bands}
        self.sites: List[Site] = self._make_sites()
        self._cells: Dict[str, dict] = {}
        for bi, b in enumerate(self.cc.bands):
            carry = [s.idx for s in self.sites
                     if hash_uniform(self.cfg.area.seed, 501, bi, s.idx) < b.coverage_fraction] or [0]
            site_i, sec_i, az, xyz = [], [], [], []
            for si in carry:
                s = self.sites[si]
                for k, a in enumerate(s.azimuths):
                    site_i.append(si)
                    sec_i.append(k)
                    az.append(a)
                    xyz.append((s.x, s.y, s.z))
            self._cells[b.id] = dict(site=np.array(site_i), sector=np.array(sec_i), az=np.array(az),
                                     xyz=np.array(xyz), carry=carry, band_index=bi)

    # ------------------------------------------------------------------ layout
    def _make_sites(self) -> List[Site]:
        cc, S = self.cc, self.S
        frame = self.ctx.gd.frame
        a = self.cfg.area
        rng = np.random.default_rng(a.seed + 77)
        raw = []
        if cc.layout == "hex":
            n = cc.n_rings
            for q in range(-n, n + 1):
                for r in range(max(-n, -q - n), min(n, -q + n) + 1):
                    x = cc.isd_m * (q + r / 2.0) + rng.normal(0, cc.jitter_m)
                    y = cc.isd_m * (math.sqrt(3) / 2 * r) + rng.normal(0, cc.jitter_m)
                    az0 = rng.uniform(0, 120)
                    raw.append((f"s{q}_{r}", x, y, cc.site_height_agl_m, [az0, az0 + 120, az0 + 240]))
        elif cc.layout == "list":
            for s in cc.sites:
                if s.lat is not None and s.lon is not None:
                    x, y, _ = frame.ll_to_enu(s.lat, s.lon, 0.0)
                else:
                    x, y = s.x, s.y
                raw.append((s.id, float(x), float(y), s.height_agl_m, list(s.azimuths_deg)))
        elif cc.layout == "opencellid":
            lat0, lon0 = a.origin_lat, a.origin_lon
            dlat = a.size_y_m / 2 / 111000.0
            dlon = a.size_x_m / 2 / (111000.0 * math.cos(math.radians(lat0)))
            seen = set()
            for i, c in enumerate(load_opencellid(cc.opencellid_csv, (lat0 - dlat, lon0 - dlon, lat0 + dlat, lon0 + dlon))):
                x, y, _ = frame.ll_to_enu(c["lat"], c["lon"], 0.0)
                key = (int(x // 40), int(y // 40))
                if key in seen:
                    continue
                seen.add(key)
                az0 = 360.0 * hash_uniform(a.seed, 502, i)
                raw.append((f"{c['radio']}{c['cell']}", float(x), float(y), cc.site_height_agl_m, [az0, az0 + 120, az0 + 240]))
        else:
            raise ValueError(f"unknown cellular layout {cc.layout}")
        sites = []
        for i, (sid, x, y, h, az) in enumerate(raw):
            z = float(S.top_at(x, y)) + h
            env = ("UMa" if float(S.built_frac_at(x, y)) > 0.08 else "RMa") if cc.environment == "auto" else cc.environment
            sites.append(Site(sid, i, x, y, z, h, az, env))
        return sites

    def cells(self, band_id: str):
        c = self._cells[band_id]
        return [dict(cell=i, site=self.sites[s].id, sector=int(k), az=float(a), x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
                for i, (s, k, a, xyz) in enumerate(zip(c["site"], c["sector"], c["az"], c["xyz"]))]

    # ------------------------------------------------------------------ load model
    def load(self, band_id: str, t):
        cd = self._cells[band_id]
        b = self.bands[band_id]
        t = np.asarray(t, dtype=float)
        nc = len(cd["site"])
        ids = np.arange(nc)
        u = hash_uniform(self.cfg.area.seed, 503, cd["band_index"], cd["site"], cd["sector"])
        base = b.load_mean * (0.6 + 0.8 * u)
        if self.cc.diurnal_load:
            h = local_hour(self.ctx.start_utc, t, self.cfg.area.origin_lon)
            diur = 0.55 + 0.45 * np.cos(2 * np.pi * (h - 20.0) / 24.0)
        else:
            diur = np.ones_like(t)
        phi = 2 * np.pi * hash_uniform(self.cfg.area.seed, 504, ids)
        wob = 1.0 + 0.15 * np.sin(2 * np.pi * t[:, None] / (240.0 + 37.0 * ids[None, :]) + phi[None, :])
        return np.clip(base[None, :] * diur[:, None] * wob, 0.02, 0.95)

    # ------------------------------------------------------------------ measurement of all cells
    def measure(self, band_id: str, pos, t, fading_mode: str = "full", ue=(2.0, "dipole")):
        ctx, S = self.ctx, self.S
        b = self.bands[band_id]
        cd = self._cells[band_id]
        pos = as_points(pos)
        N = len(pos)
        t = np.broadcast_to(np.asarray(t, dtype=float), (N,)).copy()
        f = b.freq_mhz * 1e6
        fg = f / 1e9
        carry = cd["carry"]
        ns = len(carry)
        nc = len(cd["site"])
        mode = self.cc.model
        fc = self.cfg.fading
        use_wx = f >= self.cfg.propagation.weather_min_ghz * 1e9
        h_ut = np.maximum(pos[:, 2] - S.terrain_at(pos[:, 0], pos[:, 1]), 1.5)

        PL = np.zeros((N, ns))
        LOS = np.zeros((N, ns), dtype=bool)
        KDB = np.zeros((N, ns))
        DS = np.zeros((N, ns))
        ELV = np.zeros((N, ns))
        D3 = np.zeros((N, ns))
        ATM = np.zeros((N, ns))
        SF = np.zeros((N, ns))
        G_UE = np.zeros((N, ns))
        G_BS = np.zeros((N, nc))
        site_pos = np.array([[self.sites[si].x, self.sites[si].y, self.sites[si].z] for si in carry])
        for j, si in enumerate(carry):
            site = self.sites[si]
            sp = site_pos[j]
            path = terrestrial_path(S, sp, pos, f, self.cfg.propagation, two_ray=False, cfg_fading=fc)
            d3, d2 = path["d3d"], path["d2d"]
            h_bs = max(site.z - float(S.terrain_at(site.x, site.y)), 5.0)
            if mode == "geometric":
                los = path["los"]
                pl = fspl_db(d3, f) + path["loss_diff"] + path["loss_veg"]
                sig = np.where(los, fc.shadow_sigma_los_db, fc.shadow_sigma_nlos_db)
            else:
                if mode == "3gpp":
                    pl_los_p = p_los_3gpp(site.env, d2, h_ut)
                    u = hash_uniform(self.cfg.area.seed, 601, si, np.floor(pos[:, 0] / 30), np.floor(pos[:, 1] / 30),
                                     np.floor(pos[:, 2] / 15))
                    los = u < pl_los_p
                    extra = 0.0
                else:   # hybrid: geometric LOS decides the 3GPP branch
                    los = path["los"]
                    extra = path["loss_veg"]
                pl = pl_3gpp_db(site.env, los, d3, d2, h_bs, h_ut, fg) + extra
                sig = sigma_sf_3gpp(site.env, los, h_ut)
            atm = ctx.weather.gas_atten_path(sp, pos, f)
            if use_wx:
                atm = atm + ctx.weather.rain_atten_path(sp, pos, f, t)
                atm = atm + ctx.weather.fog_atten_path(sp, pos, f, t, S.terrain_at(site.x, site.y), S.terrain_at(pos[:, 0], pos[:, 1]))
            if fading_mode != "none":
                shc = ctx.get_shadow(f"cell:{band_id}:common", fc.shadow_dcorr_m * 2.0)
                shs = ctx.get_shadow(f"cell:{band_id}:{si}", fc.shadow_dcorr_m * 2.0)
                rho = self.cc.inter_site_shadow_corr
                z = math.sqrt(rho) * shc.sample(pos[:, 0], pos[:, 1], pos[:, 2]) + math.sqrt(1 - rho) * shs.sample(pos[:, 0], pos[:, 1], pos[:, 2])
                SF[:, j] = z * sig
            PL[:, j], LOS[:, j], KDB[:, j] = pl, los, np.where(los, path["k_factor_db"], -30.0)
            DS[:, j], ELV[:, j], D3[:, j], ATM[:, j] = path["delay_spread_ns"], path["elev_deg"], d3, atm
            el_bs_to_ue = np.degrees(np.arctan2(pos[:, 2] - site.z, np.maximum(d2, 1.0)))
            G_UE[:, j] = _ue_gain(ue, el_bs_to_ue)
            bearing = np.degrees(np.arctan2(pos[:, 0] - site.x, pos[:, 1] - site.y))
            for k in np.nonzero(cd["site"] == si)[0]:
                rel = bearing - cd["az"][k]
                G_BS[:, k] = sector_gain_dbi(rel, el_bs_to_ue + b.downtilt_deg, b.bs_gain_dbi, b.hpbw_h_deg, b.hpbw_v_deg, b.sla_v_db)
        site_col = {si: j for j, si in enumerate(carry)}
        col = np.array([site_col[s] for s in cd["site"]])
        P = b.tx_power_dbm + G_BS - PL[:, col] - SF[:, col] - ATM[:, col]       # BS-side EIRP minus loss (before UE gain)
        P = P + G_UE[:, col]
        rsrp = P - 10 * np.log10(12 * b.n_prb)
        return dict(P=P, rsrp=rsrp, PL=PL, LOS=LOS, K=KDB, DS=DS, ELV=ELV, D3=D3, ATM=ATM, SF=SF, G_UE=G_UE, G_BS=G_BS,
                    col=col, site_pos=site_pos, t=t, pos=pos)

    # ------------------------------------------------------------------ link evaluation
    def evaluate(self, spec: LinkSpec, pos, t, fading_mode="full", serving_override=None, nlimit=None) -> LinkQuality:
        ctx = self.ctx
        b = self.bands[spec.band]
        cd = self._cells[spec.band]
        m = self.measure(spec.band, pos, t, fading_mode, ue=(spec.drone_gain_dbi, spec.drone_pattern))
        pos, t = m["pos"], m["t"]
        N = len(pos)
        f = b.freq_mhz * 1e6
        bw = b.bw_mhz * 1e6
        rows = np.arange(N)
        nc = m["P"].shape[1]
        if serving_override is None:
            serv = np.argmax(m["P"], axis=1)
        else:
            serv = np.clip(np.broadcast_to(np.asarray(serving_override, dtype=int), (N,)), 0, nc - 1)
        load = self.load(spec.band, t)
        load_s = load[rows, serv]
        col_s = m["col"][serv]
        pl_s, sf_s, atm_s = m["PL"][rows, col_s], m["SF"][rows, col_s], m["ATM"][rows, col_s]
        k_s = 10 ** (m["K"][rows, col_s] / 10.0)
        k_db_s = m["K"][rows, col_s]
        gue_s = m["G_UE"][rows, col_s]
        gbs_s = m["G_BS"][rows, serv]
        d3_s, los_s = m["D3"][rows, col_s], m["LOS"][rows, col_s]
        # --- small-scale fading on the serving link only
        ff = np.zeros(N)
        if fading_mode == "full":
            for si in np.unique(cd["site"][serv]):
                sel = cd["site"][serv] == si
                fgen = ctx.get_fast(f"cell:{spec.band}:{si}", f)
                ff[sel] = fgen.gain_db(pos[sel], t[sel], k_db_s[sel], d3_s[sel])
        # downlink-side RSSI is needed for RSRQ whichever direction the data link uses
        noise_dl = ctx.interf.noise_dbm(pos, f, bw, b.ue_nf_db)
        pw = undb(m["P"]) * load
        pw[rows, serv] = undb(m["P"][rows, serv])
        dl_rssi = db(pw.sum(1) + undb(noise_dl))
        if spec.direction == "downlink":
            S_dbm = m["P"][rows, serv] + ff
            others = undb(m["P"]) * load
            others[rows, serv] = 0.0
            ext = ctx.interf.power_dbm(pos, t, f, bw, rx_gain_dbi=0.0)
            I_tot = db(others.sum(1) + undb(ext))
            noise = noise_dl
            bw_eff = bw * np.maximum(0.1, 1.0 - load_s)
        else:
            ul_bw = b.ul_prb_alloc * 180e3
            pl_loss = pl_s + atm_s + sf_s
            tx = np.minimum(b.ue_tx_dbm, b.ul_p0_dbm + 10 * np.log10(b.ul_prb_alloc) + b.ul_alpha * pl_loss)
            S_dbm = tx + gue_s + gbs_s - pl_loss + ff
            # load shows up as an interference-over-thermal rise at the base station
            iot_db = np.minimum(-10 * np.log10(1.0 - load_s), 12.0)
            sp = m["site_pos"][col_s]
            ext = ctx.interf.power_dbm(sp, t, f, ul_bw, rx_gain_dbi=b.bs_gain_dbi)
            noise = thermal_noise_dbm(ul_bw, b.bs_nf_db)
            I_tot = db(undb(ext) + undb(noise) * (undb(iot_db) - 1.0))
            bw_eff = ul_bw
            bw = ul_bw
        n_plus_i = db(undb(noise) + undb(I_tot))
        sinr = S_dbm - n_plus_i
        snr = S_dbm - noise
        per, thr = radio_abstraction(spec, sinr, bw_eff, 1.0, se_alpha=b.se_alpha, se_max=b.max_se)
        air = b.air_latency_ms * (1.0 + 0.5 * load_s / (1.0 - load_s))
        lat, lat99 = retx_latency(air + b.core_latency_ms, b.harq_ms, per, jitter_ms=2.0 * load_s)
        rssi = dbm_sum(S_dbm, I_tot, noise)
        lq = LinkQuality(spec.id, "cellular", t.copy())
        lq.rssi_dbm, lq.signal_dbm, lq.noise_dbm, lq.interference_dbm = rssi, S_dbm, noise, I_tot
        lq.snr_db, lq.sinr_db = snr, sinr
        lq.pathloss_db = pl_s + atm_s
        lq.los, lq.shadow_db, lq.fading_db = los_s, -sf_s, ff
        lq = finalize(spec, lq, per, thr, lat, lat99, sinr, snr, k_lin=k_s,
                      mean_sinr_db=sinr - ff)
        # --- extra diagnostics / predictor features
        rsrp_s = m["rsrp"][rows, serv]
        K = min(self.cc.k_neighbors, nc)
        order = np.argsort(-m["rsrp"], axis=1)[:, :K]
        lq.extra.update(
            serving_cell=serv, serving_site=cd["site"][serv], serving_sector=cd["sector"][serv],
            rsrp_dbm=rsrp_s, rsrq_db=10 * np.log10(b.n_prb) + rsrp_s - dl_rssi,
            cell_load=load_s, neighbor_cells=order, neighbor_rsrp_dbm=np.take_along_axis(m["rsrp"], order, 1),
            best_cell=np.argmax(m["P"], axis=1), elev_deg=m["ELV"][rows, col_s], k_factor_db=k_db_s,
            delay_spread_ns=m["DS"][rows, col_s], dist_m=d3_s, atmos_db=atm_s, direction=spec.direction,
            ho_margin_db=m["rsrp"][rows, np.argmax(m["P"], axis=1)] - rsrp_s)
        return lq


def _ue_gain(ue, el_to_bs_deg):
    """Drone antenna gain towards a BS. The BS is seen from the drone at -elevation (as seen from the BS)."""
    gain, pattern = ue
    el_at_ue = -np.asarray(el_to_bs_deg, dtype=float)
    if pattern == "dipole":
        return gain + (dipole_gain_dbi(el_at_ue) - 2.15)
    return np.full_like(el_at_ue, gain, dtype=float)
