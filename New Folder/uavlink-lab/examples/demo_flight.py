"""End-to-end demo: build the radio world, fly a drone across it, query every link, and draw figures.

    python examples/demo_flight.py [outdir]

Produces fig_world.png, fig_coverage.png, fig_flight.png, fig_satellite.png and flight_log.csv.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from radioenv import JammerConfig, RadioEnvConfig, RadioEnvironment   # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

# ---- validated categorical slots (light surface) ------------------------------------------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
C = {"c2_915": "#2a78d6", "video_58": "#eb6834", "cell_lte1800": "#1baf7a", "cell_nr3500": "#eda100", "sat_ku": "#e87ba4"}
NAMES = {"c2_915": "C2 915 MHz", "video_58": "Video 5.8 GHz", "cell_lte1800": "LTE 1.8 GHz (UL)",
         "cell_nr3500": "NR 3.5 GHz (UL)", "sat_ku": "LEO Ku (UL)"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
                     "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False})

# ---- scenario -----------------------------------------------------------------------------------
cfg = RadioEnvConfig().replace(**{"weather.scenario": "moderate_rain", "weather.cloud_cover": 0.6, "area.seed": 7})
cfg.interference.jammers = [JammerConfig(id="jam0", kind="noise", x=-950.0, y=-700.0, z_agl_m=3.0, freq_mhz=5800.0,
                                         bw_mhz=80.0, eirp_dbm=30.0, windows=[[40.0, 90.0]])]
env = RadioEnvironment(cfg)
print(env.describe()["geodata"])

# ---- flight track (12 m/s, keeps 60 m above the highest surface within 40 m) --------------------
wp = np.array([[-1100, -800], [-600, -450], [0, 0], [600, 500], [1200, 900], [1750, 1450]], float)
seg = np.hypot(*np.diff(wp, axis=0).T)
s_total = seg.sum()
v = 12.0
t = np.arange(0.0, s_total / v, 0.5)
sdist = t * v
cum = np.r_[0, np.cumsum(seg)]
x = np.interp(sdist, cum, wp[:, 0])
y = np.interp(sdist, cum, wp[:, 1])
top = np.max([env.surface_height(x + dx, y + dy) for dx in (-40, 0, 40) for dy in (-40, 0, 40)], axis=0)
z = np.convolve(np.pad(top + 60.0, 20, mode="edge"), np.ones(41) / 41, mode="valid")[: len(t)]
z = np.maximum(z, top + 40.0)
P = np.c_[x, y, z]
res = env.query_track(P, t)
print(f"flight {t[-1]:.0f} s, {len(t)} samples")

# ---- blind time -----------------------------------------------------------------------------
ctrl_primary = np.asarray(res["c2_915"].ctrl_reliable)
vid_primary = np.asarray(res["video_58"].video_usable)
blind_primary = ~ctrl_primary & ~vid_primary
ctrl_any = np.any([np.asarray(q.ctrl_reliable) for q in res.values()], axis=0)
vid_any = np.any([np.asarray(q.video_usable) for q in res.values()], axis=0)
blind_best = ~ctrl_any & ~vid_any
dt = t[1] - t[0]
print(f"blind time (neither reliable control nor video): primary pair {blind_primary.sum() * dt:.0f} s "
      f"({100 * blind_primary.mean():.0f} %), best of all links {blind_best.sum() * dt:.0f} s ({100 * blind_best.mean():.0f} %)")
print(f"control-only lost {100 * (~ctrl_primary).mean():.0f} %, video-only lost {100 * (~vid_primary).mean():.0f} % (primary pair)")

import csv
with open(os.path.join(OUT, "flight_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "x", "y", "z"] + [f"{k}_{m}" for k in res for m in ("rssi_dbm", "snr_db", "sinr_db", "latency_ms", "throughput_mbps", "per", "outage_prob")])
    for i in range(len(t)):
        w.writerow([t[i], x[i], y[i], z[i]] + [float(getattr(res[k], m)[i]) for k in res for m in
                    ("rssi_dbm", "snr_db", "sinr_db", "latency_ms", "throughput_mbps", "per", "outage_prob")])

# ---- fig 1: the world ------------------------------------------------------------------------
gd, g = env.gd, env.gd.grid
ext = [g.x0, g.x1, g.y0, g.y1]
fig, ax = plt.subplots(2, 2, figsize=(11, 10))
panels = [("Terrain elevation (m MSL)", gd.terrain, "Greys"), ("Building height (m)", np.where(gd.bldg > 0, gd.bldg, np.nan), "Blues"),
          ("Vegetation canopy (m)", np.where(gd.canopy > 0, gd.canopy, np.nan), "Greens")]
for a, (title, arr, cm) in zip(ax.ravel()[:3], panels):
    im = a.imshow(arr, origin="lower", extent=ext, cmap=cm)
    a.set_title(title)
    fig.colorbar(im, ax=a, shrink=0.8)
cl_cols = ["#ecebe7", "#2a78d6", "#008300", "#eda100", "#eb6834"]
a = ax[1, 1]
a.imshow(gd.clutter, origin="lower", extent=ext, cmap=ListedColormap(cl_cols), vmin=0, vmax=4)
a.set_title("Clutter class")
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=c, label=n) for c, n in zip(cl_cols, ["open", "water", "vegetation", "suburban", "urban"])],
         loc="lower right", fontsize=8, facecolor=SURF, framealpha=0.9, frameon=True)
for a in ax.ravel():
    sx = [s.x for s in env.cellular.sites]
    sy = [s.y for s in env.cellular.sites]
    a.plot(sx, sy, "^", color=INK, ms=5, mec=SURF, mew=1)
    a.plot(*env.station_xyz()[:2], "*", color="#eb6834", ms=14, mec=INK, mew=0.8)
    a.plot(x, y, "-", color=INK, lw=1.4)
    a.set_xlabel("East (m)")
    a.set_ylabel("North (m)")
    a.set_aspect("equal")
ax[0, 0].annotate("GCS", env.station_xyz()[:2], xytext=(10, 8), textcoords="offset points", fontsize=9)
ax[0, 0].plot([], [], "^", color=INK, label="cell site")
ax[0, 0].plot([], [], "-", color=INK, label="flight track")
ax[0, 0].legend(loc="upper left", fontsize=8, facecolor=SURF, framealpha=0.9, frameon=True)
fig.suptitle(f"Synthetic digital twin — {env.cfg.area.name} (seed {cfg.area.seed}), {len(env.cellular.sites)} cell sites", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_world.png"), dpi=130)
plt.close(fig)

# ---- fig 2: coverage maps at 100 m AGL ----------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(14, 8.6))
for a, lid in zip(ax.ravel(), ["c2_915", "video_58", "cell_lte1800", "cell_nr3500", "sat_ku"]):
    m = env.coverage_map(lid, z_agl=100.0, t=100.0, nx=90, ny=90, fading="shadow")
    im = a.imshow(m["sinr_db"], origin="lower", extent=ext, cmap="Blues", vmin=-20, vmax=35)
    a.contour(m["x"], m["y"], m["sinr_db"], levels=[env.link(lid).req_sinr_db], colors=[INK], linewidths=0.9)
    a.plot(*env.station_xyz()[:2], "*", color="#eb6834", ms=12, mec=INK, mew=0.8)
    a.set_title(f"{NAMES[lid]} — SINR (dB), 100 m AGL")
    a.set_aspect("equal")
    a.set_xlabel("East (m)")
    fig.colorbar(im, ax=a, shrink=0.8)
ax.ravel()[-1].axis("off")
ax.ravel()[-1].text(0.0, 0.7, "Black contour: decode threshold\nStar: ground station\nt = 100 s, moderate rain,\nshadowing on, fast fading off",
                    fontsize=10, color=INK2, va="top")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_coverage.png"), dpi=130)
plt.close(fig)

# ---- fig 3: flight ---------------------------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 1.1, 1.1]))
for k in C:
    ax[0].plot(t, res[k].sinr_db, color=C[k], lw=1.5, label=NAMES[k])
    ax[1].plot(t, np.minimum(res[k].latency_ms, 400), color=C[k], lw=1.5)
ax[0].set_ylabel("SINR (dB)")
ax[0].set_ylim(-70, 50)
ax[0].grid(axis="y")
ax[0].axvspan(40, 90, color="#000", alpha=0.06, lw=0)
ax[0].text(42, 47, "5.8 GHz jammer on", fontsize=8, color=INK2, va="top")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8)
ax[0].set_title("Link SINR along the flight (moderate rain, fast fading on; clipped at +50 dB)")
ax[1].axhline(150, color=INK2, lw=1, ls="--")
ax[1].text(2, 158, "150 ms control limit", fontsize=8, color=INK2)
ax[1].set_ylabel("One-way latency (ms, capped 400)")
ax[1].grid(axis="y")
for a_, (lab, blind) in zip(ax[2:], [("Primary pair\n(C2 915 + video 5.8)", blind_primary), ("Best of all\n7 links", blind_best)]):
    a_.fill_between(t, 0, 1, where=blind, color="#e34948", step="mid", lw=0)
    a_.set_ylim(0, 1)
    a_.set_yticks([])
    a_.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    a_.text(2, 0.5, f"blind {blind.sum() * dt:.0f} s ({100 * blind.mean():.0f} %)", ha="left", va="center", fontsize=9, color=INK)
ax[2].set_title("Blind time: neither reliable control nor usable video", fontsize=10)
ax[-1].set_xlabel("Time (s)")
ax[0].set_xlim(0, t[-1])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_flight.png"), dpi=130)
plt.close(fig)

# ---- fig 4: satellite, clear vs heavy rain (hovering drone) -------------------------------------
hov = np.array([0.0, 0.0, env.surface_height(0, 0) + 100.0])
th = np.arange(0.0, 600.0, 0.5)
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for scen, col, lab in (("clear", "#2a78d6", "clear"), ("heavy_rain", "#eb6834", "heavy rain")):
    e2 = RadioEnvironment(cfg.replace(**{"weather.scenario": scen, "weather.cloud_cover": 0.3 if scen == "clear" else 0.8}))
    q = e2.query("sat_ku", np.tile(hov, (len(th), 1)), th)
    ax[0].plot(th, q.sinr_db, color=col, lw=1.2, label=lab)
    ax[1].plot(th, q.extra["rain_db"] + q.extra["cloud_db"], color=col, lw=1.2)
    if scen == "clear":
        ax[2].plot(th, q.extra["sat_elev_deg"], color=INK2, lw=1.2)
        ho = th[q.extra["handover_interruption"]]
        for h in ho:
            ax[2].axvline(h, color="#e34948", lw=0.8, alpha=0.7)
ax[0].set_ylabel("Uplink SINR (dB)")
ax[0].legend(loc="lower left")
ax[0].set_title("LEO Ku uplink, hovering at 100 m AGL")
ax[1].set_ylabel("Rain + cloud\nattenuation (dB)")
ax[2].set_ylabel("Serving satellite\nelevation (°)")
ax[2].set_xlabel("Time (s)  —  red lines: 15 s slot boundaries (handover interruption)")
for a_ in ax:
    a_.grid(axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_satellite.png"), dpi=130)
plt.close(fig)
print("wrote figures to", os.path.abspath(OUT))
