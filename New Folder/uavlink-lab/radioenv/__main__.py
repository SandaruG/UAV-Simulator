"""Command line:  python -m radioenv <command> ...

  describe [--preset NAME] [--config FILE]
  config   [--preset NAME]                     print the full JSON config (edit it, pass back with --config)
  query    --link ID --x X --y Y --z Z [--t T] [--preset NAME] [--config FILE]
  map      --link ID --out FILE.png [--z-agl 100] [--t 0] [--preset NAME] [--config FILE]
"""
import argparse
import json
import sys

import numpy as np

from . import RadioEnvConfig, RadioEnvironment, presets


def _cfg(a) -> RadioEnvConfig:
    if getattr(a, "config", None):
        return RadioEnvConfig.from_json(a.config)
    return presets.get(a.preset or "demo", seed=a.seed, weather=a.weather)


def main(argv=None):
    p = argparse.ArgumentParser(prog="radioenv", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--preset", choices=presets.names(), default=None)
        sp.add_argument("--config", help="JSON config file (overrides --preset)")
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--weather", default="clear")

    for name in ("describe", "config", "query", "map"):
        common(sub.add_parser(name))
    q = sub.choices["query"]
    q.add_argument("--link", required=True)
    for k in "xyz":
        q.add_argument(f"--{k}", type=float, required=True)
    q.add_argument("--t", type=float, default=0.0)
    q.add_argument("--agl", action="store_true", help="treat --z as metres above the surface instead of MSL")
    m = sub.choices["map"]
    m.add_argument("--link", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--z-agl", type=float, default=100.0)
    m.add_argument("--t", type=float, default=0.0)
    a = p.parse_args(argv)

    cfg = _cfg(a)
    if a.cmd == "config":
        print(cfg.to_json())
        return 0
    env = RadioEnvironment(cfg)
    if a.cmd == "describe":
        print(json.dumps(env.describe(), indent=2, default=str))
    elif a.cmd == "query":
        z = a.z + (float(env.surface_height(a.x, a.y)) if a.agl else 0.0)
        lq = env.query(a.link, np.array([a.x, a.y, z]), a.t)
        d = {k: (v.item() if hasattr(v, "item") else v) for k, v in lq.to_dict().items()}
        print(json.dumps(d, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    elif a.cmd == "map":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mp = env.coverage_map(a.link, z_agl=a.z_agl, t=a.t, nx=120, ny=120)
        g = env.gd.grid
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mp["sinr_db"], origin="lower", extent=[g.x0, g.x1, g.y0, g.y1], cmap="Blues")
        fig.colorbar(im, label="SINR (dB)")
        ax.set_title(f"{a.link} SINR at {a.z_agl:.0f} m AGL, t={a.t:.0f} s")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        fig.savefig(a.out, dpi=130, bbox_inches="tight")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
