"""LinkQuality container and the PHY/MAC abstraction (SINR -> PER, throughput, latency, outage)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .config import LinkSpec
from .util import db, rician_outage_prob, undb

_ARRAY_FIELDS = ("rssi_dbm", "signal_dbm", "noise_dbm", "interference_dbm", "snr_db", "sinr_db", "pathloss_db",
                 "los", "shadow_db", "fading_db", "per", "throughput_mbps", "latency_ms", "latency_p99_ms",
                 "available", "ctrl_reliable", "video_usable", "quality", "outage_prob", "margin_db")


@dataclass
class LinkQuality:
    """Instantaneous quality of one link at N query points. Fields are numpy arrays of shape (N,)
    (or python scalars when the query was a single point)."""
    link_id: str
    kind: str
    t: Any
    rssi_dbm: Any = None            # total received power (signal + interference + noise)
    signal_dbm: Any = None          # wanted-signal power incl. shadowing and fast fading
    noise_dbm: Any = None
    interference_dbm: Any = None
    snr_db: Any = None
    sinr_db: Any = None
    pathloss_db: Any = None
    los: Any = None
    shadow_db: Any = None
    fading_db: Any = None
    per: Any = None                 # packet error rate at the link's operating point
    throughput_mbps: Any = None
    latency_ms: Any = None          # mean one-way latency to the GCS application
    latency_p99_ms: Any = None
    available: Any = None           # link can carry traffic at all
    ctrl_reliable: Any = None       # PER and latency within the control-grade limits
    video_usable: Any = None        # throughput >= min video rate and PER acceptable
    quality: Any = None             # 0..1 convenience score
    outage_prob: Any = None         # P(SINR < threshold) over small-scale fading at this mean level
    margin_db: Any = None           # SINR - decoding threshold
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, with_extra: bool = True) -> dict:
        d = {k: getattr(self, k) for k in ("link_id", "kind", "t") + _ARRAY_FIELDS}
        if with_extra:
            d.update(self.extra)
        return d

    def __getitem__(self, i):
        out = LinkQuality(self.link_id, self.kind, np.asarray(self.t)[i] if np.ndim(self.t) else self.t)
        for k in _ARRAY_FIELDS:
            v = getattr(self, k)
            setattr(out, k, v[i] if np.ndim(v) else v)
        out.extra = {k: (v[i] if np.ndim(v) and not isinstance(v, str) else v) for k, v in self.extra.items()}
        return out


def squeeze_result(lq: LinkQuality) -> LinkQuality:
    """Convert length-1 arrays to python scalars."""
    def sq(v):
        if isinstance(v, np.ndarray) and v.shape == (1,):
            x = v[0]
            return x.item() if hasattr(x, "item") else x
        return v
    for k in ("t",) + _ARRAY_FIELDS:
        setattr(lq, k, sq(getattr(lq, k)))
    lq.extra = {k: sq(v) for k, v in lq.extra.items()}
    return lq


# --------------------------------------------------------------------------- abstraction
def per_curve(sinr_db, req_db, slope_db):
    """Logistic block-error curve: 50 % PER at req_db, ~10->90 % over 4.4*slope dB."""
    return 1.0 / (1.0 + np.exp((np.asarray(sinr_db) - req_db) / max(slope_db, 1e-3)))


def spectral_efficiency(sinr_db, alpha, se_max, floor_db):
    """Truncated Shannon (3GPP TR 36.942 style)."""
    se = alpha * np.log2(1.0 + undb(sinr_db))
    return np.where(np.asarray(sinr_db) < floor_db, 0.0, np.minimum(se, se_max))


def radio_abstraction(spec: LinkSpec, sinr_db, bw_hz, prb_share=1.0, se_alpha=None, se_max=None):
    sinr_db = np.asarray(sinr_db, dtype=float)
    alpha = spec.se_alpha if se_alpha is None else se_alpha
    smax = spec.se_max if se_max is None else se_max
    if spec.mode == "fixed":
        per = per_curve(sinr_db, spec.req_sinr_db, spec.per_slope_db)
        thr = spec.rate_mbps * (1 - per)
    else:
        se = spectral_efficiency(sinr_db, alpha, smax, spec.req_sinr_db)
        ok = sinr_db >= spec.req_sinr_db
        # link adaptation targets ~10 % first-transmission BLER; HARQ leaves a small residual that
        # shrinks with margin above the lowest-MCS threshold
        per = np.where(ok, 0.002 + 0.08 * np.exp(-(sinr_db - spec.req_sinr_db) / 3.0), 1.0)
        thr = se * bw_hz / 1e6 * prb_share * (1 - per)
    return per, thr


def retx_latency(base_ms, retx_ms, per, jitter_ms=0.0):
    """Mean and p99 one-way latency with ARQ: geometric number of attempts."""
    p = np.clip(per, 0.0, 0.999)
    mean = base_ms + retx_ms * p / (1 - p) + jitter_ms
    with np.errstate(divide="ignore", invalid="ignore"):
        n99 = np.where(p > 1e-3, np.ceil(np.log(0.01) / np.log(np.maximum(p, 1e-6))) - 1, 0.0)
    p99 = base_ms + retx_ms * n99 + 2 * jitter_ms
    cap = base_ms + 1000.0
    return np.minimum(mean, cap), np.minimum(p99, cap)


def finalize(spec: LinkSpec, lq: LinkQuality, per, thr, lat, lat99, sinr_db, snr_db, k_lin=None, avail=None,
             mean_sinr_db=None):
    margin = np.asarray(sinr_db) - spec.req_sinr_db
    quality = np.clip(margin / spec.margin_full_db, 0, 1) * (1 - per)
    lq.per, lq.throughput_mbps, lq.latency_ms, lq.latency_p99_ms = per, thr, lat, lat99
    lq.margin_db = margin
    if avail is None:
        avail = per < 0.5
    lq.available = avail
    lq.ctrl_reliable = avail & (per <= spec.reliable_per) & (lat <= spec.latency_limit_ms) & (lat99 <= 1.5 * spec.latency_limit_ms)
    lq.video_usable = avail & (thr >= spec.min_video_mbps) & (per <= 0.1)
    lq.quality = np.where(avail, quality, 0.0)
    k = np.zeros_like(np.asarray(sinr_db, dtype=float)) if k_lin is None else k_lin
    ref = sinr_db if mean_sinr_db is None else mean_sinr_db
    lq.outage_prob = rician_outage_prob(spec.req_sinr_db, ref, k)
    return lq
