"""VERIFY TEST 1, part c.  Read only; no simulation.
(1) r10 s42: moving-block bootstrap (blocks of 8 rollouts, 400 resamples, seed 0) of the short-window hinge t0
    ([0.524M, 2.10M]) for tail mu, probe pitch and q3; reports t0(pitch) - t0(tail) quantiles.  Residual
    bootstrap around the fitted hinge, so it carries the noise's autocorrelation only within blocks.
(2) ECS: per continuation, from the probe rows after S: 65,536-step bins (8 rollouts) of tail mu and pitch;
    first differences; cross-correlation at lags -3..+3, per continuation and pooled per lineage.  Also the
    noise-matched departure (k = 3 SD of the first 4 bins' first differences, cumulative) is NOT used: the
    continuations start mid-rise, so there is no flat reference.  Instead: for two named continuations
    (s42_A_1101, s42_B_1211) the 65,536-step series are printed.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_posture")
from vt1_timing_lib import f2, probe, ECS  # noqa: E402

OUT = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_posture"
R2D = 180 / math.pi
L = []
P = L.append


def hinge_fit(tt, yy, grid):
    best = None
    for t0 in grid:
        X = np.column_stack([np.ones(len(tt)), np.maximum(0, tt - t0)])
        c, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ c
        s = float(r @ r)
        if best is None or s < best[0]:
            best = (s, t0, c)
    return best


pr = probe(f2("s42", "policy_probe.npz"))
pitch = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
sel = (pr["t"] >= 524288) & (pr["t"] <= 2097152)
tt = pr["t"][sel].astype(float)
grid = np.arange(524288, 2097152 - 3 * 8192, 8192, dtype=float)
rng = np.random.default_rng(0)
sigs = {"tail mu": pr["mu"][sel, 14], "pitch": pitch[sel], "q3": pr["bm"][sel, 6] * R2D}
fits = {k: hinge_fit(tt, y, grid) for k, y in sigs.items()}
P("1. r10 s42 hinge on [0.524M, 2.10M], moving-block residual bootstrap (block 8 rollouts, 400 resamples)")
for k, (s, t0, c) in fits.items():
    P(f"   point: {k:8s} t0 {t0/1e6:.3f}M slope {c[1]*1e6:+.3f}/M")
n = len(tt)
boots = {k: [] for k in sigs}
for b in range(400):
    starts = rng.integers(0, n - 8, size=n // 8 + 1)
    idx = np.concatenate([np.arange(s, s + 8) for s in starts])[:n]
    for k, y in sigs.items():
        s, t0, c = fits[k]
        X = np.column_stack([np.ones(n), np.maximum(0, tt - t0)])
        fitted = X @ c
        res = (y - fitted)[idx]
        yb = fitted + res
        boots[k].append(hinge_fit(tt, yb, grid)[1])
for k in sigs:
    q = np.quantile(boots[k], [0.025, 0.5, 0.975])
    P(f"   boot {k:8s} t0 2.5/50/97.5%: {q[0]/1e6:.3f} / {q[1]/1e6:.3f} / {q[2]/1e6:.3f}M")
d = np.array(boots["pitch"]) - np.array(boots["tail mu"])
q = np.quantile(d, [0.025, 0.5, 0.975])
P(f"   t0(pitch) - t0(tail mu): 2.5/50/97.5% {q[0]/1e6:+.3f} / {q[1]/1e6:+.3f} / {q[2]/1e6:+.3f}M; share > 0: {np.mean(d > 0):.3f}")
P("   (paired resamples use the same block indices for both signals)")
P("")

P("2. ECS continuations: first-difference cross-correlation of tail mu and probe pitch, 65,536-step bins after S")
P("   r(lag k) = corr(d tail[t], d pitch[t+k]); k > 0 = pitch after tail")
for lin, S in (("s42", 3145728), ("s80", 4194304)):
    pooled_a, pooled_b = {k: [] for k in range(-3, 4)}, {k: [] for k in range(-3, 4)}
    rows = []
    for nm in sorted(x for x in os.listdir(ECS) if x.startswith(lin)):
        p = probe(f"{ECS}/{nm}/01_stance/policy_probe.npz")
        pit = np.degrees(np.arcsin(np.clip(-p["bm"][:, 42], -1, 1)))
        e = np.arange(S, p["t"][-1] + 1, 65536)
        a = np.array([p["mu"][(p["t"] > x) & (p["t"] <= y), 14].mean() for x, y in zip(e[:-1], e[1:])])
        b = np.array([pit[(p["t"] > x) & (p["t"] <= y)].mean() for x, y in zip(e[:-1], e[1:])])
        da, db = np.diff(a), np.diff(b)
        r = {}
        for k in range(-3, 4):
            x, y = (da[: len(da) - k], db[k:]) if k >= 0 else (da[-k:], db[: len(db) + k])
            r[k] = np.corrcoef(x, y)[0, 1]
            # pool standardized within continuation
            pooled_a[k].extend(((x - x.mean()) / x.std()).tolist())
            pooled_b[k].extend(((y - y.mean()) / y.std()).tolist())
        rows.append(r)
        P(f"   {nm}: n {len(da)}  " + "  ".join(f"{k:+d}:{r[k]:+.2f}" for k in range(-3, 4)))
        if nm in ("s42_A_1101", "s42_B_1211"):
            P("      bins (end M, tail mu, pitch): " + "  ".join(f"{y/1e6:.3f}:{u:+.3f}/{v:+.2f}" for y, u, v in zip(e[1:], a, b)))
    med = {k: np.median([r[k] for r in rows]) for k in range(-3, 4)}
    pool = {k: np.corrcoef(pooled_a[k], pooled_b[k])[0, 1] for k in range(-3, 4)}
    P(f"   {lin} median over continuations: " + "  ".join(f"{k:+d}:{med[k]:+.2f}" for k in range(-3, 4)))
    P(f"   {lin} pooled (within-standardized, n {len(pooled_a[0])} at lag 0): " + "  ".join(f"{k:+d}:{pool[k]:+.2f}" for k in range(-3, 4)))
txt = "\n".join(L)
open(f"{OUT}/vt1c_boot_ecs.txt", "w").write(txt + "\n")
print(txt)
