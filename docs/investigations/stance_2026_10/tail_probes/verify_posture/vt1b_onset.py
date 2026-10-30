"""VERIFY TEST 1, part b: does the "together" timing depend on thresholds?  Read only; no simulation.

Two threshold-free / noise-matched onset estimates, stated before running:
  (1) noise-matched detection: 65,536-step bin means (8 rollouts) of each probe signal; reference = bins in
      [0.524M, 1.049M); a departure is the bin mean beyond ref_mean +- k * ref_SD (k = 2, 3) in the named direction,
      sustained for 4 consecutive bins (262,144 steps); reported as the first bin end of the sustained run.
      Training-episode CoM / pitch use 65,536-step bin medians the same way.
  (2) hinge fit on a short window [0.524M, 2.10M] (both signals near-linear there), t0 profiled.
Runs: r10 s42 (rise), s42 re-formed rail (reference [6.55M, 6.82M)), s60 main rise (reference [5.0M, 5.6M)).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_posture")
from vt1_timing_lib import f2, probe, train_eps  # noqa: E402

R2D = 180 / math.pi
OUT = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/verify_posture"


def bins(t, y, lo, hi, w=65536, agg=np.mean):
    e = np.arange(lo, hi + 1, w)
    out = []
    for a, b in zip(e[:-1], e[1:]):
        s = (t > a) & (t <= b) & ~np.isnan(y)
        out.append(agg(y[s]) if s.any() else np.nan)
    return e[1:], np.array(out)


def detect(te, ye, ref_lo, ref_hi, direction, k, run=4):
    ref = (te > ref_lo) & (te <= ref_hi)
    m, sd = np.nanmean(ye[ref]), np.nanstd(ye[ref], ddof=1)
    ok = (direction * (ye - m) > k * sd) & (te > ref_hi)
    c = 0
    for i, v in enumerate(ok):
        c = c + 1 if v else 0
        if c >= run:
            return int(te[i - run + 1]), m, sd
    return None, m, sd


def hinge(t, y, lo, hi, step):
    s = (t >= lo) & (t <= hi) & ~np.isnan(y)
    tt, yy = t[s].astype(float), y[s]
    best = None
    for t0 in np.arange(lo, hi - 3 * step, step):
        X = np.column_stack([np.ones(len(tt)), np.maximum(0, tt - t0)])
        c, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ c
        if best is None or r @ r < best[1]:
            best = (int(t0), float(r @ r), float(c[1]))
    return best


L = []
P = L.append
P("VERIFY TEST 1b: noise-matched onset detection and short-window hinge (vt1b_onset.py)")
for run, ref_lo, ref_hi, hi, dirs in (
        ("s42", 524288, 1048576, 3670016, {"tail mu": +1, "q3 angle": +1, "pitch": -1, "CoM train": +1, "pitch train": -1,
                                           "neck": +1, "head joint": +1}),
        ("s42", 6553600, 6815744, 7602176, {"tail mu": +1, "q3 angle": +1, "pitch": -1, "CoM train": +1, "pitch train": -1,
                                            "neck": +1, "head joint": -1}),
        ("s60", 4980736, 5636096, 8126464, {"tail mu": +1, "q3 angle": +1, "pitch": +1, "CoM train": -1, "pitch train": +1,
                                            "neck": -1, "head joint": -1})):
    pr = probe(f2(run, "policy_probe.npz"))
    tt, tcom, tpit = train_eps(f2(run, "train_episodes.csv"))
    pitch = np.degrees(np.arcsin(np.clip(-pr["bm"][:, 42], -1, 1)))
    sig = {"tail mu": (pr["t"], pr["mu"][:, 14], np.mean), "q3 angle": (pr["t"], pr["bm"][:, 6] * R2D, np.mean),
           "pitch": (pr["t"], pitch, np.mean), "CoM train": (tt, tcom, np.median), "pitch train": (tt, tpit, np.median),
           "neck": (pr["t"], pr["bm"][:, 0] * R2D, np.mean), "head joint": (pr["t"], pr["bm"][:, 2] * R2D, np.mean)}
    P(f"\n{run}: reference ({ref_lo/1e6:.3f}, {ref_hi/1e6:.3f}], search to {hi/1e6:.3f}M; direction as named (+1 up)")
    for name, (t, y, agg) in sig.items():
        te, ye = bins(t, y, ref_lo - ((ref_lo) % 65536), hi, agg=agg)
        res = []
        for k in (2, 3, 5):
            d, m, sd = detect(te, ye, ref_lo, ref_hi, dirs[name], k)
            res.append(f"k{k} {d/1e6:.3f}M" if d else f"k{k} -")
        P(f"  {name:12s} dir {dirs[name]:+d} ref {m:+.3f} sd {sd:.3f}: " + ", ".join(res))
    if run == "s42" and ref_lo == 524288:
        P("  hinge on [0.524M, 2.10M] (t0, slope/M):")
        for name, (t, y, agg) in sig.items():
            if "train" in name:
                te, ye = bins(t, y, 0, 2162688, agg=agg)
                b = hinge(te, ye, 524288, 2097152, 65536)
            else:
                b = hinge(t, y, 524288, 2097152, 8192)
            P(f"    {name:12s} t0 {b[0]/1e6:.3f}M slope {b[2]*1e6:+.3f}/M")
txt = "\n".join(L)
open(f"{OUT}/vt1b_onset.txt", "w").write(txt + "\n")
print(txt)
