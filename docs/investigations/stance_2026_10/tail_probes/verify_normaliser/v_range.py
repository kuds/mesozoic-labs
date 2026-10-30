"""Verifier: how much the swap result depends on the sweep range (the tanh network is not linear in z under the start's
statistics, whose 0-6 deg sweep reaches z = 7.1).  Network evaluation on my seed-3042 states only; no simulation.

PYTHONPATH=<verify_probe/code>:<verify_probe> python -B v_range.py > v_range.txt

For every s42 end (12): loop gain (median over the 6 states of the slope x 5.183) fitted over 0-6 deg (the verifier's
definition), 0-3 deg, 3-6 deg and 4.5-6 deg (around the closed-loop operating point, the rail's +5.2 deg), under
(a) own statistics, (b) all 63 dims from the start, (c) tail_3 angle from the start; and the start's weights under its
own and each end's statistics.  Counts of ends with gain > 1 per range; ratio (b)/(a) per range.
"""
import sys

import numpy as np
import torch

from v_analyze import S42, load_states
from v_common import PLANT, SE, START, Net, Norm

torch.set_num_threads(1)
TH = np.arange(0.0, 6.01, 0.25)
RANGES = {"0-6": (0.0, 6.0), "0-3": (0.0, 3.0), "3-6": (3.0, 6.0), "4.5-6": (4.5, 6.0)}


def gains(net, norm, X):
    out = {k: [] for k in RANGES}
    for s in X:
        b = np.repeat(np.asarray(s, dtype=np.float32)[None, :], len(TH), axis=0)
        b[:, 6] = np.radians(TH).astype(np.float32)
        mu = net.mean(np.stack([norm(o) for o in b]))[:, 14]
        for k, (lo, hi) in RANGES.items():
            m = (TH >= lo - 1e-9) & (TH <= hi + 1e-9)
            out[k].append(np.polyfit(TH[m], mu[m], 1)[0])
    return {k: PLANT * float(np.median(v)) for k, v in out.items()}


def main():
    st = load_states("v_ckpts.states_*.npz")
    s0, e0 = SE["s42"]
    netS, nS = Net(START["s42"], s0), Norm(START["s42"], s0)
    XS = st["ck__start_s42__3042"]
    rows = []
    for r in S42:
        netE, nE = Net(r, e0), Norm(r, e0)
        X = st[f"ck__{r}@{e0}__3042"]
        rows.append({"own": gains(netE, nE, X), "all_S": gains(netE, nE.mixed(nS), X),
                     "t3q_S": gains(netE, nE.mixed(nS, dims=(6,)), X),
                     "startW_ownN_XS": gains(netS, nS, XS), "startW_endN_XS": gains(netS, nS.mixed(nE), XS),
                     "startW_ownN_XE": gains(netS, nS, X), "startW_endN_XE": gains(netS, nS.mixed(nE), X)})
    print("loop gain by sweep range, s42 ends (median over 12; [min..max]; # > 1)")
    for c in ("own", "all_S", "t3q_S", "startW_ownN_XS", "startW_endN_XS"):
        line = []
        for k in RANGES:
            v = np.array([d[c][k] for d in rows])
            line.append(f"{k}: {np.median(v):+.3f} [{v.min():+.3f}..{v.max():+.3f}] >1: {int((v > 1).sum())}")
        print(f"  {c:15s} " + " | ".join(line))
    for k in RANGES:
        ra = np.array([d["all_S"][k] / d["own"][k] for d in rows])
        rs = np.array([d["startW_endN_XS"][k] / d["startW_ownN_XS"][k] for d in rows])
        print(f"  range {k:5s}: end W, start/own stats ratio median {np.median(ra):.2f} [{ra.min():.2f}..{ra.max():.2f}] "
              f"({int((ra > 1).sum())}/12 >1); start W, end/own stats ratio {np.median(rs):.2f} ({int((rs < 1).sum())}/12 <1)")
    print("2x2 on X_end by range: symmetric normaliser share of G(We,Ne)-G(Ws,Ns), median over 12 [min..max]; parts")
    for k in RANGES:
        ee = np.array([d["own"][k] for d in rows]); es = np.array([d["all_S"][k] for d in rows])
        se = np.array([d["startW_endN_XE"][k] for d in rows]); ss = np.array([d["startW_ownN_XE"][k] for d in rows])
        sh = 0.5 * ((ee - es) + (se - ss)) / (ee - ss)
        print(f"  {k:5s}: G {np.median(ee):+.3f} / {np.median(es):+.3f} / {np.median(se):+.3f} / {np.median(ss):+.3f}; "
              f"normaliser part at end W {np.median(ee - es):+.3f} ({int((ee - es < 0).sum())}/12 <0), at start W "
              f"{np.median(se - ss):+.3f} ({int((se - ss < 0).sum())}/12 <0); share {np.median(sh):+.2f} [{sh.min():+.2f}..{sh.max():+.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
