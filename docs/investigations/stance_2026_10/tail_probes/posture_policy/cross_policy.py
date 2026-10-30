"""Test 2, across policies (written on resume).  Read only; plain numpy; no simulation.

python -B cross_policy.py > cross_policy.txt

Within each lineage's 12 ends (n = 12; correlational, descriptive): Spearman rank correlations between
  - the policy's own held pelvis pitch (baseline, window 5-10 s, median over 20 seeds; run/base),
  - its tail offset c0 and loop gain G (sweep.json, section D "none"),
  - its pitch pathway (sweep.json B: d cmd14 / d nose-up deg, all channels),
  - its rail share (fraction of seeds with window own14 mean >= 0.99) and raw tail_3 command (run/base),
  - its closed-loop response to the "toward" illusion (run/toward_*: median over seeds of illusion minus baseline raw14),
  - the offset drop under the ORIENT+LEGS+FEET transplant from the other lineage's ends (sweep_extra.json).
Exact permutation p-values are not computed; with n = 12, |rho| >= 0.59 corresponds to two-sided p < 0.05 (table value).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    for v in np.unique(x):
        idx = np.where(x == v)[0]
        if len(idx) > 1:
            r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def load(d):
    out = defaultdict(dict)
    p = HERE / d / "episodes.jsonl"
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            out[r["who"]][r["seed"]] = r
    return out


def main():
    sw = json.load(open(HERE / "sweep.json"))["policies"]
    sx = json.load(open(HERE / "sweep_extra.json"))["policies"]
    base = load("run/base")
    for lin in ("s42", "s80"):
        tw = load(f"run/toward_{lin}")
        ws = [w for w in sw if w.startswith(lin)]
        col = {
            "held_pitch_deg": [np.median([r["pitch_deg"] for r in base[w].values()]) for w in ws],
            "offset_c0": [sw[w]["D_transplant"]["none"]["offset_c0"] for w in ws],
            "loop_gain_G": [sw[w]["D_transplant"]["none"]["gain"] for w in ws],
            "pitch_path(dcmd/d noseup)": [sw[w]["B_families"]["pitch_illusion_all(per deg nose-up)"]["cmd14_slope"] for w in ws],
            "rail_share": [np.mean([r["own_mean"][14] >= 0.99 for r in base[w].values()]) for w in ws],
            "raw14_base": [np.median([r["raw14_mean"] for r in base[w].values()]) for w in ws],
            "d_raw14_toward": [np.median([tw[w][s]["raw14_mean"] - base[w][s]["raw14_mean"] for s in tw[w]]) for w in ws],
            "d_c0_orient_legs_transplant": [sx[w]["other_ends:ORIENT+LEGS+FEET"]["offset_c0"] - sx[w]["none"]["offset_c0"]
                                            for w in ws],
        }
        print("=" * 100)
        print(f"{lin} ends (n = {len(ws)}): values")
        for w_i, w in enumerate(ws):
            print(f"   {w:11s} " + " ".join(f"{k} {v[w_i]:+.3f}" for k, v in col.items()))
        print(f"{lin} ends: Spearman rho (n = {len(ws)}; |rho| >= 0.59 ~ p < 0.05 two-sided; correlational)")
        ks = list(col)
        for i, a in enumerate(ks):
            print("   " + f"{a:28s}" + " ".join(f"{spearman(col[a], col[b]):+.2f}" if j > i else "  .  " for j, b in enumerate(ks)))
        print("   columns: " + ", ".join(ks))


if __name__ == "__main__":
    main()
