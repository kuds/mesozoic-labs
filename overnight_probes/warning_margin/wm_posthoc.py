"""POST HOC (not pre-registered) descriptive views of episodes.jsonl: approach rate of the bias margin between
checkpoints, predictor errors on the near-rail subset (M < 0.4), within-checkpoint seed spread against the seed's
baseline command, and the per-seed outliers.  No simulation.  Writes wm_posthoc.txt."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import wm_common as C

SEEDS = list(range(3042, 3062))
summ = json.loads((C.HERE / "wm_summary.json").read_text())
pred = json.loads((C.HERE / "wm_predict.json").read_text())
eps = [json.loads(x) for x in (C.HERE / "episodes.jsonl").read_text().splitlines() if x.strip()]
by = defaultdict(list)
for e in eps:
    by[(e["run"], e["ckpt_steps"], e["seed"])].append(e)
LAB = {"s80 10.49M": ("s80", 10_485_760), "s80 11.53M": ("s80", 11_534_336), "s80 12.00M": ("s80", 12_001_280),
       "s60 6.29M*": ("s60", 6_291_456), "s60 7.34M+": ("s60", 7_340_032), "s42 2.10M*": ("s42", 2_097_152),
       "s42 3.15M+": ("s42", 3_145_728)}
FOB = {"s42": 3_670_016, "s60": 8_126_464, "s80": None}
L = []
L.append("POST HOC (not in PREREG.txt).  M = pre-registered bias margin (11th smallest of 20 per-seed thresholds).")
L.append("")
L.append("A. Approach rate of M between measured checkpoints (per 1,048,576 steps); FOB = first tail over-bar episode")
for run, labs in (("s42", ["s42 2.10M*", "s42 3.15M+"]), ("s60", ["s60 6.29M*", "s60 7.34M+"]),
                  ("s80", ["s80 10.49M", "s80 11.53M", "s80 12.00M"])):
    for a, b in zip(labs, labs[1:]):
        ma, mb = summ["ckpts"][a]["M_bias"], summ["ckpts"][b]["M_bias"]
        da = LAB[b][1] - LAB[a][1]
        rate = (mb - ma) / da * 1_048_576
        tail = ""
        if FOB[run]:
            tail = f" | FOB {FOB[run]:,} = {(FOB[run] - LAB[b][1]) / 1e6:.2f}M after {b}"
        else:
            tail = f" | at this rate M reaches 0 after a further {mb / -rate:.1f} x 1.05M steps" if rate < 0 else ""
        L.append(f"  {run}: {a} M {ma:.3f} -> {b} M {mb:.3f}: {rate:+.3f} per 1.05M{tail}")
L.append("")
L.append("B. Predictor errors (M - pred) on the near-rail subset M < 0.4 (4 checkpoints) and excluding s42 2.10M (6)")
near = [k for k in LAB if summ["ckpts"][k]["M_bias"] < 0.4]
no210 = [k for k in LAB if k != "s42 2.10M*"]
for key in ("bstar_P", "bstar_L", "bstar_S", "gap_open"):
    for nm, sub in (("near", near), ("excl 2.10M", no210)):
        errs = [summ["ckpts"][k]["M_bias"] - pred[f"{LAB[k][0]}_{LAB[k][1]}"][key] for k in sub]
        L.append(f"  {key:9s} {nm:10s} (n {len(sub)}): MAE {np.mean(np.abs(errs)):.3f}, mean signed {np.mean(errs):+.3f}, "
                 f"range [{min(errs):+.3f}, {max(errs):+.3f}]")
L.append("")
L.append("C. Within a checkpoint: per-seed bias threshold T_s against the seed's baseline window command (Spearman),")
L.append("   and the seed with the largest T_s")


def rank(x):
    x = np.asarray(x, float)
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


for k, (run, steps) in LAB.items():
    T, base, t3 = [], [], []
    for s in SEEDS:
        rr = by[(run, steps, s)]
        b = [r for r in rr if r["kind"] == "none"][0]
        parked = [r["value"] for r in rr if r["kind"] == "bias" and r["parked"]]
        T.append(min(parked) if parked else 2.0)
        base.append(b["cmd_win"])
        t3.append(b["tail3_deg_win"])
    rho = float(np.corrcoef(rank(T), rank(base))[0, 1])
    i = int(np.argmax(T))
    L.append(f"  {k:11s}: rho(T_s, base cmd) {rho:+.2f} | max T_s {T[i]:.3f} at seed {SEEDS[i]} (base cmd {base[i]:+.3f}, "
             f"tail_3 {t3[i]:+.2f} deg; checkpoint median base {np.median(base):+.3f}) | T_s IQR "
             f"[{np.quantile(T, .25):.3f}, {np.quantile(T, .75):.3f}]")
(C.HERE / "wm_posthoc.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
