"""Verifier: the lens's numbers re-derived from recompute.json (this folder's own reduction of the lens's tracks).

python -I -B verify_claims.py recompute.json PR_POLICY_DEMO_JSONL > verify_claims.txt
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations

import numpy as np
from scipy.stats import binom, mannwhitneyu, spearmanr

D = json.load(open(sys.argv[1]))
mine, lens = D["mine"], D["lens"]
M1 = [str(s) for s in range(3042, 3062)]
HELD = [str(s) for s in range(3062, 3072)]


def p_sign(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def vals(run, who, key, seeds, W="5.0"):
    return np.array([mine[run][who][s][W][key] for s in seeds])


def P(*a):
    print(*a)


policies = [w for w in mine["run"] if w != "statue"]
starts = [w for w in policies if w.startswith("start_")]
s42 = [w for w in policies if w.startswith("s42_")]
s80 = [w for w in policies if w.startswith("s80_")]
P(f"policies {len(policies)} (starts {len(starts)}, s42 {len(s42)}, s80 {len(s80)}); M1 episodes stood (len 1001): "
  f"{sum(v is not None for w in mine['run'] for v in mine['run'][w].values())}/540")

# ── 0. my reduction against the lens's in-run metrics ───────────────────────
P("\n0. MY REDUCTION (float32 tracks) AGAINST THE LENS'S IN-RUN METRICS (float64), window 5 s, all 920 episodes")
pairs = {"com_lat_mm": ("com_lateral_mean_m", 1000), "roll_deg": ("root_roll_mean_deg", 1), "pitch_deg": ("root_pitch_mean_deg", 1),
         "com_fa_mm": ("com_fore_aft_mean_m", 1000), "speed_rms_mmps": ("com_speed_rms_mps", 1000), "turn_window_deg": ("root_turn_deg", 1)}
for mk, (lk, sc) in pairs.items():
    diffs, signflips, n = [], 0, 0
    for run in mine:
        for who in mine[run]:
            for s, ep in mine[run][who].items():
                lm = lens[run][who][s]["5.0"]
                a, b = ep["5.0"][mk], lm[lk] * sc
                diffs.append(abs(a - b))
                n += 1
                signflips += (np.sign(a) != np.sign(b))
    P(f"  {mk:16s} vs {lk:26s}: max |diff| {max(diffs):.2e}  sign disagreements {signflips}/{n}")
d = [abs(mine[r][w][s]["turn_from_spawn_deg"] - lens[r][w][s]["5.0"]["root_turn_from_spawn_deg"]) for r in mine for w in mine[r] for s in mine[r][w]]
P(f"  turn_from_spawn  max |diff| {max(d):.2e} deg")

# ── 1. POL-1 / POL-5: sign tests, CoM-P1 flags, creep ───────────────────────
P("\n1. SAME-SIDE LEAN (sign test) AND CREEP, my numbers")
for W in ("2.0", "5.0", "9.0"):
    best = []
    for who in policies:
        for key, fl in (("com_lat_mm", 1.0), ("roll_deg", 0.10)):
            v = vals("run", who, key, M1, W)
            pos, neg = int((v > 0).sum()), int((v < 0).sum())
            best.append((p_sign(pos, neg), who, key, pos, neg, float(np.median(v)), bool(p_sign(pos, neg) <= 0.005 and abs(np.median(v)) >= fl)))
    best.sort()
    flagged = [b for b in best if b[6]]
    P(f"  window {W}: keys at p<=0.005 (no floor): {sum(b[0] <= 0.005 for b in best)}/52; CoM-P1 flags (with floor): {len(flagged)}; "
      f"smallest p: " + "; ".join(f"{w} {k} {p_:+d}/{n_:d} p {p:.4f} med {m:+.4f}" for p, w, k, p_, n_, m, _ in best[:3]))
creep_m1 = max(mine["run"][w][s]["5.0"]["creep_mmps"] for w in policies for s in M1)
arg = max(((mine["run"][w][s]["5.0"]["creep_mmps"], w, s) for w in policies for s in M1))
creep_h = max(((mine["run_held"][w][s]["5.0"]["creep_mmps"], w, s) for w in mine["run_held"] for s in HELD))
creep_st = max([mine["run"]["statue"][s]["5.0"]["creep_mmps"] for s in M1] + [ep["5.0"]["creep_mmps"] for ep in mine["run_null"]["statue"].values()])
n_ge1 = sum(mine[r][w][s]["5.0"]["creep_mmps"] >= 1.0 for r in ("run", "run_held") for w in mine[r] if w != "statue" for s in mine[r][w])
P(f"  feet creep LS speed 5-10 s: policies M1 max {creep_m1:.3f} mm/s ({arg[1]} {arg[2]}); held-out max {creep_h[0]:.3f} ({creep_h[1]} {creep_h[2]}); "
  f"statue (M1 + null, 120) max {creep_st:.3f}; policy episodes >= 1 mm/s: {n_ge1}")
# held-out CoM-P1 (n=10)
hf = []
for who in mine["run_held"]:
    for key, fl in (("com_lat_mm", 1.0), ("roll_deg", 0.10)):
        v = vals("run_held", who, key, HELD)
        pos, neg = int((v > 0).sum()), int((v < 0).sum())
        if p_sign(pos, neg) <= 0.005:
            hf.append((who, key, pos, neg, float(np.median(v)), abs(np.median(v)) >= fl))
P(f"  held-out 3062-3071: keys at p<=0.005: {hf}")
# statue panels
st_all = {**{s: mine["run"]["statue"][s] for s in M1}, **mine["run_null"]["statue"]}
seeds_null = sorted(mine["run_null"]["statue"], key=int)
panels = [M1] + [seeds_null[i:i + 20] for i in range(0, 100, 20)]
ps = []
for pan in panels:
    for key in ("com_lat_mm", "roll_deg"):
        v = np.array([st_all[s]["5.0"][key] for s in pan])
        ps.append(p_sign(int((v > 0).sum()), int((v < 0).sum())))
P(f"  statue: 6 panels x 2 keys, smallest sign-test p {min(ps):.4f}; keys at p<=0.005: {sum(p <= 0.005 for p in ps)}/12")
for key in ("com_lat_mm", "roll_deg"):
    v = np.array([st_all[s]["5.0"][key] for s in st_all])
    P(f"  pooled statue {key}: {int((v > 0).sum())}+/{int((v < 0).sum())}- of {v.size}, p {p_sign(int((v > 0).sum()), int((v < 0).sum())):.3f}; |v| 95th pct {np.percentile(np.abs(v), 95):.4f}")
fa = np.array([st_all[s]["5.0"]["creep_fa_mmps"] for s in st_all])
P(f"  statue feet creep direction: {int((fa > 0).sum())} forward / {int((fa < 0).sum())} backward of {fa.size}")

# ── 2. POL-2: r9 ─────────────────────────────────────────────────────────────
P("\n2. r9 POSITIVE CONTROL")
r9 = mine["run_r9"]["r9_4194304"]
for key in ("com_lat_mm", "roll_deg"):
    v = np.array([r9[s]["5.0"][key] for s in M1])
    P(f"  r9 5 s {key}: {int((v > 0).sum())}+/{int((v < 0).sum())}-, p {p_sign(int((v > 0).sum()), int((v < 0).sum())):.2e}, median {np.median(v):+.4f}")
demo = [json.loads(line) for line in open(sys.argv[2])]
dpol = {r["seed"]: r["metrics"] for r in demo if r.get("who") == "policy" and "seed" in r}
mine2 = np.array([r9[str(s)]["2.0"]["roll_deg"] for s in range(3042, 3052)])
lens2 = np.array([lens["run_r9"]["r9_4194304"][str(s)]["2.0"]["root_roll_mean_deg"] for s in range(3042, 3052)])
pr2 = np.array([dpol[s]["root_roll_mean_deg"] for s in range(3042, 3052)])
P(f"  2 s roll seeds 3042-3051: PR demo median {np.median(pr2):.5f} min {pr2.min():.5f} max {pr2.max():.5f}; lens run_r9 max|diff to PR| {np.max(np.abs(lens2 - pr2)):.2e}; "
  f"mine (float32 tracks) max|diff to PR| {np.max(np.abs(mine2 - pr2)):.2e}")
prc = np.array([dpol[s]["com_lateral_mean_m"] for s in range(3042, 3052)])
lc = np.array([lens["run_r9"]["r9_4194304"][str(s)]["2.0"]["com_lateral_mean_m"] for s in range(3042, 3052)])
P(f"  2 s com_lateral seeds 3042-3051: lens vs PR max|diff| {np.max(np.abs(lc - prc)):.2e} m")

# ── 3. POL-4: groups ─────────────────────────────────────────────────────────
P("\n3. RAIL (s42) vs NO RAIL (s80): per-policy medians over M1, exact two-sided Mann-Whitney")
med = {w: {k: float(np.median(vals("run", w, k, M1))) for k in ("com_lat_mm", "roll_deg", "pitch_deg", "com_fa_mm", "speed_rms_mmps")} for w in mine["run"]}
for k in ("com_lat_mm", "roll_deg", "pitch_deg", "com_fa_mm"):
    a, b = [med[w][k] for w in s42], [med[w][k] for w in s80]
    P(f"  {k:12s} s42 {np.median(a):+8.3f}  s80 {np.median(b):+8.3f}  p {mannwhitneyu(a, b, method='exact').pvalue:.5f}   "
      f"start_s42 {med['start_s42'][k]:+8.3f} start_s80 {med['start_s80'][k]:+8.3f}")
sd = {w: {k: float(np.std(vals("run", w, k, M1), ddof=1)) for k in ("com_lat_mm", "roll_deg")} for w in mine["run"]}
for k in ("com_lat_mm", "roll_deg"):
    a, b = [sd[w][k] for w in s42], [sd[w][k] for w in s80]
    P(f"  SD over seeds {k:10s}: statue {sd['statue'][k]:.4f}; s42 median {np.median(a):.4f} [{min(a):.4f},{max(a):.4f}]; s80 median {np.median(b):.4f} "
      f"[{min(b):.4f},{max(b):.4f}]; p {mannwhitneyu(a, b, method='exact').pvalue:.4f}; starts {sd['start_s42'][k]:.4f} / {sd['start_s80'][k]:.4f}")
# change from start, by lineage (the rail's effect would show here)
for k in ("pitch_deg", "com_fa_mm"):
    da = [med[w][k] - med["start_s42"][k] for w in s42]
    db = [med[w][k] - med["start_s80"][k] for w in s80]
    P(f"  change from own start {k}: s42 median {np.median(da):+.3f}, s80 median {np.median(db):+.3f} (start-to-end shift of each lineage)")
# A vs B
for g in (s42, s80):
    A = [w for w in g if w[4] == "A"]
    B = [w for w in g if w[4] == "B"]
    out = []
    for k in ("com_lat_mm", "roll_deg", "pitch_deg", "com_fa_mm", "speed_rms_mmps"):
        out.append(f"{k} p {mannwhitneyu([med[w][k] for w in A], [med[w][k] for w in B], method='exact').pvalue:.3f}")
    P(f"  arm A vs B within {g[0][:3]}: " + "; ".join(out))

# ── 4. POL-6: spawn-dependent lean ───────────────────────────────────────────
P("\n4. SPAWN-DEPENDENT LEAN")
for g, ws in (("s42", s42), ("s80", s80)):
    rho = [spearmanr(vals("run", x, "com_lat_mm", M1), vals("run", y, "com_lat_mm", M1))[0] for x, y in combinations(ws, 2)]
    rst = [spearmanr(vals("run", "statue", "com_lat_mm", M1), vals("run", w, "com_lat_mm", M1))[0] for w in ws]
    big = sum(abs(v) > 5 for w in ws for v in vals("run", w, "com_lat_mm", M1))
    grp_med = np.median([med[w]["com_lat_mm"] for w in ws])
    P(f"  {g}: mean pairwise Spearman {np.mean(rho):+.2f}; per-policy rho vs statue median {np.median(rst):+.2f}; |CoM lat| > 5 mm: {big}/{len(ws) * 20}; "
      f"median of per-policy medians {grp_med:+.3f} mm")
st = vals("run", "statue", "com_lat_mm", M1)
P(f"  statue: |CoM lat| > 5 mm {int((np.abs(st) > 5).sum())}/20, max |.| {np.abs(st).max():.3f} mm")
at59 = sorted(((mine["run"][w]["3059"]["5.0"]["com_lat_mm"], w) for w in policies))[:6]
P(f"  seed 3059 most-right policies: " + ", ".join(f"{w} {v:+.2f}" for v, w in at59) + f"; statue {mine['run']['statue']['3059']['5.0']['com_lat_mm']:+.3f}")
# where do the >5 mm episodes come from
from collections import Counter
c = Counter(s for w in s80 for s in M1 if abs(mine["run"][w][s]["5.0"]["com_lat_mm"]) > 5)
P(f"  s80 > 5 mm episodes by seed: {dict(c)}")
# is it held? the same keys at 9 s
held9 = [(w, mine['run'][w]['3059']['9.0']['com_lat_mm']) for _, w in at59[:4]]
P(f"  the same four at window 9-10 s: " + ", ".join(f"{w} {v:+.2f}" for w, v in held9))

# ── 5. POL-7: rock, settle, turn ─────────────────────────────────────────────
P("\n5. ROCK, SETTLE, TURN FROM SPAWN")
for g, ws in (("statue", ["statue"]), ("starts", starts), ("s42", s42), ("s80", s80)):
    sp = [mine["run"][w][s]["5.0"]["speed_rms_mmps"] for w in ws for s in M1]
    se = np.array([mine["run"][w][s]["settle_s"] for w in ws for s in M1])
    tf = np.array([mine["run"][w][s]["turn_from_spawn_deg"] for w in ws for s in M1])
    P(f"  {g:7s} CoM speed RMS 5-10 s median {np.median(sp):.3f} mm/s; settled by 3 s {np.mean(se <= 3):.2f}, by 5 s {np.mean(se <= 5):.2f}; "
      f"|turn from spawn| < 1 deg {np.mean(np.abs(tf) < 1):.3f}, max {np.abs(tf).max():.1f} deg")
st_turn = np.array([mine["run"]["statue"][s]["turn_from_spawn_deg"] for s in M1])
rhos = [spearmanr(st_turn, [mine["run"][w][s]["turn_from_spawn_deg"] for s in M1])[0] for w in policies]
P(f"  turn from spawn, policy vs statue at the same spawn: Spearman median {np.median(rhos):+.2f} [{min(rhos):+.2f}, {max(rhos):+.2f}] over 26 policies")
diff = np.array([mine["run"][w][s]["turn_from_spawn_deg"] - mine["run"]["statue"][s]["turn_from_spawn_deg"] for w in policies for s in M1])
P(f"  policy minus statue turn: median |.| {np.median(np.abs(diff)):.2f} deg, 90th pct {np.percentile(np.abs(diff), 90):.2f}")

# ── 6. COMMANDS (hip roll sum) ───────────────────────────────────────────────
P("\n6. HIP-ROLL COMMANDS (actuators 4, 8 as the lens names them), mean over M1 seeds of the 2-10 s mean")
sums, each = [], []
for w in policies:
    cm = np.array([mine["run"][w][s]["cmd_from_2s"] for s in M1])
    sums.append(cm[:, 4].mean() + cm[:, 8].mean())
    each.append(max(abs(cm[:, 4].mean()), abs(cm[:, 8].mean())))
P(f"  |r + l| max {max(np.abs(sums)):.4f}; max |each| {max(each):.4f}")

# ── 7. analytic null rates ───────────────────────────────────────────────────
P("\n7. ANALYTIC NULL RATES")
p20 = 2 * binom.sf(16, 20, 0.5)
p40 = 2 * binom.sf(29, 40, 0.5)
P(f"  P(>= 17 of 20 one way) = {p20:.6f} ({round(p20 * 2**20)}/1048576); union of 2 keys <= {2 * p20:.5f}; P(>= 30 of 40 one way) = {p40:.5f}")
