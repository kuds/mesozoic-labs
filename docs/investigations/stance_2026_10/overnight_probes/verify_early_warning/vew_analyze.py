"""Verifier's analysis of vew_rows.jsonl against the early-warning result (read only on early_warning/ew_series.csv).

    python -B vew_analyze.py > vew_analyze.txt
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EW = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/com_revisit/notes/early_warning")
rows = [json.loads(l) for l in open(HERE / "vew_rows.jsonl")]
ew = pd.read_csv(EW / "ew_series.csv")
THR = 0.311


def ewv(run, steps, col):
    m = ew[(ew["run"] == run) & (ew["steps"] == steps)]
    return float(m[col].iloc[0]) if len(m) else float("nan")


def f(x, w=7, d=3):
    return f"{x:+{w}.{d}f}" if x is not None and math.isfinite(x) else f"{'nan':>{w}s}"


print("=" * 118)
print("A. REPRODUCTION (my code, seeds 5001-5010 = the early-warning set) against ew_series.csv")
print("=" * 118)
print("  run          steps     | G_own mine   ew     | c0_own mine   ew   | FP mine    ew    | own14 mine  ew   | G_ref mine  ew   | xchk mu/norm")
mx = {"G": 0, "c0": 0, "FP": 0, "own": 0, "Gr": 0}
for r in rows:
    a = r["A"]
    vals = [(a["P"]["G"], ewv(r["run"], r["steps"], "G_own"), "G"), (a["P"]["c0"], ewv(r["run"], r["steps"], "c0_own"), "c0"),
            (a["P"]["FP"], ewv(r["run"], r["steps"], "FP_own"), "FP"), (a["own14_median"], ewv(r["run"], r["steps"], "own14"), "own"),
            (r["REF"]["P"]["G"], ewv(r["run"], r["steps"], "G_ref"), "Gr")]
    for m, e, k in vals:
        if math.isfinite(e):
            mx[k] = max(mx[k], abs(m - e))
    print(f"  {r['run']:12s} {r['steps']:9d} | " + " | ".join(f"{f(m)} {f(e)}" for m, e, _ in vals)
          + f" | {r['xcheck_mu_vs_sb3']:.0e}/{r['xcheck_norm']:.0e}")
print("  max |mine - ew|: " + ", ".join(f"{k} {v:.1e}" for k, v in mx.items()))

print()
print("=" * 118)
print("B. STATE-SET SENSITIVITY: G (0-1 deg sweep, x5.16) on seeds A = 5001-5010 (60 states), B = 6001-6020 (120 held-out")
print("   states), AB pooled; [2.5%, 97.5%] = bootstrap over episodes.  Rule G > 0.311 fires?  own14 = own clipped command")
print("=" * 118)
print("  run          steps     |  G_A [95%]                 |  G_B [95%]                 | G_AB    | fire A/B/AB | c0_A    c0_B   | own_A  own_B | REF G")
for r in rows:
    A, B, AB = r["A"]["P"], r["B"]["P"], r["AB"]["P"]
    fire = "/".join("Y" if x["G"] > THR else "n" for x in (A, B, AB))
    print(f"  {r['run']:12s} {r['steps']:9d} | {f(A['G'])} [{f(A['G_boot_q025'])},{f(A['G_boot_q975'])}] | "
          f"{f(B['G'])} [{f(B['G_boot_q025'])},{f(B['G_boot_q975'])}] | {f(AB['G'])} | {fire:11s} | {f(A['c0'])} {f(B['c0'])} | "
          f"{f(r['A']['own14_median'], 6)} {f(r['B']['own14_median'], 6)} | {f(r['REF']['P']['G'])}")

print()
print("=" * 118)
print("C. SWEEP-DEFINITION SENSITIVITY (seeds A states): G over 0-1 deg (P, primary), 0-0.5 deg (H), own angle +-0.25 deg")
print("   (L, the local derivative at the operating point), 0-6 deg (LEG, legacy); share of states with a positive slope")
print("=" * 118)
print("  run          steps     |   G_P     G_H     G_L   G_LEG  | theta_own deg | states slope>0 (P) | LEG clip from deg")
for r in rows:
    a = r["A"]
    print(f"  {r['run']:12s} {r['steps']:9d} | {f(a['P']['G'])} {f(a['H']['G'])} {f(a['L']['G'])} {f(a['LEG']['G'])} | "
          f"{a['L']['theta_deg_median']:+8.2f}      | {a['P']['frac_states_G_pos']:.2f}               | {a['LEG']['first_clipped_deg']}")

print()
print("=" * 118)
print("D. CLIPPING AND SATURATION on the 0-1 deg sweep (seeds A + B states + REF): tail_3 z range, swept points at the")
print("   clip, linearity (slope 0.5-1 / slope 0-0.5 deg), quadratic coefficient per deg^2, share of hidden units |tanh|>0.99")
print("   at 0 and 1 deg (layer 1, layer 2); normaliser tail_3 SD and the angle where the clip is reached")
print("=" * 118)
print("  run          steps     | z min..max (A,B,REF) | n clip | lin A   lin B  | quad A   | h1 sat 0/1      h2 sat 0/1      | t3 sd  clip@deg")
for r in rows:
    zs = [r[k]["P"]["z_min"] for k in ("A", "B", "REF")] + [r[k]["P"]["z_max"] for k in ("A", "B", "REF")]
    nc = sum(r[k]["P"]["n_clipped"] for k in ("A", "B", "REF"))
    a = r["A"]["P"]
    print(f"  {r['run']:12s} {r['steps']:9d} | {min(zs):+6.2f}..{max(zs):+5.2f}       | {nc:6d} | {f(a['lin_ratio_hi_lo'], 6)} {f(r['B']['P']['lin_ratio_hi_lo'], 6)} | {f(a['quad_coef_median'], 8, 4)} | "
          f"{a['sat_h1_at0']:.3f}/{a['sat_h1_at1']:.3f}     {a['sat_h2_at0']:.3f}/{a['sat_h2_at1']:.3f}     | {r['t3_sd_deg']:.3f}  {r['deg_at_clip']:6.2f}")

print()
print("=" * 118)
print("E. EPISODES: own command and rail share per state set (n terminated; episodes with |cmd|>=0.99 share > 0.10)")
print("=" * 118)
for r in rows:
    print(f"  {r['run']:12s} {r['steps']:9d} | A: own14 med {r['A']['own14_median']:+.3f} frac99 med {r['A']['frac99_median']:.3f} over-bar {r['A']['n_over_bar']}/10 term {r['A']['n_terminated']}"
          f" | B: own14 med {r['B']['own14_median']:+.3f} frac99 med {r['B']['frac99_median']:.3f} over-bar {r['B']['n_over_bar']}/20 term {r['B']['n_terminated']}"
          f" | own14 range B {min(e['own14'] for e in r['B']['episodes'] if e['own14'] is not None):+.2f}..{max(e['own14'] for e in r['B']['episodes'] if e['own14'] is not None):+.2f}")

print()
print("=" * 118)
print("F. THE RULE'S MARGINS against the state-set spread")
print("=" * 118)
get = {(r["run"], r["steps"]): r for r in rows}
for key in [("s42", 3145728), ("s60", 7340032), ("s80", 11534336), ("s80", 12001280), ("s80_B_1401", 5767168)]:
    if key in get:
        r = get[key]
        gs = [r[k]["P"]["G"] for k in ("A", "B", "AB")]
        sd = r["A"]["P"]["G_boot_sd"]
        print(f"  {key[0]:10s} {key[1]:9d}: G A/B/AB {gs[0]:+.3f}/{gs[1]:+.3f}/{gs[2]:+.3f}; |A-B| {abs(gs[0] - gs[1]):.3f}; "
              f"bootstrap SD (A) {sd:.3f}; margin to 0.311 (A) {gs[0] - THR:+.3f} = {(gs[0] - THR) / sd:+.1f} SD")
cpu = [r["cpu_s"]["rollouts_and_sweeps"] for r in rows]
print(f"\n  CPU per checkpoint for 30 episodes + all sweeps: median {np.median(cpu):.0f} s (so about {np.median(cpu) / 3:.0f} s per 10 episodes)")

print()
print("=" * 118)
print("G. LINEAR MARGIN TO THE RAIL (no new simulation; an [inferred] prediction for the proposed closed-loop nudge test).")
print("   A constant bias b on the tail_3 command moves the linear fixed point to (c0 + b) / (1 - G); the rail (cmd 1) is")
print("   reached at b* = 1 - G - c0 = (1 - G)(1 - FP) on the 0-1 deg sweep (P), or b*_L = (1 - G_L)(1 - own) with the")
print("   local slope at the operating point (L).  Seeds A states.")
print("=" * 118)
print("  run          steps     |  G_P     c0     b*_P  |  G_L    own    b*_L")
for r in rows:
    a = r["A"]
    gP, c0, gL, own = a["P"]["G"], a["P"]["c0"], a["L"]["G"], a["own14_median"]
    print(f"  {r['run']:12s} {r['steps']:9d} | {f(gP)} {f(c0)} {f(1 - gP - c0)} | {f(gL)} {f(own)} {f((1 - gL) * (1 - own))}")

print()
print("=" * 118)
print("H. THE PRE-STATED RULE UNDER EACH SWEEP DEFINITION (my measurements, seeds A).  threshold = midpoint of (r10 s80")
print("   maximum over the s80 checkpoints I measured, r10 s42 maximum before its FOB 3,670,016).  Test: r10 s60 checkpoints")
print("   before its FOB 8,126,464 (6.29M, 7.34M measured).  Also seeds B.")
print("=" * 118)
FOB42, FOB60 = 3670016, 8126464
for seeds in ("A", "B"):
    for key in ("P", "H", "L", "LEG"):
        s80 = [(r["steps"], r[seeds][key]["G"]) for r in rows if r["run"] == "s80"]
        s42 = [(r["steps"], r[seeds][key]["G"]) for r in rows if r["run"] == "s42" and r["steps"] < FOB42]
        s60 = sorted((r["steps"], r[seeds][key]["G"]) for r in rows if r["run"] == "s60" and r["steps"] < FOB60)
        m80 = max(s80, key=lambda x: x[1])
        m42 = max(s42, key=lambda x: x[1])
        thr = 0.5 * (m80[1] + m42[1])
        fire60 = [s for s, g in s60 if g > thr]
        g734 = dict(s60).get(7340032, float("nan"))
        print(f"  seeds {seeds} G_{key:3s}: s80 max {m80[1]:+.3f} at {m80[0]} (n {len(s80)}); s42 pre-FOB max {m42[1]:+.3f} at {m42[0]}; thr {thr:+.3f};"
              f" s60 7.34M {g734:+.3f} ({g734 - thr:+.3f}) -> {'FIRES' if fire60 else 'MISSES'} s60 before FOB")
