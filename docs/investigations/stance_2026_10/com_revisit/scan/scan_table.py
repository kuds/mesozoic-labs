"""Incidence tables from scan.jsonl (no simulation).  Usage: python scan_table.py scan.jsonl"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact

from scanlib import clopper_pearson

rows = [json.loads(line) for line in open(sys.argv[1])]
groups = defaultdict(list)
for r in rows:
    groups[(r["model"], r["block"], r["noise"])].append(r)


def ci(k, n):
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} = {100 * k / n:.2f} % [95 % CI {100 * lo:.2f}-{100 * hi:.2f} %]"


def mx(rs, key, scale=1.0):
    vals = [scale * r[key] for r in rs if key in r]
    if not vals:
        return "n/a"
    i = int(np.argmax(vals))
    seeds = [r["seed"] for r in rs if key in r]
    return f"{vals[i]:.4g} (seed {seeds[i]})"


order = [("compy_anat", "statue"), ("compy_anat", "test"), ("compy_anat", "train"), ("velo", "statue"),
         ("velo", "test"), ("velo", "train")]
print("Per model and block: incidence of each flag and of creep (all three), exact Clopper-Pearson 95 % intervals.")
print("speed = com_speed_end_rms > 1 mm/s; travel = foot travel after 1 s > 5 mm; oscill = per-foot load sd over the"
      " last 2 s > 1 % body weight; creep = all three.")
for key in sorted(groups, key=lambda k: (order.index((k[0], k[1])) if (k[0], k[1]) in order else 99, k[2] or 0)):
    rs = groups[key]
    model, block, noise = key
    stood = [r for r in rs if r["stood"]]
    n = len(rs)
    seeds = sorted(r["seed"] for r in rs)
    print(f"\n== {model} {block} (reset noise {noise}): {n} episodes, seeds {seeds[0]}-{seeds[-1]}")
    print(f"   stood the full horizon: {len(stood)}/{n}; fell: {[ (r['seed'], r['termination'], r['steps']) for r in rs if not r['stood']]}")
    for flag in ("flag_speed", "flag_travel", "flag_oscill", "creep", "any_flag"):
        k = sum(1 for r in stood if r[flag])
        hit = [r["seed"] for r in stood if r[flag]]
        print(f"   {flag:12s} {ci(k, n)}  seeds {hit[:20]}")
    kbar = sum(1 for r in stood if r["n_problems"])
    print(f"   failing a spread bar of the test class: {kbar}/{n} seeds {[r['seed'] for r in stood if r['n_problems']][:20]}")
    print(f"   largest com_speed_end_rms (mm/s): {mx(stood, 'com_speed_end_rms_mps', 1e3)}")
    print(f"   largest foot travel after 1 s (mm): {mx(stood, 'foot_travel_after1s_m', 1e3)}")
    print(f"   largest per-foot load sd over the last 2 s (fraction of body weight): {mx(stood, 'load_sd_tail_bw_max')}")
    print(f"   largest root vz RMS over the last 2 s (mm/s): {mx(stood, 'root_vz_tail_rms_mmps')}")
    print(f"   largest CoM travel over the window (mm): {mx(stood, 'com_travel_window_mm')}")
    print(f"   largest unloaded-substep fraction on a foot, last 2 s: {mx([dict(r, u=max(r['unloaded_tail_frac'])) for r in stood], 'u')}")
    print(f"   largest floor load on non-foot animal geoms, last 2 s (N): {mx(stood, 'other_floor_tail_max_n')}")
    osc = [r for r in stood if r["flag_oscill"]]
    if osc:
        print(f"   oscillation frequencies (Hz): {[(r['seed'], round(r['osc_freq_hz'], 2), round(r['osc_peak_share'], 2)) for r in osc[:20]]}")
    # median ring-down of root vz per second
    prof = np.array([r["root_vz_rms_per_s_mmps"] for r in stood])
    print(f"   root vz RMS per second, median over spawns (mm/s): {np.round(np.median(prof, axis=0), 4).tolist()}")
    print(f"   root vz RMS per second, max over spawns (mm/s):    {np.round(np.max(prof, axis=0), 4).tolist()}")
    wall = [r["wall_s"] for r in rs]
    print(f"   wall per episode {np.mean(wall):.2f} s")

# comparison with the robot's measured incidence at the same noise (other lenses' files, cited in NOTES)
ROBOT = {
    "0.02, the same 280 seeds (PR + lens: 111, 143 and 8 of 200-399)": (10, 280),
    "0.02, seeds 400-599 (verifier C6)": (6, 200),
    "0.01 (verifier M2: 202 spawns)": (0, 202),
}
print("\nFisher exact (two-sided) against the robot's base incidence, creep counts:")
for key in sorted(groups):
    model, block, noise = key
    if block == "statue":
        continue
    rs = groups[key]
    k = sum(1 for r in rs if r.get("creep"))
    for label, (kr, nr) in ROBOT.items():
        p = fisher_exact([[k, len(rs) - k], [kr, nr - kr]])[1]
        print(f"   {model} {block} {k}/{len(rs)} vs robot {label} {kr}/{nr}: p = {p:.3g}")
