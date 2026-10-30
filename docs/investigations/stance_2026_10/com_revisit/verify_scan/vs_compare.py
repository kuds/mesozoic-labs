"""Compare the verifier's re-runs (vs_sample.jsonl) with the scan's rows (scan.jsonl) and its controls; no simulation.
Usage: python vs_compare.py <scan.jsonl> <vs_sample.jsonl>"""
import json
import sys
from collections import defaultdict

import numpy as np

scan = {}
for x in open(sys.argv[1]):
    r = json.loads(x)
    scan[(r["model"], r["seed"], r["noise"])] = r
mine = [json.loads(x) for x in open(sys.argv[2])]

print("== 1. Robot controls with the scan's bars (1 mm/s, 5 mm, 0.01 BW) applied to MY metrics")
for r in mine:
    if r["model"] != "robot":
        continue
    if r["role"] == "known_creeper" or r["seed"] in (102, 141):
        print(f"  seed {r['seed']:3d} ({r['role']}): test speed {1e3 * r['test_com_speed_end_rms_mps']:.4f} mm/s, my speed "
              f"{1e3 * r['my_end_speed_mps']:.4f}; foot travel after 1 s {1e3 * r['my_foot_travel_after1s_m']:.2f} mm; load sd "
              f"{r['my_load_sd_bw_max']:.4f} BW at {r['my_osc_hz']:.2f} Hz; unloaded {np.round(r['my_unloaded_tail_frac'], 4).tolist()}; "
              f"CoM travel window {r['my_com_travel_window_mm']:.2f} mm -> speed {r['flag_speed_test']}/{r['flag_speed_mine']} "
              f"travel {r['flag_travel']} oscill {r['flag_oscill']} creep3 {r['creep3']}")
rest = [r for r in mine if r["model"] == "robot" and r["role"] == "resting"]
print(f"  resting robot seeds ({len(rest)}: 0-29, 102, 141): any flag on {[r['seed'] for r in rest if r['any_flag']]}; largest "
      f"speed {1e3 * max(r['my_end_speed_mps'] for r in rest):.4f} mm/s (test {1e3 * max(r['test_com_speed_end_rms_mps'] for r in rest):.4f}), "
      f"travel {1e3 * max(r['my_foot_travel_after1s_m'] for r in rest):.2f} mm, load sd {max(r['my_load_sd_bw_max'] for r in rest):.2e} BW")
cr = [r for r in mine if r["model"] == "robot" and r["role"] == "known_creeper"]
print(f"  known creepers flagged creep3: {sum(r['creep3'] for r in cr)}/{len(cr)}; smallest per-criterion values among them: "
      f"speed(test) {1e3 * min(r['test_com_speed_end_rms_mps'] for r in cr):.3f} mm/s, travel {1e3 * min(r['my_foot_travel_after1s_m'] for r in cr):.2f} mm, "
      f"load sd {min(r['my_load_sd_bw_max'] for r in cr):.4f} BW")

print("\n== 2. Agreement with scan.jsonl, per cell (model, noise)")
cells = defaultdict(list)
for r in mine:
    if r["model"] == "robot":
        continue
    cells[(r["model"], r["noise"])].append(r)
for key, rs in cells.items():
    model, noise = key
    n = len(rs)
    nf = sum(1 for r in rs if r["role"] == "flagged")
    agree_flags = 0
    diffs = defaultdict(list)
    disagreements = []
    for r in rs:
        s = scan[(model, r["seed"], noise)]
        mine_flags = (r["flag_speed_test"], r["flag_travel"], r["flag_oscill"])
        their = (s["flag_speed"], s["flag_travel"], s["flag_oscill"])
        if mine_flags == their and r["flag_speed_mine"] == s["flag_speed"]:
            agree_flags += 1
        else:
            disagreements.append((r["seed"], mine_flags, r["flag_speed_mine"], their))
        diffs["test_speed_exact"].append(r["test_com_speed_end_rms_mps"] == s["com_speed_end_rms_mps"])
        diffs["travel_absdiff_m"].append(abs(r["my_foot_travel_after1s_m"] - s["foot_travel_after1s_m"]))
        diffs["loadsd_reldiff"].append(abs(r["my_load_sd_bw_max"] - s["load_sd_tail_bw_max"]) / max(s["load_sd_tail_bw_max"], 1e-300))
        diffs["loadsd_absdiff_bw"].append(abs(r["my_load_sd_bw_max"] - s["load_sd_tail_bw_max"]))
        diffs["comtravel_absdiff_mm"].append(abs(r["my_com_travel_window_mm"] - s["com_travel_window_mm"]))
        diffs["vz_tail_absdiff"].append(abs(r["my_root_vz_tail_rms_mmps"] - s["root_vz_tail_rms_mmps"]))
        diffs["myspeed_over_test"].append(r["my_end_speed_mps"] / max(r["test_com_speed_end_rms_mps"], 1e-300))
    print(f"  {model} noise {noise}: {n} re-run ({nf} scan-flagged, {n - nf} unflagged); stood {sum(r['stood'] for r in rs)}/{n}; "
          f"flags identical on {agree_flags}/{n}; disagreements {disagreements[:5]}")
    print(f"     test end speed bit-identical {sum(diffs['test_speed_exact'])}/{n}; |foot travel diff| max {max(diffs['travel_absdiff_m']):.2e} m; "
          f"load sd |diff| max {max(diffs['loadsd_absdiff_bw']):.2e} BW (rel {max(diffs['loadsd_reldiff']):.2e}); |CoM travel diff| max "
          f"{max(diffs['comtravel_absdiff_mm']):.2e} mm (my CoM = mass-weighted xipos); root vz tail |diff| max {max(diffs['vz_tail_absdiff']):.2e} mm/s; "
          f"my/test end speed ratio {min(diffs['myspeed_over_test']):.3f}-{max(diffs['myspeed_over_test']):.3f}")
    print(f"     my maxima: test speed {1e3 * max(r['test_com_speed_end_rms_mps'] for r in rs):.4g} mm/s, my speed "
          f"{1e3 * max(r['my_end_speed_mps'] for r in rs):.4g} mm/s, travel {1e3 * max(r['my_foot_travel_after1s_m'] for r in rs):.3f} mm, "
          f"load sd {max(r['my_load_sd_bw_max'] for r in rs):.3g} BW, unloaded frac {max(max(r['my_unloaded_tail_frac']) for r in rs)}, "
          f"total tail load {min(r['my_total_load_tail_bw'] for r in rs):.6f}-{max(r['my_total_load_tail_bw'] for r in rs):.6f} BW")
    other_whole = max(r["my_other_floor_max_n_whole"] for r in rs)
    geoms = defaultdict(lambda: [0, 0.0, 0])
    for r in rs:
        for g, (c, f, leg) in r["floor_geoms_whole"].items():
            geoms[g][0] += 1
            geoms[g][1] = max(geoms[g][1], f)
            geoms[g][2] = leg
    print(f"     whole-episode floor contact: non-leg geoms max {other_whole:.3g} N; geoms ever touching the floor "
          f"(spawns touching, max N): {sorted((g, v[0], round(v[1], 3)) for g, v in geoms.items())}")

print("\n== 3. Statues")
for r in mine:
    if r["noise"] is None:
        s = scan[(r["model"], 0, None)]
        print(f"  {r['model']}: test speed identical {r['test_com_speed_end_rms_mps'] == s['com_speed_end_rms_mps']}; flags any {r['any_flag']}; "
              f"floor geoms {sorted(r['floor_geoms_whole'])}")
