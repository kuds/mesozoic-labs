"""E2 review fixes (findings E2-R1 to E2-R4, 28 Nov 2026). DESK readings of E2's stored outputs; no new episode.

Everything here is decided after E2's results were seen ("changed after a result"); no verdict, rule, seed or
threshold of the pre-registration is changed. Reads only:
- the E2 result files (sim/results/candidates_2026-11-28/E2/*), left unchanged (their sha256 is recorded);
- E2's B1 and E2(b) per-episode files in the scratch folder (candidates/E2/episodes/), written by e2_run.py.

R1: recomputes E2(a)'s speed check from the B1 1 ms episode files with its own percentile code (not
    candidate_checks.weighted_percentile_from_hist), compares with e2a_summary.json, and lists every failing
    (policy, block, r, placement, drive).
R2: per E2(b) run, the reduced drives' speeds against E2(a)'s numerical limits (83 rpm p99, 91.3 rpm peak) and the
    share of samples at or over 83 rpm. E2(b) logs walking and pushes only (no standing), so this compares numbers;
    it is not E2(a)'s check.
R3: the X6 reading as run (p99 <= 108, peak <= 1.1 x 108 = 118.8 rpm) and the plain 108 rpm reading (p99 and peak
    <= 108 rpm), both supporting, never judged.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/mesozoic-labs/hardware/triceratops_build/sim")
import candidate_checks as C  # noqa: E402

T0 = time.time()
RES = C.RESULTS / "E2"
EP = Path("/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E2/episodes")
POLS = ("fore128-s0", "fore128-s1", "fore128-s2", "fore128-s3")
CAP, PK = 83.0, 1.1 * 83.0
X6_ALLOW = 108.0  # rl/analysis/gait_pipe1.py RPM_ALLOW "front elbow" (prereg section 3 item 6)
assert abs(C.CAP_RPM - CAP) < 1e-12 and abs(C.PEAK_FACTOR * C.CAP_RPM - PK) < 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(run, pol, cond, seed, npz=False):
    j = json.loads((EP / run / pol / f"{cond}__{seed}.json").read_text())
    if npz:
        j["rpm_hist"] = np.load(EP / run / pol / f"{cond}__{seed}.npz")["rpm_hist"]
    return j


def p99_mix(hists, weights, rpm_bin, q=0.99):
    """Own implementation: each pooled histogram normalised per drive, mixed by weight; the upper edge of the first
    bin whose cumulative share reaches q."""
    out = []
    for d in range(hists[0].shape[0]):
        mix = sum(w * h[d].astype(float) / max(h[d].sum(), 1.0) for h, w in zip(hists, weights))
        c = np.cumsum(mix) / mix.sum()
        out.append((int(np.nonzero(c >= q - 1e-12)[0][0]) + 1) * rpm_bin)
    return np.array(out)


stored = json.loads((RES / "e2a_summary.json").read_text())
sc_stored = {(s["policy"], s["block"], float(s["r"]), s["placement"]): s for s in stored["speed_checks"]}

# ------------------------------------------------------------------------------------------------ R1: E2(a) speed
r1_rows, max_dp99, max_dpk = [], 0.0, 0.0
for pol in POLS:
    for blk, seeds in C.E1_BLOCKS.items():
        H, peaks, acts, rbin = {}, [], None, None
        for cond in ("walk_0.25", "stand"):
            hs = []
            for s in seeds:
                e = load("B1_dt0.001", pol, cond, s, npz=True)
                assert not e["unstable"] and not e["fell"]
                assert all(abs(x - 1.0) < 1e-12 for x in e["r"])   # B1 logs at r = 1
                acts = acts or e["actuators"]
                assert e["actuators"] == acts
                rbin = float(e["rpm_bin"])
                hs.append(e["rpm_hist"])
                peaks.append(np.array(e["rpm_peak"]))
            H[cond] = np.sum(hs, axis=0)
        p99_1 = p99_mix([H["walk_0.25"], H["stand"]], [0.75, 0.25], rbin)
        pk_1 = np.max(np.vstack(peaks), axis=0)
        for r in C.REDUCTIONS:
            for pl in C.PLACEMENTS:
                red = C.reduced_joints(pl)
                idx = [acts.index(j) for j in red]
                p, k = p99_1[idx] * r, pk_1[idx] * r
                st = sc_stored[(pol, blk, float(r), pl)]
                assert list(st["drives"]) == list(red)
                max_dp99 = max(max_dp99, float(np.max(np.abs(p - np.array(st["p99_rpm"])))))
                max_dpk = max(max_dpk, float(np.max(np.abs(k - np.array(st["peak_rpm"])))))
                ok = bool(np.all(p <= CAP) and np.all(k <= PK))
                assert ok == bool(st["pass"]), (pol, blk, r, pl)
                for j, pp, kk in zip(red, p, k):
                    if pp > CAP or kk > PK:
                        r1_rows.append({"policy": pol, "block": blk, "r": r, "placement": pl, "drive": j,
                                        "drive_type": C.drive_of(j), "p99_rpm": round(float(pp), 3),
                                        "peak_rpm": round(float(kk), 3),
                                        "fails_on": "+".join(x for x, b in (("p99", pp > CAP), ("peak", kk > PK)) if b)})
print(f"R1 recomputation vs e2a_summary.json speed_checks: max |dp99| {max_dp99:.3g} rpm, max |dpeak| {max_dpk:.3g} rpm,"
      f" pass flags 120 of 120 equal")
first_fail = {}
for x in r1_rows:
    key = (x["placement"], x["policy"], x["block"], x["drive"])
    if key not in first_fail or x["r"] < first_fail[key]["r"]:
        first_fail[key] = x
r1_first = sorted(first_fail.values(), key=lambda x: (x["placement"], x["policy"], x["block"], x["drive"]))
print("R1 failing (policy, block, r, placement, drive) rows:", len(r1_rows))
for x in r1_first:
    print(f"  first failure: {x['placement']:5s} {x['policy']} {x['block']} {x['drive']} ({x['drive_type']}) at r = "
          f"{x['r']}: p99 {x['p99_rpm']:.2f}, peak {x['peak_rpm']:.2f} rpm ({x['fails_on']})")
non_s3 = sorted({(x["policy"], x["placement"], x["drive"]) for x in r1_rows if x["policy"] != "fore128-s3"})
print("R1 non-s3 failures (policy, placement, drive):", non_s3)
print("R1 non-s3 failures at hip pitch:", [x for x in r1_rows if x["policy"] != "fore128-s3" and x["placement"] == "pitch"])

# ------------------------------------------------------------------------------------------- R2: E2(b) drive speed
e2b = json.loads((RES / "e2b_summary.json").read_text())
RED = C.reduced_joints("both")
r2 = {}
for run in ("B1_dt0.001", "B025_dt0.00025", "R_dt0.00025", "rod_dt0.00025", "belt_dt0.00025", "cable_dt0.00025",
            "belt_1span_dt0.00025"):
    st = e2b["run_stats"][run]
    worst99, worstpk, worst99_walk, worstpk_walk = (0, None), (0, None), (0, None), (0, None)
    over = {"p99_over_83": [], "peak_over_91.3": []}
    for pol in POLS:
        for cond, s in st[pol].items():
            for j in RED:
                p, k = s["drive_rpm_p99"][j], s["drive_rpm_peak"][j]
                if p > worst99[0]:
                    worst99 = (p, f"{pol} {cond} {j}")
                if k > worstpk[0]:
                    worstpk = (k, f"{pol} {cond} {j}")
                if cond == "walk_0.25" and p > worst99_walk[0]:
                    worst99_walk = (p, f"{pol} {j}")
                if cond == "walk_0.25" and k > worstpk_walk[0]:
                    worstpk_walk = (k, f"{pol} {j}")
                if p > CAP:
                    over["p99_over_83"].append(f"{pol} {cond} {j} {p}")
                if k > PK:
                    over["peak_over_91.3"].append(f"{pol} {cond} {j} {k}")
    # per episode: any reduced drive's peak over 83 / 91.3 rpm (2-20 s); at-cap share (whole episode)
    n_ep, n83, n913, cap_max, cap_by_pol = 0, 0, 0, None, {}
    for pol in POLS:
        for cond in C.E2B_CONDITIONS:
            for s in C.E1_BLOCKS["A"] + C.E1_BLOCKS["B"]:
                e = load(run, pol, cond, s)
                n_ep += 1
                pk = max(e["rpm_peak"][e["actuators"].index(j)] for j in RED)
                n83 += pk > CAP
                n913 += pk > PK
                if e["at_cap_share"] is not None:
                    m = max(e["at_cap_share"])
                    cap_max = m if cap_max is None else max(cap_max, m)
                    cap_by_pol[pol] = max(cap_by_pol.get(pol, 0.0), m)
    r2[run] = {"r": 1.25 if not run.startswith("B") else 1.0, "capped": not run.startswith("B"),
               "worst_p99_rpm": worst99, "worst_peak_rpm": worstpk,
               "walk_0.25_worst_p99_rpm": worst99_walk, "walk_0.25_worst_peak_rpm": worstpk_walk,
               "episodes": n_ep, "episodes_any_reduced_drive_peak_over_83": int(n83),
               "episodes_any_reduced_drive_peak_over_91.3": int(n913),
               "at_cap_share_max_one_drive_one_episode": cap_max, "at_cap_share_max_by_policy": cap_by_pol,
               "over_limits_pooled_per_condition": over}
    print(f"R2 {run}: worst p99 {worst99[0]} rpm ({worst99[1]}), worst peak {worstpk[0]} rpm ({worstpk[1]}); walk 0.25 "
          f"only: p99 {worst99_walk[0]} ({worst99_walk[1]}), peak {worstpk_walk[0]} ({worstpk_walk[1]}); episodes with a "
          f"reduced drive's peak > 83: {n83}/{n_ep}, > 91.3: {n913}/{n_ep}; at-cap share max "
          f"{None if cap_max is None else round(100 * cap_max, 3)} %")

# -------------------------------------------------------------------------------------------------- R3: X6 reading
grid = list(csv.DictReader((RES / "e2a_grid.csv").open()))
heat = {(g["transmission"], float(g["r"]), g["placement"], g["policy"], g["block"]):
        (g["C1_pass"] == "True" and g["C2_pass"] == "True") for g in grid}


def x6_reading(peak_cap_x6):
    rows = {}
    for (pol, blk, r, pl), s in sc_stored.items():
        ok, bad = True, []
        for j, p, k in zip(s["drives"], s["p99_rpm"], s["peak_rpm"]):
            cap = X6_ALLOW if C.drive_of(j) == "X6-60" else CAP
            pcap = peak_cap_x6 if C.drive_of(j) == "X6-60" else PK
            g = p <= cap and k <= pcap
            ok &= g
            if not g:
                bad.append(f"{j} p99 {p:.1f} peak {k:.1f}")
        rows[(pol, blk, r, pl)] = (ok, bad)
    return rows


readings = {"as_run_p99_108_peak_118.8": x6_reading(1.1 * X6_ALLOW), "plain_p99_108_peak_108": x6_reading(X6_ALLOW)}
r3 = {}
for name, rows in readings.items():
    speed_only, full = [], []
    for tr in ("rod", "belt", "cable"):
        for r in C.REDUCTIONS:
            for pl in C.PLACEMENTS:
                keys = [(p, b, float(r), pl) for p in POLS for b in ("A", "B")]
                sp = all(rows[k][0] for k in keys)
                if sp and tr == "rod":
                    speed_only.append(f"{pl} r {r}")
                if sp and all(heat[(tr, float(r), pl, k[0], k[1])] for k in keys):
                    full.append(f"{tr} {pl} r {r}")
    knees_fail = sorted({(k[0], k[2]) for k, v in rows.items() if not v[0] and k[3] == "knees"})
    r3[name] = {"speed_check_alone_passes_all_8_policy_blocks": speed_only,
                "rows_passing_C1_C2_and_this_speed_reading": full,
                "knees_speed_failures_policy_r": knees_fail}
    print(f"R3 {name}: speed alone passes all 8 policy-blocks at {speed_only}; C1+C2+speed all pass: {full or 'none'};"
          f" knee-placement speed failures (policy, r): {knees_fail}")

# ----------------------------------------------------------------------------------------- files left unchanged
unchanged = {p.name: sha(p) for p in sorted(RES.iterdir()) if p.is_file() and not p.name.startswith("e2_review")
             and not p.name.startswith("E2_REVIEW")}
out = {"experiment": "E2 review fixes (E2-R1 to E2-R4), DESK readings of stored E2 outputs; no new episode",
       "changed_after_a_result": True,
       "note": "Every reading here was chosen after E2's results were seen. No verdict, rule, seed or threshold of "
               "design/PREREG_CANDIDATES_2026_11_28.md is changed. E2(a) verdict: 0 of 45 PASS (unchanged); E2(b): "
               "rod, belt, cable COMPATIBLE under the pre-registered torque-envelope SpeedCap (unchanged).",
       "R1_e2a_speed": {"recompute_max_abs_diff_p99_rpm": max_dp99, "recompute_max_abs_diff_peak_rpm": max_dpk,
                        "pass_flags_equal": "120 of 120", "failing_rows": r1_rows, "first_failure": r1_first,
                        "non_s3_failures": [list(x) for x in non_s3]},
       "R2_e2b_drive_speed": {"limits_compared": "E2(a)'s numbers: p99 <= 83 rpm, peak <= 91.3 rpm (E2(a) applies them "
                                                 "to the walk 0.25 / stand mix; E2(b) has walk 0.25 and the two pushes, "
                                                 "pooled per condition over 8 seeds, 2-20 s, every physics step)",
                              "runs": r2},
       "R3_x6_reading": {"x6_allowance_rpm": X6_ALLOW, "source": "rl/analysis/gait_pipe1.py RPM_ALLOW 'front elbow'",
                         "readings": r3,
                         "as_run_script": {"path": "candidates/E2/code/e2a_x6_beside.py",
                                           "sha256": sha(Path(__file__).with_name("e2a_x6_beside.py")),
                                           "also_in": "e2_checks.json scripts"}},
       "files_left_unchanged_sha256": unchanged,
       "wall_s": time.time() - T0,
       "provenance": C.provenance(__file__)}
(RES / "e2_review_addendum.json").write_text(json.dumps(out, indent=1, default=str))
with (RES / "e2_review_speed_failures.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(r1_rows[0]))
    w.writeheader()
    w.writerows(r1_rows)
print("wrote", RES / "e2_review_addendum.json", "and e2_review_speed_failures.csv")
