"""The foreleg comparison's (D22) appendix: every judged measure per run, and the main figures of what
PREREG_FORELEG_D22.md lists as reported, not judged, read from the tools' stored output. Simulation only; the choice
itself is judge_d22.py's. The full per-run record is each run's `analytics/results/<run>/gait_report.md` and
`robustness/results/<run>/summary_<run>.md`.

Inputs (all in the repository): `analysis/results/d22/` (judge_d22.json, foot_loads_<run>.json, demo/<run>/),
`analytics/results/<run>/gait_report.json`, `robustness/results/<run>/sweep_<run>.json.gz`, with run 3's two seeds
(`run3-s0`, `run3-s1`, 188 mm) as a reference column. Run 3's foreleg cannot be built (F18) and its fore-assembly
differs besides the length (PLAN_FORELEG_D22.md), so it is a reference, not a third arm.

Usage: python hardware/triceratops_build/rl/analysis/d22_appendix.py [--out REPORT_FORELEG_D22_APPENDIX.md]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

RL = Path(__file__).resolve().parents[1]
D22 = RL / "analysis" / "results" / "d22"
LENGTHS, SEEDS = (128, 160), (0, 1, 2, 3)
RUNS = {f"{n} s{s}": f"fore{n}-s{s}" for n in LENGTHS for s in SEEDS}
REF = {"188 s0 (run 3)": "run3-s0", "188 s1 (run 3)": "run3-s1"}
ALL = {**RUNS, **REF}
FEET = ("left_fore", "right_fore", "left_hind", "right_hind")
FRONT = ("left_fore_roll", "left_fore_pitch", "left_fore_flexion", "right_fore_roll", "right_fore_pitch",
         "right_fore_flexion")
sys.path.insert(0, str(RL / "analysis"))
import judge_d22 as jd  # noqa: E402  (its measure(): run 3's columns get the D22 measures from run 3's stored outputs)
# PR #697's own figures from scripted controllers (commit 65a44fad, refinement60/mid160/README.md), for scale only.
PR697 = [
    ("slow four-beat walk at 17 mm/s, highest single-foot raw / 10 ms load, N", "87.7 / 84.7", "89.8 / 86.2"),
    ("six +-40 deg pivots, highest single-foot raw / 10 ms load, N", "302.5 / 210.2", "224.2 / 190.3"),
    ("six moving courses (seeds 101-103), highest raw / 10 ms load, N", "416.3 / 260.8", "280.7 / 215.1"),
    ("six more moving courses (seeds 104-106), highest raw / 10 ms load, N", "424.0 / 329.6", "302.7 / 247.8"),
    ("worst front elbow RMS torque in the slow walk, N m", "3.70", "4.58"),
]


def f(x, nd=2):
    if x is None:
        return "-"
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def gait_all(run):
    return json.loads((RL / "analytics" / "results" / run / "gait_report.json").read_text())


def gait(run, case="target_0.25"):
    return gait_all(run)["cases"][case]["summary"]


def sweep(run):
    return json.load(gzip.open(RL / "robustness" / "results" / run / f"sweep_{run}.json.gz", "rt"))


def table(rows, cols, label_fn_pairs, head_label=""):
    rows.append(f"| {head_label} | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for label, fn in label_fn_pairs:
        rows.append(f"| {label} | " + " | ".join(fn(c) for c in cols) + " |")
    rows.append("")


def foot(col):
    return json.loads((D22 / f"foot_loads_{ALL[col]}.json").read_text())


def run3_measures():
    """Run 3's two seeds through judge_d22.measure: its gait report and sweep as stored, and its foot loads measured
    with the D22 run commit's foot_loads.py (26 Nov; equal to the pre-registration's figures)."""
    return {k: jd.measure(jd.j3._load(str(RL / "analytics" / "results" / v / "gait_report.json")),
                          jd.j3._load(str(RL / "robustness" / "results" / v / f"sweep_{v}.json.gz")), foot(k))
            for k, v in REF.items()}


def judged(rows, judge):
    rows.append("## A. The judged measures and the choice (judge_d22.py)\n")
    rows.append(f"Run commit {judge['run_commit']}. Checks: " + "; ".join(
        f"{k} {'pass' if not v else 'FAIL: ' + ', '.join(v)}" for k, v in judge["problems"].items()) + ".\n")
    rows.append(f"**Choice: {judge['choice']['choice']} mm** ({judge['choice']['why']}).\n")
    r = {k: judge["runs"][f"{n}_s{s}"] for k, (n, s) in ((f"{n} s{s}", (n, s)) for n in LENGTHS for s in SEEDS)}
    r.update(run3_measures())
    cols = list(ALL)
    rows.append("Run 3's two columns (188 mm, a reference) are `judge_d22.measure` on run 3's stored gait report and "
                "sweep and on its foot loads, measured on 26 Nov with the D22 run commit's `foot_loads.py` (the "
                "same figures as the pre-registration's).\n")
    yes = lambda b: "pass" if b else "FAIL"  # noqa: E731
    table(rows, cols, [
        ("F1, flat walking (speed m/s; falls of 5)", lambda c: f"{yes(r[c]['rules']['F1'])} "
         f"({f(r[c]['rule_values']['F1']['speed_mps'], 3)}; {r[c]['rule_values']['F1']['falls']})"),
        ("F2, every drive under its rating (worst drive, RMS / continuous)", lambda c: f"{yes(r[c]['rules']['F2'])} "
         f"({r[c]['rule_values']['F2']['worst_joint']} {f(r[c]['rule_values']['F2']['worst'], 3)})"),
        ("L, worst front foot, 99th percentile of the 10 ms mean, mean of 5 episodes, N", lambda c: f(r[c]["L"], 1)),
        ("the same for the hind feet (reported), N", lambda c: f(r[c]["L_hind"], 1)),
        ("H_all, worst drive RMS / continuous, 0.25 m/s", lambda c: f(r[c]["H_all"], 3)),
        ("H_front, worst of the six front drives", lambda c: f(r[c]["H_front"], 3)),
        ("H_elbow, worst elbow (reported, not judged)", lambda c: f(r[c]["H_elbow"], 3)),
        ("N, falls at run 3's judged sweep points and three gait cases", lambda c: str(r[c]["N"])),
        ("foot-load episodes that fell, of 5", lambda c: str(r[c]["foot_falls"])),
    ], "measure")
    rows.append("Four-seed means (the order of D24 stopped at the walk gate, so the mean tests below were not "
                "applied; shown only to say whether they would have changed the outcome):\n")
    m = {n: {k: sum(judge["runs"][f"{n}_s{s}"][k] for s in SEEDS) / 4 for k in ("L", "N", "H_all", "H_front",
                                                                                  "H_elbow")} for n in LENGTHS}
    rows.append("| mean of 4 seeds | 128 mm | 160 mm | the rule that was not applied | would 160 mm pass it |\n"
                "|---|---|---|---|---|")
    rows.append(f"| L, front-foot load, N | {f(m[128]['L'], 1)} | {f(m[160]['L'], 1)} | L(160) <= 0.85 x L(128) = "
                f"{f(0.85 * m[128]['L'], 1)} | {'yes' if m[160]['L'] <= 0.85 * m[128]['L'] else 'no'} "
                f"(ratio {f(m[160]['L'] / m[128]['L'], 3)}) |")
    rows.append(f"| N, falls | {f(m[128]['N'], 2)} | {f(m[160]['N'], 2)} | N(160) <= N(128) + 7 | "
                f"{'yes' if m[160]['N'] <= m[128]['N'] + 7 else 'no'} |")
    rows.append(f"| H_all | {f(m[128]['H_all'], 3)} | {f(m[160]['H_all'], 3)} | H_all(160) <= H_all(128) + 0.035 | "
                f"{'yes' if m[160]['H_all'] <= m[128]['H_all'] + 0.035 else 'no'} |")
    rows.append(f"| H_front | {f(m[128]['H_front'], 3)} | {f(m[160]['H_front'], 3)} | H_front(160) <= "
                f"H_front(128) + 0.10 | {'yes' if m[160]['H_front'] <= m[128]['H_front'] + 0.10 else 'no'} |")
    rows.append(f"| H_elbow (reported) | {f(m[128]['H_elbow'], 3)} | {f(m[160]['H_elbow'], 3)} | none | - |\n")
    return r


def rules_by_length(rows, r, judge):
    rows.append("## B. Run 3's rules for each length (PASS: every seed; FAIL: none; MIXED otherwise)\n")
    rows.append("| rule | 128 mm (s0 s1 s2 s3) | 160 mm (s0 s1 s2 s3) | 188 mm, run 3 (s0 s1; reference) |\n"
                "|---|---|---|---|")
    for k in ("F1", "F2", "T1", "T2", "T3", "T4", "T5"):
        cells = []
        for v in ([judge["runs"][f"{n}_s{s}"]["rules"][k] for s in SEEDS] for n in LENGTHS):
            word = "PASS" if all(v) else "FAIL" if not any(v) else "MIXED"
            cells.append(f"{word} ({' '.join('P' if x else 'F' for x in v)})")
        v = [r[c]["rules"][k] for c in REF]
        cells.append(f"{'PASS' if all(v) else 'FAIL' if not any(v) else 'MIXED'} "
                     f"({' '.join('P' if x else 'F' for x in v)})")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("Rule values per run (T1 and T2: no fall and speed at least 0.8 of nominal; T3: drift on the four "
                "terrain points, limit 0.30 m; T4: friction 0.3, drift "
                "limit 0.20 m and speed at least 0.8 of nominal; T5: falls of 8 at -1.25 / -1.0 / +1.0 / +1.25 m/s, "
                "passing with none at +-1.0 and at most 2 at +-1.25. F1 and T1-T4 also need no fall; F2 has no fall "
                "condition):\n")
    cols = list(ALL)
    rv = {c: r[c]["rule_values"] for c in cols}
    table(rows, cols, [
        ("T1 step 3 cm: falls; speed / nominal", lambda c: f"{rv[c]['T1']['falls']}; "
         f"{f(rv[c]['T1']['share_of_nominal'], 2)}"),
        ("T2 ramp 8 deg: falls; speed / nominal", lambda c: f"{rv[c]['T2']['falls']}; "
         f"{f(rv[c]['T2']['share_of_nominal'], 2)}"),
        ("T3 drift, m: step 2 / step 3 / ramp 5 / ramp 8", lambda c: " / ".join(
            f(rv[c]["T3"]["points"][p]["drift_m"], 2) for p in ("step_2cm", "step_3cm", "ramp_5deg", "ramp_8deg"))),
        ("T3 falls (all four points)", lambda c: str(sum(rv[c]["T3"]["points"][p]["falls"]
                                                         for p in rv[c]["T3"]["points"]))),
        ("T4 friction 0.3: falls; drift m; speed / nominal", lambda c: f"{rv[c]['T4']['falls']}; "
         f"{f(rv[c]['T4']['drift_m'], 2)}; {f(rv[c]['T4']['share_of_nominal'], 2)}"),
        ("T5 pushes: falls", lambda c: " / ".join(str(rv[c]["T5"]["falls"][k]) for k in ("-1.25", "-1", "+1",
                                                                                         "+1.25"))),
    ], "rule value")


def gait_tables(rows):
    g = {k: gait(v) for k, v in ALL.items()}
    cols = list(ALL)
    rows.append("## C. Gait report, 0.25 m/s target case (randomised, 10 ms delay, 5 episodes)\n")
    rows.append("**Speed and straightness**\n")
    table(rows, cols, [
        ("forward speed, m/s", lambda c: f(g[c]["speed"]["forward_speed_mps"], 3)),
        ("sideways drift, mean of episodes, m", lambda c: f(g[c]["speed"]["sideways_drift_m"], 2)),
        ("sideways drift, worst episode, m", lambda c: f(g[c]["speed"]["sideways_drift_m_worst"], 2)),
        ("heading change, worst episode, deg", lambda c: f(g[c]["speed"]["heading_change_deg_worst"], 1)),
    ])
    rows.append("**Drive heat: RMS torque / continuous rating, mean of the 5 episodes (F2's figure)**\n")
    joints = list(g["188 s0 (run 3)"]["drives"]["per_joint"])
    table(rows, cols, [(j, lambda c, j=j: f(g[c]["drives"]["per_joint"][j]["rms_over_continuous"]["mean"], 3))
                       for j in joints] + [
        ("worst single drive: time at the episode's torque limit, worst episode, %", lambda c: f(100 * max(
            d["at_episode_limit"]["worst"] for d in g[c]["drives"]["per_joint"].values()), 1))])
    rows.append("**Gait, power and body**\n")
    table(rows, cols, [
        ("pattern", lambda c: g[c]["footfall"]["gait"]["class"]),
        ("stride frequency, all feet, Hz", lambda c: f(g[c]["footfall"]["stride_frequency_hz_all_feet"], 2)),
        ("stride length, mean, m", lambda c: f(g[c]["footfall"]["stride_length_m"]["mean"], 3)),
        ("irregular strides, worst foot, %", lambda c: f(100 * max(
            g[c]["footfall"]["irregular_stride_fraction"].values()), 0)),
        ("front feet: swing clearance, mean of the two, mm", lambda c: f(sum(
            g[c]["feet"]["swing_clearance_mm"][ft]["mean"] for ft in FEET[:2]) / 2, 0)),
        ("hind feet: swing clearance, mean of the two, mm", lambda c: f(sum(
            g[c]["feet"]["swing_clearance_mm"][ft]["mean"] for ft in FEET[2:]) / 2, 0)),
        ("front feet: touchdown speed, mean of the two, m/s", lambda c: f(sum(
            g[c]["feet"]["touchdown_speed_mps"][ft]["mean"] for ft in FEET[:2]) / 2, 2)),
        ("impact peak, median, worst foot, body weights", lambda c: f(max(
            g[c]["feet"]["impact_peak_bw_median"].values()), 2)),
        ("front share of the weight", lambda c: f(g[c]["weight"]["front_share"], 3)),
        ("electrical power, shared bus, W", lambda c: f(g[c]["power"]["electrical_shared_bus_w"], 1)),
        ("walking time per pack, shared bus, 30 W hotel, min", lambda c: f(
            g[c]["power"]["run_time_min_shared_bus"]["hotel_30w"], 0)),
        ("body pitch, mean (negative: nose down), deg", lambda c: f(g[c]["body"]["pitch_deg"]["mean"], 2)),
        ("body roll, mean / SD, deg", lambda c: f"{f(g[c]['body']['roll_deg']['mean'], 2)} / "
         f"{f(g[c]['body']['roll_deg']['sd'], 2)}"),
        ("body height, mean, mm", lambda c: f(g[c]["body"]["height_mm"]["mean"], 0)),
    ])
    labels = gait_all("fore128-s0")["cases"]
    rows.append("**The gait report's other cases** (5 episodes each; the \"other foot\" case swaps in "
                "`trike60_C_closed.xml`, which has the 188 mm foreleg as well as another foot, so it says nothing "
                "about 128 against 160 mm)\n")
    rows.append("| case | measure | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    for case in ("cmd_0.15", "nominal_0.25", "otherfoot_0.25"):
        cs = {k: gait(v, case) for k, v in ALL.items()}
        name = labels[case]["label"]
        for label, fn in (
                ("falls of 5", lambda t: str(t["falls"])),
                ("forward speed, m/s", lambda t: f((t.get("speed") or {}).get("forward_speed_mps"), 3)),
                ("worst drive RMS / continuous", lambda t: f(max(
                    d["rms_over_continuous"]["mean"] for d in t["drives"]["per_joint"].values()), 3)),
                ("worst front drive RMS / continuous", lambda t: f(max(
                    t["drives"]["per_joint"][j]["rms_over_continuous"]["mean"] for j in FRONT), 3))):
            rows.append(f"| {name} | {label} | " + " | ".join(fn(cs[k]) for k in cols) + " |")
    rows.append("")


def foot_tables(rows):
    fl = {k: foot(k) for k in ALL}
    rows.append("## D. Foot loads (foot_loads.py: 5 episodes, 20 s at 0.25 m/s, flat floor, randomisation off, noise "
                "on, 10 ms delay; from 2 s; run 3's two seeds measured on 26 Nov with the D22 run commit's tool)\n")
    table(rows, list(ALL), [
        ("episodes that fell, of 5", lambda c: str(fl[c]["falls"])),
        ("worst front foot, 99th percentile of the 10 ms mean (L), N", lambda c: f(fl[c]["front_p99_window_n"], 1)),
        ("worst hind foot, the same, N", lambda c: f(fl[c]["hind_p99_window_n"], 1)),
        ("largest 10 ms mean, any foot, any episode, N", lambda c: f(fl[c]["worst_foot_window_max_n"], 1)),
        ("largest 1 ms sample, any foot, any episode, N", lambda c: f(fl[c]["worst_foot_raw_max_n"], 1)),
    ], "measure")
    rows.append("PR #697's figures from its own scripted controllers, for scale only (other controllers, other "
                "motions; its turning loads have no counterpart here, since these policies walk straight):\n")
    rows.append("| PR #697 (commit 65a44fad, refinement60/mid160/README.md) | 128 mm | 160 mm |\n|---|---|---|")
    for label, a, b in PR697:
        rows.append(f"| {label} | {a} | {b} |")
    rows.append("")


def demo_tables(rows):
    d = {k: json.loads((D22 / "demo" / v / "demo_course.json").read_text()) for k, v in RUNS.items()}
    rows.append("## E. The demo course (demo_course.py: 90 s walking at 0.25 m/s and 30 s standing, repeated for one "
                "pack; temperatures and battery are model estimates)\n")
    rows.append("Provenance: demo_course.py records no checkpoint hash. Each was run from the run commit's export on "
                "the run's downloaded final pair (the folder of the same name; the pairs' sha256 are in "
                "runs/trike60-d22_LOG.md), on 26 Nov; the printout is stored beside each output.\n")
    cont = {k: v["continuous_nm"] for k, v in gait_all("fore128-s0")["drives"].items()}
    group = lambda j: ("roll" if j.endswith("roll") else "pitch" if j.endswith("pitch")  # noqa: E731
                       else "front elbow" if "fore" in j else "hind knee")

    def stand_worst(c, seg="stand"):
        sg = d[c]["segments"][seg]
        ratio = {j: t / cont[group(j)] for j, t in zip(sg["joints"], sg["tau_drive_rms"])}
        j = max(ratio, key=ratio.get)
        return j, ratio[j]

    def hottest(c, tau="tau_480s"):
        t = d[c]["temperatures"]
        j = max(t, key=lambda j: t[j][tau]["peak_c"])
        return j, t[j][tau]

    table(rows, list(RUNS), [
        ("valid (no segment fell)", lambda c: "yes" if d[c]["valid"] else "NO"),
        ("speed in the 0.25 m/s segment, m/s", lambda c: f(d[c]["segments"]["walk_0.25"]["speed_mps"], 3)),
        ("hottest winding, peak, 8 min time constant, C (drive)", lambda c: f"{f(hottest(c)[1]['peak_c'], 1)} "
         f"({hottest(c)[0]})"),
        ("the same with 4 / 15 min time constants, C", lambda c: f"{f(hottest(c, 'tau_240s')[1]['peak_c'], 1)} / "
         f"{f(hottest(c, 'tau_900s')[1]['peak_c'], 1)}"),
        ("hottest front winding, peak, 8 min, C (drive)", lambda c: (lambda j: f"{f(d[c]['temperatures'][j]['tau_480s']['peak_c'], 1)} ({j})")(
            max(FRONT, key=lambda j: d[c]["temperatures"][j]["tau_480s"]["peak_c"]))),
        ("standing segment: worst drive RMS / continuous (drive)", lambda c: f"{f(stand_worst(c)[1], 2)} "
         f"({stand_worst(c)[0]})"),
        ("standing segment: feet carrying no load throughout", lambda c: ", ".join(
            k.replace("_sole", "") for k, v in d[c]["segments"]["stand"]["loads"]["feet"].items()
            if v["normal_n"]["max"] == 0.0) or "none"),
        ("walking segment (0.25 m/s): worst drive RMS / continuous (drive)", lambda c: f"{f(stand_worst(c, 'walk_0.25')[1], 2)} "
         f"({stand_worst(c, 'walk_0.25')[0]})"),
        ("any winding reaching its limit", lambda c: "yes" if any(
            v[t]["minutes_to_limit"] is not None for v in d[c]["temperatures"].values() for t in v) else "no"),
        ("runtime per pack, 30 W hotel (25 / 40 W), min", lambda c: f"{f(d[c]['battery']['runtime_minutes'], 0)} "
         f"({f(d[c]['battery']['runtime_minutes_by_hotel_w']['25'], 0)} / "
         f"{f(d[c]['battery']['runtime_minutes_by_hotel_w']['40'], 0)})"),
        ("pack current at low charge: mean / peak, A", lambda c: f"{f(d[c]['battery']['current_mean_a']['low_v'], 1)}"
         f" / {f(d[c]['battery']['current_peak_all_segments_a']['low_v'], 1)}"),
        ("loads below: over the three measured segments (0.25 and 0.15 m/s walking, standing)", lambda c: ""),
        ("joint force, largest 99th percentile, body weights (joint)", lambda c: f"{f(d[c]['loads_headline']['joint_force_p99_bw']['value'], 2)} ({d[c]['loads_headline']['joint_force_p99_bw']['where']})"),
        ("bending, largest, N m (joint)", lambda c: f"{f(d[c]['loads_headline']['bending_max_nm']['value'], 1)} "
         f"({d[c]['loads_headline']['bending_max_nm']['where']})"),
        ("foot force, largest, body weights (foot)", lambda c: f"{f(d[c]['loads_headline']['foot_max_bw']['value'], 2)}"
         f" ({d[c]['loads_headline']['foot_max_bw']['where']})"),
    ], "measure")


TRAINING = "servo delay 0-10 ms; mass x0.9-1.1; kp x0.9-1.1; force limit x0.85-1.0; friction 0.3-1.0; steps to 4 cm; " \
           "ramps to 10 deg; pushes to 1.0 m/s; no payload"


def in_training(family, x):
    """Whether a sweep point lies inside the D22 runs' training ranges (run records' env_config)."""
    if family.startswith("delay"):
        return x <= 10
    if family == "friction":
        return x is None or 0.3 <= x <= 1.0
    lims = {"mass, every body": (0.9, 1.1), "force limit": (0.85, 1.0), "position gain kp": (0.9, 1.1)}
    if family in lims:
        return lims[family][0] <= x <= lims[family][1]
    if family == "payload on the chassis":
        return x == 0.0
    if family == "sideways push":
        return abs(x) <= 1.0
    if family in ("step up", "ramp up"):
        return x is None or (x[0] == "step" and x[1] <= 0.04) or (x[0] == "ramp" and x[1] <= 10)
    raise ValueError(family)


def atypical(family, x):
    """Inside the ranges, but beyond what training produced. The sweep's ramps do not end, so the robot climbs for the
    whole episode, against training's 0.8-2.0 m up, a top and a ramp down. Its mass and force-limit points scale every
    body or every drive at once, where training draws each body's mass and each drive's limit separately."""
    return ((family == "ramp up" and x is not None) or (family == "mass, every body" and x != 1.0)
            or (family == "force limit" and x != 1.0))


def whole_robot_mass_range(draws=200_000, seed=0):
    """The whole robot's mass factor that training's per-body draws give (U(0.9, 1.1) per body, weighted by the 128 mm
    model's body masses): min, max and SD over `draws` episodes."""
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(RL.parent / "sim" / "generated" / "trike60_C_closed_capsule_fore128.xml"))
    w = m.body_mass
    s = (np.random.default_rng(seed).uniform(0.9, 1.1, size=(draws, m.nbody)) * w).sum(1) / w.sum()
    return float(s.min()), float(s.max()), float(s.std())


def sweep_tables(rows):
    s = {k: sweep(v) for k, v in ALL.items()}
    cols = list(ALL)
    rows.append("## F. Every sweep point (nominal model, noise on, 8 episodes, 0.25 m/s unless stated)\n")
    over = {}
    for k in cols:
        seen, cnt = set(), {False: [0, []], True: [0, []]}
        for fam in s[k]["families"]:
            for r in fam["rows"]:
                if r["key"] in seen or not in_training(fam["family"], r["x"]):
                    continue
                seen.add(r["key"])
                b = cnt[atypical(fam["family"], r["x"])]
                b[0] += 1
                h = s[k]["points"][r["key"]]["summary"].get("heat_worst")
                if h is not None and h >= 1.0:
                    b[1].append(h)
        over[k] = cnt
    lo, hi, sd = whole_robot_mass_range()
    rows.append(f"Points inside the training ranges ({TRAINING}) where the worst drive is at or over its continuous "
                "rating. \"Typical of training\": like what training produced. \"Beyond what training produced\": the "
                "sweep's ramps, which do not end, so the robot climbs for the whole episode (about 4 m), against "
                "training's 0.8-2.0 m up, a top and a ramp down; and its x0.9 and x1.1 masses on every body at once and "
                "x0.85 force limit on every drive at once, where training draws them per body and per drive (the whole "
                f"robot's mass factor in training: {lo:.3f}-{hi:.3f}, SD {sd:.4f}, over 200,000 simulated draws on the "
                "128 mm model):\n")

    def cell(b):
        if not b[1]:
            return f"{len(b[1])} of {b[0]}"
        rng = f(min(b[1]), 3) if min(b[1]) == max(b[1]) else f"{f(min(b[1]), 3)}-{f(max(b[1]), 3)}"
        return f"{len(b[1])} of {b[0]} ({rng})"
    table(rows, cols, [("typical of training", lambda c: cell(over[c][False])),
                       ("beyond what training produced", lambda c: cell(over[c][True]))], "")
    rows.append("Each cell: falls of 8; speed m/s; sideways drift, mean of its absolute value, m (the figure T3 and T4 "
                "read); worst joint RMS / continuous rating. "
                "Speed, drift and heat are over the episodes that did not fall; \"all fell\" has none.\n")
    for fam in s["188 s0 (run 3)"]["families"]:
        rows.append(f"**{fam['family']}** ({fam['x_key']})\n")
        rows.append("| x | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
        for r in fam["rows"]:
            cells = []
            for k in cols:
                p = s[k]["points"].get(r["key"])
                if p is None:
                    cells.append("not run")
                    continue
                p = p["summary"]
                cells.append(f"{p['falls']}, all fell" if p["falls"] == p["episodes"] else
                             f"{p['falls']}; {f(p.get('speed_mps'), 3)}; {f(p.get('drift_abs_mean_m'), 2)}; "
                             f"{f(p.get('heat_worst'), 2)}")
            x = "model" if r["x"] is None else (" ".join(map(str, r["x"])) if isinstance(r["x"], list) else r["x"])
            rows.append(f"| {x} | " + " | ".join(cells) + " |")
        rows.append("")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "REPORT_FORELEG_D22_APPENDIX.md"))
    a = ap.parse_args(argv)
    judge = json.loads((D22 / "judge_d22.json").read_text())
    rows = ["# The foreleg comparison (D22), appendix: the judged measures and the main reported figures (simulation)\n",
            "Generated by `analysis/d22_appendix.py` from the stored tool outputs. Columns: the eight D22 runs "
            "(upper foreleg length and seed) and, where the same tool measured it, run 3's two seeds (188 mm), as a "
            "reference only: its foreleg cannot be built (F18) and its fore-assembly differs besides the length. "
            "Rules and the choice: PREREG_FORELEG_D22.md; the report: REPORT_FORELEG_D22.md. The full per-run "
            "record (balance, slip, duty factors, head steadiness, the command sweep and every other figure the tools "
            "give) is each run's `analytics/results/<run>/gait_report.md` and `robustness/results/<run>/"
            "summary_<run>.md`.\n",
            "Proportions (geometry, PLAN_FORELEG_D22.md): upper foreleg over upper hind leg 0.681 (128 mm) and "
            "0.851 (160 mm); the specimen RGM.1332500's humerus over femur 0.682. Posture: the body's mean pitch, "
            "roll and height are in C; the elbow's angle was not measured by these tools.\n"]
    r = judged(rows, judge)
    rules_by_length(rows, r, judge)
    gait_tables(rows)
    foot_tables(rows)
    demo_tables(rows)
    sweep_tables(rows)
    Path(a.out).write_text("\n".join(rows) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
