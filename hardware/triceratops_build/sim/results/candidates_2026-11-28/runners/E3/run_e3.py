"""E3 runner: per-policy heat at 60 cm (design/PREREG_CANDIDATES_2026_11_28.md section 2 "E3", section 3 item 7,
section 4 "E3"). DESK (the build's thermal model and arithmetic) on SIMULATED torques (MuJoCo 3.10.0 demo-course
segments of the build's env); nothing is measured. Every number comes from sim/candidate_checks.py (frozen, sha256
008aa308debdf1b6...) and the build tools it calls (rl/demo_course.py run, design/form_factor_study.py, the build's
thermal model rl/telemetry/thermal.py via rl/analysis/thermal_ambient.py); this file only schedules the demo courses,
stores them and tabulates the pre-registered rule.

Subcommands (at most 2 worker processes):
  run      the 14 pre-registered new demo courses: seed0 710000 for every policy, and 700000 for run3-s0, run3-s1 and
           run2b (``demo_course.run`` with the default loop, 90 s walking at 0.25 m/s + 30 s standing, one pack's
           minutes, through ``candidate_checks.demo_make``: the sweep's loaders after the sha256 check)
  repro    SUPPORTING harness check, not part of the rule: the 8 stored demo courses (fore128 x4, fore160 x4, seed0
           700000) re-run through ``demo_make`` and compared with the stored JSON (do the stored courses, made through
           steer_eval's loader, and the new ones come from the same pipeline?)
  judge    the tables, the per-policy verdicts and the harness checks; writes sim/results/candidates_2026-11-28/E3/

Operational readings fixed in this file BEFORE any E3 demo course ran (28 Nov 2026; the pre-registration is silent on
them; none loosens a rule):
  (i)  A demo course with a fallen segment (``valid`` False in rl/demo_course.py) does not count as meeting the heat
       rules on that course, whatever its C1 and C2 (a fall is not a pass); one that fell before the measuring window
       has no heat figure and does not meet either. (Stricter than reading C1 / C2 alone.)
  (ii) The demo courses run at demo_course.run's default room (None, i.e. the calibration's 25 C), as E2's block-B
       courses did, so that the fore128 seed0 710000 courses can be compared with E2's bit for bit; the room enters
       only the course's own 25 C temperature field, never its torques or minutes. C1 and C2 are then computed at 35 C
       by ``candidate_checks.policy_heat`` (method A, 8 min; 4 and 15 min beside), as pre-registered.
  (iii) "Each drive's demo-loop RMS over continuous": the loop RMS (90 s walking + 30 s standing, weighted) of the
       drive-side torque over the build's continuous rating (X4 7.5 N m, X6 12 N m), as ``policy_heat`` gives it;
       walking and standing RMS (N m) beside it.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

BUILD = Path("/home/user/mesozoic-labs/hardware/triceratops_build")
sys.path.insert(0, str(BUILD / "sim"))
import candidate_checks as C  # noqa: E402

RUNNER = Path(__file__).resolve()
OUT = C.CAND_SCRATCH / "E3"
DEMO = OUT / "demo"
REPRO = OUT / "repro"
RES = C.RESULTS / "E3"
WORKERS = 2
FROZEN_CODE_SHA16 = "008aa308debdf1b6"
LOOP = [("walk_0.25", 90.0), ("stand", 30.0)]
POLICIES = list(C.POLICY_DIRS)                     # fore128 x4, fore160 x4, run3-s0, run3-s1, run2b
STORED = [p for p in POLICIES if (C.DEMO_DIR / p / "demo_course.json").exists()]   # fore128 x4, fore160 x4
MODEL_LABEL = {
    "trike60_C_closed_capsule_fore128.xml": "adopted 60 cm model (128 mm foreleg, capsule foot)",
    "trike60_C_closed_capsule_fore160.xml": "60 cm, 160 mm foreleg, capsule foot (not the adopted model)",
    "trike60_C_closed_capsule.xml": "60 cm, 188 mm foreleg, capsule foot (not the adopted model)",
    "trike60_C_closed_ankleroll.xml": "60 cm, box foot with a passive roll ankle (not the adopted model)",
}
# Cross-check figures (DESK, the stored demo courses at 35 C, method A, 8 min), with their sources:
REPORT_HEAT35_C1 = {"fore128-s0": 76.3, "fore128-s1": 74.4, "fore128-s2": 93.3, "fore128-s3": 98.3,
                    "fore160-s0": 123.2, "fore160-s1": 97.2, "fore160-s2": 92.6, "fore160-s3": 80.6}
REPORT_HEAT35_SRC = "rl/analysis/REPORT_HEAT_35C.md section 3 table, row '35 C | A' (27-28 Nov 2026)"
STUDY_TABLE = {  # design/FORM_FACTOR_STUDY_50_80CM.md section 1.4, "Each policy's deciding drive and its mirror"
    "fore128-s0": ("left_hind_pitch", 0.761, "right_hind_pitch", 0.632, 0.700, 68.8),
    "fore128-s1": ("left_fore_pitch", 0.746, "right_fore_pitch", 0.585, 0.682, 66.8),
    "fore128-s2": ("left_fore_pitch", 0.881, "right_fore_pitch", 0.544, 0.732, 72.6),
    "fore128-s3": ("left_hind_pitch", 0.914, "right_hind_pitch", 0.509, 0.740, 73.4)}
STUDY_SRC = "design/FORM_FACTOR_STUDY_50_80CM.md section 1.4, deciding drive / mirror table (27 Nov 2026)"


def jdump(x):
    return json.dumps(x, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else
                      (float(o) if isinstance(o, (np.floating,)) else str(o))) + "\n"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_frozen():
    got = C.sha256_file(C.HERE / "candidate_checks.py")[:16]
    if got != FROZEN_CODE_SHA16:
        raise SystemExit(f"sim/candidate_checks.py sha256 {got} differs from the frozen {FROZEN_CODE_SHA16}")


# ---------------------------------------------------------------------------------------------------- demo courses
def _init_worker():
    import torch
    torch.set_num_threads(1)


def demo_path(root: Path, policy: str, seed0: int) -> Path:
    return root / policy / f"demo_course_seed{seed0}.json"


def demo_task(task):
    root, policy, seed0 = task
    out = demo_path(Path(root), policy, seed0)
    if out.exists():
        return str(out)
    import demo_course
    make, label = C.demo_make(policy)
    t0 = time.time()
    res = demo_course.run(make, label, LOOP, None, seed0)
    res["_wall_s"] = time.time() - t0
    res["_policy"] = policy
    res["_model"] = Path(C.run_env_config(policy)["model_path"]).name
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float) + "\n")
    tmp.replace(out)
    mins = res["minutes"]
    log(f"demo {policy} seed0 {seed0}: valid={res['valid']} fallen={res['fallen_segments']} "
        f"minutes={'None' if mins is None else f'{mins:.2f}'} wall={res['_wall_s']:.1f}s")
    return str(out)


def run_pool(tasks):
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS, initializer=_init_worker) as pool:
        for _ in pool.imap_unordered(demo_task, tasks):
            pass


def prereg_courses() -> list:
    """(policy, seed0, source) of every judged demo course: the stored one (700000) where it exists, else new; and a
    new one at 710000 (section 3 item 7)."""
    out = []
    for p in POLICIES:
        out.append((p, 700000, "stored" if p in STORED else "new"))
        out.append((p, 710000, "new"))
    return out


def cmd_run(a):
    check_frozen()
    for p in POLICIES:                       # the sha256 check before anything runs (load_policy repeats it)
        C.verify_checkpoint(p)
    tasks = [(str(DEMO), p, s0) for p, s0, src in prereg_courses() if src == "new"]
    assert len(tasks) == 14, len(tasks)
    (OUT / "runner_sha256_at_run.txt").write_text(f"{C.sha256_file(RUNNER)}  {RUNNER}\n"
                                                   f"{C.sha256_file(C.HERE / 'candidate_checks.py')}  "
                                                   f"sim/candidate_checks.py\nstarted {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    run_pool(tasks)
    log(f"run: {len(tasks)} demo courses, wall {time.time() - t0:.0f} s")


def cmd_repro(a):
    check_frozen()
    tasks = [(str(REPRO), p, 700000) for p in STORED]
    t0 = time.time()
    run_pool(tasks)
    log(f"repro: {len(tasks)} demo courses, wall {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------------- tables
def load_course(policy: str, seed0: int, source: str) -> tuple:
    if source == "stored":
        p = C.DEMO_DIR / policy / "demo_course.json"
    else:
        p = demo_path(DEMO, policy, seed0)
    d = json.loads(p.read_text())
    assert int(d["seed0"]) == seed0, (policy, seed0, d["seed0"])
    return d, p


def course_row(policy: str, seed0: int, source: str) -> dict:
    d, p = load_course(policy, seed0, source)
    model = Path(C.run_env_config(policy)["model_path"]).name
    segs = d["segments"]
    base = {"policy": policy, "seed0": seed0, "source": source, "file": str(p), "file_sha256": C.sha256_file(p),
            "model": model, "model_label": MODEL_LABEL.get(model, model), "valid": bool(d["valid"]),
            "fallen_segments": d["fallen_segments"], "minutes": d["minutes"],
            "segment_speed_mps": {k: s.get("speed_mps") for k, s in segs.items()},
            "segment_fell": {k: bool(s["fell"]) for k, s in segs.items()},
            "segment_termination": {k: s.get("termination") for k, s in segs.items()}}
    measured = all(s.get("measured") for s in segs.values()) and d["minutes"] is not None
    if not measured:
        return {**base, "measured": False, "MEETS_on_course": False,
                "why_not": "a segment fell before the measuring window: no heat figure (reading (i))"}
    h = C.policy_heat(d)
    joints = segs["stand"]["joints"]
    walk_rms = np.sqrt(np.asarray(segs["walk_0.25"]["tau_drive_sq_mean"], dtype=float))
    stand_rms = np.sqrt(np.asarray(segs["stand"]["tau_drive_sq_mean"], dtype=float))
    h25 = C.heat_rules(d, segs["walk_0.25"]["tau_drive_sq_mean"], segs["stand"]["tau_drive_sq_mean"], d["minutes"],
                       room_c=25.0)
    own25 = None
    if d.get("temperatures"):
        from telemetry.thermal import tc_key
        k = tc_key(C.SCORED_TC_S)
        own25 = max(v[k]["peak_c"] for v in d["temperatures"].values())
    meets_course = bool(h["MEETS"] and d["valid"])
    per_drive = [{"joint": j, "drive": "X6-60" if j.endswith("fore_flexion") else "X4-36-L",
                  "continuous_nm": 12.0 if j.endswith("fore_flexion") else 7.5,
                  "loop_rms_over_continuous": h["loop_ratio"][j], "hottest_winding_c_35c_8min": h["peak_c_8min"][j],
                  "walk_rms_nm": float(walk_rms[i]), "stand_rms_nm": float(stand_rms[i])}
                 for i, j in enumerate(joints)]
    return {**base, "measured": True,
            "C1_c": h["C1_c"], "C1_joint": h["C1_joint"], "C1_by_tc_c": h["C1_by_tc_c"], "C1_pass": h["C1_pass"],
            "C2": h["C2"], "C2_joint": h["C2_joint"], "C2_pass": h["C2_pass"],
            "deciding": h["deciding"], "even_split_C2": h["even_split_C2"], "even_split_C1_c": h["even_split_C1_c"],
            "MEETS_rules_on_course": bool(h["MEETS"]), "MEETS_on_course": meets_course,
            "why_not": None if meets_course else
            ("fallen segment (reading (i))" if h["MEETS"] else
             ", ".join(x for x, ok in (("C1", h["C1_pass"]), ("C2", h["C2_pass"])) if not ok) + " over the line"),
            "check_25c": {"policy_heat_path_C1_c_at_25c": h25["C1_c"], "demo_course_own_C1_c_at_25c": own25},
            "per_drive": per_drive}


def harness_checks(rows: dict) -> dict:
    out = {"stored_vs_REPORT_HEAT_35C": {}, "stored_vs_study_table": {}, "c1_25c_two_paths_max_abs_diff_c": None,
           "fore128_710000_vs_E2_block_B": {}, "repro_700000": {}}
    for p, c1 in REPORT_HEAT35_C1.items():
        r = rows[(p, 700000)]
        out["stored_vs_REPORT_HEAT_35C"][p] = {"report_c": c1, "here_c": round(r["C1_c"], 4),
                                               "agrees_to_0.05": abs(r["C1_c"] - c1) <= 0.05}
    for p, (dj, dr, mj, mr, ev, evc1) in STUDY_TABLE.items():
        r = rows[(p, 700000)]
        dm = r["deciding"]
        out["stored_vs_study_table"][p] = {
            "study": {"deciding": dj, "ratio": dr, "mirror": mj, "mirror_ratio": mr, "even_split": ev,
                      "even_split_C1_c": evc1},
            "here": {"deciding": dm["joint"], "ratio": round(dm["ratio"], 4), "mirror": dm["mirror"],
                     "mirror_ratio": round(dm["mirror_ratio"], 4), "even_split": round(r["even_split_C2"], 4),
                     "even_split_C1_c": round(r["even_split_C1_c"], 3)},
            "agrees": (dm["joint"] == dj and dm["mirror"] == mj and abs(dm["ratio"] - dr) <= 5e-4
                       and abs(dm["mirror_ratio"] - mr) <= 5e-4 and abs(r["even_split_C2"] - ev) <= 5e-4
                       and abs(r["even_split_C1_c"] - evc1) <= 0.05)}
    diffs = [abs(r["check_25c"]["policy_heat_path_C1_c_at_25c"] - r["check_25c"]["demo_course_own_C1_c_at_25c"])
             for r in rows.values() if r.get("measured") and r["check_25c"]["demo_course_own_C1_c_at_25c"] is not None]
    out["c1_25c_two_paths_max_abs_diff_c"] = max(diffs) if diffs else None
    out["c1_25c_two_paths_n"] = len(diffs)
    e2 = C.CAND_SCRATCH / "E2" / "demo"
    for p in C.E1_POLICIES:
        q = demo_path(e2, p, 710000)
        if not q.exists():
            out["fore128_710000_vs_E2_block_B"][p] = "E2 file not present"
            continue
        a = json.loads(demo_path(DEMO, p, 710000).read_text())
        b = json.loads(q.read_text())
        same = all(a["segments"][s]["tau_drive_sq_mean"] == b["segments"][s]["tau_drive_sq_mean"]
                   for s in a["segments"]) and a["minutes"] == b["minutes"]
        out["fore128_710000_vs_E2_block_B"][p] = {"identical_tau_drive_sq_mean_and_minutes": bool(same),
                                                  "e2_file": str(q), "e2_file_sha256": C.sha256_file(q)}
    for p in STORED:
        q = demo_path(REPRO, p, 700000)
        if not q.exists():
            out["repro_700000"][p] = "not run"
            continue
        a = json.loads(q.read_text())
        b = C.stored_demo(p)
        rel = 0.0
        for s in b["segments"]:
            x = np.asarray(a["segments"][s]["tau_drive_sq_mean"], dtype=float)
            y = np.asarray(b["segments"][s]["tau_drive_sq_mean"], dtype=float)
            rel = max(rel, float(np.max(np.abs(x - y) / np.abs(y))))
        out["repro_700000"][p] = {"max_rel_diff_tau_drive_sq_mean": rel,
                                  "minutes_stored": b["minutes"], "minutes_rerun": a["minutes"],
                                  "C1_c_rerun_35c": C.policy_heat(a)["C1_c"] if a["minutes"] else None}
    return out


def cmd_judge(a):
    check_frozen()
    t0 = time.time()
    courses = prereg_courses()
    rows = {(p, s0): course_row(p, s0, src) for p, s0, src in courses}
    verdict = {}
    for p in POLICIES:
        r7, r71 = rows[(p, 700000)], rows[(p, 710000)]
        verdict[p] = {"model": r7["model"], "model_label": r7["model_label"],
                      "courses": {str(s0): {"source": rows[(p, s0)]["source"], "valid": rows[(p, s0)]["valid"],
                                            **({k: rows[(p, s0)][k] for k in ("C1_c", "C1_joint", "C1_pass", "C2",
                                                                               "C2_joint", "C2_pass")}
                                               if rows[(p, s0)]["measured"] else {"measured": False}),
                                            "MEETS_on_course": rows[(p, s0)]["MEETS_on_course"]}
                                  for s0 in (700000, 710000)},
                      "MEETS_HEAT_RULES_AT_60CM": bool(r7["MEETS_on_course"] and r71["MEETS_on_course"])}
    meets = [p for p, v in verdict.items() if v["MEETS_HEAT_RULES_AT_60CM"]]
    hc = harness_checks(rows)
    RES.mkdir(parents=True, exist_ok=True)
    extra = {f"model:{m}": BUILD / "sim" / "generated" / m for m in MODEL_LABEL}
    summary = {
        "experiment": "E3 per-policy heat (DESK on SIMULATED torques)", "date": C.DATE,
        "prereg": "design/PREREG_CANDIDATES_2026_11_28.md (section 2 E3, section 3 item 7, section 4 E3)",
        "evidence": "DESK: the build's thermal model (rl/telemetry/thermal.py via rl/analysis/thermal_ambient.py, "
                    "method A, fitted at the 25 C rating room, run in a 35 C room; open air, no shell) and arithmetic, "
                    "on SIMULATED drive torques from rl/demo_course.py segments (MuJoCo 3.10.0); nothing measured",
        "rule": "MEETS THE HEAT RULES AT 60 CM: C1 (hottest winding over the demo, 8 min time constant) <= 90 C and C2 "
                "(worst drive's loop RMS over its continuous rating; X4-36-L 7.5 N m, X6-60 12 N m, drive side) <= "
                "0.80, at 35 C with today's drives (D0), on BOTH demo courses (seed0 700000 and 710000). Per-policy "
                "reading, not the study's worst-of-four criterion.",
        "operational_readings_fixed_before_any_E3_run": [
            "(i) a demo course with a fallen segment does not count as meeting on that course (stricter)",
            "(ii) demo courses run at demo_course.run's default room; C1/C2 recomputed at 35 C by policy_heat",
            "(iii) per-drive figure = loop RMS (90 s walk + 30 s stand) over the continuous rating; walk and stand "
            "RMS beside"],
        "units": {"C1_c": "deg C (winding, peak over the demo)", "C2": "dimensionless (RMS / continuous rating)",
                  "rms_nm": "N m, drive side", "minutes": "min (one pack's runtime on the loop)",
                  "speed_mps": "m/s, 2-20 s"},
        "verdict": verdict, "MEET": meets, "do_not_meet": [p for p in POLICIES if p not in meets],
        "fore128_meet": [p for p in meets if p.startswith("fore128")],
        "courses": {f"{p}@{s0}": rows[(p, s0)] for p, s0, _ in courses},
        "harness_checks": hc,
        "sources": {"report_heat35": REPORT_HEAT35_SRC, "study_table": STUDY_SRC,
                    "ratings": "sim/data/drives.json (maker figures via the repo catalogue, read 7 Nov 2026)",
                    "room_and_rule": "D48 (35 C room), bench B3 (90 C winding); C2 line 0.80 (study C2)"},
        "seeds": {"seed0": [700000, 710000], "segments_per_course": "seed0, seed0+1, seed0+2 (walk 0.25, walk 0.15, "
                                                                    "stand), demo_course.run"},
        "wall_s_judge": None,
        "provenance": C.provenance(RUNNER, extra=extra),
    }
    summary["wall_s_judge"] = time.time() - t0
    (RES / "e3_summary.json").write_text(jdump(summary))
    # headline CSV, one row per (policy, demo course)
    with (RES / "e3_policy_heat.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "model", "seed0", "source", "valid", "minutes", "walk_0.25_speed_mps", "C1_c_35c_8min",
                    "C1_c_4min", "C1_c_15min", "C1_joint", "C1_pass", "C2", "C2_joint", "C2_pass", "deciding_drive",
                    "deciding_ratio", "mirror_drive", "mirror_ratio", "deciding_stand_rms_nm", "mirror_stand_rms_nm",
                    "even_split_C2", "even_split_C1_c", "meets_on_course", "MEETS_HEAT_RULES_AT_60CM"])
        for p, s0, src in courses:
            r = rows[(p, s0)]
            if not r["measured"]:
                w.writerow([p, r["model"], s0, src, r["valid"], r["minutes"]] + [""] * 17 +
                           [r["MEETS_on_course"], verdict[p]["MEETS_HEAT_RULES_AT_60CM"]])
                continue
            dm, tc = r["deciding"], r["C1_by_tc_c"]
            w.writerow([p, r["model"], s0, src, r["valid"], f"{r['minutes']:.2f}",
                        f"{r['segment_speed_mps']['walk_0.25']:.4f}", f"{r['C1_c']:.2f}",
                        f"{tc['tau_240s']:.2f}", f"{tc['tau_900s']:.2f}",
                        r["C1_joint"], r["C1_pass"], f"{r['C2']:.4f}", r["C2_joint"], r["C2_pass"], dm["joint"],
                        f"{dm['ratio']:.4f}", dm["mirror"], f"{dm['mirror_ratio']:.4f}", f"{dm['stand_rms_nm']:.3f}",
                        f"{dm['mirror_stand_rms_nm']:.3f}", f"{r['even_split_C2']:.4f}", f"{r['even_split_C1_c']:.2f}",
                        r["MEETS_on_course"], verdict[p]["MEETS_HEAT_RULES_AT_60CM"]])
    with (RES / "e3_per_drive.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "seed0", "source", "joint", "drive", "continuous_nm", "loop_rms_over_continuous",
                    "hottest_winding_c_35c_8min", "walk_rms_nm", "stand_rms_nm"])
        for p, s0, src in courses:
            r = rows[(p, s0)]
            for q in r.get("per_drive", []):
                w.writerow([p, s0, src, q["joint"], q["drive"], q["continuous_nm"],
                            f"{q['loop_rms_over_continuous']:.4f}", f"{q['hottest_winding_c_35c_8min']:.2f}",
                            f"{q['walk_rms_nm']:.3f}", f"{q['stand_rms_nm']:.3f}"])
    # printout
    for p, s0, src in courses:
        r = rows[(p, s0)]
        if not r["measured"]:
            print(f"{p:11s} {s0} {src:6s} NOT MEASURED valid={r['valid']} fallen={r['fallen_segments']}")
            continue
        dm = r["deciding"]
        print(f"{p:11s} {s0} {src:6s} valid={r['valid']!s:5s} min={r['minutes']:6.2f} v={r['segment_speed_mps']['walk_0.25']:.4f} "
              f"C1={r['C1_c']:6.2f} ({r['C1_joint']}) C2={r['C2']:.4f} ({r['C2_joint']}) "
              f"mirror {dm['mirror']} {dm['mirror_ratio']:.4f} even {r['even_split_C2']:.4f}/{r['even_split_C1_c']:.2f} "
              f"meets_course={r['MEETS_on_course']}")
    for p, v in verdict.items():
        print(f"VERDICT {p}: MEETS THE HEAT RULES AT 60 CM = {v['MEETS_HEAT_RULES_AT_60CM']} ({v['model_label']})")
    print("MEET:", meets)
    print("harness:", json.dumps({k: v for k, v in hc.items()}, default=str)[:4000])
    print("wrote", RES)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("repro")
    sub.add_parser("judge")
    a = ap.parse_args(argv)
    {"run": cmd_run, "repro": cmd_repro, "judge": cmd_judge}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
