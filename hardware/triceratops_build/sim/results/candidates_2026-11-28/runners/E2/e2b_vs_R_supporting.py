"""E2(b) supporting reading, changed after a result (skeptic finding S5, 28 Nov 2026).

Pairs each series-stiffness variant (rod, belt, cable; belt_1span beside) against the rigid reduced variant R
(r 1.25, capped, eta 0.95, no compliance) on the same seeds and conditions, with the build's own e2b_compare.
R and the variants share r, cap, plant and timestep (0.25 ms). Rod (eta 0.95) differs from R in compliance alone.
Belt (0.98) and cable (0.96) also differ in efficiency, which enters twice: in the drive-side logging (StepLogger:
tau / (r eta) motoring, tau eta / r braking) and in the reduced actuator's joint-side force limit
(write_variant: r x eta x 0.70 x the drive's peak, so 3.2 % and 1.1 % above R's). A second reading re-maps each
variant's reduced drives to R's 0.95 in the logging (C.relog_drive_sq); it removes the logging difference only,
not the force-limit one.

Reads only stored episode JSON files written by e2_run.py (this session's own outputs). No simulation runs.
Not a verdict: E2(b)'s rule is against B1 and B025 (prereg section 4).
Run with: python -B e2b_vs_R_supporting.py   (from a scratch working directory)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_run as E  # noqa: E402  (sets nothing on import beyond paths and constants)

C = E.C
OUT_CSV = C.RESULTS / "E2" / "e2b_vs_R_supporting.csv"
OUT_JSON = C.RESULTS / "E2" / "e2b_vs_R_supporting.json"
VARIANTS = ("rod", "belt", "cable", "belt_1span")
R_ETA = C.SOURCES["eta_default"]["value"]   # 0.95, R's efficiency


def main():
    rid_R = E.run_id("R", C.E2B_VARIANTS["R"]["physics_dt"])
    rows, summary = [], {}
    eta_R = {j: R_ETA for j in C.reduced_joints("both")}
    for v in VARIANTS:
        rid = E.run_id(v, C.E2B_VARIANTS[v]["physics_dt"])
        role = C.E2B_VARIANTS[v]["role"]
        d_own, d_eta, dspeed, falls = [], [], [], 0
        for pol in C.E1_POLICIES:
            for cond in C.E2B_CONDITIONS:
                base = E.rows_of(rid_R, pol, cond)
                a = C.e2b_compare(E.rows_of(rid, pol, cond), base)
                b = C.e2b_compare(E.rows_of(rid, pol, cond, eta_R), base)
                row = {"variant": v, "role": role, "policy": pol, "condition": cond, "against": "R",
                       "paired_up": len(a["paired_up"]), "new_falls_vs_R": len(a["new_falls"]),
                       "unstable": len(a["unstable"]),
                       "speed_variant": round(a["speed_variant"], 4), "speed_R": round(a["speed_baseline"], 4),
                       "d_speed_vs_R": round(a["speed_variant"] - a["speed_baseline"], 4),
                       "worst_ratio_R": round(a["worst_ratio_baseline"], 4),
                       "worst_ratio_variant_own_eta": round(a["worst_ratio_variant"], 4),
                       "d_ratio_vs_R_own_eta": round(a["worst_ratio_variant"] - a["worst_ratio_baseline"], 4),
                       "worst_ratio_variant_at_eta_0.95": round(b["worst_ratio_variant"], 4),
                       "d_ratio_vs_R_at_eta_0.95": round(b["worst_ratio_variant"] - b["worst_ratio_baseline"], 4),
                       "within_0.03_at_eta_0.95": bool(b["worst_ratio_variant"] - b["worst_ratio_baseline"] <= 0.03)}
                rows.append(row)
                d_own.append(row["d_ratio_vs_R_own_eta"])
                d_eta.append(row["d_ratio_vs_R_at_eta_0.95"])
                dspeed.append(row["d_speed_vs_R"])
                falls += row["new_falls_vs_R"]
        ep = E.load_ep(rid, C.E1_POLICIES[0], C.E2B_CONDITIONS[0], E.seeds_all()[0])
        etas = sorted({round(float(e), 4) for a, e in zip(ep["actuators"], ep["eta"]) if a in eta_R})
        summary[v] = {"role": role, "eta_reduced_drives": etas,
                      "pairs": len(d_own), "new_falls_vs_R": falls,
                      "d_speed_vs_R_mps": [min(dspeed), max(dspeed)],
                      "d_ratio_vs_R_own_eta": [min(d_own), max(d_own)],
                      "d_ratio_vs_R_at_eta_0.95": [min(d_eta), max(d_eta)]}
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    me = Path(__file__).resolve()
    out = {"experiment": "E2(b) supporting", "label": "changed after a result (skeptic S5); supporting, not judged",
           "date": "2026-11-28", "evidence": "SIMULATION (stored E2(b) episodes) re-read on the desk",
           "what": "each series variant against R (r 1.25, capped, eta 0.95, rigid) on the same seeds 700000-700003 "
                   "and 710000-710003, conditions walk_0.25, push_left_0.5, push_right_0.5; ratio = worst drive's "
                   "pooled RMS / continuous rating (X4 7.5, X6 12 N m), as e2b_compare. Rod differs from R in "
                   "compliance alone; belt and cable also in efficiency (logging and force limit); the eta 0.95 "
                   "columns remove the logging difference only",
           "summary": summary,
           "provenance": {"script": str(me), "script_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
                          "candidate_checks_sha256": hashlib.sha256((C.BUILD / "sim" / "candidate_checks.py")
                                                                    .read_bytes()).hexdigest()
                          if hasattr(C, "BUILD") else None,
                          "episodes": str(E.EP)}}
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    for v, s in summary.items():
        print(v, json.dumps(s))


if __name__ == "__main__":
    main()
