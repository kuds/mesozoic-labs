"""One row per diagnostic-panel variant: the stance5 screen's counterfactual panels, summarized.  Report only.

    python3 -I panels_summary.py PANELS_ROOT OUT.csv [--judged STANCE5_JUDGED]

PANELS_ROOT/<run id>/panels_<slot>.json are panels5.py's files (schema stance5-panels/v1), one per judged checkpoint
(the six J slots) of the three stance5 band runs (the control's were rolled for stance4 and are summarized in
../../stance4/results/panels_summary.csv).  Each rolls the judge's 40-episode
panel (seeds 3042-3081, heading 0) under up to four variants: standard (no change), mirror_legs (the left leg's
hinges set to the right leg's at reset), prey_ahead (no lateral prey offset) and mirror_prey_ahead (both).  Every
number below is the file's own: the stance_quality/v2 clean count, K2a and K2b (the judge's statistics and bars),
the full-horizon and fall counts, and the split statistics over the full-horizon episodes (window stagger sd in mm,
the r - l whole-foot CoP split's sd, the common CoP's mean and the lowest foot CoP, in span half-lengths), with the
zero-action statue's K2a on the same spawn and env, and, from the file's band and ankle blocks, the band readouts
over the full-horizon episodes' window foot-steps (the whole-foot CoP outside [-0.6, +0.4], the pad CoP outside
[-0.65, +0.65], the guard at 0 and on its ramp, the band paid per episode) and the ankle asymmetry (the panel sd of the
right-minus-left window-mean ankle servo error, deg, and actuator torque, N.m).  Nothing is recomputed and nothing
enters a verdict.  With --judged, each file's model sha256 is compared with the judge file of the same run and slot,
and a standard variant's K2a and K2b with the judge's; any difference stops the script.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

SCHEMA = "stance5-panels/v1"
RUNS = (
    ("stance5_LTSB_s52", "LTSB (stance5)", 52),
    ("stance5_LTSB_s60", "LTSB (stance5)", 60),
    ("stance5_LTSB_s42", "LTSB (stance5)", 42),
)
SLOTS = (
    ("4194304", "4.2M"),
    ("5242880", "5.2M"),
    ("6291456", "6.3M"),
    ("8388608", "8.4M"),
    ("10485760", "10.5M"),
    ("final", "final"),
)
VARIANTS = ("standard", "mirror_legs", "prey_ahead", "mirror_prey_ahead")
COLUMNS = [
    "run_id",
    "arm",
    "seed",
    "slot",
    "checkpoint",
    "variant",
    "verdict",
    "clean",
    "n",
    "full_horizon",
    "falls",
    "hop_or_fall",
    "k2a",
    "k2b",
    "roll_rail",
    "stagger_window_sd_mm",
    "cop_split_sd",
    "cop_common_mean",
    "heel_min",
    "statue_k2a",
    "band_window_foot_steps",
    "band_whole_out",
    "band_pad_out",
    "band_guard_0",
    "band_guard_ramp",
    "band_paid_mean",
    "ankle_servo_err_sd_deg",
    "ankle_torque_sd_nm",
    "checks_ok",
    "model_sha256",
]


def f3(v):
    return "" if v is None else f"{v:.3f}"


def f1(v):
    return "" if v is None else f"{v:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("panels_root")
    ap.add_argument("out_csv")
    ap.add_argument("--judged", help="the stance5 judge files' root, to cross-check model sha256 and K2a/K2b")
    a = ap.parse_args()
    rows, checked = [], 0
    for run, arm, seed in RUNS:
        for slot, label in SLOTS:
            path = os.path.join(a.panels_root, run, f"panels_{slot}.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                d = json.load(fh)
            assert d.get("schema") == SCHEMA and d.get("report_only") is True, path
            assert str(d.get("steps")) == slot, path
            jroot = a.judged
            judge = None
            if jroot:
                with open(os.path.join(jroot, run, f"judge_{slot}.json")) as fh:
                    judge = json.load(fh)
                assert judge.get("model_sha256") == d.get("model_sha256"), (path, "model sha256")
            for name in VARIANTS:
                v = (d.get("variants") or {}).get(name)
                if v is None:
                    continue
                ch = v.get("checks") or {}
                k2eq = ch.get("analyze_split_k2_flags_equal_judge")
                ok = bool(ch.get("clean_independent_of_statue")) and (k2eq is None or all(k2eq.values()))
                if judge is not None and name == "standard":
                    s = judge["panel"]["summary"]
                    assert (v["k2a"], v["k2b"]) == (s["k2a_count"], s["k2b_count"]), (path, "standard K2")
                    checked += 1
                band = v.get("band") or {}
                paid = band.get("paid_per_episode") or {}
                ankle = v.get("ankle_asymmetry") or {}
                rows.append(
                    {
                        "run_id": run,
                        "arm": arm,
                        "seed": seed,
                        "slot": label,
                        "checkpoint": d.get("checkpoint"),
                        "variant": name,
                        "verdict": v.get("verdict"),
                        "clean": v.get("clean"),
                        "n": v.get("episodes"),
                        "full_horizon": v.get("full_horizon"),
                        "falls": v.get("falls"),
                        "hop_or_fall": v.get("hop_or_fall_episodes"),
                        "k2a": v.get("k2a"),
                        "k2b": v.get("k2b"),
                        "roll_rail": v.get("roll_rail_full_horizon"),
                        "stagger_window_sd_mm": f1(v.get("stagger_window_sd_mm")),
                        "cop_split_sd": f3(v.get("cop_split_sd")),
                        "cop_common_mean": f3(v.get("cop_common_mean")),
                        "heel_min": f3(v.get("heel_min")),
                        "statue_k2a": (v.get("statue_same_env") or {}).get("k2a"),
                        "band_window_foot_steps": band.get("window_foot_steps"),
                        "band_whole_out": band.get("whole_out"),
                        "band_pad_out": band.get("pad_out"),
                        "band_guard_0": band.get("guard_0"),
                        "band_guard_ramp": band.get("guard_ramp"),
                        "band_paid_mean": f3(paid.get("mean")),
                        "ankle_servo_err_sd_deg": f3((ankle.get("servo_err_deg") or {}).get("sd_r_minus_l")),
                        "ankle_torque_sd_nm": f1((ankle.get("act_Nm") or {}).get("sd_r_minus_l")),
                        "checks_ok": ok,
                        "model_sha256": d.get("model_sha256"),
                    }
                )
    extra = sorted(
        set(glob.glob(os.path.join(a.panels_root, "*", "panels_*.json")))
        - {os.path.join(a.panels_root, r, f"panels_{s}.json") for r, _, _ in RUNS for s, _ in SLOTS}
    )
    if extra:
        print("not summarized:", ", ".join(os.path.relpath(p, a.panels_root) for p in extra))
    if a.judged:
        print(f"model sha256 equal to the judge file's in every file checked; standard K2a/K2b equal in {checked}")
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
