"""Within-episode time course (100-step blocks) of the main posture measures, from posture_roll.py's arrays.

    python -P posture_timecourse.py --rolls ROLLS_DIR --out OUT.json

Reuses posture_measure.per_step (imported by path), so the definitions are the same.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("posture_measure_defs", os.path.join(HERE, "posture_measure.py"))
PM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PM)

KEYS = ("mean_femur_from_vertical_deg", "mean_knee_included_deg", "mean_ankle_included_deg", "pelvis_pitch_deg",
        "com_ahead_of_hip_pct_femur", "com_ahead_of_padmid_padstat", "mean_wholefoot_cop_padstat",
        "mean_rec_pad_cop_signed", "mean_digit_share", "mean_knee_ahead_of_vGRF_m", "mean_knee_height_m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rolls", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    episodes = [json.loads(line) for line in open(os.path.join(args.rolls, "episodes.jsonl"))]
    blocks = [(a, a + 100) for a in range(0, 1000, 100)]
    res: dict = {"blocks": blocks, "targets": {}}
    for t in ("statue", "s60", "s52"):
        per = {k: [] for k in KEYS}
        for ep in (e for e in episodes if e["target"] == t):
            z = dict(np.load(os.path.join(args.rolls, f"steps_{t}_{ep['seed']}.npz")))
            ps = PM.per_step(z)
            ps["mean_knee_height_m"] = 0.5 * (z["r_knee"][:, 2] + z["l_knee"][:, 2])
            for b in ("femur_from_vertical_deg", "knee_included_deg", "ankle_included_deg", "wholefoot_cop_padstat",
                      "rec_pad_cop_signed", "knee_ahead_of_vGRF_m"):
                ps[f"mean_{b}"] = 0.5 * (ps[f"r_{b}"] + ps[f"l_{b}"])
            ps["mean_digit_share"] = 0.5 * (ps["r_rec_digits_N"] / ps["r_rec_leg_N"] + ps["l_rec_digits_N"] / ps["l_rec_leg_N"])
            for k in KEYS:
                per[k].append([float(np.nanmean(ps[k][a:b])) for a, b in blocks])
        res["targets"][t] = {k: {"mean": np.mean(v, axis=0).round(4).tolist(), "sd": np.std(v, axis=0, ddof=1).round(4).tolist()}
                             for k, v in per.items()}
    json.dump(res, open(args.out, "w"), indent=1)
    for t, d in res["targets"].items():
        print(t)
        for k in KEYS:
            print(f"  {k:34s}", " ".join(f"{x:8.3f}" for x in d[k]["mean"]))


if __name__ == "__main__":
    main()
