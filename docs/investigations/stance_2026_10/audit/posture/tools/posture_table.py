"""Write the posture comparison table (text) from posture_summary.json, posture_episodes.json and the time course.

    python -P posture_table.py --dir OUT_DIR

Every model number printed here is read from those files; the published column is fixed text with its source.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

T = ("statue", "s60", "s52")
ROWS = [
    ("Femur from vertical, deg (+ knee ahead of hip)", "mean_femur_from_vertical_deg", "femur_from_vertical_deg", 1),
    ("Tibia from vertical, deg (+ ankle ahead of knee)", "mean_tibia_from_vertical_deg", "tibia_from_vertical_deg", 1),
    ("Metatarsus from vertical, deg", "mean_metatarsus_from_vertical_deg", "metatarsus_from_vertical_deg", 1),
    ("Knee included angle, deg (180 straight)", "mean_knee_included_deg", "knee_included_deg", 1),
    ("Ankle included angle, deg", "mean_ankle_included_deg", "ankle_included_deg", 1),
    ("MTP included angle (metatarsus vs digit III), deg", "mean_mtp_included_deg", "mtp_included_deg", 1),
    ("Digit III from horizontal, deg", "mean_digit3_from_horizontal_deg", None, 2),
    ("Pelvis pitch, deg (+ nose down)", "pelvis_pitch_deg", "pelvis_pitch_deg", 2),
    ("COM ahead of hip, % femur", "com_ahead_of_hip_pct_femur", "com_ahead_of_hip_pct_femur", 1),
    ("COM below hip, % femur", "com_below_hip_pct_femur", "com_below_hip_pct_femur", 1),
    ("COM over foot, pad statistic (signed)", "mean_com_padstat", "com_ahead_of_padmid_padstat", 2),
    ("COM over foot, footprint fraction from rear", "mean_com_footfrac", "com_footfrac", 3),
    ("Pad CoP signed (recorder, gate normalisation)", "mean_rec_pad_cop_signed", None, 2),
    ("Pad CoP gate statistic, worse foot", "max_feet_rec_pad_cop_gate", None, 2),
    ("Whole-foot CoP, pad statistic (signed)", "mean_wholefoot_cop_padstat", None, 2),
    ("Whole-foot CoP, footprint fraction from rear", "mean_wholefoot_cop_footfrac", None, 3),
    ("Whole-foot CoP ahead of digit-III MTP, m", "mean_wholefoot_cop_ahead_of_mtp_m", None, 3),
    ("Knee ahead of foot's vertical GRF line, m", "mean_knee_ahead_of_vGRF_m", None, 3),
    ("Knee ahead of vertical GRF line, % femur", "mean_knee_ahead_of_vGRF_pct_femur", None, 1),
    ("Knee to actual GRF line (signed), m", "mean_knee_to_GRFline_m", None, 3),
    ("Hip ahead of vertical GRF line, m", "mean_hip_ahead_of_vGRF_m", None, 3),
    ("Ankle ahead of vertical GRF line, m", "mean_ankle_ahead_of_vGRF_m", None, 3),
    ("GRF tilt from vertical, deg (+ forward)", "mean_GRF_lean_deg", None, 2),
    ("Pad load share (per-foot mean)", "mean_pad_share", None, 3),
    ("Digit load share (per-foot mean)", "mean_digit_share", None, 3),
    ("  right / left digit share", ("r_digit_share", "l_digit_share"), None, 3),
    ("  digit II / III / IV share", ("mean_d2_share", "mean_d3_share", "mean_d4_share"), None, 3),
    ("Right foot share of both feet' load", "r_load_share_of_both", None, 3),
    ("COM horizontal speed RMS, m/s", "com_hspeed_rms_mps", None, 4),
]


def fmt(a, nd):
    return f"{a['mean']:.{nd}f} ± {a['sd']:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    S = json.load(open(os.path.join(args.dir, "posture_summary.json")))
    E = json.load(open(os.path.join(args.dir, "posture_episodes.json")))
    TC = json.load(open(os.path.join(args.dir, "posture_timecourse.json")))
    home = S["home_keyframe"]
    lines = []
    m = S["meta"]
    lines.append("T. rex anatomical, main 2e734e29 (physics r9), stance stage UNMODIFIED (options off), heading 0, "
                 f"deterministic, seeds {m['seeds'][0]}-{m['seeds'][-1]} ({len(m['seeds'])} per target), horizon {m['horizon']}")
    for k, v in m["checkpoints"].items():
        lines.append(f"  {k}: {v['zip']} sha256 {v['zip_sha256'][:16]}... vecnorm sha256 {v['vecnorm_sha256'][:16]}...")
    lines.append("Values: mean ± sd over the 10 episodes of each episode's window mean; per-foot quantities averaged over"
                 " the two feet unless split.  Windows: w40 = steps 40-999 (40-step settle), w200 = steps 200-999 (the"
                 " gate's settle), last40 = steps 960-999.")
    for t in T:
        d = S["targets"][t]
        lines.append(f"  {t}: full-horizon {d['full_horizon']}/{d['n_episodes']}, clean on the non-statue v2 criteria "
                     f"{d['clean_nonstatue_criteria']}/{d['n_episodes']}")
    lines.append("Checks: " + json.dumps(S["checks"]))
    for w in ("w40", "w200", "last40"):
        lines.append("")
        lines.append(f"=== window {w}")
        lines.append(f"{'quantity':52s}{'home keyframe':>16s}" + "".join(f"{t:>22s}" for t in T))
        for label, key, hkey, nd in ROWS:
            hv = f"{home[hkey]:.{nd}f}" if hkey and hkey in home else ""
            if isinstance(key, tuple):
                cells = []
                for t in T:
                    W = S["targets"][t]["windows"][w]
                    cells.append(" / ".join(f"{W[k]['mean']:.{nd}f}" for k in key))
                lines.append(f"{label:52s}{hv:>16s}" + "".join(f"{c:>22s}" for c in cells))
            else:
                lines.append(f"{label:52s}{hv:>16s}" + "".join(f"{fmt(S['targets'][t]['windows'][w][key], nd):>22s}" for t in T))
    lines.append("")
    lines.append("Home keyframe geometry (mj_forward at key 'home'): " + json.dumps({k: round(v, 4) for k, v in home.items()}))
    lines.append("")
    lines.append("Digit load position along the digit (0 = MTP axis, 1 = tip), repo pad telemetry, gate window, mean of 10 episodes:")
    for t in T:
        eps = [e for e in E if e["target"] == t]
        vals = {f"{s}_toe_d{k}": float(np.nanmean([e["repo_pad_telemetry"][f"{s}_toe_d{k}_load_pos"] for e in eps]))
                for s in "rl" for k in (2, 3, 4)}
        lines.append(f"  {t}: " + ", ".join(f"{k} {v:.3f}" for k, v in vals.items()))
    lines.append("")
    lines.append("Time course (100-step blocks, mean over episodes): COM pad statistic / femur deg / digit share")
    for t in T:
        d = TC["targets"][t]
        lines.append(f"  {t} COM padstat: " + " ".join(f"{x:+.2f}" for x in d["com_ahead_of_padmid_padstat"]["mean"]))
        lines.append(f"  {t} femur deg  : " + " ".join(f"{x:.1f}" for x in d["mean_femur_from_vertical_deg"]["mean"]))
        lines.append(f"  {t} digit share: " + " ".join(f"{x:.2f}" for x in d["mean_digit_share"]["mean"]))
        lines.append(f"  {t} knee height m: " + " ".join(f"{x:.3f}" for x in d["mean_knee_height_m"]["mean"]))
    lines.append("")
    lines.append("Per-episode worse-foot gate statistic (w200) and signed pad CoP r/l:")
    for t in T:
        eps = [e for e in E if e["target"] == t]
        lines.append(f"  {t}: " + "; ".join(
            f"{e['seed']}: {e['windows']['w200']['max_feet_rec_pad_cop_gate']:.2f} "
            f"({e['windows']['w200']['r_rec_pad_cop_signed']:+.2f}/{e['windows']['w200']['l_rec_pad_cop_signed']:+.2f})" for e in eps))
    txt = "\n".join(lines) + "\n"
    open(os.path.join(args.dir, "posture_table.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
