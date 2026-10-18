"""Compare my rolls (vx_roll.py) with the posture audit's per-episode w200 values, seed by seed.

    python -P vx_compare.py MY_JSON POSTURE_EPISODES_JSON TARGET OUT_TXT
"""
import json
import sys

import numpy as np

mine = json.load(open(sys.argv[1]))
theirs = json.load(open(sys.argv[2]))
target = sys.argv[3]
pairs = [
    ("femur from vertical (deg)", lambda w, s: w[f"{s}_femur_from_vertical_deg"], lambda t, s: t[f"{s}_femur_from_vertical_deg"]),
    ("knee included (deg)", lambda w, s: w[f"{s}_knee_incl_deg"], lambda t, s: t[f"{s}_knee_included_deg"]),
    ("ankle included (deg)", lambda w, s: w[f"{s}_ankle_incl_deg"], lambda t, s: t[f"{s}_ankle_included_deg"]),
    ("MTP included (deg)", lambda w, s: w[f"{s}_mtp_incl_deg"], lambda t, s: t[f"{s}_mtp_included_deg"]),
    ("metatarsus from vertical (deg)", lambda w, s: w[f"{s}_metatarsus_from_vertical_deg"], lambda t, s: t[f"{s}_metatarsus_from_vertical_deg"]),
    ("COM proj pad stat (signed)", lambda w, s: w[f"{s}_com_padstat"], lambda t, s: t[f"{s}_com_padstat"]),
    ("COM footprint frac", lambda w, s: w[f"{s}_com_footfrac"], lambda t, s: t[f"{s}_com_footfrac"]),
    ("whole-foot CoP stat", lambda w, s: w[f"{s}_wholefoot_cop_stat"], lambda t, s: t[f"{s}_wholefoot_cop_padstat"]),
    ("pad CoP stat (mine: boundary sample; theirs: recorder)", lambda w, s: w[f"{s}_pad_cop_stat"], lambda t, s: t[f"{s}_rec_pad_cop_signed"]),
    ("knee ahead of vertical through CoP (m)", lambda w, s: w[f"{s}_knee_ahead_of_vcop_m"], lambda t, s: t[f"{s}_knee_ahead_of_vGRF_m"]),
]
glob = [
    ("COM ahead of hip (% femur)", lambda w: w["com_ahead_hip_pct_femur"], lambda t: t["com_ahead_of_hip_pct_femur"]),
    ("COM below hip (% femur)", lambda w: -w["com_above_hip_pct_femur"], lambda t: t["com_below_hip_pct_femur"]),
    ("pelvis pitch nose-down (deg)", lambda w: w["pelvis_pitch_nose_down_deg"], lambda t: t["pelvis_pitch_deg"]),
]
lines = []
tmap = {(e["target"], e["seed"]): e for e in theirs}
agg = {}
for seed, ep in mine["episodes"].items():
    w = ep["window_means"]
    t_ep = tmap[(target, int(seed))]
    t = t_ep["windows"]["w200"]
    lines.append(f"seed {seed}: length mine {ep['length']} theirs {t_ep['length']}")
    for name, fm, ft in glob:
        a, b = fm(w), ft(t)
        lines.append(f"  {name:55s} mine {a:9.4f}  theirs {b:9.4f}  diff {a - b:+.4f}")
        agg.setdefault(name, []).append((a, b))
    for name, fm, ft in pairs:
        for s in "rl":
            a, b = fm(w, s), ft(t, s)
            lines.append(f"  {name + ' ' + s:55s} mine {a:9.4f}  theirs {b:9.4f}  diff {a - b:+.4f}")
            agg.setdefault(name, []).append((a, b))
    lines.append(f"  {'gate stat worse foot (mine boundary)':55s} mine {w['gate_stat_worse_foot']:9.4f}  theirs(recorder) {t_ep['max_sole_cop_fore_aft']:9.4f}")
    lines.append(f"  {'digit share r/l (mine)':55s} {w['r_digit_share']:.4f} / {w['l_digit_share']:.4f}")
lines.append("aggregate over seeds and feet: mean mine / mean theirs / max |diff|")
for name, v in agg.items():
    a = np.array(v)
    lines.append(f"  {name:55s} {a[:, 0].mean():9.4f} {a[:, 1].mean():9.4f} {np.abs(a[:, 0] - a[:, 1]).max():.4f}")
open(sys.argv[4], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
