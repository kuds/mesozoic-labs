"""Print the K3 report's markdown tables from tables/*.csv (REPORT ONLY).

    python3 k3tables.py ROLLOUTS_DIR
"""
import csv
import os
import sys
from collections import Counter

import numpy as np

R = sys.argv[1]
T = os.path.join(R, "tables")
C = {(r["run"], r["steps"], r["cond"]): r for r in csv.DictReader(open(os.path.join(T, "conditions.csv")))}
E = list(csv.DictReader(open(os.path.join(T, "episodes.csv"))))
P = {(r["run"], r["steps"], r["cond"]): r for r in csv.DictReader(open(os.path.join(T, "pulse.csv")))} if os.path.exists(os.path.join(T, "pulse.csv")) else {}
G = {(r["run"], r["steps"]): r for r in csv.DictReader(open(os.path.join(T, "policy_gain.csv")))} if os.path.exists(os.path.join(T, "policy_gain.csv")) else {}
SHORT = {"stance5_LTSB_s42": "band s42", "stance5_LTSB_s52": "band s52", "stance5_LTSB_s60": "band s60",
         "stance4_LTS_s42": "ctrl s42", "stance4_LTS_s52": "ctrl s52", "stance4_LTS_s60": "ctrl s60"}
STEP = {"4194304": "4.2M", "5242880": "5.2M", "6291456": "6.3M", "7340032": "7.3M", "8388608": "8.4M", "10485760": "10.5M", "final": "final"}
S42 = [("stance5_LTSB_s42", s) for s in ("4194304", "5242880", "6291456", "7340032", "8388608", "10485760", "final")]
FINALS = [("stance5_LTSB_s42", "final"), ("stance5_LTSB_s52", "final"), ("stance5_LTSB_s60", "final"),
          ("stance4_LTS_s42", "final"), ("stance4_LTS_s52", "final"), ("stance4_LTS_s60", "final")]
ALL = S42 + FINALS[1:]


def name(k):
    return f"{SHORT[k[0]]} {STEP[k[1]]}"


def v(key, col, fmt="{:.2f}", scale=1.0):
    r = C.get(key)
    if r is None or r.get(col, "") in ("", "nan"):
        return "-"
    return fmt.format(float(r[col]) * scale)


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


print("### T1 falls of 20 per noise condition (same draws)\n")
conds = ["full", "legs", "tail", "neckhead", "notail", "hip_pitch", "hip_roll", "knee", "ankle", "legs_no_hippitch", "legs_no_hiproll",
         "legs_no_knee", "legs_no_ankle", "hp_sym", "hp_anti", "t3hold0_full"]
rows = []
for k in ALL:
    rows.append([name(k)] + [C[(k[0], k[1], c)]["falls"] if (k[0], k[1], c) in C else "" for c in conds])
table(["checkpoint"] + conds, rows)

print("### T2 what the joint-noise falls look like (condition full)\n")
rows = []
for k in ALL:
    fs = [r for r in E if (r["run"], r["steps"]) == k and r["cond"] == "full" and r["fell"] == "1"]
    if not fs:
        continue
    f = lambda col: np.array([float(r[col]) for r in fs if r.get(col, "") != ""])
    terms = ", ".join(f"{a} {b}" for a, b in Counter(r["term"] for r in fs).most_common())
    dirs = ", ".join(f"{a} {b}" for a, b in Counter(r["direction"] for r in fs).most_common())
    tail = sum(1 for r in fs if r.get("contact_tail_steps_before_end", "") != "")
    other = sum(1 for r in fs for c in ("head", "tibia", "torso", "arms", "thigh", "metatarsus") if r.get(f"contact_{c}_steps_before_end", "") != "")
    unl = Counter(r["unloaded_foot"] or "none" for r in fs)
    rows.append([name(k), len(fs), terms, dirs, f"{np.median(f('time_to_fall_s')):.2f} ({f('time_to_fall_s').min():.2f}-{f('time_to_fall_s').max():.2f})",
                 f"{np.median(f('unload_to_fall_s')):.2f}", ", ".join(f"{a} {b}" for a, b in unl.most_common()),
                 f"{np.median(f('end_pitch_dev_deg')):+.0f} / {np.median(np.abs(f('end_roll_deg'))):.0f}",
                 f"{1000 * np.median(f('stagger_absmax_2s_pre_m')):.0f} ({1000 * f('stagger_absmax_2s_pre_m').min():.0f})",
                 f"{np.median(f('slip_max_loaded_pre_mps')):.2f} / {np.median(f('slip_max_loaded_fall_mps')):.2f}",
                 f"{tail} / {other}"])
table(["checkpoint", "falls", "termination", "direction (COM travel onset->end)", "time to fall s, med (range)", "final unload->end s",
       "foot unloaded at the end", "end pitch dev / abs roll deg", "max abs stagger 2 s pre-onset mm, med (min)",
       "loaded-foot slip pre / fall m/s", "falls with tail / other body contact (last 100 steps)"], rows)

print("### T3 pre-failure CoP, margins, headroom under the noise (full) and without it (noise0)\n")
rows = []
for k in ALL:
    f_, n0 = (k[0], k[1], "full"), (k[0], k[1], "noise0")
    rows.append([name(k), C[f_]["falls"], f"{v(n0, 'wcop_med')} / {v(f_, 'wcop_med')}", v(f_, "wcop_toe_share"), v(f_, "wcop_heel_share"),
                 v(f_, "split_share"), f"{v(f_, 'toe_margin_p5')} / {v(f_, 'heel_margin_p5')}",
                 f"{v(f_, 'xcom_toe_margin_p1_m', '{:.0f}', 1000)} / {v(f_, 'xcom_heel_margin_p1_m', '{:.0f}', 1000)}",
                 f"{v(f_, 'ankle_ff_p99')} / {v(f_, 'ankle_ff_max')}", f"{v(n0, 'com_fa_mean', '{:.0f}', 1000)} / {v(f_, 'com_fa_mean', '{:.0f}', 1000)}"])
table(["checkpoint", "falls", "whole-foot CoP median det / noise", "toe-edge share", "heel-edge share", "split share",
       "toe / heel margin p5 (span half-lengths)", "XCoM toe / heel margin p1 mm", "ankle |force|/limit p99 / max", "COM ahead MTP III det / noise mm"], rows)

print("### T4 recovery under the noise (full), pre-failure steps\n")
rows = []
for k in ALL:
    f_ = (k[0], k[1], "full")
    rows.append([name(k), C[f_]["falls"], v(f_, "stagger_sd_m", "{:.0f}", 1000), v(f_, "hip_pitch_diff_sd_rad", "{:.3f}"),
                 f"{v(f_, 'ar50fh_roll_deg')} / {v(f_, 'ar50fh_com_lat')} / {v(f_, 'ar50fh_com_fa')}",
                 f"{v(f_, 'roll_sd')}", f"{C[f_]['excursions_recovered']}/{C[f_]['excursions']}",
                 f"{C[f_]['unload_events_replanted']}/{C[f_]['unload_events_ge10']}", v(f_, "unload_step_share", "{:.4f}"),
                 G.get(k, {}).get("gain_dmu_per_rad_dq", "-"), G.get(k, {}).get("r2", "-")])
table(["checkpoint", "falls", "stagger sd mm", "hip-pitch r-l sd rad", "0.5 s persistence roll / COM lat / COM fa (full-horizon eps)",
       "roll sd deg", "posture excursions recovered", "foot unloads (>=10 steps) replanted", "unloaded foot-step share",
       "policy d(mu_r-mu_l)/d(q_r-q_l)", "R2"], rows)

if P:
    print("### T5 deterministic differential hip-pitch pulse (0.2 s at step 400; falls of 20, stagger peak mm, settle s)\n")
    rows = []
    for k in ALL:
        if (k[0], k[1], "pulse_anti") not in P and (k[0], k[1], "pulse_anti20") not in P:
            continue
        cells = []
        for c in ("pulse_anti10", "pulse_anti20", "pulse_anti", "pulse_sym"):
            p = P.get((k[0], k[1], c))
            cells.append("-" if p is None else f"{p['falls']} / {float(p['stagger_peak_mm_med']):.0f} / {p['settle_s_med'] if p['settle_s_med'] not in ('', 'nan') else '-'}")
        p20 = P.get((k[0], k[1], "pulse_anti20"), {})
        cells.append(p20.get("dmu_diff_during_med", "-"))
        cells.append(p20.get("dmu_diff_after_100_med", "-"))
        rows.append([name(k)] + cells)
    table(["checkpoint", "anti 0.1", "anti 0.2", "anti 0.3", "common 0.3", "policy d(mu_r-mu_l) during the 0.2 pulse", "same, next 1 s"], rows)

print("### T6 s42 per checkpoint (Q4)\n")
Q = list(csv.DictReader(open(os.path.join(T, "q4_trend.csv"))))
rows = []
for r in Q:
    k = (r["run"], r["steps"])
    rows.append([name(k), r["judge_joint_falls"], r["falls_full"], r["det_t3_ge09_share"], r["satroll_t3_frac90"], r["noise_t3_ge09_share"],
                 r["t3hold0_falls"], r["own_sigma_mean"], r["own_sigma_legs"], r["det_com_fa_mm"], r["det_pitch_deg"],
                 f"{r['det_q_hip_pitch']} / {r['det_q_knee']} / {r['det_q_ankle']}", r["noise_stagger_sd_mm"], r["noise_hip_pitch_diff_sd"]])
table(["checkpoint", "judge joint falls", "rolled falls", "det tail_3 share |cmd|>=0.9", "satroll frac90", "noise tail_3 share",
       "falls, tail_3 held at 0", "own sigma mean", "own sigma legs", "det COM ahead MTP III mm", "det pelvis pitch deg",
       "det hip pitch / knee / ankle rad", "noise stagger sd mm", "noise hip-pitch r-l sd rad"], rows)
