"""Build k3diag/files/summary.json from extract.json + noise_episodes.csv (+ satroll / satdiag tables).  Files only.

    python3 -I analyze.py K3DIAG_FILES_DIR
"""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import re
import statistics
import sys

SP = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot"
RUNS = ["stance5_LTSB_s52", "stance5_LTSB_s60", "stance5_LTSB_s42", "stance4_LTS_s52", "stance4_LTS_s60", "stance4_LTS_s42"]
KEY_SLOTS = ["6291456", "7340032", "8388608", "10485760", "final"]
J_SLOTS = ["4194304", "5242880", "6291456", "7340032", "8388608", "10485760", "final"]


def f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def direction(r):
    """Fore-aft direction of one fall from its cause and window readouts (window = steps 200..L-1)."""
    cause = r["cause"]
    cm = f(r["foot_cop_mean"])
    rc, lc = f(r["r_foot_cop_fore_aft"]), f(r["l_foot_cop_fore_aft"])
    lateral = (rc is None) != (lc is None) or (rc is not None and lc is not None and abs(rc - lc) >= 0.6)
    if int(r["length"]) <= 200 or cm is None:
        d = "no_window"
    elif cause in ("nosedive", "head_contact", "skull_low"):
        d = "forward"
    elif cause == "tail_contact":
        d = "backward"
    elif cm >= 0.3:
        d = "forward"
    elif cm <= -0.3:
        d = "backward"
    else:
        d = "unclassified"
    return d, lateral


def main(out):
    ex = json.load(open(os.path.join(out, "extract.json")))
    rows = list(csv.DictReader(open(os.path.join(out, "noise_episodes.csv"))))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["run"], r["slot"], r["test"])].append(r)

    S = {"schema": "stance5-k3diag-files/v1", "date": "2026-10-22", "report_only": True,
         "sources": {"judged": [f"{SP}/stance5/judged/<RUN>/judge_<slot>.json", f"{SP}/stance4/judged/<RUN>/judge_<slot>.json"],
                     "ledgers": f"{SP}/stance{{4,5}}/runs/<RUN>/01_stance/{{eval,train}}_episodes.csv",
                     "satroll": f"{SP}/stance5/tools/satroll/stance5_LTSB_s42_<slot>.json",
                     "satdiag": f"{SP}/stance4/satdiag/rollout/tables/checkpoint_table.txt",
                     "derived": [os.path.join(out, "extract.json"), os.path.join(out, "noise_episodes.csv.gz"),
                                 os.path.join(out, "ledger_bins.csv")],
                     "scripts": [os.path.join(out, "scripts/extract.py"), os.path.join(out, "scripts/analyze.py")]},
         "conventions": {
             "joint_test": "noise_matched_control_joint: a = clip(clip(mu) + sigma_ctrl z), sigma_ctrl = frozen_params H4.control_sigma[seed] (mean 0.161), z ~ default_rng(spawn seed); spawns 3042-3061",
             "own_test": "noise_matched_own: a = clip(mu + sigma_own z) in policy space",
             "same_draws": "the z sequence of a spawn seed is the same in every run and checkpoint; band and control of one seed use the same sigma_ctrl vector, so their joint tests are paired draw for draw",
             "window": "per-episode readouts are window means / maxima over steps 200..L-1 (a fall's window ends at the fall); none is an instantaneous value just before the fall",
             "cop_units": "whole-foot CoP foot_cop_fore_aft: span half-lengths, -1 heel line, +1 toe line (pad edges 0.070-0.210 m ahead of the ankle); pad CoP sole_cop_fore_aft: pad units",
             "margins": "toe_margin_nm / heel_margin_nm: window mean of F*(toe_line - x) / F*(x - heel_line), N m (floor-load moment left before the CoP reaches the toe / heel line); *_min = the smaller foot",
             "control_dt_s": 0.01,
             "fall": "episode shorter than 1000 steps",
             "direction_rule": "forward: nosedive/head_contact/skull_low, or 'fallen' with both-feet mean whole-foot CoP >= +0.3; backward: tail_contact, or 'fallen' with mean CoP <= -0.3; lateral_suspect: one foot never loaded in the window or |CoP_l - CoP_r| >= 0.6 (an inference: the files hold no lateral readout)",
             "termination_reasons": "fallen = pelvis below 0.70 m; nosedive = forward_z more than 0.493 below its natural value; tail/head_contact = that geom on the floor (clone base_env.py)"}}

    # ---------------------------------------------------------------- per run x slot: the K3 table
    table = {}
    for run in RUNS:
        R = ex["runs"][run]
        table[run] = {}
        for slot in J_SLOTS + ["9437184", "11534336", "1048576", "2097152", "3145728"]:
            s = R["slots"].get(slot)
            if not s or "joint" not in s:
                continue
            table[run][slot] = {"joint_falls": s["joint"]["falls"], "own_falls": s["own"]["falls"],
                                "joint_causes": s["joint"]["causes"], "policy_std_mean": round(s["own"]["policy_std_mean"], 4),
                                "panel_det_falls": s["panel"]["falls"], "panel_sat_flagged_of_40": s["panel"]["sat_flagged"]}
    S["k3_table"] = table

    # ---------------------------------------------------------------- Q1 what the falls look like
    q1 = {"per_run": {}, "by_cause_all_runs": {}, "fallers_vs_survivors": {}}
    allfalls = [r for r in rows if r["test"] == "joint" and r["fell"] == "True"]
    for run in RUNS:
        fr = [r for r in allfalls if r["run"] == run and r["slot"] in KEY_SLOTS]
        dirs = collections.Counter()
        lat = 0
        for r in fr:
            d, l = direction(r)
            dirs[d] += 1
            lat += int(l)
        lens = [int(r["length"]) for r in fr]
        q1["per_run"][run] = {
            "slots": KEY_SLOTS, "joint_falls": len(fr), "causes": dict(collections.Counter(r["cause"] for r in fr)),
            "direction": dict(dirs), "lateral_suspect": lat,
            "time_to_fall_steps": {"median": med(lens), "min": min(lens) if lens else None, "max": max(lens) if lens else None,
                                   "before_400": sum(1 for x in lens if x < 400)},
            "per_slot": {}}
        for slot in KEY_SLOTS:
            sr = [r for r in fr if r["slot"] == slot]
            dd = collections.Counter(direction(r)[0] for r in sr)
            q1["per_run"][run]["per_slot"][slot] = {"falls": len(sr), "causes": dict(collections.Counter(r["cause"] for r in sr)),
                                                     "direction": dict(dd), "lengths": sorted(int(r["length"]) for r in sr)}
    keys = ["length", "foot_cop_mean", "r_foot_cop_fore_aft", "l_foot_cop_fore_aft", "r_sole_cop_fore_aft", "l_sole_cop_fore_aft",
            "toe_margin_min_nm", "heel_margin_min_nm", "max_tilt_deg", "max_lift_m", "com_mtp3_m", "xc_m", "gate_excess_max",
            "cop_std_max_m", "r_digit_share", "l_digit_share"]
    causes = collections.defaultdict(list)
    for r in allfalls:
        causes[r["cause"]].append(r)
    for c, rs in causes.items():
        q1["by_cause_all_runs"][c] = {"n": len(rs), **{k: mean(f(r[k]) for r in rs) for k in keys}}
    for run in RUNS:
        for slot in KEY_SLOTS:
            rs = by.get((run, slot, "joint"), [])
            if not any(r["fell"] == "True" for r in rs):
                continue
            fl = [r for r in rs if r["fell"] == "True"]
            sv = [r for r in rs if r["fell"] == "False"]
            det = [r for r in by.get((run, slot, "panel"), []) if r["fell"] == "False"]
            q1["fallers_vs_survivors"][f"{run}@{slot}"] = {
                grp: {"n": len(g), **{k: mean(f(r[k]) for r in g) for k in keys}}
                for grp, g in (("fallers", fl), ("survivors", sv), ("deterministic_panel_full_horizon", det))}
    # separation of fallers and survivors by the window maxima of sole tilt / corner lift
    sep = {}
    for run in RUNS:
        sv = [r for r in rows if r["run"] == run and r["test"] == "joint" and r["slot"] in KEY_SLOTS and r["fell"] == "False"]
        fl = [r for r in allfalls if r["run"] == run and r["slot"] in KEY_SLOTS and f(r["max_tilt_deg"]) is not None]
        sep[run] = {"survivors_max_tilt_deg": round(max(f(r["max_tilt_deg"]) for r in sv), 3),
                    "survivors_max_lift_m": round(max(f(r["max_lift_m"]) for r in sv), 4),
                    "survivors_median_tilt_deg": med(f(r["max_tilt_deg"]) for r in sv),
                    "fallers_min_tilt_deg": round(min(f(r["max_tilt_deg"]) for r in fl), 3) if fl else None,
                    "fallers_min_lift_m": round(min(f(r["max_lift_m"]) for r in fl), 4) if fl else None,
                    "survivors_over_roll_rail": sum(1 for r in sv if f(r["max_tilt_deg"]) > 2.0 or f(r["max_lift_m"]) > 0.004),
                    "survivors_tilt_over_3p5": sum(1 for r in sv if f(r["max_tilt_deg"]) > 3.5),
                    "falls": len([r for r in allfalls if r["run"] == run and r["slot"] in KEY_SLOTS])}
    q1["tilt_lift_separation_key_slots"] = sep
    # deterministic -> noise shift of the survivors' whole-foot CoP (all runs)
    shift = {}
    for run in RUNS:
        for slot in KEY_SLOTS:
            sv = [r for r in by.get((run, slot, "joint"), []) if r["fell"] == "False"]
            det = [r for r in by.get((run, slot, "panel"), []) if r["fell"] == "False"]
            shift[f"{run}@{slot}"] = {"det_cop_mean": mean(f(r["foot_cop_mean"]) for r in det),
                                      "noise_survivor_cop_mean": mean(f(r["foot_cop_mean"]) for r in sv),
                                      "det_com_mtp3_m": mean(f(r["com_mtp3_m"]) for r in det),
                                      "noise_survivor_com_mtp3_m": mean(f(r["com_mtp3_m"]) for r in sv),
                                      "noise_survivor_cop_std_m": mean(f(r["cop_std_max_m"]) for r in sv),
                                      "noise_survivor_gate_excess": mean(f(r["gate_excess_max"]) for r in sv)}
    q1["det_to_noise_shift"] = shift
    S["q1_what_falls_look_like"] = q1

    # ---------------------------------------------------------------- Q2 actuator groups (what files can say)
    q2 = {"own_sigma_groups": {}, "control_over_own": {}, "control_sigma_groups": {}, "common_noise_sigma_groups": None}
    for run in RUNS:
        for slot in KEY_SLOTS:
            s = ex["runs"][run]["slots"][slot]
            q2["own_sigma_groups"][f"{run}@{slot}"] = {k: round(v, 4) for k, v in s["own_sigma_groups"].items()}
            q2["control_over_own"][f"{run}@{slot}"] = {k: round(v, 2) for k, v in s["control_over_own_groups"].items()}
    fp = json.load(open(f"{SP}/symmetry/design/frozen_params.json"))["criteria"]["H4"]["control_sigma"]
    G = {"neck_head": [0, 1, 2], "hip_pitch": [3, 7], "hip_roll": [4, 8], "knee": [5, 9], "ankle": [6, 10], "tail": [11, 12, 13, 14]}
    for sd in ("42", "52", "60"):
        v = fp[sd]["vector"]
        q2["control_sigma_groups"][sd] = {g: round(sum(v[i] for i in idx) / len(idx), 4) for g, idx in G.items()}
    cn = json.load(open(f"{SP}/stance5/judged/stance5_LTSB_s42/judge_common_noise_final.json"))["common_noise"]
    q2["common_noise_sigma_groups"] = {g: round(sum(cn["sigma_vector"][i] for i in idx) / len(idx), 4) for g, idx in G.items()}
    q2["extras_final"] = {}
    for run in RUNS:
        sc = run.split("_")[0]
        c = json.load(open(f"{SP}/{sc}/judged/{run}/judge_common_noise_final.json"))["common_noise"]
        t = json.load(open(f"{SP}/{sc}/judged/{run}/judge_tail_final.json"))["tail"]
        fr = json.load(open(f"{SP}/{sc}/judged/{run}/judge_fresh_final.json"))["panel"]
        cf = [e for e in c["episodes_detail"] if not e["reached_horizon"]]
        tf = [e for e in t["episodes_detail"] if not e["reached_horizon"]]
        q2["extras_final"][run] = {
            "common_noise_falls_of_40": len(cf), "common_noise_causes": dict(collections.Counter(e["termination_reason"] for e in cf)),
            "tail_seed_falls_of_32": len(tf), "tail_seed_causes": dict(collections.Counter(e["termination_reason"] for e in tf)),
            "tail_seed_statue_falls": t.get("statue_falls"),
            "fresh_panel_falls_of_40": fr["summary"]["falls"], "fresh_panel_terminations": fr["terminations"]}
    # one-sided joint noise on a command pinned at +1 (arithmetic, not measured)
    sig_t3 = fp["42"]["vector"][14]
    q2["pinned_tail3_joint_noise"] = {"sigma": round(sig_t3, 4), "mean_shift_when_pinned": round(-sig_t3 / math.sqrt(2 * math.pi), 4),
                                      "joint_shift_rad_via_satdiag_map": round(-0.086 * sig_t3 / math.sqrt(2 * math.pi), 4),
                                      "note": "joint space clips the command first, so a tail_3 command pinned at +1 only receives downward kicks (half of them): mean shift -sigma/sqrt(2 pi); satdiag's map q = -0.093 + 0.086 a makes that about -0.007 rad"}
    S["q2_actuator_groups"] = q2

    # ---------------------------------------------------------------- Q3 band vs control under the same draws
    q3 = {"falls": {}, "paired": ex["paired_joint_falls"], "survivor_margins": {}, "det_split": {}, "recovery": {}}
    for sd in ("52", "60", "42"):
        b, c = ex["runs"][f"stance5_LTSB_s{sd}"], ex["runs"][f"stance4_LTS_s{sd}"]
        q3["falls"][f"s{sd}"] = {slot: {"band": b["slots"][slot]["joint"]["falls"], "control": c["slots"][slot]["joint"]["falls"]}
                                 for slot in KEY_SLOTS}
    tb = sum(v["band"] for s in q3["falls"].values() for v in s.values())
    tc = sum(v["control"] for s in q3["falls"].values() for v in s.values())
    q3["falls_total_key_slots"] = {"band": tb, "control": tc, "episodes_each": 300,
                                   "band_excl_s42": sum(v["band"] for k, s in q3["falls"].items() if k != "s42" for v in s.values()),
                                   "control_excl_s42": sum(v["control"] for k, s in q3["falls"].items() if k != "s42" for v in s.values())}
    for run in RUNS:
        for slot in KEY_SLOTS:
            sv = [r for r in by[(run, slot, "joint")] if r["fell"] == "False"]
            det = [r for r in by[(run, slot, "panel")] if r["fell"] == "False"]
            q3["survivor_margins"][f"{run}@{slot}"] = {
                "noise_toe_margin_min_nm_median": med(f(r["toe_margin_min_nm"]) for r in sv),
                "noise_heel_margin_min_nm_median": med(f(r["heel_margin_min_nm"]) for r in sv),
                "noise_cop_mean": mean(f(r["foot_cop_mean"]) for r in sv),
                "noise_cop_std_m": mean(f(r["cop_std_max_m"]) for r in sv),
                "det_toe_margin_min_nm_mean": mean(f(r["toe_margin_min_nm"]) for r in det),
                "det_heel_margin_min_nm_mean": mean(f(r["heel_margin_min_nm"]) for r in det)}
            p = ex["runs"][run]["slots"][slot]["panel"]
            q3["det_split"][f"{run}@{slot}"] = {"per_foot_cop_max": p["full_horizon"]["foot_cop_max"],
                                                "per_foot_cop_min": p["full_horizon"]["foot_cop_min"],
                                                "split": round(p["full_horizon"]["foot_cop_max"] - p["full_horizon"]["foot_cop_min"], 4),
                                                "xc_median_m": p["xc_median_m"], "com_mtp3_median_m": p["com_mtp3_median_m"]}
        sep_run = S["q1_what_falls_look_like"]["tilt_lift_separation_key_slots"][run]
        q3["recovery"][run] = {"falls": sep_run["falls"], "survived_roll_rail_excursions": sep_run["survivors_over_roll_rail"],
                               "survived_tilt_over_3p5deg": sep_run["survivors_tilt_over_3p5"]}
    q3["ankle_torque_headroom"] = "not in the judge files: the noise rows hold no actuator forces; the eval ledger's actuator_force_frac_max (deterministic, max over 15 actuators, held by head_pitch or tail_1 per satdiag) reads 0.27-0.36 in all runs"
    q3["band_det_readouts"] = {f"{run}@{slot}": ex["runs"][run]["slots"][slot]["panel"].get("band")
                               for run in RUNS[:3] for slot in J_SLOTS}
    q3["band_det_readouts_statue"] = ex["runs"]["stance5_LTSB_s42"]["slots"]["final"]["panel"]["statue"].get("band")
    S["q3_band_vs_control"] = q3

    # ---------------------------------------------------------------- Q4 s42's rise: tail habit or something else
    sat_t3 = {}
    for slot in ("4194304", "5242880", "6291456"):
        d = json.load(open(f"{SP}/stance5/tools/satroll/stance5_LTSB_s42_{slot}.json"))["checkpoints"][slot]
        sat_t3[slot] = {"frac90": round(d["named"]["frac90"], 4), "frac99": round(d["named"]["frac99"], 4),
                        "mean_command": round(d["named"]["mean_command"], 4), "rolled_flagged_of_20": d["flagged_over_bar"]}
    s4tab = {}
    for line in open(f"{SP}/stance4/satdiag/rollout/tables/checkpoint_table.txt"):
        m = re.match(r"stance4_LTS_s42\s+(\S+) flagged\s+(\d+)/20 \| top by >=0.9: (\S+) ([0-9.]+)/([0-9.]+) .*\| tail_3 mean ([+-][0-9.]+)", line)
        if m:
            s4tab[m.group(1)] = {"flagged_of_20": int(m.group(2)), "top": m.group(3), "frac90": float(m.group(4)),
                                 "frac99": float(m.group(5)), "tail3_mean_command": float(m.group(6))}
    trend = {}
    for run in ("stance5_LTSB_s42", "stance4_LTS_s42", "stance5_LTSB_s52", "stance5_LTSB_s60"):
        R = ex["runs"][run]
        bins = {b["bin_end"]: b for b in R["ledger_bins"]}
        trend[run] = {}
        for slot in ["4194304", "5242880", "6291456", "7340032", "8388608", "9437184", "10485760", "11534336", "final"]:
            s = R["slots"].get(slot)
            if not s or "joint" not in s:
                continue
            ts = 12001280 if slot == "final" else int(slot)
            b = min(bins.values(), key=lambda x: abs(x["bin_end"] - ts))
            fr = [r for r in allfalls if r["run"] == run and r["slot"] == slot]
            dd = collections.Counter(direction(r)[0] for r in fr)
            p = s["panel"]
            trend[run][slot] = {
                "joint_falls": s["joint"]["falls"], "joint_direction": dict(dd), "own_falls": s["own"]["falls"],
                "panel_sat_flagged_of_40": p["sat_flagged"], "panel_sat_fraction_flagged_mean": p["sat_fraction_flagged_mean"],
                "sat_vs_fall_same_spawn": s.get("sat_vs_joint_fall"),
                "policy_std_mean": round(s["own"]["policy_std_mean"], 4),
                "own_sigma_legs_mean": round(statistics.fmean([s["own_sigma"][a] for a in ("r_hip_pitch", "l_hip_pitch", "r_knee", "l_knee", "r_ankle", "l_ankle")]), 4),
                "own_sigma_tail3": round(s["own_sigma"]["tail_3_pitch"], 4),
                "det_com_mtp3_median_m": p["com_mtp3_median_m"], "det_xc_median_m": p["xc_median_m"],
                "det_cop_mean": p["full_horizon"]["foot_cop_mean"], "det_toe_margin_min_nm": p["full_horizon"]["toe_margin_min_nm"],
                "ledger_bin_end": b["bin_end"], "eval_com_mm": round(1000 * b["eval_com_mean_m"], 1) if b["eval_com_mean_m"] is not None else None,
                "eval_pitch_deg": b["eval_pitch_mean_deg"], "eval_hip_q_deg": b["eval_hip_pitch_q_deg"],
                "eval_knee_q_deg": b["eval_knee_q_deg"], "eval_ankle_q_deg": b["eval_ankle_q_deg"],
                "eval_sat_flagged_of_40": b["eval_sat_flagged"], "train_fall_rate": round(b["train_fall_rate"], 4),
                "train_causes": b["train_causes"]}
            if run == "stance5_LTSB_s42" and slot in sat_t3:
                trend[run][slot]["satroll_tail3"] = sat_t3[slot]
            if run == "stance4_LTS_s42" and slot in s4tab:
                trend[run][slot]["satdiag_tail3"] = s4tab[slot]
    q4 = {"trend": trend, "s42_band_fall_spawn_recurrence_7p3M_to_final": ex["s42_band_fall_spawn_recurrence"],
          "train_ledger_falls_per_524288_bin": {run: [{"bin_end": b["bin_end"], "rate": round(b["train_fall_rate"], 4),
                                                         "causes": b["train_causes"]} for b in ex["runs"][run]["ledger_bins"]
                                                        if b["bin_end"] >= 6291456] for run in RUNS}}
    S["q4_s42_rise"] = q4
    with open(os.path.join(out, "summary_data.json"), "w") as fh:
        json.dump(S, fh, indent=1)
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
