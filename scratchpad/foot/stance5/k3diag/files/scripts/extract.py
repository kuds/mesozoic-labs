"""K3 diagnosis from files only (report only): mine the stance5 (band) and stance4 (control) judge files and ledgers.

Reads JSON/CSV only; loads no checkpoint and imports nothing from the repo.  Run:
    python3 -I extract.py OUT_DIR
Writes OUT_DIR/noise_episodes.csv (one row per noise-test episode), OUT_DIR/ledger_bins.csv (train/eval ledger per
524,288-step bin) and OUT_DIR/extract.json (everything the summary is built from).
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
RUNS = [("stance5", "stance5_LTSB_s52", "band", 52), ("stance5", "stance5_LTSB_s60", "band", 60),
        ("stance5", "stance5_LTSB_s42", "band", 42), ("stance4", "stance4_LTS_s52", "control", 52),
        ("stance4", "stance4_LTS_s60", "control", 60), ("stance4", "stance4_LTS_s42", "control", 42)]
SLOTS = ["1048576", "2097152", "3145728", "4194304", "5242880", "6291456", "7340032", "8388608", "9437184",
         "10485760", "11534336", "final"]
STEPS = {s: (12001280 if s == "final" else int(s)) for s in SLOTS}
ACT = ["neck_pitch", "neck_yaw", "head_pitch", "r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "l_hip_pitch",
       "l_hip_roll", "l_knee", "l_ankle", "tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch"]
GROUPS = {"neck_head": [0, 1, 2], "hip_pitch": [3, 7], "hip_roll": [4, 8], "knee": [5, 9], "ankle": [6, 10],
          "tail": [11, 12, 13, 14]}
FEET = ("r", "l")
EP_KEYS = ["foot_cop_fore_aft", "foot_cop_fore_aft_gate", "sole_cop_fore_aft", "sole_cop_fore_aft_gate", "digit_share",
           "toe_margin_nm", "heel_margin_nm", "foot_cop_std_m"]
SAT_RE = re.compile(r"max_actuator_saturation_fraction ([0-9.]+) >")


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.fmean(xs) if xs else None


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else None


def rnd(x, n=4):
    if isinstance(x, float):
        return round(x, n)
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [rnd(v, n) for v in x]
    return x


def ep_row(e):
    pad = e.get("pad") or {}
    r = {"seed": e["seed"], "length": e["length"], "fell": not e["reached_horizon"],
         "cause": e.get("termination_reason") or "truncated", "reward": e.get("reward"),
         "max_sole_cop": e.get("max_sole_cop_fore_aft"), "max_foot_cop": e.get("max_foot_cop_fore_aft"),
         "max_tilt_deg": e.get("max_sole_tilt_deg"), "max_lift_m": e.get("max_sole_corner_lift_m"),
         "xc_m": e.get("xc_both_feet_m"), "com_mtp3_m": e.get("com_mtp3_head_m"),
         "com_site_m": e.get("com_fore_aft_footsite_m"), "k2a": e.get("k2a_over"), "k2b": e.get("k2b_outside")}
    for f in FEET:
        for k in EP_KEYS:
            r[f"{f}_{k}"] = pad.get(f"{f}_{k}")
    # derived: the feet's mean and max signed whole-foot CoP, the smaller toe and heel moment margin
    fc = [r[f"{f}_foot_cop_fore_aft"] for f in FEET if r[f"{f}_foot_cop_fore_aft"] is not None]
    r["foot_cop_mean"] = mean(fc)
    r["foot_cop_max"] = max(fc) if fc else None
    r["foot_cop_min"] = min(fc) if fc else None
    r["foot_cop_lr_diff"] = (r["l_foot_cop_fore_aft"] - r["r_foot_cop_fore_aft"]) if len(fc) == 2 else None
    tm = [r[f"{f}_toe_margin_nm"] for f in FEET if r[f"{f}_toe_margin_nm"] is not None]
    hm = [r[f"{f}_heel_margin_nm"] for f in FEET if r[f"{f}_heel_margin_nm"] is not None]
    r["toe_margin_min_nm"] = min(tm) if tm else None
    r["heel_margin_min_nm"] = min(hm) if hm else None
    r["toe_margin_mean_nm"] = mean(tm)
    r["heel_margin_mean_nm"] = mean(hm)
    # unloaded-or-rocking excess: gate minus |signed| (gate counts an unloaded window step as 1)
    gx = [r[f"{f}_foot_cop_fore_aft_gate"] - abs(r[f"{f}_foot_cop_fore_aft"]) for f in FEET
          if r[f"{f}_foot_cop_fore_aft_gate"] is not None and r[f"{f}_foot_cop_fore_aft"] is not None]
    r["gate_excess_max"] = max(gx) if gx else None
    r["cop_std_max_m"] = max([r[f"{f}_foot_cop_std_m"] for f in FEET if r[f"{f}_foot_cop_std_m"] is not None] or [None]) \
        if any(r[f"{f}_foot_cop_std_m"] is not None for f in FEET) else None
    return r


def group_stats(rows):
    """Means over a set of episode rows of the readouts used in the answers."""
    keys = ["length", "foot_cop_mean", "foot_cop_max", "foot_cop_min", "foot_cop_lr_diff", "toe_margin_min_nm",
            "heel_margin_min_nm", "toe_margin_mean_nm", "heel_margin_mean_nm", "max_sole_cop", "max_foot_cop",
            "max_tilt_deg", "max_lift_m", "xc_m", "com_mtp3_m", "com_site_m", "gate_excess_max", "cop_std_max_m",
            "r_digit_share", "l_digit_share"]
    out = {"n": len(rows)}
    for k in keys:
        out[k] = mean(r[k] for r in rows)
    return out


def noise_block(n):
    rows = [ep_row(e) for e in n["episodes_detail"]]
    falls = [r for r in rows if r["fell"]]
    surv = [r for r in rows if not r["fell"]]
    causes = collections.Counter(r["cause"] for r in falls)
    blk = {"falls": len(falls), "fall_seeds": [r["seed"] for r in falls], "fall_lengths": [r["length"] for r in falls],
           "causes": dict(causes), "fall_length_median": med([r["length"] for r in falls]),
           "k2a_count": n.get("k2a_count"), "k2b_count": n.get("k2b_count"),
           # survivors whose pad CoP gate exceeded 0.8 (an excursion survived) / whole-foot gate over 0.8
           "survivors_pad_cop_gt_0p8": n.get("cop_gt_bar_gate_full_horizon"),
           "survivors_foot_cop_gt_0p8": n.get("foot_cop_gt_bar_full_horizon"),
           "survivors_k2b_outside": sum(1 for r in surv if r["k2b"]),
           "roll_rail_full_horizon": n.get("roll_rail_full_horizon"),
           "sigma_mean": n.get("sigma_mean"), "policy_std_mean": n.get("policy_std_mean"),
           "fallers": group_stats(falls), "survivors": group_stats(surv), "all": group_stats(rows),
           "reward_mean_full": n.get("reward_mean_full"), "xc_median_full_horizon_m": n.get("xc_median_full_horizon_m"),
           "com_mtp3_head_median_full_horizon_m": n.get("com_mtp3_head_median_full_horizon_m")}
    return blk, rows


def panel_block(p):
    rows = [ep_row(e) for e in p["episodes_detail"]]
    st = [ep_row(e) for e in p["statue_episodes_detail"]]
    sat = {}
    for e in p["episodes_detail"]:
        v = None
        for reason in e.get("reasons") or []:
            m = SAT_RE.search(reason)
            if m:
                v = float(m.group(1))
        sat[e["seed"]] = v
    s, ss = p["summary"], p["statue_on_same_seeds"]
    full = [r for r in rows if not r["fell"]]
    stfull = [r for r in st if not r["fell"]]
    blk = {"falls": s["falls"], "fall_seeds": s["fall_seeds"], "fall_lengths": s["fall_lengths"],
           "terminations": p.get("terminations"), "n_clean": p.get("n_clean"),
           "sat_flagged": p["criterion_failures"].get("max_actuator_saturation_fraction"),
           "sat_fraction_flagged_mean": mean(v for v in sat.values() if v is not None),
           "sat_fraction_by_seed": {str(k): v for k, v in sat.items()},
           "mean_reward": p.get("mean_reward"), "statue_mean_reward": p.get("statue_mean_reward"),
           "xc_median_m": s["xc_median_full_horizon_m"], "com_mtp3_median_m": s["com_mtp3_head_median_full_horizon_m"],
           "com_site_median_m": s["com_fore_aft_footsite_median_full_horizon_m"],
           "per_foot_signed_cop_median": {f: s["per_foot"][f]["foot_signed_cop_median"] for f in FEET},
           "per_foot_pad_cop_median": {f: s["per_foot"][f]["signed_cop_median"] for f in FEET},
           "per_foot_signed_cop_min": {f: s["per_foot"][f]["foot_signed_cop_min"] for f in FEET},
           "per_foot_signed_cop_max": {f: s["per_foot"][f]["foot_signed_cop_max"] for f in FEET},
           "full_horizon": group_stats(full),
           "statue": {"falls": ss["falls"], "xc_median_m": ss["xc_median_full_horizon_m"],
                      "com_mtp3_median_m": ss["com_mtp3_head_median_full_horizon_m"],
                      "com_site_median_m": ss["com_fore_aft_footsite_median_full_horizon_m"],
                      "per_foot_signed_cop_median": {f: ss["per_foot"][f]["foot_signed_cop_median"] for f in FEET},
                      "full_horizon": group_stats(stfull)}}
    if p.get("band_summary"):
        b = p["band_summary"]
        blk["band"] = {"whole_out": b["full_horizon"]["whole_out"], "pad_out": b["full_horizon"]["pad_out"],
                       "window_foot_steps": b["full_horizon"]["window_foot_steps"],
                       "window_steps_paying": b["full_horizon"]["window_steps_paying"],
                       "paid_mean": b["paid_per_full_horizon_episode"]["mean"] if b.get("paid_per_full_horizon_episode") else None,
                       "share_min": b["full_horizon"]["share_min"]}
        sb = ss.get("band_summary")
        if sb:
            blk["statue"]["band"] = {"whole_out": sb["full_horizon"]["whole_out"], "pad_out": sb["full_horizon"]["pad_out"],
                                     "window_foot_steps": sb["full_horizon"]["window_foot_steps"],
                                     "paid_mean": sb["paid_per_full_horizon_episode"]["mean"]}
    return blk, rows


def sigma_groups(vec):
    return {g: mean(vec[i] for i in idx) for g, idx in GROUPS.items()}


def ledger_bins(path_train, path_eval, binw=524288):
    tr = list(csv.DictReader(open(path_train)))
    ev = list(csv.DictReader(open(path_eval)))
    bins = collections.OrderedDict()
    for x in tr:
        b = (int(x["timesteps"]) - 1) // binw
        d = bins.setdefault(b, {"train_episodes": 0, "train_falls": 0, "train_causes": collections.Counter(),
                                "train_fall_com_last": [], "train_len_falls": [], "eval_episodes": 0, "eval_falls": 0,
                                "eval_causes": collections.Counter(), "eval_com": [], "eval_pitch": [], "eval_knee": [],
                                "eval_ankle": [], "eval_hip": [], "eval_toe_margin": [], "eval_xcom_toe_min": [],
                                "eval_ankle_arm": [], "eval_force_frac_max": [], "eval_sat_flag": 0})
        d["train_episodes"] += 1
        if x["cause"] != "truncated":
            d["train_falls"] += 1
            d["train_causes"][x["cause"]] += 1
            d["train_len_falls"].append(int(x["length"]))
            try:
                d["train_fall_com_last"].append(float(x["com_fore_aft_last_m"]))
            except ValueError:
                pass
    for x in ev:
        b = (int(x["timesteps"]) - 1) // binw
        d = bins.get(b)
        if d is None:
            continue
        d["eval_episodes"] += 1
        if x["full_horizon"] != "1":
            d["eval_falls"] += 1
            d["eval_causes"][x["cause"]] += 1

        def f(k):
            try:
                v = float(x.get(k, "nan"))
                return None if math.isnan(v) else v
            except ValueError:
                return None
        d["eval_com"].append(f("com_fore_aft_mean_m"))
        d["eval_pitch"].append(f("root_pitch_mean_deg"))
        d["eval_knee"].append(f("leg_q_l_r_mean_knee_deg"))
        d["eval_ankle"].append(f("leg_q_l_r_mean_ankle_deg"))
        d["eval_hip"].append(f("leg_q_l_r_mean_hip_pitch_deg"))
        d["eval_toe_margin"].append(mean([f("r_toe_margin_nm"), f("l_toe_margin_nm")]))
        d["eval_xcom_toe_min"].append(f("xcom_toe_margin_min_m"))
        d["eval_ankle_arm"].append(mean([f("r_ankle_moment_arm_m"), f("l_ankle_moment_arm_m")]))
        d["eval_force_frac_max"].append(f("actuator_force_frac_max"))
        if "max_actuator_saturation_fraction" in (x.get("screen_v2_reasons") or ""):
            d["eval_sat_flag"] += 1
    out = []
    for b, d in bins.items():
        out.append({"bin_start": b * binw, "bin_end": (b + 1) * binw, "train_episodes": d["train_episodes"],
                    "train_falls": d["train_falls"],
                    "train_fall_rate": d["train_falls"] / d["train_episodes"] if d["train_episodes"] else None,
                    "train_causes": dict(d["train_causes"]),
                    "train_fall_com_last_mean_m": mean(d["train_fall_com_last"]),
                    "eval_episodes": d["eval_episodes"], "eval_falls": d["eval_falls"], "eval_causes": dict(d["eval_causes"]),
                    "eval_sat_flagged": d["eval_sat_flag"],
                    "eval_com_mean_m": mean(d["eval_com"]), "eval_pitch_mean_deg": mean(d["eval_pitch"]),
                    "eval_hip_pitch_q_deg": mean(d["eval_hip"]), "eval_knee_q_deg": mean(d["eval_knee"]),
                    "eval_ankle_q_deg": mean(d["eval_ankle"]), "eval_toe_margin_nm": mean(d["eval_toe_margin"]),
                    "eval_xcom_toe_margin_min_m": mean(d["eval_xcom_toe_min"]),
                    "eval_ankle_moment_arm_m": mean(d["eval_ankle_arm"]),
                    "eval_force_frac_max": mean(d["eval_force_frac_max"])})
    return out


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    result = {"runs": {}}
    ep_csv = []
    for screen, run, arm, seed in RUNS:
        R = {"screen": screen, "arm": "LTSB" if arm == "band" else "LTS", "role": arm, "seed": seed, "slots": {},
             "extras": {}}
        for slot in SLOTS:
            p = f"{SP}/{screen}/judged/{run}/judge_{slot}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            S = {"timesteps": STEPS[slot]}
            if d.get("noise_matched_control_joint"):
                S["joint"], jrows = noise_block(d["noise_matched_control_joint"])
                S["own"], orows = noise_block(d["noise_matched_own"])
                own_vec = d["noise_matched_own"]["sigma_vector"]
                ctrl_vec = d["noise_matched_control_joint"]["sigma_vector"]
                S["own_sigma"] = dict(zip(ACT, own_vec))
                S["own_sigma_groups"] = sigma_groups(own_vec)
                S["control_over_own_groups"] = {g: mean(ctrl_vec[i] / own_vec[i] for i in idx) for g, idx in GROUPS.items()}
                for test, rows in (("joint", jrows), ("own", orows)):
                    for r in rows:
                        ep_csv.append({"run": run, "role": arm, "seed": seed, "slot": slot, "test": test, **r})
            S["panel"], prow = panel_block(d["panel"])
            for r in prow:
                ep_csv.append({"run": run, "role": arm, "seed": seed, "slot": slot, "test": "panel", **r})
            # same spawn, deterministic saturation vs joint-noise fall
            if "joint" in S:
                sat = S["panel"]["sat_fraction_by_seed"]
                fl = set(S["joint"]["fall_seeds"])
                flagged = [s for s in range(3042, 3062) if sat.get(str(s)) is not None]
                unflag = [s for s in range(3042, 3062) if sat.get(str(s)) is None]
                S["sat_vs_joint_fall"] = {"flagged_spawns": len(flagged), "falls_in_flagged": sum(1 for s in flagged if s in fl),
                                          "unflagged_spawns": len(unflag), "falls_in_unflagged": sum(1 for s in unflag if s in fl),
                                          "panel_det_fall_seeds_in_3042_3061": [s for s in S["panel"]["fall_seeds"] if s <= 3061]}
            R["slots"][slot] = S
        for extra in ("fresh_final", "tail_final", "common_noise_final"):
            p = f"{SP}/{screen}/judged/{run}/judge_{extra}.json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            if extra == "fresh_final":
                pb, _ = panel_block(d["panel"])
                R["extras"][extra] = {k: pb[k] for k in ("falls", "fall_seeds", "fall_lengths", "terminations", "sat_flagged",
                                                         "xc_median_m", "com_mtp3_median_m", "per_foot_signed_cop_median")}
            else:
                blk = d["tail"] if extra == "tail_final" else d["common_noise"]
                rows = blk.get("episodes_detail") or blk.get("rows") or blk.get("policy") or []
                R["extras"][extra] = {k: blk.get(k) for k in ("n", "falls", "fall_seeds", "sigma_mean", "reference")
                                      if k in blk}
                R["extras"][extra]["keys"] = sorted(blk.keys())
        tr = f"{SP}/{screen}/runs/{run}/01_stance/train_episodes.csv"
        ev = f"{SP}/{screen}/runs/{run}/01_stance/eval_episodes.csv"
        R["ledger_bins"] = ledger_bins(tr, ev)
        result["runs"][run] = R
    # paired spawns: band vs control of the same seed (same sigma vector, same z draws: default_rng(episode seed))
    pairs = {}
    for seed in (42, 52, 60):
        b, c = result["runs"][f"stance5_LTSB_s{seed}"], result["runs"][f"stance4_LTS_s{seed}"]
        for slot in ("6291456", "8388608", "10485760", "final"):
            if slot in b["slots"] and slot in c["slots"]:
                fb, fc = set(b["slots"][slot]["joint"]["fall_seeds"]), set(c["slots"][slot]["joint"]["fall_seeds"])
                pairs[f"s{seed}_{slot}"] = {"band_only": sorted(fb - fc), "control_only": sorted(fc - fb), "both": sorted(fb & fc)}
    result["paired_joint_falls"] = pairs
    # recurrence of fall spawns across s42 band checkpoints
    rec = collections.Counter()
    for slot in ("7340032", "8388608", "10485760", "final"):
        for s in result["runs"]["stance5_LTSB_s42"]["slots"][slot]["joint"]["fall_seeds"]:
            rec[s] += 1
    result["s42_band_fall_spawn_recurrence"] = dict(sorted(rec.items()))
    with open(os.path.join(out_dir, "extract.json"), "w") as fh:
        json.dump(rnd(result, 5), fh, indent=1)
    cols = list(ep_csv[0].keys())
    with open(os.path.join(out_dir, "noise_episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in ep_csv:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(out_dir, "ledger_bins.csv"), "w", newline="") as fh:
        w = None
        for run, R in result["runs"].items():
            for b in R["ledger_bins"]:
                row = {"run": run, **{k: (json.dumps(v) if isinstance(v, dict) else (round(v, 5) if isinstance(v, float) else v))
                                      for k, v in b.items()}}
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    w.writeheader()
                w.writerow(row)
    print("wrote", out_dir, len(ep_csv), "episode rows")


if __name__ == "__main__":
    main(sys.argv[1])
