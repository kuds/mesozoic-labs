"""EARLY-WARNING CURVE: ledger events and the probe bank's tail_3, per run (read only; no simulation).

    python -B ew_events.py > ew_events.txt   (writes ew_events.json)

Ledger (eval_episodes.csv: 10 deterministic episodes per evaluation, every 131,072 steps):
  over the BAR      = saturated_actuator_fraction > 0.10 (share of window steps at |command| >= 0.99 on the most
                      saturated actuator; CRIT6 line 166);  TAIL over-bar = that actuator is a tail actuator
                      (tail_1_pitch, tail_1_yaw, tail_2_pitch, tail_3_pitch; indices 11-14).
  ANY / TAIL FLAG   = >= 5 of the evaluation's 10 episodes over the bar (TAIL: with a tail actuator named).
  ANY / TAIL ALARM  = the second of two consecutive flagged evaluations.
  CMD_L             = mean over the 10 episodes of tail_3_pitch_act_command_mean (window-mean clipped command).
  RAIL (ledger)     = first evaluation with CMD_L >= 0.90;  PARK (ledger) = first with CMD_L >= 0.60.
  CMD_L >= 0.30     = first evaluation with CMD_L >= 0.30 (a ledger-only early marker, for comparison).
Probe (policy_probe.npz, one row per PPO update, 8,192 steps): bank_action_now[:, :, 14] (the deterministic
action on the fixed 10-state bank under the current normalisation), mean over the bank and over its two
statue_settled rows; rollout_mu_mean[:, 14] (the training rows' mean Gaussian mean).
ECS ledgers hold the parent run's evaluations up to the start S; events are reported overall and after S.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import ew_common as C

TAIL_ACTS = {"tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"}
START = {"s42": 3_145_728, "s80": 4_194_304}


def ledger_path(run):
    return (C.r10_dir(run) / "eval_episodes.csv.copy1") if run in C.R10_RUNS else (C.ecs_dir(run) / "eval_episodes.csv")


def probe_path(run):
    return (C.r10_dir(run) / "policy_probe.npz.copy1") if run in C.R10_RUNS else (C.ecs_dir(run) / "policy_probe.npz")


def per_eval(run):
    d = pd.read_csv(ledger_path(run)).copy()
    d = d[["timesteps", "saturated_actuator_fraction", "saturated_actuator", "tail_3_pitch_act_command_mean"]].copy()
    d["over"] = d["saturated_actuator_fraction"] > 0.10
    d["tail_over"] = d["over"] & d["saturated_actuator"].isin(TAIL_ACTS)
    d["t3_over"] = d["over"] & (d["saturated_actuator"] == "tail_3_pitch_act")
    g = d.groupby("timesteps")
    e = pd.DataFrame({
        "n": g.size(), "n_over": g["over"].sum(), "n_tail_over": g["tail_over"].sum(), "n_t3_over": g["t3_over"].sum(),
        "cmd_l": g["tail_3_pitch_act_command_mean"].mean(), "cmd_l_max": g["tail_3_pitch_act_command_mean"].max(),
        "acts": g["saturated_actuator"].agg(lambda s: ",".join(sorted(set(str(x) for x in s if isinstance(x, str))))),
    }).reset_index()
    e["any_flag"] = e["n_over"] >= 5
    e["tail_flag"] = e["n_tail_over"] >= 5
    e["any_alarm"] = e["any_flag"] & e["any_flag"].shift(1, fill_value=False)
    e["tail_alarm"] = e["tail_flag"] & e["tail_flag"].shift(1, fill_value=False)
    return e


def first(e, col, after=0):
    s = e[(e[col]) & (e["timesteps"] > after)]
    return int(s["timesteps"].iloc[0]) if len(s) else None


def events(run):
    e = per_eval(run)
    after = START.get(run[:3], 0) if run not in C.R10_RUNS else 0
    out = {"n_evals": int(len(e)), "last_eval": int(e["timesteps"].max()), "after": after}
    e["over_any_ep"] = e["n_over"] > 0
    e["over_tail_ep"] = e["n_tail_over"] > 0
    e["park"] = e["cmd_l"] >= 0.60
    e["rail"] = e["cmd_l"] >= 0.90
    e["cmd30"] = e["cmd_l"] >= 0.30
    for k in ("over_any_ep", "over_tail_ep", "any_flag", "tail_flag", "any_alarm", "tail_alarm", "park", "rail", "cmd30"):
        out[f"first_{k}"] = first(e, k, after)
        out[f"first_{k}_overall"] = first(e, k, 0)
    out["n_any_flag"] = int(e["any_flag"].sum())
    out["n_tail_flag"] = int(e["tail_flag"].sum())
    out["n_over_eps"] = int(e["n_over"].sum())
    out["n_tail_over_eps"] = int(e["n_tail_over"].sum())
    out["over_acts"] = sorted({a for s in e.loc[e["n_over"] > 0, "acts"] for a in s.split(",") if a})
    out["cmd_l_max_eval"] = float(e["cmd_l"].max())
    out["series"] = e[["timesteps", "n_over", "n_tail_over", "cmd_l", "any_flag", "tail_flag"]].to_dict("list")
    return out


def probe(run):
    with np.load(probe_path(run), allow_pickle=False) as d:
        t = d["update_timesteps"]
        a = d["bank_action_now"][:, :, C.A_T3]
        lab = d["bank_labels"]
        r = d["rollout_mu_mean"][:, C.A_T3]
        beyond = d["rollout_mu_beyond_high_share"][:, C.A_T3]
    ss = lab == "statue_settled"
    return {"t": t.tolist(), "bank_mean": a.mean(axis=1).tolist(), "bank_ss": a[:, ss].mean(axis=1).tolist(),
            "bank_rows": a.tolist(), "rollout_mu": r.tolist(), "mu_beyond_high": beyond.tolist()}


def main() -> int:
    res = {}
    for run in list(C.R10_RUNS) + C.ECS_RUNS:
        res[run] = {"events": events(run), "probe": probe(run)}
    with open(C.HERE / "ew_events.json", "w") as fh:
        json.dump(res, fh)
    keys = ["first_over_any_ep", "first_over_tail_ep", "first_any_flag", "first_tail_flag", "first_any_alarm",
            "first_tail_alarm", "first_cmd30", "first_park", "first_rail"]
    print("Ledger events per run (steps; '-' = never; ECS: after the start S unless marked '*' = at or before S)")
    print("run          " + " ".join(f"{k[6:]:>13s}" for k in keys) + "  nAnyF nTailF  maxCMD_L  over-bar actuators")
    for run, v in res.items():
        ev = v["events"]
        cells = []
        for k in keys:
            x = ev[k]
            if x is None and ev[k + "_overall"] is not None:
                cells.append(f"{ev[k + '_overall']:>12d}*")
            else:
                cells.append(f"{x:>13d}" if x is not None else f"{'-':>13s}")
        print(f"{run:12s} " + " ".join(cells) + f"  {ev['n_any_flag']:5d} {ev['n_tail_flag']:6d}  {ev['cmd_l_max_eval']:+.3f}  {','.join(ev['over_acts'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
