"""TRACK TAIL (scratch, report only): collate measure.py / balance.py / training telemetry into results.csv.

    python3 collect.py            (plain python3, numpy only; reads JSON / npz / csv written by the experiment)

Writes tailexp/results.csv (one row per measured checkpoint: start, control, p1, and the original run's own 5,242,880 and
6,291,456 as references) and tailexp/train_telemetry.csv (one row per training rollout of each arm).
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

import numpy as np

T = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance5/tailexp"
TAIL3 = 14
STEPS = ["4325376", "4456448", "4587520", "4718592", "4849664", "4980736", "5111808", "5242880", "final"]
WINDOW = 131072


def load_measure(label: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(T, "measure", f"{label}__summary_*.json")):
        d = json.load(open(f))
        if d.get("label") != label:
            continue
        for st, c in d["checkpoints"].items():
            if c.get("present", True):
                out[st] = c
    return out


def load_balance(label: str, st: str):
    p = os.path.join(T, "balance", f"{label}__{st}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def r(x, n=4):
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return round(float(x), n) if isinstance(x, float) else x


def telemetry(arm: str):
    p = os.path.join(T, arm, "telemetry.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def main() -> int:
    rows = []
    sets = [("start", "start", ["4194304"]), ("control", "control", STEPS), ("p1", "p1", STEPS),
            ("orig", "orig", ["5242880", "6291456"])]
    tele = {a: telemetry(a) for a in ("control", "p1")}
    for arm, label, steps in sets:
        meas = load_measure(label)
        for st in steps:
            c = meas.get(st)
            if c is None:
                continue
            t3 = c["tail3"]
            per = c["per_actuator"]
            others = [p for i, p in enumerate(per) if i != TAIL3]
            o90 = max(others, key=lambda p: p["frac90"])
            ob1 = max(others, key=lambda p: p["raw_beyond1"])
            oraw = max(others, key=lambda p: abs(p["mean_raw"]))
            k3 = c.get("k3") or {}
            row = {
                "arm": arm, "checkpoint": st, "num_timesteps": c["num_timesteps"],
                "t3_frac90": r(t3["frac90"]), "t3_frac99": r(t3["frac99"]), "t3_cmd_mean": r(t3["mean_command"]),
                "t3_raw_mean": r(t3["raw_mean_window"]), "t3_raw_mean_pinned": r(t3["raw_mean_on_pinned_steps"]),
                "t3_raw_max": r(t3["raw_max"]), "t3_raw_beyond1": r(t3["raw_beyond1"]), "t3_std": r(t3["policy_std"]),
                "falls": c["falls"], "reward_mean": r(c["reward_mean"], 1), "clean": c["clean"], "clean_of": c["clean_of"],
                "flagged_sat": c["flagged_over_bar"],
                "other_rails": ";".join(f"{k}:{v}" for k, v in sorted(c["failing_rails"].items())
                                        if k != "max_actuator_saturation_fraction"),
                "k3_joint_falls": (k3.get("noise_matched_control_joint") or {}).get("falls", ""),
                "k3_own_falls": (k3.get("noise_matched_own") or {}).get("falls", ""),
                "other_max_frac90": r(o90["frac90"]), "other_max_frac90_act": o90["actuator"],
                "other_max_frac99": r(max(p["frac99"] for p in others)),
                "other_max_raw_beyond1": r(ob1["raw_beyond1"]), "other_max_raw_beyond1_act": ob1["actuator"] if ob1["raw_beyond1"] > 0 else "",
                "other_max_abs_raw_mean": r(abs(oraw["mean_raw"])), "other_max_abs_raw_mean_act": oraw["actuator"],
            }
            b = load_balance(label, st)
            if b is not None:
                d, n = b["tests"]["det_sigma0_joint"], b["tests"]["noise_control_joint"]
                row.update({
                    "det_k2a": d["k2a_count"], "det_k2b": d["k2b_count"],
                    "det_xc_mm": r(1000 * d["xc_median_full_horizon_m"], 2) if d["xc_median_full_horizon_m"] is not None else "",
                    "det_com_footsite_mm": r(1000 * d["com_fore_aft_footsite_median_full_horizon_m"], 2)
                    if d["com_fore_aft_footsite_median_full_horizon_m"] is not None else "",
                    "det_gate_cop_med": r(d["gate_cop_median"]), "det_roll_rail": d["roll_rail_full_horizon"],
                    "noise_falls": n["falls"], "noise_k2a": n["k2a_count"], "noise_k2b": n["k2b_count"],
                    "noise_xc_mm": r(1000 * n["xc_median_full_horizon_m"], 2) if n["xc_median_full_horizon_m"] is not None else "",
                    "noise_com_footsite_mm": r(1000 * n["com_fore_aft_footsite_median_full_horizon_m"], 2)
                    if n["com_fore_aft_footsite_median_full_horizon_m"] is not None else "",
                    "noise_gate_cop_med": r(n["gate_cop_median"]), "noise_cop_gt_0p8": n["cop_gt_bar_gate"],
                    "noise_roll_rail": n["roll_rail_full_horizon"], "noise_reward_full": r(n["reward_mean_full"], 1),
                })
                if (d["falls"], round(d["reward_mean_full"] or 0, 6)) != (c["falls"], round(c["reward_mean"], 6)) and c["falls"] == 0:
                    row["det_crosscheck"] = f"balance sigma0 reward {d['reward_mean_full']} vs measure {c['reward_mean']}"
                if k3 and n["falls"] != row["k3_joint_falls"]:
                    raise SystemExit(f"{arm} {st}: balance joint-noise falls {n['falls']} != measure K3 {row['k3_joint_falls']}")
            tz = tele.get(arm)
            if tz is not None and st != "final":
                N = int(st)
                ts = tz["num_timesteps"]
                i = np.where(ts == N)[0]
                w = (ts > N - WINDOW) & (ts <= N)
                if i.size:
                    i = int(i[0])
                    mb = tz["mu_beyond"][i].copy()
                    row.update({"train_t3_mu_mean": r(tz["mu_mean"][i, TAIL3]), "train_t3_mu_beyond": r(tz["mu_beyond"][i, TAIL3]),
                                "train_t3_cmd_ge_0p99": r(tz["cmd_ge_0p99"][i, TAIL3]),
                                "train_t3_sample_clip_hi": r(tz["sample_at_clip_hi"][i, TAIL3])})
                    mb[TAIL3] = -1
                    row.update({"train_other_mu_beyond_max": r(mb.max()), "train_other_mu_beyond_act": int(mb.argmax())})
                if w.any():
                    row.update({"train_w16_t3_mu_mean": r(tz["mu_mean"][w, TAIL3].mean()),
                                "train_w16_t3_mu_beyond": r(tz["mu_beyond"][w, TAIL3].mean()),
                                "train_w16_t3_cmd_ge_0p99": r(tz["cmd_ge_0p99"][w, TAIL3].mean()),
                                "train_w16_penalty": r(tz["mean_bound_penalty_buffer"][w].mean(), 8),
                                "train_w16_ep_rew_mean": r(np.nanmean(tz["ep_rew_mean"][w]), 1),
                                "train_w16_approx_kl": r(np.nanmean(tz["prev_train_approx_kl"][w]), 5)})
            rows.append(row)
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(T, "results.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)
    # per-rollout training telemetry
    trows = []
    for arm, tz in tele.items():
        if tz is None:
            continue
        for i in range(len(tz["num_timesteps"])):
            mb = tz["mu_beyond"][i].copy()
            mb[TAIL3] = -1
            trows.append({"arm": arm, "num_timesteps": int(tz["num_timesteps"][i]),
                          "t3_mu_mean": r(tz["mu_mean"][i, TAIL3]), "t3_mu_beyond": r(tz["mu_beyond"][i, TAIL3]),
                          "t3_mu_q50": r(tz["mu_tail3_q"][i, 2]), "t3_mu_q95": r(tz["mu_tail3_q"][i, 4]),
                          "t3_cmd_ge_0p99": r(tz["cmd_ge_0p99"][i, TAIL3]), "t3_sample_clip_hi": r(tz["sample_at_clip_hi"][i, TAIL3]),
                          "t3_log_std": r(tz["log_std"][i, TAIL3]), "penalty_buffer": r(tz["mean_bound_penalty_buffer"][i], 8),
                          "other_mu_beyond_max": r(mb.max()), "other_mu_beyond_act": int(mb.argmax()),
                          "ep_rew_mean": r(tz["ep_rew_mean"][i], 1), "ep_len_mean": r(tz["ep_len_mean"][i], 1),
                          "prev_approx_kl": r(tz["prev_train_approx_kl"][i], 5), "prev_clip_fraction": r(tz["prev_train_clip_fraction"][i]),
                          "prev_mean_bound_loss": r(tz["prev_train_mean_bound_loss"][i], 8),
                          "prev_mean_bound_beyond_fraction": r(tz["prev_train_mean_bound_beyond_fraction"][i], 5)})
    if trows:
        with open(os.path.join(T, "train_telemetry.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(trows[0]))
            wr.writeheader()
            for row in trows:
                wr.writerow(row)
    print(json.dumps({"results_rows": len(rows), "telemetry_rows": len(trows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
