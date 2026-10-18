"""Reduce the push-and-release battery to push_summary.json.

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen). Report only; certifies
nothing; no training.

Inputs (under the battery's push/ directory):
  repo_probe/sweep_<role>.pkl                 Part A: main's probe reports (build_stance_gate_report with impulse)
  telemetry/push_<role>.json                  Part B: per-episode telemetry reductions
  telemetry/push_series_<role>.npz            Part B: per-episode series (CoM speed, CoP, pad CoP, XCoM margin)
"""
from __future__ import annotations

import json
import math
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(sys.argv[1])
ROLES = ("s60_6M", "s52_6M", "s42_6M", "statue")
DIRS = ("fwd", "back", "left", "right")
DIR_LABEL = {"fwd": "forward +x", "back": "backward -x", "left": "lateral +y (left)", "right": "lateral -y (right)"}
REPO_EPISODES = 8
RECOVERY_SHARE = 0.95  # stance_report._RECOVERY_SHARE
HOLD = 50
UNIT = {"fwd": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


def finite(values):
    return [float(x) for x in values if x is not None and math.isfinite(x)]


def med(values):
    v = finite(values)
    return None if not v else float(statistics.median(v))


def vmax(values):
    v = finite(values)
    return None if not v else float(max(v))


def vmin(values):
    v = finite(values)
    return None if not v else float(min(v))


def r(x, nd=3):
    return None if x is None else round(float(x), nd)


def first_settle(ok, hold):
    run = 0
    for k in range(len(ok)):
        run = run + 1 if ok[k] else 0
        if run >= hold:
            return k - hold + 1
    return None


tele, series = {}, {}
for role in ROLES:
    p = BASE / "telemetry" / f"push_{role}.json"
    if p.exists():
        tele[role] = json.loads(p.read_text())
        q = BASE / "telemetry" / f"push_series_{role}.npz"
        series[role] = np.load(q) if q.exists() else None

summary = {
    "label": "OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen, MirrorKLPPO + "
             "experiment_* keys); report only; certifies nothing; no training",
    "telemetry_sources": {role: str(BASE / "telemetry" / f"push_{role}.json") for role in tele},
    "series_sources": {role: str(BASE / "telemetry" / f"push_series_{role}.npz") for role in tele},
    "repo_probe_sources": {},
    "roles": {},
}

# ------------------------------------------------------------------ Part A: main's probe rows, and the crosscheck
repo = {}
for role in ROLES:
    p = BASE / "repo_probe" / f"sweep_{role}.pkl"
    if p.exists():
        repo[role] = pickle.load(open(p, "rb"))
        summary["repo_probe_sources"][role] = str(p)
summary["repo_probe_written"] = {role: str(BASE / "repo_probe" / role / "stance_gate_probe_impulse.txt")
                                 for role in ("s60_6M", "s52_6M", "s42_6M")}
repo_rows, crosscheck = {}, {}
for role, reports in repo.items():
    rows = {}
    for rep in reports:
        imp = rep["impulse"]
        rows[(imp["axis_label"], round(imp["speed"], 6))] = rep
    repo_rows[role] = {
        f"{k[0]} @ {k[1]:g}": {"full_horizon_fraction": v["metrics"]["full_horizon_fraction"],
                               "survived": sum(1 for e in v["episode_evidence"] if e["reached_horizon"]),
                               "n": v["episodes"], "length_mean": v["metrics"]["episode_length_mean"],
                               "terminations": v["terminations"], "first_seed": v["seed"]}
        for k, v in rows.items()}
    if role in tele:
        compared = equal_len = 0
        max_dr = 0.0
        for row in tele[role]["rows"]:
            key = (row["impulse"]["axis_label"], round(row["speed"], 6))
            if key not in rows:
                continue
            ev = {e["seed"]: e for e in rows[key]["episode_evidence"]}
            for e in row["episodes"]:
                if e["seed"] in ev:
                    compared += 1
                    equal_len += int(float(ev[e["seed"]]["length"]) == float(e["length"]))
                    max_dr = max(max_dr, abs(float(ev[e["seed"]]["reward"]) - float(e["reward"])))
        crosscheck[role] = {"episodes_compared": compared, "episodes_with_equal_length": equal_len,
                            "max_abs_reward_diff": max_dr}
summary["repo_probe_rows"] = repo_rows
summary["crosscheck_telemetry_vs_repo_probe"] = crosscheck

# ------------------------------------------------------------------ Part B
for role, data in tele.items():
    z = series[role]
    out = {"policy": data["policy"], "model": data["model"], "model_sha256": data["model_sha256"],
           "vecnorm": data["vecnorm"], "vecnorm_sha256": data["vecnorm_sha256"], "first_seed": data["first_seed"],
           "episodes_per_row": data["episodes_per_row"], "impulse_step": data["impulse_step"],
           "conventions": data["conventions"], "control": None, "directions": {}}
    # no-push baseline over the same window and the role-adaptive settle threshold
    ctrl = [row for row in data["rows"] if row["direction"] == "control"][0]
    ce = ctrl["episodes"]
    vh_ctrl = z["control_0_vh"] if z is not None else None
    base_p99 = float(np.nanpercentile(vh_ctrl, 99)) if vh_ctrl is not None else None
    tau = max(0.03, 1.5 * base_p99) if base_p99 is not None else 0.03
    ctrl_win_cop = {d: [e["win_cop_peak_disp_m"][d] for e in ce] for d in DIRS}
    out["control"] = {
        "survived": sum(e["survived"] for e in ce), "n": len(ce),
        "vh_p99_all_post_impulse_steps": r(base_p99, 4),
        "adaptive_settle_threshold_mps": r(tau, 4),
        "win_vh_max_median": r(med([e["win_vh_max"] for e in ce]), 4),
        "win_vh_max_max": r(vmax([e["win_vh_max"] for e in ce]), 4),
        "win_pad_cop_fore_aft_max_median": r(med([e["win_pad_cop_fore_aft_max"] for e in ce])),
        "win_pad_cop_fore_aft_max_max": r(vmax([e["win_pad_cop_fore_aft_max"] for e in ce])),
        "win_pad_cop_outer_max_median": r(med([e["win_pad_cop_outer_max"] for e in ce])),
        "win_cop_peak_disp_m_median": {d: r(med(v), 4) for d, v in ctrl_win_cop.items()},
        "win_cop_peak_disp_m_max": {d: r(vmax(v), 4) for d, v in ctrl_win_cop.items()},
        "pre_pad_cop_fore_aft_max_median": r(med([e["pre_pad_cop_fore_aft_max"] for e in ce])),
    }
    if z is not None:
        pc = z["control_0_pad_cop"]  # (E, 300, feet, [fore-aft +toe, outer +])
        out["control"]["pad_cop_fore_aft_signed_median_per_foot_steps_200_499"] = [
            r(float(v)) for v in np.nanmedian(pc[..., 0], axis=(0, 1))]
        out["control"]["pad_cop_outer_signed_median_per_foot_steps_200_499"] = [
            r(float(v)) for v in np.nanmedian(pc[..., 1], axis=(0, 1))]
    out["recovery_time_adaptive_definition"] = (
        f"seconds from the impulse until horizontal CoM speed stays below {tau:.4f} m/s = max(0.03, 1.5 x the "
        "role's own no-push p99 over steps 200-999 of the control row) for 50 steps, survivors only")
    for d in DIRS:
        rows = sorted([row for row in data["rows"] if row["direction"] == d], key=lambda x: x["speed"])
        if not rows:
            continue
        per_seed_pre, pre_mismatch = {}, 0.0
        for row in rows:
            for e in row["episodes"]:
                b = e["b"]
                vals = (e["omega0"], b["cop_pad"], b["xcom_pad"], b["cop_full"], b["xcom_full"], b["com_pad"])
                if e["seed"] in per_seed_pre:
                    pre_mismatch = max(pre_mismatch, max(abs(a - c) for a, c in zip(vals, per_seed_pre[e["seed"]])))
                else:
                    per_seed_pre[e["seed"]] = vals
        seeds = sorted(per_seed_pre)
        w0 = [per_seed_pre[s][0] for s in seeds]
        names = ("cop_pad", "xcom_pad", "cop_full", "xcom_full", "com_pad")
        bvals = {name: [per_seed_pre[s][i + 1] for s in seeds] for i, name in enumerate(names)}
        pred = {name: [per_seed_pre[s][0] * per_seed_pre[s][i + 1] for s in seeds] for i, name in enumerate(names[:4])}
        surv = {s: {} for s in seeds}
        for row in rows:
            for e in row["episodes"]:
                surv[e["seed"]][row["speed"]] = e["survived"]
        grid = [row["speed"] for row in rows]
        per_seed, non_monotone = [], 0
        for i, s in enumerate(seeds):
            ok = surv[s]
            best = 0.0
            for v in grid:
                if ok.get(v):
                    best = v
                else:
                    break
            first_fall = next((v for v in grid if not ok.get(v, False)), None)
            above = first_fall is not None and any(ok.get(v) for v in grid if v > first_fall)
            non_monotone += int(above)
            per_seed.append({"seed": s, "observed_max_survived_mps": best, "first_fall_mps": first_fall,
                             "survived_above_first_fall": above,
                             **{f"b_{k}_m": bvals[k][i] for k in names}, "omega0": w0[i],
                             **{f"pred_dv_{k}": pred[k][i] for k in pred}})
        obs = np.array([p["observed_max_survived_mps"] for p in per_seed])
        ffall = np.array([np.nan if p["first_fall_mps"] is None else p["first_fall_mps"] for p in per_seed])

        def spearman(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
                return None
            ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
            return float(np.corrcoef(ra, rb)[0, 1])

        def ratio(num, den):
            q = num / np.asarray(den, float)
            return r(float(np.nanmedian(q))) if np.isfinite(q).any() else None

        speeds = {}
        for row in rows:
            eps = row["episodes"]
            sv = [e for e in eps if e["survived"]]
            fl = [e for e in eps if not e["survived"]]
            key = f"{d}_{row['speed']:g}"
            adaptive, xm3, moving_end = [], [], 0
            if z is not None and f"{key}_vh" in z.files:
                vh = z[f"{key}_vh"]
                xmar = z[f"{key}_xcom_margin_pad"]
                for j, e in enumerate(eps):
                    if e["survived"]:
                        k = first_settle(np.nan_to_num(vh[j], nan=np.inf) < tau, HOLD)
                        adaptive.append(None if k is None else (k + 1) * 0.01)
                        xm3.append(float(np.nanmin(xmar[j])) if np.isfinite(xmar[j]).any() else None)
                        moving_end += int(vh[j, -1] > 0.05)
            speeds[f"{row['speed']:g}"] = {
                "survived": row["survived"], "n": row["n"],
                "survived_repo_seeds_first8": sum(e["survived"] for e in eps[:REPO_EPISODES]),
                "length_mean": r(row["length_mean"], 1),
                "fall_reasons": dict(Counter(e["reason"] for e in fl)),
                "time_to_fall_s_median": r(med([e["time_to_fall_s"] for e in fl]), 2),
                "recovery_time_s_v0.03_median": r(med([e["recovery_time_s_v0.03"] for e in sv]), 2),
                "recovery_time_s_v0.03_max": r(vmax([e["recovery_time_s_v0.03"] for e in sv]), 2),
                "recovery_time_s_v0.05_median": r(med([e["recovery_time_s_v0.05"] for e in sv]), 2),
                "recovery_time_s_v0.02_median": r(med([e["recovery_time_s_v0.02"] for e in sv]), 2),
                "recovery_time_s_adaptive_median": r(med(adaptive), 2),
                "recovery_time_s_adaptive_max": r(vmax(adaptive), 2),
                "survivors_never_settled_adaptive": sum(1 for a in adaptive if a is None),
                "pad_cop_recovery_time_s_median": r(med([e["pad_cop_recovery_time_s"] for e in sv]), 2),
                "win_pad_cop_fore_aft_max_median": r(med([e["win_pad_cop_fore_aft_max"] for e in eps])),
                "win_pad_cop_fore_aft_max_survivors_median": r(med([e["win_pad_cop_fore_aft_max"] for e in sv])),
                "win_pad_cop_outer_max_median": r(med([e["win_pad_cop_outer_max"] for e in eps])),
                "win_cop_peak_disp_along_push_m_median": r(med([e["win_cop_peak_disp_along_push_m"] for e in eps]), 4),
                "win_cop_peak_disp_along_push_m_survivors_median": r(med(
                    [e["win_cop_peak_disp_along_push_m"] for e in sv]), 4),
                "win_cop_peak_fraction_of_b_pad_median": r(med([e["win_cop_peak_fraction_of_b_pad"] for e in eps])),
                "win_cop_peak_fraction_of_b_pad_survivors_median": r(med(
                    [e["win_cop_peak_fraction_of_b_pad"] for e in sv])),
                "win_cop_peak_fraction_of_b_pad_survivors_max": r(vmax(
                    [e["win_cop_peak_fraction_of_b_pad"] for e in sv])),
                "cop_peak_fraction_of_b_pad_fallers_median": r(med([e["cop_peak_fraction_of_b_pad"] for e in fl])),
                "cop_min_margin_pad_m_survivors_median": r(med([e["cop_min_margin_pad_m"] for e in sv]), 4),
                "post_pad_cop_edge_fraction_survivors_median": r(med([e["post_pad_cop_edge_fraction"] for e in sv])),
                "post_whole_foot_cop_max_median": r(med([e["post_whole_foot_cop_max"] for e in eps])),
                "post_whole_foot_cop_min_median": r(med([e["post_whole_foot_cop_min"] for e in eps])),
                "xcom_min_margin_pad_m_survivors_min": r(vmin([e["xcom_min_margin_pad_m"] for e in sv]), 4),
                "xcom_min_margin_pad_first3s_m_survivors_min": r(vmin(xm3), 4),
                "survivors_moving_at_horizon": moving_end,
                "xcom_min_margin_pad_m_survivors_median": r(med([e["xcom_min_margin_pad_m"] for e in sv]), 4),
                "survivors_with_xcom_outside_pad": sum(1 for e in sv if (e["xcom_outside_pad_steps"] or 0) > 0),
                "xcom_min_margin_full_m_survivors_min": r(vmin([e["xcom_min_margin_full_m"] for e in sv]), 4),
                "com_peak_disp_along_push_m_survivors_median": r(med([e["com_peak_disp_along_push_m"] for e in sv]), 4),
                "com_final_disp_along_push_m_survivors_median": r(med([e["com_final_disp_along_push_m"] for e in sv]), 4),
                "post_1s_window_mean_pad_cop_fore_aft_max_survivors_median": r(med(
                    [e["post_1s_window_mean_pad_cop_fore_aft_max"] for e in sv])),
                "post_1s_window_mean_pad_cop_fore_aft_max_survivors_max": r(vmax(
                    [e["post_1s_window_mean_pad_cop_fore_aft_max"] for e in sv])),
                "fallers_with_xcom_outside_pad": sum(1 for e in fl if (e["xcom_outside_pad_steps"] or 0) > 0),
                "fallers_cop_min_margin_pad_m_median": r(med([e["cop_min_margin_pad_m"] for e in fl]), 4),
                "fallers_liftoff": sum(1 for e in fl if sum(e["liftoff_steps_per_foot"]) > 0),
                "stepped": sum(1 for e in eps if e["stepped"]),
                "survivors_stepped": sum(1 for e in sv if e["stepped"]),
                "liftoff_any": sum(1 for e in eps if sum(e["liftoff_steps_per_foot"]) > 0),
                "foot_max_horiz_disp_m_survivors_max": r(vmax([max(e["foot_max_horiz_disp_m"]) for e in sv]), 4),
                "win_vh_max_median": r(med([e["win_vh_max"] for e in eps]), 4),
            }
        full_env = 0.0
        for row in rows:
            if row["survived"] / row["n"] >= RECOVERY_SHARE:
                full_env = max(full_env, row["speed"])
        full_env_mono = 0.0
        for row in rows:
            if row["survived"] / row["n"] >= RECOVERY_SHARE:
                full_env_mono = row["speed"]
            else:
                break
        half_env = 0.0
        for row in rows:
            if row["survived"] / row["n"] >= 0.5:
                half_env = row["speed"]
            else:
                break
        out["directions"][d] = {
            "label": DIR_LABEL[d],
            "grid_mps": grid,
            "survival_by_speed": {f"{row['speed']:g}": f"{row['survived']}/{row['n']}" for row in rows},
            "speeds": speeds,
            "envelope_full_recovery_mps": full_env,
            "envelope_full_recovery_monotone_mps": full_env_mono,
            "envelope_ge50pct_monotone_mps": half_env,
            "first_speed_with_any_fall_mps": next((row["speed"] for row in rows if row["survived"] < row["n"]), None),
            "last_speed_with_any_survivor_mps": max((row["speed"] for row in rows if row["survived"] > 0), default=None),
            "margin_arithmetic": {
                "pre_impulse_identical_across_rows_max_abs_diff": pre_mismatch,
                "omega0_median": r(med(w0), 4),
                "com_height_m_median": r(9.81 / med(w0) ** 2, 4),
                "b_m_median": {k: r(med(v), 4) for k, v in bvals.items()},
                "b_m_min": {k: r(vmin(v), 4) for k, v in bvals.items()},
                "b_m_max": {k: r(vmax(v), 4) for k, v in bvals.items()},
                "pred_dv_mps_median": {k: r(med(v)) for k, v in pred.items()},
                "pred_dv_mps_min": {k: r(vmin(v)) for k, v in pred.items()},
                "pred_dv_mps_max": {k: r(vmax(v)) for k, v in pred.items()},
                "observed_per_seed_max_survived_mps_median": r(float(np.median(obs))),
                "observed_per_seed_max_survived_mps_min": r(float(np.min(obs))),
                "observed_per_seed_max_survived_mps_max": r(float(np.max(obs))),
                "observed_per_seed_first_fall_mps_median": (r(float(np.nanmedian(ffall)))
                                                            if np.isfinite(ffall).any() else None),
                "ratio_observed_max_survived_over_pred_cop_pad_median": ratio(obs, pred["cop_pad"]),
                "ratio_observed_max_survived_over_pred_cop_full_median": ratio(obs, pred["cop_full"]),
                "ratio_first_fall_over_pred_cop_pad_median": ratio(ffall, pred["cop_pad"]),
                "ratio_first_fall_over_pred_cop_full_median": ratio(ffall, pred["cop_full"]),
                "seeds_pred_cop_pad_between_max_survived_and_first_fall": int(sum(
                    1 for p in per_seed if p["first_fall_mps"] is not None
                    and p["observed_max_survived_mps"] <= p["pred_dv_cop_pad"] <= p["first_fall_mps"])),
                "spearman_observed_max_survived_vs_pred_cop_pad": r(spearman(obs, pred["cop_pad"])),
                "ratio_observed_max_survived_over_pred_xcom_pad_median": ratio(obs, pred["xcom_pad"]),
                "spearman_observed_max_survived_vs_pred_xcom_pad": r(spearman(obs, pred["xcom_pad"])),
                "seeds_pred_cop_pad_within_one_grid_step_of_bracket": int(sum(
                    1 for p in per_seed if p["first_fall_mps"] is not None
                    and p["observed_max_survived_mps"] - 0.05 - 1e-9 <= p["pred_dv_cop_pad"]
                    <= p["first_fall_mps"] + 0.05 + 1e-9)),
                "seeds_pred_xcom_pad_within_one_grid_step_of_bracket": int(sum(
                    1 for p in per_seed if p["first_fall_mps"] is not None
                    and p["observed_max_survived_mps"] - 0.05 - 1e-9 <= p["pred_dv_xcom_pad"]
                    <= p["first_fall_mps"] + 0.05 + 1e-9)),
                "seeds_surviving_above_their_first_fall": non_monotone,
                "n_seeds": len(seeds),
                "per_seed": per_seed,
            },
        }
    summary["roles"][role] = out

(BASE / "push_summary.json").write_text(json.dumps(summary, indent=1, default=float))
print("wrote", BASE / "push_summary.json")
