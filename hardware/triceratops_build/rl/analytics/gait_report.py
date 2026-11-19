"""Gait-quality report of a trained 60 cm Triceratops policy. Simulation only; one policy is reported, not judged.

Usage (from anywhere; CPU; about 90 s with 2 workers for 35 episodes):
  python hardware/triceratops_build/rl/analytics/gait_report.py --ckpt-dir DIR --run NAME [--out DIR] [--workers 2]
DIR holds one SB3 checkpoint pair (``*_<steps>_steps.zip`` and ``*_vecnormalize_<steps>_steps.pkl``) and the run's
``run_record.json``. Writes ``results/<NAME>/`` beside this file: ``gait_report.json`` (every metric, per episode and
per case), ``gait_report.md`` (the summary) and the charts (PNG, each labelled SIMULATION and with the run name).

Episodes (as ``rl/analysis/gait_pipe1.py``, whose definitions this extends): deterministic actions, servo delay
10 ms, seeds 100000-100004, randomisation and observation noise on, 20 s, measured over t >= 2 s:
  * 0.25 m/s (the museum target) and 0.15 m/s;
  * 0.25 m/s on the nominal model (no randomisation, no observation noise): the time-series charts use its seed
    100000 (t = 6-9 s), because a clean episode reads more easily;
  * 0.25 m/s on the roll-ankle model ``trike60_C_closed_ankleroll.xml`` (for the roll-ankle angle range);
  * a speed sweep 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.27 m/s on seeds 100000-100002 (0.15 and 0.25 reuse the
    episodes above).
Every physics step (1 ms) is logged through the env's ``substep_hook``.

Safety: nothing stored in the checkpoint runs. The policy weights load with ``torch.load(weights_only=True)`` into a
freshly built ELU 256-256 policy, the normalisation statistics through a restricted unpickler; both loaders are
imported from ``rl/analysis/gait_pipe1.py``. The environment is ``rl/trike_env.py`` (or an exported copy of ``rl/``
named by TRIKE_ENV_DIR, as gait_pipe1.py reads it), rebuilt from the run record's env config.

Definitions: README.md in this folder. Metric code that needs no simulator: gait_core.py (unit-tested).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

HERE = Path(__file__).resolve().parent
RL = HERE.parent
BUILD = RL.parent
REPO = BUILD.parents[1]
for p in (str(HERE), str(RL / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

import numpy as np  # noqa: E402

import gait_core as gc  # noqa: E402
import gait_pipe1 as gp  # noqa: E402  (puts TRIKE_ENV_DIR or rl/ first on sys.path and imports trike_env from it)
from environments.shared.floor_contact.reduction import decode_normal_forces, decode_settings  # noqa: E402

CORNERS = gc.CORNERS
SEEDS = gp.SEEDS
DELAY = gp.DELAY
WINDOW_S = gp.WINDOW_S
THERMAL, LIMIT, RPM_ALLOW = gp.THERMAL, gp.LIMIT, gp.RPM_ALLOW
GROUPS = tuple(THERMAL)
GEN = BUILD / "sim" / "generated"
DRIVES_JSON = BUILD / "sim" / "data" / "drives.json"
CATALOG_JSON = REPO / "hardware" / "trex_form_study" / "designlab" / "data" / "actuators.json"
REFERENCE_JSON = RL / "analysis" / "results" / "gait_pipe1_2026-11-19.json"
DEFAULT_MODEL = "trike60_C_closed.xml"
ANKLEROLL_MODEL = "trike60_C_closed_ankleroll.xml"

GROUP_DRIVE = {"roll": "X4-36-L", "pitch": "X4-36-L", "front elbow": "X6-60", "hind knee": "X4-36-L"}
CATALOG_ID = {"X4-36-L": "myactuator_rmd_x4_p36_36_e_l", "X6-60": "myactuator_rmd_x6_p20_60_e"}
BATTERY_WH, BATTERY_USABLE = 146.5, 0.80
HOTEL_W, HOTEL_RANGE_W = 30.0, (25.0, 40.0)
SWEEP = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.27)
SWEEP_SEEDS = SEEDS[:3]
HEAD_LOWPASS_HZ = 20.0
HEAD_BOB_HZ = 5.0  # head acceleration is also split at 5 Hz: <= 5 Hz the visible bob, 5-20 Hz the step impacts' shake
TRACE_T = (6.0, 9.0)
NEAR_EDGE_M = 0.005
LOAD_RATE_LAG_S, LOAD_RATE_SPAN_S, IMPACT_SPAN_S = 0.005, 0.030, 0.050
SCUFF_MM = 5.0
STEP_MIN_S = 0.050
GAP_BINS_MS = ((0, 10), (10, 20), (20, 50), (50, 80), (80, 10000))

CASES = {  # key: (label, model file, config changes, command)
    "target_0.25": ("0.25 m/s (target), randomised", DEFAULT_MODEL, {}, 0.25),
    "cmd_0.15": ("0.15 m/s, randomised", DEFAULT_MODEL, {}, 0.15),
    "nominal_0.25": ("0.25 m/s, nominal model, observation noise off", DEFAULT_MODEL,
                     {"domain_randomization": False, "observation_noise": False}, 0.25),
    "ankleroll_0.25": ("0.25 m/s, roll-ankle model, randomised", ANKLEROLL_MODEL, {}, 0.25),
}
REFERENCE_CASES = {"target_0.25": "target 0.25 m/s", "cmd_0.15": "0.15 m/s",
                   "nominal_0.25": "sensitivity: nominal model, no randomisation or noise, 0.25 m/s",
                   "ankleroll_0.25": "sensitivity: roll-ankle model, 0.25 m/s"}


# ---------------------------------------------------------------- drive data
def drive_table() -> dict:
    """Per joint group, the joint-side figures used here (N m, rpm), and the loss-model data with its source."""
    drives = json.loads(DRIVES_JSON.read_text())
    eta_t = float(drives["transmission_efficiency"])
    cat = {a["id"]: a for a in json.loads(CATALOG_JSON.read_text())["actuators"]}
    out = {}
    for g, name in GROUP_DRIVE.items():
        d = drives["leg_drives"][name]
        eta = eta_t if g in ("front elbow", "hind knee") else 1.0  # the flexion joints work through a 1:1 linkage
        c = cat[CATALOG_ID[name]]
        out[g] = {"drive": name, "transmission_efficiency": eta, "model_limit_nm": LIMIT[g],
                  "maker_peak_nm": d["peak_nm"] * eta, "maker_rated_nm": d["rated_nm"] * eta,
                  "continuous_nm": THERMAL[g], "speed_allowance_rpm": RPM_ALLOW[g],
                  "kt_nm_per_a_rms": c["kt_nm_per_a"], "phase_resistance_ohm": c["phase_resistance_ohm"],
                  "loss_data_source": f"{CATALOG_JSON.relative_to(REPO)} id {CATALOG_ID[name]} (output-side Kt, "
                                      "A rms; per-phase Y-equivalent resistance)"}
        assert abs(d["thermal_eq_nm"] * eta - THERMAL[g]) < 1e-9, (g, d["thermal_eq_nm"], eta)
        assert abs(0.70 * d["peak_nm"] * eta - LIMIT[g]) < 1e-9, (g, d["peak_nm"], eta)
    return out


# ---------------------------------------------------------------- one logged episode
def run_episode(env, policy, stats, seed: int, command: float, full: bool = True) -> dict:
    """Run one deterministic episode; log every physics step (``full``) or only the control steps."""
    import mujoco
    import torch

    mean, var, clip, eps = stats
    obs, _ = env.reset(seed=seed, options={"command_speed": command, "delay_steps": DELAY})
    m, d = env.model, env.data
    idx = np.asarray(env._leg_act, dtype=np.int64)
    ch = int(env._chassis)
    log: dict = {"names": [m.actuator(int(i)).name for i in idx], "limit_ep": m.actuator_forcerange[idx, 1].copy(),
                 "mass": float(m.body_mass.sum()), "gravity": float(-m.opt.gravity[2]),
                 "heading0": float(env._heading), "contact_n": float(env.cfg.contact_threshold_n),
                 "friction": env.factors.get("friction") if env.factors else None}
    log["limit_nominal"] = env._nominal["actuator_forcerange"][idx, 1].copy()
    steps = {"t": [], "v": [], "pos": []}
    if full:
        N = int(round(env.cfg.episode_seconds / env.physics_dt)) + 2 * env.cfg.frame_skip
        A = {k: np.zeros((N,) + s) for k, s in (
            ("t", ()), ("tau", (12,)), ("omega", (12,)), ("load", (4,)), ("fn", (4,)), ("fpx", (4,)), ("fpy", (4,)),
            ("sole_p", (4, 3)), ("sole_m", (4, 9)), ("sole_vel", (4, 3)), ("pos", (3,)), ("xmat", (9,)),
            ("com", (3,)), ("head_p", (3,)), ("head_v", (3,)), ("ankle", (4,)), ("ankle_roll", (4,)))}
        floor, soles, touch = int(env._floor), np.asarray(env._soles, dtype=np.int64), np.asarray(env._touch_adr)
        geom_foot = np.full(m.ngeom, -1, dtype=np.int64)
        geom_foot[soles] = np.arange(4)
        elliptic, edge_off = decode_settings(m)
        head = m.body("head_yaw").id
        ank = np.array([m.jnt_qposadr[m.joint(f"{c}_ankle").id] for c in CORNERS])
        roll_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{c}_ankle_roll") for c in CORNERS]
        ankr = np.array([m.jnt_qposadr[j] for j in roll_ids]) if min(roll_ids) >= 0 else None
        if ankr is not None:
            log["ankle_roll_limit_deg"] = float(np.degrees(m.jnt_range[roll_ids[0], 1]))
        log["sole_size"] = m.geom_size[soles[0]].copy()
        v6 = np.zeros(6)
        k = [0]
        OBJ_GEOM, OBJ_BODY = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
        objvel = mujoco.mj_objectVelocity

        def hook(e):
            i = k[0]
            if i >= N:
                return
            A["t"][i] = d.time
            A["tau"][i] = d.actuator_force[idx]
            A["omega"][i] = d.actuator_velocity[idx]
            A["load"][i] = d.sensordata[touch]
            if d.ncon:
                con = d.contact
                gg = con.geom
                fl = gg == floor
                other = np.where(fl[:, 0], gg[:, 1], gg[:, 0])
                foot = np.where(fl[:, 0] ^ fl[:, 1], geom_foot[other], -1)
                ok = foot >= 0
                if ok.any():
                    fn = decode_normal_forces(con, d.efc_force, elliptic=elliptic, edge_offsets=edge_off)[ok]
                    ft, ps = foot[ok], con.pos[ok]
                    A["fn"][i] = np.bincount(ft, fn, 4)
                    A["fpx"][i] = np.bincount(ft, fn * ps[:, 0], 4)
                    A["fpy"][i] = np.bincount(ft, fn * ps[:, 1], 4)
            A["sole_p"][i] = d.geom_xpos[soles]
            A["sole_m"][i] = d.geom_xmat[soles]
            for j in range(4):
                objvel(m, d, OBJ_GEOM, int(soles[j]), v6, 0)
                A["sole_vel"][i, j] = v6[3:]
            A["pos"][i] = d.xpos[ch]
            A["xmat"][i] = d.xmat[ch]
            A["com"][i] = d.subtree_com[ch]
            A["head_p"][i] = d.xipos[head]
            objvel(m, d, OBJ_BODY, head, v6, 0)
            A["head_v"][i] = v6[3:]
            A["ankle"][i] = d.qpos[ank]
            if ankr is not None:
                A["ankle_roll"][i] = d.qpos[ankr]
            k[0] = i + 1

        env.substep_hook = hook
    fell, term, info = False, "", {}
    try:
        while True:
            o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            with torch.no_grad():
                act, _, _ = policy(torch.as_tensor(o[None]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(act.numpy()[0])
            steps["t"].append(info["time_s"])
            steps["v"].append(info["forward_speed"])
            steps["pos"].append(d.xpos[ch].copy())
            if terminated or truncated:
                fell, term = bool(terminated), info["termination"]
                break
    finally:
        env.substep_hook = None
    log.update(fell=fell, termination=term, episode_forward_speed=float(info.get("episode_forward_speed", np.nan)),
               length_s=float(info.get("time_s", 0.0)), step_t=np.asarray(steps["t"]), step_v=np.asarray(steps["v"]),
               step_pos=np.asarray(steps["pos"]), debounce_steps=int(getattr(env, "_debounce_steps", 10)),
               ratings=np.array([THERMAL[gp.group_of(n)] for n in log["names"]]), has_ankle_roll=False)
    if full:
        n = k[0]
        for key, arr in A.items():
            log[key] = arr[:n]
        log["has_ankle_roll"] = ankr is not None
        R = log["xmat"].reshape(-1, 3, 3)
        log["rp"] = np.stack([np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2])),
                              np.degrees(np.arcsin(-np.clip(R[:, 2, 0], -1, 1)))], axis=1)
        log["yaw"] = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        log["sole_v"] = np.hypot(log["sole_vel"][..., 0], log["sole_vel"][..., 1])
    return log


# ---------------------------------------------------------------- metrics of one episode
def window_speed_from_steps(log: dict) -> float:
    t, p = log["step_t"], log["step_pos"]
    w = t >= WINDOW_S
    if w.sum() < 2:
        return float("nan")
    h = np.array([math.cos(log["heading0"]), math.sin(log["heading0"])])
    i0 = int(np.argmax(w)) - 1 if int(np.argmax(w)) > 0 else 0  # position at the window's start
    return float((p[-1, :2] - p[i0, :2]) @ h / (t[-1] - t[i0]))


def analyse(a: dict, command: float, drives: dict, trace: bool) -> tuple[dict, dict]:
    """(metrics, chart payload) of one fully logged episode."""
    t = a["t"]
    dt = float(np.median(np.diff(t)))
    w = t >= WINDOW_S
    i0 = int(np.argmax(w))
    n_all = t.size
    h = np.array([math.cos(a["heading0"]), math.sin(a["heading0"])])
    lat = np.array([-h[1], h[0]])
    bw = a["mass"] * a["gravity"]
    thr = a["contact_n"]
    met: dict = {"seed": a.get("seed"), "command_mps": command, "fell": a["fell"], "termination": a["termination"],
                 "friction": a["friction"], "mass_kg": a["mass"], "window_s": float(t[w][-1] - t[w][0] + dt)}
    pay: dict = {}

    # --- the existing analysis's own measures, by its own function, on this log (exact reuse) ---
    met["pipe1_measures"] = gp.measures(a, command)
    pm = met["pipe1_measures"]
    try:
        import gait_metrics  # the training evaluation's measures (rl/gait_metrics.py), when its env is the current one
        met["training_eval_measures"] = gait_metrics.measures(a, command)
    except Exception as exc:  # an older exported env lacks names gait_metrics imports
        met["training_eval_measures"] = {"unavailable": repr(exc)}

    # --- h. speed ---
    sw = a["step_t"] >= WINDOW_S
    met["speed"] = {"forward_speed_mps": pm["forward_speed_mps"], "episode_forward_speed_mps": a["episode_forward_speed"],
                    "sideways_drift_m": pm["sideways_drift_m"], "heading_change_deg": pm["heading_change_deg"],
                    "mean_abs_step_error_mps": float(np.mean(np.abs(a["step_v"][sw] - command))),
                    "abs_mean_error_mps": abs(pm["forward_speed_mps"] - command),
                    "step_speed_sd_mps": float(np.std(a["step_v"][sw]))}
    pay["step_t"], pay["step_v"] = a["step_t"].tolist(), a["step_v"].tolist()

    # --- a. footfalls ---
    raw = a["load"] > thr
    deb = gc.debounce_contact(raw, a["debounce_steps"])  # the env's rule (10 ms): touchdown counts as in training
    stp = gc.step_contact(raw, a["debounce_steps"], int(round(STEP_MIN_S / dt)))  # steps (module docstring)
    pos_h = a["pos"][:, :2] @ h
    td = {c: gc.touchdowns(stp[:, j]) for j, c in enumerate(CORNERS)}
    tdw = {c: v[v >= i0] for c, v in td.items()}
    td_env = {c: (lambda v: v[v >= i0])(gc.touchdowns(deb[:, j])) for j, c in enumerate(CORNERS)}
    T = met["window_s"]
    ff: dict = {"duty_factor": {}, "duty_factor_env_debounce": {}, "duty_factor_raw": {}, "steps_per_s": {},
                "touchdowns_per_s": {}, "stride_period_s_median": {}, "stride_frequency_hz": {},
                "irregular_stride_fraction": {}, "stance_ms": {}, "swing_ms": {}, "stance_travel_m": {},
                "mean_load_bw": {}, "contact_gaps": {}}
    stance_ms, swing_ms, travel, periods = {}, {}, {}, {}
    low_z = None
    for j, c in enumerate(CORNERS):
        ff["duty_factor_raw"][c] = float(raw[w, j].mean())
        ff["duty_factor_env_debounce"][c] = float(deb[w, j].mean())
        ff["duty_factor"][c] = float(stp[w, j].mean())
        ff["touchdowns_per_s"][c] = float(td_env[c].size / T)
        ff["steps_per_s"][c] = float(tdw[c].size / T)
        per = np.diff(tdw[c]) * dt
        periods[c] = per
        med = float(np.median(per)) if per.size else None
        ff["stride_period_s_median"][c] = med
        ff["stride_frequency_hz"][c] = 1.0 / med if med else None
        ff["irregular_stride_fraction"][c] = float(np.mean(np.abs(per / med - 1) > 0.25)) if med else None
        st = gc.segments(stp[:, j], i0, n_all, True)
        sg = gc.segments(stp[:, j], i0, n_all, False)
        stance_ms[c] = np.array([(b - s) * dt * 1000 for s, b in st])
        swing_ms[c] = np.array([(b - s) * dt * 1000 for s, b in sg])
        travel[c] = np.array([pos_h[b - 1] - pos_h[s] for s, b in st])
        ff["stance_ms"][c] = gc.stats(stance_ms[c])
        ff["swing_ms"][c] = gc.stats(swing_ms[c])
        ff["stance_travel_m"][c] = gc.stats(travel[c])
        ff["mean_load_bw"][c] = float(a["fn"][w, j].mean() / bw)
    corners = gc.sole_bottom_corners(a["sole_p"], a["sole_m"], a["sole_size"])  # samples x 4 x 4 x 3
    low_z = corners[..., 2].min(axis=2)  # samples x 4: each sole's lowest corner height above the floor (z = 0)
    for j, c in enumerate(CORNERS):  # raw unloaded gaps by length, with how far the sole rose in them
        gaps = [(n, 1000 * low_z[s:s + n, j].max()) for v, s, n in gc.runs(raw[:, j])
                if not v and s >= i0 and s + n < n_all]
        rows = {}
        for lo_ms, hi_ms in GAP_BINS_MS:
            sel = [cl for n, cl in gaps if lo_ms <= n * dt * 1000 < hi_ms]
            rows[f"{lo_ms}-{hi_ms} ms"] = {"count": len(sel), "median_lift_mm": float(np.median(sel)) if sel else None,
                                          "max_lift_mm": float(max(sel)) if sel else None}
        ff["contact_gaps"][c] = rows
    lf = tdw["left_fore"]
    strides_m = np.array([pos_h[b] - pos_h[s] for s, b in zip(lf[:-1], lf[1:])])
    ff["stride_length_m"] = gc.stats(strides_m)
    off, lock, phases = {}, {}, {}
    for c in CORNERS[1:]:
        ph = gc.phase_offsets(lf, tdw[c])
        phases[c] = ph
        off[c], lock[c] = gc.circular_mean(ph)
    ff["phase_offset_after_LF"] = {c: {"mean_cycles": off[c], "resultant_length": lock[c], "strides": int(phases[c].size)}
                                   for c in CORNERS[1:]}
    ff["gait"] = gc.classify_gait(off, lock)
    ff["diagonal_timing"] = gc.diagonal_timing(off)
    n_down = stp[w].sum(1)
    LF, RF, LH, RH = (stp[w, j] for j in range(4))
    two = n_down == 2
    ff["feet_down_fraction"] = {"4": float((n_down == 4).mean()), "3": float((n_down == 3).mean()),
                                "2 diagonal": float((two & ((LF & RH) | (RF & LH))).mean()),
                                "2 same side": float((two & ((LF & LH) | (RF & RH))).mean()),
                                "2 front": float((two & LF & RF).mean()), "2 hind": float((two & LH & RH).mean()),
                                "1": float((n_down == 1).mean()), "0": float((n_down == 0).mean())}
    met["footfall"] = ff
    pay["stride_phases"] = {c: phases[c].tolist() for c in CORNERS[1:]}
    pay["stance_ms"] = {c: stance_ms[c].tolist() for c in CORNERS}
    pay["stance_travel_m"] = {c: travel[c].tolist() for c in CORNERS}
    pay["stride_periods_s"] = {c: periods[c].tolist() for c in CORNERS}
    pay["stride_length_m"] = strides_m.tolist()
    try:
        import gait_metrics
        ref_td = gait_metrics.debounced_touchdowns(raw, a["debounce_steps"], i0)
        met["footfall"]["touchdowns_equal_training_eval"] = bool(
            np.array_equal(ref_td, np.array([td_env[c].size for c in CORNERS])))
    except Exception:
        met["footfall"]["touchdowns_equal_training_eval"] = None

    # --- b. weight distribution (floor truth: sole-floor normal forces) ---
    fn = a["fn"]
    load_bw = fn / bw
    tot = fn[w].sum(1)
    share = {c: float(fn[w, j].mean() / tot.mean()) for j, c in enumerate(CORNERS)}
    met["weight"] = {"body_weight_n": bw, "mean_total_load_bw": float(tot.mean() / bw), "share": share,
                     "front_share": share["left_fore"] + share["right_fore"],
                     "left_share": share["left_fore"] + share["left_hind"],
                     "peak_load_bw": {c: float(load_bw[w, j].max()) for j, c in enumerate(CORNERS)},
                     "p99_load_bw": {c: float(np.percentile(load_bw[w, j], 99)) for j, c in enumerate(CORNERS)},
                     "touch_vs_floor_mean_abs_n": {c: float(np.abs(a["load"][w, j] - fn[w, j]).mean())
                                                   for j, c in enumerate(CORNERS)},
                     "touch_vs_floor_state_agreement": {c: float(((a["load"][w, j] > thr) == (fn[w, j] > thr)).mean())
                                                        for j, c in enumerate(CORNERS)}}
    prof = gc.stride_resample(load_bw, lf, 101)
    pay["stride_load_profiles"] = prof.astype(np.float32).tolist()

    # --- c. balance ---
    # Headline: the support polygon of the feet in stance on the step contact the footfalls use (50 ms rule). Raw 2 N
    # contact counts every 1-2 ms solver rebound of one foot of a diagonal pair as a lift, leaving one foot "down"
    # and the CoM ~0.2 m outside; the raw and env-debounce (10 ms) figures are kept, labelled, for comparison.
    cxy = corners[..., :2]
    com = a["com"][:, :2]

    def balance(down):
        fsum = np.where(down, fn, 0).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cop_ = np.stack([np.where(down, a["fpx"], 0).sum(1) / fsum, np.where(down, a["fpy"], 0).sum(1) / fsum], 1)
        cop_[fsum <= 0] = np.nan
        mc_, mm_ = gc.margins_over_time(cxy[w], down[w], cop_[w], com[w])
        sup = np.isfinite(mm_)
        nfeet = down[w].sum(1)
        bal = {
            "support_defined_fraction": float(sup.mean()), "no_foot_down_fraction": float((nfeet == 0).mean()),
            "cop_margin_negative_fraction": float(np.mean(mc_[np.isfinite(mc_)] < -gc.OUTSIDE_TOL_M))
            if np.isfinite(mc_).any() else None,
            "cop_near_edge_fraction": float(np.mean(mc_[np.isfinite(mc_)] < NEAR_EDGE_M)) if np.isfinite(mc_).any() else None,
            "cop_margin_mm": gc.stats(1000 * mc_), "com_margin_mm": gc.stats(1000 * mm_),
            "com_margin_negative_fraction": float(np.mean(mm_[sup] < -gc.OUTSIDE_TOL_M)) if sup.any() else None,
            "com_margin_negative_fraction_by_feet": {
                str(k): (float(np.mean(mm_[sup & (nfeet == k)] < -gc.OUTSIDE_TOL_M)) if (sup & (nfeet == k)).any() else None)
                for k in (1, 2, 3, 4)},
            "feet_down_fraction": {str(k): float((nfeet == k).mean()) for k in range(5)}}
        return bal, cop_, mc_, mm_

    met["balance"], cop, mc, mm = balance(stp)
    met["balance"]["contact"] = "step contact (env 10 ms debounce, then gaps and blips under 50 ms merged)"
    met["balance_env_debounce"] = balance(deb)[0]
    met["balance_env_debounce"]["contact"] = "env 10 ms debounce"
    met["balance_raw_contact"] = balance(fn > thr)[0]
    met["balance_raw_contact"]["contact"] = "raw: sole load above 2 N at each 1 ms step (solver rebounds count as lifts)"

    # --- d. drives ---
    tau, om = a["tau"][w], a["omega"][w]
    rpm = om * 60 / (2 * math.pi)
    pj = {}
    hist = {}
    for j, nm in enumerate(a["names"]):
        g = gp.group_of(nm)
        dv = drives[g]
        pj[nm] = {"group": g, "rms_nm": pm["per_joint"][nm]["rms_nm"],
                  "rms_over_continuous": pm["per_joint"][nm]["rms_over_thermal"],
                  "at_nominal_limit": pm["per_joint"][nm]["sat_nominal"],
                  "at_episode_limit": pm["per_joint"][nm]["sat_episode_limit"],
                  "peak_nm": float(np.abs(tau[:, j]).max()), "peak_rpm": float(np.abs(rpm[:, j]).max()),
                  "above_maker_rated_fraction": float(np.mean(np.abs(tau[:, j]) > dv["maker_rated_nm"])),
                  "above_speed_allowance_fraction": float(np.mean(np.abs(rpm[:, j]) > dv["speed_allowance_rpm"]))}
    for g in GROUPS:
        cols = [j for j, nm in enumerate(a["names"]) if gp.group_of(nm) == g]
        dv = drives[g]
        xe = np.linspace(-1.4 * dv["speed_allowance_rpm"], 1.4 * dv["speed_allowance_rpm"], 81)
        ye = np.linspace(-1.08 * dv["maker_peak_nm"], 1.08 * dv["maker_peak_nm"], 81)
        H, _, _ = np.histogram2d(rpm[:, cols].ravel(), tau[:, cols].ravel(), bins=(xe, ye))
        hist[g] = {"rpm_edges": xe.tolist(), "nm_edges": ye.tolist(), "counts": H.astype(int).tolist(),
                   "outside": int(rpm[:, cols].size - H.sum())}
    met["drives"] = {"per_joint": pj, "groups": pm["drives"]}
    pay["drive_hist"] = hist

    # --- e. power ---
    # Electrical ESTIMATES (README): copper loss from the drive torque, which is the joint torque / linkage efficiency
    # while the drive motors (torque x speed >= 0) and joint torque x efficiency while it brakes; mechanical power at
    # the drive likewise p / eta or p x eta. Two bus models: per drive with no reuse (each drive draws its positive
    # power plus copper loss; braking energy is lost), and a shared bus (the 12 drives' net draw each 1 ms step, braking
    # power feeding the others; a net surplus is dumped, since the ideal diodes block it from the battery). Unmodelled
    # gearbox, iron and driver losses would raise both.
    p = tau * om
    eta = np.array([drives[gp.group_of(nm)]["transmission_efficiency"] for nm in a["names"]])
    kt = np.array([drives[gp.group_of(nm)]["kt_nm_per_a_rms"] for nm in a["names"]])
    res = np.array([drives[gp.group_of(nm)]["phase_resistance_ohm"] for nm in a["names"]])
    el = gc.electrical_power(tau, om, eta, kt, res)
    p_drive, p_cu = el["drive_mech"], el["copper"]
    p_elec, p_bus = el["per_drive"], el["shared_bus"]  # per drive with no reuse; shared bus
    v = pm["forward_speed_mps"]
    P = {"mech_abs_w": float(np.abs(p).sum(1).mean()), "mech_positive_w": float(np.maximum(p, 0).sum(1).mean()),
         "mech_negative_w": float(np.minimum(p, 0).sum(1).mean()), "copper_w": float(p_cu.sum(1).mean()),
         "electrical_per_drive_w": float(p_elec.mean()), "electrical_per_drive_peak_w": float(p_elec.max()),
         "electrical_shared_bus_w": float(p_bus.mean()), "electrical_shared_bus_peak_w": float(p_bus.max()),
         "shared_bus_surplus_fraction": float(np.mean((p_drive + p_cu).sum(1) < 0))}
    P["by_group_w"] = {g: {"positive_mech": float(np.maximum(p[:, cols], 0).sum(1).mean()),
                           "copper": float(p_cu[:, cols].sum(1).mean())}
                       for g in GROUPS for cols in [[j for j, nm in enumerate(a["names"]) if gp.group_of(nm) == g]]}
    weight = a["mass"] * a["gravity"]
    P["cost_of_transport_mech"] = P["mech_abs_w"] / (weight * v) if v > 0.02 else None
    for model in ("per_drive", "shared_bus"):
        e_w = P[f"electrical_{model}_w"]
        P[f"cost_of_transport_electrical_{model}_with_hotel"] = (e_w + HOTEL_W) / (weight * v) if v > 0.02 else None
        P[f"run_time_min_{model}"] = {f"hotel_{int(hw)}w": 60.0 * BATTERY_WH * BATTERY_USABLE / (e_w + hw)
                                      for hw in (HOTEL_RANGE_W[0], HOTEL_W, HOTEL_RANGE_W[1])}
    met["power"] = P

    # --- f. body and head ---
    rp = a["rp"][w]
    z = a["pos"][w, 2]
    met["body"] = {
        "roll_deg": {"mean": float(rp[:, 0].mean()), "rms": float(np.sqrt((rp[:, 0] ** 2).mean())),
                     "sd": float(rp[:, 0].std()), "range_p1_p99": float(np.ptp(np.percentile(rp[:, 0], [1, 99]))),
                     "peak_abs": float(np.abs(rp[:, 0]).max())},
        "pitch_deg": {"mean": float(rp[:, 1].mean()), "rms": float(np.sqrt((rp[:, 1] ** 2).mean())),
                      "sd": float(rp[:, 1].std()), "range_p1_p99": float(np.ptp(np.percentile(rp[:, 1], [1, 99]))),
                      "peak_abs": float(np.abs(rp[:, 1]).max())},
        "height_mm": {"mean": float(1000 * z.mean()), "sd": float(1000 * z.std()),
                      "range_p1_p99": float(1000 * np.ptp(np.percentile(z, [1, 99]))),
                      "dominant_hz": gc.dominant_frequency(z, dt)},
        "roll_dominant_hz": gc.dominant_frequency(rp[:, 0], dt), "pitch_dominant_hz": gc.dominant_frequency(rp[:, 1], dt)}
    hv = a["head_v"]
    acc_raw = np.gradient(hv, dt, axis=0)
    acc = gc.lowpass_fft(acc_raw[w], dt, HEAD_LOWPASS_HZ)
    acc_lo = gc.lowpass_fft(acc_raw[w], dt, HEAD_BOB_HZ)  # the visible bob and sway
    acc_hi = acc - acc_lo  # HEAD_BOB_HZ-HEAD_LOWPASS_HZ: the step impacts' shake

    def frame_axes(x):
        return np.stack([x[:, :2] @ h, x[:, :2] @ lat, x[:, 2]], 1)  # fore-aft, lateral, vertical

    comp, comp_lo, comp_hi = frame_axes(acc), frame_axes(acc_lo), frame_axes(acc_hi)
    mag = np.linalg.norm(acc, axis=1)
    mag_lo, mag_hi = np.linalg.norm(acc_lo, axis=1), np.linalg.norm(acc_hi, axis=1)
    axes3 = ("fore_aft", "lateral", "vertical")
    hp = a["head_p"][w]
    one_s = int(round(1.0 / dt))
    bob = hp[:, 2] - gc.moving_average(hp[:, 2], one_s)
    sway = hp[:, :2] @ lat
    sway = sway - gc.moving_average(sway, one_s)
    met["head"] = {"lowpass_hz": HEAD_LOWPASS_HZ,
                   "acc_rms_mps2": float(np.sqrt((mag ** 2).mean())), "acc_peak_mps2": float(mag.max()),
                   "acc_p99_mps2": float(np.percentile(mag, 99)),
                   "acc_rms_by_axis_mps2": {k: float(np.sqrt((comp[:, i] ** 2).mean())) for i, k in enumerate(axes3)},
                   "band_split_hz": [HEAD_BOB_HZ, HEAD_LOWPASS_HZ],
                   "acc_rms_le5hz_mps2": float(np.sqrt((mag_lo ** 2).mean())), "acc_peak_le5hz_mps2": float(mag_lo.max()),
                   "acc_rms_5_20hz_mps2": float(np.sqrt((mag_hi ** 2).mean())), "acc_peak_5_20hz_mps2": float(mag_hi.max()),
                   "acc_rms_le5hz_by_axis_mps2": {k: float(np.sqrt((comp_lo[:, i] ** 2).mean())) for i, k in enumerate(axes3)},
                   "acc_rms_5_20hz_by_axis_mps2": {k: float(np.sqrt((comp_hi[:, i] ** 2).mean()))
                                                   for i, k in enumerate(axes3)},
                   "acc_peak_unfiltered_mps2": float(np.linalg.norm(acc_raw[w], axis=1).max()),
                   "bob_mm_p1_p99": float(1000 * np.ptp(np.percentile(bob, [1, 99]))),
                   "sway_mm_p1_p99": float(1000 * np.ptp(np.percentile(sway, [1, 99]))),
                   "bob_dominant_hz": gc.dominant_frequency(hp[:, 2], dt)}

    # --- g. feet ---
    feet: dict = {"swing_clearance_mm": {}, "scuff_fraction": {}, "touchdown_speed_mps": {}, "peak_load_rate_bw_per_s": {},
                  "impact_peak_bw": {}, "stance_slip": {}, "ankle_pitch_deg_p1_p99": {}}
    clear, tdv, rate, imp = {}, {}, {}, {}
    lag, span, ispan = (int(round(x / dt)) for x in (LOAD_RATE_LAG_S, LOAD_RATE_SPAN_S, IMPACT_SPAN_S))
    for j, c in enumerate(CORNERS):
        sg = gc.segments(stp[:, j], i0, n_all, False)
        clear[c] = np.array([1000 * low_z[s:b, j].max() for s, b in sg])
        feet["swing_clearance_mm"][c] = gc.stats(clear[c])
        feet["scuff_fraction"][c] = float(np.mean(clear[c] < SCUFF_MM)) if clear[c].size else None
        tds = tdw[c][(tdw[c] >= 1) & (tdw[c] + span + lag < n_all)]
        tdv[c] = -a["sole_vel"][tds - 1, j, 2]
        rate[c] = np.array([((fn[k + lag:k + span + lag, j] - fn[k:k + span, j]) / (lag * dt)).max() / bw for k in tds])
        imp[c] = np.array([fn[k:k + ispan, j].max() / bw for k in tds])
        feet["touchdown_speed_mps"][c] = gc.stats(tdv[c])
        feet["peak_load_rate_bw_per_s"][c] = gc.stats(rate[c])
        feet["impact_peak_bw"][c] = gc.stats(imp[c])
        sl = pm["stance_slip"][c]
        feet["stance_slip"][c] = {"mean_mps": sl["mean_mps"], "p95_mps": sl["p95_mps"], "slip_ratio": sl["slip_ratio"]}
        feet["ankle_pitch_deg_p1_p99"][c] = [float(x) for x in np.percentile(np.degrees(a["ankle"][w, j]), [1, 99])]
    if a["has_ankle_roll"]:
        q = np.degrees(a["ankle_roll"][w])
        lim = a["ankle_roll_limit_deg"]
        feet["ankle_roll_deg"] = {c: {"p1": float(np.percentile(q[:, j], 1)), "p99": float(np.percentile(q[:, j], 99)),
                                      "min": float(q[:, j].min()), "max": float(q[:, j].max()),
                                      "stance_p1_p99": [float(x) for x in np.percentile(q[stp[w, j], j], [1, 99])],
                                      "at_limit_fraction": float(np.mean(np.abs(q[:, j]) >= 0.98 * lim))}
                                  for j, c in enumerate(CORNERS)}
        feet["ankle_roll_limit_deg"] = lim
    met["feet"] = feet
    edges = np.linspace(0, 1.0, 51)
    sv = a["sole_v"][w]
    pay["slip_hist"] = {"edges": edges.tolist(),
                        "counts": {c: np.histogram(np.clip(sv[raw[w, j], j], 0, 0.9999), edges)[0].tolist()
                                   for j, c in enumerate(CORNERS)}}
    pay["clearance_mm"] = {c: clear[c].tolist() for c in CORNERS}
    pay["touchdown_speed_mps"] = {c: tdv[c].tolist() for c in CORNERS}
    pay["load_rate_bw_per_s"] = {c: rate[c].tolist() for c in CORNERS}
    if a["has_ankle_roll"]:
        pay["ankle_roll_deg"] = {c: np.percentile(np.degrees(a["ankle_roll"][w, j]), np.arange(0, 101, 1)).tolist()
                                 for j, c in enumerate(CORNERS)}

    # --- time-series slice for the charts ---
    if trace:
        s = (t >= TRACE_T[0]) & (t < TRACE_T[1])
        sw_ = s[w]
        pay["trace"] = {
            "t": t[s].tolist(), "stp": stp[s].astype(int).tolist(), "deb": deb[s].astype(int).tolist(),
            "raw": raw[s].astype(int).tolist(),
            "load_bw": load_bw[s].tolist(), "cop": cop[s].tolist(), "com": com[s].tolist(),
            "corners_xy": cxy[s].round(5).tolist(), "down_ft": stp[s].astype(int).tolist(),
            "down_raw": (fn[s] > thr).astype(int).tolist(),
            "cop_margin": mc[sw_].tolist(), "com_margin": mm[sw_].tolist(),
            "p_pos": np.maximum(p, 0)[sw_].sum(1).tolist(), "p_abs": np.abs(p)[sw_].sum(1).tolist(),
            "p_cu": p_cu[sw_].sum(1).tolist(), "p_elec": p_elec[sw_].tolist(), "p_bus": p_bus[sw_].tolist(),
            "roll": a["rp"][s, 0].tolist(), "pitch": a["rp"][s, 1].tolist(), "height_mm": (1000 * a["pos"][s, 2]).tolist(),
            "head_acc": comp[sw_].tolist(), "head_acc_le5hz": comp_lo[sw_].tolist(),
            "head_bob_mm": (1000 * bob[sw_]).tolist(),
            "heading": [float(h[0]), float(h[1])]}
    return met, pay


# ---------------------------------------------------------------- worker processes
_W: dict = {}


def _init(ckpt: dict, env_config: dict, net, log_std_init) -> None:
    import torch

    torch.set_num_threads(1)
    _W.update(ckpt=ckpt, ec=env_config, net=net, log_std=log_std_init, envs={}, policy=None,
              stats=gp.load_obs_stats(Path(ckpt["pkl"])), drives=drive_table())


def _env(model_file: str, changes: dict):
    key = (model_file, tuple(sorted(changes.items())))
    if key not in _W["envs"]:
        cfg = gp.TrikeEnvConfig.from_dict({**_W["ec"], "model_path": str(GEN / model_file)}).replace(**changes)
        env = gp.TrikeEnv(cfg)
        assert _W["stats"][0].shape == env.observation_space.shape, (_W["stats"][0].shape, env.observation_space.shape)
        _W["envs"][key] = env
        if _W["policy"] is None:
            _W["policy"] = gp.build_policy(env, Path(_W["ckpt"]["zip"]), _W["net"], _W["log_std"])
            first = _W["policy"].mlp_extractor.policy_net[0].in_features
            assert first == env.observation_space.shape[0], (first, env.observation_space.shape)
    return _W["envs"][key]


def _job(spec: dict) -> dict:
    t0 = time.time()
    env = _env(spec["model"], spec["changes"])
    log = run_episode(env, _W["policy"], _W["stats"], spec["seed"], spec["command"], full=spec["full"])
    log["seed"] = spec["seed"]
    out = {"spec": spec, "fell": log["fell"], "termination": log["termination"], "length_s": log["length_s"],
           "episode_forward_speed_mps": log["episode_forward_speed"], "window_speed_steps_mps": window_speed_from_steps(log),
           "friction": log["friction"]}
    if spec["full"]:
        out["metrics"], out["payload"] = analyse(log, spec["command"], _W["drives"], trace=spec["trace"])
    out["wall_s"] = time.time() - t0
    return out


# ---------------------------------------------------------------- across episodes
def _mean(vals):
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.mean(v)) if v else None


def _max(vals):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return float(max(v)) if v else None


def summarise_case(eps: list[dict], pays: list[dict]) -> dict:
    """Across a case's episodes: means of per-episode figures, the worst episode where the existing analysis reports
    it, and stride-level statistics pooled over all episodes."""
    M = [e for e in eps]
    s: dict = {"episodes": len(M), "falls": int(sum(e["fell"] for e in M))}
    s["pipe1_summary"] = gp.summarise([e["pipe1_measures"] for e in M])
    sp = {k: _mean([e["speed"][k] for e in M]) for k in M[0]["speed"]}
    sp["sideways_drift_m_worst"] = max((e["speed"]["sideways_drift_m"] for e in M), key=abs)
    sp["heading_change_deg_worst"] = max((e["speed"]["heading_change_deg"] for e in M), key=abs)
    s["speed"] = sp
    # footfalls: pooled
    ff: dict = {}
    for k in ("duty_factor", "duty_factor_env_debounce", "duty_factor_raw", "steps_per_s", "touchdowns_per_s",
              "mean_load_bw"):
        ff[k] = {c: _mean([e["footfall"][k][c] for e in M]) for c in CORNERS}
    ff["contact_gaps"] = {c: {b: {"count": int(sum(e["footfall"]["contact_gaps"][c][b]["count"] for e in M)),
                                  "median_lift_mm": _mean([e["footfall"]["contact_gaps"][c][b]["median_lift_mm"] for e in M]),
                                  "max_lift_mm": _max([e["footfall"]["contact_gaps"][c][b]["max_lift_mm"] for e in M])}
                              for b in M[0]["footfall"]["contact_gaps"][c]} for c in CORNERS}
    pool = lambda key: {c: np.concatenate([np.asarray(p[key][c]) for p in pays]) for c in CORNERS}  # noqa: E731
    per, stn, trav = pool("stride_periods_s"), pool("stance_ms"), pool("stance_travel_m")
    ff["stride_period_s_median"] = {c: float(np.median(per[c])) if per[c].size else None for c in CORNERS}
    ff["stride_frequency_hz"] = {c: (1 / v if v else None) for c, v in ff["stride_period_s_median"].items()}
    allp = np.concatenate([per[c] for c in CORNERS])
    ff["stride_frequency_hz_all_feet"] = float(1 / np.median(allp)) if allp.size else None
    ff["irregular_stride_fraction"] = {c: float(np.mean(np.abs(per[c] / np.median(per[c]) - 1) > 0.25))
                                       if per[c].size else None for c in CORNERS}
    ff["stance_ms"] = {c: gc.stats(stn[c]) for c in CORNERS}
    ff["stance_travel_m"] = {c: gc.stats(trav[c]) for c in CORNERS}
    ff["swing_ms"] = {c: _mean([e["footfall"]["swing_ms"][c]["median"] for e in M]) for c in CORNERS}
    sl = np.concatenate([np.asarray(p["stride_length_m"]) for p in pays])
    ff["stride_length_m"] = gc.stats(sl)
    off, lock = {}, {}
    for c in CORNERS[1:]:
        ph = np.concatenate([np.asarray(p["stride_phases"][c]) for p in pays])
        off[c], lock[c] = gc.circular_mean(ph)
    ff["phase_offset_after_LF"] = {c: {"mean_cycles": off[c], "resultant_length": lock[c]} for c in CORNERS[1:]}
    ff["gait"] = gc.classify_gait(off, lock)
    ff["diagonal_timing"] = gc.diagonal_timing(off)
    ff["gait_per_episode"] = [e["footfall"]["gait"]["class"] for e in M]
    ff["feet_down_fraction"] = {k: _mean([e["footfall"]["feet_down_fraction"][k] for e in M])
                                for k in M[0]["footfall"]["feet_down_fraction"]}

    def si(key_l, key_r, val):
        return gc.symmetry_index(val(key_l), val(key_r))

    duty = lambda c: ff["duty_factor"][c]  # noqa: E731
    sms = lambda c: ff["stance_ms"][c]["mean"]  # noqa: E731
    stl = lambda c: ff["stance_travel_m"][c]["mean"]  # noqa: E731
    load = lambda c: ff["mean_load_bw"][c]  # noqa: E731
    ff["symmetry_index_pct"] = {
        name: {"fore (LF vs RF)": si("left_fore", "right_fore", f), "hind (LH vs RH)": si("left_hind", "right_hind", f),
               "sides (LF+LH vs RF+RH)": gc.symmetry_index(f("left_fore") + f("left_hind"),
                                                           f("right_fore") + f("right_hind"))}
        for name, f in (("duty_factor", duty), ("stance_duration", sms), ("stance_length", stl), ("load", load))}
    ff["touchdowns_equal_training_eval"] = all(e["footfall"]["touchdowns_equal_training_eval"] for e in M) \
        if all(e["footfall"]["touchdowns_equal_training_eval"] is not None for e in M) else None
    s["footfall"] = ff
    # weight
    wt = {"share": {c: _mean([e["weight"]["share"][c] for e in M]) for c in CORNERS},
          "front_share": _mean([e["weight"]["front_share"] for e in M]),
          "left_share": _mean([e["weight"]["left_share"] for e in M]),
          "mean_total_load_bw": _mean([e["weight"]["mean_total_load_bw"] for e in M]),
          "peak_load_bw_worst": {c: _max([e["weight"]["peak_load_bw"][c] for e in M]) for c in CORNERS},
          "p99_load_bw_mean": {c: _mean([e["weight"]["p99_load_bw"][c] for e in M]) for c in CORNERS},
          "touch_vs_floor_state_agreement_min": min(min(e["weight"]["touch_vs_floor_state_agreement"].values())
                                                    for e in M),
          "touch_vs_floor_mean_abs_n_max": max(max(e["weight"]["touch_vs_floor_mean_abs_n"].values()) for e in M)}
    prof = np.concatenate([np.asarray(p["stride_load_profiles"]).reshape(-1, 101, 4) for p in pays])
    wt["strides"] = int(prof.shape[0])
    wt["stride_profile_every_5pct"] = {
        c: {"mean": prof[:, ::5, j].mean(0).round(4).tolist(),
            "p10": np.percentile(prof[:, ::5, j], 10, axis=0).round(4).tolist(),
            "p90": np.percentile(prof[:, ::5, j], 90, axis=0).round(4).tolist()} for j, c in enumerate(CORNERS)}
    s["weight"] = wt
    # balance: the headline on step contact, and the env-debounce and raw-contact variants
    for key in ("balance", "balance_env_debounce", "balance_raw_contact"):
        b = {k: _mean([e[key][k] for e in M]) for k in
             ("support_defined_fraction", "no_foot_down_fraction", "cop_margin_negative_fraction",
              "cop_near_edge_fraction", "com_margin_negative_fraction")}
        b["contact"] = M[0][key]["contact"]
        b["com_margin_median_mm"] = _mean([e[key]["com_margin_mm"]["median"] for e in M])
        b["com_margin_p10_mm"] = _mean([e[key]["com_margin_mm"]["p10"] for e in M])
        b["cop_margin_median_mm"] = _mean([e[key]["cop_margin_mm"]["median"] for e in M])
        b["com_margin_negative_fraction_by_feet"] = {
            k: _mean([e[key]["com_margin_negative_fraction_by_feet"][k] for e in M]) for k in ("1", "2", "3", "4")}
        b["feet_down_fraction"] = {k: _mean([e[key]["feet_down_fraction"][k] for e in M]) for k in "01234"}
        s[key] = b
    # drives
    names = list(M[0]["drives"]["per_joint"])
    s["drives"] = {"per_joint": {nm: {k: {"mean": _mean([e["drives"]["per_joint"][nm][k] for e in M]),
                                          "worst": _max([e["drives"]["per_joint"][nm][k] for e in M])}
                                      for k in M[0]["drives"]["per_joint"][nm] if k != "group"}
                                 for nm in names},
                   "groups": s["pipe1_summary"]["drives"]}
    # power
    keys = ("mech_abs_w", "mech_positive_w", "mech_negative_w", "copper_w", "electrical_per_drive_w",
            "electrical_shared_bus_w", "shared_bus_surplus_fraction", "cost_of_transport_mech",
            "cost_of_transport_electrical_per_drive_with_hotel", "cost_of_transport_electrical_shared_bus_with_hotel")
    P = {k: _mean([e["power"][k] for e in M]) for k in keys}
    P["electrical_per_drive_peak_w"] = _max([e["power"]["electrical_per_drive_peak_w"] for e in M])
    P["electrical_shared_bus_peak_w"] = _max([e["power"]["electrical_shared_bus_peak_w"] for e in M])
    P["by_group_w"] = {g: {k: _mean([e["power"]["by_group_w"][g][k] for e in M]) for k in ("positive_mech", "copper")}
                       for g in GROUPS}
    for model in ("per_drive", "shared_bus"):
        rk = f"run_time_min_{model}"
        P[rk] = {k: _mean([e["power"][rk][k] for e in M]) for k in M[0]["power"][rk]}
    s["power"] = P
    # body and head
    s["body"] = {ax: {k: _mean([e["body"][ax][k] for e in M]) for k in M[0]["body"][ax]}
                 for ax in ("roll_deg", "pitch_deg", "height_mm")}
    s["body"]["roll_deg"]["peak_abs_worst"] = _max([e["body"]["roll_deg"]["peak_abs"] for e in M])
    s["body"]["pitch_deg"]["peak_abs_worst"] = _max([e["body"]["pitch_deg"]["peak_abs"] for e in M])
    hd = {k: _mean([e["head"][k] for e in M]) for k, v in M[0]["head"].items() if isinstance(v, (int, float))}
    hd["band_split_hz"] = M[0]["head"]["band_split_hz"]
    for k in ("acc_peak_mps2", "acc_peak_le5hz_mps2", "acc_peak_5_20hz_mps2"):
        hd[k.replace("_mps2", "_mps2_worst")] = _max([e["head"][k] for e in M])
    for k in ("acc_rms_by_axis_mps2", "acc_rms_le5hz_by_axis_mps2", "acc_rms_5_20hz_by_axis_mps2"):
        hd[k] = {ax: _mean([e["head"][k][ax] for e in M]) for ax in ("fore_aft", "lateral", "vertical")}
    s["head"] = hd
    # feet: pooled
    ft: dict = {}
    for key, pk in (("swing_clearance_mm", "clearance_mm"), ("touchdown_speed_mps", "touchdown_speed_mps"),
                    ("peak_load_rate_bw_per_s", "load_rate_bw_per_s")):
        ft[key] = {c: gc.stats(np.concatenate([np.asarray(p[pk][c]) for p in pays])) for c in CORNERS}
    ft["scuff_fraction"] = {c: float(np.mean(np.concatenate([np.asarray(p["clearance_mm"][c]) for p in pays]) < SCUFF_MM))
                            for c in CORNERS}
    ft["impact_peak_bw_median"] = {c: _mean([e["feet"]["impact_peak_bw"][c]["median"] for e in M]) for c in CORNERS}
    ft["stance_slip"] = {c: {"mean_mps": _mean([e["feet"]["stance_slip"][c]["mean_mps"] for e in M]),
                             "p95_mps_worst": _max([e["feet"]["stance_slip"][c]["p95_mps"] for e in M]),
                             "slip_ratio_mean": _mean([e["feet"]["stance_slip"][c]["slip_ratio"] for e in M]),
                             "slip_ratio_worst": _max([e["feet"]["stance_slip"][c]["slip_ratio"] for e in M])}
                         for c in CORNERS}
    ft["ankle_pitch_deg_p1_p99"] = {c: [_mean([e["feet"]["ankle_pitch_deg_p1_p99"][c][0] for e in M]),
                                        _mean([e["feet"]["ankle_pitch_deg_p1_p99"][c][1] for e in M])] for c in CORNERS}
    if "ankle_roll_deg" in M[0]["feet"]:
        ft["ankle_roll_limit_deg"] = M[0]["feet"]["ankle_roll_limit_deg"]
        ft["ankle_roll_deg"] = {c: {"min_worst": min(e["feet"]["ankle_roll_deg"][c]["min"] for e in M),
                                    "max_worst": max(e["feet"]["ankle_roll_deg"][c]["max"] for e in M),
                                    "p1_mean": _mean([e["feet"]["ankle_roll_deg"][c]["p1"] for e in M]),
                                    "p99_mean": _mean([e["feet"]["ankle_roll_deg"][c]["p99"] for e in M]),
                                    "at_limit_fraction_mean": _mean([e["feet"]["ankle_roll_deg"][c]["at_limit_fraction"]
                                                                     for e in M])}
                                for c in CORNERS}
    s["feet"] = ft
    return s


def cross_check(cases: dict) -> dict:
    """This tool's log through gait_pipe1.measures against the committed results of gait_pipe1.py, per episode."""
    if not REFERENCE_JSON.exists():
        return {"available": False}
    ref = json.loads(REFERENCE_JSON.read_text())
    out = {"reference": str(REFERENCE_JSON.relative_to(REPO)), "cases": {}}
    for key, label in REFERENCE_CASES.items():
        if key not in cases or label not in ref["cases"]:
            continue
        rows = ref["cases"][label]["episodes"]
        mine = [e["pipe1_measures"] for e in cases[key]["episodes"]]
        diff: dict = {}

        def upd(name, a, b):
            if a is None or b is None:
                return
            diff[name] = max(diff.get(name, 0.0), abs(float(a) - float(b)))

        for r, m in zip(rows, mine):
            upd("forward_speed_mps", r["forward_speed_mps"], m["forward_speed_mps"])
            upd("sideways_drift_m", r["sideways_drift_m"], m["sideways_drift_m"])
            upd("power_w", r["power_w"], m["power_w"])
            for nm in r["per_joint"]:
                upd("per_joint_rms_nm", r["per_joint"][nm]["rms_nm"], m["per_joint"][nm]["rms_nm"])
                upd("per_joint_at_nominal_limit", r["per_joint"][nm]["sat_nominal"], m["per_joint"][nm]["sat_nominal"])
            for c in CORNERS:
                upd("duty_factor_raw", r["duty_factor"][c], m["duty_factor"][c])
                upd("slip_ratio", r["stance_slip"][c]["slip_ratio"], m["stance_slip"][c]["slip_ratio"])
            upd("roll_rms_deg", r["roll_deg"]["rms"], m["roll_deg"]["rms"])
            upd("pitch_rms_deg", r["pitch_deg"]["rms"], m["pitch_deg"]["rms"])
        rs = ref["cases"][label]["summary"]
        ms = cases[key]["summary"]["pipe1_summary"]
        out["cases"][key] = {"reference_case": label, "episodes_compared": min(len(rows), len(mine)),
                             "max_abs_difference": diff,
                             "reference_speed_mps": rs["forward_speed_mps"], "this_speed_mps": ms["forward_speed_mps"],
                             "reference_power_w": rs["power_w"], "this_power_w": ms["power_w"],
                             "identical": all(v < 1e-9 for v in diff.values())}
    return out


def git_blob_sha256(commit: str, path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(REPO)
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], capture_output=True, check=True).stdout
        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return None


def find_pair(cdir: Path) -> tuple[Path, Path, int]:
    best = None
    for z in cdir.glob("*_steps.zip"):
        m = re.fullmatch(r"(.+)_(\d+)_steps", z.stem)
        if not m:
            continue
        pk = cdir / f"{m.group(1)}_vecnormalize_{m.group(2)}_steps.pkl"
        if pk.exists() and (best is None or int(m.group(2)) > best[2]):
            best = (z, pk, int(m.group(2)))
    if best is None:
        raise SystemExit(f"no checkpoint pair (*_<n>_steps.zip + *_vecnormalize_<n>_steps.pkl) in {cdir}")
    return best


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--run", required=True, help="run name: the results folder (and the chart label unless --label)")
    ap.add_argument("--label", default=None, help="the run's name on the charts and in the Markdown (default: --run)")
    ap.add_argument("--out", default=None, help="output folder (default: results/<run> beside this file)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--cache", default=None, help="also write the chart payload here (JSON) for --charts-from")
    ap.add_argument("--charts-from", default=None, help="redraw charts and Markdown from a cached payload; no runs")
    ap.add_argument("--quick", action="store_true", help="development only: 2 seeds, 10 s episodes, 3 sweep points")
    args = ap.parse_args(argv)
    if args.quick:
        global SEEDS, SWEEP, SWEEP_SEEDS
        SEEDS, SWEEP, SWEEP_SEEDS = SEEDS[:2], (0.0, 0.15, 0.25), SEEDS[:1]
    out = Path(args.out) if args.out else HERE / "results" / args.run
    out.mkdir(parents=True, exist_ok=True)

    if args.charts_from:
        cached = json.loads(Path(args.charts_from).read_text())
        results, payload = cached["results"], cached["payload"]
    else:
        results, payload = simulate(args)
        if args.cache:
            Path(args.cache).write_text(json.dumps({"results": results, "payload": payload}))
    import gait_charts

    results["label"] = args.label or results.get("label") or args.run
    results["charts"] = gait_charts.make_all(results, payload, out)
    # The repository keeps files under 500 KB: gait_report.json holds every summary (per case and the sweep);
    # the per-episode detail goes to gait_report_episodes.json.gz (gzip, written reproducibly with mtime 0).
    full = _round(results)
    summary = {k: v for k, v in full.items() if k != "cases"}
    summary["cases"] = {name: {k: v for k, v in c.items() if k != "episodes"} for name, c in full["cases"].items()}
    summary["episodes_file"] = "gait_report_episodes.json.gz"
    (out / "gait_report.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    import gzip
    episodes = {name: c.get("episodes") for name, c in full["cases"].items()}
    with open(out / "gait_report_episodes.json.gz", "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write((json.dumps(episodes, default=float) + "\n").encode())
    for f in ("gait_report.json", "gait_report_episodes.json.gz"):
        assert (out / f).stat().st_size < 500 * 1024, (f, (out / f).stat().st_size)
    (out / "gait_report.md").write_text(markdown(results))
    for f in results["charts"]:
        size = (out / f).stat().st_size
        assert size < 500 * 1024, (f, size)
    print("wrote", out)
    return 0


def _round(obj, digits: int = 6):
    """Floats to ``digits`` significant digits for the written JSON (every comparison is made before)."""
    if isinstance(obj, float):
        return float(f"{obj:.{digits}g}") if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, np.generic):
        return _round(obj.item(), digits)
    return obj


def simulate(args) -> tuple[dict, dict]:
    cdir = Path(args.ckpt_dir)
    zp, pk, steps = find_pair(cdir)
    record = json.loads((cdir / "run_record.json").read_text())
    st = record["settings"]
    ec = dict(st["env_config"])
    if args.quick:
        ec["episode_seconds"] = 10.0
    drives = drive_table()
    jobs = []
    for key, (label, model, changes, cmd) in CASES.items():
        for s in SEEDS:
            jobs.append({"case": key, "model": model, "changes": changes, "command": cmd, "seed": s, "full": True,
                         "trace": s == SEEDS[0]})
    for c in SWEEP:
        if any(abs(c - CASES[k][3]) < 1e-9 for k in ("target_0.25", "cmd_0.15")):
            continue  # reused from the full cases (same seeds, settings and command)
        for s in SWEEP_SEEDS:
            jobs.append({"case": "sweep", "model": DEFAULT_MODEL, "changes": {}, "command": c, "seed": s, "full": False,
                         "trace": False})
    t0 = time.time()
    init = ({"zip": str(zp), "pkl": str(pk)}, ec, st["net"], st["log_std_init"])
    if args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init, initargs=init) as pool:
            res = pool.map(_job, jobs, chunksize=1)
    else:
        _init(*init)
        res = [_job(j) for j in jobs]
    wall = time.time() - t0
    print(f"{len(jobs)} episodes in {wall:.0f} s")

    env_file = Path(gp.trike_env.__file__).resolve()
    commit = record["sessions"][0]["git"]["commit"]
    results: dict = {
        "run": args.run, "date": args.date, "simulation_only": True,
        "tool": "hardware/triceratops_build/rl/analytics/gait_report.py",
        "policy": {"checkpoint": zp.name, "vecnormalize": pk.name, "steps": steps,
                   "zip_sha256": gp.sha256(zp), "vecnormalize_sha256": gp.sha256(pk), "run_commit": commit,
                   "net": st["net"], "activation": "ELU", "loading": "torch.load(weights_only=True) into a fresh policy; "
                   "restricted unpickler for the normalisation statistics (rl/analysis/gait_pipe1.py)"},
        "environment": {"trike_env_py": (str(env_file.relative_to(REPO)) if env_file.is_relative_to(REPO)
                                         else "TRIKE_ENV_DIR export: " + env_file.name),
                        "trike_env_sha256": gp.sha256(env_file),
                        "run_commit_trike_env_sha256": git_blob_sha256(commit, RL / "trike_env.py"),
                        "from_TRIKE_ENV_DIR": bool(os.environ.get("TRIKE_ENV_DIR")),
                        "env_config": "the run record's settings.env_config, model path re-pointed to sim/generated",
                        "model_sha256": {f: gp.sha256(GEN / f) for f in (DEFAULT_MODEL, ANKLEROLL_MODEL)}},
        "settings": {"seeds": list(SEEDS), "delay_ms": DELAY, "window_from_s": WINDOW_S, "deterministic": True,
                     "episode_s": ec.get("episode_seconds", 20.0), "contact_threshold_n": ec.get("contact_threshold_n"),
                     "debounce_ms": 10, "sweep_commands": list(SWEEP), "sweep_seeds": list(SWEEP_SEEDS),
                     "trace_window_s": list(TRACE_T), "head_lowpass_hz": HEAD_LOWPASS_HZ,
                     "battery_wh": BATTERY_WH, "battery_usable": BATTERY_USABLE, "hotel_w": HOTEL_W,
                     "hotel_range_w": list(HOTEL_RANGE_W), "wall_s": wall, "workers": args.workers},
        "drives": drives, "cases": {}, "sweep": {}}
    payload: dict = {"cases": {}}
    for key, (label, model, changes, cmd) in CASES.items():
        rr = sorted([r for r in res if r["spec"]["case"] == key], key=lambda r: r["spec"]["seed"])
        eps = [r["metrics"] for r in rr]
        pays = [r["payload"] for r in rr]
        results["cases"][key] = {"label": label, "model": model, "changes": changes, "command_mps": cmd,
                                 "summary": summarise_case(eps, pays), "episodes": eps}
        payload["cases"][key] = {"trace": pays[0].get("trace"), "step_t": pays[0]["step_t"], "step_v": pays[0]["step_v"],
                                 "stride_load_profiles": [p["stride_load_profiles"] for p in pays],
                                 "drive_hist": pays, "clearance_mm": [p["clearance_mm"] for p in pays],
                                 "touchdown_speed_mps": [p["touchdown_speed_mps"] for p in pays],
                                 "load_rate_bw_per_s": [p["load_rate_bw_per_s"] for p in pays],
                                 "slip_hist": [p["slip_hist"] for p in pays],
                                 "ankle_roll_deg": [p.get("ankle_roll_deg") for p in pays]}
        # drive histograms summed over episodes (the per-episode payload is dropped)
        hsum = {}
        for g in GROUPS:
            H = sum(np.asarray(p["drive_hist"][g]["counts"]) for p in pays)
            hsum[g] = {"rpm_edges": pays[0]["drive_hist"][g]["rpm_edges"], "nm_edges": pays[0]["drive_hist"][g]["nm_edges"],
                       "counts": H.tolist(), "outside": int(sum(p["drive_hist"][g]["outside"] for p in pays))}
        payload["cases"][key]["drive_hist"] = hsum
    sweep = {}
    for c in SWEEP:
        rows = []
        for s in SWEEP_SEEDS:
            r = next((r for r in res if r["spec"]["seed"] == s and abs(r["spec"]["command"] - c) < 1e-9
                      and r["spec"]["model"] == DEFAULT_MODEL and not r["spec"]["changes"]), None)
            if r is None:
                continue
            wsp = r["metrics"]["speed"]["forward_speed_mps"] if r["spec"]["full"] else r["window_speed_steps_mps"]
            rows.append({"seed": s, "forward_speed_mps": wsp, "episode_forward_speed_mps": r["episode_forward_speed_mps"],
                         "fell": r["fell"], "termination": r["termination"], "length_s": r["length_s"],
                         "friction": r["friction"], "source": "full case" if r["spec"]["full"] else "sweep"})
        sweep[f"{c:.2f}"] = {"command_mps": c, "episodes": rows, "falls": int(sum(x["fell"] for x in rows)),
                             "forward_speed_mps_mean": _mean([x["forward_speed_mps"] for x in rows])}
    results["sweep"] = sweep
    payload["sweep"] = sweep
    results["cross_check"] = cross_check(results["cases"])
    return results, payload


# ---------------------------------------------------------------- Markdown
def _f(x, fmt="{:.2f}", none="-"):
    return none if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def markdown(R: dict) -> str:
    C = R["cases"]
    T, Q, N = C["target_0.25"]["summary"], C["cmd_0.15"]["summary"], C["nominal_0.25"]["summary"]
    A = C["ankleroll_0.25"]["summary"]
    pol = R["policy"]
    run = R.get("label") or R["run"]
    L = [f"# Gait quality of {run} (SIMULATION)", "",
         f"Written by `rl/analytics/gait_report.py` on {R['date']}. Policy `{pol['checkpoint']}` ({pol['steps']:,} steps; "
         f"sha256 {pol['zip_sha256'][:12]}...), trained at commit {pol['run_commit'][:8]}. Deterministic actions, servo "
         f"delay {R['settings']['delay_ms']} ms, seeds {R['settings']['seeds'][0]}-{R['settings']['seeds'][-1]}, "
         f"randomisation and observation noise on unless stated, 20 s episodes measured from t = "
         f"{R['settings']['window_from_s']:.0f} s. One policy: every figure is reported, not judged. Definitions: "
         "`rl/analytics/README.md`. Everything is simulation of a model whose masses are estimates.", ""]
    ff = T["footfall"]
    g = ff["gait"]
    dtm = ff.get("diagonal_timing") or {}
    L += ["## Answer first", "",
          f"- **Speed:** {T['speed']['forward_speed_mps']:.3f} m/s at the 0.25 m/s command and "
          f"{Q['speed']['forward_speed_mps']:.3f} m/s at 0.15; falls {T['falls']} + {Q['falls']} of "
          f"{T['episodes'] + Q['episodes']} (sweep: {sum(v['falls'] for v in R['sweep'].values())} falls in "
          f"{sum(len(v['episodes']) for v in R['sweep'].values())} episodes).",
          f"- **Gait:** {g['class']} ({g['detail']}); a diagonal gait whose RF+LH pair follows the LF+RH pair after "
          f"{dtm.get('pair_interval_LFRH_to_RFLH', float('nan')):.2f} of a stride (0.5 for an even trot); "
          f"stride {ff['stride_frequency_hz_all_feet']:.2f} Hz, "
          f"{100 * ff['stride_length_m']['mean']:.1f} cm per stride; phase after LF: RF "
          f"{ff['phase_offset_after_LF']['right_fore']['mean_cycles']:.2f}, LH "
          f"{ff['phase_offset_after_LF']['left_hind']['mean_cycles']:.2f}, RH "
          f"{ff['phase_offset_after_LF']['right_hind']['mean_cycles']:.2f} of a stride.",
          f"- **Balance** (support polygon of the feet in stance, step contact): the whole-robot centre of mass lies "
          f"outside it {100 * T['balance']['com_margin_negative_fraction']:.1f} % of the time; the centre of "
          f"pressure is within {NEAR_EDGE_M * 1000:.0f} mm of its edge {100 * T['balance']['cop_near_edge_fraction']:.0f} % "
          f"of the time. (Raw 2 N contact, which counts 1-2 ms solver rebounds as lifts, gives "
          f"{100 * T['balance_raw_contact']['com_margin_negative_fraction']:.0f} %.)",
          f"- **Drives:** worst RMS torque / continuous rating {max(v['rms_over_continuous']['mean'] for v in T['drives']['per_joint'].values()):.2f} "
          f"(mean of episodes; {max(T['drives']['per_joint'], key=lambda n: T['drives']['per_joint'][n]['rms_over_continuous']['mean'])}).",
          f"- **Power:** {T['power']['mech_abs_w']:.0f} W joint mechanical (sum |torque x speed|); leg electrical "
          f"estimate {T['power']['electrical_shared_bus_w']:.0f} W on a shared bus to "
          f"{T['power']['electrical_per_drive_w']:.0f} W per drive with no reuse of braking energy; about "
          f"{T['power']['run_time_min_per_drive'][f'hotel_{int(HOTEL_W)}w']:.0f}-"
          f"{T['power']['run_time_min_shared_bus'][f'hotel_{int(HOTEL_W)}w']:.0f} min walking on one battery set with a "
          f"{HOTEL_W:.0f} W hotel load (assumed). Estimates: unmodelled gearbox, iron and driver losses would raise them.",
          f"- **Steadiness:** body roll {T['body']['roll_deg']['sd']:.1f} deg and pitch {T['body']['pitch_deg']['sd']:.1f} deg "
          f"SD, height {T['body']['height_mm']['range_p1_p99']:.0f} mm range; head bob "
          f"{T['head']['bob_mm_p1_p99']:.0f} mm and sway {T['head']['sway_mm_p1_p99']:.0f} mm (1-99 % range) at "
          f"{T['head']['bob_dominant_hz']:.1f} Hz; head acceleration {T['head']['acc_rms_mps2']:.1f} m/s^2 RMS to 20 Hz, of "
          f"which {T['head']['acc_rms_le5hz_mps2']:.1f} m/s^2 at or below 5 Hz (the visible bob, "
          f"{T['head']['acc_rms_le5hz_mps2'] / 9.81:.2f} g) and {T['head']['acc_rms_5_20hz_mps2']:.1f} m/s^2 at 5-20 Hz "
          "(the step impacts' shake).", ""]

    def case_line(key):
        s = C[key]["summary"]
        return (f"| {C[key]['label']} | {s['speed']['forward_speed_mps']:.3f} | {s['falls']} | "
                f"{s['footfall']['gait']['class']} | {s['footfall']['stride_frequency_hz_all_feet']:.2f} | "
                f"{100 * s['footfall']['stride_length_m']['mean']:.1f} | {s['power']['mech_abs_w']:.1f} | "
                f"{s['power']['electrical_shared_bus_w']:.0f}-{s['power']['electrical_per_drive_w']:.0f} | "
                f"{100 * s['balance']['com_margin_negative_fraction']:.1f} % | "
                f"{100 * max(v for v in s['footfall']['irregular_stride_fraction'].values() if v is not None):.0f} % | "
                f"{sum(s['footfall']['contact_gaps'][c][b]['count'] for c in CORNERS for b in ('20-50 ms', '50-80 ms'))} |")

    L += ["## Cases (5 episodes each)", "",
          "| Case | Speed, m/s | Falls | Gait | Stride, Hz | Stride, cm | Mech. power, W | Elec. est. (bus-per drive), W "
          "| CoM outside support (step contact) "
          "| Irregular strides, worst foot | Contact gaps of 20-80 ms (all feet) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + [case_line(k) for k in C] + [
          "", "Irregular strides: step-to-step periods more than 25 % off the median. Gaps of 20-80 ms lie between the "
          "contact-solver rebounds (shorter, sole lift about 1 mm) and real swings (longer); many of them mean the feet "
          "shuffle, and the 50 ms step rule splits them less cleanly (README).", ""]
    # a
    L += ["## a. Footfalls (`footfall.png`)", "",
          "Contact: touch load above 2 N (equal to the floor-truth sole force here). Steps: the env's 10 ms debounce, "
          "then breaks and blips under 50 ms merged (see README). 0.25 m/s, randomised, 5 episodes.", "",
          "| | LF | RF | LH | RH |", "|---|---|---|---|---|",
          "| Duty factor: steps / env debounce / raw | " + " | ".join(
              f"{ff['duty_factor'][c]:.2f} / {ff['duty_factor_env_debounce'][c]:.2f} / {ff['duty_factor_raw'][c]:.2f}"
              for c in CORNERS) + " |",
          "| Steps per s | " + " | ".join(f"{ff['steps_per_s'][c]:.2f}" for c in CORNERS) + " |",
          "| Touchdowns per s, env debounce (training evaluation's count) | " + " | ".join(
              f"{ff['touchdowns_per_s'][c]:.2f}" for c in CORNERS) + " |",
          "| Stride frequency, Hz (median period) | " + " | ".join(f"{_f(ff['stride_frequency_hz'][c])}" for c in CORNERS) + " |",
          "| Stance, ms (median) | " + " | ".join(f"{_f(ff['stance_ms'][c]['median'], '{:.0f}')}" for c in CORNERS) + " |",
          "| Swing, ms (median) | " + " | ".join(f"{_f(ff['swing_ms'][c], '{:.0f}')}" for c in CORNERS) + " |",
          "| Phase after LF, cycles (R) | 0 | " + " | ".join(
              f"{ff['phase_offset_after_LF'][c]['mean_cycles']:.2f} ({ff['phase_offset_after_LF'][c]['resultant_length']:.2f})"
              for c in CORNERS[1:]) + " |", "",
          f"Gait class: **{g['class']}** ({g['detail']}; largest phase error to the nearest template "
          f"{_f(g['distance'])} cycles; tolerance 0.125). Per episode: {', '.join(ff['gait_per_episode'])}. "
          + (f"Diagonal timing: Hildebrand limb phase left {dtm['limb_phase_left']:.2f}, right "
             f"{dtm['limb_phase_right']:.2f} (trot 0.5); each fore foot lands {dtm['fore_after_diagonal_hind_LF_RH']:+.2f} "
             f"(LF after RH) and {dtm['fore_after_diagonal_hind_RF_LH']:+.2f} (RF after LH) cycles after its diagonal hind "
             f"partner; the RF+LH pair follows the LF+RH pair after {dtm['pair_interval_LFRH_to_RFLH']:.2f} of a stride "
             "(0.5 when even). " if dtm else "") +
          f"Stride length {100 * ff['stride_length_m']['mean']:.1f} cm (LF touchdown to LF touchdown). Irregular strides "
          f"(period more than 25 % off the median): " + ", ".join(
              f"{gc.SHORT[c]} {100 * ff['irregular_stride_fraction'][c]:.0f} %" for c in CORNERS) + ".", "",
          "Feet down (steps): " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in ff["feet_down_fraction"].items()) + ".", "",
          "Symmetry index (Robinson, %, 0 = symmetric): " + "; ".join(
              f"{name.replace('_', ' ')}: fore {_f(v['fore (LF vs RF)'], '{:.0f}')}, hind {_f(v['hind (LH vs RH)'], '{:.0f}')}, "
              f"sides {_f(v['sides (LF+LH vs RF+RH)'], '{:.0f}')}" for name, v in ff["symmetry_index_pct"].items()) + ".", ""]
    # b
    wt = T["weight"]
    L += ["## b. Weight distribution (`weight_distribution.png`)", "",
          f"Floor-truth sole normal forces. Load share LF {100 * wt['share']['left_fore']:.0f} %, RF "
          f"{100 * wt['share']['right_fore']:.0f} %, LH {100 * wt['share']['left_hind']:.0f} %, RH "
          f"{100 * wt['share']['right_hind']:.0f} %: front {100 * wt['front_share']:.0f} % / rear "
          f"{100 * (1 - wt['front_share']):.0f} %, left {100 * wt['left_share']:.0f} % / right {100 * (1 - wt['left_share']):.0f} %. "
          "Peak foot load (worst episode, 1 ms) / 99th percentile (mean), x body weight: " + ", ".join(
              f"{gc.SHORT[c]} {wt['peak_load_bw_worst'][c]:.2f} / {wt['p99_load_bw_mean'][c]:.2f}" for c in CORNERS)
          + f". {wt['strides']} strides pooled. Touch sensor and floor truth agree on loaded / unloaded in at least "
          f"{100 * wt['touch_vs_floor_state_agreement_min']:.1f} % of samples.", ""]
    # c
    b = T["balance"]
    L += ["## c. Balance (`balance.png`)", "",
          "Support polygon: hull of the sole corners of the feet in stance on the step contact the footfalls use (the "
          "env's 10 ms debounce, then gaps and blips under 50 ms merged); CoP from those soles' contact forces. "
          f"CoM ground projection outside the polygon {100 * b['com_margin_negative_fraction']:.1f} % of the time (by feet "
          "in stance: " + ", ".join(f"{k}: {_f(100 * v if v is not None else None, '{:.0f}')} %"
                                    for k, v in b["com_margin_negative_fraction_by_feet"].items())
          + "; time with one foot in stance " + f"{100 * b['feet_down_fraction']['1']:.1f} %"
          + f"); CoM margin median {b['com_margin_median_mm']:.0f} mm. CoP outside: "
          f"{100 * b['cop_margin_negative_fraction']:.2f} % (it cannot leave the polygon by construction; a check); CoP "
          f"within 5 mm of the edge {100 * b['cop_near_edge_fraction']:.0f} %, median margin {b['cop_margin_median_mm']:.0f} mm. "
          f"No foot in stance {100 * b['no_foot_down_fraction']:.1f} % of the time.", "",
          "| Contact used for the polygon (0.25 m/s, randomised) | CoM outside | One foot down | CoP within 5 mm of an edge |",
          "|---|---|---|---|"] + [
          f"| {T[k]['contact']} | {100 * T[k]['com_margin_negative_fraction']:.1f} % | "
          f"{100 * T[k]['feet_down_fraction']['1']:.1f} % | {100 * T[k]['cop_near_edge_fraction']:.0f} % |"
          for k in ("balance", "balance_env_debounce", "balance_raw_contact")] + [
          "", "Footnote: with the raw 2 N contact every 1-2 ms solver rebound of one foot of a diagonal pair leaves one foot "
          "'down' and the CoM about 0.2 m outside; that figure (the first version of this report's headline) measures the "
          "contact model, not the gait. Nominal model: " + ", ".join(
              f"{C['nominal_0.25']['summary'][k]['contact'].split(' (')[0].split(':')[0]} "
              f"{100 * C['nominal_0.25']['summary'][k]['com_margin_negative_fraction']:.1f} %"
              for k in ("balance", "balance_env_debounce", "balance_raw_contact")) + ".", ""]
    # d
    L += ["## d. Drives (`drive_envelopes.png`, `drive_loads.png`)", "",
          "| Joint | RMS / continuous, mean (worst) | At nominal limit, mean (worst) | Peak, N m | Peak, rpm | Above maker rated |",
          "|---|---|---|---|---|---|"]
    for nm, v in T["drives"]["per_joint"].items():
        L.append(f"| {nm} | {v['rms_over_continuous']['mean']:.2f} ({v['rms_over_continuous']['worst']:.2f}) | "
                 f"{100 * v['at_nominal_limit']['mean']:.1f} % ({100 * v['at_nominal_limit']['worst']:.1f} %) | "
                 f"{v['peak_nm']['worst']:.1f} | {v['peak_rpm']['worst']:.0f} | {100 * v['above_maker_rated_fraction']['mean']:.1f} % |")
    dr = R["drives"]
    L += ["", "Joint-side figures (N m; flexion joints x 0.90 linkage efficiency): " + "; ".join(
        f"{g}: model limit {v['model_limit_nm']:.1f}, maker peak {v['maker_peak_nm']:.1f}, maker rated "
        f"{v['maker_rated_nm']:.2f}, continuous {v['continuous_nm']:.2f}, speed allowance {v['speed_allowance_rpm']:.0f} rpm"
        for g, v in dr.items()) + ".", ""]
    # e
    L += ["## e. Power and energy (`power.png`)", "",
          "| Case | Sum abs(torque x speed), W | Positive work, W | Copper loss, W | Elec. est., shared bus / per drive, W "
          "| CoT (mech.) | CoT (elec. + hotel), bus / per drive | Run time at 30 W hotel, min, bus / per drive "
          "| Run time, per drive, 25 / 40 W hotel |", "|---|---|---|---|---|---|---|---|---|"]
    for k in C:
        p = C[k]["summary"]["power"]
        rb, rp_ = p["run_time_min_shared_bus"], p["run_time_min_per_drive"]
        L.append(f"| {C[k]['label']} | {p['mech_abs_w']:.1f} | {p['mech_positive_w']:.1f} | {p['copper_w']:.1f} | "
                 f"{p['electrical_shared_bus_w']:.1f} / {p['electrical_per_drive_w']:.1f} | {_f(p['cost_of_transport_mech'])} | "
                 f"{_f(p['cost_of_transport_electrical_shared_bus_with_hotel'])} / "
                 f"{_f(p['cost_of_transport_electrical_per_drive_with_hotel'])} | {rb['hotel_30w']:.0f} / {rp_['hotel_30w']:.0f} | "
                 f"{rp_['hotel_25w']:.0f} / {rp_['hotel_40w']:.0f} |")
    L += ["", "Loss model (stated, simple). Each drive's torque is the joint torque / linkage efficiency (0.90 on the "
          "flexion joints) while it motors and joint torque x efficiency while it brakes; its copper loss is "
          "3 R (tau_drive / Kt)^2, with Kt (output side, N m per A rms) and the per-phase resistance from the T. rex "
          "actuator catalog (X4-36-L 1.8 N m/A, 0.20 ohm; X6-60 2.1 N m/A, 0.205 ohm). **Per drive, no reuse:** each "
          "drive draws its positive mechanical power plus its copper loss; braking energy is lost. **Shared bus:** the 12 "
          "drives' net draw each 1 ms step, max(0, sum of mechanical power at the drive + copper loss), braking power "
          "feeding the other drives and any surplus dumped (the power board's ideal diodes block it from the battery). "
          + "Electrical figures are estimates: unmodelled gearbox, iron and driver losses would raise them; the per-drive figure ignores bus-level reuse of braking energy (which would lower it), the shared-bus figure assumes all of it is reused within the same 1 ms step." + " Not included: the head and tail servos. Hotel load "
          f"{HOTEL_W:.0f} W (range {HOTEL_RANGE_W[0]:.0f}-{HOTEL_RANGE_W[1]:.0f} W) is an assumption, not a parts-list "
          "figure: Jetson Orin Nano Super (7-25 W), OAK-D Pro W, Teensy and CAN, IMU, safety MCU, drive standby, DC-DC "
          f"losses. Battery {BATTERY_WH} Wh x {BATTERY_USABLE:.0%} usable. Torque (copper loss) dominates the electrical "
          "figures.", ""]
    # f
    L += ["## f. Body and head steadiness (`steadiness.png`)", "",
          "| Case | Roll SD, deg | Pitch SD, deg | Height range, mm | Head bob, mm | Head sway, mm | Head acc. RMS to 20 Hz, "
          "m/s^2 | of which <= 5 Hz | of which 5-20 Hz | Head acc. peak, m/s^2 (<= 5 Hz) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in C:
        s = C[k]["summary"]
        L.append(f"| {C[k]['label']} | {s['body']['roll_deg']['sd']:.2f} | {s['body']['pitch_deg']['sd']:.2f} | "
                 f"{s['body']['height_mm']['range_p1_p99']:.1f} | {s['head']['bob_mm_p1_p99']:.1f} | "
                 f"{s['head']['sway_mm_p1_p99']:.1f} | {s['head']['acc_rms_mps2']:.2f} | {s['head']['acc_rms_le5hz_mps2']:.2f} | "
                 f"{s['head']['acc_rms_5_20hz_mps2']:.2f} | {s['head']['acc_peak_mps2_worst']:.1f} "
                 f"({s['head']['acc_peak_le5hz_mps2_worst']:.1f}) |")
    L += ["", f"Head acceleration: the head's (head_yaw body) centre of mass, velocity differentiated, low-passed at "
          f"{HEAD_LOWPASS_HZ:.0f} Hz, gravity excluded, and split at {HEAD_BOB_HZ:.0f} Hz: at or below it the visible bob "
          "and sway, above it the shake of the step impacts (the two bands' mean squares add). Bob and sway: 1-99 % range "
          "after removing a 1 s moving average.", ""]
    # g
    ft = T["feet"]
    L += ["## g. Feet (`feet.png`)", "", "| | LF | RF | LH | RH |", "|---|---|---|---|---|",
          "| Swing clearance, mm, median (10th pct) | " + " | ".join(
              f"{_f(ft['swing_clearance_mm'][c]['median'], '{:.1f}')} ({_f(ft['swing_clearance_mm'][c]['p10'], '{:.1f}')})"
              for c in CORNERS) + " |",
          f"| Swings under {SCUFF_MM:.0f} mm | " + " | ".join(f"{100 * ft['scuff_fraction'][c]:.0f} %" for c in CORNERS) + " |",
          "| Touchdown speed, m/s, median (95th) | " + " | ".join(
              f"{_f(ft['touchdown_speed_mps'][c]['median'])} ({_f(ft['touchdown_speed_mps'][c]['p95'])})" for c in CORNERS) + " |",
          "| Peak load rate, body weights/s, median (95th) | " + " | ".join(
              f"{_f(ft['peak_load_rate_bw_per_s'][c]['median'], '{:.0f}')} ({_f(ft['peak_load_rate_bw_per_s'][c]['p95'], '{:.0f}')})"
              for c in CORNERS) + " |",
          "| Stance slip speed, mean / 95th pct worst, m/s | " + " | ".join(
              f"{_f(ft['stance_slip'][c]['mean_mps'])} / {_f(ft['stance_slip'][c]['p95_mps_worst'])}" for c in CORNERS) + " |",
          "| Slip ratio, mean (worst) | " + " | ".join(
              f"{_f(ft['stance_slip'][c]['slip_ratio_mean'])} ({_f(ft['stance_slip'][c]['slip_ratio_worst'])})" for c in CORNERS) + " |"]
    if "ankle_roll_deg" in A["feet"]:
        ar = A["feet"]["ankle_roll_deg"]
        L += [f"| Roll-ankle angle, deg, min / max (roll-ankle model; limit +-{A['feet']['ankle_roll_limit_deg']:.0f}) | "
              + " | ".join(f"{ar[c]['min_worst']:+.1f} / {ar[c]['max_worst']:+.1f}" for c in CORNERS) + " |"]
    L += [""]
    # h
    L += ["## h. Speed tracking (`speed_tracking.png`)", "",
          "| Command, m/s | " + " | ".join(f"{float(k):.2f}" for k in R["sweep"]) + " |",
          "|---|" + "---|" * len(R["sweep"]),
          "| Speed, m/s (mean of 3 seeds) | " + " | ".join(_f(v["forward_speed_mps_mean"], "{:.3f}") for v in R["sweep"].values()) + " |",
          "| Falls | " + " | ".join(str(v["falls"]) for v in R["sweep"].values()) + " |", "",
          f"At 0.25 m/s: mean |step speed - command| {T['speed']['mean_abs_step_error_mps']:.3f} m/s (control-step speeds "
          f"swing with the stride, SD {T['speed']['step_speed_sd_mps']:.3f} m/s); |mean speed - command| "
          f"{T['speed']['abs_mean_error_mps']:.3f} m/s. Sideways drift worst {T['speed']['sideways_drift_m_worst']:+.2f} m, "
          f"heading change worst {T['speed']['heading_change_deg_worst']:+.1f} deg over 18 s.", ""]
    # cross check
    cc = R["cross_check"]
    L += ["## Cross-check against `rl/analysis/results/gait_pipe1_2026-11-19.json`", "",
          f"Environment: `{R['environment']['trike_env_py']}` "
          f"(sha256 {R['environment']['trike_env_sha256'][:12]}...; the run commit's file: "
          f"{_f(R['environment']['run_commit_trike_env_sha256'], '{}')[:12]}...). This tool's log, passed through "
          "`gait_pipe1.measures`, against the committed per-episode figures:", "",
          "| Case | Episodes | Speed now / then, m/s | Power now / then, W | Largest difference | Identical |", "|---|---|---|---|---|---|"]
    for k, v in cc.get("cases", {}).items():
        big = max(v["max_abs_difference"].items(), key=lambda kv: kv[1]) if v["max_abs_difference"] else ("-", 0)
        L.append(f"| {C[k]['label']} | {v['episodes_compared']} | {v['this_speed_mps']:.4f} / {v['reference_speed_mps']:.4f} | "
                 f"{v['this_power_w']:.2f} / {v['reference_power_w']:.2f} | {big[0]} {big[1]:.1e} | {v['identical']} |")
    L += ["", f"Debounced touchdowns equal `rl/gait_metrics.py`'s count in every episode: "
          f"{all(C[k]['summary']['footfall']['touchdowns_equal_training_eval'] for k in C)}.", ""]
    L += ["## What this does not show", "",
          "Simulation only, of a model whose masses and contact are estimates; one policy at one checkpoint (a verdict "
          "needs two seeds or two controllers). RMS over 18 s stands in for steady walking, not a thermal simulation. "
          "The electrical figures are estimates from a simple loss model (unmodelled gearbox, iron and driver losses would "
          "raise them; the per-drive figure ignores bus-level reuse of braking energy, which would lower it), and the "
          "hotel load is assumed. Head "
          "acceleration is the head body's, not a camera's or a visitor's view of it. Charts: " +
          ", ".join(f"`{c}`" for c in R.get("charts", [])) + ".", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
