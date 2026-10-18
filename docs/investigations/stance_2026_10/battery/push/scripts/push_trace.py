"""Push-and-release telemetry: main's impulse probe rollout, replicated step for step, with read-only balance telemetry.

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen). Report only; certifies
nothing; no training.

Rollout = environments/shared/reporting/stance_report.py run_panel/_roll_episodes with an impulse, line for line:
one env per (role, impulse row) panel built from main's stance stage config (options off), validated against the
current plant identity; env.reset(seed=first_seed + index); deterministic policy via stance_report._load_policy
(frozen VecNormalize) or stance_report._zero_action_predict (the statue); at the control boundary where
steps == impulse.step, stance_report._apply_root_impulse(env, impulse) (qvel[0:3] += delta_v) BEFORE the action is
computed from the pre-impulse observation, exactly as main does. Nothing in the rollout draws a random number after
reset; the impulse is a deterministic velocity change.

Telemetry, all read-only on env.data:
  * main's SubstepContactRecorder (floor truth: per-foot pad CoP sole_cop, whole-foot CoP, foot sites, leg loads);
  * a substep hook chained under it that reads contact normal forces from efc_force (elliptic cone: the normal is
    efc_force[efc_address]) and positions, for the net floor CoP;
  * whole-body CoM position and velocity from a SCRATCH MjData (qpos/qvel copied in, mj_kinematics + mj_comPos +
    mj_jacSubtreeCom), so env.data is never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--vecnorm", default=None)
ap.add_argument("--speeds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,1.0,2.0")
ap.add_argument("--dirs", default="fwd,back,left,right")
ap.add_argument("--episodes", type=int, default=16)
ap.add_argument("--first-seed", type=int, default=None, help="default: stance_report.PUBLICATION_SEED_START")
ap.add_argument("--out", required=True)
ap.add_argument("--series-steps", type=int, default=300, help="post-impulse steps kept per episode in the series npz "
                "(the battery's main runs used the default 300; added afterwards for the single-episode XCoM check)")
args = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(1)
import mujoco  # noqa: E402

import environments  # noqa: E402

assert "main_ro" in environments.__file__, environments.__file__
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.gait.constants import CONTACT_THRESHOLD_N  # noqa: E402
from environments.shared.gait.morphology import Morphology  # noqa: E402
from environments.shared.gait.recorder import SubstepContactRecorder, _digit_axis_ends, _sole_corners  # noqa: E402
from environments.shared.gait.stance_metrics import whole_foot_cop  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity, validate_environment_plant  # noqa: E402
from environments.shared.reporting.stance_report import (  # noqa: E402
    PUBLICATION_SEED_START,
    RootImpulse,
    _apply_root_impulse,
    _load_policy,
    _zero_action_predict,
)
from environments.shared.species_registry import env_class as model_env_class  # noqa: E402

G = 9.81
V_TOLS = (0.02, 0.03, 0.05)  # m/s: horizontal CoM speed counted as settled (primary 0.03)
V_TOL = 0.03
HOLD = 50  # control steps (0.5 s) the settle condition must hold
WIN = 100  # control steps (1 s) after the impulse for the windowed peaks
SERIES = args.series_steps  # control steps after the impulse kept per episode in the series npz (default 300)
PRE = 20  # control steps before the impulse averaged for the pre-impulse CoP
DIRS = {"fwd": ((1.0, 0.0), "forward +x"), "back": ((-1.0, 0.0), "backward -x"),
        "left": ((0.0, 1.0), "lateral +y"), "right": ((0.0, -1.0), "lateral -y")}
DIR_KEYS = ("fwd", "back", "left", "right")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


MID = ModelId("tyrannosaurus_rex", "anatomical")
CFG = load_stage_config(MID, 1)
CUR = CFG["curriculum_kwargs"]
assert CUR["gate_kind"] == "stance_quality_heading/v1"
ENV_KW = dict(CFG["env_kwargs"])
SETTLE = int(CUR["settle_steps"])
HORIZON = int(ENV_KW.get("max_episode_steps", 1000))
ENV_CLASS = model_env_class(MID)
PLANT = current_plant_identity(MID)
FIRST = PUBLICATION_SEED_START if args.first_seed is None else args.first_seed

if args.model is None:
    predict = _zero_action_predict(ENV_CLASS, ENV_KW)
    policy_note = "zero-action statue"
else:
    predict, policy_note = _load_policy(args.model, args.vecnorm, lambda: ENV_CLASS(**ENV_KW), plant_identity=PLANT)


# ----------------------------------------------------------------------------- convex geometry (world xy)
def hull(points):
    pts = sorted(set(map(tuple, np.round(np.asarray(points, dtype=np.float64), 12))))
    if len(pts) < 3:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)  # counter-clockwise


def halfplanes(poly):
    """Outward normals n_i and offsets c_i of a CCW convex polygon: inside iff n_i . x <= c_i for all i."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    e = b - a
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n, np.einsum("ij,ij->i", n, a)


def ray_margin(hp, q, u):
    """Signed distance along unit u from q to the polygon boundary (the exit point); + when q is inside.

    For q inside: the distance the point can travel along u before leaving. For q past an edge it is negative.
    """
    if hp is None or not np.all(np.isfinite(q)):
        return float("nan")
    n, c = hp
    nu = n @ u
    slack = c - n @ q
    pos = nu > 1e-12
    if not pos.any():
        return float("nan")
    return float(np.min(slack[pos] / nu[pos]))


def signed_dist(hp, q):
    """Signed Euclidean distance from q to a convex polygon's boundary (+ inside; - outside, a lower bound)."""
    if hp is None or not np.all(np.isfinite(q)):
        return float("nan")
    n, c = hp
    slack = c - n @ q
    return float(slack.min())


# ----------------------------------------------------------------------------- read-only substep hook (net CoP)
class NetCopProbe:
    def __init__(self, env, morph):
        u = env.unwrapped
        self.u = u
        m = u.model
        assert int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC), "net CoP decode assumes the elliptic cone"
        self.is_floor = np.asarray(morph.is_floor, dtype=bool)
        self.is_animal = np.asarray(morph.is_animal, dtype=bool)
        self.geom_foot = np.asarray(morph.geom_foot, dtype=np.int64)
        self.nf = morph.n_feet
        self.rows = []

    def __call__(self):
        d = self.u.data
        ncon = int(d.ncon)
        fn = np.zeros(self.nf)
        net = np.zeros(3)
        if ncon:
            con = d.contact
            g = np.asarray(con.geom)[:ncon]
            f1 = self.is_floor[g[:, 0]]
            f2 = self.is_floor[g[:, 1]]
            other = np.where(f1, g[:, 1], g[:, 0])
            keep = (f1 != f2) & self.is_animal[other]
            if keep.any():
                addr = np.asarray(con.efc_address)[:ncon][keep]
                efc = np.asarray(d.efc_force)
                normal = np.where(addr >= 0, efc[np.maximum(addr, 0)], 0.0)
                pos = np.asarray(con.pos)[:ncon][keep]
                net = np.array([normal.sum(), (normal * pos[:, 0]).sum(), (normal * pos[:, 1]).sum()])
                foot = self.geom_foot[other[keep]]
                for fi in range(self.nf):
                    fn[fi] = normal[foot == fi].sum()
        self.rows.append((net, fn))

    def close_step(self, frame_skip):
        rows = self.rows[-frame_skip:]
        self.rows = []
        net = np.sum([r[0] for r in rows], axis=0)
        fn = np.mean([r[1] for r in rows], axis=0)
        cop = net[1:] / net[0] if net[0] > 1e-9 else np.array([np.nan, np.nan])
        return cop, net[0] / len(rows), fn


# ----------------------------------------------------------------------------- one panel (one impulse row)
def first_settle(ok, start, length, hold):
    """First index k >= start with ok[k:k+hold] all True and k+hold <= length; None if none."""
    run = 0
    for k in range(start, length):
        run = run + 1 if ok[k] else 0
        if run >= hold:
            return k - hold + 1
    return None


def panel(impulse, dkey):
    env = ENV_CLASS(**ENV_KW)
    try:
        validate_environment_plant(env, PLANT, artifact=f"{MID} push-and-release telemetry environment")
    except BaseException:
        env.close()
        raise
    u = env.unwrapped
    m = u.model
    morph = Morphology.from_env(env, MID)
    probe = NetCopProbe(env, morph)
    u._substep_probe_hook = probe  # the recorder chains it (calls it first) and restores it on detach
    recorder = SubstepContactRecorder(env, morph).attach()
    scratch = mujoco.MjData(m)
    root = int(morph.root_body)
    jac = np.zeros((3, m.nv))
    corners_local = [_sole_corners(m, f) for f in morph.feet]
    digit_ids = np.array([g for g, j in zip(morph.support_geom_ids, morph.support_geom_mtp) if j >= 0], dtype=np.int64)
    digit_foot = np.array([morph.geom_foot[g] for g in digit_ids], dtype=np.int64)
    _, digit_tip = _digit_axis_ends(m, digit_ids, [j for j in morph.support_geom_mtp if j >= 0])
    episodes = []
    series = []
    try:
        for index in range(args.episodes):
            seed = FIRST + index
            obs, _ = env.reset(seed=seed)
            reset_predict = getattr(predict, "reset", None)
            if callable(reset_predict):
                reset_predict()
            rec = {k: [] for k in ("com", "vcom", "cop", "fz", "fn", "hp_pad", "hp_full", "loaded")}
            pre = {}
            total, steps = 0.0, 0
            terminated = truncated = False
            reason = ""

            def kin():
                scratch.qpos[:] = u.data.qpos
                scratch.qvel[:] = u.data.qvel
                mujoco.mj_kinematics(m, scratch)
                mujoco.mj_comPos(m, scratch)
                mujoco.mj_jacSubtreeCom(m, scratch, jac, root)
                return scratch.subtree_com[root].copy(), jac @ scratch.qvel

            def polygons(loaded):
                pad_pts, tip_pts = [], []
                for fi, f in enumerate(morph.feet):
                    if not loaded[fi] or corners_local[fi] is None:
                        continue
                    R = scratch.geom_xmat[f.sole_geom].reshape(3, 3)
                    c0 = scratch.geom_xpos[f.sole_geom]
                    pad_pts.append((c0 + corners_local[fi] @ R.T)[:, :2])
                for gi, g in enumerate(digit_ids):
                    if loaded[digit_foot[gi]]:
                        Rg = scratch.geom_xmat[g].reshape(3, 3)
                        tip_pts.append((scratch.geom_xpos[g] + Rg[:, 2] * digit_tip[gi])[:2])
                pad = hull(np.concatenate(pad_pts)) if pad_pts else np.zeros((0, 2))
                full = hull(np.concatenate(pad_pts + [np.array(tip_pts).reshape(-1, 2)])) if pad_pts else pad
                return (halfplanes(pad) if pad.shape[0] >= 3 else None,
                        halfplanes(full) if full.shape[0] >= 3 else None)

            while True:
                if impulse is not None and steps == impulse.step:
                    com0, v0 = kin()
                    loaded = rec["fn"][-1] > CONTACT_THRESHOLD_N
                    hp_pad, hp_full = polygons(loaded)
                    l0 = float(com0[2])
                    w0 = math.sqrt(G / l0)
                    xcom0 = com0[:2] + v0[:2] / w0
                    cop_pre = np.nanmean(np.array(rec["cop"][-PRE:]), axis=0)
                    pre = dict(com=com0, vcom=v0, l=l0, omega0=w0, xcom=xcom0, cop=cop_pre,
                               b={k: {"cop_pad": ray_margin(hp_pad, cop_pre, np.array(DIRS[k][0])),
                                      "xcom_pad": ray_margin(hp_pad, xcom0, np.array(DIRS[k][0])),
                                      "com_pad": ray_margin(hp_pad, com0[:2], np.array(DIRS[k][0])),
                                      "cop_full": ray_margin(hp_full, cop_pre, np.array(DIRS[k][0])),
                                      "xcom_full": ray_margin(hp_full, xcom0, np.array(DIRS[k][0]))}
                                  for k in DIR_KEYS},
                               hp_pad=hp_pad, hp_full=hp_full)
                    _apply_root_impulse(env, impulse)
                    _, v1 = kin()
                    pre["dv_applied"] = (v1 - v0).tolist()
                action = np.asarray(predict(obs), dtype=np.float64).ravel()
                obs, reward, terminated, truncated, info = env.step(action)
                recorder.end_step(action, float(reward))
                cop, fz, fn = probe.close_step(u.frame_skip)
                com, vcom = kin()
                rec["com"].append(com)
                rec["vcom"].append(vcom)
                rec["cop"].append(cop)
                rec["fz"].append(fz)
                rec["fn"].append(fn)
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    reason = info.get("termination_reason", "terminated" if terminated else "truncated")
                    trace = recorder.end_episode(terminated=bool(terminated), truncated=bool(truncated),
                                                 termination_reason=str(reason))
                    break
            episodes.append(reduce_episode(seed, steps, total, terminated, truncated, reason, rec, pre, trace,
                                           impulse, dkey, morph))
            series.append(episode_series(rec, trace, pre, dkey))
    finally:
        try:
            recorder.detach()
        finally:
            if "_substep_probe_hook" in vars(u):
                delattr(u, "_substep_probe_hook")
            env.close()
    return episodes, stack_series(series)


def reduce_episode(seed, steps, total, terminated, truncated, reason, rec, pre, trace, impulse, dkey, morph):
    com = np.asarray(rec["com"])
    vcom = np.asarray(rec["vcom"])
    cop = np.asarray(rec["cop"])
    vh = np.linalg.norm(vcom[:, :2], axis=1)
    k0 = SETTLE  # record index k describes the state after control step k+1; index SETTLE = first post-impulse step
    survived = bool(truncated and steps == HORIZON)
    row = dict(seed=seed, length=steps, reward=total, survived=survived, terminated=bool(terminated),
               reason=str(reason), time_to_fall_s=None if survived else (steps - k0) * 0.01)
    # pad CoP from main's recorder (fore-aft +, outer +) / half-size, per foot; unloaded steps are NaN
    sc = np.asarray(trace.sole_cop, dtype=np.float64)
    loaded = np.isfinite(sc[..., 0]) & (np.asarray(trace.sole_contacts_mean) > 0)
    fa = np.where(loaded, np.abs(sc[..., 0]), np.nan)
    lat = np.where(loaded, np.abs(sc[..., 1]), np.nan)
    spanned, wcop, wloaded = whole_foot_cop(trace)
    wcop = np.where(wloaded, wcop, np.nan)
    leg_max = np.asarray(trace.leg_floor_max, dtype=np.float64)
    foot_pos = np.asarray(trace.foot_pos, dtype=np.float64)

    def nanmax(a):
        a = np.asarray(a, dtype=np.float64)
        return None if not np.isfinite(a).any() else float(np.nanmax(a))

    def nanmin(a):
        a = np.asarray(a, dtype=np.float64)
        return None if not np.isfinite(a).any() else float(np.nanmin(a))

    pre_win = slice(max(0, k0 - PRE), k0)
    row["pre_vh_max"] = nanmax(vh[pre_win])
    row["pre_pad_cop_fore_aft_max"] = nanmax(fa[pre_win])
    row["pre_pad_cop_outer_max"] = nanmax(lat[pre_win])
    wpost = slice(k0, min(steps, k0 + WIN))
    row["win_pad_cop_fore_aft_max"] = nanmax(fa[wpost])
    row["win_pad_cop_outer_max"] = nanmax(lat[wpost])
    row["win_vh_max"] = nanmax(vh[wpost])
    for direction, (unit, _label) in DIRS.items():
        if pre and steps > k0:
            row.setdefault("win_cop_peak_disp_m", {})[direction] = nanmax((cop[wpost] - pre["cop"]) @ np.array(unit))
    if impulse is None or impulse.speed == 0:
        post = slice(k0, steps)
        row["post_vh_p50"] = None if steps <= k0 else float(np.percentile(vh[post], 50))
        row["post_vh_p99"] = None if steps <= k0 else float(np.percentile(vh[post], 99))
        row["post_vh_max"] = nanmax(vh[post])
        row["post_pad_cop_fore_aft_max"] = nanmax(fa[post])
        row["post_pad_cop_outer_max"] = nanmax(lat[post])
        row["post_whole_foot_cop_max"] = nanmax(wcop[post])
        row["post_whole_foot_cop_min"] = nanmin(wcop[post])
        return row
    u = np.array(DIRS[dkey][0])
    post = slice(k0, steps)
    n_post = steps - k0
    row["omega0"] = pre["omega0"]
    row["com_height_m"] = pre["l"]
    row["b"] = pre["b"][dkey]
    row["b_all_dirs"] = pre["b"]
    row["pred_dv_cop_pad"] = pre["omega0"] * pre["b"][dkey]["cop_pad"]
    row["pred_dv_xcom_pad"] = pre["omega0"] * pre["b"][dkey]["xcom_pad"]
    row["pred_dv_cop_full"] = pre["omega0"] * pre["b"][dkey]["cop_full"]
    row["pred_dv_xcom_full"] = pre["omega0"] * pre["b"][dkey]["xcom_full"]
    row["dv_applied"] = pre["dv_applied"]
    row["pre_vcom"] = pre["vcom"].tolist()
    # CoM motion after the shove
    disp = (com[post, :2] - pre["com"][:2]) @ u
    row["com_peak_disp_along_push_m"] = nanmax(disp)
    row["com_final_disp_along_push_m"] = float(disp[-1]) if n_post else None
    row["post_vh_max"] = nanmax(vh[post])
    # recovery: horizontal CoM speed below V_TOL for HOLD consecutive steps, entirely inside the episode
    for tol in V_TOLS:
        k = first_settle(vh < tol, k0, steps, HOLD)
        row[f"recovery_time_s_v{tol:g}"] = None if (k is None or not survived) else (k - k0 + 1) * 0.01
    row["recovery_time_s"] = row[f"recovery_time_s_v{V_TOL:g}"]
    # net CoP excursion along the push direction (m) relative to the pre-impulse mean, and its margin to the pad hull
    cdisp = (cop[post] - pre["cop"]) @ u
    row["cop_peak_disp_along_push_m"] = nanmax(cdisp)
    wdisp = (cop[wpost] - pre["cop"]) @ u
    row["win_cop_peak_disp_along_push_m"] = nanmax(wdisp)
    row["win_cop_peak_fraction_of_b_pad"] = (None if row["win_cop_peak_disp_along_push_m"] is None
                                             or not pre["b"][dkey]["cop_pad"] > 0
                                             else row["win_cop_peak_disp_along_push_m"] / pre["b"][dkey]["cop_pad"])
    row["cop_peak_fraction_of_b_pad"] = (None if row["cop_peak_disp_along_push_m"] is None
                                         or not pre["b"][dkey]["cop_pad"] > 0
                                         else row["cop_peak_disp_along_push_m"] / pre["b"][dkey]["cop_pad"])
    hp = pre["hp_pad"]
    cop_margin = np.array([ray_margin(hp, c, u) for c in cop[post]]) if hp is not None else np.array([np.nan])
    row["cop_min_margin_pad_m"] = nanmin(cop_margin)
    # XCoM against the PRE-IMPULSE pad hull (feet planted; a step moves the real polygon, see foot displacement)
    xcom = com[post, :2] + vcom[post, :2] / pre["omega0"]
    xm = np.array([ray_margin(hp, x, u) for x in xcom]) if hp is not None else np.array([np.nan])
    row["xcom_min_margin_pad_m"] = nanmin(xm)
    row["xcom_outside_pad_steps"] = int(np.sum(xm < 0)) if np.isfinite(xm).any() else None
    hpf = pre["hp_full"]
    xmf = np.array([ray_margin(hpf, x, u) for x in xcom]) if hpf is not None else np.array([np.nan])
    row["xcom_min_margin_full_m"] = nanmin(xmf)
    # per-foot pad CoP (main's recorder): peaks after the shove, time on the edge
    row["post_pad_cop_fore_aft_max"] = nanmax(fa[post])
    row["post_pad_cop_outer_max"] = nanmax(lat[post])
    edge = np.nanmax(np.where(np.isfinite(fa[post]), fa[post], 0.0), axis=1) >= 0.95 if n_post else np.array([])
    row["post_pad_cop_edge_fraction"] = float(edge.mean()) if n_post else None
    row["post_whole_foot_cop_max"] = nanmax(wcop[post])
    row["post_whole_foot_cop_min"] = nanmin(wcop[post])
    # cert-style window mean over the first second after the shove (|fore-aft|, unloaded = 1), max over feet
    w = slice(k0, min(steps, k0 + 100))
    if steps >= k0 + 100:
        fw = np.where(np.isfinite(fa[w]), fa[w], 1.0)
        row["post_1s_window_mean_pad_cop_fore_aft_max"] = float(fw.mean(axis=0).max())
    else:
        row["post_1s_window_mean_pad_cop_fore_aft_max"] = None
    # pad CoP recovery: max over feet |fore-aft| back under max(0.9, pre max) for HOLD steps
    thr = max(0.9, row["pre_pad_cop_fore_aft_max"] or 0.0)
    okc = np.nan_to_num(np.nanmax(np.where(np.isfinite(fa), fa, 1.0), axis=1), nan=1.0) <= thr
    kc = first_settle(okc, k0, steps, HOLD)
    row["pad_cop_recovery_time_s"] = None if (kc is None or not survived) else (kc - k0 + 1) * 0.01
    # stepping: any lift-off of a leg after the shove, and how far each foot site moved horizontally
    lift = leg_max[post] <= CONTACT_THRESHOLD_N
    row["liftoff_steps_per_foot"] = lift.sum(axis=0).astype(int).tolist()
    ref = foot_pos[k0 - 1, :, :2]
    row["foot_max_horiz_disp_m"] = np.max(np.linalg.norm(foot_pos[post, :, :2] - ref, axis=2), axis=0).tolist()
    row["stepped"] = bool(lift.any() and max(row["foot_max_horiz_disp_m"]) > 0.02)
    return row


def episode_series(rec, trace, pre, dkey):
    """Per-episode series for re-derivation: vh from the impulse to the end; the rest over SERIES steps after it."""
    k0 = SETTLE
    steps = len(rec["com"])
    com = np.asarray(rec["com"])
    vcom = np.asarray(rec["vcom"])
    cop = np.asarray(rec["cop"])
    vh = np.full(HORIZON - k0, np.nan)
    if steps > k0:
        vh[: steps - k0] = np.linalg.norm(vcom[k0:, :2], axis=1)
    n = max(0, min(steps, k0 + SERIES) - k0)
    out = {"vh": vh}
    com_d = np.full((SERIES, 2), np.nan)
    cop_d = np.full((SERIES, 2), np.nan)
    pad = np.full((SERIES, 2, 2), np.nan)
    xm = np.full(SERIES, np.nan)
    xs = np.full(SERIES, np.nan)
    if pre and n:
        com_d[:n] = com[k0:k0 + n, :2] - pre["com"][:2]
        cop_d[:n] = cop[k0:k0 + n] - pre["cop"]
        sc = np.asarray(trace.sole_cop, dtype=np.float64)
        loaded = np.isfinite(sc[..., 0]) & (np.asarray(trace.sole_contacts_mean) > 0)
        pad[:n] = np.where(loaded[k0:k0 + n, :, None], sc[k0:k0 + n], np.nan)
        if dkey is not None and pre.get("hp_pad") is not None:
            u = np.array(DIRS[dkey][0])
            xcom = com[k0:k0 + n, :2] + vcom[k0:k0 + n, :2] / pre["omega0"]
            xm[:n] = [ray_margin(pre["hp_pad"], x, u) for x in xcom]
            xs[:n] = [signed_dist(pre["hp_pad"], x) for x in xcom]
    out.update(com_dxy=com_d, cop_dxy=cop_d, pad_cop=pad, xcom_margin_pad=xm)
    if args.series_steps != 300:  # the check runs only; the main runs' series files keep their original keys
        out["xcom_signed_dist_pad"] = xs
    return out


def stack_series(series):
    return {k: np.stack([s[k] for s in series]).astype(np.float32) for k in series[0]} if series else {}


# ----------------------------------------------------------------------------- the sweep
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
speeds = sorted({float(s) for s in args.speeds.split(",") if float(s) > 0})
dirs = [d for d in args.dirs.split(",") if d]
rows_spec = [(None, "control", 0.0)] + [(d, d, s) for s in speeds for d in dirs]
summary = dict(
    label="OFFLINE EVALUATION of an uncertified scratch-recipe checkpoint; report only, certifies nothing",
    role=args.role, policy=policy_note, environments_file=environments.__file__,
    model=args.model, model_sha256=None if args.model is None else sha(args.model),
    vecnorm=args.vecnorm, vecnorm_sha256=None if args.vecnorm is None else sha(args.vecnorm),
    env_overrides={}, settle_steps=SETTLE, impulse_step=SETTLE, horizon=HORIZON, first_seed=FIRST,
    episodes_per_row=args.episodes,
    conventions=dict(
        directions={k: {"unit_xy": v[0], "axis_label": v[1]} for k, v in DIRS.items()},
        recovery_time="seconds from the impulse to the first step from which horizontal CoM speed stays < "
                      f"{V_TOL} m/s (also reported at {V_TOLS}) for {HOLD} consecutive steps (0.5 s) inside a "
                      "surviving episode; None if never",
        windowed="win_* keys: peaks over the first WIN=100 control steps (1 s) after the impulse; the control row "
                 "carries the same window as the no-push baseline",
        series_npz="push_series_<role>.npz: per row <dir>_<speed>_<key>, episodes stacked in seed order: vh "
                   "(E, 800) from the impulse; com_dxy, cop_dxy (E, 300, 2) from the pre-impulse CoM / mean CoP; "
                   "pad_cop (E, 300, 2 feet, fore-aft/outer) main's recorder, NaN unloaded; xcom_margin_pad (E, 300)",
        pad_cop_recovery_time="seconds until max over feet |pad CoP fore-aft| (main's recorder, half-lengths, "
                              f"unloaded = 1) stays <= max(0.9, pre-impulse {PRE}-step max) for {HOLD} steps",
        b="ray distance along the push direction from the pre-impulse point (CoP: mean net floor CoP over the "
          f"{PRE} steps before the impulse; XCoM/CoM: at the impulse boundary) to the edge of the convex hull of the "
          "loaded pads' bottom corners (pad) or pads + loaded digit tips (full)",
        pred_dv="omega0 * b, omega0 = sqrt(g / CoM height) at the impulse boundary",
        survived="episode reached the horizon (1000 steps), the probe's full-horizon criterion",
    ),
    rows=[],
)
compact = {}
t0 = time.time()
for dkey, tag, speed in rows_spec:
    t1 = time.time()
    if dkey is None:
        impulse = RootImpulse(step=SETTLE, delta_v=(0.0, 0.0, 0.0), axis_label="none (control)")
    else:
        ux, uy = DIRS[dkey][0]
        impulse = RootImpulse(step=SETTLE, delta_v=(speed * ux, speed * uy, 0.0), axis_label=DIRS[dkey][1])
    eps, trace = panel(impulse, dkey)
    row = dict(direction=tag, speed=speed, impulse=impulse.as_dict(), episodes=eps,
               survived=sum(e["survived"] for e in eps), n=len(eps),
               length_mean=float(np.mean([e["length"] for e in eps])), seconds=time.time() - t1)
    summary["rows"].append(row)
    for k, v in trace.items():
        compact[f"{tag}_{speed:g}_{k}"] = v
    print(json.dumps({"role": args.role, "dir": tag, "speed": speed, "survived": row["survived"], "n": row["n"],
                      "len": row["length_mean"], "s": round(row["seconds"], 1)}), flush=True)
    (out / f"push_{args.role}.json").write_text(json.dumps(summary, indent=1, default=float))
np.savez_compressed(out / f"push_series_{args.role}.npz", **compact)
summary["seconds"] = time.time() - t0
(out / f"push_{args.role}.json").write_text(json.dumps(summary, indent=1, default=float))
print("done", args.role, round(summary["seconds"], 1))
