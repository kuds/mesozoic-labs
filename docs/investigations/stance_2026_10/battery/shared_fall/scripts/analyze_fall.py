"""Reduce the seed-7047 shared-fall replays to events, a summary JSON/text and a compact figure.

OFFLINE EVALUATION of uncertified scratch-recipe checkpoints; report only; certifies nothing.
Reads replays/ (written by fall_replay.py) and the earlier audit CSVs (cross-check only).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

SP = Path("scratchpad")
OUT = SP / "foot/battery/shared_fall"
REP = OUT / "replays"
HOWCLOSE = SP / "foot/next_signals/howclose"
ROLES = ("s60_6M", "s52_6M", "statue")
YAWS = (-90, -45, 0, 45, 90, 180)

meta = {r: json.loads((REP / f"replay_{r}.json").read_text()) for r in ROLES}
N = meta["statue"]["names"]
Z = {r: dict(np.load(REP / f"trace_{r}_y+0.npz")) for r in ROLES}
H = {r: dict(np.load(REP / f"headings_{r}.npz")) for r in ROLES}
DT = N["control_dt"]
BW = N["body_weight_n"]
ACT = [a.replace("_act", "") for a in N["actuator_names"]]
JN = N["joint_names"]
HOME = np.array(N["home_qpos_j"])
ADR = np.array(N["actuator_qposadr_minus7"])
LEG = ["r_hip_pitch", "r_hip_roll", "r_knee", "r_ankle", "l_hip_pitch", "l_hip_roll", "l_knee", "l_ankle"]


def f(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def step_of(i):
    return None if i is None else int(i) + 1  # 1-based control step (step k ends at t = k * dt)


def first_true(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return None if not idx.size else int(idx[0]) + start


def stays_from(mask, start=0):
    """First index from which mask is True through the end (None if the last entry is False)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.size or not mask[-1]:
        return None
    bad = np.flatnonzero(~mask[start:])
    return start if not bad.size else int(bad[-1]) + 1 + start


# ----------------------------------------------------------------------------- cross-check vs the audit certificates
def audit_rows(label):
    path = HOWCLOSE / f"audit7042_r9armL_{label}_6291456_optoff" / "stance_heading_panel_selected.csv"
    return [r for r in csv.DictReader(open(path)) if r["panel_seed"] == "7047"], str(path)


crosscheck = {}
for label, role in (("s60", "s60_6M"), ("s52", "s52_6M")):
    rows, path = audit_rows(label)
    ep = {round(e["yaw_deg"]): e for e in meta[role]["episodes"]}
    st = {round(e["yaw_deg"]): e for e in meta["statue"]["episodes"]}
    cells = []
    for r in rows:
        yaw = round(float(r["yaw_deg"]))
        mine = ep[yaw] if r["role"] == "policy" else st[yaw]
        cells.append(dict(
            role=r["role"], yaw=yaw, cert_length=int(r["length"]), replay_length=mine["length"],
            cert_reward=float(r["reward"]), replay_reward=mine["reward"],
            cert_max_sole_cop_fore_aft=r["max_sole_cop_fore_aft"],
            replay_max_sole_cop_fore_aft=mine["metrics"].get("max_sole_cop_fore_aft"),
        ))
    crosscheck[role] = dict(
        source=path,
        all_lengths_equal=all(c["cert_length"] == c["replay_length"] for c in cells),
        max_abs_reward_diff=max(abs(c["cert_reward"] - c["replay_reward"]) for c in cells),
        cells=cells,
    )

# ----------------------------------------------------------------------------- heading consistency
headings = {}
for r in ROLES:
    eps = {round(e["yaw_deg"]): e for e in meta[r]["episodes"]}
    ref = H[r]["y+0_com_hf"]
    rows = []
    for y in YAWS:
        tag = f"y{y:+d}"
        c = H[r][f"{tag}_com_hf"]
        n = min(len(c), len(ref))
        mp = H[r][f"{tag}_m_xcom_pad"]
        rows.append(dict(
            yaw_deg=y, length=eps[y]["length"], termination_reason=eps[y]["termination_reason"],
            reward=eps[y]["reward"], max_abs_com_dev_vs_yaw0_m=f(np.abs(c[:n] - ref[:n]).max(), 6),
            xcom_pad_toe_margin_negative_for_good_from_step=step_of(stays_from(mp[:, 1] < 0, 10))
            if eps[y]["length"] < 1000 else None,
            pitch_down_deg_at_end=f(H[r][f"{tag}_pitch_down_deg"][-1], 2),
        ))
    headings[r] = rows

# ----------------------------------------------------------------------------- per-role events (heading 0)
fwd_line = {}
events = {}
for r in ROLES:
    z = Z[r]
    T = len(z["reward"])
    t = (np.arange(T) + 1) * DT
    pc = z["pad_center_hf"].mean(1)[:, 0]  # pad-centre fwd coordinate
    toe_line = z["pad_toe_hf"][:, :, 0].mean(1)
    heel_line = z["pad_heel_hf"][:, :, 0].mean(1)
    tip_d3 = z["digit_tips_hf"][:, [1, 4], 0].mean(1)  # central digit tips (r_d3, l_d3)
    xf = z["xcom_hf"][:, 0]
    cf = z["com_hf"][:, 0]
    vf = z["vcom_hf"][:, 0]
    sole_fore = z["rec_sole_cop"][:, :, 0]
    both_toe = np.all(sole_fore >= 0.99, axis=1)
    foot_cop_norm = (z["rec_foot_cop"] - 0.14) / 0.07  # -1 heel line, +1 toe line of the declared span (0.07..0.21 m)
    m_pad, m_full, m_cont = z["m_xcom_pad"], z["m_xcom_full"], z["m_xcom_contact"]
    m_com = z["m_com_pad"]
    w0 = z["omega0"]
    dq = np.degrees(z["qpos_j"][:, ADR] - HOME[ADR])  # (T, nu) actuated joints vs home
    sup = z["rec_support_floor_mean"]  # (T, 8): r pad,d2,d3,d4, l pad,d2,d3,d4
    pad_load = sup[:, [0, 4]]
    digit_load = np.stack([sup[:, 1:4].sum(1), sup[:, 5:8].sum(1)], axis=1)
    fw = z["fw_hf"]
    shear = np.hypot(fw[:, :, 0], fw[:, :, 1]) / (1.2 * np.maximum(fw[:, :, 2], 1e-9))
    lift = z["rec_sole_corner_lift"]
    sp = z["rec_sole_pitch_deg"]
    W = 10  # skip the spawn touchdown (steps 1-10) for every "from" search
    ev = {}
    ev["length"] = T
    ev["termination_reason"] = meta[r]["episodes"][[round(e["yaw_deg"]) for e in meta[r]["episodes"]].index(0)][
        "termination_reason"]
    ev["spawn"] = dict(com_minus_pad_centre_fwd_m=f(cf[0] - pc[0]), xcom_minus_pad_centre_fwd_m=f(xf[0] - pc[0]),
                       pad_toe_line_minus_pad_centre_m=f(toe_line[0] - pc[0]),
                       com_fwd_velocity_mps=f(vf[0]), omega0_per_s=f(w0[0]))
    ev["pad_cop_at_toe_edge_both_feet_first_step"] = step_of(first_true(both_toe, 2))
    ev["pad_cop_at_toe_edge_both_feet_fraction_steps_11_to_end"] = f(both_toe[W:].mean(), 3)
    ev["pad_cop_at_heel_edge_any_foot_fraction_steps_11_to_end"] = f(np.any(sole_fore[W:] <= -0.99, 1).mean(), 3)
    ev["xcom_vs_pad_toe_line"] = dict(
        first_negative_step=step_of(first_true(m_pad[:, 1] < 0, W)),
        negative_for_good_from_step=step_of(stays_from(m_pad[:, 1] < 0, W)),
        min_m=f(np.nanmin(m_pad[W:, 1])), max_m=f(np.nanmax(m_pad[W:, 1])),
    )
    ev["com_vs_pad_toe_line"] = dict(
        first_negative_step=step_of(first_true(m_com[:, 1] < 0, W)),
        negative_for_good_from_step=step_of(stays_from(m_com[:, 1] < 0, W)),
        min_m=f(np.nanmin(m_com[W:, 1])),
    )
    i_full = stays_from(m_full[:, 0] < 0, W)
    ev["xcom_vs_pad_plus_digit_tip_hull"] = dict(
        first_outside_step=step_of(first_true(m_full[:, 0] < 0, W)),
        outside_for_good_from_step=step_of(i_full),
        min_signed_m=f(np.nanmin(m_full[W:, 0])),
    )
    ev["xcom_vs_loaded_contact_hull"] = dict(
        first_outside_step=step_of(first_true(m_cont[:, 0] < 0, W)),
        outside_for_good_from_step=step_of(stays_from(m_cont[:, 0] < 0, W)),
        min_signed_m=f(np.nanmin(m_cont[W:, 0])),
    )
    # inverted-pendulum check: growth rate of the forward CoM speed after leaving the full hull
    if i_full is not None and T - i_full > 15:
        seg = slice(i_full, T - 3)
        lam = np.polyfit(t[seg], np.log(np.maximum(vf[seg], 1e-6)), 1)[0]
        ev["topple"] = dict(fit_steps=[step_of(i_full), step_of(T - 4)], forward_speed_growth_rate_per_s=f(lam, 3),
                            omega0_mean_per_s=f(w0[seg].mean(), 3),
                            forward_speed_at_hull_exit_mps=f(vf[i_full]), forward_speed_at_end_mps=f(vf[-1]))
    # slow-creep phase rate (steps 31-81)
    ev["creep_steps_31_81"] = dict(
        xcom_fwd_rate_mm_per_step=f((xf[80] - xf[30]) / 50 * 1000, 3) if T > 81 else None,
    )
    psi0 = math.radians(meta[r]["episodes"][[round(e["yaw_deg"]) for e in meta[r]["episodes"]].index(0)]["psi0_deg"])
    fax = np.array([math.cos(psi0), math.sin(psi0)])
    pel = cf + (z["pelvis_pos"][:, :2] - z["com"][:, :2]) @ fax  # pelvis fwd in the heading frame
    ev["pelvis_minus_pad_centre_fwd_m"] = {str(s): f(pel[s - 1] - pc[s - 1]) for s in (1, 21, 41, 61, 81, 101, 121, 141)
                                           if s <= T}
    ev["com_minus_pelvis_fwd_m"] = {str(s): f(cf[s - 1] - pel[s - 1]) for s in (1, 21, 41, 61, 81, 101, 121, 141)
                                    if s <= T}
    ev["heel"] = dict(
        sole_pitch_deg_mean_steps_11_60=[f(v, 2) for v in sp[W:60].mean(0)],
        corner_lift_mm_mean_steps_11_60=[f(v * 1000, 1) for v in lift[W:60].mean(0)],
        corner_lift_mm_end=[f(v * 1000, 1) for v in lift[-1]],
        sole_pitch_deg_closest_to_flat_steps_11_to_end=[f(v, 2) for v in sp[W:].max(0)],
        corner_lift_mm_min_steps_11_to_end=[f(v * 1000, 1) for v in lift[W:].min(0)],
        sole_pitch_deg_spawn=[f(v, 2) for v in sp[0]], corner_lift_mm_spawn=[f(v * 1000, 1) for v in lift[0]],
        sole_pitch_deg_end=[f(v, 1) for v in sp[-1]],
    )
    ev["loads_fraction_BW"] = {
        str(s): dict(pad=[f(v / BW, 3) for v in pad_load[s - 1]], digits=[f(v / BW, 3) for v in digit_load[s - 1]],
                     nonleg=f(z["rec_nonleg_floor_mean"][s - 1] / BW, 3))
        for s in (11, 41, 81, 121, 161, T) if s <= T}
    ev["whole_foot_cop_norm_heel_m1_toe_p1"] = {str(s): [f(v, 3) for v in foot_cop_norm[s - 1]]
                                                for s in (11, 41, 81, 121, 161, T) if s <= T}
    ev["digit_load_position_0mtp_1tip"] = {str(s): [f(v, 3) for v in z["rec_support_load_pos"][s - 1][[1, 2, 3, 5, 6, 7]]]
                                           for s in (41, 121) if s <= T}
    # actuators
    afmax = np.nanmax(z["afrac_max"], 0)
    sat_a = (np.abs(z["action"]) >= 0.99)
    ev["actuators"] = dict(
        max_force_fraction_of_forcerange={a: f(v, 3) for a, v in zip(ACT, afmax)},
        max_force_fraction_leg=f(max(afmax[ACT.index(a)] for a in LEG), 3),
        any_substep_at_force_limit={a: bool(v > 0) for a, v in zip(ACT, np.nanmax(z["at_limit_frac"], 0))},
        action_abs_ge_0p99_fraction={a: f(v, 3) for a, v in zip(ACT, sat_a.mean(0)) if v > 0},
        action_abs_ge_0p99_first_step={a: step_of(first_true(sat_a[:, i])) for i, a in enumerate(ACT) if sat_a[:, i].any()},
        max_abs_action_leg=f(np.abs(z["action"][:, [ACT.index(a) for a in LEG]]).max(), 3),
        action_at_steps={str(s): {a: f(z["action"][s - 1, i], 3) for i, a in enumerate(ACT)} for s in (41, 101, T) if s <= T},
    )
    home_ctrl = np.array(N["home_ctrl"])
    ev["applied_target_deg_vs_home_ctrl"] = {
        str(s): {a: f(np.degrees(z["ctrl"][s - 1, i] - home_ctrl[i]), 2) for i, a in enumerate(ACT)}
        for s in (21, 41, 61, 101) if s <= T}
    cr = np.array(N["ctrlrange"])  # = the joint ranges of the position-servoed joints
    qa = z["qpos_j"][:, ADR]
    near = (qa <= cr[:, 0] + math.radians(0.5)) | (qa >= cr[:, 1] - math.radians(0.5))
    ev["joint_within_0p5deg_of_range_limit_fraction"] = {a: f(v, 3) for a, v in zip(ACT, near.mean(0)) if v > 0}
    toe_idx = [JN.index(j) for j in JN if "toe" in j]
    ev["mtp_deg_vs_home"] = {str(s): {JN[k]: f(np.degrees(z["qpos_j"][s - 1, k] - HOME[k]), 2) for k in toe_idx}
                             for s in (41, 121, T) if s <= T}
    ev["joint_deg_vs_home"] = dict(
        at_steps={str(s): {a: f(dq[s - 1, i], 2) for i, a in enumerate(ACT)} for s in (21, 61, 101, T) if s <= T},
        max_abs_over_episode={a: f(np.abs(dq[:, i]).max(), 2) for i, a in enumerate(ACT)},
    )
    ev["friction"] = dict(
        net_shear_over_mu_normal_max_steps_11_to_end=[f(v, 3) for v in np.nanmax(shear[W:], 0)],
        net_shear_over_mu_normal_mean_steps_11_60=[f(v, 3) for v in shear[W:60].mean(0)],
        forceweighted_contact_cone_use_max_steps_11_to_end=[f(v, 3) for v in np.nanmax(z["ratio_mean"][W:], 0)],
        foot_site_slide_mm_step11_to_end=[f(v * 1000, 1) for v in
                                          np.linalg.norm(z["rec_foot_pos"][-1, :, :2] - z["rec_foot_pos"][W, :, :2], axis=1)],
    )
    d = z["com_hf"][-1, :2] - z["com_hf"][W, :2]
    ev["fall_direction"] = dict(
        com_displacement_fwd_lat_m_step11_to_end=[f(d[0]), f(d[1])],
        direction_deg_from_heading=f(math.degrees(math.atan2(d[1], d[0])), 1),
        pitch_down_deg_end=f(z["pitch_down_deg"][-1], 2), roll_deg_end=f(z["roll_deg"][-1], 2),
        yaw_change_deg_end=f(z["yaw_change_deg"][-1], 2), com_height_m_end=f(z["com_hf"][-1, 2], 3),
        head_tip_z_m_end=f(z["head_tip_z"][-1], 3), forward_z_end=f(z["forward_z"][-1], 4),
        nosedive_forward_z_threshold=f(N["natural_forward_z"] - N["nosedive_threshold"], 4),
    )
    if r == "statue":
        ev["statue_extremes_steps_11_to_1000"] = dict(
            xcom_pad_signed_margin_min_m=f(np.nanmin(m_pad[W:, 0])),
            xcom_pad_toe_margin_min_m=f(np.nanmin(m_pad[W:, 1])),
            xcom_pad_heel_margin_min_m=f(np.nanmin(m_pad[W:, 2])),
            pitch_down_deg_range=[f(z["pitch_down_deg"][W:].min(), 2), f(z["pitch_down_deg"][W:].max(), 2)],
            pad_cop_fore_range=[f(np.nanmin(sole_fore[W:]), 3), f(np.nanmax(sole_fore[W:]), 3)],
            xcom_fwd_rel_pad_centre_range_m=[f((xf - pc)[W:].min()), f((xf - pc)[W:].max())],
        )
    events[r] = ev
    fwd_line[r] = dict(t=t, pc=pc, toe=toe_line, heel=heel_line, tip=tip_d3, xf=xf, cf=cf, vf=vf, pel=pel,
                       sole_fore=sole_fore.mean(1), fcop=np.nanmean(foot_cop_norm, 1), pad=pad_load.sum(1) / BW,
                       dig=digit_load.sum(1) / BW, mfull=m_full[:, 0], mpad=m_pad[:, 1], pitch=z["pitch_down_deg"],
                       head=z["head_tip_z"], afleg=z["afrac_max"][:, [ACT.index(a) for a in LEG]].max(1),
                       neck_a=z["action"][:, ACT.index("neck_pitch")], sp=sp.mean(1), shear=shear.max(1),
                       hip=dq[:, [ACT.index("r_hip_pitch"), ACT.index("l_hip_pitch")]].mean(1),
                       i_padtoe=-1 if ev["xcom_vs_pad_toe_line"]["negative_for_good_from_step"] is None
                       else ev["xcom_vs_pad_toe_line"]["negative_for_good_from_step"] - 1,
                       i_full=-1 if i_full is None else i_full, dq=dq)

# ----------------------------------------------------------------------------- kinematic attribution (feet held fixed)
import environments  # noqa: E402

assert "main_ro" in environments.__file__, environments.__file__
import mujoco  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.species_registry import env_class as model_env_class  # noqa: E402

MID = ModelId("tyrannosaurus_rex", "anatomical")
env = model_env_class(MID)(**dict(load_stage_config(MID, 1)["env_kwargs"]))
m = env.unwrapped.model
key = int(getattr(env.unwrapped, "_reset_keyframe_id", 0))
env.close()
d = mujoco.MjData(m)
root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
soles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("r_plantar_geom", "l_plantar_geom")]


def pose(qfull):
    d.qpos[:] = qfull
    d.qvel[:] = 0
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    return (d.subtree_com[root].copy(), [d.geom_xpos[s].copy() for s in soles],
            [d.geom_xmat[s].reshape(3, 3).copy() for s in soles])


# The authored home stance itself
mujoco.mj_resetDataKeyframe(m, d, key)
mujoco.mj_forward(m, d)
com_h, ps_h, Rs_h = pose(d.qpos.copy())
padc_h = 0.5 * (ps_h[0] + ps_h[1])
home_static = dict(
    keyframe="home", com_minus_pad_centre_fwd_m=f(com_h[0] - padc_h[0]),
    pad_toe_line_minus_pad_centre_m=f(0.07),
    com_toe_margin_m=f(padc_h[0] + 0.07 - com_h[0]), com_height_m=f(com_h[2] - min(p[2] for p in ps_h) + 0.01, 3),
    note="world x = animal forward at the home keyframe; pad half-length 0.07 m",
)

GROUPS = {
    "hip_pitch": ["r_hip_pitch", "l_hip_pitch"], "hip_roll": ["r_hip_roll", "l_hip_roll"],
    "knee": ["r_knee", "l_knee"], "ankle": ["r_ankle", "l_ankle"],
    "toes(passive)": [j for j in JN if "toe" in j],
    "neck_head": ["neck_pitch", "neck_yaw", "head_pitch"],
    "tail": ["tail_1_pitch", "tail_1_yaw", "tail_2_pitch", "tail_3_pitch"],
}
GROUPS["all_leg_joints"] = GROUPS["hip_pitch"] + GROUPS["hip_roll"] + GROUPS["knee"] + GROUPS["ankle"] + GROUPS["toes(passive)"]
GROUPS["all_joints"] = list(JN)


def attribution(z, s):
    """CoM fwd shift (m) that each joint group's deviation from home causes at step s, each foot held where it is."""
    i = s - 1
    q = np.r_[z["root_qpos"][i], z["qpos_j"][i]].astype(np.float64)
    com0, ps0, Rs0 = pose(q)
    out = {}
    for g, joints in GROUPS.items():
        qc = q.copy()
        for j in joints:
            k = JN.index(j)
            qc[7 + k] = HOME[k]
        comc, psc, Rsc = pose(qc)
        shifts = []
        for fi in range(2):  # anchor each foot at its actual pose; the CoM offset is carried in that foot's frame
            rel = Rsc[fi].T @ (comc - psc[fi])
            com_cf = ps0[fi] + Rs0[fi] @ rel
            shifts.append(com0[0] - com_cf[0])
        out[g] = f(float(np.mean(shifts)), 4)
    return out


attrib = {}
for r in ROLES:
    T = len(Z[r]["reward"])
    attrib[r] = {str(s): attribution(Z[r], s) for s in (21, 61, 101) if s <= T}

# ----------------------------------------------------------------------------- context: audit panel at heading 0 (if run)
context = None
CTX = OUT / "context"
if (CTX / "replay_ctx_statue.json").exists():
    context = {}
    for r in ("ctx_s60_6M", "ctx_s52_6M", "ctx_statue"):
        js = json.loads((CTX / f"replay_{r}.json").read_text())
        npz = np.load(CTX / f"headings_{r}.npz")
        rows = []
        for e in js["episodes"]:
            s = e["seed"]
            tag = f"s{s}_y+0"
            sc = npz[f"{tag}_rec_sole_cop"][:, :, 0]
            mp = npz[f"{tag}_m_xcom_pad"][:, 1]
            mf = npz[f"{tag}_m_xcom_full"][:, 0]
            xf = npz[f"{tag}_xcom_hf"][:, 0]
            pcx = npz[f"{tag}_pad_center_hf"].mean(1)[:, 0]
            win = slice(10, 200)  # steps 11-200: the settle window, where both 7047 falls are decided
            rows.append(dict(
                seed=s, length=e["length"], reason=e["termination_reason"],
                spawn_xcom_minus_pad_centre_m=f(xf[0] - pcx[0]),
                spawn_mean_sole_pitch_deg=f(np.nanmean(npz[f"{tag}_rec_sole_pitch_deg"][0]), 2),
                mean_xcom_minus_pad_centre_m_steps_11_200=f((xf - pcx)[win].mean()),
                min_xcom_pad_toe_margin_m_steps_11_200=f(np.nanmin(mp[win])),
                min_xcom_full_hull_margin_m_steps_11_200=f(np.nanmin(mf[win])),
                pad_cop_both_toe_edge_frac_steps_11_200=f(np.all(sc[win] >= 0.99, 1).mean(), 3),
                mean_sole_pitch_deg_steps_11_200=f(np.nanmean(npz[f"{tag}_rec_sole_pitch_deg"][win]), 2),
            ))
        context[r] = dict(source=str(CTX / f"replay_{r}.json"), rows=rows)

# ----------------------------------------------------------------------------- write
summary = dict(
    label="OFFLINE EVALUATION of uncertified scratch-recipe checkpoints (r9 arm L symmetry screen, 6,291,456 steps); "
          "report only; certifies nothing",
    seed=7047, sources={r: str(REP / f"replay_{r}.json") for r in ROLES},
    traces={r: str(REP / f"trace_{r}_y+0.npz") for r in ROLES},
    checkpoints={r: dict(model=meta[r]["model"], model_sha256=meta[r]["model_sha256"], vecnorm=meta[r]["vecnorm"],
                         vecnorm_sha256=meta[r]["vecnorm_sha256"]) for r in ROLES},
    conventions=dict(
        heading_frame="fwd = the root's heading after the spawn turn, lat = its left; origin = spawn midpoint of the "
                      "two plantar pads; all margins in m",
        xcom="CoM_xy + v_xy / omega0, omega0 = sqrt(9.81 / CoM height above the floor), boundary state after each step",
        pad_hull="convex hull of the floor-facing corners of each LOADED foot's plantar pad (the declared heel line "
                 "0.07 m and toe line 0.21 m from the ankle axis)",
        full_hull="pad hull plus the tips of the loaded feet's three digit capsules",
        contact_hull="convex hull of the contact points carrying > 0.1 N at the step's last substep",
        toe_margin="front edge of the hull at the XCoM's lateral coordinate minus the XCoM's fwd coordinate "
                   "(negative = XCoM ahead of the toe line)",
        signed_margin="distance to the hull boundary, + inside, - outside",
        pad_cop="recorder sole_cop: force-weighted pad CoP / pad half-length, +1 = pad toe edge, -1 = heel edge",
        whole_foot_cop="recorder foot_cop normalised on the declared span: -1 heel line, +1 toe line (beyond +1 = "
                       "load on the digits ahead of the pad)",
        actuator_force_fraction="|actuator_force| / forcerange bound, max over the 5 substeps of a step",
        step="1-based control step; step k ends at t = k * 0.01 s; steps 1-200 are the gate's settle window",
    ),
    crosscheck_vs_audit_certificates=crosscheck,
    headings=headings,
    events=events,
    home_keyframe_static=home_static,
    kinematic_attribution_com_fwd_shift_m=dict(
        definition="CoM forward position minus the CoM with that joint group set to its home value, each foot held at "
                   "its actual pose (mean of the two feet); + means the group's deviation from home carries the CoM "
                   "forward. Kinematic only (no re-balancing); the closed chain is broken for leg groups.",
        by_role_step=attrib,
    ),
    context_audit_panel_heading0=context,
)
(OUT / "shared_fall_summary.json").write_text(json.dumps(summary, indent=1, default=str))
np.savez_compressed(OUT / "shared_fall_series.npz", **{
    f"{r}_{k}": np.asarray(v, dtype=np.float32) for r, dct in fwd_line.items() for k, v in dct.items()})
print("wrote", OUT / "shared_fall_summary.json")

# ----------------------------------------------------------------------------- figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"s60_6M": "#2a78d6", "s52_6M": "#eb6834", "statue": "#52514e"}
LAB = {"s60_6M": "s60 6M (falls, head contact)", "s52_6M": "s52 6M (falls, nosedive)", "statue": "statue (stands)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
TMAX = 2.5
DRAW = ("statue", "s52_6M", "s60_6M")  # statue first so the policies sit on top where they overlap
fig, axes = plt.subplots(5, 2, figsize=(11.5, 15), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#fcfcfb")


def lines(ax, key, ylab, title, ref=None, mark=None):
    for r in DRAW:
        s = fwd_line[r]
        k = s["t"] <= TMAX
        y = np.asarray(s[key] if not callable(key) else key(s))
        ax.plot(s["t"][k], y[k], color=COL[r], lw=1.6, label=LAB[r])
        if mark and r != "statue" and s[mark] >= 0:
            i = int(s[mark])
            ax.plot(s["t"][i], y[i], "o", ms=6, color=COL[r], mec="#fcfcfb", mew=1.5, zorder=5)
    if ref:
        for val, txt in ref:
            ax.axhline(val, color=MUTED, lw=0.9, ls="--", zorder=0)
            ax.annotate(txt, (TMAX, val), xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                        color=MUTED, fontsize=7.5)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=9, color=INK)


ax = axes[0, 0]
lines(ax, lambda s: s["xf"] - s["pc"], "m ahead of pad centre",
      "XCoM fore-aft vs support lines (dot: XCoM past pad toe line for good)", mark="i_padtoe")
st = fwd_line["statue"]
for val, txt, ls in ((np.mean(st["toe"] - st["pc"]), "pad toe line", "--"),
                     (np.mean(st["heel"] - st["pc"]), "pad heel line", "--"),
                     (np.mean(st["tip"] - st["pc"]), "central digit tips", ":")):
    ax.axhline(val, color=MUTED, lw=0.9, ls=ls, zorder=0)
    ax.annotate(txt, (TMAX, val), xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom", color=MUTED,
                fontsize=7.5)
ax.set_ylim(-0.12, 0.45)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=7.5)

lines(axes[0, 1], "mfull", "m (+ inside)", "XCoM signed margin to pad + digit-tip hull (dot: leaves it for good)",
      ref=[(0.0, "hull edge")], mark="i_full")
axes[0, 1].set_ylim(-0.4, 0.2)
lines(axes[1, 0], "sole_fore", "pad CoP / half-length", "Pad CoP fore-aft (mean of feet): +1 toe edge, -1 heel edge",
      ref=[(1.0, "toe edge"), (-1.0, "heel edge")])
axes[1, 0].set_ylim(-1.15, 1.25)
lines(axes[1, 1], "dig", "fraction of body weight", "Digit load, both feet (the pads carry the rest)")
lines(axes[2, 0], lambda s: s["pel"] - s["pc"], "m ahead of pad centre", "Pelvis fore-aft over the pads")
lines(axes[2, 1], "sp", "deg (toe up +)", "Pad pitch (mean of feet): negative = heel raised", ref=[(0.0, "flat")])
axes[2, 1].set_ylim(-25, 3)
lines(axes[3, 0], "pitch", "deg nose-down", "Trunk pitch")
lines(axes[3, 1], "hip", "deg vs home", "Hip pitch vs home (mean of legs; + = leg back under the pelvis)",
      ref=[(0.0, "home")])
lines(axes[4, 0], "afleg", "|force| / forcerange", "Largest leg-actuator force fraction (none near its limit)",
      ref=[(1.0, "force limit")])
axes[4, 0].set_ylim(0, 1.1)
lines(axes[4, 1], "neck_a", "action", "Neck-pitch command (+ = nose down); the only command at its bound",
      ref=[(1.0, "bound")])
axes[4, 1].set_ylim(-0.3, 1.1)
for a in axes[4]:
    a.set_xlabel("time since reset (s)")
for a in axes.ravel():
    a.axvline(2.0, color=GRID, lw=1.2, zorder=0)
axes[0, 0].annotate("settle window ends", (2.0, 0.40), xytext=(3, 0), textcoords="offset points", color=MUTED,
                    fontsize=7.5)
fig.suptitle("Seed 7047, heading 0 (the six headings repeat it): s60 and s52 6M fall forward, the statue stands.\n"
             "Offline evaluation of uncertified scratch-recipe checkpoints; certifies nothing.", fontsize=9.5,
             color=INK, x=0.01, ha="left")
fig.savefig(OUT / "shared_fall_7047.png", dpi=100, facecolor=fig.get_facecolor())
print("wrote", OUT / "shared_fall_7047.png")
