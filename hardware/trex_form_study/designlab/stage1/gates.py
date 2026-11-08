"""Stage 1 fail-fast gates G1 (plant) and G2 (references), ``STAGE1_PLAN.md`` section 3, steps 2 and 3.

G1, the plant:

* the hashes match (``plant.verify`` on the model the plant loads; ``refs.source`` on the rebuilt one);
* with zero command (every servo at its home target, no feedforward) the body stands ``STAND_S`` on hard floor and
  on low pile: no fall (``plant.FALL_*``) and a trunk tilt from home that stays below 12 deg
  (``carpet.UPRIGHT_TILT_DEG``, the Stand behaviour's success limit);
* the tendon map equals ``sizing._motor_space`` to ``TENDON_TOL`` on random torques and speeds, through the
  compiled model's own transmission (motor torques from the sizing map, applied as raw motor torques, give back
  the joint torques; joint speeds give the sizing's motor speeds), and the ankle reaches, but never exceeds,
  ``|tau_p| + |tau_r| = 2 peak``;
* seeded runs repeat exactly: two fresh builds with the same seed (random start noise and random servo targets
  drawn from the plant's generator) give bit-identical traces, and another seed does not;
* the plant (envelope included) runs at least ``MIN_STEPS_PER_S`` physics steps per second per core with zero
  command on hard floor, alone and with one process per core.

G2, the references:

* every reference matches its records to 6 digits (``refs.compare``, recorded in ``refs.json``);
* the feedforward replay of the nominal walk on the plant with the envelope walks at least ``MIN_REPLAY_STRIDES``
  strides.  The replay is the reference with feedforward and servo PD only (R0, no feedback): every 2 ms the servos
  get the reference joint angles and rates and the inverse-dynamics torque plus the MJCF's passive joint terms
  (damping x qd_ref + frictionloss x tanh(qd_ref / 0.05)), from the reference's first frame (already walking),
  until a fall or ``REPLAY_STRIDES``.  It is run with and without the envelope; the other references are replayed
  the same way and reported, not gated.  The design study's replay on the screened MJCF (joint servos, no
  envelope) lasted 13.6 / 16.6 / 60 strides (``DESIGN_STUDY_STRIDES``; 60 = no fall in 60).

Reported with G2, not gated: the open-loop replay has no feedback, so on a marginally stable body its fall time
depends on tiny differences in the start.  ``replay_spread`` repeats the nominal replay from ``SPREAD_RUNS`` start
states whose root velocity is offset by at most ``SPREAD_OFFSET_M_S`` (seeded); and the nominal replay is also run
on the plant built with ``ankle_gains="joint"`` (the screened model's joint-space ankle stiffness), the one plant
choice the spread was found to depend on.  G2 itself is the unperturbed run on the default plant, as the plan
states.

Nothing is tuned to pass a gate: every number comes from the plant and references as built.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet, statics
from . import drive, plant
from .refs import Reference

STAND_S = 20.0
STAND_FLOORS = ("hard", "low")
TENDON_TOL = 1e-9
TENDON_SAMPLES = 500
REPEAT_S = 2.0
MIN_STEPS_PER_S = 20_000
RATE_STEPS = 20_000
REPLAY_STRIDES = 60.0
MIN_REPLAY_STRIDES = 10.0
DESIGN_STUDY_STRIDES = {"60-N-T-10": 13.6, "60-N-H-10": 16.6, "75-M-H-10": 60.0}
FRICTION_SMOOTH = 0.05  # rad/s, the frictionloss compensation's tanh width
SPREAD_RUNS = 10
SPREAD_OFFSET_M_S = 1e-3
SPREAD_SEED = 0


# ------------------------------------------------------------------------------------------------
# G1
# ------------------------------------------------------------------------------------------------
def stand(body: str, floor: str, seconds: float = STAND_S) -> dict:
    """Zero-command stance from the settled home pose."""
    t0 = time.perf_counter()
    p = plant.build(body, floor)
    build_s = time.perf_counter() - t0
    m, d = p.m, p.d
    p.reset()
    p.hold()
    root = p.trunk
    com0 = d.subtree_com[root].copy()
    n = int(round(seconds / m.opt.timestep))
    tilt_max, z_min, fell_at = 0.0, float(d.qpos[2]), None
    sat = np.zeros(m.nu, bool)
    t1 = time.perf_counter()
    for k in range(n):
        p.step()
        sat |= np.abs(d.actuator_force) >= m.actuator_forcerange[:, 1] - 1e-9
        tilt_max = max(tilt_max, p.tilt_from_home())
        z_min = min(z_min, float(d.qpos[2]))
        if p.fallen():
            fell_at = float(d.time)
            break
    wall = time.perf_counter() - t1
    tilt_max_deg = float(np.degrees(tilt_max))
    com = d.subtree_com[root]
    return dict(floor=floor, seconds=float(d.time), fell_at_s=fell_at,
                stands=fell_at is None and tilt_max_deg < carpet.UPRIGHT_TILT_DEG, tilt_from_home_max_deg=tilt_max_deg,
                tilt_from_home_end_deg=float(np.degrees(p.tilt_from_home())), root_z_min_m=z_min,
                com_drift_mm=1e3 * float(np.linalg.norm(com[:2] - com0[:2])), com_drop_mm=1e3 * float(com0[2] - com[2]),
                foot_load_end_N=[float(v) for v in p.foot_loads()],
                saturated=[p.drives[i].actuator for i in np.flatnonzero(sat)], build_s=build_s, wall_s=wall,
                steps_per_s=(k + 1) / wall, settle=p.meta["settle"],
                floor_info=p.meta["floor_info"])


def tendon_map(body: str, samples: int = TENDON_SAMPLES, seed: int = 0) -> dict:
    """The compiled transmission against ``sizing._motor_space`` (module docstring)."""
    p = plant.build(body, "hard", settle=False)
    m, d = p.m, p.d
    rng = np.random.default_rng(seed)
    peak = np.array([dr.peak for dr in p.drives])
    err_tau = err_w = err_tau_G = err_w_G = 0.0
    m.actuator_gainprm[:, 0] = 1.0  # raw motor torques: force = ctrl
    m.actuator_biasprm[:] = 0.0
    m.actuator_forcelimited[:] = 0
    for _ in range(samples):
        tau = rng.uniform(-1.0, 1.0, len(p.joints)) * peak
        omega = rng.uniform(-10.0, 10.0, len(p.joints))
        f_s, w_s = drive.sizing_motor_values(p.joints, tau, omega)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = p.q_home
        d.qpos[p.qadr] += rng.uniform(-0.2, 0.2, len(p.joints))
        d.qvel[p.dofs] = omega
        d.ctrl[:] = f_s
        mujoco.mj_forward(m, d)
        err_tau = max(err_tau, float(np.max(np.abs(d.qfrc_actuator[p.dofs] - tau))))
        err_w = max(err_w, float(np.max(np.abs(d.actuator_velocity - w_s))))
        err_tau_G = max(err_tau_G, float(np.max(np.abs(drive.motor_torques(p.G, tau) - f_s))))
        err_w_G = max(err_w_G, float(np.max(np.abs(p.G @ omega - w_s))))
    # the linkage limit: motors at +-peak (forces clamped by the force range) give at most 2 peak on one axis
    m.actuator_forcelimited[:] = 1
    m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1] = -peak, peak
    ankle = {}
    for side in statics.SIDES:
        ia, ib = (p.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll"))
        worst = 0.0
        for _ in range(samples):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = p.q_home
            d.ctrl[:] = rng.uniform(-10.0, 10.0, m.nu) * peak
            mujoco.mj_forward(m, d)
            worst = max(worst, abs(d.qfrc_actuator[p.dofs[ia]]) + abs(d.qfrc_actuator[p.dofs[ib]]))
        corners = []
        for ca, cb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = p.q_home
            d.ctrl[ia], d.ctrl[ib] = 10 * ca * peak[ia], 10 * cb * peak[ib]
            mujoco.mj_forward(m, d)
            corners.append([float(d.qfrc_actuator[p.dofs[ia]]), float(d.qfrc_actuator[p.dofs[ib]])])
        ankle[side] = dict(peak_Nm=float(peak[ia]), max_abs_pitch_plus_abs_roll_Nm=float(worst),
                           corners_pitch_roll_Nm=corners,
                           within=bool(worst <= 2 * peak[ia] + TENDON_TOL),
                           reaches=bool(all(abs(abs(a) + abs(b) - 2 * peak[ia]) <= TENDON_TOL for a, b in corners)))
    ok = max(err_tau, err_w, err_tau_G, err_w_G) <= TENDON_TOL and all(v["within"] and v["reaches"]
                                                                        for v in ankle.values())
    return dict(samples=samples, tolerance=TENDON_TOL, max_err_joint_torque_Nm=err_tau, max_err_motor_speed_rad_s=err_w,
                max_err_plant_matrix_torque_Nm=err_tau_G, max_err_plant_matrix_speed_rad_s=err_w_G, ankle_limit=ankle,
                matches=bool(ok))


def _trace_hash(body: str, seed: int, seconds: float = REPEAT_S) -> str:
    """A fresh build, a random start and random servo targets (50 Hz) from the plant's seeded generator."""
    p = plant.build(body, "hard", seed=seed)
    p.reset(joint_noise=0.01, root_velocity_noise=0.02)
    d = p.d
    h = hashlib.sha256()
    q0 = p.q_home[p.qadr]
    for k in range(int(round(seconds / p.m.opt.timestep))):
        if k % 10 == 0:
            p.command(q0 + p.rng.normal(0.0, 0.02, len(q0)))
        p.step()
        for arr in (d.qpos, d.qvel, d.sensordata, d.actuator_force):
            h.update(arr.tobytes())
    return h.hexdigest()


def repeatability(body: str, seed: int = 1) -> dict:
    a, b, c = _trace_hash(body, seed), _trace_hash(body, seed), _trace_hash(body, seed + 1)
    return dict(seconds=REPEAT_S, seed=seed, trace_sha256=a, repeat_sha256=b, other_seed=seed + 1, other_sha256=c,
                repeats=a == b, other_seed_differs=a != c, ok=a == b and a != c)


def _rate(body: str, steps: int = RATE_STEPS) -> float:
    p = plant.build(body, "hard")
    p.reset()
    p.hold()
    p.step(1000)
    t0 = time.perf_counter()
    p.step(steps)
    return steps / (time.perf_counter() - t0)


def step_rate(body: str, steps: int = RATE_STEPS, procs: int | None = None) -> dict:
    """Plant steps per second (zero command, hard floor, envelope on): one process, then one per core at once."""
    procs = procs or os.cpu_count() or 1
    single = _rate(body, steps)
    with mp.get_context("spawn").Pool(procs) as pool:
        par = pool.starmap(_rate, [(body, steps)] * procs)
    per_core = float(min(single, np.mean(par)))
    return dict(steps=steps, single_process=single, processes=procs, per_process=par, per_process_mean=float(np.mean(par)),
                per_core=per_core, threshold=MIN_STEPS_PER_S, ok=per_core >= MIN_STEPS_PER_S)


def g1(body: str, out_dir: Path | None = None) -> dict:
    """G1; with ``out_dir`` (``refs.prepare`` output) the rebuilt model's hash check is included as well."""
    t0 = time.perf_counter()
    xml, _ = plant.model_files(body)
    hashes = dict(plant_model=plant.verify(body, xml))
    path = Path(out_dir) / "refs.json" if out_dir is not None else None
    if path is not None and path.exists():
        hashes["reference_model"] = json.loads(path.read_text())["model"]
    out = dict(hashes=hashes)
    out["stand"] = {f: stand(body, f) for f in STAND_FLOORS}
    out["tendon_map"] = tendon_map(body)
    out["repeatability"] = repeatability(body)
    out["step_rate"] = step_rate(body)
    out["pass"] = bool(all(h["matches_record"] and h["matches_plan"] for h in hashes.values())
                       and all(s["stands"] for s in out["stand"].values()) and out["tendon_map"]["matches"]
                       and out["repeatability"]["ok"] and out["step_rate"]["ok"])
    out["seconds"] = time.perf_counter() - t0
    return out


# ------------------------------------------------------------------------------------------------
# G2
# ------------------------------------------------------------------------------------------------
def _yaw(R) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def replay(p: plant.Plant, ref: Reference, strides: float = REPLAY_STRIDES, start_velocity_offset=None) -> dict:
    """Feedforward replay (module docstring) of ``ref`` on ``p``; ``start_velocity_offset`` (m/s, world) is added
    to the root velocity of the start frame."""
    if list(ref.joints) != p.joints:
        raise ValueError("reference and plant joints differ")
    m, d, env = p.m, p.d, p.envelope
    series = ref.joint_series(p.qadr, p.dofs)
    damp, fric = m.dof_damping[p.dofs].copy(), m.dof_frictionloss[p.dofs].copy()
    qd0 = ref.qd[0].copy()
    if start_velocity_offset is not None:
        qd0[:3] += np.asarray(start_velocity_offset, float)
    p.set_state(ref.q[0], qd0)
    x0 = d.qpos[:2].copy()
    R0 = d.xmat[p.trunk].reshape(3, 3).copy()
    n = int(round(strides * ref.period / m.opt.timestep))
    ratio_max = np.zeros(m.nu)
    sat_steps = np.zeros(m.nu)
    past = 0
    fell_at = None
    yaw_prev, yaw_turned = _yaw(R0), 0.0
    t0 = time.perf_counter()
    k = 0
    for k in range(n):
        q, qd, tau = ref.at(d.time, series)
        p.command(q, qd, tau + damp * qd + fric * np.tanh(qd / FRICTION_SMOOTH))
        p.step()
        r = np.abs(env.moment @ d.qvel) / env.rated
        np.maximum(ratio_max, r, out=ratio_max)
        past += bool((r > 1.0).any())
        sat_steps += np.abs(d.actuator_force) >= m.actuator_forcerange[:, 1] - 1e-9
        yaw = _yaw(d.xmat[p.trunk].reshape(3, 3))
        yaw_turned += (yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi  # unwrapped
        yaw_prev = yaw
        if p.fallen():
            fell_at = float(d.time)
            break
    wall = time.perf_counter() - t0
    steps = k + 1
    dur = float(d.time)
    heading = _yaw(R0)
    dxy = d.qpos[:2] - x0
    fwd = np.array([np.cos(heading), np.sin(heading)])
    worst = int(np.argmax(ratio_max))
    return dict(reference=ref.name, envelope=env.enabled, strides_max=strides, strides=dur / ref.period,
                fell=fell_at is not None, fall_time_s=fell_at, seconds=dur,
                mean_forward_speed_m_s=float(dxy @ fwd / max(dur, 1e-9)),
                lateral_drift_m=float(dxy @ np.array([-fwd[1], fwd[0]])),
                yaw_turned_deg=float(np.degrees(yaw_turned)), mean_yaw_rate_rad_s=yaw_turned / max(dur, 1e-9),
                reference_yaw_rate_rad_s=float(ref.stride_motion[0]) / ref.period,
                steps_past_corner_frac=past / steps,
                speed_ratio_max={dr.actuator: float(v) for dr, v in zip(p.drives, ratio_max, strict=True)},
                worst_speed_ratio=[p.drives[worst].actuator, float(ratio_max[worst])],
                saturated_frac={p.drives[i].actuator: float(sat_steps[i] / steps) for i in np.flatnonzero(sat_steps)},
                wall_s=wall, steps_per_s=steps / wall)


def replay_spread(p: plant.Plant, ref: Reference, runs: int = SPREAD_RUNS, offset: float = SPREAD_OFFSET_M_S,
                  seed: int = SPREAD_SEED, strides: float = REPLAY_STRIDES) -> dict:
    """Strides of ``runs`` replays whose start root velocity is offset in x and y by up to ``offset`` (uniform,
    seeded): how much the open-loop fall time depends on the start (reported with G2, not gated)."""
    rng = np.random.default_rng(seed)
    v = np.array([replay(p, ref, strides, np.r_[rng.uniform(-offset, offset, 2), 0.0])["strides"] for _ in range(runs)])
    return dict(runs=runs, max_offset_m_s=offset, seed=seed, strides=[float(x) for x in v], median=float(np.median(v)),
                min=float(v.min()), max=float(v.max()), runs_at_least_min_strides=int(np.sum(v >= MIN_REPLAY_STRIDES)))


def g2(body: str, out_dir: Path, strides: float = REPLAY_STRIDES) -> dict:
    """G2 from the references saved in ``out_dir`` (``refs.prepare``)."""
    t0 = time.perf_counter()
    doc = json.loads((Path(out_dir) / "refs.json").read_text())
    checks = {name: dict(matches=e["matches_records"],
                         records=[dict(record=c["record"], leaves=c["leaves"], max_rel_diff=c["max_rel_diff"],
                                       n_mismatched=c["n_mismatched"], n_missing=c["n_missing"])
                                  for c in e["record_checks"]])
              for name, e in doc["references"].items()}
    nom = doc["references"]["nominal"]
    plants = {flag: plant.build(body, "hard", envelope=flag) for flag in (True, False)}
    rep = {}
    for name, e in doc["references"].items():
        ref = Reference.load(Path(out_dir) / e["file"])
        rep[name] = {("with_envelope" if flag else "without_envelope"): replay(p, ref, strides)
                     for flag, p in plants.items()}
    walked = rep["nominal"]["with_envelope"]["strides"]
    nominal_ref = Reference.load(Path(out_dir) / nom["file"])
    alt = plant.build(body, "hard", ankle_gains="joint")
    info = dict(nominal_spread_with_envelope=replay_spread(plants[True], nominal_ref, strides=strides),
                joint_space_ankle_gains=dict(
                    note="plant built with ankle_gains='joint' (each ankle motor at half kp: the screened model's "
                         "joint-space ankle stiffness); not the gated plant",
                    nominal_with_envelope=replay(alt, nominal_ref, strides),
                    nominal_spread_with_envelope=replay_spread(alt, nominal_ref, strides=strides)))
    out = dict(references_match=checks, all_match=all(c["matches"] for c in checks.values()),
               nominal_speed_m_s=round(nom["speed_m_s"], 6), nominal_min_cop_margin_single_m=round(
                   nom["min_cop_margin_single_m"], 6),
               replay=rep, nominal_strides_with_envelope=walked,
               nominal_strides_without_envelope=rep["nominal"]["without_envelope"]["strides"],
               design_study_strides_screened_model=DESIGN_STUDY_STRIDES[body], min_strides=MIN_REPLAY_STRIDES,
               reported_not_gated=info)
    out["pass"] = bool(out["all_match"] and walked >= MIN_REPLAY_STRIDES)
    out["seconds"] = time.perf_counter() - t0
    return out
