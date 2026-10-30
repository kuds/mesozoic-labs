"""Test 3: the r9 "does the plant force the pin" check, redone on r10 in the postures the policies actually hold.

PYTHONPATH=<policies/code_copy>:<this folder> python -B run_plant.py static BASE_RUN_DIR OUT.json
PYTHONPATH=<policies/code_copy>:<this folder> python -B run_plant.py replay BASE_RUN_DIR OUT.jsonl [--seeds 3042,3051,3061]

Simulation only, no training, no policy evaluation (the replays use the commands recorded in the baseline rollouts).

The r9 method (tailround/mech/probe_plant.py, copied as r9_probe_plant_copy.py) measured, on the statue only: the
tail_3 joint angle reached against the command a in [-1, 1] (part C), the torque balance on the tail joints (B, D) and
the reward terms that read the tail (probe_reward.py).  Here the same quantities are measured in each posture held in
the baseline deterministic rollouts (run_base.py, seeds 3042-3061, 1,000 steps; posture = the window mean of qpos over
steps 501-1000 with the quaternion renormalised; applied command = the window mean of the action filter state):

static (mj_forward only, no time stepping; one MjData per posture, qvel = 0):
  - gravity torque on tail_1_pitch, tail_2_pitch, tail_3_pitch (= -qfrc_bias at qvel 0; + lifts the tail), their spring
    torque k (springref - q) and servo torque kp (ctrl - q) at the posture's own applied command;
  - tail_3's static equilibrium q3(a) for a = -1, -0.75, ..., +1 with everything else frozen in the posture, solved as
    verify_clamp/static_tail.py does (kp (c - q) + k (qref - q) - qfrc_bias(q) = 0, gravity re-evaluated at q), so the
    plant map's slope and intercept per posture; the command a needed to hold the tail_home_pose target (0.00022 rad);
  - the tail_3 segment's world pitch and the tail tip height, the pelvis pitch;
  - the residual tail_3 torque at the posture's own (q3, command): how far from static balance the policy holds it.

replay (open loop): the policy's own recorded 15 commands of a baseline episode, entry 14 replaced by a constant
a in {-1, -0.5, 0, +0.5, +1} (and, as a check, not replaced: must reproduce the baseline return), the env's own step,
filter, scaling and reward.  The statue's commands are zeros.  Per episode, window means over steps 501-1000 of: the
tail_3 angle, tail_instability (|tail-tip angular velocity|, what tail_stability prices), the tail_tip_angvel gyro
sensor's norm, every reward_* term and the total, the pelvis pitch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from pp_common import ALL_WHO, TAIL3, make_env, stance_ctor

A_GRID = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
A_REPLAY = [-1.0, -0.5, 0.0, 0.5, 1.0]
TAIL_PITCH = ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch")
TAIL_ACT = ("tail_1_pitch_act", "tail_2_pitch_act", "tail_3_pitch_act")


def scale(u, a: np.ndarray) -> np.ndarray:
    return u._scale_action(np.asarray(a, dtype=np.float64))


def static_one(u, d, qpos: np.ndarray, filt: np.ndarray) -> dict:
    m = u.model
    jid = {n: m.joint(n).id for n in TAIL_PITCH}
    qa = {n: int(m.jnt_qposadr[jid[n]]) for n in TAIL_PITCH}
    da = {n: int(m.jnt_dofadr[jid[n]]) for n in TAIL_PITCH}
    aid = {n: m.actuator(a).id for n, a in zip(TAIL_PITCH, TAIL_ACT)}
    assert aid["tail_3_pitch"] == TAIL3
    ctrl = scale(u, filt)

    def fwd(q):
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    fwd(qpos)
    out = {"pelvis_pitch_deg": math.degrees(math.asin(float(d.xmat[u.pelvis_id].reshape(3, 3)[2, 0])))}
    for n in TAIL_PITCH:
        q = float(d.qpos[qa[n]])
        k = float(m.jnt_stiffness[jid[n]])
        ref = float(m.qpos_spring[qa[n]])
        kp = float(m.actuator_gainprm[aid[n], 0])
        grav = -float(d.qfrc_bias[da[n]])
        spring = k * (ref - q)
        servo = kp * (float(ctrl[aid[n]]) - q)
        out[n] = {"q_deg": math.degrees(q), "gravity_Nm": grav, "spring_Nm": spring, "servo_Nm": servo,
                  "net_static_Nm": grav + spring + servo, "cmd": float(filt[aid[n]])}
    R3 = d.xmat[m.body("tail_3").id].reshape(3, 3)
    v = R3 @ np.array([-1.0, 0.0, 0.0])
    out["tail3_segment_world_pitch_deg"] = math.degrees(math.atan2(v[2], -v[0]))
    out["tail_tip_z_m"] = float(d.site_xpos[u.tail_tip_site_id][2])
    # static equilibrium of tail_3 alone against a, everything else frozen in the posture
    n = "tail_3_pitch"
    k = float(m.jnt_stiffness[jid[n]])
    ref = float(m.qpos_spring[qa[n]])
    kp = float(m.actuator_gainprm[TAIL3, 0])
    eq = []
    for a in A_GRID:
        v15 = np.array(filt, dtype=np.float64, copy=True)
        v15[TAIL3] = a
        c = float(scale(u, v15)[TAIL3])
        q = float(qpos[qa[n]])
        for _ in range(60):
            qq = qpos.copy()
            qq[qa[n]] = q
            fwd(qq)
            g = float(d.qfrc_bias[da[n]])
            q = (kp * c + k * ref - g) / (kp + k)
        eq.append(math.degrees(q))
    out["q3_eq_deg_vs_a"] = eq
    slope, icpt = np.polyfit(A_GRID, eq, 1)
    out["plant_slope_deg_per_unit"] = float(slope)
    out["plant_intercept_deg"] = float(icpt)
    # command needed to hold the tail_home_pose target
    target = float(u._tail_home_qpos[3])
    qq = qpos.copy()
    qq[qa[n]] = target
    fwd(qq)
    g = float(d.qfrc_bias[da[n]])
    c_req = target + (g - k * (ref - target)) / kp
    cmax = float(m.actuator_ctrlrange[TAIL3, 1])
    out["a_to_hold_target"] = c_req / cmax
    out["gravity_Nm_at_target_tail3"] = -g
    fwd(qpos)
    return out


def do_static(base: Path, out_path: Path) -> None:
    ctor = stance_ctor()
    env = make_env(ctor)
    env.reset(seed=3042)
    u = env.unwrapped
    d = mujoco.MjData(u.model)
    res = {"a_grid": A_GRID, "tail_home_target_rad": [float(x) for x in u._tail_home_qpos], "postures": {}}
    for who in ALL_WHO:
        z = np.load(base / "arrays" / f"{who}.npz")
        rows = []
        for i, seed in enumerate(z["seeds"]):
            r = static_one(u, d, z["qpos_mean"][i], z["filt_mean"][i])
            r["seed"] = int(seed)
            rows.append(r)
        res["postures"][who] = rows
        print(who, "static done", flush=True)
    # the home keyframe as the r9-style reference (noise-free, home controls)
    mujoco.mj_resetDataKeyframe(u.model, d, int(u._reset_keyframe_id))
    res["home_keyframe"] = static_one(u, d, d.qpos.copy(), np.zeros(u.model.nu))
    env.close()
    out_path.write_text(json.dumps(res, indent=1))


def do_replay(base: Path, out_path: Path, seeds: list[int]) -> None:
    ctor = stance_ctor()
    m0 = make_env(ctor)
    q3 = int(m0.unwrapped.model.jnt_qposadr[m0.unwrapped.model.joint("tail_3_pitch").id])
    gyro_adr = None
    mm = m0.unwrapped.model
    for s in range(mm.nsensor):
        if mm.sensor(s).name == "tail_tip_angvel":
            gyro_adr = int(mm.sensor_adr[s])
    m0.close()
    out = open(out_path, "a")
    for who in ALL_WHO:
        z = np.load(base / "arrays" / f"{who}.npz")
        allseeds = list(z["seeds"])
        t0 = time.time()
        for seed in seeds:
            i = allseeds.index(seed)
            acts = z["own_actions"][i].astype(np.float64) if who != "statue" else np.zeros((1000, 15))
            for a in ["own"] + A_REPLAY:
                env = make_env(ctor)
                u = env.unwrapped
                dd = u.data
                env.reset(seed=int(seed))
                rew, infos, t3, gy, fz = [], [], [], [], []
                steps = 0
                for t in range(1000):
                    act = acts[t].copy()
                    if a != "own":
                        act[TAIL3] = a
                    _, r, term, trunc, info = env.step(act.astype(np.float32) if who != "statue" else act)
                    steps += 1
                    rew.append(float(r))
                    infos.append({k: float(v) for k, v in info.items() if k.startswith("reward_") or k == "tail_instability"})
                    t3.append(float(dd.qpos[q3]))
                    gy.append(float(np.linalg.norm(dd.sensordata[gyro_adr:gyro_adr + 3])))
                    fz.append(float(dd.xmat[u.pelvis_id].reshape(3, 3)[2, 0]))
                    if term or trunc:
                        break
                env.close()
                w = slice(500, steps)
                keys = sorted(infos[0])
                row = {"who": who, "seed": int(seed), "a": a, "steps": steps, "return": float(np.sum(rew)),
                       "per_step_500_1000": {k: float(np.mean([x[k] for x in infos[w]])) for k in keys},
                       "reward_per_step_500_1000": float(np.mean(rew[w])),
                       "tail3_deg": math.degrees(float(np.mean(t3[w]))),
                       "tail_tip_gyro_norm": float(np.mean(gy[w])),
                       "pitch_deg": math.degrees(math.asin(float(np.mean(fz[w]))))}
                out.write(json.dumps(row) + "\n")
                out.flush()
        print(f"{who}: replays {time.time() - t0:.1f} s", flush=True)
    out.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("static", "replay"))
    ap.add_argument("base", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", default="3042,3051,3061")
    a = ap.parse_args()
    if a.mode == "static":
        do_static(a.base, a.out)
    else:
        do_replay(a.base, a.out, [int(s) for s in a.seeds.split(",")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
