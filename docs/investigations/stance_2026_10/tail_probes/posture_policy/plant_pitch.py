"""Test 3 addendum (written on resume): the plant map under a PURE pelvis rotation, everything else frozen.

PYTHONPATH=<policies/code_copy>:<this folder> python -B plant_pitch.py run/base plant_pitch.json > plant_pitch.txt

Kinematics and statics only (mj_forward, qvel = 0; no time stepping, no policy, no training).  run_plant.py static
compares the postures the policies hold, which differ in many joints at once.  Here one posture at a time (the statue's,
the s42 ends' and the s80 ends' typical held qpos = the median over policies of the per-policy median over seeds of the
window-mean qpos, quaternion renormalised; and each of the 12 s42 ends' own median posture) is rotated about the
pelvis's own y axis by D in {-2.2, -1.1, 0, +1.1, +2.2} deg (positive = nose UP, the run_base/pitch_illusion convention),
and for each D:
  - gravity torque on tail_1_pitch, tail_2_pitch, tail_3_pitch (-qfrc_bias at qvel 0) with the tail at the posture's
    own angles, and on tail_3 with tail_3 at the tail_home target;
  - tail_3's static equilibrium q3(a), a in {-1, 0, +1}, gravity re-evaluated at q (run_plant.static_one's solver);
  - the command a needed to hold the tail_home target;
  - the tail_3 segment's world pitch.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from pp_common import ALL_WHO, NAMES, TAIL3, make_env, stance_ctor

D_GRID = [-2.2, -1.1, 0.0, 1.1, 2.2]


def posture(base: Path, who: str) -> np.ndarray:
    q = np.median(np.load(base / "arrays" / f"{who}.npz")["qpos_mean"], axis=0)
    q[3:7] /= np.linalg.norm(q[3:7])
    return q


def rotate(q: np.ndarray, nose_up_deg: float) -> np.ndarray:
    rq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rq, np.array([0.0, 1.0, 0.0]), -math.radians(nose_up_deg))
    nq = np.zeros(4)
    mujoco.mju_mulQuat(nq, q[3:7], rq)
    out = q.copy()
    out[3:7] = nq
    return out


def measure(u, d, q: np.ndarray) -> dict:
    m = u.model
    names = ("tail_1_pitch", "tail_2_pitch", "tail_3_pitch")
    jid = {n: m.joint(n).id for n in names}
    qa = {n: int(m.jnt_qposadr[jid[n]]) for n in names}
    da = {n: int(m.jnt_dofadr[jid[n]]) for n in names}
    k = float(m.jnt_stiffness[jid["tail_3_pitch"]])
    ref = float(m.qpos_spring[qa["tail_3_pitch"]])
    kp = float(m.actuator_gainprm[TAIL3, 0])
    cmax = float(m.actuator_ctrlrange[TAIL3, 1])

    def fwd(x):
        d.qpos[:] = x
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    fwd(q)
    out = {"pelvis_pitch_deg": math.degrees(math.asin(float(d.xmat[u.pelvis_id].reshape(3, 3)[2, 0])))}
    for n in names:
        out[f"grav_{n}_Nm"] = -float(d.qfrc_bias[da[n]])
    R3 = d.xmat[m.body("tail_3").id].reshape(3, 3)
    v = R3 @ np.array([-1.0, 0.0, 0.0])
    out["tail3_world_pitch_deg"] = math.degrees(math.atan2(v[2], -v[0]))
    eq = []
    for a in (-1.0, 0.0, 1.0):
        c = a * cmax
        x = float(q[qa["tail_3_pitch"]])
        for _ in range(60):
            qq = q.copy()
            qq[qa["tail_3_pitch"]] = x
            fwd(qq)
            g = float(d.qfrc_bias[da["tail_3_pitch"]])
            x = (kp * c + k * ref - g) / (kp + k)
        eq.append(math.degrees(x))
    out["q3_eq_deg_a-1_0_+1"] = eq
    target = float(u._tail_home_qpos[3])
    qq = q.copy()
    qq[qa["tail_3_pitch"]] = target
    fwd(qq)
    g = float(d.qfrc_bias[da["tail_3_pitch"]])
    out["grav_tail3_at_target_Nm"] = -g
    out["a_to_hold_target"] = (target + (g - k * (ref - target)) / kp) / cmax
    return out


def main() -> int:
    base = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    env = make_env(stance_ctor())
    env.reset(seed=3042)
    u = env.unwrapped
    d = mujoco.MjData(u.model)
    pos = {"statue": posture(base, "statue")}
    for lin in ("s42", "s80"):
        per = [posture(base, w) for w in NAMES if w.startswith(lin)]
        q = np.median(np.stack(per), axis=0)
        q[3:7] /= np.linalg.norm(q[3:7])
        pos[f"{lin}_ends_typical"] = q
    for w in NAMES:
        if w.startswith("s42"):
            pos[w] = posture(base, w)
    res = {"D_grid_nose_up_deg": D_GRID, "postures": {}}
    for name, q in pos.items():
        res["postures"][name] = {f"{D:+.1f}": measure(u, d, rotate(q, D)) for D in D_GRID}
    env.close()
    out_path.write_text(json.dumps(res, indent=1))
    keys = ["pelvis_pitch_deg", "tail3_world_pitch_deg", "grav_tail_1_pitch_Nm", "grav_tail_2_pitch_Nm",
            "grav_tail_3_pitch_Nm", "grav_tail3_at_target_Nm", "a_to_hold_target"]
    print("PURE PELVIS ROTATION, everything else frozen (D = extra nose-up, deg).  q3_eq in deg at a = -1 / 0 / +1.")
    for name in ("statue", "s42_ends_typical", "s80_ends_typical"):
        print(f"  {name}")
        for D in D_GRID:
            r = res["postures"][name][f"{D:+.1f}"]
            print(f"    D {D:+.1f}: " + " ".join(f"{k} {r[k]:+.4f}" for k in keys)
                  + " q3_eq " + "/".join(f"{x:+.4f}" for x in r["q3_eq_deg_a-1_0_+1"]))
    print("  per-deg slopes (least squares over D), 12 s42 ends' own postures: median [range]")
    for k in keys[1:] + ["q3_eq_a0", "q3_eq_a+1"]:
        sl = []
        for w in [x for x in pos if x.startswith("s42_") and x != "s42_ends_typical"]:
            ys = []
            for D in D_GRID:
                r = res["postures"][w][f"{D:+.1f}"]
                ys.append(r["q3_eq_deg_a-1_0_+1"][1] if k == "q3_eq_a0" else r["q3_eq_deg_a-1_0_+1"][2]
                          if k == "q3_eq_a+1" else r[k])
            sl.append(np.polyfit(D_GRID, ys, 1)[0])
        sl = np.asarray(sl)
        print(f"    d {k:24s} / d nose-up deg: {np.median(sl):+.5f} [{sl.min():+.5f},{sl.max():+.5f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
