"""Observation layout of the T. rex anatomical stance env (bipedal-target-heading-free/v1, 63 values).

PYTHONPATH=<policies/code_copy> python -B probe_layout.py > layout.txt

Read only; builds one env from the stance config's constructor, resets it, and prints for every hinge joint its
observation index for the angle (qposadr - 7) and the velocity ((nq - 7) + dofadr - 6), checked against the live
_get_obs() output, then the indices of the remaining blocks in _get_obs()'s concatenation order.  No training.
"""

import json
import sys

import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.species_registry import env_class

MID = ModelId("tyrannosaurus_rex", "anatomical")


def main() -> int:
    ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
    env = env_class(MID)(**ctor)
    u = env.unwrapped
    m, d = u.model, u.data
    obs, _ = env.reset(seed=3042)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(m.nu))
    nq7 = m.nq - 7
    nv6 = m.nv - 6
    print(f"nq {m.nq} nv {m.nv} obs size {obs.shape[0]}; joint angles 0..{nq7 - 1}, joint velocities {nq7}..{nq7 + nv6 - 1}")
    rows = []
    for j in range(m.njnt):
        name = m.joint(j).name
        if int(m.jnt_type[j]) == 0:  # free
            continue
        qa = int(m.jnt_qposadr[j]) - 7
        va = nq7 + int(m.jnt_dofadr[j]) - 6
        ok_q = bool(np.float32(d.qpos[qa + 7]) == obs[qa])
        ok_v = bool(np.float32(d.qvel[va - nq7 + 6]) == obs[va])
        rows.append({"joint": name, "angle_idx": qa, "vel_idx": va, "check_angle": ok_q, "check_vel": ok_v})
        print(f"  {name:16s} angle obs[{qa:2d}]  velocity obs[{va:2d}]  live check {ok_q} {ok_v}")
    k = nq7 + nv6
    blocks = [("gravity", 3), ("pelvis_gyro", 3), ("heading_velocity", 3), ("pelvis_accel", 3), ("foot_contact", 2),
              ("prey_direction_body", 3), ("prey_distance", 1), ("command", 3)]
    names = [None] * obs.shape[0]
    for r in rows:
        names[r["angle_idx"]] = r["joint"] + ":q"
        names[r["vel_idx"]] = r["joint"] + ":v"
    for b, n in blocks:
        print(f"  {b:20s} obs[{k}..{k + n - 1}]")
        for i in range(n):
            names[k + i] = f"{b}[{i}]"
        k += n
    print(f"  total {k} (obs size {obs.shape[0]})")
    assert k == obs.shape[0]
    assert m.actuator(14).name == "tail_3_pitch_act"
    env.close()
    json.dump({"names": names, "joints": rows}, open(sys.argv[1] if len(sys.argv) > 1 else "layout.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
