"""Verifier: observation indices of the tail joints, by perturbation (independent of the probe's equality check).

PYTHONPATH=<verify_probe/code> python -B v_layout.py > v_layout.txt

Builds the stance env, steps it a little, then for every hinge joint perturbs its qpos (and separately its qvel) by
a distinct amount, calls _get_obs() and records which observation entries changed.  Joint angle entries should move
alone (1 entry) for qpos; for qvel only the velocity entry should move (heading velocity etc. read the free joint).
Also prints which joint actuator 14 drives (trnid) and its name.  No training; nothing is written but stdout.
"""
import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.species_registry import env_class

MID = ModelId("tyrannosaurus_rex", "anatomical")
kw = dict(load_stage_config(MID, "stance")["env_kwargs"])
env = env_class(MID)(**kw)
u = env.unwrapped
m, d = u.model, u.data
obs, _ = env.reset(seed=7)
for _ in range(20):
    obs, *_ = env.step(np.zeros(m.nu))
print(f"obs len {obs.shape[0]}  nq {m.nq} nv {m.nv} nu {m.nu} njnt {m.njnt}")
base = u._get_obs().copy()
assert np.array_equal(base, obs), "obs from step != _get_obs() right after"
q0, v0 = d.qpos.copy(), d.qvel.copy()
for j in range(m.njnt):
    if int(m.jnt_type[j]) != 3:  # hinge only
        print(f"  joint {j} {m.joint(j).name} type {int(m.jnt_type[j])} (skipped)")
        continue
    qa, va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    d.qpos[:] = q0; d.qvel[:] = v0
    d.qpos[qa] += 0.0123
    o1 = u._get_obs()
    dq = np.nonzero(o1 != base)[0]
    d.qpos[:] = q0; d.qvel[:] = v0
    d.qvel[va] += 0.0456
    o2 = u._get_obs()
    dv = np.nonzero(o2 != base)[0]
    d.qpos[:] = q0; d.qvel[:] = v0
    tag = "  <-- tail" if m.joint(j).name.startswith("tail") else ""
    print(f"  {m.joint(j).name:16s} qposadr {qa:2d} dofadr {va:2d}  obs changed by qpos: {list(dq)}  by qvel: {list(dv)}"
          f"  d(obs)={o1[dq] - base[dq] if len(dq) else []}{tag}")
for a in range(m.nu):
    jid = int(m.actuator_trnid[a, 0])
    if a in (11, 12, 13, 14) or "tail" in m.actuator(a).name:
        print(f"  actuator {a:2d} {m.actuator(a).name:22s} drives joint {m.joint(jid).name}")
env.close()
