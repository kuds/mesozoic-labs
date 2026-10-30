"""Static check of why command +1 holds tail_3 at about +5.2 deg (no dynamics: mj_forward at the home keyframe only).

PYTHONPATH=<policies/code_copy> python -B static_tail.py > static_tail.txt

Reads the model's tail_3 servo (gain, bias, force range), the joint's stiffness and spring reference, and the
gravity (bias) torque on the tail_3 dof at the home pose with tail_3 at 0 and at +5.2 deg; then solves the static
balance kp (c - q) + k (q_ref - q) - g(q) = 0 for c = 0 and c = +0.2094 rad, and compares the two-stiffness split
kp / (kp + k) with the measured +5.2 deg.  Also the action-to-ctrl map through the env's own _scale_action.
"""

import math

import mujoco
import numpy as np

from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.species_registry import env_class

MID = ModelId("tyrannosaurus_rex", "anatomical")
env = env_class(MID)(**dict(load_stage_config(MID, "stance")["env_kwargs"]))
u = env.unwrapped
m = u.model
d = mujoco.MjData(m)
a = 14
j = m.joint("tail_3_pitch").id
qa, da = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
print("actuator", m.actuator(a).name, "trn joint", m.joint(int(m.actuator_trnid[a, 0])).name)
print("gainprm[0]", m.actuator_gainprm[a, 0], "biasprm[:3]", m.actuator_biasprm[a, :3].tolist(),
      "forcerange", m.actuator_forcerange[a].tolist(), "ctrlrange", m.actuator_ctrlrange[a].tolist())
print("joint stiffness", m.jnt_stiffness[j], "springref (qpos_spring)", m.qpos_spring[qa], "rad =",
      math.degrees(m.qpos_spring[qa]), "deg; range deg", np.degrees(m.jnt_range[j]).tolist(),
      "damping", m.dof_damping[da], "armature", m.dof_armature[da], "frictionloss", m.dof_frictionloss[da])
kp = float(m.actuator_gainprm[a, 0])
k = float(m.jnt_stiffness[j])
qref = float(m.qpos_spring[qa])
for qdeg in (0.0, 5.2):
    mujoco.mj_resetDataKeyframe(m, d, int(u._reset_keyframe_id))
    d.qpos[qa] = math.radians(qdeg)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    print(f"tail_3 at {qdeg:+.1f} deg: gravity/bias torque qfrc_bias {d.qfrc_bias[da]:+.4f} N m; passive (spring) "
          f"{d.qfrc_passive[da]:+.4f} N m; spring+gravity net {d.qfrc_passive[da] - d.qfrc_bias[da]:+.4f} N m")
g0 = None
mujoco.mj_resetDataKeyframe(m, d, int(u._reset_keyframe_id))
d.qpos[qa] = 0.0
mujoco.mj_forward(m, d)
g0 = float(d.qfrc_bias[da])
for c in (0.0, float(m.actuator_ctrlrange[a, 1])):
    # solve kp (c - q) + k (qref - q) - g(q) = 0 by fixed point with g re-evaluated at q
    q = 0.0
    for _ in range(50):
        mujoco.mj_resetDataKeyframe(m, d, int(u._reset_keyframe_id))
        d.qpos[qa] = q
        mujoco.mj_forward(m, d)
        g = float(d.qfrc_bias[da])
        q = (kp * c + k * qref - g) / (kp + k)
    print(f"static equilibrium for ctrl {c:+.4f} rad: q {math.degrees(q):+.3f} deg; servo torque {kp * (c - q):+.3f} N m")
print(f"two-stiffness split kp/(kp+k) = {kp / (kp + k):.4f}; x 12 deg = {12 * kp / (kp + k):.3f} deg; "
      f"spring preload k*qref {k * qref:+.3f} N m vs gravity torque at 0 {g0:+.3f} N m")
for act in (-1.0, 0.0, 0.5, 1.0):
    v = np.zeros(15)
    v[a] = act
    print(f"_scale_action: action {act:+.1f} -> ctrl {u._scale_action(v)[a]:+.5f} rad")
env.close()
