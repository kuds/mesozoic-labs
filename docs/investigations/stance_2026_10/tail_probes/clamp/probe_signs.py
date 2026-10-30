"""Sign conventions by forward kinematics at the home keyframe (no dynamics): which way + on neck_pitch, head_pitch,
tail_3_pitch moves the snout / tail tip, and the whole-body CoM shift.  Read only."""
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
root = u.pelvis_id
snout = m.geom("snout").id
tip = u.tail_tip_site_id
for joint, delta_deg in (("neck_pitch", 5.0), ("head_pitch", 5.0), ("tail_3_pitch", 5.2)):
    out = []
    for sgn in (0.0, 1.0):
        mujoco.mj_resetDataKeyframe(m, d, int(u._reset_keyframe_id))
        d.qpos[m.jnt_qposadr[m.joint(joint).id]] += sgn * np.radians(delta_deg)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        out.append((d.geom_xpos[snout].copy(), d.site_xpos[tip].copy(), d.subtree_com[int(m.jnt_bodyid[0])].copy()))
    (s0, t0, c0), (s1, t1, c1) = out
    fwd = d.xmat[root].reshape(3, 3)[:, 0]
    print(f"{joint} +{delta_deg} deg: snout dz {1e3*(s1-s0)[2]:+.1f} mm, dx(fwd) {1e3*(s1-s0)@fwd:+.1f} mm; tail tip dz {1e3*(t1-t0)[2]:+.1f} mm; "
          f"whole-body CoM d(fwd) {1e3*(c1-c0)@fwd:+.3f} mm, dz {1e3*(c1-c0)[2]:+.3f} mm")
