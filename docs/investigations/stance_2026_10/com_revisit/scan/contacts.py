"""Which geoms carry each model's floor load (statue and one 0.02 spawn; at the first control step and at the end), and
the light joint's place in the load path; plus the Velociraptor's lateral pre-stress decay time (velo_ext.jsonl)."""
import json
import mujoco
import numpy as np
import scanlib as S
from environments.shared.tests.com_stance_helpers import run_tracked_episode
from environments.shared.gait.morphology import Morphology

f6 = np.zeros(6)
for model, light in (("compy_anat", ("l_ankle", "r_ankle")), ("velo", ("l_claw", "r_claw"))):
    for noise, seed in ((None, 0), (0.02, 313 if model == "compy_anat" else 274)):
        inst = S.CLASSES[model]()
        env = inst.make_env() if noise is None else inst.make_env(reset_noise_scale=noise)
        u = env.unwrapped; m, d = u.model, u.data
        morph = Morphology.from_env(env, S.CLASSES[model].model_id)
        snaps = {}
        def post(env_, step, action):
            if step in (1, u.max_episode_steps):
                load = {}
                for i in range(d.ncon):
                    c = d.contact[i]
                    g = c.geom2 if morph.is_floor[c.geom1] else c.geom1
                    mujoco.mj_contactForce(m, d, i, f6)
                    load[m.geom(int(g)).name] = load.get(m.geom(int(g)).name, 0.0) + f6[0]
                snaps[step] = {k: round(v, 3) for k, v in sorted(load.items())}
        run_tracked_episode(env, S.CLASSES[model].model_id, seed=seed, post_step=post, morphology=morph)
        # bodies below the light joint
        jb = [int(m.jnt_bodyid[m.joint(n).id]) for n in light]
        below = []
        for b in range(m.nbody):
            x = b
            while x > 0 and x not in jb:
                x = int(m.body_parentid[x])
            if x in jb:
                below.append(m.body(b).name)
        geoms_below = [m.geom(g).name for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in below]
        print(f"{model} noise {noise} seed {seed}: floor load by geom (N) at step 1 {snaps.get(1)}; at the last step {snaps.get(u.max_episode_steps)}")
        print(f"   bodies carried by the light joints {light}: {below}; their geoms {geoms_below}; "
              f"armature {[float(m.dof_armature[m.jnt_dofadr[m.joint(n).id]]) for n in light]}")
        env.close()

print("Velociraptor lateral pre-stress decay (velo_ext.jsonl): fit of log|mean lateral force| on seconds 10-58")
for line in open("velo_ext.jsonl"):
    r = json.loads(line)
    lat = np.array(r["lateral_force_per_s_n"])
    s = np.arange(10, 59)
    y = np.log(np.abs(lat[10:59, 0]) + 1e-12)
    k, c = np.polyfit(s, y, 1)
    print(f"   noise {r['noise']} seed {r['seed']}: tau {-1 / k:.2f} s; |F_y| s1 {abs(lat[1,0]):.3f} s59 {abs(lat[59,0]):.3f} N; "
          f"CoM speed s59 {r['com_speed_rms_per_s_mmps'][59]:.4f} mm/s")
