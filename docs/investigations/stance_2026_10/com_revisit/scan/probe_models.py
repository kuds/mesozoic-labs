"""Read-only look at the two models as the CoM test classes build them (no rollout beyond one timed episode)."""
import time
import mujoco
import numpy as np
from environments.compsognathus_longipes.tests.test_com_stance_quality import TestAnatomicalComStanceQuality
from environments.velociraptor_mongoliensis.tests.test_com_stance_quality import TestComStanceQuality as TestVelo
from environments.shared.gait.morphology import Morphology
from environments.shared.tests.com_stance_helpers import run_tracked_episode

for CLS in (TestAnatomicalComStanceQuality, TestVelo):
    inst = CLS()
    env = inst.make_env(reset_noise_scale=0.02)
    u = env.unwrapped; m = u.model
    morph = Morphology.from_env(env, CLS.model_id)
    print(f"== {CLS.model_id}: class {CLS.__name__}, window {CLS.window_start_s}, horizon_steps {CLS.horizon_steps}, "
          f"max_episode_steps {u.max_episode_steps}, dt {u.dt}, timestep {m.opt.timestep}, frame_skip {u.frame_skip}")
    print(f"   integrator {m.opt.integrator} cone {m.opt.cone} solver {m.opt.solver} iters {m.opt.iterations} tol {m.opt.tolerance} noslip {m.opt.noslip_iterations} impratio {m.opt.impratio}")
    print(f"   nq {m.nq} nv {m.nv} nu {m.nu} ngeom {m.ngeom} nbody {m.nbody}; mass {m.body_subtreemass[morph.root_body]:.4f} kg; root body {m.body(morph.root_body).name}")
    print(f"   feet: {[ (f.label, m.site(f.primary_site).name) for f in morph.feet]}")
    gf = np.asarray(morph.geom_foot)
    print(f"   leg geoms per foot: {[ [m.geom(g).name for g in np.flatnonzero(gf == i)] for i in range(len(morph.feet))]}")
    print(f"   floor geoms {[m.geom(g).name for g in morph.floor_geoms]}")
    print(f"   reset_noise_scale attr {getattr(u, '_reset_noise_scale', None)}; keyframe {getattr(u, '_reset_keyframe_id', None)}")
    # actuators
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]; dof = m.jnt_dofadr[j]
        print(f"     act {m.actuator(a).name:24s} joint {m.joint(j).name:20s} kp {m.actuator_gainprm[a,0]:.3g} kv {-m.actuator_biasprm[a,2]:.3g} "
              f"armature {m.dof_armature[dof]:.3g} damping {m.dof_damping[dof]:.3g} frange {m.actuator_forcerange[a]} gear {m.actuator_gear[a,0]}")
    ncon = []
    t0 = time.time()
    u._substep_probe_hook = lambda: ncon.append(int(u.data.ncon))
    ep = run_tracked_episode(env, CLS.model_id, seed=0, morphology=morph)
    print(f"   one episode seed 0 at 0.02: stood {ep.stood} steps {ep.steps} {time.time()-t0:.1f} s; ncon per substep mean {np.mean(ncon):.1f} max {max(ncon)}")
    env.close()
