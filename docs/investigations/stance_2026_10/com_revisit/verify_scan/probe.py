"""Probe: feet, leg subtrees, collision geoms of the three models (no stepping beyond reset)."""
import mujoco, numpy as np
from environments.compsognathus_longipes.tests.test_com_stance_quality import TestAnatomicalComStanceQuality as CA, TestRobotComStanceQuality as CR
from environments.velociraptor_mongoliensis.tests.test_com_stance_quality import TestComStanceQuality as CV
from environments.shared.gait.morphology import Morphology
for name, CLS in (("compy_anat", CA), ("velo", CV), ("robot", CR)):
    env = CLS().make_env()
    u = env.unwrapped; m = u.model
    morph = Morphology.from_env(env, CLS.model_id)
    print(f"== {name}: nbody {m.nbody} ngeom {m.ngeom} nv {m.nv} nu {m.nu} h {m.opt.timestep} frame_skip {u.frame_skip} dt {u.dt} max_steps {u.max_episode_steps} hook {u._substep_probe_hook}")
    print("   root body", morph.root_body, m.body(morph.root_body).name, "mass total", float(m.body_subtreemass[morph.root_body]), float(sum(m.body_mass)))
    for f in morph.feet:
        s = int(f.primary_site)
        print("   foot", f.label, "site", m.site(s).name, "body", m.body(int(m.site_bodyid[s])).name)
    gf = np.asarray(morph.geom_foot)
    print("   floor geoms", [m.geom(g).name for g in range(m.ngeom) if morph.is_floor[g]])
    coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    print("   collidable animal geoms:", [(m.geom(g).name, m.body(int(m.geom_bodyid[g])).name, int(gf[g])) for g in coll if morph.is_animal[g]])
    print("   root children:", [m.body(b).name for b in range(m.nbody) if m.body_parentid[b] == morph.root_body])
    env.close()
