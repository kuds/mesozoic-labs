"""Reviewer probe (not part of PR #655): the PR's statue window rewards and t4d lean cases, re-run on whatever
code `checkout` points to (here: the r10 round's run branch code copy, read only). Open loop, no network."""
import json, sys
import numpy as np
from multiprocessing import Pool
D = "scratchpad"
sys.path.insert(0, D)
LEAN52 = {"r_hip_pitch_act": 0.104, "l_hip_pitch_act": 0.104, "r_ankle_act": -0.084, "l_ankle_act": -0.084}
CASES = {"statue": {}, "lean0.25": {k: v * 0.25 for k, v in LEAN52.items()}, "lean0.5": {k: v * 0.5 for k, v in LEAN52.items()},
         "hip+0.02": {"r_hip_pitch_act": 0.02, "l_hip_pitch_act": 0.02}}
SEEDS = [None, 0, 1, 2, 3]
def job(args):
    ref, case, seed = args
    from common import make_env, names
    from t2_pins import lean_rollout, make_fn
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    if ref is not None:
        u.leg_home_pose_reference = ref
    _, an = names(u)
    rec = lean_rollout(env, u, make_fn(dict(pins=CASES[case]), u.model.nu, an), seed)
    out = dict(ref=ref, case=case, seed=seed, n=rec["n"], ret=float(rec["R"].sum()),
               cfg=dict(leg_ref=u.leg_home_pose_reference, leg_w=float(u.leg_home_pose_weight), leg_tol=float(u.leg_home_pose_tolerance),
                        lateral_lean_w=float(getattr(u, "lateral_lean_weight", float("nan"))),
                        stagger_w=float(getattr(u, "stance_stagger_weight", float("nan"))), cop_band_w=float(getattr(u, "stance_cop_band_weight", float("nan"))),
                        settle=int(u.stance_width_settle_steps)))
    if rec["n"] >= 1000:
        out["win"] = {f"{a}-{b}": dict(total=float(rec["R"][a:b].mean()), terms={k: float(v[a:b].mean()) for k, v in rec["T"].items()})
                      for a, b in ((0, 50), (50, 200), (150, 200), (200, 250), (200, 1000))}
    return out
if __name__ == "__main__":
    jobs = [(None, c, s) for c in CASES for s in SEEDS] + [("keyframe", c, s) for c in CASES for s in SEEDS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=2)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print("done", len(res))
