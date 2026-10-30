"""Diagnostic: does the small-lean gain over the statue come from the keyframe-referenced leg_home_pose?"""
import sys, json, numpy as np
from multiprocessing import Pool
D = "scratchpad"
sys.path.insert(0, D)
LEAN52 = {"r_hip_pitch_act": 0.104, "l_hip_pitch_act": 0.104, "r_ankle_act": -0.084, "l_ankle_act": -0.084}
CASES = {"statue": {}, "lean0.25": {k: v * 0.25 for k, v in LEAN52.items()}, "lean0.5": {k: v * 0.5 for k, v in LEAN52.items()},
         "hip+0.02": {"r_hip_pitch_act": 0.02, "l_hip_pitch_act": 0.02}}
def job(args):
    ref, case, seed = args
    from common import make_env, names
    from t2_pins import lean_rollout, make_fn
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    u.leg_home_pose_reference = ref
    _, an = names(u)
    rec = lean_rollout(env, u, make_fn(dict(pins=CASES[case]), u.model.nu, an), seed)
    return dict(ref=ref, case=case, seed=seed, n=rec["n"], ret=float(rec["R"].sum()), leg=float(rec["T"]["reward_leg_home_pose"].sum()))
if __name__ == "__main__":
    jobs = [(ref, c, s) for ref in ("keyframe", "settled") for c in CASES for s in (None, 0, 1)]
    with Pool(2) as p:
        res = p.map(job, jobs)
    for ref in ("keyframe", "settled"):
        base = np.mean([r["ret"] for r in res if r["ref"] == ref and r["case"] == "statue"])
        bl = np.mean([r["leg"] for r in res if r["ref"] == ref and r["case"] == "statue"])
        print(ref, "statue %.1f" % base, " ".join("| %s Δret %+.1f (leg_home_pose %+.1f, stand %d/3)" % (c, np.mean([r["ret"] for r in res if r["ref"] == ref and r["case"] == c]) - base,
              np.mean([r["leg"] for r in res if r["ref"] == ref and r["case"] == c]) - bl, sum(r["n"] >= 1000 for r in res if r["ref"] == ref and r["case"] == c)) for c in CASES if c != "statue"))
