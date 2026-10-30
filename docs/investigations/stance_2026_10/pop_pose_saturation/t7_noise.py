"""Track 4/5: statue (mean action 0) under per-step Gaussian exploration noise, by channel group; statue fall rate at reset noise."""
import json
import sys
import numpy as np
from multiprocessing import Pool

D = "scratchpad"
sys.path.insert(0, D)
GROUPS = {
    "all": None,
    "legs": ["r_hip_pitch_act", "r_hip_roll_act", "r_knee_act", "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act"],
    "neck_head": ["neck_pitch_act", "neck_yaw_act", "head_pitch_act"],
    "tail": ["tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"],
}


def job(args):
    kind, group, sigma, seed = args
    from common import make_env, names, term_keys
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    _, an = names(u)
    mask = np.ones(u.model.nu) if GROUPS.get(group) is None else np.array([1.0 if a in GROUPS[group] else 0.0 for a in an])
    rng = np.random.default_rng(seed * 7 + 13)
    obs, _ = env.reset(seed=seed)
    R, end, n = [], "truncated", 0
    terms = {}
    tilt = []
    for t in range(1000):
        a = np.clip(rng.normal(0.0, sigma, u.model.nu) * mask, -1, 1) if sigma > 0 else np.zeros(u.model.nu)
        obs, r, term, trunc, info = env.step(a.astype(np.float32))
        R.append(r)
        tilt.append(info.get("tilt_angle", np.nan))
        for k in term_keys(info):
            terms[k] = terms.get(k, 0.0) + float(info[k])
        n += 1
        if term or trunc:
            end = info.get("termination_reason", "truncated") if term else "truncated"
            break
    return dict(kind=kind, group=group, sigma=sigma, seed=seed, n=n, end=end, ret=float(np.sum(R)), terms=terms,
                max_tilt_deg=float(np.degrees(np.nanmax(tilt))))


if __name__ == "__main__":
    jobs = [("statue100", "none", 0.0, s) for s in range(1000, 1100)]
    for g in GROUPS:
        for sg in (0.02, 0.05, 0.08, 0.11, 0.14, 0.22):
            jobs += [("noise", g, sg, s) for s in range(200, 210)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=5)
    json.dump(res, open(f"{D}/t7_noise.json", "w"))
    print("done", len(res))
