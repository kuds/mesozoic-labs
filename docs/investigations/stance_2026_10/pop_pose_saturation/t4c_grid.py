"""Track 4: static return surface over (leg lean scale x neck command) — is the lean+neck counterweight a coordinate-wise trap?"""
import json
import sys
import numpy as np
from multiprocessing import Pool

D = "scratchpad"
sys.path.insert(0, D)
SEEDS = [None, 0, 1]
LEAN52 = {"r_hip_pitch_act": 0.104, "l_hip_pitch_act": 0.104, "r_ankle_act": -0.084, "l_ankle_act": -0.084}
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
NECKS = (0.0, -0.25, -0.5, -0.75, -0.9, -1.0)


def job(args):
    s, nk, seed = args
    from common import make_env, names
    from t2_pins import lean_rollout, make_fn
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    _, an = names(u)
    pins = {k: v * s for k, v in LEAN52.items()}
    pins["neck_pitch_act"] = nk
    rec = lean_rollout(env, u, make_fn(dict(pins=pins), u.model.nu, an), seed)
    out = dict(s=s, neck=nk, seed=seed, n=rec["n"], end=rec["end"], ret=float(rec["R"].sum()))
    if rec["n"] >= 1000:
        out["w200"] = {k: float(v[200:].sum()) for k, v in rec["T"].items()}
        out["w200"]["_total"] = float(rec["R"][200:].sum())
        out["comx"] = float(1e3 * rec["comx"][300:400].mean())
    return out


if __name__ == "__main__":
    jobs = [(s, nk, seed) for s in SCALES for nk in NECKS for seed in SEEDS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=3)
    json.dump(res, open(f"{D}/t4c_grid.json", "w"))
    print("done", len(res))
