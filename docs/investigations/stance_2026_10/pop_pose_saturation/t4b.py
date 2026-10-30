"""Track 4: unwinding a statically-stable counterbalanced pair (leg lean + neck counterweight) together vs one half."""
import json
import sys
import numpy as np
from multiprocessing import Pool

D = "scratchpad"
sys.path.insert(0, D)
SEEDS = [None, 0, 1, 2, 3]
LEAN52 = {"r_hip_pitch_act": 0.104, "l_hip_pitch_act": 0.104, "r_ankle_act": -0.084, "l_ankle_act": -0.084}
LEAN60 = {"r_hip_pitch_act": 0.114, "l_hip_pitch_act": 0.114, "r_ankle_act": -0.081, "l_ankle_act": -0.081}
PAIRS = {"A": {**{k: v * 0.5 for k, v in LEAN52.items()}, "neck_pitch_act": -1.0}, "B": {**LEAN60, "neck_pitch_act": -1.0}}
LEGS = list(LEAN52)


def make_fn(pins, keys, tu, T, an, nu):
    vec = np.zeros(nu)
    mask = np.zeros(nu)
    for k, v in pins.items():
        vec[an.index(k)] = v
    for k in keys:
        mask[an.index(k)] = 1.0

    def f(t):
        if tu is None or t < tu:
            return vec
        s = float(np.clip(1.0 - (t - tu + 1) / max(T, 1), 0, 1))
        return vec * (1 - mask) + vec * mask * s
    return f


def job(args):
    label, pins, keys, tu, T, seed = args
    from common import make_env, names
    from t2_pins import lean_rollout
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    _, an = names(u)
    rec = lean_rollout(env, u, make_fn(pins, keys, tu, T, an, u.model.nu), seed)
    out = dict(label=label, seed=seed, n=rec["n"], end=rec["end"], ret=float(rec["R"].sum()))
    out["R"] = rec["R"].round(5).tolist()
    out["T"] = {k: v.round(5).tolist() for k, v in rec["T"].items()}
    out["comx"] = (1e3 * rec["comx"]).round(2).tolist()
    return out


if __name__ == "__main__":
    jobs = [("statue", {}, [], None, 1, s) for s in SEEDS]
    for p, pins in PAIRS.items():
        jobs += [(f"{p}:hold", pins, [], None, 1, s) for s in SEEDS]
        for mode, keys in (("both", list(pins)), ("neck_only", ["neck_pitch_act"]), ("legs_only", LEGS)):
            for T in (1, 50, 200):
                jobs += [(f"{p}:{mode}:T{T}", pins, keys, 400, T, s) for s in SEEDS]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=3)
    json.dump(res, open(f"{D}/t4b.json", "w"))
    print("done", len(res))
