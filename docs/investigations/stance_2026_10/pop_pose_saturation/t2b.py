"""Track 2/4/5 follow-up: static leg-offset tolerance, neck/tail counterweight tests, early lean unwinds."""
import json
import sys
import numpy as np
from multiprocessing import Pool

D = "scratchpad"
sys.path.insert(0, D)
SEEDS = [None, 0, 1, 2, 3]
LEAN52 = {"r_hip_pitch_act": 0.104, "l_hip_pitch_act": 0.104, "r_ankle_act": -0.084, "l_ankle_act": -0.084}
LEAN60 = {"r_hip_pitch_act": 0.114, "l_hip_pitch_act": 0.114, "r_ankle_act": -0.081, "l_ankle_act": -0.081}


def job(args):
    label, spec, seed = args
    from common import make_env, names
    from t2_pins import lean_rollout, make_fn
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    _, an = names(u)
    rec = lean_rollout(env, u, make_fn(spec, u.model.nu, an), seed)
    n = rec["n"]
    out = dict(label=label, seed=seed, n=n, end=rec["end"], ret=float(rec["R"].sum()))
    if n >= 100:
        w = slice(60, 100)
        out["early"] = dict(comx_mm=float(1e3 * rec["comx"][w].mean()), tilt_deg=float(np.degrees(rec["tilt"][w].mean())),
                            fz=float(rec["fz"][w].mean()), z=float(rec["z"][w].mean()),
                            q_deg={an[i]: float(np.degrees(rec["q"][w, i].mean())) for i in range(len(an))})
    if n >= 1000:
        out["w200_1000"] = {k: float(v[200:].sum()) for k, v in rec["T"].items()}
        out["w200_1000"]["_total"] = float(rec["R"][200:].sum())
        w = slice(300, 400)
        out["ss"] = dict(comx_mm=float(1e3 * rec["comx"][w].mean()), tilt_deg=float(np.degrees(rec["tilt"][w].mean())),
                         fz=float(rec["fz"][w].mean()), z=float(rec["z"][w].mean()),
                         q_deg={an[i]: float(np.degrees(rec["q"][w, i].mean())) for i in range(len(an))})
    return out


if __name__ == "__main__":
    from common import make_env, names
    env, u = make_env()
    _, AN = names(u)
    m = u.model
    home = u._home_ctrl.copy()
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    deg_per_unit = {AN[i]: (float(np.degrees(home[i] - lo[i])), float(np.degrees(hi[i] - home[i]))) for i in range(m.nu)}
    env.close()
    jobs = [("statue", dict(pins={}), s) for s in SEEDS]
    groups = {
        "bi_hip_pitch": {"r_hip_pitch_act": 1, "l_hip_pitch_act": 1},
        "bi_knee": {"r_knee_act": 1, "l_knee_act": 1},
        "bi_ankle": {"r_ankle_act": 1, "l_ankle_act": 1},
        "bi_roll_mirror": {"r_hip_roll_act": 1, "l_hip_roll_act": -1},
        "bi_roll_same": {"r_hip_roll_act": 1, "l_hip_roll_act": 1},
        "r_hip_pitch": {"r_hip_pitch_act": 1},
        "r_hip_roll": {"r_hip_roll_act": 1},
        "r_knee": {"r_knee_act": 1},
        "r_ankle": {"r_ankle_act": 1},
    }
    for g, pat in groups.items():
        for x in (0.02, 0.05, 0.1, -0.02, -0.05, -0.1):
            jobs += [(f"tol:{g}:{x:+.2f}", dict(pins={k: v * x for k, v in pat.items()}), s) for s in SEEDS]
    combos = {
        "LEAN52": LEAN52, "LEAN52+neck-1": {**LEAN52, "neck_pitch_act": -1}, "LEAN52+neck-0.5": {**LEAN52, "neck_pitch_act": -0.5},
        "LEAN60+neck-1": {**LEAN60, "neck_pitch_act": -1}, "LEAN52+tail3+1": {**LEAN52, "tail_3_pitch_act": 1},
        "LEAN52+head-1": {**LEAN52, "head_pitch_act": -1}, "LEAN52x0.5+neck-1": {**{k: v * 0.5 for k, v in LEAN52.items()}, "neck_pitch_act": -1},
        "NECK-1": {"neck_pitch_act": -1},
    }
    for c, pat in combos.items():
        jobs += [(f"cb:{c}", dict(pins=pat), s) for s in SEEDS]
    for tu in (60, 100):
        for T in (1, 10, 50):
            jobs += [(f"unw:LEAN60:t{tu}:T{T}", dict(pins=LEAN60, unwind=(tu, T)), s) for s in SEEDS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=4)
    json.dump(dict(deg_per_unit=deg_per_unit, res=res), open(f"{D}/t2b.json", "w"))
    print("done", len(res))
