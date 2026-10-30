"""Tracks 2/4/5: actuator pins (null-space), static displaced poses, and unwinding transients (statue-based)."""
import json
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "scratchpad")
OUT = "scratchpad/t2"
SEEDS = [None, 0, 1, 2, 3, 4]  # None = noise-free reset


def lean_rollout(env, u, act_fn, seed, steps=1000):
    from common import custom_reset, term_keys
    if seed is None:
        obs, _ = custom_reset(env, u, 0)
    else:
        obs, _ = env.reset(seed=seed)
    m, d = u.model, u.data
    root = int(m.jnt_bodyid[0])
    feet = [m.body(n).id for n in ("r_metatarsus", "l_metatarsus", "r_toe_d3", "l_toe_d3")]
    T, R, TILT, FZ, Z, COMX, COMZ, AF, Q, V = {}, [], [], [], [], [], [], [], [], []
    end = "truncated"
    for t in range(steps):
        a = np.clip(np.asarray(act_fn(t), dtype=np.float32), -1, 1)
        obs, r, term, trunc, info = env.step(a)
        R.append(r)
        for k in term_keys(info):
            T.setdefault(k, []).append(info[k])
        TILT.append(info["tilt_angle"]); FZ.append(info["forward_z"]); Z.append(d.qpos[2])
        sup = d.xpos[feet].mean(0)
        COMX.append(d.subtree_com[root][0] - sup[0]); COMZ.append(d.subtree_com[root][2])
        AF.append(d.actuator_force.copy()); V.append(np.linalg.norm(d.qvel[0:3]))
        Q.append(np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] for i in range(m.nu)]))
        if term or trunc:
            end = info.get("termination_reason", "truncated") if term else "truncated"
            break
    return dict(R=np.array(R), T={k: np.array(v) for k, v in T.items()}, tilt=np.array(TILT), fz=np.array(FZ),
                z=np.array(Z), comx=np.array(COMX), comz=np.array(COMZ), af=np.array(AF), q=np.array(Q),
                v=np.array(V), n=len(R), end=end)


def make_fn(spec, nu, an):
    """spec: dict(pins={name: value}, t_on=0, ramp=(t0,t1) or None, unwind=(t_u, T) or None)."""
    vec = np.zeros(nu)
    for k, v in spec["pins"].items():
        vec[an.index(k)] = v
    ramp = spec.get("ramp")
    unwind = spec.get("unwind")

    def f(t):
        s = 1.0
        if ramp is not None:
            t0, t1 = ramp
            s = float(np.clip((t - t0) / max(t1 - t0, 1), 0, 1))
        if unwind is not None:
            tu, T = unwind
            if t >= tu:
                s *= float(np.clip(1.0 - (t - tu + 1) / max(T, 1), 0, 1))
        return vec * s
    return f


def job(args):
    label, spec, seed = args
    from common import make_env, names
    global _ENV
    if "_ENV" not in globals():
        _ENV = make_env()
    env, u = _ENV
    _, an = names(u)
    rec = lean_rollout(env, u, make_fn(spec, u.model.nu, an), seed)
    fr = u.model.actuator_forcerange[:, 1]
    jr = np.array([u.model.jnt_range[u.model.actuator_trnid[i, 0]] for i in range(u.model.nu)])
    n = rec["n"]
    out = dict(label=label, seed=seed, n=n, end=rec["end"], ret=float(rec["R"].sum()))
    for w0, w1 in ((0, 50), (50, 200), (200, 400), (400, 700), (700, 1000)):
        if n >= w1:
            out[f"w{w0}_{w1}"] = {k: float(v[w0:w1].sum()) for k, v in rec["T"].items()}
            out[f"w{w0}_{w1}"]["_total"] = float(rec["R"][w0:w1].sum())
    if n >= 1000:
        ss = slice(300, 400)  # steady state while pinned (before any unwind at 400)
        out["ss"] = dict(
            tilt_deg=float(np.degrees(rec["tilt"][ss].mean())), fz=float(rec["fz"][ss].mean()),
            z=float(rec["z"][ss].mean()), comx_mm=float(1e3 * rec["comx"][ss].mean()), comz_mm=float(1e3 * rec["comz"][ss].mean()),
            vroot=float(rec["v"][ss].mean()),
            force_frac={an[i]: float(np.abs(rec["af"][ss, i]).mean() / fr[i]) for i in range(len(an))},
            q_deg={an[i]: float(np.degrees(rec["q"][ss, i].mean())) for i in range(len(an))},
            at_stop={an[i]: bool(min(abs(rec["q"][ss, i].mean() - jr[i, 0]), abs(rec["q"][ss, i].mean() - jr[i, 1])) < np.radians(1.0)) for i in range(len(an))},
        )
        out["tail"] = dict(tilt_deg=float(np.degrees(rec["tilt"][900:].mean())), comx_mm=float(1e3 * rec["comx"][900:].mean()),
                           z=float(rec["z"][900:].mean()), q_deg={an[i]: float(np.degrees(rec["q"][900:, i].mean())) for i in range(len(an))})
        # per-step reward trace around the unwind point (for transients)
        out["trace_390_520"] = rec["R"][390:520].round(5).tolist()
        out["trace_terms_390_520"] = {k: v[390:520].round(5).tolist() for k, v in rec["T"].items()
                                      if k in ("reward_speed", "reward_drift", "reward_posture", "reward_nosedive", "reward_height",
                                               "reward_spin", "reward_tail", "reward_alive", "reward_bilateral_support",
                                               "reward_smoothness", "reward_action_jerk", "reward_action_saturation", "reward_energy",
                                               "reward_tail_home_pose", "reward_neck_posture", "reward_leg_home_pose",
                                               "reward_foot_load_balance", "reward_head_clearance")}
    return out


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, "scratchpad")
    from common import make_env, names
    env, u = make_env()
    _, AN = names(u)
    env.close()
    which = sys.argv[1]
    jobs = []
    if which == "pins":
        jobs += [("statue", dict(pins={}), s) for s in SEEDS]
        for a in AN:
            for sgn in (1.0, -1.0):
                jobs += [(f"pin:{a}:{sgn:+.0f}", dict(pins={a: sgn}), s) for s in SEEDS]
    elif which == "patterns":
        P491 = {"head_pitch_act": -1, "r_hip_roll_act": 1, "l_hip_roll_act": -1, "tail_1_pitch_act": 1,
                "tail_1_yaw_act": 1, "tail_2_pitch_act": 1, "tail_3_pitch_act": 1}
        P491_NOROLL = {k: v for k, v in P491.items() if "hip_roll" not in k}
        LEAN52 = {"r_hip_pitch_act": 0.104, "l_hip_pitch_act": 0.104, "r_ankle_act": -0.084, "l_ankle_act": -0.084}
        LEAN60 = {"r_hip_pitch_act": 0.114, "l_hip_pitch_act": 0.114, "r_ankle_act": -0.081, "l_ankle_act": -0.081}
        pats = {
            "statue": {}, "TAIL3_RAIL": {"tail_3_pitch_act": 1.0}, "NECK_RAIL": {"neck_pitch_act": -1.0},
            "P491": P491, "P491_noroll": P491_NOROLL, "LEAN52": LEAN52, "LEAN60": LEAN60,
            "TAIL3_08": {"tail_3_pitch_act": 0.8},
        }
        for name, pins in pats.items():
            jobs += [(name, dict(pins=pins), s) for s in SEEDS]
        # documented within-episode onset profile for the tail rail (0 until ~40, ~1 by ~216)
        jobs += [("TAIL3_RAIL_onsetramp", dict(pins={"tail_3_pitch_act": 1.0}, ramp=(40, 216)), s) for s in SEEDS]
        # random static DC offsets (passive stability of displaced poses)
        rng = np.random.default_rng(7)
        for rms in (0.1, 0.2, 0.3, 0.5):
            for k in range(6):
                v = rng.normal(0, rms, size=len(AN))
                jobs += [(f"RAND_rms{rms}_{k}", dict(pins={a: float(x) for a, x in zip(AN, v)}), s) for s in (None, 0)]
        # unwinding experiments
        for name in ("TAIL3_RAIL", "NECK_RAIL", "P491_noroll", "P491", "LEAN60"):
            for T in (1, 10, 50, 200):
                jobs += [(f"UNWIND:{name}:T{T}", dict(pins=pats[name], unwind=(400, T)), s) for s in SEEDS]
        # re-pin (reverse direction) for symmetry: statue for 400 steps then ramp into the pin over T
        for name in ("TAIL3_RAIL", "NECK_RAIL"):
            for T in (1, 50):
                jobs += [(f"REPIN:{name}:T{T}", dict(pins=pats[name], ramp=(400, 400 + T)), s) for s in SEEDS]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=4)
    with open(f"{OUT}/{which}.json", "w") as f:
        json.dump(res, f)
    print(which, len(res), "episodes")
