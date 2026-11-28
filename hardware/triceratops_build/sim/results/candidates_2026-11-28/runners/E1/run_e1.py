"""E1 clash reach (design/PREREG_CANDIDATES_2026_11_28.md, section 4 E1), runner. Uses only sim/candidate_checks.py's
frozen code for the episodes, logging, geometry, stops and verdict. Scratch: candidates/E1/ in the session scratch.

Stages (each resumable; an existing per-episode file is kept, never rerun):
  episodes  : the 224 policy episodes (4 fore128 policies x 7 conditions x seeds 700000-700003, 710000-710003) and the
              24 trot episodes (J1-J4, push back 0.5, push right 0.5; seeds 101-104), each with a StepLogger
  clearance : #9 (both sides) and #10 clearance, and each with the soles aside, at every logged pose
  stops     : every candidate stop (17) and the coupled limit (#9, #10) on every controller's pooled samples
  summarise : e1_verdict, the reported tables, written to sim/results/candidates_2026-11-28/E1/
Usage: python run_e1.py {episodes,clearance,stops,summarise,probe} [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

BUILD = Path("/home/user/mesozoic-labs/hardware/triceratops_build")
sys.path.insert(0, str(BUILD / "sim"))
import numpy as np  # noqa: E402

import candidate_checks as C  # noqa: E402

RUNNER = Path(__file__).resolve()
SCR = C.CAND_SCRATCH / "E1"
EPI = SCR / "episodes"
CLR = SCR / "clearance"
STP = SCR / "stops"
OUT = C.RESULTS / "E1"
CONTROLLERS = list(C.E1_POLICIES) + ["trot"]
TROT_CONDS = ("J1", "J2", "J3", "J4") + tuple(C.TROT_PUSHES)


def jobs():
    out = []
    for run in C.E1_POLICIES:
        for cond in C.POLICY_CONDITIONS:
            for blk, seeds in C.E1_BLOCKS.items():
                for s in seeds:
                    out.append((run, cond, blk, s))
    for cond in TROT_CONDS:
        for s in C.E1_TROT_SEEDS:
            out.append(("trot", cond, "trot", s))
    return out


def ep_name(ctrl, cond, seed):
    return f"{ctrl}__{cond}__{seed}"


class LazyLogger:
    """A StepLogger built on the episode's own env at its first reset (the trot's episode functions make their env
    inside; the logger needs it for the joint addresses and the actuator order)."""

    def __init__(self):
        self.lg = None

    def on_reset(self, env):
        if self.lg is None:
            self.lg = C.StepLogger(env)
        self.lg.on_reset(env)

    def __call__(self, env):
        self.lg(env)

    def result(self):
        return self.lg.result()


_W = {}


def _init():
    import torch
    torch.set_num_threads(1)


def _policy_env(run):
    if run not in _W:
        ec = C.run_env_config(run)
        assert Path(ec["model_path"]).name == C.MODEL.name, ec["model_path"]
        env = C.make_env(C.policy_env_config(ec, C.MODEL, physics_dt=0.001))
        stats, policy = C.load_policy(run, env)
        _W[run] = (env, stats, policy)
    return _W[run]


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def run_job(job):
    ctrl, cond, blk, seed = job
    name = ep_name(ctrl, cond, seed)
    npz, js = EPI / f"{name}.npz", EPI / f"{name}.json"
    if npz.exists() and js.exists():
        return name, "kept"
    t0 = time.time()
    if ctrl == "trot":
        walk60 = C._walk60()
        P = walk60.WalkParams()
        lg = LazyLogger()
        if cond in ("J1", "J2", "J3", "J4"):
            rec = C.trot_judged_episode(C.MODEL, cond, seed, P, 0.0, extra_hooks=(lg,))
        else:
            axis, dv = C.TROT_PUSHES[cond]
            rec = C.trot_push_episode(C.MODEL, axis, dv, seed, P, 0.0, extra_hooks=(lg,))
        res = lg.result()
        fell = bool(rec["fell"])
        term = rec["termination"]
    else:
        env, stats, policy = _policy_env(ctrl)
        lg = C.StepLogger(env)
        rec = C.policy_episode(env, seed, cond, policy=policy, stats=stats, hooks=(lg,))
        res = lg.result()
        fell = bool(rec["fell"])
        term = rec["termination"]
    np.savez_compressed(npz, t=res["t"], q=res["q"], sq_motor=res["sq_motor"], sq_brake=res["sq_brake"],
                        drive_sq=res["drive_sq"], rpm_hist=res["rpm_hist"].astype(np.int32),
                        rpm_peak=res["rpm_peak"], r=res["r"], eta=res["eta"],
                        window_samples=np.int64(res["window_samples"]),
                        q_nonfinite_dropped=np.int64(res["q_nonfinite_dropped"]), rpm_bin=np.float64(res["rpm_bin"]))
    meta = {"controller": ctrl, "condition": cond, "block": blk, "seed": seed, "fell": fell, "termination": term,
            "unstable": term == C.UNSTABLE, "joints": res["joints"], "actuators": res["actuators"],
            "n_poses": int(len(res["t"])), "t_last_pose_s": float(res["t"][-1]) if len(res["t"]) else None,
            "q_nonfinite_dropped": int(res["q_nonfinite_dropped"]), "window_samples": int(res["window_samples"]),
            "wall_s": time.time() - t0, "record": _jsonable(rec)}
    js.write_text(json.dumps(meta, default=str))
    return name, f"{'FELL ' + term if fell else 'ok'} {time.time() - t0:.1f}s"


_G = {}


def _geom():
    if "g" not in _G:
        _G["g"] = C.ClashGeometry()
    return _G["g"]


def clear_job(name):
    out = CLR / f"{name}.npz"
    if out.exists():
        return name, "kept"
    t0 = time.time()
    q = np.load(EPI / f"{name}.npz")["q"]
    b = _geom().batch(q)
    np.savez_compressed(out, **{k: v for k, v in b.items()})
    return name, f"{len(q)} poses {time.time() - t0:.1f}s"


def pooled(ctrl, fallen_out=False):
    names, Q, B, fell, idx = [], [], {}, [], []
    for job in jobs():
        if job[0] != ctrl:
            continue
        n = ep_name(job[0], job[1], job[3])
        meta = json.loads((EPI / f"{n}.json").read_text())
        if fallen_out and meta["fell"]:
            continue
        q = np.load(EPI / f"{n}.npz")["q"].astype(np.float64)
        c = np.load(CLR / f"{n}.npz")
        Q.append(q)
        for k in c.files:
            B.setdefault(k, []).append(c[k])
        names.append(n)
        fell.append(meta["fell"])
        idx.append(np.arange(len(q)))
    Q = np.vstack(Q)
    B = {k: np.concatenate(v) for k, v in B.items()}
    ep_of = np.concatenate([np.full(len(i), j) for j, i in enumerate(idx)])
    return names, Q, B, ep_of


def stop_job(arg):
    ctrl, sid = arg
    out = STP / f"{ctrl}__{sid.replace(':', '_').replace(' ', '_').replace('/', '_')}.json"
    if out.exists():
        return arg, "kept"
    t0 = time.time()
    _, Q, B, _ = pooled(ctrl)
    stop = [s for s in C.candidate_stops() if s["id"] == sid][0]
    r = C.evaluate_stop(_geom(), Q, B, stop)
    r["wall_s"] = time.time() - t0
    out.write_text(json.dumps(r))
    return arg, f"clipped {r['clipped']} {time.time() - t0:.1f}s"


def pool_map(fn, items, workers):
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for i, (k, msg) in enumerate(pool.imap_unordered(fn, items, chunksize=1)):
            print(f"[{i + 1}/{len(items)} {time.time() - t0:.0f}s] {k}: {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("episodes", "clearance", "stops", "summarise", "probe"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    assert a.workers <= 2
    for p in (EPI, CLR, STP):
        p.mkdir(parents=True, exist_ok=True)
    if a.stage == "probe":
        return probe()
    if a.stage == "episodes":
        js = jobs()
        assert len(js) == 248, len(js)
        # the trot's 24 first (no policy load), then the policies run by run so a worker keeps one env and policy
        js = [j for j in js if j[0] == "trot"] + [j for j in js if j[0] != "trot"]
        pool_map(run_job, js, a.workers)
    elif a.stage == "clearance":
        names = [ep_name(j[0], j[1], j[3]) for j in jobs()]
        pool_map(clear_job, names, a.workers)
    elif a.stage == "stops":
        stops = [s["id"] for s in C.candidate_stops()]
        items = [(c, s) for c in CONTROLLERS for s in stops]
        pool_map(stop_job, items, a.workers)
    elif a.stage == "summarise":
        import summarise_e1
        summarise_e1.summarise()
    return 0


def probe():
    """Timing only, on seeds outside every judged set (seed 1, 2 s), no clearance printed."""
    run = "fore128-s0"
    t0 = time.time()
    env, stats, policy = _policy_env(run)
    env2 = C.make_env(env.cfg.replace(episode_seconds=2.0))
    t1 = time.time()
    lg = C.StepLogger(env2)
    rec = C.policy_episode(env2, 1, "walk_0.25", policy=policy, stats=stats, hooks=(lg,))
    t2 = time.time()
    res = lg.result()
    t3 = time.time()
    g = _geom()
    t35 = time.time()
    g.batch(res["q"][:100])
    t4 = time.time()
    print(f"load {t1 - t0:.1f}s; 2 s episode {t2 - t1:.1f}s ({len(res['q'])} poses); geometry init {t35 - t3:.1f}s;"
          f" 100 poses clearance {t4 - t35:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
