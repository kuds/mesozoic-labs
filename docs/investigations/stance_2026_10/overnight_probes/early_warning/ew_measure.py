"""EARLY-WARNING CURVE: measure one fixed set of loop readouts at every checkpoint of the named runs.
NO TRAINING: checkpoints are opened read only; simulation (deterministic rollouts) and network evaluation only.

    PYTHONPATH=<r10round/ecs/code>:<this folder> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 \
        venv/bin/python -B ew_measure.py OUT.jsonl RUN[,RUN...] [--seeds 5001-5010] [--steps N,N]

Per checkpoint (definitions in ew_common.py):
  REF  = sweeps on the probe bank's 10 fixed raw observations (no simulation);
  OWN  = 10 deterministic 1,000-step rollouts (seeds 5001-5010, env.reset(seed=s)); per episode the own (clipped)
         tail_3 command mean over steps 501-1000, the raw mu14 at the snapshots, the tail_3 angle and the pelvis
         pitch (qpos quaternion, + nose up) means over steps 501-1000, 25-step bins of the own command over the
         episode; then the same sweeps on the 60 snapshot states.
Checkpoints whose weights and normaliser are byte-identical to one already measured (the ECS starts) are not
re-run: the row is copied with "dup_of".  CPU seconds (time.process_time) are recorded per part.
Writes OUT.jsonl (one row per checkpoint) and states/<run>_<steps>.npz (snapshot states, float64).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import ew_common as C

torch.set_num_threads(1)


def rollouts(env, net: C.Net, seeds) -> tuple[list[dict], np.ndarray]:
    u = env.unwrapped
    eps, X = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        own14, t3, pit = [], [], []
        term = trunc = False
        k = 0
        for k in range(1, C.HORIZON + 1):
            a = net.act(obs)
            own14.append(float(np.clip(a[C.A_T3], -1, 1)))
            obs, r, term, trunc, _ = env.step(a)
            t3.append(float(obs[C.I_T3]))
            pit.append(C.pitch_up_deg(np.array(u.data.qpos[3:7])))
            if k in C.SNAP_STEPS:
                X.append(np.array(obs, dtype=np.float64, copy=True))
            if term or trunc:
                break
        full = (k == C.HORIZON) and not term
        w = slice(500, 1000)
        eps.append({"seed": seed, "steps": k, "full": bool(full), "terminated": bool(term),
                    "own14_win": float(np.mean(own14[w])) if len(own14) > 500 else None,
                    "frac99_win": float(np.mean(np.abs(own14[w]) >= 0.99)) if len(own14) > 500 else None,
                    "tail3_deg_win": math.degrees(float(np.mean(t3[w]))) if len(t3) > 500 else None,
                    "pitch_deg_win": float(np.mean(pit[w])) if len(pit) > 500 else None,
                    "own14_bins25": [round(float(np.mean(own14[i:i + 25])), 5) for i in range(0, len(own14), 25)]})
    return eps, np.array(X, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("runs")
    ap.add_argument("--seeds", default="5001-5010")
    ap.add_argument("--steps", default="")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    want = {int(s) for s in a.steps.split(",") if s}
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    factory = lambda: env_class(mid)(**ctor)  # noqa: E731
    env = factory()
    assert env.unwrapped.model.actuator(C.A_T3).name == "tail_3_pitch_act"
    bank_obs, bank_labels = C.bank()
    done, by_key = set(), {}
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["run"], r["steps"]))
            by_key.setdefault(r["key"], r)
    (C.HERE / "states").mkdir(exist_ok=True)
    with open(a.out, "a") as fh:
        for run in a.runs.split(","):
            for steps, z, p in C.checkpoints(run):
                if want and steps not in want:
                    continue
                if (run, steps) in done:
                    continue
                key = C.ckpt_key(z, p)
                if key in by_key:
                    src = by_key[key]
                    row = dict(src)
                    row.update({"run": run, "steps": steps, "zip": str(z), "dup_of": [src["run"], src["steps"]]})
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"{run} {steps}: duplicate of {src['run']} {src['steps']}", flush=True)
                    continue
                t0, w0 = time.process_time(), time.time()
                net = C.Net(z, p, factory)
                t_load = time.process_time() - t0
                t1 = time.process_time()
                ref = C.measure_states(net, bank_obs)
                ref_ss = C.measure_states(net, bank_obs[bank_labels == "statue_settled"])
                mu_bank = net.mu(bank_obs)[:, C.A_T3]
                t_ref = time.process_time() - t1
                t2 = time.process_time()
                eps, X = rollouts(env, net, seeds)
                t_roll = time.process_time() - t2
                t3 = time.process_time()
                own = C.measure_states(net, X)
                t_own = time.process_time() - t3
                full = [e for e in eps if e["full"]]
                row = {"run": run, "steps": steps, "key": key, "zip": str(z), "seeds": seeds,
                       "n_episodes": len(eps), "n_full": len(full),
                       "own14_win_median": float(np.median([e["own14_win"] for e in eps if e["own14_win"] is not None])),
                       "own14_win_mean": float(np.mean([e["own14_win"] for e in eps if e["own14_win"] is not None])),
                       "frac99_win_median": float(np.median([e["frac99_win"] for e in eps if e["frac99_win"] is not None])),
                       "n_over_bar": int(sum((e["frac99_win"] or 0) > 0.10 for e in eps)),
                       "tail3_deg_win_median": float(np.median([e["tail3_deg_win"] for e in eps if e["tail3_deg_win"] is not None])),
                       "pitch_deg_win_median": float(np.median([e["pitch_deg_win"] for e in eps if e["pitch_deg_win"] is not None])),
                       "ref": ref, "ref_statue_settled": ref_ss, "own": own,
                       "mu14_bank": mu_bank.round(6).tolist(),
                       "episodes": eps,
                       "cpu_s": {"load": t_load, "ref_sweeps": t_ref, "rollouts": t_roll, "own_sweeps": t_own},
                       "wall_s": time.time() - w0}
                np.savez_compressed(C.HERE / "states" / f"{run}_{steps}.npz", X=X)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                by_key[key] = row
                print(f"{run} {steps}: own14 {row['own14_win_median']:+.3f} | OWN c0 {own['primary']['c0']:+.3f} "
                      f"G {own['primary']['G']:+.3f} FP {own['primary']['FP']:+.3f} pitch {own['pitch']['pitch_slope']:+.3f} | "
                      f"REF c0 {ref['primary']['c0']:+.3f} G {ref['primary']['G']:+.3f} FP {ref['primary']['FP']:+.3f} | "
                      f"cpu ref {t_ref:.2f} roll {t_roll:.1f} own {t_own:.2f} s", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
