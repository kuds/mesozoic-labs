"""Verifier's added test of the runner's inferred "self-reinforcing loop" (F12): does the policy's tail_3 command rise
with the tail_3 joint angle it observes, with everything else in its observation held fixed?

PYTHONPATH=<policies/code_copy> python -B verify_loop.py OUT.jsonl --who NAME[,NAME] [--seed 3042]

Simulation and network evaluation only; no training; checkpoints opened read only.  Uses verify_sim.Policy (this
folder).  Per policy: one 1,000-step episode unforced and one with tail_3 replaced as in the clamp (held 0 for s42
and start checkpoints, forced +1 for s80), D-9's override.  At steps 500, 600, ..., 1000 of each, the observation
is copied and only its tail_3_pitch joint-position entry (obs index qposadr - 7) is set to theta in 0..6 deg; the
policy's raw Gaussian mean (pre-clip) is read for tail_3 (14), head_pitch (2) and neck_pitch (0).  Slopes are
least-squares per state, then the median over the 12 states.  The plant's steady-state gain (deg of tail_3 per unit
command) is measured here from the statue's own tail_3 equilibrium (0 and +1) and recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from environments.shared.config import load_stage_config
from environments.shared.species_registry import env_class
from verify_sim import MID, Policy, paths

THETAS_DEG = np.arange(0.0, 6.01, 1.0)


def collect(env, pol, seed, force):
    obs, _ = env.reset(seed=seed)
    states = []
    for step in range(1, 1001):
        a = np.array(pol.act(obs), copy=True)
        if force is not None:
            a[14] = force
        obs, r, term, trunc, info = env.step(a)
        if step % 100 == 0 and step >= 500:
            states.append(np.array(obs, copy=True))
        if term or trunc:
            break
    return states


def sweep(pol, obs, idx):
    batch = np.repeat(obs[None, :], len(THETAS_DEG), axis=0)
    batch[:, idx] = np.radians(THETAS_DEG).astype(np.float32)
    nb = np.stack([pol.norm(b) for b in batch])
    with torch.no_grad():
        mean = pol.model.policy.get_distribution(torch.as_tensor(nb)).distribution.mean.numpy().astype(np.float64)
    out = {}
    for j, name in ((14, "tail3"), (2, "head_pitch"), (0, "neck_pitch")):
        slope, icpt = np.polyfit(THETAS_DEG, mean[:, j], 1)
        out[name] = {"slope_per_deg": float(slope), "at0": float(mean[0, j]), "at5": float(mean[5, j]),
                     "curve": mean[:, j].round(4).tolist()}
    return out


def plant_gain(ctor):
    env = env_class(MID)(**ctor)
    m = env.unwrapped.model
    adr = int(m.jnt_qposadr[m.joint("tail_3_pitch").id])
    res = {}
    for f in (0.0, 1.0):
        env.reset(seed=3042)
        a = np.zeros(15)
        a[14] = f
        q = []
        for step in range(1, 1001):
            env.step(a)
            if step > 500:
                q.append(env.unwrapped.data.qpos[adr])
        res[f] = math.degrees(float(np.mean(q)))
    env.close()
    return res, adr - 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--who", required=True)
    ap.add_argument("--seed", type=int, default=3042)
    a = ap.parse_args()
    ctor = dict(load_stage_config(MID, "stance")["env_kwargs"])
    gain, idx = plant_gain(ctor)
    with open(a.out, "a") as out:
        out.write(json.dumps({"plant_tail3_deg_at_cmd": gain, "obs_index_tail3": idx}) + "\n")
        for who in a.who.split(","):
            pol = Policy(*paths(who))
            clamp = 1.0 if who.startswith("s80") else 0.0
            for cond, force in (("unforced", None), ("clamped", clamp)):
                env = env_class(MID)(**ctor)
                states = collect(env, pol, a.seed, force)
                env.close()
                sw = [sweep(pol, s, idx) for s in states]
                row = {"who": who, "cond": cond, "force": force, "n_states": len(states),
                       "obs_tail3_deg_at_states": [math.degrees(float(s[idx])) for s in states]}
                for name in ("tail3", "head_pitch", "neck_pitch"):
                    row[name] = {k: float(np.median([x[name][k] for x in sw])) for k in ("slope_per_deg", "at0", "at5")}
                    row[name]["curve_median"] = np.median([x[name]["curve"] for x in sw], axis=0).round(4).tolist()
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(who, "done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
