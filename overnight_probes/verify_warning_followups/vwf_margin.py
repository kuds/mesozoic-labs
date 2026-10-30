"""Verifier's own re-run of the margin-to-rail sweep (warning_margin PREREG.txt design), written independently of
warning_margin/wm_common.py and wm_run.py.  No training; checkpoints read only; deterministic closed loop.

    PYTHONPATH=<r10round/ecs/code> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 venv/bin/python -B vwf_margin.py \
        --plan run:steps:seed0:seed1:obs12 [...] --out vwf_margin_episodes.jsonl

Differences from the original harness (deliberate, to test it):
  - the bias is applied as a PRE-CLIPPED command (a[14] = clip(own14 + b, -1, 1)); the original passes own14 + b and
    relies on the env's clip (TyrannosaurusRexEnv._filter_action / BaseDinoEnv._scale_action).  Identical results
    confirm that path.
  - window readout, parked rule, bisection and M are recomputed here from the per-step command array.
Definitions as warning_margin/PREREG.txt section 2: window = steps 501-1000; railed step = command >= 0.99; PARKED =
more than 250 railed window steps; BIAS bracket (0, 2.0], 7 halvings, T_s = upper end; OBS 12 deg run once per seed.
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

torch.set_num_threads(1)
F2 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad/foot/stance6/fetch/F2")
OBS_T3, ACT_T3 = 6, 14


def paths(run, steps):
    m = F2 / f"stance6_r10LTSB_{run}" / "copies" / "01_stance" / "models"
    if steps == 12_001_280:
        return m / "stage1_final.zip.copy1", m / "stage1_final_vecnorm.pkl.copy1"
    return m / f"stage1_{steps}_steps.zip.copy1", m / f"stage1_vecnormalize_{steps}_steps.pkl.copy1"


def run_episode(env, model, norm, seed, kind, value):
    obs, _ = env.reset(seed=seed)
    cmds = np.full(1000, np.nan)
    owns = np.full(1000, np.nan)
    n_steps = 0
    terminated = False
    for i in range(1000):
        seen = obs.copy()
        if kind == "obs":
            seen[OBS_T3] = seen[OBS_T3] + math.radians(value)
        act = np.array(model.predict(norm.normalize_obs(seen), deterministic=True)[0], dtype=np.float32, copy=True)
        own = float(act[ACT_T3])
        if kind == "bias":
            act[ACT_T3] = np.float32(min(1.0, max(-1.0, own + value)))
        owns[i] = own
        cmds[i] = min(1.0, max(-1.0, float(act[ACT_T3])))
        obs, _r, term, trunc, _ = env.step(act)
        n_steps = i + 1
        if term or trunc:
            terminated = bool(term)
            break
    w = cmds[500:1000]
    n_railed = int(np.sum(w >= 0.99))  # NaN (missing steps) compares False -> not railed
    return {"seed": seed, "kind": kind, "value": value, "n_steps": n_steps, "terminated": terminated,
            "n_railed": n_railed, "parked": n_railed > 250,
            "cmd_mean_win": float(np.nanmean(w)) if np.isfinite(w).any() else None,
            "own_mean_win": float(np.nanmean(owns[500:1000])) if np.isfinite(owns[500:1000]).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    kw = dict(load_stage_config(mid, "stance")["env_kwargs"])

    def factory():
        return env_class(mid)(**kw)

    env = factory()
    assert env.unwrapped.model.actuator(ACT_T3).name == "tail_3_pitch_act"
    out = open(a.out, "a")
    for item in a.plan:
        run, steps, s0, s1, obs12 = item.split(":")
        steps, s0, s1, obs12 = int(steps), int(s0), int(s1), obs12 == "1"
        z, p = paths(run, steps)
        model, norm, _ = load_sb3_checkpoint(str(z), str(p), factory)
        t0 = time.time()
        for seed in range(s0, s1 + 1):
            def ep(kind, value):
                r = run_episode(env, model, norm, seed, kind, value)
                r.update(run=run, ckpt=steps)
                out.write(json.dumps(r) + "\n")
                out.flush()
                return r
            base = ep("none", 0.0)
            if base["parked"]:
                T = 0.0
            else:
                lo, hi = 0.0, 2.0
                for _ in range(7):
                    mid_v = 0.5 * (lo + hi)
                    if ep("bias", mid_v)["parked"]:
                        hi = mid_v
                    else:
                        lo = mid_v
                T = hi
            o12 = ep("obs", 12.0)["parked"] if obs12 else None
            print(f"{run} {steps} seed {seed}: base {base['cmd_mean_win']:+.4f} T_bias {T:.6f} obs12 parked {o12} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
