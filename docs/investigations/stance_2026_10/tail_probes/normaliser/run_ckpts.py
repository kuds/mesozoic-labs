"""Step 2 rollouts: every checkpoint of every continuation (and the start pair once per lineage), own statistics,
deterministic, unforced, 1,000 steps, seed 3042 (the verifier's sweep seed) unless --seeds says otherwise.

PYTHONPATH=<policies/code_copy>:<this folder> python -B run_ckpts.py OUT_DIR [--lin s42,s80] [--seeds 3042] [--only LABEL,...]

Writes OUT_DIR/episodes.jsonl (one row per checkpoint and seed, with the own-statistics loop-gain sweep on that
episode's states) and OUT_DIR/states/<label>_<seed>.npy (the 6 observations at steps 500..1000, float32), so the
statistics swaps can be evaluated on these states afterwards without simulating again.  No training; read only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from probe_common import NAMES, SE, START_RUN, Agent, episode, label, make_factory, steps_of, sweep_states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--lin", default="s42,s80")
    ap.add_argument("--seeds", default="3042")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    only = {w for w in a.only.split(",") if w}
    (a.out / "states").mkdir(parents=True, exist_ok=True)
    factory = make_factory()
    done = set()
    if (a.out / "episodes.jsonl").exists():
        for line in open(a.out / "episodes.jsonl"):
            r = json.loads(line)
            done.add((r["label"], r["seed"]))
    out = open(a.out / "episodes.jsonl", "a")
    todo = []
    for lin in a.lin.split(","):
        todo.append((START_RUN[lin], SE[lin][0]))
        for run in NAMES:
            if run.startswith(lin):
                todo += [(run, st) for st in steps_of(lin)[1:]]
    for run, st in todo:
        lab = label(run, st)
        if only and lab not in only:
            continue
        if all((lab, s) in done for s in seeds):
            continue
        t0 = time.time()
        agent = Agent(run, st, factory)
        for seed in seeds:
            if (lab, seed) in done:
                continue
            env = factory()
            row, states = episode(env, agent, seed)
            env.close()
            np.save(a.out / "states" / f"{lab}_{seed}.npy", states)
            sw = sweep_states(agent, states) if len(states) else None
            row.update({"label": lab, "run": run, "step": st, "lin": run[:3], "stats": "own", "sweep": sw,
                        "n_states": int(len(states))})
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"{lab}: {len(seeds)} ep {time.time() - t0:.1f} s gain {sw['loop_gain'] if sw else None}", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
