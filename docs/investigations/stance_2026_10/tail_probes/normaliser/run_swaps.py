"""Step 3: statistics swaps in closed loop.  Deterministic, unforced, 1,000 steps, seeds 3042-3061.

PYTHONPATH=<policies/code_copy>:<this folder> python -B run_swaps.py OUT_DIR --lin s42 --conds own,endW_startN,...
                                                                   [--seeds 3042-3061] [--only RUN,...]

Conditions (per end run R of the lineage; S = the lineage's start pair, E = the end step):
  own                 R's end weights with R's end statistics (baseline; also the start pair with its own, once)
  endW_startN         R's end weights, all 63 dimensions normalised with S's statistics
  endW_startN_tail    R's end weights, its own statistics except the 8 tail dimensions (angles and velocities of
                      tail_1_pitch, tail_1_yaw, tail_2_pitch, tail_3_pitch: obs 3-6, 24-27) taken from S
  endW_startN_t3q     as above, only the tail_3 angle (obs 6) taken from S
  startW_endN         S's weights, all 63 dimensions normalised with R's end statistics
  startW_endN_tail    S's weights, its own statistics except the 8 tail dimensions taken from R's end
Per episode: return, window return (steps 200-1000), the policy's own tail_3 command (mean and share >= 0.99 over
steps 500-1000), tail_3 angle, reward terms; and the loop-gain sweep on that episode's own states under the condition's
statistics.  Writes OUT_DIR/episodes.jsonl and OUT_DIR/states/<cond>__<who>.npy (seeds x 6 x 63, float32).
No training; checkpoints read only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from probe_common import NAMES, SE, START_RUN, T3Q, TAIL_DIMS, Agent, episode, make_factory, stats, sweep_states

CONDS = ("own", "endW_startN", "endW_startN_tail", "endW_startN_t3q", "startW_endN", "startW_endN_tail")


def plan(lin: str, conds: list[str], only: set[str]):
    """(cond, who, weights (run, step), donor (run, step) or None, dims or None)."""
    s, e = SE[lin]
    srun = START_RUN[lin]
    out = []
    if "own" in conds and (not only or f"start_{lin}" in only):
        out.append(("own", f"start_{lin}", (srun, s), None, None))
    for run in NAMES:
        if not run.startswith(lin) or (only and run not in only):
            continue
        for c in conds:
            if c == "own":
                out.append((c, run, (run, e), None, None))
            elif c == "endW_startN":
                out.append((c, run, (run, e), (srun, s), None))
            elif c == "endW_startN_tail":
                out.append((c, run, (run, e), (srun, s), TAIL_DIMS))
            elif c == "endW_startN_t3q":
                out.append((c, run, (run, e), (srun, s), (T3Q,)))
            elif c == "startW_endN":
                out.append((c, run, (srun, s), (run, e), None))
            elif c == "startW_endN_tail":
                out.append((c, run, (srun, s), (run, e), TAIL_DIMS))
    return sorted(out, key=lambda x: (x[2] != (srun, s), x[2]))  # stable: one model load per weights


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--lin", required=True, choices=("s42", "s80"))
    ap.add_argument("--conds", required=True)
    ap.add_argument("--seeds", default="3042-3061")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    conds = [c for c in a.conds.split(",") if c]
    assert all(c in CONDS for c in conds), conds
    only = {w for w in a.only.split(",") if w}
    (a.out / "states").mkdir(parents=True, exist_ok=True)
    factory = make_factory()
    done = set()
    if (a.out / "episodes.jsonl").exists():
        for line in open(a.out / "episodes.jsonl"):
            r = json.loads(line)
            done.add((r["cond"], r["who"], r["seed"]))
    out = open(a.out / "episodes.jsonl", "a")
    agents: dict = {}
    for cond, who, wts, donor, dims in plan(a.lin, conds, only):
        if all((cond, who, s) in done for s in seeds):
            continue
        if any((cond, who, s) in done for s in seeds):
            raise SystemExit(f"{cond} {who} is partly done; remove its rows first")
        t0 = time.time()
        if wts not in agents:
            agents.clear()  # keep at most one loaded model (the plan is ordered by run)
            agents[wts] = Agent(*wts, factory)
        ag = agents[wts]
        if donor is None:
            ag.set_stats(None, tag="own")
        else:
            m, v = stats(*donor)
            ag.set_stats(m, v, dims=dims, tag=cond)
        st_all = []
        rows = []
        for seed in seeds:
            if (cond, who, seed) in done:
                continue
            env = factory()
            row, states = episode(env, ag, seed)
            env.close()
            sw = sweep_states(ag, states) if len(states) else None
            row.update({"cond": cond, "who": who, "lin": a.lin, "weights": list(wts), "donor": list(donor) if donor else None,
                        "dims": list(dims) if dims else None, "sweep": sw, "n_states": int(len(states))})
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            pad = np.full((6, 63), np.nan, dtype=np.float32)
            pad[: len(states)] = states
            st_all.append(pad)
        np.save(a.out / "states" / f"{cond}__{who}.npy", np.asarray(st_all, dtype=np.float32))
        ag.set_stats(None, tag="own")
        g = [r["sweep"]["loop_gain"] for r in rows if r["sweep"]]
        print(f"{cond} {who}: {len(rows)} ep {time.time() - t0:.1f} s; median return {np.median([r['return'] for r in rows]):.1f}; "
              f"own14 {np.nanmedian([r['own14_mean_5_10'] for r in rows]):+.3f}; gain {np.median(g) if g else float('nan'):+.3f}; "
              f"falls {sum(r['terminated'] for r in rows)}", flush=True)
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
