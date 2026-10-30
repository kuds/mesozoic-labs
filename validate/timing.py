"""Throughput from the no-update V2 runs (files only): CPU and wall seconds per rollout from the evidence clock, the
evaluation's extra cost, and the estimate for one continuation (176 rollouts, 11 evaluations, 175 updates) and for
8 continuations at 4 at a time.  The update's cost is not measured here (no optimizer step is allowed in this build):
it is taken from the ECS (S2/r10round/ecs), whose real continuations and no-update validation runs both ran this code
on this machine: per rollout (learn_seconds / rollouts), real minus no-update."""
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ECS = Path("/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/ecs")
EVAL_EVERY = 16


def t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()


rows = []
for run in sorted(Path(sys.argv[1]).iterdir()):
    meta = json.loads((run / "run_meta.json").read_text())
    if meta.get("status") != "done":
        continue
    ev = [json.loads(x) for x in (run / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    cpu = [e["clock"]["cpu_seconds"] for e in ev]
    wall = [t(e["clock"]["utc"]) for e in ev]
    steps = [e["num_timesteps"] for e in ev]
    d_cpu = [b - a for a, b in zip(cpu, cpu[1:])]
    d_wall = [b - a for a, b in zip(wall, wall[1:])]
    # the evaluation at S + 131,072 k runs on the last step of the rollout that ends there
    with_eval = [d for d, s in zip(d_cpu, steps[1:]) if (s - 1048576) % 131072 == 0]
    plain = [d for d, s in zip(d_cpu, steps[1:]) if (s - 1048576) % 131072 != 0]
    rows.append(dict(run=run.name, cpu_total=meta.get("cpu_seconds"), wall_total=meta.get("wall_seconds"),
                     first_rollout_end_cpu=cpu[0], cpu_per_rollout_median=st.median(plain),
                     cpu_rollout_with_eval=st.median(with_eval) if with_eval else None,
                     wall_per_rollout_median=st.median(d_wall)))
for r in rows:
    print(json.dumps(r))
cpu_roll = st.median(r["cpu_per_rollout_median"] for r in rows)
evals = [r["cpu_rollout_with_eval"] for r in rows if r["cpu_rollout_with_eval"]]
cpu_eval = (st.median(evals) - cpu_roll) if evals else 0.0
setup = st.median(r["first_rollout_end_cpu"] - r["cpu_per_rollout_median"] for r in rows)
# ECS: per-rollout wall in real continuations (3 at a time) and in no-update V2 runs (one at a time)
real, nou = [], []
for m in ECS.glob("runs/*/run_meta.json"):
    d = json.loads(m.read_text())
    if d.get("status") == "done" and d.get("learn_seconds"):
        real.append(d["learn_seconds"] / ((d["T"] - d["S"]) / 8192))
for m in ECS.glob("validate/v2/*/run_meta.json"):
    d = json.loads(m.read_text())
    if d.get("status") == "done" and d.get("learn_seconds"):
        nou.append(d["learn_seconds"] / ((d["T"] - d["S"]) / 8192))
upd = st.median(real) - st.median(nou)
print(f"ECS: real continuations {len(real)}: median {st.median(real):.2f} s/rollout (3 at a time); no-update V2 "
      f"{len(nou)}: median {st.median(nou):.2f} s/rollout; the update's share {upd:.2f} s/rollout [inferred]")
n_roll, n_eval = 176, 11
one = setup + n_roll * (cpu_roll + upd) + n_eval * cpu_eval
print(f"BS no-update: {cpu_roll:.2f} CPU-s per rollout (median over runs), +{cpu_eval:.2f} CPU-s per evaluation, setup "
      f"{setup:.1f} CPU-s")
print(f"ONE CONTINUATION (estimate): {one:.0f} CPU-s = {one / 60:.0f} min on one free CPU "
      f"({n_roll} rollouts x ({cpu_roll:.2f} + {upd:.2f} update) + {n_eval} evals + setup)")
for k in (0, 1, 2, 3):
    f = max(1.0, (4 + k) / 4)
    print(f"8 continuations, 4 at a time, 2 waves, with {k} other busy process(es) on the 4 CPUs: about "
          f"{2 * one * f / 3600:.1f} h wall (x{f:.2f})")
