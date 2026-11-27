"""The policy's action noise (PPO's exploration std, train_health.jsonl's action_std, the mean over the 12 actions of
exp(log_std)) at fixed points of training, from the D22 runs' training-health logs. Simulation records only.

The D22 runs trained the straight task on the 128 mm and 160 mm models with run 3's preset and train.py's default
schedules: learning rate 3e-4 to 3e-5 over 10 M steps, late ramp 1 M to 3 M, push and friction ramp 1 M to 4 M,
log_std_init -1.6 (std 0.2). The steering dev runs (D31) keep the same schedules, so these runs show what noise to
expect when the heat cost reaches full weight (3 M). They had no r_heat_peak, which also prices noise.

The D22 runs differ from the steering dev runs in task as well: the straight task (no turn commands) with run 3's
terrain (steps and ramps from 1 M), 0-10 ms servo delays and no whole-robot mass or payload draws. Their noise is a
guide for the dev runs, not a prediction.

Usage:
  python d22_action_std.py --health-dir DIR --out OUT.json
DIR holds <run>.jsonl, each a copy of a run's train_health.jsonl (read as data; nothing in it is executed). The rows
are read with telemetry.health's readers: a torn line is skipped, and a resumed run's discarded branch is dropped
(latest_branch). A point with no rollout within 8192 steps (one rollout of 8 envs x 1024) is recorded as null (not
reached, or a gap). The same tool reads a dev run's log for its noise check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telemetry.health import latest_branch, read_health  # noqa: E402

POINTS = (250_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000)


MAX_GAP = 8192  # one rollout of 8 envs x 1024 steps


def trajectory(path: Path, points=POINTS) -> dict:
    lines = [r for r in latest_branch(read_health(path), "timesteps") if "action_std" in r]
    if not lines:
        raise SystemExit(f"{path}: no rows with timesteps and action_std")
    out = {"rollouts": len(lines), "first_timesteps": lines[0]["timesteps"], "last_timesteps": lines[-1]["timesteps"],
           "sessions": sorted({x.get("session") for x in lines}, key=str), "action_std": {}}
    for p in points:
        d = min(lines, key=lambda x: abs(x["timesteps"] - p))
        if abs(d["timesteps"] - p) > MAX_GAP:
            out["action_std"][str(p)] = None  # the run has no rollout near this point (not reached, or a gap)
            continue
        out["action_std"][str(p)] = {"timesteps": d["timesteps"], "action_std": round(float(d["action_std"]), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--health-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {p.stem: trajectory(p) for p in sorted(Path(a.health_dir).glob("*.jsonl"))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    for run, t in res.items():
        print(run, " ".join(f"{int(k) / 1e6:g}M " + (f"{v['action_std']:.3f}" if v else "-")
                            for k, v in t["action_std"].items()))


if __name__ == "__main__":
    main()
