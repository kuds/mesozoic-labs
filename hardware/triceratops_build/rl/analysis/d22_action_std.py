"""The policy's action noise (PPO's exploration std, train_health.jsonl's action_std, the mean over the 12 actions of
exp(log_std)) at fixed points of training, from the D22 runs' training-health logs. Simulation records only.

The D22 runs trained the straight task on the 128 mm and 160 mm models with run 3's preset and train.py's default
schedules: learning rate 3e-4 to 3e-5 over 10 M steps, late ramp 1 M to 3 M, push and friction ramp 1 M to 4 M,
log_std_init -1.6 (std 0.2). The steering dev runs (D31) keep the same schedules, so these runs show what noise to
expect when the heat cost reaches full weight (3 M). They had no r_heat_peak, which also prices noise.

Usage:
  python d22_action_std.py --health-dir DIR --out OUT.json
DIR holds <run>.jsonl, each a copy of a run's train_health.jsonl (read as data; nothing in it is executed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POINTS = (250_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 7_500_000, 10_000_000)


def trajectory(path: Path) -> dict:
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    out = {"rollouts": len(lines), "first_timesteps": lines[0]["timesteps"], "last_timesteps": lines[-1]["timesteps"],
           "sessions": sorted({x["session"] for x in lines}), "action_std": {}}
    for p in POINTS:
        d = min(lines, key=lambda x: abs(x["timesteps"] - p))
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
        print(run, " ".join(f"{int(k) / 1e6:g}M {v['action_std']:.3f}" for k, v in t["action_std"].items()))


if __name__ == "__main__":
    main()
