"""Foot loads of a checkpoint walking flat ground (D22, D23): each foot's normal force at every 1 ms physics step,
raw and averaged over 10 ms, in 5 episodes of 20 s at 0.25 m/s (seeds 300000-300004), measured from 2 s.
Simulation only.

The episodes are the demo course's walking segment (`demo_course.measure_segment`'s setting): the run's model and
env config on the flat floor (`flat_floor`), randomisation off, observation noise on, a fixed 10 ms delay, the
deterministic policy loaded with the safe loaders (`steer_eval.load_policy`). The environment is the one
`TRIKE_ENV_DIR` points at (an export of the run commit's rl/), as for the gait report and the sweep, else the tree's.
Per episode and foot: the maximum and 99th percentile of the 10 ms mean, and the maximum raw sample (N and body
weights). The summary gives, as means over the episodes, the worst front foot's 99th percentile of the 10 ms mean
(``front_p99_window_n``: the figure D24's rule reads), the worst hind foot's and the worst foot's.

Usage: python hardware/triceratops_build/rl/analysis/foot_loads.py --ckpt-dir DIR --out FILE.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RL = Path(__file__).resolve().parents[1]
for p in (RL, RL / "analysis", RL / "analytics"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import gait_pipe1  # noqa: E402,F401  (puts TRIKE_ENV_DIR first and imports that trike_env before anything else does)

SEEDS = tuple(range(300000, 300005))
SECONDS, FROM_S, COMMAND = 20.0, 2.0, 0.25


def measure(make, seed: int) -> dict:
    import trike_env as te
    from telemetry.loads import LoadRecorder

    env, act = make(COMMAND)
    env.cfg = env.cfg.replace(episode_seconds=SECONDS)
    env.max_episode_steps = int(round(SECONDS / env.dt))
    m = env.model
    rec = LoadRecorder(m, [], [f"{c}_sole" for c in te.CORNERS])

    def hook(e):
        if e.data.time >= FROM_S:
            rec(e.model, e.data)

    obs, _ = env.reset(seed=seed, options={"command_speed": COMMAND, "delay_steps": 10})
    env.substep_hook = hook
    fell, term, t = False, "", 0.0
    try:
        while True:
            obs, _, terminated, truncated, info = env.step(act(obs))
            t = info["time_s"]
            if terminated or truncated:
                fell, term = bool(terminated), info["termination"]
                break
    finally:
        env.substep_hook = None
    s = rec.summary(env.nominal_weight, physics_dt=env.physics_dt)
    feet = {}
    for c in te.CORNERS:
        f = s["feet"].get(f"{c}_sole", {})
        win, raw = f.get("normal_window_n"), f.get("normal_n")
        feet[c] = {"window_max_n": win and win["max"], "window_p99_n": win and win["p99"],
                   "raw_max_n": raw and raw["max"], "window_max_bw": win and win["max"] / env.nominal_weight}
    def worst(names):
        v = [feet[c]["window_p99_n"] for c in names if feet[c]["window_p99_n"] is not None]
        return max(v) if len(v) == len(names) else None

    return {"seed": seed, "fell": fell, "termination": term, "seconds": t, "foot_samples": s["foot_samples"],
            "window_s": s.get("window_s"), "weight_n": env.nominal_weight, "feet": feet,
            "front_p99_window_n": worst(("left_fore", "right_fore")),
            "hind_p99_window_n": worst(("left_hind", "right_hind")), "worst_foot_p99_window_n": worst(te.CORNERS)}


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(ckpt_dir: Path) -> dict:
    """What was measured: the checkpoint pair (the newest, as the other tools take it), the run's model file and the
    env code, with their sha256, and the run commit from the run record."""
    import gait_report
    import trike_env as te

    zp, pk, steps = gait_report.find_pair(ckpt_dir)
    rec = json.loads((ckpt_dir / "run_record.json").read_text())
    model = Path(rec["settings"]["env_config"]["model_path"]).name
    return {"checkpoint": zp.name, "steps": steps, "zip_sha256": _sha(zp), "vecnormalize_sha256": _sha(pk),
            "model": model, "model_sha256": _sha(te.MODEL_DIR / model), "trike_env_sha256": _sha(Path(te.__file__)),
            "run_commit": rec["sessions"][-1]["git"]["commit"]}


def run(ckpt_dir: Path) -> dict:
    import demo_course

    make, label = demo_course.policy_factory(ckpt_dir)
    eps = [measure(make, s) for s in SEEDS]
    def mean_of(key):  # undefined if any episode fell or lacks the figure
        v = [e[key] for e in eps if not e["fell"] and e[key] is not None]
        return float(np.mean(v)) if len(v) == len(eps) else None

    return {"simulation_only": True, "tool": "hardware/triceratops_build/rl/analysis/foot_loads.py", "policy": label,
            "provenance": provenance(ckpt_dir),
            "settings": {"seeds": list(SEEDS), "seconds": SECONDS, "from_s": FROM_S, "command_mps": COMMAND,
                         "delay_ms": 10, "floor": "flat", "randomisation": False, "observation_noise": True},
            "episodes": eps, "falls": sum(e["fell"] for e in eps),
            "front_p99_window_n": mean_of("front_p99_window_n"), "hind_p99_window_n": mean_of("hind_p99_window_n"),
            "worst_foot_p99_window_n": mean_of("worst_foot_p99_window_n"),
            "worst_foot_window_max_n": max((v["window_max_n"] for e in eps for v in e["feet"].values()
                                            if v["window_max_n"] is not None), default=None),
            "worst_foot_raw_max_n": max((v["raw_max_n"] for e in eps for v in e["feet"].values()
                                         if v["raw_max_n"] is not None), default=None)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Path(a.ckpt_dir))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"{res['policy']}: falls {res['falls']}/{len(SEEDS)}; 99th percentile of the 10 ms mean, mean of episodes: "
          f"worst front foot {res['front_p99_window_n']} N, worst hind foot {res['hind_p99_window_n']} N; largest "
          f"10 ms mean {res['worst_foot_window_max_n']} N; largest 1 ms sample {res['worst_foot_raw_max_n']} N")
    return 0


if __name__ == "__main__":
    sys.exit(main())
