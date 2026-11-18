"""Evaluate a trained policy on the 60 cm Triceratops walking task. Simulation only.

Usage:
  python hardware/triceratops_build/rl/evaluate.py --run-dir RUN_DIR [--episodes 10] [--delays 10 0]
  python hardware/triceratops_build/rl/evaluate.py --checkpoint path/trike60_<n>_steps.zip [--vecnormalize ...]
  python hardware/triceratops_build/rl/evaluate.py --policy zero --episodes 5     (baseline: stand still)

Episodes run with domain randomisation ON (friction, masses, kp, force limits, observation noise, as in training)
and the servo delay FIXED per block: 10 ms (10 physics steps) always, plus any other delays given with --delays.
The commanded speed is drawn per episode as in training, or fixed with --command. Episode seeds are
``--seed + i``, identical across delay blocks, so the blocks differ only in the delay.

Reported per delay (JSON, printed and written to --out, default RUN_DIR/eval/eval_<steps>.json):
  mean_forward_speed_mps      distance along the commanded heading / episode time, mean over episodes
  mean_command_mps            mean commanded speed
  mean_abs_speed_error_mps    per episode the mean over control steps of |v_fwd(step) - v_cmd|, then mean
  mean_abs_avg_speed_error_mps per episode |mean v_fwd - v_cmd|, then mean
  falls, fall_rate            episodes that terminated (tilt, height, body contact, unstable) before the time limit
  mean_episode_length_s
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from trike_env import TrikeEnv, TrikeEnvConfig  # noqa: E402

NAME_PREFIX = "trike60"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="a train.py --out directory: its newest intact checkpoint pair is used")
    src.add_argument("--checkpoint", help="an SB3 .zip; its VecNormalize sidecar is found by name or --vecnormalize")
    src.add_argument("--policy", choices=("zero",), help="a baseline instead of a checkpoint: zero = all-zero actions")
    ap.add_argument("--vecnormalize", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--delays", type=int, nargs="+", default=[10], help="servo delays in physics steps (ms)")
    ap.add_argument("--command", type=float, default=None, help="fixed commanded speed, m/s (default: drawn)")
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--model", default=None, help="MJCF path (default: the run's, else trike60_C_closed.xml)")
    ap.add_argument("--no-randomization", action="store_true", help="nominal model and no observation noise")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the policy mean")
    ap.add_argument("--out", default=None, help="JSON output path")
    return ap.parse_args(argv)


def resolve_checkpoint(args):
    """(zip, vecnormalize pkl, steps) or (None, None, None) for a baseline policy."""
    if args.policy:
        return None, None, None
    if args.run_dir:
        from environments.shared.curriculum.checkpoints import newest_intact_periodic_pair

        pair, skipped = newest_intact_periodic_pair(Path(args.run_dir) / "checkpoints", NAME_PREFIX)
        for s in skipped:
            print(f"WARNING: skipping checkpoint {s}", flush=True)
        if pair is None:
            raise SystemExit(f"no intact checkpoint pair in {Path(args.run_dir) / 'checkpoints'}")
        steps, zip_path, vn_path = pair
        return zip_path, vn_path, steps
    zip_path = Path(args.checkpoint)
    stem = zip_path.stem  # trike60_<n>_steps
    steps = None
    vn = Path(args.vecnormalize) if args.vecnormalize else None
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
        steps = int(parts[1])
        if vn is None:
            vn = zip_path.with_name(f"{parts[0]}_vecnormalize_{steps}_steps.pkl")
    if vn is None or not vn.exists():
        raise SystemExit(f"no VecNormalize statistics for {zip_path} (give --vecnormalize)")
    return zip_path, vn, steps


def env_config(args) -> TrikeEnvConfig:
    cfg = TrikeEnvConfig()
    if args.run_dir and (Path(args.run_dir) / "run_record.json").exists():
        rec = json.loads((Path(args.run_dir) / "run_record.json").read_text())
        if rec.get("env_config"):
            cfg = TrikeEnvConfig.from_dict(rec["env_config"])
            if not Path(cfg.model_path).exists():  # the run was trained on another machine: same file, this checkout
                cfg = cfg.replace(model_path=str(HERE.parent / "sim" / "generated" / Path(cfg.model_path).name))
    if args.model:
        cfg = cfg.replace(model_path=str(Path(args.model).resolve()))
    if args.no_randomization:
        cfg = cfg.replace(domain_randomization=False, observation_noise=False)
    else:
        cfg = cfg.replace(domain_randomization=True, observation_noise=True)
    return cfg


def make_policy(zip_path, vn_path, env, deterministic: bool):
    if zip_path is None:
        return lambda obs: np.zeros(12, dtype=np.float32)
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.policy_loading import load_sb3_model

    torch.set_num_threads(1)  # one small MLP forward per step: threads only add overhead
    vn = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: env]))
    vn.training, vn.norm_reward = False, False
    model = load_sb3_model(zip_path, algorithm="ppo", device="cpu")

    def act(obs):
        a, _ = model.predict(vn.normalize_obs(obs[None, :]), deterministic=deterministic)
        return a[0]

    return act


def run_episode(env: TrikeEnv, policy, seed: int, delay: int, command):
    opts = {"delay_steps": delay}
    if command is not None:
        opts["command_speed"] = command
    obs, info = env.reset(seed=seed, options=opts)
    v_cmd = info["command_speed"]
    errs, dist, n, term, reason = [], 0.0, 0, False, ""
    while True:
        obs, r, term, trunc, inf = env.step(policy(obs))
        n += 1
        errs.append(abs(inf["forward_speed"] - v_cmd))
        dist += inf["forward_speed"] * env.dt
        if term or trunc:
            reason = inf["termination"]
            break
    t = n * env.dt
    v = dist / t
    return {"seed": seed, "delay_steps": delay, "command_mps": v_cmd, "forward_speed_mps": v,
            "abs_speed_error_mps": float(np.mean(errs)), "abs_avg_speed_error_mps": abs(v - v_cmd),
            "fell": bool(term), "termination": reason, "length_s": t, "friction": info.get("friction")}


def summarize(eps: list[dict]) -> dict:
    def mean(k):
        return float(np.mean([e[k] for e in eps]))

    falls = sum(e["fell"] for e in eps)
    return {"episodes": len(eps), "mean_forward_speed_mps": mean("forward_speed_mps"),
            "mean_command_mps": mean("command_mps"), "mean_abs_speed_error_mps": mean("abs_speed_error_mps"),
            "mean_abs_avg_speed_error_mps": mean("abs_avg_speed_error_mps"), "falls": falls,
            "fall_rate": falls / len(eps), "mean_episode_length_s": mean("length_s"), "per_episode": eps}


def main(argv=None) -> int:
    args = parse_args(argv)
    zip_path, vn_path, steps = resolve_checkpoint(args)
    cfg = env_config(args)
    env = TrikeEnv(cfg)
    policy = make_policy(zip_path, vn_path, env, deterministic=not args.stochastic)
    delays = list(dict.fromkeys([10] + list(args.delays)))  # 10 ms always, first
    results = {}
    for delay in delays:
        eps = [run_episode(env, policy, args.seed + i, delay, args.command) for i in range(args.episodes)]
        results[f"delay_{delay}ms"] = summarize(eps)
        s = results[f"delay_{delay}ms"]
        print(f"delay {delay:2d} ms: {s['episodes']} episodes, forward speed {s['mean_forward_speed_mps']:.3f} m/s "
              f"(command {s['mean_command_mps']:.3f}), |error| {s['mean_abs_speed_error_mps']:.3f} m/s, "
              f"falls {s['falls']}, length {s['mean_episode_length_s']:.1f} s", flush=True)
    report = {"checkpoint": str(zip_path) if zip_path else f"baseline:{args.policy}",
              "vecnormalize": str(vn_path) if vn_path else None, "steps": steps, "episodes": args.episodes,
              "seed": args.seed, "command": args.command, "deterministic": not args.stochastic,
              "env_config": cfg.to_dict(), "results": results}
    out = args.out
    if out is None and args.run_dir:
        out = str(Path(args.run_dir) / "eval" / f"eval_{steps}.json")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1))
        print(f"Wrote {out}", flush=True)
    else:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_episode"} for k, v in results.items()},
                         indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
