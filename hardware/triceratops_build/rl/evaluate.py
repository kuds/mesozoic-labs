"""Evaluate a trained policy on the 60 cm Triceratops walking task. Simulation only.

Usage:
  python hardware/triceratops_build/rl/evaluate.py --run-dir RUN_DIR [--episodes 10] [--delays 10 0] [--sweep]
  python hardware/triceratops_build/rl/evaluate.py --checkpoint path/trike60_<n>_steps.zip [--vecnormalize ...]
  python hardware/triceratops_build/rl/evaluate.py --policy zero --model trike60_C_closed --episodes 5

Episodes run with domain randomisation ON (friction, masses, kp, force limits, observation noise, as in training).
Actions are the policy's mean (deterministic) unless --stochastic; the VecNormalize statistics are used in eval mode
(training=False, norm_reward=False: they are read, never updated).

Two kinds of blocks, all at a FIXED servo delay:
  * per delay in --delays (10 ms always, first): --episodes episodes with the command drawn per episode as in
    training, or fixed with --command;
  * --sweep [SPEEDS]: at 10 ms, --sweep-episodes episodes at each fixed command (default 0, 0.1, 0.2, 0.25, 0.27 m/s).
    The museum target, 0.25 m/s, is reported on its own line and under "target" in the JSON.
Episode seeds are ``--seed + i`` in every block, so blocks differ only in delay or command.

The model is the run's (run_record.json beside the checkpoints) unless --model names one; with neither, it refuses.

Reported per block (JSON, printed and written to --out, default RUN_DIR/eval/eval_<steps>.json):
  mean_forward_speed_mps        distance along the commanded heading / episode time, mean over episodes
  mean_command_mps              mean commanded speed
  mean_abs_speed_error_mps      per episode the mean over control steps of |v_fwd(step) - v_cmd|, then mean
  mean_abs_avg_speed_error_mps  per episode |mean v_fwd - v_cmd|, then mean
  falls, fall_rate              episodes that terminated (tilt, height, body contact, unstable) before the time limit
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

import gait_metrics  # noqa: E402
import numpy as np  # noqa: E402
from trike_env import MODEL_DIR, MODELS, MUSEUM_TARGET_SPEED_MPS, TrikeEnv, TrikeEnvConfig, flat_floor  # noqa: E402

NAME_PREFIX = "trike60"
SWEEP_DEFAULT = (0.0, 0.1, 0.2, 0.25, 0.27)


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
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="fixed-command sweep at 10 ms; no values: 0 0.1 0.2 0.25 0.27")
    ap.add_argument("--sweep-episodes", type=int, default=None, help="episodes per sweep command (default --episodes)")
    ap.add_argument("--seed", type=int, default=100_000)
    ap.add_argument("--model", default=None, help="MJCF path or model name (default: the run's own)")
    ap.add_argument("--seconds", type=float, default=None, help="episode length override, s (quick checks only)")
    ap.add_argument("--no-randomization", action="store_true", help="nominal model and no observation noise")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the policy mean")
    ap.add_argument("--out", default=None, help="JSON output path")
    return ap.parse_args(argv)


def resolve_checkpoint(args):
    """(zip, vecnormalize pkl, steps, run_dir or None); (None, None, None, None) for a baseline policy."""
    if args.policy:
        return None, None, None, None
    if args.run_dir:
        from trike_checkpoints import scan_pairs

        scan = scan_pairs(Path(args.run_dir) / "checkpoints", retries=2, wait_s=10.0)
        if scan.io_error:
            raise SystemExit(f"REFUSED: the checkpoint files could not be read ({scan.io_error}); run again later")
        for s in scan.skipped:
            print(f"WARNING: skipping checkpoint {s}", flush=True)
        if scan.pair is None:
            raise SystemExit(f"no intact checkpoint pair in {Path(args.run_dir) / 'checkpoints'}")
        steps, zip_path, vn_path = scan.pair
        return zip_path, vn_path, steps, Path(args.run_dir)
    zip_path = Path(args.checkpoint)
    steps = None
    vn = Path(args.vecnormalize) if args.vecnormalize else None
    parts = zip_path.stem.rsplit("_", 2)  # trike60_<n>_steps
    if len(parts) == 3 and parts[2] == "steps" and parts[1].isdigit():
        steps = int(parts[1])
        if vn is None:
            vn = zip_path.with_name(f"{parts[0]}_vecnormalize_{steps}_steps.pkl")
    if vn is None or not vn.exists():
        raise SystemExit(f"no VecNormalize statistics for {zip_path} (give --vecnormalize)")
    run_dir = zip_path.resolve().parent.parent
    return zip_path, vn, steps, run_dir if (run_dir / "run_record.json").exists() else None


def run_env_config(run_dir: Path | None) -> dict | None:
    """The env config a run trained with (from its run_record.json), or None (no record, or an unreadable one:
    then --model must name the model, and the rest of the config is the default)."""
    if run_dir is None or not (run_dir / "run_record.json").exists():
        return None
    try:
        rec = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        print(f"WARNING: {run_dir / 'run_record.json'} could not be read ({exc!r}); the run's env config is unknown",
              flush=True)
        return None
    return (rec.get("settings") or {}).get("env_config") or rec.get("env_config")


def env_config(args, run_dir: Path | None) -> TrikeEnvConfig:
    recorded = run_env_config(run_dir)
    if recorded is not None:
        cfg = TrikeEnvConfig.from_dict(recorded)
        name = Path(cfg.model_path).name  # recorded by file name: use this checkout's copy of that file
        cfg = cfg.replace(model_path=str(MODEL_DIR / name))
    elif args.model:
        cfg = TrikeEnvConfig()
    else:
        raise SystemExit("REFUSED: no readable run_record.json beside the checkpoint names the model; give --model "
                         f"(one of {sorted(MODELS)} or a path)")
    if args.model:
        path = MODELS.get(args.model) or Path(args.model).resolve()
        if recorded is not None and Path(path).name != Path(cfg.model_path).name:
            print(f"WARNING: --model {Path(path).name} differs from the run's model {Path(cfg.model_path).name}",
                  flush=True)
        cfg = cfg.replace(model_path=str(path))
    if not Path(cfg.model_path).exists():
        raise SystemExit(f"model file not found: {cfg.model_path}")
    cfg = cfg.replace(domain_randomization=not args.no_randomization, observation_noise=not args.no_randomization)
    cfg = flat_floor(cfg)  # a run trained on terrain is evaluated on the flat floor, as every other run
    if args.seconds:
        cfg = cfg.replace(episode_seconds=args.seconds)
    return cfg


def make_policy(zip_path, vn_path, env, deterministic: bool):
    """A callable obs -> action; for a checkpoint it carries ``.vecnormalize`` (eval mode) and ``.model``."""
    if zip_path is None:
        def zero(obs):
            return np.zeros(12, dtype=np.float32)

        zero.vecnormalize = None
        return zero
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.policy_loading import load_sb3_model

    torch.set_num_threads(1)  # one small MLP forward per step: threads only add overhead
    vn = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: env]))
    vn.training, vn.norm_reward = False, False  # eval mode: statistics are read, never updated
    model = load_sb3_model(zip_path, algorithm="ppo", device="cpu")

    def act(obs):
        a, _ = model.predict(vn.normalize_obs(obs[None, :]), deterministic=deterministic)
        return a[0]

    act.vecnormalize, act.model = vn, model
    return act


def run_episode(env: TrikeEnv, policy, seed: int, delay: int, command):
    """One episode, logged every physics step: speed figures plus the gait and drive measures (gait_metrics)."""
    log = gait_metrics.record_episode(env, policy, seed=seed, command=command, delay_steps=delay)
    v_cmd, v = log["command"], log["episode_forward_speed"]
    return {"seed": seed, "delay_steps": delay, "command_mps": v_cmd, "forward_speed_mps": v,
            "abs_speed_error_mps": float(np.mean(np.abs(log["step_forward_speed"] - v_cmd))),
            "abs_avg_speed_error_mps": abs(v - v_cmd), "fell": log["fell"], "termination": log["termination"],
            "length_s": log["time_s"], "friction": log["reset_info"].get("friction"),
            "measures": gait_metrics.measures(log, v_cmd)}


def summarize(eps: list[dict]) -> dict:
    def mean(k):
        return float(np.mean([e[k] for e in eps]))

    falls = sum(e["fell"] for e in eps)
    return {"episodes": len(eps), "mean_forward_speed_mps": mean("forward_speed_mps"),
            "mean_command_mps": mean("command_mps"), "mean_abs_speed_error_mps": mean("abs_speed_error_mps"),
            "mean_abs_avg_speed_error_mps": mean("abs_avg_speed_error_mps"), "falls": falls,
            "fall_rate": falls / len(eps), "mean_episode_length_s": mean("length_s"),
            "gait": gait_metrics.summarise([e["measures"] for e in eps]), "per_episode": eps}


def line(label: str, s: dict) -> str:
    return (f"{label}: {s['episodes']} episodes, forward speed {s['mean_forward_speed_mps']:.3f} m/s (command "
            f"{s['mean_command_mps']:.3f}), |error| {s['mean_abs_speed_error_mps']:.3f} m/s, falls {s['falls']}, "
            f"length {s['mean_episode_length_s']:.1f} s")


def main(argv=None) -> int:
    args = parse_args(argv)
    zip_path, vn_path, steps, run_dir = resolve_checkpoint(args)
    cfg = env_config(args, run_dir)
    env = TrikeEnv(cfg)
    policy = make_policy(zip_path, vn_path, env, deterministic=not args.stochastic)
    results: dict = {}
    for delay in list(dict.fromkeys([10] + list(args.delays))):  # 10 ms always, first
        eps = [run_episode(env, policy, args.seed + i, delay, args.command) for i in range(args.episodes)]
        results[f"delay_{delay}ms"] = summarize(eps)
        print(line(f"delay {delay:2d} ms", results[f"delay_{delay}ms"]), flush=True)
    target = None
    if args.sweep is not None:
        speeds = list(args.sweep) or list(SWEEP_DEFAULT)
        n_ep = args.sweep_episodes or args.episodes
        sweep = {}
        for c in speeds:
            eps = [run_episode(env, policy, args.seed + i, 10, c) for i in range(n_ep)]
            sweep[f"{c:.2f}"] = summarize(eps)
            print(line(f"sweep, delay 10 ms, command {c:.2f} m/s", sweep[f"{c:.2f}"]), flush=True)
        results["sweep_delay_10ms"] = sweep
        key = f"{MUSEUM_TARGET_SPEED_MPS:.2f}"
        if key in sweep:
            target = {k: v for k, v in sweep[key].items() if k != "per_episode"}
            print(line(f"TARGET {MUSEUM_TARGET_SPEED_MPS:.2f} m/s at 10 ms", sweep[key]), flush=True)
            print(f"TARGET {MUSEUM_TARGET_SPEED_MPS:.2f} m/s at 10 ms, gait: {gait_metrics.one_line(sweep[key]['gait'])}",
                  flush=True)
    report = {"checkpoint": str(zip_path) if zip_path else f"baseline:{args.policy}",
              "vecnormalize": str(vn_path) if vn_path else None, "steps": steps, "episodes": args.episodes,
              "seed": args.seed, "command": args.command, "deterministic": not args.stochastic,
              "model": Path(cfg.model_path).name, "env_config": cfg.to_dict(),
              "target": {"speed_mps": MUSEUM_TARGET_SPEED_MPS, "delay_ms": 10, "summary": target},
              "results": results}
    out = args.out
    if out is None and args.run_dir:
        out = str(Path(args.run_dir) / "eval" / f"eval_{steps}.json")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(out).with_name(Path(out).name + ".tmp")
        tmp.write_text(json.dumps(report, indent=1))
        os.replace(tmp, out)
        print(f"Wrote {out}", flush=True)
    else:
        print(json.dumps({k: v for k, v in report.items() if k not in ("env_config", "results")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
