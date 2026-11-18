"""PPO training for the 60 cm Triceratops walking task (``trike_env.TrikeEnv``). Simulation only.

Usage (from anywhere):
  python hardware/triceratops_build/rl/train.py --out RUN_DIR --total-steps 20000000 --n-envs 8 [--seed 0]
  python hardware/triceratops_build/rl/train.py --out RUN_DIR --total-steps 20000000 --n-envs 8 --resume

``--total-steps`` is the run's TOTAL environment-step budget, not an increment: a resumed run continues from its
newest intact checkpoint pair (``RUN_DIR/checkpoints/trike60_<steps>_steps.zip`` + ``trike60_vecnormalize_<steps>
_steps.pkl``) with ``reset_num_timesteps=False`` and stops at the same total, so the learning-rate schedule and the
step count line up whether or not the run was interrupted. A run that has already reached its total exits at once.

Writes into RUN_DIR:
  checkpoints/          periodic pairs (SB3 CheckpointCallback, save_vecnormalize=True), plus a pair at the end
                        of every session (also on Ctrl-C / SIGTERM)
  run_record.json       arguments, env config, PPO settings, package versions, commit, one entry per session with
                        its measured env steps per second
  monitor/              SB3 Monitor CSVs (episode reward, length, final forward speed and command), one per env
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")  # no rendering here; avoids a GL backend loading next to torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (str(HERE), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
# SubprocVecEnv workers import trike_env and environments.* by name: give them the same path.
os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE), str(REPO_ROOT)] + (
    [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))

NAME_PREFIX = "trike60"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory (checkpoints, run_record.json)")
    ap.add_argument("--total-steps", type=int, required=True, help="total env steps of the run (not an increment)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--model", default=None, help="MJCF path (default: ../sim/generated/trike60_C_closed.xml)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="continue from the newest intact checkpoint pair in --out")
    ap.add_argument("--checkpoint-every", type=int, default=250_000, help="env steps between periodic checkpoints")
    ap.add_argument("--vec", choices=("auto", "subproc", "dummy"), default="auto",
                    help="auto: SubprocVecEnv when --n-envs > 1, else DummyVecEnv")
    ap.add_argument("--torch-threads", type=int, default=1, help="torch intra-op threads in the learner process")
    # PPO
    ap.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr-start", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=3e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--net", default="256,256", help="hidden sizes of the policy and value MLPs")
    # env
    ap.add_argument("--env-config", default=None, help="JSON file of TrikeEnvConfig fields to override")
    ap.add_argument("--verbose", type=int, default=0, help="SB3's own logging (1: a table every rollout)")
    ap.add_argument("--log-every", type=int, default=100_000, help="env steps between one-line progress reports")
    return ap.parse_args(argv)


def versions() -> dict:
    import gymnasium
    import mujoco
    import numpy
    import stable_baselines3
    import torch

    return {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__}


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(["git", "-C", str(REPO_ROOT), *cmd], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def env_config(args):
    from trike_env import TrikeEnvConfig

    cfg = TrikeEnvConfig()
    if args.model:
        cfg = cfg.replace(model_path=str(Path(args.model).resolve()))
    if args.env_config:
        with open(args.env_config, encoding="utf-8") as f:
            cfg = TrikeEnvConfig.from_dict({**cfg.to_dict(), **json.load(f)})
    return cfg


def comparable_config(d: dict) -> dict:
    """An env config as a run compares it across sessions: lists as tuples, the model by file name only (the
    checkout's absolute path differs between machines)."""
    from trike_env import TrikeEnvConfig

    c = TrikeEnvConfig.from_dict(d).to_dict()
    c["model_path"] = Path(c["model_path"]).name
    return c


def build_venv(cfg, args, monitor_dir: Path):
    from functools import partial

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    monitor_dir.mkdir(parents=True, exist_ok=True)
    fns = [partial(_monitored_env, cfg.to_dict(), str(monitor_dir / f"env{i}")) for i in range(args.n_envs)]
    use_subproc = args.vec == "subproc" or (args.vec == "auto" and args.n_envs > 1)
    return SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)


def _monitored_env(cfg_dict: dict, monitor_path: str):
    from stable_baselines3.common.monitor import Monitor

    from trike_env import make_env

    return Monitor(make_env(cfg_dict), filename=monitor_path,
                   info_keywords=("forward_speed", "command_speed", "termination"))


def find_resume_pair(ckpt_dir: Path):
    from environments.shared.curriculum.checkpoints import newest_intact_periodic_pair

    return newest_intact_periodic_pair(ckpt_dir, NAME_PREFIX)


def progress_callback(every: int, total: int):
    """A callback printing one line every ``every`` env steps: steps, rate, recent episode return/length/speed."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Progress(BaseCallback):
        def _on_training_start(self) -> None:
            self.t0, self.n0, self.next = time.time(), self.model.num_timesteps, self.model.num_timesteps + every

        def _on_step(self) -> bool:
            n = self.model.num_timesteps
            if n >= self.next:
                self.next += every
                rate = (n - self.n0) / max(time.time() - self.t0, 1e-9)
                eps = list(self.model.ep_info_buffer)
                if eps:
                    r = sum(e["r"] for e in eps) / len(eps)
                    ln = sum(e["l"] for e in eps) / len(eps)
                    v = sum(e.get("forward_speed", 0.0) for e in eps) / len(eps)
                    ep = f"last {len(eps)} episodes: return {r:.1f}, length {ln:.0f} steps, final speed {v:.3f} m/s"
                else:
                    ep = "no episode finished yet"
                print(f"steps {n} of {total} ({100.0 * n / total:.1f}%), {rate:.0f} env steps/s; {ep}", flush=True)
            return True

    return Progress()


def save_pair(model, venv, ckpt_dir: Path) -> Path:
    steps = int(model.num_timesteps)
    zip_path = ckpt_dir / f"{NAME_PREFIX}_{steps}_steps.zip"
    model.save(str(zip_path))
    venv.save(str(ckpt_dir / f"{NAME_PREFIX}_vecnormalize_{steps}_steps.pkl"))
    return zip_path


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.curriculum.schedules import LinearSchedule
    from environments.shared.policy_loading import load_sb3_model

    torch.set_num_threads(max(1, args.torch_threads))
    out = Path(args.out).resolve()
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    record_path = out / "run_record.json"

    pair, skipped = find_resume_pair(ckpt_dir)
    for s in skipped:
        print(f"WARNING: skipping checkpoint {s}", flush=True)
    if not args.resume and (pair is not None or skipped):
        print(f"REFUSED: {ckpt_dir} already holds checkpoints; pass --resume to continue that run, or use another "
              "--out. Nothing was changed.", flush=True)
        return 2
    if args.resume and pair is None and skipped:
        print("REFUSED: --resume found checkpoints but none is intact (see the warnings above). Nothing was "
              "changed.", flush=True)
        return 2

    cfg = env_config(args)
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    if pair is not None and record.get("env_config") and (
            comparable_config(record["env_config"]) != comparable_config(cfg.to_dict())):
        print("REFUSED: the env config differs from the one this run was started with (run_record.json). "
              "Nothing was changed.", flush=True)
        return 2

    if pair is not None:
        steps0 = pair[0]
        if steps0 >= args.total_steps:
            print(f"Run already complete: newest checkpoint is at {steps0} steps >= --total-steps "
                  f"{args.total_steps}. Nothing to do.", flush=True)
            return 0

    net = [int(x) for x in args.net.split(",") if x]
    ppo_kwargs = dict(
        n_steps=args.n_steps, batch_size=min(args.batch_size, args.n_steps * args.n_envs), n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range, ent_coef=args.ent_coef,
        vf_coef=0.5, max_grad_norm=1.0, learning_rate=LinearSchedule(args.lr_start, args.lr_end),
    )
    policy_kwargs = dict(net_arch=dict(pi=net, vf=net), activation_fn=torch.nn.ELU, log_std_init=args.log_std_init)

    venv = build_venv(cfg, args, out / "monitor")
    session_seed = args.seed if pair is None else args.seed + int(pair[0])  # a resumed session draws new episodes

    if pair is None:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, seed=args.seed, device=args.device,
                    verbose=args.verbose, **ppo_kwargs)
        print(f"Starting a new run in {out} (0 of {args.total_steps} steps).", flush=True)
    else:
        steps0, zip_path, vn_path = pair
        venv = VecNormalize.load(str(vn_path), venv)
        venv.training, venv.norm_reward = True, True
        model = load_sb3_model(zip_path, algorithm="ppo", env=venv, device=args.device)
        model.verbose = args.verbose
        model.set_random_seed(session_seed)  # PPO.load re-seeds the envs with the run's first seed; draw anew
        print(f"Resuming {out} from {zip_path.name} ({model.num_timesteps} of {args.total_steps} steps).",
              flush=True)

    session = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv[1:] if argv is None
        else list(argv), "args": vars(args), "versions": versions(), "git": git_state(),
        "cpu_count": os.cpu_count(), "resumed_from": None if pair is None else pair[1].name,
        "steps_start": int(model.num_timesteps), "session_seed": session_seed,
    }
    record.setdefault("created_utc", session["started_utc"])
    record["env_config"] = cfg.to_dict()
    record["ppo"] = {k: (repr(v) if k == "learning_rate" else v) for k, v in ppo_kwargs.items()}
    record["policy_kwargs"] = {"net_arch": net, "activation_fn": "ELU", "log_std_init": args.log_std_init}
    record["total_steps"] = args.total_steps
    record.setdefault("sessions", []).append(session)
    record_path.write_text(json.dumps(record, indent=1, default=str))

    ckpt_cb = CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.n_envs), save_path=str(ckpt_dir),
                                 name_prefix=NAME_PREFIX, save_vecnormalize=True, verbose=1)
    callbacks = [ckpt_cb, progress_callback(args.log_every, args.total_steps)]

    def on_term(signum, frame):  # Colab's stop button / kill: end like Ctrl-C so the last pair is saved
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_term)
    remaining = args.total_steps - int(model.num_timesteps)
    t0, n0 = time.time(), int(model.num_timesteps)
    status = "completed"
    try:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=pair is None,
                    progress_bar=False)
    except KeyboardInterrupt:
        status = "interrupted"
        print("Interrupted: saving a checkpoint pair before exiting.", flush=True)
    wall = time.time() - t0
    last = save_pair(model, venv, ckpt_dir)
    session.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": status,
                    "steps_end": int(model.num_timesteps), "wall_s": wall,
                    "env_steps_per_s": (int(model.num_timesteps) - n0) / wall if wall > 0 else None,
                    "last_checkpoint": last.name})
    record_path.write_text(json.dumps(record, indent=1, default=str))
    print(f"Session {status}: {session['steps_end']} of {args.total_steps} steps; "
          f"{session['env_steps_per_s'] or 0:.0f} env steps/s over {wall:.0f} s; saved {last.name}", flush=True)
    try:
        venv.close()
    except Exception as exc:  # workers already gone after an interrupt
        print(f"(closing the envs: {exc!r})", flush=True)
    return 0 if status == "completed" else 130


if __name__ == "__main__":
    sys.exit(main())
