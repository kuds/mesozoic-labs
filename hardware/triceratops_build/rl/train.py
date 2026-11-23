"""PPO training for the 60 cm Triceratops walking task (``trike_env.TrikeEnv``). Simulation only.

Usage (from anywhere):
  python hardware/triceratops_build/rl/train.py --out RUN_DIR --total-steps 2000000 --n-envs 8 [--seed 0]
  python hardware/triceratops_build/rl/train.py --out RUN_DIR --total-steps 2000000 --n-envs 8 --resume

``--total-steps`` is the run's TOTAL environment-step budget, not an increment. With ``--resume`` the run continues
from its newest intact checkpoint pair (``RUN_DIR/checkpoints/trike60_<steps>_steps.zip`` and
``trike60_vecnormalize_<steps>_steps.pkl``) with ``reset_num_timesteps=False`` and stops at the same total, so the
learning-rate schedule and the step count carry on whether or not the run was interrupted. A run that has already
reached its total exits at once. ``--resume`` on a folder with no checkpoint starts the run; on a folder whose
checkpoints are all STRUCTURALLY broken (missing sidecar, not a zip, missing SB3 members, a pickle that does not load)
it moves them to ``checkpoints/broken_<time>/`` and starts again from step 0. An I/O error while reading them is not
a verdict on the files: the checkpoints are read again (``--io-retries`` times, ``--io-retry-wait`` s apart) and, if
it persists, the run is refused (exit 2) with nothing moved. A quarantine is written to run_record.json the moment
the files are moved. Each session records every save and how far it got in run_record.json as it goes (atomically),
so a killed session still leaves that record. A step-0 start is refused when the folder lists no resumable
checkpoint but, since the run's last quarantine, a session recorded saves that are nowhere under the checkpoint
folder (broken_*/ and nonfinite/ included), all its saves are non-finite (nothing to resume from), or an older-format
session (no list of saves) reached the first checkpoint interval or has no recorded end. Every refusal comes before
any file is moved, removed or written.

A run keeps its settings (model, env config, PPO settings, envs, seed, total steps, schedules): a resume with other
values is refused unless ``--allow-changes`` is given, and then the changes are logged in run_record.json. The
policy's shape (``--net``, ``--log-std-init``) never changes within a run. Changing ``--total-steps`` moves the
learning-rate schedule (it is linear in steps / total): extend a finished run under a new run folder instead.

Stopping: SIGINT or SIGTERM ends training; the first one saves a final pair (written under temporary names, then
renamed) and the session's status, with both signals ignored while it saves; the exit code is then 130. The
SubprocVecEnv workers ignore SIGINT (a terminal's Ctrl-C reaches the whole process group); the main process closes
them. ``--exit-with-parent [PID]`` (Linux; the notebook passes its kernel's PID) turns the death of the parent
process (a restarted notebook kernel) into a SIGTERM, so a training process never outlives its notebook; it refuses
to start (exit 3) if the parent is already gone. A pair whose parameters are not finite (a diverged run) is saved
under ``checkpoints/nonfinite/`` and is never resumed from.

Writes into RUN_DIR (every JSON file written to a temporary name, then renamed):
  checkpoints/                   pairs every --checkpoint-every steps and at the end of every session
  run_record.json                settings, package versions, commit, one entry per session (status, steps, rate)
  eval_during_training.jsonl     one line per periodic deterministic evaluation (fixed commands, 10 ms delay)
  monitor/session<k>/            SB3 Monitor CSVs, one per env and session
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

from trike_checkpoints import (  # noqa: E402
    CANDIDATE_RE,
    NONFINITE_DIR,
    files_present,
    pair_names,
    quarantine_broken,
    remove_tmp_files,
    save_pair,
    scan_pairs,
    write_json_atomic,
)

FIXED_SETTINGS = ("net", "log_std_init")  # the policy's shape: a checkpoint cannot take another
PPO_SETTINGS = ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef")
EVAL_SEED = 900_000  # fixed episode seeds for the periodic evaluation, so evaluations compare across a run


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory (checkpoints, run_record.json)")
    ap.add_argument("--total-steps", type=int, required=True, help="total env steps of the run (not an increment)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--model", default=None,
                    help="MJCF path, or a model name: trike60_C_closed (default), trike60_C_closed_ankleroll or "
                         "trike60_C_closed_capsule")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="continue from the newest intact checkpoint pair in --out")
    ap.add_argument("--allow-changes", action="store_true",
                    help="on --resume, accept settings that differ from the run's (they are logged)")
    ap.add_argument("--exit-with-parent", nargs="?", type=int, const=0, default=None, metavar="PID",
                    help="Linux: SIGTERM when the parent process dies; with PID, refuse to start unless that "
                         "process is still the parent (the notebook passes its own PID)")
    ap.add_argument("--io-retries", type=int, default=3, help="re-reads of the checkpoints after an I/O error")
    ap.add_argument("--io-retry-wait", type=float, default=20.0, help="seconds between those re-reads")
    ap.add_argument("--checkpoint-every", type=int, default=250_000, help="env steps between checkpoint pairs")
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
    ap.add_argument("--log-std-init", type=float, default=-1.6,
                    help="initial log std of the action noise (-1.6: std 0.2; with -1.0, std 0.37, a 300k-step probe "
                         "moved only through its noise; README)")
    ap.add_argument("--net", default="256,256", help="hidden sizes of the policy and value MLPs")
    # schedules and evaluation
    ap.add_argument("--penalty-ramp-steps", type=int, default=2_000_000,
                    help="env steps over which the energy and slip penalties ramp up to full weight (0: full at once)")
    ap.add_argument("--penalty-ramp-start", type=float, default=0.1, help="their scale at step 0")
    ap.add_argument("--late-ramp-from", type=int, default=1_000_000,
                    help="env steps before which the LATE penalties (heat, saturation, short air times) are off")
    ap.add_argument("--late-ramp-to", type=int, default=3_000_000,
                    help="env steps at which the late penalties reach full weight (linear from --late-ramp-from)")
    ap.add_argument("--terrain-ramp-from", type=int, default=1_000_000,
                    help="env steps until which the terrain, pushes and low friction stay at scale 0 (when the env "
                         "config turns them on)")
    ap.add_argument("--terrain-ramp-to", type=int, default=4_000_000,
                    help="env steps at which they reach full scale (linear from --terrain-ramp-from)")
    ap.add_argument("--eval-every", type=int, default=250_000,
                    help="env steps between deterministic evaluations (0: none)")
    ap.add_argument("--eval-commands", default="0,0.15,0.25", help="commanded speeds of those evaluations, m/s")
    ap.add_argument("--eval-seconds", type=float, default=0.0, help="their episode length (0: the env's own)")
    ap.add_argument("--checks-every", type=int, default=0,
                    help="every this many steps, the periodic checks (telemetry/checks.py; 0: off): a few "
                         "deterministic episodes at each of --checks, one JSON line in checks_during_training.jsonl")
    ap.add_argument("--checks", default=",".join(CHECK_POINTS),
                    help=f"the checks' points, from {', '.join(CHECK_POINTS)}")
    # env
    ap.add_argument("--env-config", default=None, help="JSON file of TrikeEnvConfig fields to override")
    ap.add_argument("--log-every", type=int, default=100_000, help="env steps between one-line progress reports")
    ap.add_argument("--verbose", type=int, default=0, help="SB3's own logging (1: a table every rollout)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------- small helpers
def say(text: str) -> None:
    """Print one line, flushed. If the reader of stdout is gone (a killed notebook kernel), stop printing instead of
    failing: the save that follows a stop must not die on a broken pipe."""
    try:
        print(text, flush=True)
    except (BrokenPipeError, OSError, ValueError):
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_record(path: Path) -> tuple[dict, str | None]:
    """run_record.json, or ``{}`` and why it could not be read. Changes nothing: main() moves an unreadable file
    aside only once every refusal has been passed."""
    if not path.exists():
        return {}, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (ValueError, UnicodeDecodeError) as exc:
        return {}, f"{path.name} could not be read ({exc})"


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

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def resolve_model(arg: str | None) -> str:
    from trike_env import DEFAULT_MODEL, MODELS

    if not arg:
        return str(DEFAULT_MODEL)
    if arg in MODELS:
        return str(MODELS[arg])
    path = Path(arg).resolve()
    if not path.exists():
        raise SystemExit(f"--model {arg}: no such file, and not one of {sorted(MODELS)}")
    return str(path)


def env_config(args):
    from trike_env import TrikeEnvConfig

    cfg = TrikeEnvConfig(model_path=resolve_model(args.model))
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


# The periodic checks' points (robustness-sweep points, rl/robustness/sweep.py) and seeds: the sweep's own conditions
# run 3 is judged on, a few seeds each, so terrain, push and slippery-floor skill can be watched while it appears.
CHECK_POINTS = {"step_3cm": {"terrain": ("step", 0.03)}, "ramp_8deg": {"terrain": ("ramp", 8.0)},
                "friction_0.3": {"friction": 0.3}, "push_1.0": {"push_dv": 1.0}}
CHECK_SEEDS = (300_000, 300_001)


def check_points(text: str) -> dict:
    names = [x.strip() for x in str(text).split(",") if x.strip()]
    unknown = [n for n in names if n not in CHECK_POINTS]
    if unknown:
        raise SystemExit(f"--checks: unknown point(s) {unknown}; known: {sorted(CHECK_POINTS)}")
    return {n: CHECK_POINTS[n] for n in names}


def eval_commands(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if x.strip()]


def run_settings(args, cfg) -> dict:
    """What a run keeps across its sessions (JSON-shaped, so a recorded copy compares equal)."""
    return json.loads(json.dumps({
        "model": Path(cfg.model_path).name, "env_config": comparable_config(cfg.to_dict()), "n_envs": args.n_envs,
        "seed": args.seed, "total_steps": args.total_steps, "checkpoint_every": args.checkpoint_every,
        "n_steps": args.n_steps, "batch_size": min(args.batch_size, args.n_steps * args.n_envs),
        "n_epochs": args.n_epochs, "lr_start": args.lr_start, "lr_end": args.lr_end, "gamma": args.gamma,
        "gae_lambda": args.gae_lambda, "clip_range": args.clip_range, "ent_coef": args.ent_coef,
        "log_std_init": args.log_std_init, "net": [int(x) for x in args.net.split(",") if x],
        "penalty_ramp_steps": args.penalty_ramp_steps, "penalty_ramp_start": args.penalty_ramp_start,
        "late_ramp_from": args.late_ramp_from, "late_ramp_to": args.late_ramp_to,
        **({"terrain_ramp_from": args.terrain_ramp_from, "terrain_ramp_to": args.terrain_ramp_to}
           if rough(cfg) else {}),
        "eval_every": args.eval_every, "eval_commands": eval_commands(args.eval_commands),
        "eval_seconds": args.eval_seconds,
        **({"checks_every": args.checks_every, "checks": sorted(check_points(args.checks))}
           if args.checks_every > 0 else {}),
    }, default=list))


def rough(cfg) -> bool:
    """Whether the env config turns on any of the terrain-scaled features (terrain, pushes, low friction)."""
    return cfg.terrain_boxes > 0 or cfg.push_prob > 0 or cfg.friction_low_hard is not None


def flat_eval_config(cfg):
    """The env config of the evaluations during training: the run's own, with the terrain, pushes and low-friction
    end off (an older run's config is unchanged), so every run's evaluations are on the same flat floor."""
    from trike_env import flat_floor

    return flat_floor(cfg)


def settings_diff(old: dict, new: dict) -> dict:
    keys = sorted(set(old) | set(new))
    return {k: {"run": old.get(k), "now": new.get(k)} for k in keys if old.get(k) != new.get(k)}


def penalty_scale_at(steps: int, ramp_steps: int, start: float) -> float:
    """The energy and slip penalties' scale: ``start`` at step 0, linear to 1 at ``ramp_steps``."""
    if ramp_steps <= 0:
        return 1.0
    return start + (1.0 - start) * min(1.0, steps / ramp_steps)


def late_scale_at(steps: int, ramp_from: int, ramp_to: int) -> float:
    """The late penalties' scale (heat, saturation, short air times): 0 until ``ramp_from``, linear to 1 at
    ``ramp_to`` (a step at ``ramp_from`` when ``ramp_to`` <= ``ramp_from``)."""
    if steps <= ramp_from:
        return 0.0
    if ramp_to <= ramp_from:
        return 1.0
    return min(1.0, (steps - ramp_from) / (ramp_to - ramp_from))


def actual_ppo(model) -> dict:
    """The PPO settings the loaded or built model really uses (read back from it)."""
    return {"n_steps": int(model.n_steps), "batch_size": int(model.batch_size), "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma), "gae_lambda": float(model.gae_lambda),
            "clip_range": float(model.clip_range(1.0)), "ent_coef": float(model.ent_coef),
            "learning_rate": repr(model.learning_rate), "n_envs": int(model.n_envs),
            "net": list(model.policy_kwargs.get("net_arch", {}).get("pi", [])),
            "log_std_init": model.policy_kwargs.get("log_std_init")}


def math_close(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def exit_with_parent(expected_parent: int) -> str | None:
    """Linux: ask the kernel to send SIGTERM to this process when its parent dies (PR_SET_PDEATHSIG), then check
    the parent is still ``expected_parent``: a parent that died before the request would never be signalled.
    Returns why this process must not start, or None."""
    if not sys.platform.startswith("linux"):
        say("Note: --exit-with-parent works on Linux only; ignored here.")
        return None
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # 1 = PR_SET_PDEATHSIG
        say(f"WARNING: --exit-with-parent: prctl failed (errno {ctypes.get_errno()}); this process will not stop "
            "with its parent")
    if os.getppid() != expected_parent:
        return f"the parent process {expected_parent} is gone (the parent is now {os.getppid()})"
    return None


# ---------------------------------------------------------------------------------------------------- envs, pairs
def build_venv(cfg, args, monitor_dir: Path):
    from functools import partial

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    monitor_dir.mkdir(parents=True, exist_ok=True)
    use_subproc = args.vec == "subproc" or (args.vec == "auto" and args.n_envs > 1)
    fns = [partial(_monitored_env, cfg.to_dict(), str(monitor_dir / f"env{i}"), use_subproc)
           for i in range(args.n_envs)]
    return SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)


def _monitored_env(cfg_dict: dict, monitor_path: str, in_worker: bool):
    from stable_baselines3.common.monitor import Monitor
    from trike_env import make_env

    if in_worker:  # a SubprocVecEnv worker: Ctrl-C is the main process's to handle; it closes the workers
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    return Monitor(make_env(cfg_dict), filename=monitor_path,
                   info_keywords=("episode_forward_speed", "command_speed", "termination"))


def unaccounted_saves(record: dict, ckpt_dir: Path, checkpoint_every: int) -> str | None:
    """Why this run must not start at step 0 (a sentence for the refusal), or None. Asked only when the checkpoint
    folder lists no resumable checkpoint. Raises OSError if the folder cannot be listed.

    Sessions before the run's last quarantine do not count (their files were set aside; the quarantine is in
    run_record.json from the moment the files are moved). A session that records its saves (``saves``, this
    format) is accounted for when each of them is somewhere under the checkpoint folder (broken_*/ and nonfinite/
    included); one whose saves are all non-finite leaves nothing to resume from. A session in the older format
    (no ``saves``) is trusted only if it recorded an end and did not reach the first checkpoint interval."""
    sessions = list(record.get("sessions") or [])
    since = max([0] + [int(q.get("session_index", 0)) for q in record.get("quarantined") or []]
                + [i for i, x in enumerate(sessions) if x.get("quarantined")])
    present = files_present(ckpt_dir)
    missing, nonfinite = [], []
    for i, x in enumerate(sessions[since:], start=since):
        for n in [*(x.get("saves") or []), x.get("last_checkpoint"), x.get("started_from")]:
            if not isinstance(n, str):
                continue
            if n.startswith(f"{NONFINITE_DIR}/"):
                nonfinite.append(n)
            elif CANDIDATE_RE.fullmatch(n) and n not in present:
                missing.append(n)
        if "saves" not in x:  # the older record format: what it saved is not known
            reached = max(int(x.get("steps_end") or 0), int(x.get("steps_reached") or 0))
            if reached >= checkpoint_every or not x.get("ended_utc"):
                what = f"reached {reached} steps" if reached >= checkpoint_every else "has no recorded end"
                return (f"no checkpoint is listed in {ckpt_dir}, but session {i} (an older record format, which does "
                        f"not list its saves) {what} and may have saved checkpoints. Starting at step 0 could "
                        "overwrite them. Run again later; if they are really gone, use a new run folder.")
    if missing:
        return (f"no checkpoint is listed in {ckpt_dir}, but run_record.json names saves that are not there: "
                f"{', '.join(sorted(set(missing))[:4])}. Starting at step 0 could overwrite them. Run again later; "
                "if they are really gone, use a new run folder.")
    if nonfinite:
        return (f"there is no finite checkpoint to resume from: this run's saves since its last restart are all "
                f"non-finite (training diverged; they are kept in {ckpt_dir / NONFINITE_DIR}). Starting again at step 0 "
                "with the same settings would likely diverge again: use a new run folder, with other settings.")
    return None


# ---------------------------------------------------------------------------------------------------- callbacks
def make_callbacks(args, cfg, ckpt_dir: Path, out: Path, state: dict):
    from stable_baselines3.common.callbacks import BaseCallback
    from trike_env import TrikeEnv

    def first_multiple_above(n: int, every: int) -> int:
        return (n // every + 1) * every

    class PairSaver(BaseCallback):
        """A checkpoint pair at every multiple of --checkpoint-every steps (atomic; prints one line)."""

        def _on_training_start(self) -> None:
            self.next = first_multiple_above(self.model.num_timesteps, args.checkpoint_every)

        def _on_step(self) -> bool:
            n = self.model.num_timesteps
            if n >= self.next:
                while self.next <= n:
                    self.next += args.checkpoint_every
                name, finite = save_pair(self.model, self.model.get_vec_normalize_env(), ckpt_dir)
                state["last_pair"] = name
                state["record_progress"](n, save=name)  # in run_record.json before the line is printed
                if finite:
                    say(f"Saved checkpoint pair at {n} steps: {name} and {pair_names(n)[1]}")
                else:
                    say(f"WARNING: the parameters are not finite at {n} steps: saved as checkpoints/{name}, which "
                        "is never resumed from")
            return True

    class PenaltyRamp(BaseCallback):
        """The schedules, set in every env before each rollout: the penalty scale (energy, slip), the late scale
        (heat, saturation, short air times) and, when the env config uses them, the terrain scale (steps, slopes,
        pushes, low friction; 0 until --terrain-ramp-from, as late_scale_at)."""

        def _on_rollout_start(self) -> None:
            n = self.model.num_timesteps
            s = penalty_scale_at(n, args.penalty_ramp_steps, args.penalty_ramp_start)
            if s != state.get("penalty_scale"):
                self.training_env.env_method("set_penalty_scale", s)
                state["penalty_scale"] = s
            late = late_scale_at(n, args.late_ramp_from, args.late_ramp_to)
            if late != state.get("late_scale"):
                self.training_env.env_method("set_late_scale", late)
                state["late_scale"] = late
            if rough(cfg):
                ts = late_scale_at(n, args.terrain_ramp_from, args.terrain_ramp_to)
                if ts != state.get("terrain_scale"):
                    self.training_env.env_method("set_terrain_scale", ts)
                    state["terrain_scale"] = ts

        def _on_step(self) -> bool:
            return True

    class Progress(BaseCallback):
        """One line every --log-every steps: steps, rate, and the training rollouts' recent episodes."""

        def _on_training_start(self) -> None:
            self.t0, self.n0 = time.time(), self.model.num_timesteps
            self.next = first_multiple_above(self.n0, args.log_every)

        def _on_step(self) -> bool:
            n = self.model.num_timesteps
            if n >= self.next:
                while self.next <= n:
                    self.next += args.log_every
                rate = (n - self.n0) / max(time.time() - self.t0, 1e-9)
                eps = list(self.model.ep_info_buffer)
                if eps:
                    def mean(key):
                        return sum(float(e.get(key, 0.0)) for e in eps) / len(eps)

                    ep = (f"last {len(eps)} episodes: return {mean('r'):.1f}, length {mean('l'):.0f} steps, speed "
                          f"{mean('episode_forward_speed'):.3f} m/s at command {mean('command_speed'):.3f} m/s")
                else:
                    ep = "no episode finished yet"
                state["record_progress"](n)  # in run_record.json before the line is printed
                say(f"steps {n} of {args.total_steps} ({100.0 * n / args.total_steps:.1f}%), {rate:.0f} env "
                    f"steps/s, penalty scale {state.get('penalty_scale', 1.0):.2f}, late scale "
                    f"{state.get('late_scale', 1.0):.2f}"
                    + (f", terrain scale {state.get('terrain_scale', 0.0):.2f}" if rough(cfg) else "")
                    + "; training rollouts "
                    f"(stochastic actions), {ep}")
            return True

    class PeriodicEval(BaseCallback):
        """Every --eval-every steps: one episode per fixed command, deterministic actions, randomisation on, servo
        delay fixed at 10 ms, fixed seeds, on the flat floor (``flat_eval_config``); one printed line and one JSON
        line in eval_during_training.jsonl."""

        def _on_training_start(self) -> None:
            seconds = args.eval_seconds or cfg.episode_seconds
            self.env = TrikeEnv(flat_eval_config(cfg).replace(delay_steps_fixed=10, episode_seconds=seconds,
                                                              domain_randomization=True, observation_noise=True))
            self.next = first_multiple_above(self.model.num_timesteps, args.eval_every)

        def _on_step(self) -> bool:
            n = self.model.num_timesteps
            if n >= self.next:
                while self.next <= n:
                    self.next += args.eval_every
                self.evaluate(n)
            return True

        def evaluate(self, n: int) -> None:
            import gait_metrics

            vn = self.model.get_vec_normalize_env()

            def act(obs):
                a, _ = self.model.predict(vn.normalize_obs(obs[None, :]), deterministic=True)
                return a[0]

            rows = []
            for i, c in enumerate(eval_commands(args.eval_commands)):
                log = gait_metrics.record_episode(self.env, act, seed=EVAL_SEED + i, command=c, delay_steps=10)
                rows.append({"command_mps": c, "forward_speed_mps": log["episode_forward_speed"],
                             "fell": log["fell"], "termination": log["termination"], "length_s": log["time_s"],
                             "measures": gait_metrics.measures(log, c)})
            rec = {"steps": n, "time_utc": utc(), "deterministic": True, "delay_steps": 10, "episodes": rows}
            with open(out / "eval_during_training.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            parts = ", ".join(f"{r['command_mps']:.2f} -> {r['forward_speed_mps']:.3f}" for r in rows)
            falls = sum(r["fell"] for r in rows)
            say(f"eval at {n} steps (deterministic, delay 10 ms): command -> speed m/s: {parts}; falls {falls} of "
                f"{len(rows)}")
            top = rows[-1]  # the highest command: its drive, slip and step figures on one line
            say(f"eval at {n} steps, command {top['command_mps']:.2f} m/s: "
                f"{gait_metrics.one_line(gait_metrics.summarise([top['measures']]))}")

    from telemetry.health import TrainingHealth

    cbs = [PairSaver(), PenaltyRamp(), Progress(), TrainingHealth(out / "train_health.jsonl", say=say)]
    if args.eval_every > 0:
        cbs.append(PeriodicEval())
    if args.checks_every > 0:
        cbs.append(make_checks(args, cfg, out))
    return cbs


def make_checks(args, cfg, out: Path):
    """The periodic checks (telemetry/checks.py) at robustness-sweep points on the run's own config (the sweep
    builds a flat-floor env from it and adds its own step, ramp, push or friction): nominal model, noise as
    configured, 10 ms delay, 0.25 m/s, deterministic actions."""
    for sub in ("robustness", "analysis"):
        if str(HERE / sub) not in sys.path:
            sys.path.insert(0, str(HERE / sub))
    import sweep
    from telemetry.checks import PeriodicChecks

    ec = json.loads(json.dumps(cfg.to_dict(), default=list))
    envs: dict = {}

    def run_point(name, spec, seed, act):
        p = sweep.make_point(**spec)
        if name not in envs:
            envs[name] = sweep.build_env(ec, p["terrain"])
        return sweep.run_episode(envs[name], None, None, seed, p, controller=act)

    def summarise(name, eps):
        ok = [e["forward_speed_mps"] for e in eps if not e["fell"] and e.get("forward_speed_mps") is not None]
        sp = f", {sum(ok) / len(ok):.3f} m/s" if ok else ""
        return f"{name} falls {sum(e['fell'] for e in eps)}/{len(eps)}{sp}"

    return PeriodicChecks(out / "checks_during_training.jsonl", args.checks_every, check_points(args.checks),
                          run_point, seeds=CHECK_SEEDS, say=say, summarise=summarise)


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ppid0 = os.getppid()
    args = parse_args(argv)
    if args.exit_with_parent is not None:
        problem = exit_with_parent(args.exit_with_parent or ppid0)
        if problem:
            say(f"REFUSED: {problem}. Nothing was started.")
            return 3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.curriculum.schedules import LinearSchedule
    from environments.shared.policy_loading import load_sb3_model

    torch.set_num_threads(max(1, args.torch_threads))
    out = Path(args.out).resolve()
    ckpt_dir = out / "checkpoints"
    record_path = out / "run_record.json"

    # ---- 1. Look, change nothing: every refusal comes before any file is moved, removed or written.
    scan = scan_pairs(ckpt_dir, retries=args.io_retries, wait_s=args.io_retry_wait, log=say)
    if scan.io_error:
        say(f"REFUSED: the checkpoint files could not be read ({scan.io_error}), also after {args.io_retries} "
            "re-reads; nothing was moved or changed. Run again later; if it repeats, the storage has a problem.")
        return 2
    pair = scan.pair
    for s in scan.skipped:
        say(f"WARNING: skipping checkpoint {s}")
    if not args.resume and scan.candidates:
        say(f"REFUSED: {ckpt_dir} already holds checkpoints; pass --resume to continue that run, or use another "
            "--out. Nothing was changed.")
        return 2
    record, record_problem = read_record(record_path)
    if record_problem:
        say(f"WARNING: {record_problem}; its settings cannot be checked")
    cfg = env_config(args)
    settings = run_settings(args, cfg)
    recorded = record.get("settings")
    changes = settings_diff(recorded, settings) if recorded else {}
    if changes:
        listed = "; ".join(f"{k}: run {v['run']!r}, now {v['now']!r}" for k, v in changes.items() if k != "env_config")
        if "env_config" in changes:
            old, new = changes["env_config"]["run"] or {}, changes["env_config"]["now"] or {}
            fields = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            listed = "; ".join(x for x in (listed, f"env_config fields {', '.join(fields)}") if x)
        fixed = [k for k in changes if k in FIXED_SETTINGS]
        if fixed:
            say(f"REFUSED: the policy's shape ({', '.join(fixed)}) cannot change within a run ({listed}). Use a new "
                "run folder. Nothing was changed.")
            return 2
        if not args.allow_changes:
            say(f"REFUSED: these settings differ from the run's (run_record.json): {listed}. Use the run's own "
                "values, a new run folder, or --allow-changes (logged). Nothing was changed.")
            return 2
    if not scan.candidates and record.get("sessions"):  # a step-0 start of a run that has sessions already
        try:
            reason = unaccounted_saves(record, ckpt_dir, int((recorded or {}).get("checkpoint_every")
                                                             or args.checkpoint_every))
        except OSError as exc:
            reason = (f"the checkpoint folder {ckpt_dir} could not be listed ({exc!r}), so its saves cannot be "
                      "checked. Run again later.")
        if reason:  # e.g. a storage that answers "no folder" or lists it empty for a while
            say(f"REFUSED: {reason} Nothing was changed.")
            return 2
    if pair is not None and pair[0] >= args.total_steps:
        say(f"Run already complete: newest checkpoint is at {pair[0]} steps (target {args.total_steps}). "
            "Nothing to do.")
        return 0

    # ---- 2. Act.
    if changes:
        say(f"WARNING: --allow-changes: this session runs with changed settings: {listed}")
        if "total_steps" in changes and pair is not None:
            say("WARNING: total_steps changed: the learning rate follows steps / total_steps, so it jumps at this "
                "resume.")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    for name in remove_tmp_files(ckpt_dir):
        notes.append(f"removed the unfinished temporary file {name}")
        say(f"Note: removed the unfinished temporary file {name}")
    if record_problem:
        aside = record_path.with_name(f"run_record.broken_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        os.replace(record_path, aside)
        notes.append(f"{record_problem}; moved to {aside.name}")
        say(f"WARNING: {record_path.name} moved to {aside.name}")
    quarantine = None
    if pair is None and scan.skipped:  # with --resume: every checkpoint is structurally broken; keep them aside
        qdir, moved = quarantine_broken(ckpt_dir)
        quarantine = {"time_utc": utc(), "folder": str(qdir.relative_to(out)), "files": moved,
                      "reasons": scan.skipped, "session_index": len(record.get("sessions") or [])}
        record.setdefault("quarantined", []).append(quarantine)
        write_json_atomic(record_path, record)  # at once: a session killed now must not leave it unrecorded
        say(f"WARNING: no intact checkpoint pair; moved {len(moved)} broken file(s) to "
            f"{qdir.relative_to(out)}/ and starting this run again from step 0.")

    cpu = os.cpu_count() or 1
    say(f"Machine: {cpu} vCPUs; {args.n_envs} envs; Python {platform.python_version()}; torch {torch.__version__}"
          + ("; WARNING: more envs than vCPUs, each env runs slower" if args.n_envs > cpu else ""))
    ppo_kwargs = dict(
        n_steps=settings["n_steps"], batch_size=settings["batch_size"], n_epochs=args.n_epochs, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=0.5,
        max_grad_norm=1.0, learning_rate=LinearSchedule(args.lr_start, args.lr_end),
    )
    policy_kwargs = dict(net_arch=dict(pi=settings["net"], vf=settings["net"]), activation_fn=torch.nn.ELU,
                         log_std_init=args.log_std_init)

    session_index = len(record.get("sessions", []))
    venv = build_venv(cfg, args, out / "monitor" / f"session{session_index}")  # Monitor would overwrite old CSVs
    session_seed = args.seed if pair is None else args.seed + int(pair[0])  # a resumed session draws new episodes

    if pair is None:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, seed=args.seed, device=args.device,
                    verbose=args.verbose, **ppo_kwargs)
        say(f"Starting a new run in {out} (0 of {args.total_steps} steps).")
    else:
        steps0, zip_path, vn_path = pair
        venv = VecNormalize.load(str(vn_path), venv)
        venv.training, venv.norm_reward = True, True
        if "gamma" in changes:  # the reward normaliser discounts with its own copy of gamma
            venv.gamma = args.gamma
            say(f"Note: --allow-changes: VecNormalize's gamma set to {args.gamma} as well")
        # Without --allow-changes nothing is passed and the archive's own settings apply (they equal the run's).
        overrides = {k: ppo_kwargs[k] for k in PPO_SETTINGS if k in changes}
        if "lr_start" in changes or "lr_end" in changes:
            overrides["learning_rate"] = ppo_kwargs["learning_rate"]
        model = load_sb3_model(zip_path, algorithm="ppo", env=venv, device=args.device, **overrides)
        model.verbose = args.verbose
        model.set_random_seed(session_seed)  # PPO.load re-seeds the envs with the run's first seed; draw anew
        say(f"Resuming {out} from {zip_path.name} ({model.num_timesteps} of {args.total_steps} steps).")

    in_use = actual_ppo(model)
    differs = [k for k in PPO_SETTINGS if not math_close(in_use[k], settings[k])]
    if differs:  # cannot happen unless an archive was edited: say so rather than record the wrong values
        say(f"WARNING: the model uses other PPO settings than recorded: "
            f"{', '.join(f'{k} {in_use[k]!r} (recorded {settings[k]!r})' for k in differs)}")
    session = {
        "started_utc": utc(), "status": "running", "argv": sys.argv[1:] if argv is None else [str(a) for a in argv],
        "args": vars(args), "settings": settings, "ppo_in_use": in_use, "changed_settings": changes or None,
        "versions": versions(), "git": git_state(), "cpu_count": cpu,
        "started_from": pair[1].name if pair is not None else "step 0", "steps_start": int(model.num_timesteps),
        "session_seed": session_seed, "quarantined": quarantine, "notes": notes or None,
        "vecnormalize_gamma": float(venv.gamma),
        "saves": [], "steps_reached": int(model.num_timesteps),  # updated at every save and progress line
    }
    record.setdefault("created_utc", session["started_utc"])
    record["settings"] = settings
    record.setdefault("sessions", []).append(session)
    write_json_atomic(record_path, record)

    def record_progress(n: int, save: str | None = None) -> None:
        """This session's progress, and every save, into run_record.json (atomically), so a session that is killed
        still tells the next one how far it got and what it saved."""
        session["steps_reached"] = int(n)
        if save:
            session["saves"].append(save)
            session["last_checkpoint"] = save
        try:
            write_json_atomic(record_path, record)
        except OSError as exc:
            say(f"WARNING: run_record.json could not be updated ({exc!r}); training goes on")

    state: dict = {"stop": None, "last_pair": None, "record_progress": record_progress}

    def on_signal(signum, frame):  # the first SIGINT/SIGTERM ends training; any repeat is ignored
        if state["stop"] is not None:
            return
        state["stop"] = signum
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    callbacks = make_callbacks(args, cfg, ckpt_dir, out, state)
    remaining = args.total_steps - int(model.num_timesteps)
    t0, n0 = time.time(), int(model.num_timesteps)
    status, code = "completed", 0
    try:
        # The schedules' scales at this step, set before learn(): SB3 resets the envs before any callback runs, so
        # without this every session's first episode in each env ran at the envs' default scales of 1 (since
        # 19 Nov 2026; earlier runs, pipe1 and run2b among them, trained their first episodes at scale 1).
        state["penalty_scale"] = penalty_scale_at(n0, args.penalty_ramp_steps, args.penalty_ramp_start)
        state["late_scale"] = late_scale_at(n0, args.late_ramp_from, args.late_ramp_to)
        venv.env_method("set_penalty_scale", state["penalty_scale"])
        venv.env_method("set_late_scale", state["late_scale"])
        if rough(cfg):
            state["terrain_scale"] = late_scale_at(n0, args.terrain_ramp_from, args.terrain_ramp_to)
            venv.env_method("set_terrain_scale", state["terrain_scale"])
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=pair is None,
                    progress_bar=False)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # nothing may cut the save below short
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        status, code = "interrupted", 130
        say("Interrupted: saving a checkpoint pair before exiting.")
    except Exception as exc:  # a crash: still try to keep what was learned
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        status, code = f"crashed: {exc!r}", 1
        say(f"CRASHED: {exc!r}; saving a checkpoint pair before exiting.")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    wall = time.time() - t0
    finite = None
    try:
        last, finite = save_pair(model, venv, ckpt_dir)
        if not finite:
            say(f"WARNING: the parameters are not finite: saved as checkpoints/{last}, which is never resumed from")
            code = code or 1
    except Exception as exc:
        last = f"(not saved: {exc!r})"
        code = code or 1
    n_end = int(model.num_timesteps)
    if finite is not None:
        session["saves"].append(last)
    session["steps_reached"] = n_end
    rate = (n_end - n0) / wall if wall > 0 else None
    session.update({"ended_utc": utc(), "status": status, "steps_end": n_end, "wall_s": wall,
                    "env_steps_per_s": rate, "last_checkpoint": last, "last_checkpoint_finite": finite})
    write_json_atomic(record_path, record)
    say(f"Session {status}: {n_end} steps (target {args.total_steps}); {rate or 0:.0f} env steps/s over "
          f"{wall:.0f} s; last pair {last}")
    try:
        venv.close()
    except Exception as exc:  # workers already gone after an interrupt
        say(f"(closing the envs: {exc!r})")
    return code


if __name__ == "__main__":
    sys.exit(main())
