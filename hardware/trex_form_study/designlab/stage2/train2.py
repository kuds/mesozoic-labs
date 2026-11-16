"""Track B's trainer for R5 and R6 (round C design D2, D12-D16; section 3.6; Track B rules O1-O4, M1-M6, K1-K5,
K8, K9; ``stage2/design/ROUND_C_RULINGS_16NOV.md``).  A new module: ``designlab/stage1/train.py`` and ``policy.py``
are imported, not edited.

    python -m designlab.stage2.train2 run --mode R5|R6 --body B --seed N [--set V6] --out ROOT [--job NAME] [--steps N]
           [--workers W] [--resume] [--allow-version-change] [--allow-source-change] [--from-checkpoint JOBDIR:STEPS]
           [--lr X] [--monitor-every N] [--monitor-also-k K,...] [--no-gates] [--tiny] [--run-id ID] [--config JSON]
           [--retry-check]
    python -m designlab.stage2.train2 {probe-lr|probe-sigma|filter} --bodies B,... --set S --out PATH ...
    python -m designlab.stage2.train2 preflight
    python -m designlab.stage2.train2 worker ADDRESS        # a worker of the env pool (started by the pool)
    python -m designlab.stage2.train2 monitor ...           # the monitor child: designlab.stage2.monitor2's CLI

**A run** (``train``) is one job directory ``<root>/jobs/<job>``: ``run.json`` (schema ``s2-run-1``),
``updates.jsonl`` (one line per PPO update), ``progress.json`` (rewritten at every rollout start),
``gates/g8a.json`` and ``gates/rate.json``, ``checkpoints/`` (``s2load``'s four files per checkpoint) and
``monitor/`` (``monitor2``'s records).  8 envs in W worker processes (``vecpool.PoolVecEnv``; the workers are this
module's ``worker`` entry, so every process of a run carries ``designlab.stage2.train2`` on its command line);
design 2's PPO with ``n_steps`` 1024, sigma 0.1 and schedules that are objects (``schedules``), so that an archive
holds no bytecode; ``VecNormalize``; one torch thread.

**Boundaries** (rule K1, ``BOUNDARY_RULE``).  At step 0 and at the first rollout start at or past each multiple of
``checkpoint_every`` the callback saves a checkpoint with the run state and then, in every run whether interrupted
or not, reseeds SB3, torch, numpy and the vec env with ``100,000 x seed + 100 x b``, resets every env, zeroes the
reward normalisation's running returns and sets the model's last observation from that reset.  A resume loads the
newest intact checkpoint with ``force_reset=False``, takes the run state from the checkpoint's record, and performs
the same boundary at its first rollout start: a resumed run is the uninterrupted one bit for bit.

**Checks** (rules M1-M5).  At the first rollout start at or past each multiple of ``monitor_every`` and on the
final checkpoint the monitor child runs on the checkpoint just saved; each attempt is logged in ``run.json``
``monitor[]`` before the child starts and closed with its exit code.  Everything read after a check is
``monitor2.decide`` on the records: the gain, G8b with its one fix (the task weights, in place:
``env_method("set_task_weights")``), the stop rules.  A stop by a rule ends the job with exit code 0.

**Exit codes** of ``run``: 0 done, or stopped by rule M4 or M5; 3 G8a failed ("G8a fails: no run"); 4 refused (a
resume the rules refuse, or a start that cannot be made: nothing was run); 5 halted for the lead (R0L's baseline
could not be made; two attempts at one check were never closed: ``run.json`` ``halt`` holds the reason, and
``--retry-check`` on the resume lets that check run once more); any other code is a failure.

**The seams.**  ``_env_configs``, ``_env_specs``, ``_g8a_short``, ``_monitor_cmd`` and ``_validation`` are the
places where the trainer meets the env, the gate, the monitor child and the validation records; the tests replace
them with a stub env and a stub monitor.  Nothing else in the trainer knows which env it trains.

torch, stable-baselines3, mujoco and the env are imported inside the functions that need them: a worker imports
this module and none of those.
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import os  # noqa: E402
import sys  # noqa: E402

if __name__ == "__main__":  # before any numpy import (round A's module skeleton)
    for _v in tbenv.THREAD_VARS:
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage2 import s2io, schedules  # noqa: E402

SCHEMA = "s2-run-1"
PROGRESS_SCHEMA = "s2-progress-1"
RATE_SCHEMA = "s2-rate-1"
MODES = ("R5", "R6")
#: Rule O1: 8 envs in every run and session; a rollout is 8 x 1,024 steps.
N_ENVS = 8
N_STEPS = 1024
CHECKPOINT_EVERY = 250_000
MONITOR_EVERY = 2_500_000
#: Rule O2: the initial sigma of each action's range.
SIGMA = 0.1
#: Rule O3: the schedule is linear from the run's rate to this share of it over the run's budget.
LR_END_RATIO = 0.3
PROBE_RATES = (1e-4, 3e-5, 1e-5)
PROBE_SIGMAS = (0.1, 0.03)
PROBE_UPDATES = 6
PROBE_SEED = 42
#: Rule K8: the rates a fine-tune may take when the parent's schedule value does not run every minibatch step.
FINE_TUNE_RATES = (1e-4, 3e-5, 1e-5, 3e-6)
#: Rule O2's probe: the episode seeds, the sigmas, the delay (physics steps).
PROBE_EPISODES = tuple(range(7000, 7010))
PROBE_SIGMA_VALUES = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2)
PROBE_K = 5
PROBE_TASKS = ("walk_nominal", "walk_v0.4")
PROBE_TASKS_S1 = ("walk", "brisk")
#: Rule O4: the candidates, the body the rule rests on, the share of the 10 Hz survival that 20 Hz must keep.
FILTER_CUTOFFS = (10.0, 20.0)
FILTER_BODY = "75-M-H-10"
FILTER_KEEP = 0.9
#: Rule K1: the seed of boundary b of PPO seed s is ``SEED_BASE s + SEED_STEP b``; env i resets with that + i.
SEED_BASE, SEED_STEP = 100_000, 100
#: ``--tiny`` (design 3.6 "Run level"): the rollout, the budget, the checkpoint and check intervals, G8a's cut and,
#: where no probe record gives one, the learning rate.
TINY = dict(n_steps=64, steps=4096, checkpoint_every=1024, monitor_every=2048, g8a_seconds=2.0, lr=3e-5)
S1_SET = "S1"
EXIT_DONE, EXIT_FAILED, EXIT_G8A, EXIT_REFUSED, EXIT_HALT = 0, 1, 3, 4, 5
#: Where the probes' records are read from when ``--lr`` or the filter is not given (design section 6).
VALIDATION = s2io.STUDY / "stage2" / "tb" / "validation"
WORKER_CMD = ("-m", "designlab.stage2.train2", "worker")

#: The files whose hash a session records and gives to its children (design 3.6 "Sources"): every Track B module
#: of the design's section 2.1, the round-A and Stage 2 modules the bar and the look reading execute, and the four
#: repository files the reward and the loader execute.  Relative to the study directory; a file that is not there
#: yet (a later work package's) is recorded as absent.
SOURCE_FILES = tuple(f"designlab/stage2/{m}.py" for m in (
    "tbenv", "schedules", "s2load", "r5lib", "r5core", "r5env", "g8a", "lookread", "lookbar", "s2rows", "vecpool",
    "train2", "monitor2", "clip2", "gates2", "r2lfit", "bundle", "nb", "bar", "readings", "setio", "s2io",
    "motion")) + tuple(f"../../environments/shared/{m}.py" for m in (
        "direction_commands", "command_frame", "policy_loading", "curriculum/checkpoints"))

BOUNDARY_RULE = (
    "rule K1: at step 0 and at the first rollout start at or past each multiple of checkpoint_every a checkpoint "
    "is saved; then, in every run whether interrupted or not, everything random (python, numpy, torch, the action "
    f"space and the vec env) is reseeded with {SEED_BASE:,} x seed + {SEED_STEP} x b (b = that multiple / "
    "checkpoint_every; env i resets with that + i), every env is reset, the reward normalisation's running returns "
    "are zeroed and the model's last observation and episode starts are set from that reset.  A resume loads the "
    "newest intact checkpoint without letting the library reset the envs, continues the step count and the "
    "schedule, and does the same boundary before its first rollout")
CHECKPOINT_RULE = (
    "a checkpoint is s2load's four files, the record last; its record holds the run state as it is before that "
    "step's check (the task weights in force, the fix, the monitor history, the no-gain count, the checks done, "
    "any stop, the flags).  Kept locally: each check's and the newest two (rule K2).  When learn() returns, a final "
    "checkpoint is saved if the last step has none")
RETRY_RULE = (
    "the lead's ruling of 16 Nov 2026 (the fifth list, 2): after the halt for two attempts at one check that were "
    "never closed, the lead has the fault repaired and resumes with a recorded parameter (--retry-check; "
    "sessions[].retry_check), which lets that check run once more per resume; the attempt is logged with "
    "after_halt")
RESUME_RULE = (
    "rules K4, K5: a resume is refused, in this order, on no run.json; a stopped run; other settings (every "
    "argument given must be the run's; the env configs as built today, but for the named freeze and the task "
    "names, must be the run's); other sources than the last session's unless --allow-source-change; other library "
    "versions than the last session's (s2load.comparable) unless --allow-version-change; other inputs (the envs are "
    "built from the run's stored configs, which name the freeze the run started on and its tasks: a used file's "
    "sha256 or entry hash, the plant's inputs or the resolved task table that differs; a later freeze of the set "
    "is accepted when these are unchanged, and both freezes are recorded); a checkpoint whose run_uuid or settings "
    "hash is not the run's; a fingerprint that is not reproduced within 1e-5.  A checkpoint that fails its "
    "integrity check is skipped with its reason.  No intact checkpoint is no refusal: the job restarts from step 0 "
    "in place and is the same run.  The run state is taken from the checkpoint's record: run.json's live fields "
    "are rebuilt from it, later entries move to the session's discarded_after_resume (the attempts logged for a "
    "check on the checkpoint itself are kept), updates.jsonl is cut back to the resumed step, a check that is due "
    "at that step and has no record naming that checkpoint is run, and the rules are applied to the records as the "
    "uninterrupted run applies them")
UPDATES_RULE = (
    "updates.jsonl holds one line per PPO update: session, update (the run's count), steps (the env steps the "
    "policy was trained on after it), minibatch_steps_run (the optimizer steps PPO really made, counted by a "
    "wrapper around the optimizer's step: target_kl stops an update without a message, lesson 16) of "
    "minibatch_steps_possible, approx_kl, clip_fraction, explained_variance, value_loss, std_mean (the mean of "
    "exp(log_std)), lr, ep_len_mean and ep_rew_mean (over the episodes that ended in the update's rollout; the "
    "reward before normalisation; null if none ended) and wall_s (the rollout and the update)")
KL_CUT_RULE = (
    "rule O3: if the median share of minibatch steps run over a check interval (the updates after the previous "
    "check, up to this one) is under one half, run.json carries flags.kl_cut and the lead is told; it stops "
    "nothing")
RATE_RULE = (
    "gates/rate.json, per session: first_rollout (the session's first rollout: env steps per wall second, no "
    "update in it) and first_three (its first three rollouts, two updates included: S2-8's rate), each with the CPU "
    "seconds of the trainer and of the workers, the workers, the vCPUs, the resident memory of the trainer and of "
    "each worker, and the machine's load.  The clock starts after the session's first boundary, and what the "
    "trainer does at later rollout starts (a checkpoint, a check with its monitor child, a boundary) is left out of "
    "the wall seconds")
GATES_RULE = (
    "gates false (--no-gates): no G8a short form, no R0L baseline and no check, so no G8b, no fix and no stop rule; "
    "for validation runs and tests; it is a setting of the run")
FINE_TUNE_RULE = (
    "rule K8: a fine-tune is a new job directory and run id from a parent's checkpoint and statistics (the same "
    "layout), the parent's log_std, its own boundaries and checks, and a constant learning rate r: the parent's "
    "schedule value p at the parent's step if every minibatch step of each of 6 updates runs at p from the "
    "parent's checkpoint, else the largest of 1e-4, 3e-5, 1e-5, 3e-6 below p at which they all run; if none does "
    "the fine-tune does not run")
LR_PROBE_RULE = (
    "rule O3: for each body and head, for each rate from the largest down: a fresh R5 model, seed 42, training's "
    "randomisation, 8 envs on rule J5's floors, the rate held constant, 6 updates, once at sigma 0.1 and once with "
    "log_std set to ln 0.03; a rate passes on a body and head when every minibatch step ran in each update at both "
    "sigmas; the probe stops at the first rate that passes.  A body's rate is the smallest passing rate over its "
    "heads that were probed")
SIGMA_PROBE_RULE = (
    "rule O2 (reported; decides nothing): the stance-start walk_nominal and walk_v0.4 tasks at a servo delay of 5 "
    "physics steps on the nominal plant, on hard floor, with no push and no head event, 10 episodes each (episode "
    "seeds 7000-7009), actions clip(sigma z) per env step with the same z for every sigma (z from "
    "default_rng(episode seed)), through the env's clip and filter: survivals and the mean survival time per sigma")
FILTER_RULE = (
    "rule O4: rule O2's 20 episodes under each cut-off with the same z, the noise scaled so that the filtered "
    "action has the same RMS (sigma 0.1 at the first cut-off; x sqrt((a0 / (2 - a0)) / (a / (2 - a))) at another, a "
    "= 1 - exp(-2 pi f x 0.02 s)); 20 Hz is used if its mean survival time over 75-M-H-10's 20 episodes is at "
    "least 0.9 x that at 10 Hz, else 10 Hz; another body's measurement is reported and decides nothing")


class TrainError(RuntimeError):
    """The job cannot go on.  ``code`` is the exit code of ``run``."""

    code = EXIT_FAILED


class Refused(TrainError):
    """A resume the rules refuse, or a start that cannot be made: nothing was run."""

    code = EXIT_REFUSED


class Halt(TrainError):
    """The run goes to the lead before it can go on (rule M1)."""

    code = EXIT_HALT


class _StopTraining(Exception):
    """Raised at a rollout start when a rule stopped the run: ``learn`` ends without another env step."""


# ------------------------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------------------------
def sources() -> dict:
    """``{files: {path: sha256 | None}, plant_sources, panel_sources, sha256}``: the sha256 of every file of
    ``SOURCE_FILES`` (None: not there), ``plant.sources_sha256()`` and ``panels.panel_sources()``'s hash, and the
    hash of all of it.  A session records it and gives it to the monitor child, which refuses to run on another."""
    from designlab.stage1 import panels, plant

    files = {}
    for rel in SOURCE_FILES:
        p = (s2io.STUDY / rel).resolve()
        files[rel] = s2io.sha256_file(p) if p.is_file() else None
    doc = dict(files=files, plant_sources=plant.sources_sha256(), panel_sources=panels.panel_sources()["sha256"])
    return dict(doc, sha256=s2io.sha256_json(doc))


# ------------------------------------------------------------------------------------------------
# The seams (module docstring): the env, the gate, the monitor child, the validation records
# ------------------------------------------------------------------------------------------------
def _env_configs(mode: str, body: str, set_name: str, overrides: dict | None) -> list:
    """A job's 8 env configs as plain dicts (``R5Config.describe()``): env i on the floor of rule J5, the set
    pinned by its freeze today.  ``overrides`` replaces fields in every config: any field ``R5Config`` has, so a
    further named option of the env needs no change here."""
    from designlab.stage2 import r5env

    return [c.describe() for c in r5env.env_configs(mode, body, set_name, N_ENVS, dict(overrides or {}))]


def _env_specs(mode: str, cfgs: list) -> list:
    """The pool's specs: an importable factory name and plain keyword arguments, never a closure."""
    return [("designlab.stage2.r5env:make", {"mode": mode, "cfg": dict(c)}) for c in cfgs]


def _g8a_short(body: str, set_name: str, tiny: bool) -> dict:
    """G8a's short form before a new run (rule Z1; design 3.6): the env with the zero action against the bar's
    R0 row on ``walk/v0.4``, seed 3042, at 0 and 10 ms (``g8a.short``; a tiny run cuts it to 2 s).  On the Stage 1
    view: ``g8a.stock`` on the stock ``walk/brisk``, same seed and delays.  ``passes``: every episode identical
    over every step the row ran."""
    from designlab.stage2 import g8a

    cut = float(TINY["g8a_seconds"]) if tiny else None
    if set_name == S1_SET:
        eps = [g8a.stock(body, "walk/brisk", g8a.SEEDS[0], k, seconds=cut) for k in g8a.DELAYS]
        g8a.release()
        return dict(form="stock", rule=g8a.RULE, body=body, set=set_name, cut_s=cut, episodes=eps,
                    identical=sum(1 for e in eps if e["identical"]), of=len(eps),
                    passes=bool(eps) and all(e["identical"] and e["compared_steps"] == e["reference_steps"]
                                             for e in eps))
    doc = g8a.short(body, set_name) if cut is None else g8a.bar_form(body, set_name, (g8a.SHORT_CONDITION,),
                                                                    g8a.SEEDS[:1], g8a.DELAYS, cut_s=cut)
    return dict(form="bar", **doc)


def _monitor_cmd() -> list:
    """The monitor child's command line before its own arguments."""
    return [sys.executable, "-m", "designlab.stage2.train2", "monitor"]


def base_body(body: str) -> str:
    """The body a head belongs to (``75-M-H-10-X`` -> ``75-M-H-10``): rule O3's group."""
    return body[:-2] if body.endswith("-X") else body


def _validation(body: str, set_name: str) -> dict:
    """What the validation records give a run that was not told: ``{"lr_probe": {path, sha256, rate, heads} |
    None, "filter": {path, sha256, filter_hz} | None}`` (``lr_probe.json``'s rate of the body, rule O3;
    ``filter.json``'s choice, rule O4)."""
    out = dict(lr_probe=None, filter=None)
    p = VALIDATION / "lr_probe.json"
    if p.is_file():
        row = (s2io.read_json(p).get("rates_by_body") or {}).get(base_body(body))
        if row and row.get("rate") is not None:
            out["lr_probe"] = dict(path=s2io.portable(str(p)), sha256=s2io.sha256_file(p), rate=float(row["rate"]),
                                   heads=row.get("heads"))
    p = VALIDATION / "filter.json"
    if p.is_file():
        hz = (s2io.read_json(p).get("choice") or {}).get("filter_hz")
        if hz is not None:
            out["filter"] = dict(path=s2io.portable(str(p)), sha256=s2io.sha256_file(p), filter_hz=float(hz))
    return out


def _resume_view(desc: dict) -> dict:
    """What a resumed run compares of an env's description (the lead's rulings of 16 Nov 2026 on rule K4;
    ``r5env.RESUME_KEYS``): the freeze the config names (``set_freeze_started``), the task names, the resolved
    task table as built (the weights in force are run state, rule K5, and are no part of it) and the input files
    its tasks use, with the plant's inputs; not the whole description, which an accepted later freeze changes.
    The config itself is not read from the description: a resumed run builds its envs from its stored configs."""
    return s2io.plain(dict(task_names=list(desc["task_names"]), tasks=desc["tasks"], task_files=desc["task_files"],
                           set_freeze_started=desc.get("set_freeze_started"),
                           plant_inputs=(desc.get("inputs") or {}).get("plant_inputs")))


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------
def job_name(mode: str, body: str, seed: int) -> str:
    return "%s-%s-s%d" % (mode, body, int(seed))


def vcpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _rss_mb(pid: int | None = None) -> float | None:
    try:
        with open(f"/proc/{'self' if pid is None else int(pid)}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def is_boundary(n: int, every: int, rollout: int) -> bool:
    """Whether the rollout start with ``n`` steps done is the first at or past a multiple of ``every`` (0 too)."""
    return n == 0 or n // every != (n - rollout) // every


def boundary_seed(seed: int, b: int) -> int:
    return SEED_BASE * int(seed) + SEED_STEP * int(b)


def describe_schedule(s) -> dict:
    """What a session records of a model's schedule, after the check that it is no function (S2-14: a function or a
    closure would be pickled into the archive by value): ``schedules.describe`` for this study's classes; for
    another module-level class (SB3 wraps the float clip range in its own ``ConstantSchedule``) its name and its
    plain attributes."""
    inner, seen = s, 0
    while hasattr(inner, "value_schedule") and seen < 8:
        inner, seen = inner.value_schedule, seen + 1
    if inspect.isfunction(inner) or inspect.ismethod(inner) or type(inner).__module__ == "__main__":
        raise TypeError(f"{inner!r} is a function: a schedule is an instance of a module-level class (S2-14)")
    try:
        return schedules.describe(inner)
    except TypeError:
        out = {"class": f"{type(inner).__module__}.{type(inner).__name__}"}
        out.update({("value" if k == "val" else k): v for k, v in vars(inner).items()
                    if isinstance(v, (int, float, str, bool))})
        return out


def count_optimizer_steps(model) -> list:
    """Wrap ``model.policy.optimizer.step`` so that every call is counted; the counter (a one-entry list).  SB3
    advances its own update count as if every minibatch step had run (lesson 16)."""
    counter = [0]
    opt = model.policy.optimizer
    inner = opt.step

    def step(*args, **kwargs):
        counter[0] += 1
        return inner(*args, **kwargs)

    opt.step = step
    return counter


def minibatch_steps_possible(model) -> int:
    return int(model.n_epochs) * int(math.ceil(model.n_steps * model.n_envs / model.batch_size))


def read_updates(path) -> list:
    """The whole lines of an ``updates.jsonl`` (a line cut by a kill is left out)."""
    out = []
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "steps" in row:
            out.append(row)
    return out


def _cut_updates(path: Path, n: int, session: int) -> int:
    """Keep the lines with ``steps <= n``; the others (and any cut line) go to ``updates.discarded.s<k>.jsonl``.
    The number of lines moved."""
    if not path.is_file():
        return 0
    keep, drop = [], []
    for line in path.read_text().splitlines():
        try:
            ok = int(json.loads(line)["steps"]) <= n
        except (ValueError, KeyError, TypeError):
            ok = False
        (keep if ok else drop).append(line)
    if drop:
        with open(path.with_name("updates.discarded.s%d.jsonl" % session), "a", encoding="utf-8") as fh:
            fh.write("\n".join(drop) + "\n")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(x + "\n" for x in keep))
        os.replace(tmp, path)
    return len(drop)


def kl_share(updates: list, lo: int, hi: int) -> dict:
    """``KL_CUT_RULE`` on the updates with ``lo < steps <= hi``."""
    shares = [u["minibatch_steps_run"] / u["minibatch_steps_possible"] for u in updates
              if lo < int(u["steps"]) <= hi and u.get("minibatch_steps_possible")]
    med = float(np.median(shares)) if shares else None
    return dict(after_steps=int(lo), to_steps=int(hi), updates=len(shares), median_share=med,
                cut=bool(med is not None and med < 0.5))


def _parent_of(parent) -> tuple | None:
    """``(job directory, steps)`` of ``--from-checkpoint JOBDIR:STEPS`` (a string, a pair or a dict)."""
    if parent is None:
        return None
    if isinstance(parent, dict):
        return Path(parent["job_dir"]), int(parent["steps"])
    if isinstance(parent, str):
        d, sep, n = parent.rpartition(":")
        if not sep or not n.isdigit():
            raise Refused(f"--from-checkpoint {parent!r}: JOBDIR:STEPS")
        return Path(d), int(n)
    d, n = parent
    return Path(d), int(n)


# ------------------------------------------------------------------------------------------------
# A run
# ------------------------------------------------------------------------------------------------
def train(mode, body, seed, set_name="V6", out=None, job=None, steps=None, resume=False, workers=None,
          checkpoint_every=CHECKPOINT_EVERY, monitor_every=MONITOR_EVERY, monitor_also_k=(), lr=None, config=None,
          ppo=None, gates=True, allow_version_change=False, allow_source_change=False, parent=None, tiny=False,
          run_id=None, log=print, retry_check=False) -> dict:
    """One training job (module docstring); its ``run.json``.

    ``out``: the root (``<out>/jobs/<job>``); ``job``: the job's name (default ``<mode>-<body>-s<seed>``; a
    fine-tune names its own); ``steps``: the budget asked (rounded up to whole rollouts; required unless ``tiny`` or
    ``resume``); ``workers``: W (default one per vCPU, at most 8; 0: the envs in this process); ``lr``: the starting
    rate (default: the body's rate of ``lr_probe.json``); ``config``: ``R5Config`` fields for every env; ``ppo``:
    PPO constructor arguments over design 2's (smoke checks; both are settings); ``gates`` False: ``GATES_RULE``;
    ``parent``: ``(job directory, steps)`` of a fine-tune's parent (rule K8); ``tiny``: design 3.6's tiny run;
    ``retry_check``: on a resume of a run that halted after two attempts at one check were never closed, let that
    check run once more (``RETRY_RULE``; recorded in the session).

    Raises ``Refused`` (a resume the rules refuse), ``Halt`` (rule M1: the lead) or ``TrainError``; a job that a
    rule stopped returns its ``run.json`` like one that finished (``status``)."""
    return _Run(dict(mode=mode, body=body, seed=seed, set_name=set_name, out=out, job=job, steps=steps, resume=resume,
                     workers=workers, checkpoint_every=checkpoint_every, monitor_every=monitor_every,
                     monitor_also_k=monitor_also_k, lr=lr, config=config, ppo=ppo, gates=gates,
                     allow_version_change=allow_version_change, allow_source_change=allow_source_change,
                     parent=parent, tiny=tiny, run_id=run_id, log=log, retry_check=retry_check)).go()


class _Run:
    """One call of ``train``: the arguments, the job's files and the session's state."""

    def __init__(self, a: dict):
        if a["mode"] not in MODES:
            raise Refused(f"mode {a['mode']!r}: choose from {', '.join(MODES)}")
        if a["out"] is None:
            raise Refused("out: the root directory of the job (<out>/jobs/<job>)")
        self.mode, self.body, self.seed, self.set = a["mode"], str(a["body"]), int(a["seed"]), str(a["set_name"])
        self.tiny, self.gates, self.resume = bool(a["tiny"]), bool(a["gates"]), bool(a["resume"])
        if self.set == S1_SET and not self.tiny:
            raise Refused("set S1 is the Stage 1 view (development): the trainer takes it only with --tiny")
        self.parent = _parent_of(a["parent"])
        if self.parent is not None and not a["job"]:
            raise Refused("a fine-tune names its own job (--job NAME): it is a new job directory (rule K8)")
        self.job = str(a["job"]) if a["job"] else job_name(self.mode, self.body, self.seed)
        self.run_id = a["run_id"]
        self.root = Path(a["out"])
        self.jd = self.root / "jobs" / self.job
        self.path = self.jd / "run.json"
        self.config, self.ppo = dict(a["config"] or {}), dict(a["ppo"] or {})
        self.lr_asked = None if a["lr"] is None else float(a["lr"])
        self.steps_asked = None if a["steps"] is None else int(a["steps"])
        every_ck, every_mon = int(a["checkpoint_every"]), int(a["monitor_every"])
        if self.tiny:  # (design 3.6: the tiny run's intervals stand in for the defaults)
            every_ck = TINY["checkpoint_every"] if every_ck == CHECKPOINT_EVERY else every_ck
            every_mon = TINY["monitor_every"] if every_mon == MONITOR_EVERY else every_mon
        self.every_ck, self.every_mon = every_ck, every_mon
        self.also_k = tuple(int(k) for k in a["monitor_also_k"] or ())
        self.n_steps = int(self.ppo.get("n_steps", TINY["n_steps"] if self.tiny else N_STEPS))
        self.rollout = N_ENVS * self.n_steps
        if every_ck <= 0 or every_mon <= 0 or every_mon % every_ck:
            raise Refused(f"monitor_every {every_mon} must be a positive multiple of checkpoint_every {every_ck}: a "
                          "check runs on the checkpoint of its step")
        w = a["workers"]
        self.workers = min(N_ENVS, vcpus()) if w is None else max(0, min(int(w), N_ENVS))
        self.allow_version, self.allow_source = bool(a["allow_version_change"]), bool(a["allow_source_change"])
        self.retry_check, self.retry_used = bool(a.get("retry_check")), False
        self.log = a["log"] or (lambda *x: None)
        self.doc = self.session = self.venv = self.vecnorm = self.model = self.cb = None
        self.k, self.skipped, self.freeze_index, self.val, self.env_described = 0, [], None, {}, {}

    # ---------------------------------------------------------------- records
    def write(self) -> None:
        s2io.write_json(self.path, self.doc)

    def say(self, text: str) -> None:
        self.log(f"{self.job}: {text}")

    def _asked(self) -> dict:
        """The settings that are the caller's arguments (``lr`` and ``budget_asked`` None: not given)."""
        return dict(mode=self.mode, body=self.body, seed=self.seed, set=self.set, n_envs=N_ENVS, tiny=self.tiny,
                    gates=self.gates, checkpoint_every=self.every_ck, monitor_every=self.every_mon,
                    monitor_k=_monitor2().K, monitor_also_k=list(self.also_k), config=s2io.plain(self.config),
                    ppo_overrides=s2io.plain(self.ppo), n_steps=self.n_steps,
                    parent=None if self.parent is None else dict(job=self.parent[0].name, steps=self.parent[1]))

    def _refuse_settings(self, doc: dict) -> None:
        """``RESUME_RULE``: every argument given must be the run's."""
        was, now = doc["settings"], self._asked()
        diff = []
        for key, v in now.items():
            w = was.get(key)
            if key == "parent":
                w = None if w is None else dict(job=w.get("job"), steps=w.get("steps"))
            if w != v:
                diff.append(f"{key}: {w!r} -> {v!r}")
        if self.steps_asked is not None and was.get("budget") is not None:
            budget = -(-self.steps_asked // self.rollout) * self.rollout
            if budget != was["budget"]:
                diff.append(f"budget: {was['budget']} -> {budget}")
        if self.lr_asked is not None and was.get("lr") is not None and was["lr"]["start"] != self.lr_asked:
            diff.append(f"lr: {was['lr']['start']!r} -> {self.lr_asked!r}")
        if diff:
            raise Refused(f"{self.path}: other settings than the run's ({'; '.join(diff)}): refusing to resume "
                          "(rule K4: no flag)")

    # ---------------------------------------------------------------- the whole job
    def go(self) -> dict:
        from designlab.stage2 import s2load

        doc = s2io.read_json(self.path) if self.path.is_file() else None
        if doc is not None and not self.resume:
            raise Refused(f"{self.path} exists: pass --resume to continue it")
        if doc is None and self.resume:
            raise Refused(f"{self.path}: nothing to resume (no run.json)")
        if doc is not None:
            if doc.get("stopped"):
                raise Refused(f"{self.path}: the run was stopped ({doc['stopped'].get('reason')}): it is not resumed")
            self._refuse_settings(doc)
            if doc.get("settings_sha256") is not None and s2io.sha256_json(doc["settings"]) != doc["settings_sha256"]:
                raise Refused(f"{self.path}: its settings do not hash to its settings_sha256 (the file was changed)")
            if doc.get("status") == "done":
                self.say("the run is done; nothing to do")
                return doc
        elif self.steps_asked is None and not self.tiny:
            raise Refused("--steps: a new run names its budget (rule Y11 fixes it after the smoke)")
        import torch

        torch.set_num_threads(1)
        src, vers = sources(), s2load.versions()
        new = doc is None
        self.doc = doc = self._new_doc() if new else doc
        complete = doc["settings_sha256"] is not None  # (else: never reached its first rollout)
        loaded = None
        if not complete:  # (a start that cannot be made is refused before anything is written)
            self.val = _validation(self.body, self.set)
            self._rate(self.val, None, dry=True)
        try:
            # A resume refuses in RESUME_RULE's order, and nothing is written before the last refusal: the envs
            # are built from the run's stored configs (other inputs), then the sources, the versions, the
            # checkpoint's run and its fingerprint.
            if complete:
                self._open()
            flagged = self._refuse_sources_versions(doc, src, vers)
            if complete:
                loaded = self._load()
            self._begin(src, vers, flagged)
            if not complete:
                if self.gates and not (doc["gates"].get("g8a") or {}).get("passes") and not self._g8a():
                    doc["status"] = "stopped_g8a"
                    return doc
                self._open()
            self._model(loaded, new)
            if self.gates:
                self._baseline()
            self._learn()
            return self._finish()
        except BaseException as exc:
            if self.session is not None:
                self.session["error"] = f"{type(exc).__name__}: {exc}"
                if doc["status"] == "running":
                    doc["status"] = "partial"
                if isinstance(exc, Halt):  # (one of rule M5's five endings, until the lead resumes the run)
                    doc["halt"] = dict(session=self.k, reason=str(exc), at=s2io.now_utc())
            raise
        finally:
            if self.session is not None:
                self.session["ended_at"] = s2io.now_utc()
                if self.model is not None:
                    doc["num_timesteps"] = int(self.model.num_timesteps)
                self.write()
                self._progress(int(doc.get("num_timesteps") or 0))
            if self.venv is not None:
                self.venv.close()

    def _new_doc(self) -> dict:
        m2 = _monitor2()
        asked = self._asked()
        asked.update(budget=None, budget_asked=self.steps_asked)
        return dict(
            schema=SCHEMA, run_uuid=uuid.uuid4().hex, run_id=self.run_id, job=self.job, mode=self.mode,
            body=self.body, head="X" if self.body.endswith("-X") else "current", seed=self.seed, set=self.set,
            settings=asked, settings_sha256=None,
            rules=dict(boundary=BOUNDARY_RULE, checkpoint=CHECKPOINT_RULE, resume=RESUME_RULE, updates=UPDATES_RULE,
                       kl_cut=KL_CUT_RULE, rate=RATE_RULE, gates=GATES_RULE, fine_tune=FINE_TUNE_RULE,
                       retry=RETRY_RULE, monitor=dict(m2.RULES)),
            layout=None, sessions=[], checkpoints=[], kept=[],
            gates=dict(g8a=None, rate=None, g8b=None, g8b_after_fix=None), monitor=[], baseline=None,
            baseline_attempts=[], fix=None, flags=dict(kl_cut=False), no_gain_count=0, stopped=None, parent=None,
            final_check=None, halt=None, updates_file="updates.jsonl", status="running", budget=None,
            budget_asked=self.steps_asked, num_timesteps=0)

    def _refuse_sources_versions(self, doc: dict, src: dict, vers: dict) -> dict:
        """``RESUME_RULE``: the sources and the library versions against the last session's; what the new session
        records of an allowed change."""
        from designlab.stage2 import s2load

        out = {}
        if not doc["sessions"]:
            return out
        last = doc["sessions"][-1]
        was = (last.get("sources") or {}).get("sha256")
        if was != src["sha256"]:
            if not self.allow_source:
                files = (last.get("sources") or {}).get("files") or {}
                changed = sorted(f for f in src["files"] if files.get(f) != src["files"][f])
                raise Refused(f"{self.path}: other sources than the run's last session ({str(was)[:12]}... -> "
                              f"{src['sha256'][:12]}...; files: {', '.join(changed) or 'the Stage 1 or panel sources'}"
                              "): refusing to resume (pass --allow-source-change to resume on them, flagged)")
            out.update(source_change=True, sources_before=last.get("sources"))
        a, b = s2load.comparable(last.get("versions") or {}), s2load.comparable(vers)
        if a != b:
            if not self.allow_version:
                raise Refused(f"{self.path}: other library versions than the run's last session ({a} -> {b}): "
                              "refusing to resume (pass --allow-version-change to resume on them, flagged)")
            out.update(version_change=True, versions_before=last.get("versions"))
        return out

    def _begin(self, src: dict, vers: dict, flagged: dict) -> None:
        """The session's record, appended and written: from here on the job's files change."""
        import torch

        doc = self.doc
        self.k = len(doc["sessions"])
        self.session = session = dict(
            index=self.k, started_at=s2io.now_utc(), ended_at=None, pid=os.getpid(), host=socket.gethostname(),
            **s2io.git_state(own_outputs=[str(self.root.resolve())]), versions=vers,
            sources=dict(sha256=src["sha256"], files=src["files"], plant_sources=src["plant_sources"],
                         panel_sources=src["panel_sources"]),
            workers=self.workers, vcpus=vcpus(), torch_threads=int(torch.get_num_threads()),
            pythonhashseed=os.environ.get("PYTHONHASHSEED"), resumed_from=None, restarted_from_zero=False,
            skipped_checkpoints=list(self.skipped), fingerprint_max_abs_diff=None, version_change=False,
            source_change=False, freeze_index=self.freeze_index, boundary_seeds=[], discarded_after_resume=None,
            retry_check=self.retry_check, error=None, env=None, other_envs=None)
        session.update(flagged)
        session.update(self.env_described)
        doc["sessions"].append(session)
        doc["status"], doc["halt"] = "running", None
        for key in ("version_change", "source_change"):
            if session[key]:
                doc["flags"][key] = True
        self.write()

    # ---------------------------------------------------------------- G8a
    def _g8a(self) -> bool:
        self.say(f"G8a's short form on set {self.set}")
        res = _g8a_short(self.body, self.set, self.tiny)
        rec = s2io.plain(dict(schema="s2-run-g8a-1", job=self.job, run_uuid=self.doc["run_uuid"], tiny=self.tiny,
                              **res, session=self.k, at=s2io.now_utc()))
        rec["sha256"] = s2io.sha256_json(rec)
        s2io.write_json(self.jd / "gates" / "g8a.json", rec)
        self.doc["gates"]["g8a"] = dict(record="gates/g8a.json", sha256=rec["sha256"], passes=bool(res["passes"]),
                                        identical=res.get("identical"), of=res.get("of"), form=res.get("form"))
        if not res["passes"]:
            self.doc["stopped"] = dict(at_steps=0, status="stopped_g8a", at=s2io.now_utc(),
                                       reason="G8a fails: no run (the zero action is not the bar's R0 row: "
                                              f"{res.get('identical')} of {res.get('of')} episodes identical)")
            self.say("G8a fails: no run")
        self.write()
        return bool(res["passes"])

    # ---------------------------------------------------------------- the envs, the settings, the model
    def _pool(self, cfgs: list):
        from designlab.stage2 import vecpool

        specs = _env_specs(self.mode, cfgs)
        w = min(self.workers, len(specs))  # (never more workers than envs)
        self.say(f"building {len(specs)} envs in {w} worker{'' if w == 1 else 's'}")
        try:
            return vecpool.PoolVecEnv(specs, w, worker_cmd=[sys.executable, *WORKER_CMD])
        except Exception as exc:  # noqa: BLE001 - an env that refuses its inputs (rule K4) is a refused resume
            if self.resume and type(exc).__name__ == "InputError":
                raise Refused(f"{self.path}: other inputs than the run's: {exc} (rule K4: no flag)") from exc
            raise

    def _open(self) -> None:
        """The envs.  A run without settings yet: built from today's configs, and the settings are completed and
        hashed.  A run with settings: built from its stored configs (which name the freeze it started on and its
        tasks), and what the envs then describe is compared with the run's (``RESUME_RULE``: other inputs)."""
        doc = self.doc
        complete = doc["settings_sha256"] is not None
        val = (doc["settings"].get("validation") if complete else self.val) or {}
        overrides = dict(self.config)
        if val.get("filter") and "filter_hz" not in overrides:  # (rule O4's choice, unless the caller names one)
            overrides["filter_hz"] = val["filter"]["filter_hz"]
        today = _env_configs(self.mode, self.body, self.set, overrides)
        cfgs, parent = today, None
        if complete:
            cfgs = doc["settings"]["env_configs"]
            pinned = ("set_freeze", "tasks")  # (what the stored configs pin: the run's freeze and its task names)
            for i, (c, t) in enumerate(zip(cfgs, today, strict=True)):
                diff = [f"{k}: {c.get(k)!r} -> {t.get(k)!r}" for k in sorted(set(c) | set(t))
                        if k not in pinned and c.get(k) != t.get(k)]
                if diff:
                    raise Refused(f"{self.path}: other settings than the run's: env config {i} as built today "
                                  f"differs ({'; '.join(diff)}): refusing to resume (rule K4: no flag)")
        elif self.parent is not None:
            parent = self._parent_record()  # (before the pool is built: rule K8's probe builds its own)
        self.venv = venv = self._pool(cfgs)
        descs = venv.env_method("describe")
        built = [i for i, (d, c) in enumerate(zip(descs, cfgs, strict=True)) if s2io.plain(d["config"]) != c]
        if built:  # (rule K4: of the description of the envs as built, ``config`` is the stored config)
            raise TrainError(f"the config that env {built[0]} describes is not the config it was built from")
        views = [_resume_view(d) for d in descs]
        if any(v != views[0] for v in views[1:]):
            raise TrainError("the envs of the job do not agree on their tasks, input files or freeze")
        view, layout = views[0], s2io.plain(descs[0]["layout"])
        self.freeze_index = s2io.plain(dict(started=view["set_freeze_started"], now=descs[0].get("set_freeze_now")))
        # what the session's envs execute, as they describe it (the rule texts, the reward's weights and scales,
        # R6's span, every file read): env 0's description whole, the others' entries that differ from it
        first = s2io.plain(descs[0])
        self.env_described = dict(env=first, other_envs=[{k: v for k, v in s2io.plain(d).items() if first.get(k) != v}
                                                         for d in descs[1:]])
        if self.session is not None:
            self.session.update(freeze_index=self.freeze_index, **self.env_described)
        self.weights0 = {t: float(view["tasks"][t]["weight"]) for t in view["task_names"]}
        if complete:
            s = doc["settings"]
            was = {"the named freeze": s["set_freeze"], "the task names": s["task_names"],
                   "the resolved task table": s["tasks"], "the input files": s["inputs"]["files"],
                   "the plant's inputs": s["inputs"]["plant_inputs"], "the layout": s["layout"]}
            now = {"the named freeze": view["set_freeze_started"], "the task names": view["task_names"],
                   "the resolved task table": view["tasks"], "the input files": view["task_files"],
                   "the plant's inputs": view["plant_inputs"], "the layout": layout}
            diff = [k for k in was if was[k] != now[k]]
            if diff:
                files = [f for f in sorted(set(s["inputs"]["files"]) | set(view["task_files"]))
                         if s["inputs"]["files"].get(f) != view["task_files"].get(f)]
                raise Refused(f"{self.path}: other inputs than the run's ({', '.join(diff)}"
                              + (f": {', '.join(files)}" if files else "") + "): refusing to resume (rule K4: no "
                              "flag)")
        else:
            self._settings(cfgs, view, layout, val, parent)
        s = doc["settings"]
        self.layout, self.budget = layout, int(s["budget"])
        self.lr = (float(s["lr"]["start"]), float(s["lr"]["end"]))
        self.rules = _monitor2().rules(self.every_mon, self.tiny, self.set)
        self.expect = dict(run_uuid=doc["run_uuid"], settings_sha256=doc["settings_sha256"])

    def _load(self):
        """The newest intact checkpoint, loaded on the pool with ``force_reset=False`` (None: there is none)."""
        from designlab.stage2 import s2load

        n, self.skipped = s2load.newest_intact(self.jd)
        if n is None:
            return None
        try:
            ck = s2load.load_checkpoint(self.jd, n, env=self.venv, lr=self.lr, expect_run=self.expect)
        except s2load.CheckpointError as exc:
            raise Refused(f"{self.path}: {exc}: refusing to resume") from exc
        return int(n), ck

    def _model(self, loaded, new: bool) -> None:
        session = self.session
        for row in session["skipped_checkpoints"]:
            self.say(f"WARNING: skipping checkpoint {row['steps']}: {row['reason']}")
        if loaded is None:
            session["restarted_from_zero"] = bool(not new)
            if not new:
                self.say(f"restarting {self.job} from step 0")
                self._rollback(None, None)
            self.vecnorm, self.model = self._fresh()
        else:
            n, ck = loaded
            self.vecnorm, self.model = ck["vecnorm"], ck["model"]
            session.update(resumed_from=n, fingerprint_max_abs_diff=ck["fingerprint_max_abs_diff"])
            self._rollback(n, ck["record"])
            self.say(f"resuming session {self.k}: {self.job} from {n} steps")
        if int(self.model.n_steps) != self.n_steps or int(self.model.n_envs) != N_ENVS:
            raise TrainError(f"the model's rollout is {self.model.n_envs} x {self.model.n_steps}, the job's "
                             f"{N_ENVS} x {self.n_steps}")
        self.counter = count_optimizer_steps(self.model)
        session["schedules"] = dict(lr=describe_schedule(self.model.lr_schedule),
                                    clip_range=describe_schedule(self.model.clip_range))
        assert not inspect.isfunction(self.model.learning_rate)  # (S2-14, before the first save)
        if self.weights != self.weights0:  # (a fix in force: the checkpoint's run state)
            self._set_weights(self.weights)
        self.write()

    def _settings(self, cfgs: list, view: dict, layout: dict, val: dict, parent: dict | None) -> None:
        """Complete and hash the settings of a run that has none yet (a new run, or one that was cut before its
        envs were built)."""
        from designlab.stage1 import policy

        doc = self.doc
        if any(list(c.get("tasks") or ()) != list(view["task_names"]) for c in cfgs):
            raise TrainError("an env config does not name the tasks its env resolved: a run stores configs that name "
                             "their tasks, so that a later freeze of the set cannot change them (rule K4)")
        steps = self.steps_asked or doc["settings"].get("budget_asked") or (TINY["steps"] if self.tiny else None)
        if steps is None:
            raise Refused("--steps: the run has no budget yet (it was cut before its first rollout)")
        budget = -(-int(steps) // self.rollout) * self.rollout
        lr, source = self._rate(val, parent)
        end = lr if parent is not None else LR_END_RATIO * lr
        d = policy.DESIGN2
        ppo = dict(n_steps=self.n_steps, batch_size=d["batch_size"], n_epochs=d["n_epochs"], gamma=d["gamma"],
                   gae_lambda=d["gae_lambda"], clip_range=d["clip_range"], target_kl=d["target_kl"],
                   net_arch=list(d["net_arch"]), clip_obs=d["clip_obs"], clip_reward=d["clip_reward"])
        ppo.update({k: v for k, v in self.ppo.items() if k in ppo})
        s = dict(doc["settings"])
        s.update(set_freeze=view["set_freeze_started"],
                 inputs=dict(files=view["task_files"], plant_inputs=view["plant_inputs"]), tasks=view["tasks"],
                 task_names=view["task_names"], ppo=ppo,
                 lr=dict(start=lr, end=end, source=source,
                         schedule="constant" if parent is not None else "linear over the budget"),
                 log_std_init="the parent's" if parent is not None else math.log(SIGMA), budget=budget,
                 budget_asked=int(steps), env_configs=cfgs, layout=layout, validation=val,
                 parent=None if parent is None else dict(job=parent["job"], run_uuid=parent["run_uuid"],
                                                         steps=parent["steps"], files=parent["files"]))
        doc["settings"] = s2io.plain(s)
        doc["settings_sha256"] = s2io.sha256_json(doc["settings"])
        doc.update(layout=layout, budget=budget, budget_asked=int(steps), parent=parent)
        self.write()

    def _rate(self, val: dict, parent: dict | None, dry: bool = False) -> tuple:
        """The run's starting rate and where it comes from (``dry``: before a fine-tune's probe has run)."""
        if self.lr_asked is not None:
            return self.lr_asked, "given (--lr)"
        if dry and self.parent is not None:
            return None, "rule K8's probe from the parent's checkpoint"
        if parent is not None:
            r = (parent.get("lr_probe") or {}).get("rate")
            if r is None:
                raise Refused(f"{self.job}: no rate runs every minibatch step from the parent's checkpoint (rule "
                              "K8): the fine-tune does not run; it goes to the lead")
            return float(r), "rule K8's probe from the parent's checkpoint"
        if val.get("lr_probe"):
            return float(val["lr_probe"]["rate"]), "lr_probe.json (rule O3)"
        if self.tiny:
            return float(TINY["lr"]), "the tiny run's default (no probe record)"
        raise Refused(f"{self.job}: no learning rate: pass --lr or run `train2 probe-lr` first (rule O3: "
                      f"{s2io.portable(str(VALIDATION / 'lr_probe.json'))} holds no rate for {base_body(self.body)})")

    def _parent_record(self) -> dict:
        """Rule K8: the parent's checkpoint (read here; loaded by ``_fresh``) and, unless ``--lr`` was given, the
        probe of the rate from it, run once and kept."""
        from designlab.stage2 import s2load

        pdir, psteps = self.parent
        files = s2load.checkpoint_files(pdir, psteps)
        if not files["record"].is_file():
            raise Refused(f"{files['record']}: the parent's checkpoint has no record")
        rec = s2io.read_json(files["record"])
        pdoc = s2io.read_json(pdir / "run.json") if (pdir / "run.json").is_file() else {}
        out = dict(job=pdir.name, run_uuid=rec.get("run_uuid"), steps=int(psteps),
                   files={k: rec["files"][k]["sha256"] for k in s2load.FILE_KINDS}, layout=rec.get("layout"),
                   settings_sha256=rec.get("settings_sha256"), lr_schedule=(pdoc.get("settings") or {}).get("lr"),
                   budget=pdoc.get("budget"), fingerprint_max_abs_diff=None, lr_probe=None)
        if self.lr_asked is None:
            self.say(f"rule K8's probe from the parent's checkpoint {pdir.name}:{psteps}")
            probe = probe_lr([self.body], self.set, self.jd / "gates" / "parent_lr_probe.json",
                             workers=self.workers, parent=(pdir, psteps))
            out["lr_probe"] = dict(record="gates/parent_lr_probe.json", sha256=probe["sha256"],
                                   rate=probe["bodies"][self.body]["passed"], p=probe["parent"]["p"])
        return out

    def _fresh(self) -> tuple:
        """A new run's ``VecNormalize`` and model (design 3.6's code block), or a fine-tune's from its parent."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        from designlab.stage1 import policy
        from designlab.stage2 import s2load

        self.weights = dict(self.weights0)
        d = policy.DESIGN2
        if self.parent is not None:
            pdir, psteps = self.parent
            try:
                ck = s2load.load_checkpoint(pdir, psteps, env=self.venv, lr=self.lr)
            except s2load.CheckpointError as exc:
                raise Refused(f"{self.job}: the parent's checkpoint is not usable: {exc}") from exc
            want = {k: self.layout[k] for k in ("obs_dim", "actor_dim", "action_dim")}
            have = {k: ck["record"]["layout"][k] for k in want}
            if want != have:
                raise Refused(f"{self.job}: the parent's layout {have} is not this job's {want} (rule K8: the same "
                              "layout)")
            model = ck["model"]
            model.num_timesteps, model._n_updates, model._episode_num = 0, 0, 0
            self.doc["parent"]["fingerprint_max_abs_diff"] = ck["fingerprint_max_abs_diff"]
            return ck["vecnorm"], model
        vecnorm = VecNormalize(self.venv, norm_obs=True, norm_reward=True, clip_obs=d["clip_obs"],
                               clip_reward=d["clip_reward"], gamma=d["gamma"])
        kw = policy.model_kwargs(int(self.layout["actor_dim"]), log_std_init=math.log(SIGMA))
        kw.update(learning_rate=schedules.LinearSchedule(*self.lr), n_steps=self.n_steps, clip_range=d["clip_range"])
        kw.update(self.ppo)
        return vecnorm, PPO(env=vecnorm, seed=self.seed, device="cpu", verbose=0, **kw)

    def _set_weights(self, weights: dict) -> None:
        try:
            self.venv.env_method("set_task_weights", dict(weights))
        except Exception as exc:  # noqa: BLE001 - some envs may hold the new weights and some not: the pool is spent
            raise TrainError(f"set_task_weights failed in an env ({type(exc).__name__}: {exc}): the envs no longer "
                             "agree on their task weights, the session ends") from exc

    # ---------------------------------------------------------------- the run state
    def run_state(self) -> dict:
        d = self.doc
        return s2io.plain(dict(task_weights=self.weights, fix=d["fix"], monitor=d["monitor"],
                               gates=dict(g8b=d["gates"]["g8b"], g8b_after_fix=d["gates"]["g8b_after_fix"]),
                               no_gain_count=d["no_gain_count"],
                               checks_done=[e["at_steps"] for e in d["monitor"] if e.get("decided")],
                               stopped=d["stopped"], flags=d["flags"]))

    def _rollback(self, n: int | None, record: dict | None) -> None:
        """``RESUME_RULE``: run.json's live fields from the checkpoint's run state (None: the state of step 0)."""
        from designlab.stage2 import s2load

        d, m2 = self.doc, _monitor2()
        st = (record or {}).get("run_state") or dict(task_weights=self.weights0, fix=None, monitor=[],
                                                     gates=dict(g8b=None, g8b_after_fix=None), no_gain_count=0,
                                                     flags=dict(kl_cut=False))
        at = -1 if n is None else n
        disc: dict = {}
        later = [e for e in d["monitor"] if e["at_steps"] > at]
        here = [e for e in d["monitor"] if e["at_steps"] == at]
        if later:
            disc["monitor"] = later
        kept = []
        if here and n is not None:
            e = here[0]
            if e.get("checkpoint_sha256") == m2.checkpoint_hashes(record):
                derived = {k: v for k, v in e.items() if k not in ("at_steps", "multiple", "checkpoint_sha256",
                                                                   "attempts")}
                if derived:
                    disc["monitor_at_resume"] = dict(at_steps=e["at_steps"], **derived)
                for a in e["attempts"]:  # (rule M1: never closed is an interruption)
                    if a.get("closed_at") is None:
                        a["interrupted"] = True
                kept = [dict(at_steps=e["at_steps"], multiple=e.get("multiple"),
                             checkpoint_sha256=e["checkpoint_sha256"], attempts=e["attempts"])]
            else:
                disc.setdefault("monitor", []).insert(0, e)
        d["monitor"] = copy.deepcopy(st.get("monitor") or []) + kept
        for key, was, now in (("fix", d["fix"], st.get("fix")),
                              ("g8b", d["gates"]["g8b"], (st.get("gates") or {}).get("g8b")),
                              ("g8b_after_fix", d["gates"]["g8b_after_fix"],
                               (st.get("gates") or {}).get("g8b_after_fix"))):
            if was != now:
                disc[key] = was
        d["fix"] = st.get("fix")
        d["gates"]["g8b"] = (st.get("gates") or {}).get("g8b")
        d["gates"]["g8b_after_fix"] = (st.get("gates") or {}).get("g8b_after_fix")
        d["no_gain_count"] = int(st.get("no_gain_count") or 0)
        sticky = {k: True for k in ("version_change", "source_change") if d["flags"].get(k)}
        d["flags"] = dict(st.get("flags") or dict(kl_cut=False), **sticky)
        self.weights = {str(t): float(w) for t, w in (st.get("task_weights") or self.weights0).items()}
        gone = [m for m in sorted(set(d["checkpoints"]) | set(s2load.list_checkpoints(self.jd))
                                  | {r["steps"] for r in self.session["skipped_checkpoints"]}) if m > at]
        for m in gone:
            s2load.remove_checkpoint(self.jd, m)
        if gone:
            disc["checkpoints"] = gone
        d["checkpoints"] = [m for m in d["checkpoints"] if m <= at]
        d["kept"] = s2load.list_checkpoints(self.jd)
        cut = _cut_updates(self.jd / "updates.jsonl", at, self.k)
        if cut:
            disc["updates"] = dict(lines=cut, file="updates.discarded.s%d.jsonl" % self.k)
        if d.get("num_timesteps", 0) > max(at, 0):
            disc["num_timesteps"] = d["num_timesteps"]
        d["num_timesteps"] = max(at, 0)
        self.session["discarded_after_resume"] = disc or None

    # ---------------------------------------------------------------- the monitor child
    def _child(self, args: list) -> int:
        """Run the monitor child and wait for it (a subprocess: nothing forks while the pool is open)."""
        cmd = _monitor_cmd() + [str(x) for x in args]
        cmd += ["--sources", self.session["sources"]["sha256"]]
        if self.tiny:
            cmd.append("--tiny")
        if self.also_k:
            cmd += ["--also-k", ",".join("%d" % k for k in self.also_k)]
        return subprocess.run(cmd, env=tbenv.child_env("disable"), cwd=tbenv.study_dir(),
                              stdin=subprocess.DEVNULL).returncode

    def _baseline(self) -> None:
        """R0L's episodes, once per run (rule M1): no check is read against a baseline that holds an error."""
        m2, d = _monitor2(), self.doc
        path = self.jd / "monitor" / "r0l.json"
        rec = s2io.read_json(path) if path.is_file() else None
        if d["baseline"] is not None and rec is not None and rec.get("sha256") != d["baseline"].get("sha256"):
            raise Refused(f"{path}: not the baseline record run.json names (its sha256 differs)")
        if rec is None or m2.baseline_problem(rec) is not None:
            self.say("the R0L baseline")
            att = dict(session=self.k, started_at=s2io.now_utc(), closed_at=None, exit_code=None, record=None)
            d["baseline_attempts"].append(att)
            self.write()
            code = self._child(["baseline", "--job", self.jd, "--body", self.body, "--set", self.set, "--k", m2.K])
            rec = s2io.read_json(path) if path.is_file() else None
            ok = rec is not None and m2.baseline_problem(rec) is None
            att.update(closed_at=s2io.now_utc(), exit_code=int(code), record=bool(ok))
            self.write()
            if not ok:
                raise Halt(f"{self.job}: the R0L baseline could not be made (the monitor child ended with code "
                           f"{code}; {m2.baseline_problem(rec)}): rule M1: the fault is repaired and all its "
                           "episodes are run again before the run's first rollout")
        self.baseline = rec
        d["baseline"] = dict(record="monitor/r0l.json", sha256=rec["sha256"], row=rec.get("row"),
                             **{k: rec.get(k) for k in m2.SUMMARY_KEYS})
        self.write()

    def _hashes(self, n: int) -> dict:
        from designlab.stage2 import s2load

        return _monitor2().checkpoint_hashes(s2io.read_json(s2load.checkpoint_files(self.jd, n)["record"]))

    def _history(self, upto: int) -> list:
        """The records of the run's checks up to ``upto``, in order (a check counted without a record: its
        stand-in)."""
        m2, out = _monitor2(), []
        for e in self.doc["monitor"]:
            if e["at_steps"] > upto:
                continue
            if e.get("counted_without_record"):
                out.append(m2.no_record(e["at_steps"], self.set, self.tiny, k=m2.K))
                continue
            path = self.jd / e["record"]
            rec = s2io.read_json(path) if path.is_file() else None
            if rec is None or rec.get("sha256") != e.get("record_sha256"):
                raise TrainError(f"{path}: the record of the check at {e['at_steps']} steps is missing or is not "
                                 "the one run.json names: the rules read records")
            out.append(rec)
        return out

    def check(self, n: int, training_follows: bool) -> None:
        """The check on checkpoint ``n`` (module docstring): the attempts, the record, ``monitor2.decide``."""
        m2, d = _monitor2(), self.doc
        hashes, pairs = self._hashes(n), m2.pairs_for(self.set, self.tiny)
        entry = next((e for e in d["monitor"] if e["at_steps"] == n), None)
        if entry is None:
            entry = dict(at_steps=n, multiple=(n // self.every_mon) * self.every_mon, checkpoint_sha256=hashes,
                         attempts=[])
            d["monitor"].append(entry)
        path = self.jd / "monitor" / ("monitor_%d.json" % n)

        def usable():
            doc = s2io.read_json(path) if path.is_file() else None
            return doc if m2.is_record(doc, hashes, pairs) else None

        rec = usable()
        closed = sum(1 for a in entry["attempts"] if a.get("closed_at") is not None and not a.get("record"))
        while rec is None and closed < m2.ATTEMPTS:
            after_halt = sum(1 for a in entry["attempts"] if a.get("closed_at") is None) >= m2.ATTEMPTS
            if after_halt and (not self.retry_check or self.retry_used):
                raise Halt(f"{self.job}: two attempts at the check of {n} steps were never closed (the session died "
                           "while the check ran, twice): the run halts and goes to the lead (rule M1); with the "
                           "fault repaired, a resume with --retry-check lets the check run once more")
            self.retry_used = self.retry_used or after_halt
            if path.is_file():  # (a record of other files: a checkpoint saved again after a resume)
                os.replace(path, path.with_name("monitor_%d.discarded.s%d.json" % (n, self.k)))
            self.say(f"check at {n} steps (attempt {len(entry['attempts']) + 1})")
            att = dict(session=self.k, started_at=s2io.now_utc(), checkpoint_sha256=hashes, closed_at=None,
                       exit_code=None, record=None, **(dict(after_halt=True) if after_halt else {}))
            entry["attempts"].append(att)
            self.write()  # (logged before the monitor starts)
            code = self._child(["run", "--job", self.jd, "--steps", n, "--body", self.body, "--set", self.set,
                                "--k", m2.K])
            rec = usable()
            att.update(closed_at=s2io.now_utc(), exit_code=int(code), record=rec is not None)
            self.write()
            if rec is None:
                closed += 1
                self.say(f"WARNING: the monitor left no record of the check at {n} steps (exit code {code})")
        entry.update(record=None if rec is None else "monitor/monitor_%d.json" % n,
                     record_sha256=None if rec is None else rec["sha256"], counted_without_record=rec is None)
        if rec is None:  # (rule M1: counted without a record, each of its episodes entered as an error)
            entry["episodes"] = [dict(condition=c, seed=s, outcome="error") for c, s in pairs]
        prev = max([e["at_steps"] for e in d["monitor"] if e["at_steps"] < n], default=0)
        kl = kl_share(read_updates(self.jd / "updates.jsonl"), prev, n)
        st = m2.decide(self._history(n), self.baseline, dict(task_weights=self.weights0), self.rules)
        mine = next(c for c in st["checks"] if c["at_steps"] == n)
        entry.update({k: v for k, v in mine.items() if k != "at_steps"}, kl_cut=kl, decided=True)
        d["gates"]["g8b"], d["gates"]["g8b_after_fix"] = st["gates"]["g8b"], st["gates"]["g8b_after_fix"]
        d["no_gain_count"] = st["no_gain_count"]
        d["flags"]["kl_cut"] = kl["cut"]
        d["flags"]["kl_cut_ever"] = bool(d["flags"].get("kl_cut_ever") or kl["cut"])
        if not training_follows:
            # The final checkpoint's check: the budget ended the run (rule M5's five endings), so a fix that the
            # reading asks for cannot act and a stop has nothing to stop; both are recorded as what they were.
            extra = {}
            if st["fix"] is not None and d["fix"] is None:
                extra["fix_not_applied"] = st["fix"]
            if st["stopped"] is not None:
                extra["would_have_stopped"] = st["stopped"]
            if extra:
                d["final_check"] = dict(at_steps=n, ended_by="the budget", **extra)
        else:
            if st["fix"] is not None and d["fix"] is None:
                fixed = [dict(c, task_weights=[st["fix"]["weights"][t] for t in c["tasks"]])
                         for c in d["settings"]["env_configs"]]
                d["fix"] = dict(st["fix"], monitor=entry["record"], monitor_sha256=entry["record_sha256"],
                                env_configs=fixed, applied="in place: env_method('set_task_weights') on every env; "
                                "the settings keep the env configs the run started with")
                self.say(f"G8b failed at {n} steps: one fix (rule M4): the weights of "
                         f"{', '.join(st['fix']['tasks']) or 'no task'} doubled")
            if st["task_weights"] != self.weights:
                self.weights = dict(st["task_weights"])
                self._set_weights(self.weights)
            if st["stopped"] is not None:
                d["stopped"] = dict(st["stopped"], at=s2io.now_utc())
                self.say(f"stopped at {n} steps: {st['stopped']['reason']}")
        g = mine["gain"]
        self.say(f"check at {n} steps: N {g['N']} (N* {g['N_star']}), S {g['S']:.3f} s (S* {g['S_star']:.3f} s): "
                 f"{'a gain' if g['gain'] else 'no gain'}" + ("; kl_cut" if kl["cut"] else ""))
        self.write()

    # ---------------------------------------------------------------- checkpoints, the boundary, progress
    def save(self, n: int) -> None:
        from designlab.stage2 import s2load

        d = self.doc
        s2load.save_checkpoint(self.jd, n, self.model, self.vecnorm, self.layout,
                               dict(run_uuid=d["run_uuid"], run_id=d["run_id"], job=self.job,
                                    settings_sha256=d["settings_sha256"]), self.run_state())
        have = s2load.list_checkpoints(self.jd)
        keep = s2load.keep(have, self.every_mon)
        for m in have:
            if m not in keep:
                s2load.remove_checkpoint(self.jd, m)
        d["checkpoints"] = sorted(set(d["checkpoints"]) | {int(n)})
        d["kept"] = keep
        d["num_timesteps"] = int(n)
        self.write()

    def boundary(self, n: int) -> None:
        """``BOUNDARY_RULE``."""
        b = n // self.every_ck
        s = boundary_seed(self.seed, b)
        model, vecnorm = self.model, self.vecnorm
        model.set_random_seed(s)
        obs = vecnorm.reset()  # (every env with its seed; the running returns zeroed)
        model._last_obs = obs
        model._last_episode_starts = np.ones((vecnorm.num_envs,), dtype=bool)
        model._last_original_obs = vecnorm.get_original_obs()
        self.session["boundary_seeds"].append(dict(b=int(b), steps=int(n), seed=int(s)))

    def _progress(self, n: int, rate: float | None = None, last: list | None = None) -> None:
        d = self.doc
        if self.cb is not None:
            rate, last = self.cb.rate if rate is None else rate, self.cb.last if last is None else last
        mon = next((e for e in reversed(d["monitor"]) if e.get("decided")), None)
        budget = d.get("budget")
        s2io.write_json(self.jd / "progress.json", dict(
            schema=PROGRESS_SCHEMA, job=self.job, run_uuid=d["run_uuid"], run_id=d["run_id"], session=self.k,
            pid=os.getpid(), status=d["status"], steps=int(n), budget=budget, updated_at=s2io.now_utc(),
            env_steps_per_s=rate,
            eta_h=None if not rate or budget is None else max(0, budget - n) / rate / 3600.0,
            workers=self.workers, last_checkpoint=d["kept"][-1] if d["kept"] else None,
            last_monitor=None if mon is None else dict(
                at_steps=mon["at_steps"], N=mon.get("N"), S=mon.get("S"),
                mean_clock_factor=mon.get("mean_clock_factor"), gain=(mon.get("gain") or {}).get("gain")),
            g8a=(d["gates"].get("g8a") or {}).get("passes"), g8b=(d["gates"].get("g8b") or {}).get("pass"),
            fix=None if d["fix"] is None else d["fix"]["at_steps"], flags=d["flags"], stopped=d["stopped"],
            minibatch_steps_last=last))

    # ---------------------------------------------------------------- learn
    def _learn(self) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        run, model, d = self, self.model, self.doc
        updates_path = self.jd / "updates.jsonl"
        possible = minibatch_steps_possible(model)
        n_envs = self.vecnorm.num_envs

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.first, self.pending, self.last = True, None, None
                self.n0 = self.t0 = self.c0 = None
                self.paused = 0.0  # (seconds spent at rollout starts since the clock started: RATE_RULE)
                self.t_roll = time.perf_counter()
                self.ep_len, self.ep_ret = np.zeros(n_envs, int), np.zeros(n_envs)
                self.done_len, self.done_ret = [], []
                self.rate = None

            def _cpu(self) -> tuple:
                """CPU seconds of the trainer and of the pool's workers (0 with the envs in this process)."""
                me, w = os.getpid(), 0.0
                try:
                    rows = dict(run.venv.env_method("process_cpu"))
                    w = float(sum(c for pid, c in rows.items() if pid != me))
                except Exception:  # noqa: BLE001 - an env without the method: the trainer's alone
                    pass
                return time.process_time(), w

            def _flush(self) -> None:
                """The line of the update that followed the last complete rollout (``UPDATES_RULE``)."""
                p = self.pending
                if p is None:
                    return
                import torch

                self.pending = None
                v = model.logger.name_to_value

                def num(key):
                    return None if key not in v else float(v[key])

                with torch.no_grad():
                    std = float(torch.exp(model.policy.log_std).mean())
                ran = int(run.counter[0] - p["count0"])
                row = dict(session=run.k, update=p["steps"] // run.rollout, steps=p["steps"],
                           minibatch_steps_run=ran, minibatch_steps_possible=possible,
                           approx_kl=num("train/approx_kl"), clip_fraction=num("train/clip_fraction"),
                           explained_variance=num("train/explained_variance"), value_loss=num("train/value_loss"),
                           std_mean=std, lr=num("train/learning_rate"), ep_len_mean=p["ep_len_mean"],
                           ep_rew_mean=p["ep_rew_mean"], wall_s=time.perf_counter() - p["t0"])
                with open(updates_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(s2io.plain(row)) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                self.last = [ran, possible]

            def _on_rollout_start(self) -> None:
                t_in = time.perf_counter()
                n = int(model.num_timesteps)
                self._flush()
                first, self.first = self.first, False
                saved = False
                if is_boundary(n, run.every_ck, run.rollout) and not (first and run.session["resumed_from"] == n):
                    run.save(n)
                    saved = True
                if run.gates and n > 0 and is_boundary(n, run.every_mon, run.rollout) and not any(
                        e["at_steps"] == n and e.get("decided") for e in d["monitor"]):
                    run.check(n, training_follows=True)
                    if d["stopped"] is not None:
                        raise _StopTraining
                if saved or first:
                    run.boundary(n)
                    self.ep_len[:], self.ep_ret[:] = 0, 0.0
                    run.write()
                if self.n0 is None:  # (the session's clock starts after its first boundary)
                    self.n0, self.t0, self.c0 = n, time.perf_counter(), self._cpu()
                run._progress(n, self.rate, self.last)
                self.t_roll = time.perf_counter()
                if self.t0 < t_in:
                    self.paused += self.t_roll - t_in
                self.done_len, self.done_ret = [], []

            def _on_step(self) -> bool:
                rew = run.vecnorm.get_original_reward()
                self.ep_len += 1
                self.ep_ret += rew
                for i in np.flatnonzero(self.locals["dones"]):
                    self.done_len.append(int(self.ep_len[i]))
                    self.done_ret.append(float(self.ep_ret[i]))
                    self.ep_len[i], self.ep_ret[i] = 0, 0.0
                return True

            def _on_rollout_end(self) -> None:
                n = int(model.num_timesteps)
                self.pending = dict(steps=n, count0=run.counter[0], t0=self.t_roll,
                                    ep_len_mean=float(np.mean(self.done_len)) if self.done_len else None,
                                    ep_rew_mean=float(np.mean(self.done_ret)) if self.done_ret else None)
                done = (n - self.n0) // run.rollout
                wall = max(time.perf_counter() - self.t0 - self.paused, 1e-9)
                if done in (1, 3):
                    run._rate_record("first_rollout" if done == 1 else "first_three", n - self.n0, wall, self.c0,
                                     self._cpu())
                self.rate = (n - self.n0) / wall

            def _on_training_end(self) -> None:
                self._flush()

        self.cb = cb = _Callback()
        remaining = self.budget - int(model.num_timesteps)
        if remaining > 0:
            try:
                model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=False, progress_bar=False)
            except _StopTraining:
                pass

    def _rate_record(self, key: str, steps: int, wall: float, c0: tuple, c1: tuple) -> None:
        """``RATE_RULE``.  The third rollout's rate is read before its update: two updates are included."""
        d = self.doc
        pids = list(getattr(self.venv, "worker_pids", []) or [])
        row = dict(env_steps=int(steps), wall_s=wall, env_steps_per_wall_s=steps / max(wall, 1e-9),
                   cpu_s_trainer=c1[0] - c0[0], cpu_s_workers=c1[1] - c0[1], workers=min(self.workers, N_ENVS),
                   vcpus=vcpus(), n_envs=N_ENVS, rss_mb=dict(trainer=_rss_mb(), workers=[_rss_mb(p) for p in pids]),
                   load_average=list(os.getloadavg()), at=s2io.now_utc())
        path = self.jd / "gates" / "rate.json"
        rec = s2io.read_json(path) if path.is_file() else dict(schema=RATE_SCHEMA, job=self.job,
                                                               run_uuid=d["run_uuid"], rule=RATE_RULE, sessions=[])
        mine = next((s for s in rec["sessions"] if s["session"] == self.k), None)
        if mine is None:
            mine = dict(session=self.k)
            rec["sessions"].append(mine)
        mine[key] = row
        s2io.write_json(path, rec)
        d["gates"]["rate"] = dict(record="gates/rate.json", session=self.k, **{
            k: v["env_steps_per_wall_s"] for k, v in mine.items() if isinstance(v, dict)})
        if key == "first_three":
            self.say(f"first rate: {row['env_steps_per_wall_s']:.0f} env steps/s ({N_ENVS} envs, "
                     f"{row['workers']} workers, {row['vcpus']} vCPUs)")

    def _finish(self) -> dict:
        from designlab.stage2 import s2load

        d = self.doc
        n = int(self.model.num_timesteps)
        if d["stopped"] is None:
            if n not in s2load.list_checkpoints(self.jd):
                self.save(n)  # (CHECKPOINT_RULE: the last update's policy; no rollout start follows)
            if self.gates and not any(e["at_steps"] == n and e.get("decided") for e in d["monitor"]):
                self.check(n, training_follows=False)  # (rule M2: and on a run's final checkpoint)
        d["num_timesteps"] = n
        d["status"] = d["stopped"]["status"] if d["stopped"] is not None else ("done" if n >= self.budget else
                                                                              "partial")
        self.say(f"{d['status']} at {n} steps")
        return d


def _monitor2():
    from designlab.stage2 import monitor2

    return monitor2


# ------------------------------------------------------------------------------------------------
# Probes (rules O2, O3, O4, K8): each writes one record
# ------------------------------------------------------------------------------------------------
def _record_path(out, name: str) -> Path:
    out = Path(out)
    return out if out.suffix == ".json" else out / name


def _write_record(path: Path, doc: dict) -> dict:
    from designlab.stage2 import s2load

    doc = dict(doc, versions=s2load.versions(), provenance=s2io.git_state(own_outputs=[str(path.resolve())]),
               at=s2io.now_utc())
    doc = s2io.plain(doc)
    doc["sha256"] = s2io.sha256_json(doc)
    s2io.write_json(path, doc)
    return doc


def _probe_cell(venv, layout: dict, rate: float, sigma, updates: int, n_steps: int, parent=None) -> dict:
    """6 updates at a constant rate on the pool: a fresh model (seed 42), with ``log_std`` set to ln ``sigma``
    where it is not rule O2's 0.1, or the parent's checkpoint with its own ``log_std`` (rule K8); per update the
    minibatch steps run and possible."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from designlab.stage1 import policy
    from designlab.stage2 import s2load

    d = policy.DESIGN2
    if parent is not None:
        ck = s2load.load_checkpoint(parent[0], parent[1], env=venv, lr=(rate, rate))
        model = ck["model"]
        model._last_obs = None  # (the probe starts from a reset of its own envs)
    else:
        vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=d["clip_obs"],
                               clip_reward=d["clip_reward"], gamma=d["gamma"])
        kw = policy.model_kwargs(int(layout["actor_dim"]), log_std_init=math.log(SIGMA))
        kw.update(learning_rate=schedules.ConstantSchedule(rate), n_steps=int(n_steps), clip_range=d["clip_range"])
        model = PPO(env=vecnorm, seed=PROBE_SEED, device="cpu", verbose=0, **kw)
        if float(sigma) != SIGMA:
            with torch.no_grad():
                model.policy.log_std.fill_(math.log(float(sigma)))
    counter = count_optimizer_steps(model)
    possible = minibatch_steps_possible(model)
    rows = []

    class _Count(BaseCallback):
        def __init__(self):
            super().__init__()
            self.mark = None

        def _row(self):
            if self.mark is not None:
                v = model.logger.name_to_value
                with torch.no_grad():
                    std = float(torch.exp(model.policy.log_std).mean())
                rows.append(dict(minibatch_steps_run=int(counter[0] - self.mark), minibatch_steps_possible=possible,
                                 approx_kl=float(v["train/approx_kl"]) if "train/approx_kl" in v else None,
                                 clip_fraction=float(v["train/clip_fraction"]) if "train/clip_fraction" in v else None,
                                 explained_variance=float(v["train/explained_variance"])
                                 if "train/explained_variance" in v else None, std_mean=std))
                self.mark = None

        def _on_rollout_start(self) -> None:
            self._row()

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self.mark = counter[0]

        def _on_training_end(self) -> None:
            self._row()

    model.learn(total_timesteps=int(updates) * model.n_steps * model.n_envs, callback=_Count(),
                reset_num_timesteps=parent is None, progress_bar=False)
    return dict(sigma=sigma, updates=rows,
                passes=bool(len(rows) == int(updates) and all(r["minibatch_steps_run"] == possible for r in rows)))


def probe_lr(bodies, set_name, out, rates=PROBE_RATES, sigmas=PROBE_SIGMAS, updates=PROBE_UPDATES, workers=None,
             parent=None) -> dict:
    """Rule O3's learning-rate probe (``LR_PROBE_RULE``), or with ``parent = (job directory, steps)`` rule K8's
    from a parent's checkpoint (``FINE_TUNE_RULE``: one body, the parent's own sigma; the rates tried are the
    parent's schedule value p at its step, then the listed rates below p).  The record (``lr_probe.json`` under
    ``out``, or ``out`` itself when it names a ``.json``): per body and head the rates tried with every update of
    every cell and the rate that passed; ``rates_by_body`` (rule O3 as settled on 15 Nov: the smallest passing
    rate over the probed heads of a body; None where a probed head has none); ``value`` (the smallest over
    everything probed)."""
    import torch

    from designlab.stage2 import vecpool

    torch.set_num_threads(1)
    bodies = list(bodies)
    w = min(N_ENVS, vcpus()) if workers is None else max(0, min(int(workers), N_ENVS))
    n_steps = N_STEPS
    par = None
    if parent is not None:
        pdir, psteps = _parent_of(parent)
        if len(bodies) != 1:
            raise Refused("a fine-tune's probe has one body: the parent's")
        pdoc = s2io.read_json(Path(pdir) / "run.json")
        s = pdoc["settings"]
        p = schedules.LinearSchedule(s["lr"]["start"], s["lr"]["end"])(1.0 - psteps / pdoc["budget"])
        rates = [p] + [r for r in FINE_TUNE_RATES if r < p]
        sigmas = ("the parent's",)
        n_steps = int(s["ppo"]["n_steps"])
        par = dict(job=Path(pdir).name, run_uuid=pdoc["run_uuid"], steps=int(psteps), p=float(p),
                   lr_schedule=s["lr"], budget=pdoc["budget"])
    out_bodies = {}
    for body in bodies:
        val = _validation(body, set_name)
        over = dict(filter_hz=val["filter"]["filter_hz"]) if val.get("filter") else {}
        cfgs = _env_configs("R5", body, set_name, over)
        venv = vecpool.PoolVecEnv(_env_specs("R5", cfgs), min(w, len(cfgs)), worker_cmd=[sys.executable, *WORKER_CMD])
        try:
            desc = venv.env_method("describe", indices=[0])[0]
            tried, passed = [], None
            for rate in rates:
                cells = []
                for sigma in sigmas:
                    cells.append(_probe_cell(venv, desc["layout"], float(rate), sigma, updates, n_steps,
                                             parent=None if parent is None else (pdir, psteps)))
                ok = all(c["passes"] for c in cells)
                tried.append(dict(rate=float(rate), cells=cells, passes=ok))
                if ok:
                    passed = float(rate)
                    break
        finally:
            venv.close()
        out_bodies[body] = dict(tried=tried, passed=passed, floors=[c.get("floor") for c in cfgs],
                                filter_hz=cfgs[0].get("filter_hz"), set_freeze=cfgs[0].get("set_freeze"),
                                task_names=list(desc["tasks"]), validation=val)
    by_body: dict = {}
    for body, row in out_bodies.items():
        g = by_body.setdefault(base_body(body), dict(rate=None, heads={}))
        g["heads"][body] = row["passed"]
    for g in by_body.values():
        got = list(g["heads"].values())
        g["rate"] = None if any(r is None for r in got) else min(got)
    all_passed = [r["passed"] for r in out_bodies.values()]
    doc = dict(schema="s2-lr-probe-1", rule=FINE_TUNE_RULE if parent is not None else LR_PROBE_RULE, set=set_name,
               rates=[float(r) for r in rates], sigmas=list(sigmas), updates=int(updates), seed=PROBE_SEED,
               n_envs=N_ENVS, n_steps=int(n_steps), workers=w, bodies=out_bodies, rates_by_body=by_body,
               value=None if any(r is None for r in all_passed) else min(all_passed), parent=par,
               note="rule O3 as settled on 15 Nov 2026 uses rates_by_body (one rate per body); value is the "
                    "smallest over everything probed here")
    return _write_record(_record_path(out, "lr_probe.json"), doc)


def _noise_episodes(body: str, set_name: str, filter_hz: float, sigma: float, episodes=None, k: int = PROBE_K,
                    env=None) -> tuple:
    """Rule O2's episodes at one sigma on one env (built here unless given): ``(rows, env)``."""
    from designlab.stage2 import r5env

    tasks = PROBE_TASKS_S1 if set_name == S1_SET else PROBE_TASKS
    if env is None:
        cfg = r5env.R5Config(body=body, set=set_name, mode="R5", floor="hard", tasks=tasks, randomize=False,
                             push_prob=0.0, head_events=False, mid_start=0.0, filter_hz=float(filter_hz),
                             set_freeze=r5env.set_freeze(body, set_name))
        env = r5env.make("R5", cfg)
    n = env.layout["action_dim"]
    rows = []
    for task in tasks:
        for ep in (PROBE_EPISODES if episodes is None else episodes):
            env.reset(seed=0, options=dict(task=task, start="stance", episode_seed=int(ep), push=None, head=None,
                                           factors=dict(latency_steps=int(k))))
            rng = np.random.default_rng(int(ep))
            while True:
                z = rng.standard_normal(n)  # (the same z for every sigma)
                _, _, term, trunc, info = env.step(np.clip(sigma * z, -1.0, 1.0).astype(np.float32))
                if term or trunc:
                    break
            rows.append(dict(task=task, episode_seed=int(ep), survived=not term, env_steps=int(env.k_env),
                             of=int(env.n_steps), survival_s=float(env.k_env * env.dt_env),
                             fall_cause=info.get("fall_cause")))
    return rows, env


def _noise_summary(rows: list) -> dict:
    return dict(episodes=len(rows), survived=sum(r["survived"] for r in rows),
                mean_survival_s=float(np.mean([r["survival_s"] for r in rows])) if rows else None)


def probe_sigma(bodies, set_name, out, sigmas=PROBE_SIGMA_VALUES) -> dict:
    """Rule O2's sigma probe (``SIGMA_PROBE_RULE``; reported, decides nothing): per body and sigma the 20 episodes,
    how many reach the task's end and the mean survival time.  One record (``sigma_probe.json``)."""
    out_bodies = {}
    for body in bodies:
        hz = (_validation(body, set_name).get("filter") or {}).get("filter_hz", FILTER_CUTOFFS[0])
        env, per = None, {}
        for sigma in sigmas:
            rows, env = _noise_episodes(body, set_name, hz, float(sigma), env=env)
            per["%g" % sigma] = dict(sigma=float(sigma), **_noise_summary(rows), runs=rows)
        env.close()
        out_bodies[body] = dict(filter_hz=float(hz), floor="hard", k=PROBE_K, sigmas=per)
    doc = dict(schema="s2-sigma-probe-1", rule=SIGMA_PROBE_RULE, set=set_name, sigmas=[float(s) for s in sigmas],
               episode_seeds=list(PROBE_EPISODES), bodies=out_bodies, decides="nothing (reported)")
    return _write_record(_record_path(out, "sigma_probe.json"), doc)


def filter_alpha(hz: float, dt_env: float = 0.02) -> float:
    return 1.0 - math.exp(-2.0 * math.pi * float(hz) * dt_env)


def filter_sigma(hz: float, hz0: float = FILTER_CUTOFFS[0], sigma0: float = SIGMA) -> float:
    """The sigma at cut-off ``hz`` whose filtered action has the RMS of ``sigma0`` at ``hz0`` (rule O4)."""
    a0, a = filter_alpha(hz0), filter_alpha(hz)
    return sigma0 * math.sqrt((a0 / (2.0 - a0)) / (a / (2.0 - a)))


def filter_choice(bodies, set_name, out, cutoffs=FILTER_CUTOFFS) -> dict:
    """Rule O4's measurement (``FILTER_RULE``): per body and cut-off rule O2's 20 episodes at equal filtered
    action RMS.  ``choice``: on ``FILTER_BODY`` (75-M-H-10) the last cut-off if its mean survival time is at least
    0.9 x the first's, else the first; None when that body was not measured.  One record (``filter.json``)."""
    out_bodies = {}
    for body in bodies:
        per = {}
        for hz in cutoffs:
            sigma = filter_sigma(hz, cutoffs[0])
            rows, env = _noise_episodes(body, set_name, float(hz), sigma)
            env.close()
            per["%g" % hz] = dict(filter_hz=float(hz), sigma=sigma, alpha=filter_alpha(hz), **_noise_summary(rows),
                                  runs=rows)
        a, b = per["%g" % cutoffs[0]]["mean_survival_s"], per["%g" % cutoffs[-1]]["mean_survival_s"]
        out_bodies[body] = dict(cutoffs=per, ratio=None if not a else b / a,
                                role="the rule's body" if body == FILTER_BODY else "reported; decides nothing")
    choice = None
    if FILTER_BODY in out_bodies:
        r = out_bodies[FILTER_BODY]["ratio"]
        choice = dict(body=FILTER_BODY, ratio=r, keep=FILTER_KEEP,
                      filter_hz=float(cutoffs[-1] if r is not None and r >= FILTER_KEEP else cutoffs[0]))
    doc = dict(schema="s2-filter-1", rule=FILTER_RULE, set=set_name, cutoffs=[float(c) for c in cutoffs],
               episode_seeds=list(PROBE_EPISODES), bodies=out_bodies, choice=choice)
    return _write_record(_record_path(out, "filter.json"), doc)


# ------------------------------------------------------------------------------------------------
# Preflight (the notebook's check): a policy built with MUJOCO_GL assigned, one tiny rollout in a worker
# ------------------------------------------------------------------------------------------------
PREFLIGHT_LAYOUT = dict(blocks={"q": (0, 4), "qd": (4, 8), "root_vel": (8, 11), "height": (11, 12)}, actor_dim=8,
                        obs_dim=12, action_dim=3, nj=4, heading=False)


def preflight_env(tag: int = 0):
    """A 12-observation, 3-action env without a plant: the factory ``preflight`` names to its worker."""
    import gymnasium
    from gymnasium import spaces

    class PreflightEnv(gymnasium.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (PREFLIGHT_LAYOUT["obs_dim"],), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (PREFLIGHT_LAYOUT["action_dim"],), np.float32)
            self._rng, self._t = np.random.default_rng(int(tag)), 0

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._t = 0
            return self._rng.normal(size=12).astype(np.float32), {}

        def step(self, action):
            self._t += 1
            obs = self._rng.normal(size=12).astype(np.float32)
            return obs, float(-np.sum(np.square(action))), False, self._t >= 10, {}

        def where(self) -> dict:
            return dict(pid=os.getpid(), mujoco_gl=os.environ.get("MUJOCO_GL"),
                        pyopengl_platform=os.environ.get("PYOPENGL_PLATFORM"), torch="torch" in sys.modules,
                        stable_baselines3="stable_baselines3" in sys.modules,
                        threads={v: os.environ.get(v) for v in tbenv.THREAD_VARS},
                        pythonhashseed=os.environ.get("PYTHONHASHSEED"), argv=list(sys.orig_argv))

    return PreflightEnv()


def preflight(workers: int = 1) -> dict:
    """Build a policy in this process (``MUJOCO_GL`` was assigned when this module was imported), run one rollout
    and one update on two stub envs in a pool worker, save the archive and look for bytecode in it.  What the
    trainer and its worker report of themselves; ``ok``."""
    import tempfile

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from designlab.stage1 import policy
    from designlab.stage2 import s2load, vecpool

    tbenv.repo_on_path()
    from environments.shared.policy_loading import inspect_sb3_archive

    torch.set_num_threads(1)
    venv = vecpool.PoolVecEnv([("designlab.stage2.train2:preflight_env", {"tag": i}) for i in range(2)],
                              max(0, min(int(workers), 2)), worker_cmd=[sys.executable, *WORKER_CMD])
    try:
        vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0, gamma=0.99)
        kw = policy.model_kwargs(PREFLIGHT_LAYOUT["actor_dim"], log_std_init=math.log(SIGMA))
        kw.update(learning_rate=schedules.LinearSchedule(1e-4, 3e-5), n_steps=16, batch_size=32, clip_range=0.2)
        model = PPO(env=vecnorm, seed=0, device="cpu", verbose=0, **kw)
        counter = count_optimizer_steps(model)
        model.learn(total_timesteps=32, progress_bar=False)
        where = venv.env_method("where", indices=[0])[0]
        with tempfile.TemporaryDirectory() as td:
            rec = s2load.save_checkpoint(td, 32, model, vecnorm, PREFLIGHT_LAYOUT,
                                         dict(run_uuid="preflight", run_id=None, job="preflight",
                                              settings_sha256="preflight"), {})
            bytecode = sorted(inspect_sb3_archive(s2load.checkpoint_files(td, 32)["model"]).bytecode_members)
            again = s2load.load_checkpoint(td, 32)
    finally:
        venv.close()
    out = dict(mujoco_gl=os.environ.get("MUJOCO_GL"), pyopengl_platform=os.environ.get("PYOPENGL_PLATFORM"),
               torch_threads=int(torch.get_num_threads()), worker=where, steps=int(model.num_timesteps),
               minibatch_steps_run=int(counter[0]), bytecode_members=bytecode,
               fingerprint_max_abs_diff=again["fingerprint_max_abs_diff"], versions=rec["versions"])
    out["ok"] = bool(out["mujoco_gl"] == "disable" and where["mujoco_gl"] == "disable" and not where["torch"]
                     and not bytecode and out["steps"] == 32 and out["minibatch_steps_run"] > 0
                     and out["fingerprint_max_abs_diff"] <= s2load.FINGERPRINT_TOL)
    return out


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _list(text) -> list:
    return [x for x in str(text or "").split(",") if x.strip() != ""]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["worker"]:  # a worker of the env pool: numpy and gymnasium, never torch
        if len(argv) != 2:
            print("usage: python -m designlab.stage2.train2 worker ADDRESS", file=sys.stderr)
            return 2
        from designlab.stage2 import vecpool

        return vecpool.worker_main(argv[1])
    if argv[:1] == ["monitor"]:
        from designlab.stage2 import monitor2

        return monitor2.main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.train2", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run", help="one training job")
    a.add_argument("--mode", required=True, choices=MODES)
    a.add_argument("--body", required=True)
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--set", default="V6")
    a.add_argument("--out", required=True, help="the root: the job's directory is <out>/jobs/<job>")
    a.add_argument("--job", default=None)
    a.add_argument("--steps", type=int, default=None)
    a.add_argument("--workers", type=int, default=None)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--allow-version-change", action="store_true")
    a.add_argument("--allow-source-change", action="store_true")
    a.add_argument("--from-checkpoint", default=None, metavar="JOBDIR:STEPS")
    a.add_argument("--lr", type=float, default=None)
    a.add_argument("--monitor-every", type=int, default=MONITOR_EVERY)
    a.add_argument("--monitor-also-k", default="")
    a.add_argument("--no-gates", action="store_true")
    a.add_argument("--tiny", action="store_true")
    a.add_argument("--run-id", default=None)
    a.add_argument("--retry-check", action="store_true",
                   help="after a halt for two attempts at one check that were never closed: run it once more")
    a.add_argument("--config", default=None, help="a JSON object of env config fields for every env")
    for name, text in (("probe-lr", "rule O3's learning-rate probe"), ("probe-sigma", "rule O2's sigma probe"),
                       ("filter", "rule O4's filter measurement")):
        a = sub.add_parser(name, help=text)
        a.add_argument("--bodies", required=True, help="comma-separated")
        a.add_argument("--set", default="V6")
        a.add_argument("--out", required=True, help="the record (a .json path) or its directory")
        if name == "probe-lr":
            a.add_argument("--workers", type=int, default=None)
            a.add_argument("--parent", default=None, metavar="JOBDIR:STEPS", help="rule K8's probe")
    a = sub.add_parser("preflight", help="a policy built and one rollout run in a worker")
    a.add_argument("--workers", type=int, default=1)
    sub.add_parser("worker", help="ADDRESS: a worker of the env pool (started by the pool)")
    sub.add_parser("monitor", help="...: the monitor child (designlab.stage2.monitor2's arguments)")
    args = ap.parse_args(argv)
    try:
        s2io.require_one_thread()
    except s2io.S2Error as exc:
        print(f"train2 {args.cmd}: {exc}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    try:
        if args.cmd == "preflight":
            res = preflight(args.workers)
            print(json.dumps(s2io.plain(res)), flush=True)
            return EXIT_DONE if res["ok"] else EXIT_FAILED
        if args.cmd in ("probe-lr", "probe-sigma", "filter"):
            if args.cmd == "probe-lr":
                doc = probe_lr(_list(args.bodies), args.set, args.out, workers=args.workers, parent=args.parent)
                print(f"lr probe on set {args.set}: {doc['rates_by_body']}", flush=True)
            elif args.cmd == "probe-sigma":
                doc = probe_sigma(_list(args.bodies), args.set, args.out)
                print(f"sigma probe on set {args.set}: written", flush=True)
            else:
                doc = filter_choice(_list(args.bodies), args.set, args.out)
                print(f"filter on set {args.set}: {doc['choice']}", flush=True)
            return EXIT_DONE
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise Refused("start the trainer with PYTHONHASHSEED=0 (design D16: the trainer and its workers; "
                          f"it is {os.environ.get('PYTHONHASHSEED')!r})")
        doc = train(args.mode, args.body, args.seed, set_name=args.set, out=args.out, job=args.job, steps=args.steps,
                    resume=args.resume, workers=args.workers, monitor_every=args.monitor_every,
                    monitor_also_k=tuple(int(k) for k in _list(args.monitor_also_k)), lr=args.lr,
                    config=json.loads(args.config) if args.config else None, gates=not args.no_gates,
                    allow_version_change=args.allow_version_change, allow_source_change=args.allow_source_change,
                    parent=args.from_checkpoint, tiny=args.tiny, run_id=args.run_id,
                    retry_check=args.retry_check, log=lambda *x: print(*x, flush=True))
    except TrainError as exc:
        print(f"train2 {args.cmd}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return exc.code
    return {"stopped_g8a": EXIT_G8A, "partial": EXIT_FAILED}.get(doc["status"], EXIT_DONE)


__all__ = ["BOUNDARY_RULE", "CHECKPOINT_EVERY", "EXIT_DONE", "EXIT_FAILED", "EXIT_G8A", "EXIT_HALT", "EXIT_REFUSED",
           "Halt", "MONITOR_EVERY", "N_ENVS", "N_STEPS", "Refused", "RESUME_RULE", "SCHEMA", "SOURCE_FILES", "TINY",
           "TrainError", "base_body", "boundary_seed", "count_optimizer_steps", "filter_choice", "filter_sigma",
           "is_boundary", "job_name", "kl_share", "main", "preflight", "preflight_env", "probe_lr", "probe_sigma",
           "read_updates", "sources", "train"]

if __name__ == "__main__":
    from designlab.stage2 import train2 as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
