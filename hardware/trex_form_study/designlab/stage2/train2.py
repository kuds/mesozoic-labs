"""Track B's trainer for R5 and R6 (round C design D2, D12-D16; section 3.6; Track B rules O1-O4, M1-M6, K1-K5,
K8, K9; ``stage2/design/ROUND_C_RULINGS_16NOV.md``).  A new module: ``designlab/stage1/train.py`` and ``policy.py``
are imported, not edited.

    python -m designlab.stage2.train2 run --mode R5|R6 --body B --seed N [--set V6] --out ROOT [--job NAME] [--steps N]
           [--workers W] [--resume] [--allow-version-change] [--allow-source-change] [--from-checkpoint JOBDIR:STEPS]
           [--lr X] [--monitor-every N] [--monitor-also-k K,...] [--no-gates] [--tiny] [--run-id ID] [--config JSON]
           [--retry-check] [--expect-baseline PATH]
    python -m designlab.stage2.train2 {probe-lr|probe-sigma|filter} --bodies B,... --set S --out PATH
           [probe-lr: --workers W] [probe-lr, rule K8's probe: --parent JOBDIR:STEPS --mode R5|R6 --config JSON]
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
``env_method("set_task_weights")``), the stop rules.  A stop by a rule ends the job with exit code 0.  A monitor
that refuses to run (its exit code ``monitor2.EXIT_REFUSAL``: its sources have moved, the run's ``run.json`` is
missing, the checkpoint is not intact, the checkpoint is another run's, the row refuses the checkpoint while it is
built) made no attempt: the attempt is closed as ``refused`` with the monitor's reason, it is not one of the two
closed attempts after which a check counts as without a record, the run halts, and a resume runs the check again
(with ``--retry-check`` when the refused attempt was the one that parameter let run; ``monitor2.RULES["refusal"]``).
The monitor child's environment holds ``STAGE2_ALLOW_SOURCE_CHANGE=1`` exactly when the session's sources are
allowed (``ALLOW_SOURCE_ENV``): the session was given ``--allow-source-change``, or an earlier session given it ran on
the same sources (``SOURCE_ALLOWANCE_RULE``: an allowance holds for the rest of the run for the same sources, and
``run.json`` ``sources_allowed`` records the hashes).  The trainer passes the allowance to the policy row, which reads
it there (rule K9), and a value the trainer's own environment holds is not passed on.  A halt for moved sources
(the monitor's own sources check, or the row's rule K9) names ``--allow-source-change``.

**The committed baseline** (rule M3; the lead's ruling 17 of 17 Nov 2026; ``EXPECT_BASELINE_RULE``).  With
``--expect-baseline PATH`` (S2-4: ``stage2/tb/runs/s2-4/R0L_BASELINE.json``, made by ``monitor2 baseline`` without a
job and committed) the run's own R0L baseline must equal that file episode for episode (``monitor2.baseline_equal``:
the outcome, the survival and, in every episode that ran, the state digest: the lead's ruling 33 of 17 Nov 2026),
checked after the baseline and before the first rollout, else the run halts.  The file's sha256 is a setting of the run (``settings.expect_baseline``): a resume gives the same file.

**Exit codes** of ``run``: 0 done, or stopped by rule M4 or M5; 3 G8a failed ("G8a fails: no run"); 4 refused (a
resume the rules refuse, or a start that cannot be made: nothing was run; among them an ``--expect-baseline`` file
that is no baseline record, or one given without the gates); 5 halted for the lead (R0L's baseline could not be
made, or is not the committed one that ``--expect-baseline`` names; two attempts at one check were never closed:
``--retry-check`` on the resume lets that check run once more; the monitor refused to run on a check: a resume runs it
again, with ``--retry-check`` when the refused attempt was the one that parameter let run, and the halt says so;
``run.json`` ``halt`` holds the reason, for a refusal the monitor's own); any other code is a failure.

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

# ``s2io.git_state`` runs ``git status``, which rewrites the checkout's index under ``index.lock`` unless it is
# told not to.  A SIGKILL of the job's process group in that instant leaves the lock behind (seen on 16 Nov 2026
# after a kill test), and every later git command that writes the index refuses to run until someone removes it.
# The trainer's processes only read the checkout's state; their children inherit the variable.
os.environ.setdefault("GIT_OPTIONAL_LOCKS", "0")

if __name__ == "__main__":  # before any numpy import (round A's module skeleton)
    for _v in tbenv.THREAD_VARS:
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402
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
LR_PROBE_SCHEMA, FILTER_SCHEMA = "s2-lr-probe-1", "s2-filter-1"
#: How long a resume waits for a monitor child of an earlier session to be gone after its SIGKILL (``CHILD_RULE``).
CHILD_WAIT_S = 20.0
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
    "names, must be the run's); other inputs (the envs are built from the run's stored configs, which name the "
    "freeze the run started on and its tasks: a used file's sha256 or entry hash, the plant's inputs or the "
    "resolved task table that differs; a later freeze of the set is accepted when these are unchanged, and both "
    "freezes are recorded); other sources than the last session's unless --allow-source-change, or unless an "
    "earlier session given it ran on the same sources (SOURCE_ALLOWANCE_RULE); other library "
    "versions than the last session's (s2load.comparable) unless --allow-version-change; a checkpoint whose "
    "run_uuid or settings hash is not the run's; a fingerprint that is not reproduced within 1e-5.  A checkpoint "
    "that fails its integrity check is skipped with its reason.  No intact checkpoint is no refusal: the job "
    "restarts from step 0 in place and is the same run.  The run state is taken from the checkpoint's record: "
    "run.json's live fields are rebuilt from it (its list of checkpoints holds the step resumed from, whether or "
    "not the session that saved it lived to write so), later entries move to the session's discarded_after_resume "
    "(the attempts logged for a check on the checkpoint itself are kept), updates.jsonl is cut back to the resumed "
    "step, a check that is due at that step and has no record naming that checkpoint is run, and the rules are "
    "applied to the records as the uninterrupted run applies them.  A monitor_<n>.json of the resumed step that "
    "cannot be read (a copy that was cut short) is no record (rule M1): it is kept beside and the check is run")
BASELINE_RULE = (
    "rule M1: R0L's baseline is made once per run, before its first rollout; run.json baseline holds its record's "
    "sha256, its summary and, per episode, the condition, the seed, the outcome and the survival (with the state "
    "digest where the record holds one).  A resume reads monitor/r0l.json and refuses one that is not the record "
    "run.json names.  If the file is lost, or cannot be read (it is then kept beside as r0l.unreadable.s<k>.json), "
    "R0L's episodes are run again and compared with what run.json holds, episode for episode and in the summary: "
    "equal, the new record takes the old one's place and the session is named under baseline.remade; not equal, "
    "the new record is kept as r0l.differs.s<k>.json, nothing is replaced and the run halts and goes to the lead "
    "(the run's checks were read against the lost one)")
#: Rule M3 as the lead ruled it on 17 Nov 2026 (ruling 17): S2-4's own baseline against the committed one.
EXPECT_BASELINE_RULE = (
    "rule M3 (the lead's ruling 17 of 17 Nov 2026): with --expect-baseline PATH, the committed R0L baseline (S2-4: "
    "stage2/tb/runs/s2-4/R0L_BASELINE.json, made by the monitor's baseline command without a job), the run's own "
    "monitor/r0l.json must equal it episode for episode: the condition, the seed, the outcome, the survival and the "
    "state digest (monitor2.baseline_equal), with the digests compared in both in every episode that ran (the "
    "lead's ruling 33 of 17 Nov 2026: in every pair of which at least one episode ran, both hold a digest that is a "
    "non-empty string; two episodes not run, which hold none, are compared on their outcome and survival; with no "
    "episode that ran no digest was compared); it is checked after the baseline is "
    "made or read and before the first rollout, at every session, and recorded in run.json baseline_expected.  Not "
    "equal: the run halts (exit code 5) and goes to the lead; no step is trained.  A file that is missing, cannot be "
    "read or is no baseline record, or one given to a run without the gates, is refused before anything is "
    "written.  The file's sha256 is a setting of the run (settings.expect_baseline): a resume must give the same "
    "file (rule K4)")
CHILD_RULE = (
    "an attempt at a check, or at the baseline, holds its monitor child's pid, that process's start time (clock "
    "ticks since boot) and the boot id, written when the child has started.  A resume ends, with SIGKILL, every "
    "such child of an earlier session on this host that is still alive under an attempt that was never closed "
    "(the trainer was killed alone while its monitor ran), before any check or baseline is run: two monitors "
    "never write one record.  The attempt stays what it was, an interruption (rule M1); the session lists what it "
    "ended under ended_children.  The trainer waits for its monitor child without a timeout (the lead's ruling 17 "
    "of 17 Nov 2026): a timeout would close a hung attempt with an exit code and count it toward the two closed "
    "attempts; a hang is ended by whoever watches the run, who kills the job's process group, and the attempt that "
    "was never closed is an interruption")
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
    "heads that were probed.  The probe runs with rule O4's filter (measured first) and is run once per body and "
    "head: a later call adds its heads to the record it finds, with the earlier ones as they were, probes none of "
    "them again and takes each body's rate over every head the record then holds; it adds nothing to a record of "
    "other settings")
SIGMA_PROBE_RULE = (
    "rule O2 (reported; decides nothing): the stance-start walk_nominal and walk_v0.4 tasks at a servo delay of 5 "
    "physics steps on the nominal plant, on hard floor, with no push and no head event, 10 episodes each (episode "
    "seeds 7000-7009), actions clip(sigma z) per env step with the same z for every sigma (z from "
    "default_rng(episode seed)), through the env's clip and filter: survivals and the mean survival time per sigma")
FILTER_RULE = (
    "rule O4: rule O2's 20 episodes under each cut-off with the same z, the noise scaled so that the filtered "
    "action has the same RMS (sigma 0.1 at the first cut-off; x sqrt((a0 / (2 - a0)) / (a / (2 - a))) at another, a "
    "= 1 - exp(-2 pi f x 0.02 s)); 20 Hz is used if its mean survival time over 75-M-H-10's 20 episodes is at "
    "least 0.9 x that at 10 Hz, else 10 Hz; another body's measurement is reported and decides nothing.  Each "
    "body is measured once: a later call adds its bodies to the record it finds, measures none of them again and "
    "keeps the choice")
FILTER_NEED_RULE = (
    "rule O4 is measured before any run: a run that is not tiny, and rule O3's probe on any set but the Stage 1 "
    "view, is refused when filter.json holds no choice, unless the caller names the filter (config filter_hz); a "
    "tiny run then takes the env's own default")


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


#: The environment variable by which a policy row is told that it may run on moved sources, flagged (rule K9;
#: ``s2rows.SOURCE_CHANGE_ENV``): the monitor child gets it from the trainer's ``--allow-source-change`` alone
#: (the rulings of 17 Nov 2026, 10).
ALLOW_SOURCE_ENV = "STAGE2_ALLOW_SOURCE_CHANGE"


#: The lead's ruling 12 of 17 Nov 2026 (the check of the third follow-up, finding 2).
SOURCE_ALLOWANCE_RULE = (
    "the lead's ruling 12 of 17 Nov 2026: an allowance for moved sources holds for the rest of the run for the same "
    "sources.  A session given --allow-source-change records the hash of the sources it runs on in run.json "
    "sources_allowed (with its index), once per hash; a later session whose sources hash is one of them is allowed "
    "as if it had been given the parameter: its resume accepts those sources against the last session's (flagged, "
    f"as with the parameter) and its monitor children get {ALLOW_SOURCE_ENV}=1 for the policy row (rule K9); the "
    "session's source_allowance names the session whose parameter allowed them.  Sources that no session given the "
    "parameter ran on are allowed only by the parameter.  A halt for moved sources (the monitor's own sources "
    "check, or the row's refusal by rule K9) names --allow-source-change, the parameter that resolves it")
#: What a halt for moved sources adds to its reason (``SOURCE_ALLOWANCE_RULE``).
SOURCE_REMEDY = ("the sources moved (the monitor's sources check, or the row's rule K9): a resume with "
                 "--allow-source-change runs the check on them, flagged")


def allowed_sources(doc: dict) -> dict:
    """``{sources sha256: the session whose --allow-source-change allowed it}`` of a ``run.json``
    (``SOURCE_ALLOWANCE_RULE``): its ``sources_allowed``, and, for a ``run.json`` written before that key, the
    sources of every session that was given the parameter or resumed on moved sources with it."""
    out = {}
    for e in doc.get("sources_allowed") or []:
        if isinstance(e, dict) and isinstance(e.get("sha256"), str):
            out.setdefault(e["sha256"], e.get("session"))
    for i, sess in enumerate(doc.get("sessions") or []):
        h = (sess.get("sources") or {}).get("sha256")
        if isinstance(h, str) and (sess.get("allow_source_change") is True or sess.get("source_change") is True):
            out.setdefault(h, sess.get("index", i))
    return out


def moved_sources(why: dict | None) -> bool:
    """Whether a refusal of the monitor (``_Run._refusal``'s ``{kind, reason}``) is one for moved sources: the
    monitor's own sources check (kind ``sources``) or the row's rule K9 (kind ``row``, a reason that cites rule K9
    or ``ALLOW_SOURCE_ENV``, as ``s2rows``' refusal of moved sources does)."""
    if not why:
        return False
    kind, reason = why.get("kind"), str(why.get("reason") or "")
    return kind == "sources" or (kind == "row" and ("rule K9" in reason or ALLOW_SOURCE_ENV in reason))


def monitor_env(allow_source_change: bool, base=None) -> dict:
    """The monitor child's environment: ``tbenv.child_env("disable", base)`` with ``ALLOW_SOURCE_ENV`` = ``"1"``
    when the session's sources are allowed (``--allow-source-change``, or ``SOURCE_ALLOWANCE_RULE``: sources an
    earlier session given the parameter ran on), and without it otherwise (a value ``base`` holds is not passed on:
    the row's allowance is the trainer's)."""
    env = tbenv.child_env("disable", base)
    env.pop(ALLOW_SOURCE_ENV, None)
    if allow_source_change:
        env[ALLOW_SOURCE_ENV] = "1"
    return env


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


def _overrides(config: dict | None, val: dict) -> dict:
    """The env config fields that a job, or a probe, gives every env: the caller's, and rule O4's choice from the
    validation record unless the caller names a filter."""
    over = dict(config or {})
    if (val or {}).get("filter") and "filter_hz" not in over:
        over["filter_hz"] = val["filter"]["filter_hz"]
    return over


def _need_filter(who: str, config: dict | None, val: dict) -> None:
    """``FILTER_NEED_RULE``: ``Refused`` when neither the validation record nor the caller gives the filter."""
    if (val or {}).get("filter") or "filter_hz" in (config or {}):
        return
    raise Refused(f"{who}: no action filter: run `train2 filter` on {FILTER_BODY} first (rule O4: "
                  f"{s2io.portable(str(VALIDATION / 'filter.json'))} holds no choice), or name one (--config "
                  "'{\"filter_hz\": 10.0}')")


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


def require_hashseed() -> None:
    """``Refused`` unless this process runs with ``PYTHONHASHSEED=0`` (design D16: the trainer and its workers;
    the pool's workers and the monitor child get it from ``tbenv.child_env``, the trainer and a probe from whoever
    starts them)."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise Refused("start the trainer with PYTHONHASHSEED=0 (design D16: the trainer and its workers; "
                      f"it is {os.environ.get('PYTHONHASHSEED')!r})")


def _read_record(path):
    """The JSON document of a record file; None when the file is not there or cannot be read as JSON (a copy that
    was cut short: by rule M1 such a file is no record)."""
    try:
        return s2io.read_json(path)
    except (OSError, ValueError):
        return None


def _proc(pid) -> tuple | None:
    """``(start time in clock ticks since boot, state)`` of a process (``/proc/<pid>/stat``, fields 22 and 3); None
    where there is no such process, or no ``/proc``.  The start time tells a process from a later one that got its
    pid; state ``Z`` is a process that has ended and was not yet collected."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        rest = stat[stat.rindex(")") + 2:].split()
        return int(rest[19]), rest[0]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
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
          run_id=None, log=print, retry_check=False, expect_baseline=None) -> dict:
    """One training job (module docstring); its ``run.json``.

    ``out``: the root (``<out>/jobs/<job>``); ``job``: the job's name (default ``<mode>-<body>-s<seed>``; a
    fine-tune names its own); ``steps``: the budget asked (rounded up to whole rollouts; required unless ``tiny`` or
    ``resume``); ``workers``: W (default one per vCPU, at most 8; 0: the envs in this process); ``lr``: the starting
    rate (default: the body's rate of ``lr_probe.json``); ``config``: ``R5Config`` fields for every env; ``ppo``:
    PPO constructor arguments over design 2's (smoke checks; both are settings); ``gates`` False: ``GATES_RULE``;
    ``parent``: ``(job directory, steps)`` of a fine-tune's parent (rule K8); ``tiny``: design 3.6's tiny run;
    ``retry_check``: on a resume of a run that halted after two attempts at one check were never closed, let that
    check run once more (``RETRY_RULE``; recorded in the session); ``expect_baseline``: the path of the committed R0L
    baseline the run's own must equal before its first rollout (rule M3, ``EXPECT_BASELINE_RULE``).

    Raises ``Refused`` (a resume the rules refuse), ``Halt`` (rule M1: the lead) or ``TrainError``; a job that a
    rule stopped returns its ``run.json`` like one that finished (``status``)."""
    return _Run(dict(mode=mode, body=body, seed=seed, set_name=set_name, out=out, job=job, steps=steps, resume=resume,
                     workers=workers, checkpoint_every=checkpoint_every, monitor_every=monitor_every,
                     monitor_also_k=monitor_also_k, lr=lr, config=config, ppo=ppo, gates=gates,
                     allow_version_change=allow_version_change, allow_source_change=allow_source_change,
                     parent=parent, tiny=tiny, run_id=run_id, log=log, retry_check=retry_check,
                     expect_baseline=expect_baseline)).go()


class _Run:
    """One call of ``train``: the arguments, the job's files and the session's state."""

    def __init__(self, a: dict):
        require_hashseed()
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
        self.expected = expected_baseline(a.get("expect_baseline"))  # (EXPECT_BASELINE_RULE: before any write)
        if self.expected is not None and not self.gates:
            raise Refused("--expect-baseline: a run without the gates (--no-gates) makes no R0L baseline to compare "
                          "with the committed one (rule M3)")
        self.source_allowance = None  # (SOURCE_ALLOWANCE_RULE: set by _refuse_sources_versions)
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
                    parent=None if self.parent is None else dict(job=self.parent[0].name, steps=self.parent[1]),
                    expect_baseline=None if self.expected is None else self.expected["sha256"])

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
            if not self.tiny:
                _need_filter(self.job, self.config, self.val)
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
            if not new:
                self._end_children()
            if not complete:
                if self.gates and not (doc["gates"].get("g8a") or {}).get("passes") and not self._g8a():
                    doc["status"] = "stopped_g8a"
                    return doc
                self._open()
            self._model(loaded, new)
            if self.gates:
                self._baseline()
                self._expect_baseline()
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
                       retry=RETRY_RULE, baseline=BASELINE_RULE, children=CHILD_RULE, filter=FILTER_NEED_RULE,
                       source_allowance=SOURCE_ALLOWANCE_RULE, expect_baseline=EXPECT_BASELINE_RULE,
                       monitor=dict(m2.RULES)),
            layout=None, sessions=[], sources_allowed=[], checkpoints=[], kept=[],
            gates=dict(g8a=None, rate=None, g8b=None, g8b_after_fix=None), monitor=[], baseline=None,
            baseline_attempts=[], baseline_expected=[], fix=None, flags=dict(kl_cut=False), no_gain_count=0,
            stopped=None, parent=None,
            final_check=None, halt=None, updates_file="updates.jsonl", status="running", budget=None,
            budget_asked=self.steps_asked, num_timesteps=0)

    def _refuse_sources_versions(self, doc: dict, src: dict, vers: dict) -> dict:
        """``RESUME_RULE``: the sources and the library versions against the last session's; what the new session
        records of an allowed change.  The session's sources are allowed (``self.source_allowance``) by its
        ``--allow-source-change`` or, without it, when an earlier session given it ran on the same sources
        (``SOURCE_ALLOWANCE_RULE``)."""
        from designlab.stage2 import s2load

        out = {}
        allowed = allowed_sources(doc)
        if self.allow_source:
            self.source_allowance = dict(sha256=src["sha256"], by="--allow-source-change", session=len(doc["sessions"]))
        elif src["sha256"] in allowed:
            self.source_allowance = dict(sha256=src["sha256"], by="sources_allowed", session=allowed[src["sha256"]])
        if not doc["sessions"]:
            return out
        last = doc["sessions"][-1]
        was = (last.get("sources") or {}).get("sha256")
        if was != src["sha256"]:
            if self.source_allowance is None:
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
            retry_check=self.retry_check, allow_source_change=self.allow_source,
            source_allowance=self.source_allowance,
            expect_baseline=None if self.expected is None else {k: self.expected[k] for k in ("path", "sha256")},
            error=None, env=None, other_envs=None)
        session.update(flagged)
        if self.allow_source and src["sha256"] not in allowed_sources(doc):  # (SOURCE_ALLOWANCE_RULE)
            doc.setdefault("sources_allowed", []).append(dict(sha256=src["sha256"], session=self.k,
                                                              at=session["started_at"]))
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
        overrides = _overrides(self.config, val)  # (rule O4's choice, unless the caller names one)
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
                             workers=self.workers, parent=(pdir, psteps), mode=self.mode,
                             config=_overrides(self.config, self.val))
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
        # (the step resumed from is a checkpoint, also when its session was killed between the checkpoint's last
        # file and run.json)
        d["checkpoints"] = sorted({int(m) for m in d["checkpoints"] if m <= at}
                                  | ({int(n)} if n is not None else set()))
        d["kept"] = s2load.list_checkpoints(self.jd)
        cut = _cut_updates(self.jd / "updates.jsonl", at, self.k)
        if cut:
            disc["updates"] = dict(lines=cut, file="updates.discarded.s%d.jsonl" % self.k)
        if d.get("num_timesteps", 0) > max(at, 0):
            disc["num_timesteps"] = d["num_timesteps"]
        d["num_timesteps"] = max(at, 0)
        self.session["discarded_after_resume"] = disc or None

    # ---------------------------------------------------------------- the monitor child
    def _child(self, args: list, att: dict | None = None) -> int:
        """Run the monitor child and wait for it (a subprocess: nothing forks while the pool is open).  ``att``:
        the attempt's log entry, already written; it takes the child's pid, start time and boot id as soon as the
        child has started (``CHILD_RULE``).  The wait has no timeout (``CHILD_RULE``; the lead's ruling 17 of 17 Nov
        2026).  The child's environment is ``monitor_env``'s: the session's allowance
        of its sources (``--allow-source-change``, or ``SOURCE_ALLOWANCE_RULE``) passed to the row."""
        cmd = _monitor_cmd() + [str(x) for x in args]
        cmd += ["--sources", self.session["sources"]["sha256"]]
        if self.tiny:
            cmd.append("--tiny")
        if self.also_k:
            cmd += ["--also-k", ",".join("%d" % k for k in self.also_k)]
        proc = subprocess.Popen(cmd, env=monitor_env(self.source_allowance is not None), cwd=tbenv.study_dir(),
                                stdin=subprocess.DEVNULL)
        try:
            if att is not None:
                att.update(pid=int(proc.pid), pid_start=(_proc(proc.pid) or (None,))[0], boot_id=_boot_id())
                self.write()
            return int(proc.wait())
        except BaseException:  # (as subprocess.run: the child does not outlive an exception in its trainer)
            proc.kill()
            proc.wait()
            raise

    def _refusal(self, code: int, steps: int | None = None) -> dict | None:
        """``monitor2.RULES["refusal"]``: None unless the monitor child left with its refusal's exit code; then
        ``{kind, reason}`` from the note it wrote beside the record (``steps`` None: the baseline's).  Without a
        readable note ``kind`` is None and the reason names what a refusal can be."""
        m2 = _monitor2()
        if int(code) != m2.EXIT_REFUSAL:
            return None
        note = _read_record(m2.refusal_path(self.jd / "monitor", steps))
        if isinstance(note, dict) and note.get("reason"):
            return dict(kind=note.get("kind"), reason=str(note["reason"]))
        return dict(kind=None, reason=f"exit code {int(code)} and no note of its reason: its sources have moved, the "
                                      "run's run.json is missing, the checkpoint is not intact, the checkpoint is "
                                      "another run's, or the row refused the checkpoint while it was built")

    def _end_children(self) -> None:
        """``CHILD_RULE``: end the monitor children of earlier sessions that outlived their trainer."""
        d, host, boot, ended = self.doc, socket.gethostname(), _boot_id(), []
        rows = [("baseline", None, a) for a in d.get("baseline_attempts") or []]
        rows += [("check", e["at_steps"], a) for e in d["monitor"] for a in e.get("attempts") or []]
        for kind, n, a in rows:
            pid, start, s = a.get("pid"), a.get("pid_start"), a.get("session")
            if a.get("closed_at") is not None or pid is None or start is None or not isinstance(s, int):
                continue
            if s >= self.k or d["sessions"][s].get("host") != host or a.get("boot_id") != boot:
                continue

            def alive(pid=pid, start=start) -> bool:
                now = _proc(pid)
                return now is not None and now[0] == start and now[1] != "Z"

            if not alive():
                continue
            self.say(f"WARNING: the monitor child {pid} of session {s} ({kind}"
                     + ("" if n is None else f" at {n} steps") + ") outlived its trainer: it is ended")
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + CHILD_WAIT_S
            while alive():
                if time.monotonic() > deadline:
                    raise TrainError(f"{self.job}: the monitor child {pid} of session {s} is still alive "
                                     f"{CHILD_WAIT_S:g} s after its SIGKILL: no second monitor is started beside it")
                time.sleep(0.05)
            a["child_ended"] = dict(session=self.k, at=s2io.now_utc())
            ended.append(dict(kind=kind, at_steps=n, pid=int(pid), session=int(s)))
        if ended:
            self.session["ended_children"] = ended
            self.write()

    def _baseline(self) -> None:
        """R0L's episodes, once per run (rule M1): no check is read against a baseline that holds an error.  A
        record that is lost or cannot be read is made again and must be the one ``run.json`` names
        (``BASELINE_RULE``)."""
        m2, d = _monitor2(), self.doc
        path = self.jd / "monitor" / "r0l.json"
        rec, was = _read_record(path), d["baseline"]
        remade = list((was or {}).get("remade") or [])
        if rec is None and path.is_file():  # (a copy that was cut short is no record; it is kept beside)
            os.replace(path, path.with_name("r0l.unreadable.s%d.json" % self.k))
            self.say("WARNING: monitor/r0l.json cannot be read: it is kept beside and R0L's episodes are run again")
        if was is not None and rec is not None and rec.get("sha256") != was.get("sha256"):
            raise Refused(f"{path}: not the baseline record run.json names (its sha256 differs)")
        if rec is None or m2.baseline_problem(rec) is not None:
            self.say("the R0L baseline")
            att = dict(session=self.k, started_at=s2io.now_utc(), closed_at=None, exit_code=None, record=None)
            d["baseline_attempts"].append(att)
            self.write()
            code = self._child(["baseline", "--job", self.jd, "--body", self.body, "--set", self.set, "--k", m2.K],
                               att)
            rec = _read_record(path)
            ok = rec is not None and m2.baseline_problem(rec) is None
            why = None if ok else self._refusal(code)
            att.update(closed_at=s2io.now_utc(), exit_code=int(code), record=bool(ok),
                       **(dict(refused=True, refusal=why) if why else {}))
            self.write()
            if not ok:
                what = m2.baseline_problem(rec) if why is None else f"the monitor refused to run: {why['reason']}"
                moved = f"; {SOURCE_REMEDY}" if moved_sources(why) else ""
                raise Halt(f"{self.job}: the R0L baseline could not be made (the monitor child ended with code "
                           f"{code}; {what}): rule M1: the fault is repaired and all its "
                           f"episodes are run again before the run's first rollout{moved}")
            if was is not None:  # (the record run.json names was lost: the run's checks were read against it)
                same = baseline_same(was, rec)
                if not same["equal"]:
                    kept = "r0l.differs.s%d.json" % self.k
                    os.replace(path, path.with_name(kept))
                    raise Halt(f"{self.job}: R0L's baseline was lost and, made again, is not the one the run's "
                               f"checks were read against ({'; '.join(same['differ'])}): nothing is replaced, the "
                               f"new record is kept as monitor/{kept}, and the run goes to the lead")
                remade.append(dict(session=self.k, was_sha256=was.get("sha256"), **same))
                self.say("the R0L baseline was lost and is made again: equal to the one run.json names")
        self.baseline = rec
        d["baseline"] = baseline_entry(rec)
        if remade:
            d["baseline"]["remade"] = remade
        self.write()

    def _expect_baseline(self) -> None:
        """``EXPECT_BASELINE_RULE``: the run's own R0L baseline against the committed one, before the first rollout
        (and again at every later session); recorded in ``run.json`` ``baseline_expected``; not equal: ``Halt``."""
        if self.expected is None:
            return
        m2, d, exp = _monitor2(), self.doc, self.expected
        eq = m2.baseline_equal(self.baseline, exp["record"])
        differ = list(eq["differ"])
        if not eq["digests_compared"]:
            differ.append("the state digests were not compared in every episode that ran (a record holds none, or "
                          "none that is a non-empty string, or no episode ran)")
        entry = dict(session=self.k, path=exp["path"], sha256=exp["sha256"], record_sha256=exp["record_sha256"],
                     baseline_sha256=self.baseline.get("sha256"), equal=not differ, episodes=eq["episodes"],
                     digests_compared=eq["digests_compared"], differ=differ, at=s2io.now_utc())
        d.setdefault("baseline_expected", []).append(entry)
        self.write()
        if differ:
            raise Halt(f"{self.job}: the run's R0L baseline (monitor/r0l.json) is not the committed one "
                       f"({exp['path']}): {'; '.join(differ)}: rule M3: S2-4's own baseline must equal it episode for "
                       "episode (outcome, survival and, in every episode that ran, state digest), else the run "
                       "stops before its first rollout; no step is trained and the run goes to the lead")
        self.say(f"R0L's baseline equals the committed one ({exp['path']}): rule M3")

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
            rec = _read_record(path)
            if not isinstance(rec, dict) or rec.get("sha256") != e.get("record_sha256"):
                raise TrainError(f"{path}: the record of the check at {e['at_steps']} steps is missing, cannot be "
                                 "read or is not the one run.json names: the rules read records")
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
            doc = _read_record(path)  # (None too for a file that cannot be read: a copy that was cut short)
            return doc if m2.is_record(doc, hashes, pairs) else None

        rec = usable()
        # (rule M1's two: closed, no record, and the monitor ran: a refusal is no attempt, ``RULES["refusal"]``)
        closed = sum(1 for a in entry["attempts"] if a.get("closed_at") is not None and not a.get("record")
                     and not a.get("refused"))
        while rec is None and closed < m2.ATTEMPTS:
            after_halt = sum(1 for a in entry["attempts"] if a.get("closed_at") is None) >= m2.ATTEMPTS
            if after_halt and (not self.retry_check or self.retry_used):
                raise Halt(f"{self.job}: two attempts at the check of {n} steps were never closed (the session died "
                           "while the check ran, twice): the run halts and goes to the lead (rule M1); with the "
                           "fault repaired, a resume with --retry-check lets the check run once more")
            self.retry_used = self.retry_used or after_halt
            if path.is_file():  # (no record: of other files, a checkpoint saved again after a resume; or cut)
                os.replace(path, path.with_name("monitor_%d.discarded.s%d.json" % (n, self.k)))
            self.say(f"check at {n} steps (attempt {len(entry['attempts']) + 1})")
            att = dict(session=self.k, started_at=s2io.now_utc(), checkpoint_sha256=hashes, closed_at=None,
                       exit_code=None, record=None, **(dict(after_halt=True) if after_halt else {}))
            entry["attempts"].append(att)
            self.write()  # (logged before the monitor starts)
            code = self._child(["run", "--job", self.jd, "--steps", n, "--body", self.body, "--set", self.set,
                                "--k", m2.K], att)
            rec = usable()
            why = None if rec is not None else self._refusal(code, n)
            att.update(closed_at=s2io.now_utc(), exit_code=int(code), record=rec is not None,
                       **(dict(refused=True, refusal=why) if why else {}))
            self.write()
            if why is not None:  # (no attempt: it is not counted, nothing is read, and the run goes to the lead)
                # (under --retry-check the two attempts never closed stay: a plain resume would halt on them)
                again = ("a resume with --retry-check runs the check again (the refused attempt was the one "
                         "--retry-check let run after two attempts at this check were never closed)" if after_halt
                         else "a resume runs the check again")
                moved = f"; {SOURCE_REMEDY}" if moved_sources(why) else ""  # (SOURCE_ALLOWANCE_RULE)
                raise Halt(f"{self.job}: the monitor refused to run on the check of {n} steps ({why['reason']}): a "
                           "monitor that refuses to run is no attempt and does not count toward the two after "
                           "which a check counts as without a gain (rule M1; the lead's ruling of 16 Nov 2026): the "
                           f"run halts and goes to the lead; with the fault repaired, {again}{moved}")
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
                mean_clock_factor=mon.get("mean_clock_factor"), gain=(mon.get("gain") or {}).get("gain"),
                counted_without_record=bool(mon.get("counted_without_record"))),
            # (rule M1: "the lead is told"; a halt is one of rule M5's endings until the lead resumes the run)
            checks_without_record=[e["at_steps"] for e in d["monitor"] if e.get("counted_without_record")],
            halt=d.get("halt"),
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


BASELINE_EPISODE_KEYS = ("condition", "seed", "outcome", "survival_s", "state_digest", "digest")


def baseline_entry(rec: dict) -> dict:
    """What ``run.json`` ``baseline`` holds of R0L's record (``BASELINE_RULE``)."""
    m2 = _monitor2()
    return dict(record="monitor/r0l.json", sha256=rec["sha256"], row=rec.get("row"),
                **{k: rec.get(k) for k in m2.SUMMARY_KEYS},
                episodes=[{k: e[k] for k in BASELINE_EPISODE_KEYS if k in e} for e in rec["episodes"]])


def baseline_same(was: dict, rec: dict) -> dict:
    """A baseline record made again (``rec``) against what ``run.json`` holds of the lost one (``was``): the
    summary, and episode for episode the outcome, the survival and, where both hold one, the state digest
    (``monitor2.baseline_equal``).  ``{equal, differ, episodes_compared, digests_compared}``."""
    m2 = _monitor2()
    differ = [f"{k}: {was.get(k)!r} -> {rec.get(k)!r}" for k in m2.SUMMARY_KEYS
              if k in was and was.get(k) != s2io.plain(rec.get(k))]
    eps = None
    if was.get("episodes") is not None:
        eps = m2.baseline_equal(rec, dict(episodes=was["episodes"]))
        differ += eps["differ"]
    return dict(equal=not differ, differ=differ, episodes_compared=None if eps is None else eps["episodes"],
                digests_compared=None if eps is None else eps["digests_compared"])


def expected_baseline(path) -> dict | None:
    """The committed R0L baseline of ``--expect-baseline PATH`` (``EXPECT_BASELINE_RULE``): ``{path, sha256,
    record_sha256, record}``, ``path`` relative to the study directory where it lies under it (``s2io.portable``),
    ``sha256`` the file's.  None for None.  ``Refused`` when the file is missing, cannot be read or is no baseline
    record (``monitor2.baseline_problem``)."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise Refused(f"--expect-baseline {p}: no such file (rule M3: the committed R0L baseline)")
    try:
        doc = s2io.read_json(p)
    except (OSError, ValueError) as exc:
        raise Refused(f"--expect-baseline {p}: the file cannot be read ({type(exc).__name__})") from exc
    why = _monitor2().baseline_problem(doc)
    if why is not None:
        raise Refused(f"--expect-baseline {p}: {why}: it is no committed R0L baseline (rule M3)")
    return dict(path=s2io.portable(str(p.resolve())), sha256=s2io.sha256_file(p), record_sha256=doc.get("sha256"),
                record=doc)


# ------------------------------------------------------------------------------------------------
# Probes (rules O2, O3, O4, K8): each writes one record
# ------------------------------------------------------------------------------------------------
def _record_path(out, name: str) -> Path:
    out = Path(out)
    return out if out.suffix == ".json" else out / name


def _stamp(path: Path) -> dict:
    """The library versions, the checkout's state (the record itself left out) and the time."""
    from designlab.stage2 import s2load

    return dict(versions=s2load.versions(), provenance=s2io.git_state(own_outputs=[str(path.resolve())]),
                at=s2io.now_utc())


def _write_record(path: Path, doc: dict) -> dict:
    doc = s2io.plain(dict(doc, **_stamp(path)))
    doc["sha256"] = s2io.sha256_json(doc)
    s2io.write_json(path, doc)
    return doc


def _record_to_add_to(path: Path, schema: str, same: dict, what: str) -> dict | None:
    """The record a probe adds its bodies to (None: there is none yet).  ``Refused`` when the file is no record of
    ``schema`` that hashes to its own ``sha256``, or holds another value for one of ``same``'s settings: nothing is
    added to such a file and it is not replaced."""
    if not path.exists():
        return None
    doc = _read_record(path)
    if (not isinstance(doc, dict) or doc.get("schema") != schema
            or doc.get("sha256") != s2io.sha256_json({k: v for k, v in doc.items() if k != "sha256"})):
        raise Refused(f"{path}: not a {what} record that hashes to its own sha256: nothing is added to it and it is "
                      "not replaced (move it aside, or name another --out)")
    diff = [f"{k}: {doc.get(k)!r} -> {v!r}" for k, v in s2io.plain(same).items() if doc.get(k) != v]
    if diff:
        raise Refused(f"{path}: the record holds a {what} of other settings ({'; '.join(diff)}): nothing is added "
                      "to it and it is not replaced (name another --out)")
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
             parent=None, mode="R5", config=None) -> dict:
    """Rule O3's learning-rate probe (``LR_PROBE_RULE``), or with ``parent = (job directory, steps)`` rule K8's
    from a parent's checkpoint (``FINE_TUNE_RULE``: one body, the parent's own sigma; the rates tried are the
    parent's schedule value p at its step, then the listed rates below p).  The record (``lr_probe.json`` under
    ``out``, or ``out`` itself when it names a ``.json``): per body and head the rates tried with every update of
    every cell and the rate that passed; ``rates_by_body`` (rule O3 as settled on 15 Nov: the smallest passing
    rate over the probed heads of a body; None where a probed head has none); ``value`` (the smallest over
    everything the record holds).

    Rule O3's probe is a fresh R5 model on envs with rule O4's filter (``FILTER_NEED_RULE``), and each body and
    head is probed once: the bodies of this call are added to the record ``out`` already holds, which keeps its
    earlier bodies as they were; a body the record holds is refused, and so is a record of other settings.  Rule
    K8's probe runs on the fine-tune's own envs: ``mode`` and ``config`` (env config fields for every env) are the
    fine-tune's, and its record is its job's own (a record of anything else at ``out`` is not replaced)."""
    import torch

    from designlab.stage2 import vecpool

    require_hashseed()
    torch.set_num_threads(1)
    bodies = list(dict.fromkeys(str(b) for b in bodies))
    if mode not in MODES:
        raise Refused(f"mode {mode!r}: choose from {', '.join(MODES)}")
    path = _record_path(out, "lr_probe.json")
    w = min(N_ENVS, vcpus()) if workers is None else max(0, min(int(workers), N_ENVS))
    n_steps = N_STEPS
    par, have = None, {}
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
        if path.exists():  # (a fine-tune cut before its first rollout makes its probe again; nothing else is replaced)
            old = _read_record(path)
            mine = {k: par[k] for k in ("job", "run_uuid", "steps")}
            if not isinstance(old, dict) or {k: (old.get("parent") or {}).get(k) for k in mine} != mine:
                raise Refused(f"{path}: not the record of this parent's probe ({par['job']}:{par['steps']}): it is "
                              "not replaced (name another --out)")
    else:
        if mode != "R5":
            raise Refused(f"rule O3's probe is a fresh R5 model (mode {mode!r} is a fine-tune's: pass its parent)")
        same = dict(set=set_name, rates=[float(r) for r in rates], sigmas=list(sigmas), updates=int(updates),
                    seed=PROBE_SEED, n_envs=N_ENVS, n_steps=int(n_steps), parent=None)
        old = _record_to_add_to(path, LR_PROBE_SCHEMA, same, "learning-rate probe")
        have = dict((old or {}).get("bodies") or {})
        again = [b for b in bodies if b in have]
        if again:
            raise Refused(f"{path}: already probed: {', '.join(again)} (rule O3: the probe is run once per body and "
                          "head, and is not run again)")
    vals = {body: _validation(body, set_name) for body in bodies}
    if parent is None and set_name != S1_SET:  # (before any env is built: nothing is probed when one is refused)
        for body in bodies:
            _need_filter(f"probe-lr {body}", config, vals[body])
    out_bodies = {}
    for body in bodies:
        val = vals[body]
        over = _overrides(config, val)
        cfgs = _env_configs(mode, body, set_name, over)
        venv = vecpool.PoolVecEnv(_env_specs(mode, cfgs), min(w, len(cfgs)), worker_cmd=[sys.executable, *WORKER_CMD])
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
        out_bodies[body] = dict(tried=tried, passed=passed, mode=mode, config=s2io.plain(dict(config or {})),
                                floors=[c.get("floor") for c in cfgs], filter_hz=cfgs[0].get("filter_hz"),
                                set_freeze=cfgs[0].get("set_freeze"), task_names=list(desc["tasks"]),
                                validation=val, probed=dict(_stamp(path), workers=w))
    rows = dict(have, **out_bodies)  # (the record's earlier bodies as they were, then this call's)
    by_body: dict = {}
    for body, row in rows.items():
        g = by_body.setdefault(base_body(body), dict(rate=None, heads={}))
        g["heads"][body] = row["passed"]
    for g in by_body.values():
        got = list(g["heads"].values())
        g["rate"] = None if any(r is None for r in got) else min(got)
    all_passed = [r["passed"] for r in rows.values()]
    doc = dict(schema=LR_PROBE_SCHEMA, rule=FINE_TUNE_RULE if parent is not None else LR_PROBE_RULE, set=set_name,
               mode=mode, rates=[float(r) for r in rates], sigmas=list(sigmas), updates=int(updates), seed=PROBE_SEED,
               n_envs=N_ENVS, n_steps=int(n_steps), workers=w, bodies=rows, rates_by_body=by_body,
               value=None if any(r is None for r in all_passed) else min(all_passed), parent=par,
               added=list(out_bodies), filter_rule=FILTER_NEED_RULE,
               note="rule O3 as settled on 15 Nov 2026 uses rates_by_body (one rate per body, over every head the "
                    "record holds); value is the smallest over everything the record holds; each body's entry "
                    "holds when and on what it was probed (probed); the versions, provenance and time at the top "
                    "are the last call's, which added the bodies under added")
    return _write_record(path, doc)


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
    require_hashseed()
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
    0.9 x the first's, else the first; None while that body is not measured.  One record (``filter.json``): each
    body is measured once, and the bodies of this call are added to the record ``out`` already holds, which keeps
    its earlier bodies and its choice; a body the record holds is refused, and so is a record of other
    settings."""
    require_hashseed()
    bodies = list(dict.fromkeys(str(b) for b in bodies))
    path = _record_path(out, "filter.json")
    same = dict(set=set_name, cutoffs=[float(c) for c in cutoffs], episode_seeds=list(PROBE_EPISODES))
    old = _record_to_add_to(path, FILTER_SCHEMA, same, "filter measurement")
    have = dict((old or {}).get("bodies") or {})
    again = [b for b in bodies if b in have]
    if again:
        raise Refused(f"{path}: already measured: {', '.join(again)} (rule O4: measured once; it is not measured "
                      "again)")
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
                                role="the rule's body" if body == FILTER_BODY else "reported; decides nothing",
                                measured=_stamp(path))
    rows_by_body = dict(have, **out_bodies)  # (the record's earlier bodies as they were, then this call's)
    choice = (old or {}).get("choice")  # (measured once: a later call keeps it)
    if choice is None and FILTER_BODY in rows_by_body:
        r = rows_by_body[FILTER_BODY]["ratio"]
        choice = dict(body=FILTER_BODY, ratio=r, keep=FILTER_KEEP,
                      filter_hz=float(cutoffs[-1] if r is not None and r >= FILTER_KEEP else cutoffs[0]))
    doc = dict(schema=FILTER_SCHEMA, rule=FILTER_RULE, set=set_name, cutoffs=[float(c) for c in cutoffs],
               episode_seeds=list(PROBE_EPISODES), bodies=rows_by_body, choice=choice, added=list(out_bodies),
               note="each body's entry holds when it was measured (measured); the versions, provenance and time at "
                    "the top are the last call's, which added the bodies under added")
    return _write_record(path, doc)


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
    a.add_argument("--allow-source-change", action="store_true",
                   help="resume on moved sources, flagged; also passed to the policy row in the monitor child "
                        f"({ALLOW_SOURCE_ENV}=1)")
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
    a.add_argument("--expect-baseline", default=None, metavar="PATH",
                   help="the committed R0L baseline the run's own must equal before its first rollout (rule M3)")
    for name, text in (("probe-lr", "rule O3's learning-rate probe"), ("probe-sigma", "rule O2's sigma probe"),
                       ("filter", "rule O4's filter measurement")):
        a = sub.add_parser(name, help=text)
        a.add_argument("--bodies", required=True, help="comma-separated")
        a.add_argument("--set", default="V6")
        a.add_argument("--out", required=True, help="the record (a .json path) or its directory")
        if name == "probe-lr":
            a.add_argument("--workers", type=int, default=None)
            a.add_argument("--parent", default=None, metavar="JOBDIR:STEPS", help="rule K8's probe")
            a.add_argument("--mode", default="R5", choices=MODES, help="with --parent: the fine-tune's mode")
            a.add_argument("--config", default=None, help="a JSON object of env config fields for every env "
                           "(with --parent: the fine-tune's)")
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
        require_hashseed()
        if args.cmd in ("probe-lr", "probe-sigma", "filter"):
            if args.cmd == "probe-lr":
                doc = probe_lr(_list(args.bodies), args.set, args.out, workers=args.workers, parent=args.parent,
                               mode=args.mode, config=json.loads(args.config) if args.config else None)
                print(f"lr probe on set {args.set}: {doc['rates_by_body']}", flush=True)
            elif args.cmd == "probe-sigma":
                doc = probe_sigma(_list(args.bodies), args.set, args.out)
                print(f"sigma probe on set {args.set}: written", flush=True)
            else:
                doc = filter_choice(_list(args.bodies), args.set, args.out)
                print(f"filter on set {args.set}: {doc['choice']}", flush=True)
            return EXIT_DONE
        doc = train(args.mode, args.body, args.seed, set_name=args.set, out=args.out, job=args.job, steps=args.steps,
                    resume=args.resume, workers=args.workers, monitor_every=args.monitor_every,
                    monitor_also_k=tuple(int(k) for k in _list(args.monitor_also_k)), lr=args.lr,
                    config=json.loads(args.config) if args.config else None, gates=not args.no_gates,
                    allow_version_change=args.allow_version_change, allow_source_change=args.allow_source_change,
                    parent=args.from_checkpoint, tiny=args.tiny, run_id=args.run_id,
                    retry_check=args.retry_check, expect_baseline=args.expect_baseline,
                    log=lambda *x: print(*x, flush=True))
    except TrainError as exc:
        print(f"train2 {args.cmd}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return exc.code
    return {"stopped_g8a": EXIT_G8A, "partial": EXIT_FAILED}.get(doc["status"], EXIT_DONE)


__all__ = ["ALLOW_SOURCE_ENV", "BOUNDARY_RULE", "CHECKPOINT_EVERY", "EXIT_DONE", "EXIT_FAILED", "EXIT_G8A", "EXIT_HALT",
           "EXIT_REFUSED", "EXPECT_BASELINE_RULE", "Halt", "MONITOR_EVERY", "N_ENVS", "N_STEPS", "Refused",
           "RESUME_RULE", "SCHEMA",
           "SOURCE_ALLOWANCE_RULE", "SOURCE_FILES", "TINY", "TrainError", "allowed_sources", "base_body",
           "boundary_seed", "count_optimizer_steps", "expected_baseline", "filter_choice",
           "filter_sigma", "is_boundary", "job_name", "kl_share", "main", "monitor_env", "moved_sources", "preflight",
           "preflight_env",
           "probe_lr", "probe_sigma", "read_updates", "require_hashseed", "sources", "train"]

if __name__ == "__main__":
    from designlab.stage2 import train2 as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
