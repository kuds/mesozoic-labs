"""Track B's gates and validation records (round C design D17, D18, D22, D23; sections 2.2, 3.7, 4, 6; Track B
rules Z1, Z2, K3, O2-O4, X1 of ``STAGE2_TRACKB_PLAN.md``; the lead's rulings ``ROUND_C_RULINGS_16NOV.md`` and
``TRACKB_RULINGS6_16NOV.md`` ruling 2).  Simulation only: every record here is a check of the tooling and of the
references, kinematic or in the simulator; none is a measurement on hardware.

    python -m designlab.stage2.gates2 g8a --body B --set S [--record PATH] [--conditions K,...] [--seeds 3042,3043]
           [--delays 0,5] [--cut-s X] [--set-root DIR] [--g9-ref PATH]
    python -m designlab.stage2.gates2 g9-ref --body B --set S [--record PATH] [--set-root DIR] [--refcheck PATH]
    python -m designlab.stage2.gates2 g9-run --body B --set S --bar-out DIR [--stage1b-reread PATH] [--record PATH]
           [--block seeds3042] [--set-root DIR]
    python -m designlab.stage2.gates2 rates --body B --set S [--floors hard,robot_floor_carpet] [--workers 1,2,3,4]
           [--steps 16384] [--record PATH] [--g9-ref PATH]
    python -m designlab.stage2.gates2 kill-resume --body B --set S --out DIR [--record PATH] [--tiny] [--lr X]
           [--config JSON] [--g9-ref PATH]
    python -m designlab.stage2.gates2 gl-start [--out DIR] [--record PATH]
    python -m designlab.stage2.gates2 r6-start --body B --set S [--record PATH] [--out DIR] [--workers 2] [--g9-ref PATH]
    python -m designlab.stage2.gates2 dev-episode --file PATH --step NAME --command TEXT --set S --body B
           --what ROWS_OR_MODE --count N --reason TEXT [--state listed|done]
    python -m designlab.stage2.gates2 _train designlab.stage2.train2 JSON      # kill-resume's child (internal)

Every gate writes one JSON record (``--record``; without it nothing is written and the record is printed in
brief) holding its rule text, the hashes of what it read, the sources, the library versions, the commit, whether
the checkout is dirty (this command's own outputs left out), the time and its own ``sha256`` (``s2io.sha256_json``
of the record without that key), and returns it.  Exit code 0 when the gate passes, 1 when it fails, 2 on an error
(a refusal included: nothing was judged).

**What each gate is** (the records' ``rule`` texts are ``RULES``):

* ``g8a`` (rule Z1): ``g8a.bar_form`` on ``G8A_CONDITIONS`` x seeds 3042, 3043 x delays 0 and 5 physics steps; a
  condition the set holds no reference for is ``not_run`` with the set's reason.  On the Stage 1 view (``S1``) the
  development form ``g8a.stock`` on ``monitor2.S1_CONDITIONS``.  A forced difference names its condition, seed,
  delay and first differing physics step (``differences``).
* ``g9_ref`` (rule Z2's G9-ref: parts (a), (b), (c), (f); kinematic only, **run before any closed-loop episode on
  the set**): (a) round A's ``bar refcheck`` record cited by path and sha256 (the one the set's latest freeze cites,
  else ``refcheck.json`` in the set's directory when its files are the freeze's, else ``bar.refcheck`` is run and its
  record written beside this one); (b) each rung the look ladder holds for the body against
  ``stage2/look/metrics.json`` and ``stage2/look/L1_FREEZE.json``, every other walk finite with at least 4 steps,
  an ``-X`` body against the plain body's playbacks, ``V6noF5`` against the same body's V6 playbacks, and every walk
  and turn item's window with its steps, lift-offs and its reference toe-off with its side of 0.02 rad (the lead's
  ruling 7 of 16 Nov 2026); (c) the four degraded playbacks of ``lookread.controls``, the fifth of
  ``TRACKB_RULINGS6_16NOV.md`` ruling 2 (the tail scaled to a tenth about its mean: ``tail_tenth``) and the 3-step
  cut; (f) the ``setscreen`` record cited for every reference no reference check plays, and every task script of
  ``r5lib.Library``: the joint targets' jump at every join, at each joint no larger than the largest change between
  two consecutive frames of the walk it leaves (the lead's ruling 15 of 17 Nov 2026, a change after a result), 2 s
  of the tracker's zero-action targets finite.  G9-ref fails while a failing item of (a) names a reference that is
  still available in the set (the lead's ruling 14 of 17 Nov 2026); its record names those references.
* ``g9_run`` (Z2's G9-run, parts (d), (e)): the ``R0L@10ms`` companion records ``lookbar.collect`` wrote and the
  bar's 10 ms records under ``--bar-out``; round A's ``bar reread`` record of Stage 1b's 10 ms turn episodes cited
  for the 60 cm body and head.
* ``rates``: ``r5env.throughput`` per floor in this process, then a trainer-style rollout loop (a fresh PPO policy
  acting through ``VecNormalize`` on ``vecpool.PoolVecEnv``, no update) with 8 envs at each worker count.
* ``kill_resume`` (rule K3): two runs of ``train2`` under ``--out``, the second killed by ``os.killpg(..., SIGKILL)``
  when its ``progress.json`` reads 81,920 steps and resumed with 3 workers; the two final checkpoints compared.
* ``gl_start``: a ``train2 preflight`` child started with ``MUJOCO_GL=osmesa``.
* ``r6_start``: an ``R6Env`` stepped with zero and random actions and from a mid start; a 2-rollout tiny R6 run.

**Rule X1.**  ``g8a``, ``rates``, ``kill_resume`` and ``r6_start`` run closed-loop episodes: on a look set
(``LOOK_SETS``) each is refused unless ``stage2/tb/gates/<set>/<body>/g9_ref.json`` (or ``--g9-ref``) is a passed
G9-ref record of that set, body and the set's latest freeze (``require_g9_ref``).  ``g9_ref`` and ``g9_run`` run
no episode.  The chain ``stage2/tb/run_trackb.sh`` lists each closed-loop step on a look set that is no record in
``stage2/tb/dev_episodes.json`` before it runs (``dev-episode``).

Readings this module makes of rules that leave a choice are marked in its rule texts after ``MODULE_READING``.
Heavy modules (the bar, the env, torch) are imported inside the functions that need them.
"""

from __future__ import annotations

from designlab.stage2 import tbenv

tbenv.assign_gl("disable")

import os  # noqa: E402
import sys  # noqa: E402

# (as the trainer: ``s2io.git_state`` runs ``git status``; a SIGKILL of a child's process group must leave no
# ``index.lock`` in the checkout, and every child inherits the variable)
os.environ.setdefault("GIT_OPTIONAL_LOCKS", "0")

if __name__ == "__main__":  # before any numpy import (round A's module skeleton)
    for _v in tbenv.THREAD_VARS:
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import importlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from designlab.stage2 import s2io  # noqa: E402

EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
TB = s2io.STUDY / "stage2" / "tb"
GATES = TB / "gates"
VALIDATION = TB / "validation"
WORK = TB / "work"
LOOK = s2io.STUDY / "stage2" / "look"
#: The look sets (``lookbar.LOOK_SETS``; a test compares them): a closed-loop gate there needs a passed G9-ref.
LOOK_SETS = ("V6", "V6noF5")
#: The set of the look sets whose walks carry no toe-off (rule L11): G9-ref's form of rule Z2 for it.
NO_TOE_OFF_SET = "V6noF5"
STOCK_SET = "S1"
#: Rule Z1's ten condition keys: the bar's eight gated mu 0.9 conditions, then ``walk/v0.4_mu0.54`` and
#: ``walk/nominal`` (``g8a.conditions`` computes the same list from a set's records; a test compares them).
G8A_CONDITIONS = ("walk/v0.4", "turn/q90_left", "turn/q90_right", "nudge/forward", "nudge/backward", "nudge/left",
                  "nudge/right", "stop/two_step", "walk/v0.4_mu0.54", "walk/nominal")
G8A_SEEDS = (3042, 3043)
G8A_DELAYS = (0, 5)
#: G9-ref (b): the reference times a walk alone is played over (a walk item's 10 s on the nominal clock, 2 s in).
WALK_TA_REF = 2.0
WALK_WINDOW_S = 10.0
#: G9-ref (b): the rungs of the ladder by the bar's walk names, where an entry names no ladder twin.
RUNG_WALKS = {"nominal": "nominal", "v040": "v0.4"}
#: G9-ref (b): a rung of these the set does not hold fails (b) (rule Z2: a failing start, nominal or v0.4 blocks
#: the body); any other rung the set does not hold is listed.
REQUIRED_RUNGS = ("nominal", "v040")
#: G9-ref (c): the fifth degraded playback's scale of the tail about its mean (``TRACKB_RULINGS6_16NOV.md`` 2).
TAIL_SCALE = 0.1
#: G9-ref (c): the walks whose playbacks the controls are made on.
CONTROL_WALKS = ("nominal", "v0.4")
#: G9-ref (f): the seconds of the zero action played.  (A join's limit is no constant: at each joint, the largest
#: change between two consecutive frames of the walk it leaves; the lead's ruling 15 of 17 Nov 2026.)
ZERO_ACTION_S = 2.0
#: G9-ref (a), what follows a failure: a failing item that plays one of these blocks the body and head.
BLOCKING_ITEMS = ("walk/nominal", "walk/v0.4")
#: G9-run: the row and the delay.
R0L_ROW = "R0L@10ms"
RUN_MS = 10
STAGE1B_BODIES = ("60-N-T-10", "60-N-T-10-X")
STAGE1B_BODY = "60-N-T-10"
#: Rule K3's settings, and a tiny form of them on the Stage 1 view (a smoke of the command; no record of K3).
K3 = dict(mode="R5", seed=42, steps=131_072, checkpoint_every=32_768, monitor_every=131_072, kill_at=81_920,
          workers=(2, 3), tiny=False, gates=False, n_envs=8, n_steps=1024)
K3_TINY = dict(K3, steps=4096, checkpoint_every=1024, monitor_every=4096, kill_at=2560, tiny=True, n_steps=64)
#: Rule K3's body and set; a record of other settings, body or set is a check of the command and says so.
K3_ON = ("75-M-H-10", "V6")
K3_CHECK = "a check of the command (not rule K3's settings, body and set): no record of K3"
TRAINER = "designlab.stage2.train2"
CHILD_TIMEOUT_S = 4 * 3600.0
GROUP_WAIT_S = 30.0
POLL_S = 0.02
#: ``rates``: the load average under which a measurement is not provisional (lesson 10).
IDLE_LOAD = 0.5
RATE_SIGMA = 0.1
#: ``r6_start``: the steps of each kind, the training start's rollouts (tiny: 8 x 64 steps each).
R6_STEPS = 200
R6_ROLLOUTS = 2
SCHEMAS = dict(g8a="s2-gate-g8a-1", g9_ref="s2-gate-g9-ref-1", g9_run="s2-gate-g9-run-1", rates="s2-val-rates-1",
               kill_resume="s2-val-kill-resume-1", gl_start="s2-val-gl-start-1", r6_start="s2-val-r6-start-1",
               dev_episodes="s2-dev-episodes-1")

#: The mark that stands, in each rule text, between the rule's wording and this module's own reading of it.
MODULE_READING = "This module's own reading, which is not the rule's wording: "

RULES = dict(
    z1=(
        "STAGE2_TRACKB_PLAN rule Z1: for a set and a body or head the R5 env with the zero action (head neutral, "
        "nominal plant, the condition's friction, stance start with the row's start seed and Stage 1's start "
        "noise, the condition's push, the delay through the bar's own wrapper, interference termination off) "
        "against the bar's row R0@0ms and R0@10ms, on the eight gated mu 0.9 conditions the set has references "
        "for, walk/v0.4_mu0.54 and walk/nominal, panel seeds 3042 and 3043 (40 episodes).  Compared: sha256 of "
        "qpos, qvel, ctrl, sensordata after every physics step the row ran; the same end.  Pass: every episode "
        "identical (identical == of; compared_steps == reference_steps in every episode).  "
        + MODULE_READING +
        "the gate's form is the ten conditions x two seeds x two delays with no cut (complete); a record of fewer "
        "(--conditions, --seeds, --delays, --cut-s) is a check of the command and says so (complete false): its "
        "exit code is that of the episodes it ran, and it is refused (nothing runs) when its record would lie under "
        "stage2/tb/gates/, where only the gate's form is written (Z1's check reads that record).  On the Stage 1 "
        "view the development form g8a.stock on monitor2.S1_CONDITIONS stands in (no bar set exists there; it is "
        "never the gate's form)."),
    z2_ref=(
        "STAGE2_TRACKB_PLAN rule Z2, G9-ref (kinematic only; before the first closed-loop episode on the set and "
        "body): (a) every frozen reference played kinematically at the commanded clock passes its own item's "
        "readings, the quick turn also reaching 80 degrees at the slowest clock (the bar's reference check, "
        "cited); (b) on each rung the look ladder holds for the body the walk's playback agrees with the ladder's "
        "numbers (bob within 5 %, head pitch within 0.1 degree, tail phase within 0.01 cycle, toe-off within 0.02 "
        "rad and within 10 % of the ladder's number) and the set's look-format hash for that rung equals "
        "L1_FREEZE.json's; on a rung the ladder does not hold the four numbers are finite with at least 4 steps "
        "in the window; on an -X body each walk's playback agrees with the plain body's playback of the same rung "
        "within the same limits; for every walk and turn item the playback of its window (U2) holds at least 4 "
        "steps and 4 lift-offs; on V6noF5 no hash and no toe-off comparison (the toe-off 'not read'), bob, head "
        "and tail against the same body's V6 playback; (c) four degraded playbacks each fail exactly their term "
        "(bob x 0.5; a 3 degree neck sinusoid; the tail 0.2 cycle late; flat feet, on V6 only), the unchanged "
        "playback passes all four with ratio 1, a playback cut to 3 steps reads too_few_steps, and (the lead's "
        "ruling 2 of TRACKB_RULINGS6_16NOV) the tail scaled to a tenth about its mean fails the tail term and no "
        "other; (f) each reference no bar item plays has its screen record cited, every task's track is "
        "continuous at its joins and plays 2 s with finite targets: [changed after a result: the lead's ruling 15 "
        "of 17 Nov 2026 (ROUND_C_RULINGS_17NOV.md), made after the numbers the joins gave on Stage 1's own arrays "
        "(up to 1.1e-4 rad) and on V0 (up to 5.3e-5 rad), seen on 17 Nov, where nothing is wrong, and before any "
        "look set was frozen and before any closed-loop episode on one; as first written: joint targets within "
        "1e-6 rad] a join's jump at a joint does not exceed the largest change between two consecutive frames of "
        "the walk it leaves, at that joint (a join never adds a step larger than the walk's own motion); the gate "
        "records both numbers per join and joint.  If (a) fails for a reference it is treated as not available "
        "and goes to the lead; [the lead's ruling 14 of 17 Nov 2026] G9-ref fails while a failing item's "
        "reference is still available in the set, and its record names those references: the lead marks them "
        "not available, freezes the set again and runs G9-ref again; a failing start, nominal or v0.4 "
        "blocks the body and head; if (b), (c) or (f) fails no closed-loop episode runs on that set and body.  "
        "The gates list each item's reference toe-off and its side of 0.02 rad (the lead's ruling 7 of 16 Nov "
        "2026).  "
        + MODULE_READING +
        "G9-ref reads a frozen set (any set but the Stage 1 view): an unfrozen set is refused, and every file "
        "parts (b) and (c) play, and the comparand's walks, is checked against the latest freeze "
        "(setio.read_inputs, require_frozen) before anything is played (a file that differs is refused); the "
        "record lists them with their sha256 and those (f)'s library read (inputs).  "
        "(a) cites the refcheck record the set's latest freeze cites when its sha256 is the cited one, else the "
        "set directory's refcheck.json when every file it read has the latest freeze's sha256, else bar.refcheck "
        "is run; whichever record is read, one given by path included, must be of the set and body and every file "
        "it read must have the latest freeze's sha256, else (a) is not read and blocks.  An item that fails names "
        "its own reference (a walk item its walk, a turn or stop item its move) and a failing walk/nominal or "
        "walk/v0.4 item blocks; the start is played by every item and alone by none, so a failing start is read "
        "through the walk/nominal item (the start and the nominal walk alone) and blocks through it.  A failing "
        "item's reference is still available while its entry is usable as the env and the bar read it "
        "(r5lib.why_not: an entry with a file, not recorded as not available) or the set's latest freeze lists "
        "it; one that cannot be named counts as still available (still_available).  (b) plays "
        "each walk alone (whole windows of its "
        "file) over the reference times [2.0, 2.0 + 10 x 1.05] s through lookread.playback; a rung's walk is the "
        "entry whose ladder_twin names the rung, else nominal and v0.4 by name; its numbers are compared with "
        "the ladder rung its entry names as its twin (the rung of L1_FREEZE where it names none); a rung of the "
        "ladder the set does not hold fails (b) for nominal and v040 and is listed for any other.  An -X body's "
        "and V6noF5's walks are compared by name with the comparand set's walks of the same name (the comparand's "
        "files cited by path and their own sha256).  The items "
        "are the bar's walk and turn conditions at mu 0.9 that the set holds (a mu 0.54 item plays the same "
        "track).  (c) is made on the playbacks of the walks nominal and v0.4, with lookread.controls' window "
        "(the lead of lookread.LEAD_S before it).  (f): a reference no reference check plays is one that no "
        "item of the cited refcheck record that was done names among its files; its block in the set's "
        "screen.json must exist with the entry's hash (a walk: its look-format file).  A join's jump at a joint "
        "is the |difference| of that joint's target between segment i at its own end (its own formula at its "
        "full duration) and segment i + 1's first frame.  The walk it leaves is the walk (a periodic segment) "
        "segment i plays; where segment i is no walk (the start, a move or a hold), the walk the track played last "
        "before the join (the walk it left through that move), else the first walk it plays after the join; a "
        "track that plays no walk holds its joins to 0 at every joint.  A walk's largest change at a joint is "
        "over every two consecutive frames of its file's window as the track plays them, the last frame to the "
        "first included; 'does not exceed' is at most (<=).  The zero-action targets are R5Tracker.zero_command "
        "at every physics step of 2 s of simulated time at the commanded clock."),
    z2_run=(
        "STAGE2_TRACKB_PLAN rule Z2, G9-run (after R0L's records of that set and body): (d) every episode of the "
        "body's R0L@10ms records holds every reading of its item and, where its window was reached, a finite look "
        "reading, and every 10 ms bar record holds both toe-clearance readings; (e) for the 60 cm body and head "
        "the re-run Stage 1b 10 ms turn episodes of 60-N-T-10 are trace-identical to their committed records "
        "(the bar's re-read record, cited).  If (d) or (e) fails, no training run of that body and head starts.  "
        + MODULE_READING +
        "every companion record of R0L@10ms under the bar's out directory is read with the bar record it names "
        "(its sha256 checked); every bar episode must have run (no error) and hold a success value for every "
        "term its record lists (and every term of reading 3); a companion episode must be read or fell; a walk "
        "or turn episode that reached its window must hold the four look numbers of the robot, finite.  The two "
        "toe-clearance readings are the own-swing reading (toe_clearance_own_swing_m) and Stage 1's reference-"
        "clock reading (stage1_toe_clearance_m), finite in every walk and turn episode of every 10 ms bar record "
        "that reached its window (a record none of whose episodes reached it is listed).  The records expected "
        "are those of every gated condition the set holds a reference for, and walk/nominal.  (e) cites the "
        "reread record given: schema bar-reread-1 of 60-N-T-10, its rows all 10 ms rows (names ending @10ms) "
        "and its conditions all turns (keys beginning turn/), complete, every trace equal; the -X head cites the "
        "plain body's."),
    k3=(
        "STAGE2_TRACKB_PLAN rule K3: on this machine, with one set of library versions, on the real env: "
        "75-M-H-10, set V6, R5, seed 42, 8 envs, n_steps 1024, a checkpoint every 32,768 steps, 131,072 steps in "
        "all; one run uninterrupted with 2 workers; one run with 2 workers killed with SIGKILL (the whole process "
        "group) when its progress reads 81,920 steps and resumed with 3 workers.  Pass: every policy tensor, the "
        "optimizer's state, the normalisation statistics (their count included) and the fingerprint actions of "
        "the two final checkpoints are equal (tolerance zero).  If they are not, the build stops and goes to the "
        "lead; no tolerance is set.  "
        + MODULE_READING +
        "both runs are train2.train with gates false (no G8a short form, no baseline, no check: GATES_RULE, "
        "'for validation runs and tests') and monitor_every 131,072 (a multiple of the checkpoint interval, as the "
        "trainer asks); the learning rate and the filter are the validation records' (lr_probe.json, filter.json) "
        "unless given.  The kill is made by this process at its first reading of the killed run's progress.json "
        "at or past 81,920 steps (read every 20 ms); a reading past it is no kill as ruled and that run is made "
        "again.  A run that was interrupted otherwise (a restart of the machine) is moved aside and made again: "
        "the uninterrupted run has one session, the killed run two.  A run already on disk is reused only when its "
        "run.json's settings are the ones asked (mode, body, seed, set, 8 envs and n_steps 1024 as the trainer "
        "records them, tiny, gates, the checkpoint and check intervals, the budget, the config and PPO overrides, "
        "the starting rate and the filter, as train2 resolves them from the validation records when not given); "
        "otherwise it is moved aside and made again, and the record lists what differed.  The trainer is refused "
        "before any run when it would not run 8 envs of 1024 steps.  Pass also needs both runs' settings_sha256 "
        "equal and each run's settings the ones asked.  Compared, on every checkpoint both runs "
        "hold after the resumed step (the final one among them): the archive's policy.pth, "
        "policy.optimizer.pth and pytorch_variables.pth tensors, the VecNormalize statistics (observation and "
        "return running mean, variance and count, and the wrapper's clip and gamma), loaded through "
        "s2load.load_vecnormalize, and every array of the fingerprint file (observations, actions and the plain "
        "statistics); equal means the same dtype, shape and every value"),
    rates=(
        "the validation record of rates and memory (design section 6, step V4; lesson 10): r5env.throughput per "
        "floor in one process (one env, actions normal(0, 0.1), resets included), then a trainer-style rollout "
        "loop with 8 envs on rule J5's floors at 1, 2, 3, 4 workers for 16,384 steps each: a fresh PPO policy of "
        "design 2 (sigma 0.1) acting through VecNormalize on the worker pool, rollouts collected as PPO collects "
        "them and no update.  Env steps per wall second, the CPU seconds of the parent and of the workers, the "
        "resident memory of the parent and of each worker.  provisional is true unless the 1-minute load average "
        "before the run was under 0.5.  The pool's build is timed apart and left out of the rate"),
    gl_start=(
        "plan B.3 (STAGE2_PLAN.md section 4 B): a training start with MUJOCO_GL set against it.  A child "
        "'python -m designlab.stage2.train2 preflight' started with MUJOCO_GL=osmesa and PYOPENGL_PLATFORM=osmesa "
        "in its environment builds a policy, runs one rollout and one update in a pool worker and saves an "
        "archive: pass when it exits 0 with ok, MUJOCO_GL is 'disable' in the trainer and in its worker, no "
        "PYOPENGL_PLATFORM is left in either and the worker holds no torch.  Rule X1: on the view S1 (the "
        "preflight's env has no plant and plays no set)"),
    r6_start=(
        "plan B.3: R6's start.  An R6Env of the body and set (training's first env config: hard floor, "
        "training's randomisation): 200 env steps of the zero action and 200 of actions uniform on [-1, 1], "
        "episodes reset as they end; after every step the joint targets inside the joint ranges, the clock "
        "factor in [0.75, 1.25], the observation and the reward finite; a mid-task start: the history and the "
        "filter hold the reference's actions and the targets are the reference's (design D8), then up to 50 "
        "zero-action steps under the same checks; then a tiny R6 training run of 2 rollouts (8 x 64 steps each, "
        "gates off) that ends done with its final checkpoint's fingerprint reproduced; a run already under the "
        "work directory is reused only when its run.json's settings are this one's (R6, the body, the set, seed "
        "42, tiny, gates off, the budget), else it is moved aside and made again.  Rule X1: on the view S1"),
    x1=(
        "STAGE2_TRACKB_PLAN rule X1: on a look set, for each body and head, no closed-loop episode before G9-ref "
        "passed; between G9-ref and S2-4's start only Z1's episodes, R0L's baseline and panel records, Z2's "
        "G9-run, the probes of O2-O4, the rate measurement, K3's two runs, the acceptance run's tiny jobs and "
        "R2L's fits; none but Z1's, R0L's and Z2's, and the fits', is a record; each is listed in "
        "stage2/tb/dev_episodes.json with its command, set, body, rows or mode, and count"),
)


class GateError(RuntimeError):
    """A gate could not be judged (exit code 2)."""


class Refused(GateError):
    """A gate refused to run (rule X1, or its inputs): nothing was run (exit code 2)."""


# ------------------------------------------------------------------------------------------------
# Records
# ------------------------------------------------------------------------------------------------
SOURCE_MODULES = ("gates2", "g8a", "r5lib", "r5core", "r5env", "lookread", "lookbar", "s2rows", "s2load", "vecpool",
                  "train2", "monitor2", "tbenv", "schedules")


def sources() -> dict:
    """The sha256 of this module and of the Track B modules a gate executes (by path relative to the study
    directory), with the bar's and the panels' source hashes (round A's record keys)."""
    from designlab.stage1 import panels
    from designlab.stage2 import bar

    files = {}
    for name in SOURCE_MODULES:
        p = s2io.module_path(f"{name}.py")
        files[s2io.portable(str(p))] = s2io.sha256_file(p) if p.is_file() else None
    return dict(track_b=files, bar=bar.bar_sources()["sha256"], panels=panels.panel_sources()["sha256"])


def provenance(own_outputs=()) -> dict:
    """``{commit, dirty, dirty_paths, versions, blas_threads}`` with this command's own outputs left out."""
    from designlab.stage1 import panels

    return dict(**s2io.git_state(own_outputs=[str(Path(o).resolve()) for o in own_outputs if o is not None]),
                versions=panels.library_versions(), blas_threads=s2io.blas_threads())


def finish(doc: dict, record=None) -> dict:
    """The record with its ``sha256`` (of the record without that key), written to ``record`` when given."""
    doc = json.loads(json.dumps(s2io.plain(doc)))
    doc.pop("sha256", None)
    doc["sha256"] = s2io.sha256_json(doc)
    if record is not None:
        s2io.write_json(record, doc)
    return doc


def record_sha_ok(doc: dict | None) -> bool:
    """Whether a record's ``sha256`` is that of the record without it."""
    if not isinstance(doc, dict) or "sha256" not in doc:
        return False
    rest = {k: v for k, v in doc.items() if k != "sha256"}
    return s2io.sha256_json(rest) == doc["sha256"]


def cite(path) -> dict:
    """A file cited by path (relative to the study directory where it lies under it) and sha256."""
    p = Path(path)
    return dict(path=s2io.portable(str(p.resolve())), sha256=s2io.sha256_file(p))


def _read(path) -> dict | None:
    try:
        doc = s2io.read_json(path)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def gate_path(set_name: str, body: str, name: str) -> Path:
    """``stage2/tb/gates/<set>/<body>/<name>.json`` (design section 6's layout)."""
    return GATES / set_name / body / f"{name}.json"


def _under(path, directory) -> bool:
    """Whether ``path`` lies under ``directory`` (both resolved)."""
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
    except ValueError:
        return False
    return True


def _set_root(root):
    from designlab.stage2 import setio

    return setio.ROOT if root is None else Path(root)


def _latest_freeze(set_name: str, body: str, root=None) -> dict | None:
    """``{index, manifest_sha256}`` of a set's latest freeze (None: the S1 view, a set never frozen or not here)."""
    from designlab.stage2 import setio

    if set_name == STOCK_SET:
        return None
    d = setio.set_dir(set_name, body, _set_root(root))
    if not (d / "set.json").is_file():
        return None
    fz = setio.frozen(s2io.read_json(d / "set.json"))
    return None if fz is None else dict(index=fz["index"], manifest_sha256=fz["manifest_sha256"])


def require_g9_ref(body: str, set_name: str, path=None, root=None) -> dict | None:
    """Rule X1 for a gate that runs closed-loop episodes: on a look set (``LOOK_SETS``) ``Refused`` unless
    ``path`` (default ``gate_path(set, body, "g9_ref")``) is a G9-ref record (``SCHEMAS["g9_ref"]``, its own
    sha256 intact) of this set and body that passed and was made on the set's latest freeze.  Returns its citation
    (None on any other set)."""
    if set_name not in LOOK_SETS:
        return None
    p = Path(path) if path is not None else gate_path(set_name, body, "g9_ref")
    doc = _read(p) if p.is_file() else None
    why = None
    if doc is None:
        why = "there is no G9-ref record" if not p.is_file() else "the G9-ref record cannot be read"
    elif doc.get("schema") != SCHEMAS["g9_ref"] or not record_sha_ok(doc):
        why = "the G9-ref record is not one (schema, or its sha256 is not its own)"
    elif (doc.get("body"), doc.get("set")) != (body, set_name):
        why = f"the G9-ref record is of {doc.get('set')}/{doc.get('body')}"
    elif doc.get("pass_") is not True:
        why = "G9-ref did not pass"
    else:
        now = _latest_freeze(set_name, body, root)
        if now is None:
            why = f"set {set_name} of {body} is not here or not frozen"
        elif (doc.get("set_freeze") or {}).get("manifest_sha256") != now["manifest_sha256"]:
            why = (f"the G9-ref record was made on another freeze of the set "
                   f"({str((doc.get('set_freeze') or {}).get('manifest_sha256'))[:12]}..., the latest is "
                   f"{now['manifest_sha256'][:12]}...)")
    if why is not None:
        raise Refused(f"{set_name}/{body}: {why} ({s2io.portable(str(p))}): rule X1, no closed-loop episode on a "
                      "look set before G9-ref has passed there")
    return dict(cite(p), pass_=True, at=doc.get("at"), set_freeze=doc.get("set_freeze"))


# ------------------------------------------------------------------------------------------------
# G8a (rule Z1)
# ------------------------------------------------------------------------------------------------
EPISODE_KEYS = ("condition", "seed", "k", "reference_steps", "env_steps", "compared_steps", "first_difference",
                "reference_fell_step", "env_terminated_step", "ends_alike", "identical", "interference_first_env_step")


def g8a(body: str, set_name: str, record=None, *, conditions=None, seeds=None, delays=None, cut_s=None, root=None,
        g9_ref=None) -> dict:
    """Rule Z1's record (design 3.7's schema): ``g8a.bar_form(body, set_name, G8A_CONDITIONS, (3042, 3043), (0, 5))``;
    the conditions the set holds no reference for are ``not_run`` with the set's reason.  On the Stage 1 view
    ``g8a.stock`` on ``monitor2.S1_CONDITIONS``.  ``conditions``, ``seeds``, ``delays`` and ``cut_s`` make a check
    of the command (``complete`` false).  ``differences`` names every episode that is not identical: condition,
    seed, delay and first differing physics step."""
    s2io.require_one_thread()
    g9 = require_g9_ref(body, set_name, g9_ref, root)
    seeds = tuple(int(s) for s in (seeds or G8A_SEEDS))
    delays = tuple(int(k) for k in (delays if delays is not None else G8A_DELAYS))
    gate_form = (set_name != STOCK_SET and tuple(conditions or G8A_CONDITIONS) == G8A_CONDITIONS
                 and seeds == G8A_SEEDS and delays == G8A_DELAYS and cut_s is None)
    if record is not None and not gate_form and _under(record, GATES):
        raise Refused(f"{s2io.portable(str(Path(record).resolve()))}: a record of fewer conditions, seeds or delays, a "
                      "cut, or the Stage 1 view's form is a check of the command, never rule Z1's record: it is not "
                      f"written under {s2io.portable(str(GATES.resolve()))}/ (write it elsewhere)")
    from designlab.stage2 import g8a as G

    t0 = time.perf_counter()
    if set_name == STOCK_SET:
        from designlab.stage2 import monitor2

        conds = tuple(conditions or monitor2.S1_CONDITIONS)
        eps = []
        try:
            for c in conds:
                for s in seeds:
                    for k in delays:
                        e = G.stock(body, c, s, k, seconds=cut_s)
                        eps.append(dict({x: e.get(x) for x in EPISODE_KEYS}, seconds=e.get("seconds"), row=f"R0@{2 * k}ms",
                                        reference_fall_cause=e.get("reference_fall_cause"),
                                        env_fall_cause=e.get("env_fall_cause"), reference_warned=e.get("reference_warned")))
        finally:
            G.release()
        from designlab.stage1 import panels

        doc = dict(rule=G.RULE, body=body, set=set_name, set_freeze=None, cut_s=cut_s, episodes=eps,
                   identical=sum(1 for e in eps if e["identical"]), of=len(eps), not_run=[],
                   passes=bool(eps) and all(e["identical"] and e["compared_steps"] == e["reference_steps"] for e in eps),
                   key_context=panels.key_context(body), form="stock")
        complete = False
        expected = list(conds)
    else:
        conds = tuple(conditions or G8A_CONDITIONS)
        doc = G.bar_form(body, set_name, conds, seeds, delays, cut_s=cut_s, root=_set_root(root))
        doc["form"] = "bar"
        complete = (conds == G8A_CONDITIONS and seeds == G8A_SEEDS and delays == G8A_DELAYS and cut_s is None)
        expected = list(G8A_CONDITIONS)
    eps = doc["episodes"]
    differences = [dict(condition=e["condition"], seed=e["seed"], k=e["k"], delay_ms=2 * int(e["k"]),
                        first_difference=e["first_difference"], ends_alike=e["ends_alike"],
                        reference_fell_step=e["reference_fell_step"], env_terminated_step=e["env_terminated_step"])
                   for e in eps if not (e["identical"] and e["compared_steps"] == e["reference_steps"])]
    passes = bool(eps) and not differences
    out = dict(schema=SCHEMAS["g8a"], gate="G8a", rule=RULES["z1"], compare_rule=doc["rule"], body=body,
               set=set_name, form=doc["form"], set_freeze=doc.get("set_freeze"), conditions=list(conds),
               conditions_of_the_rule=expected, seeds=list(seeds), delays=list(delays), cut_s=cut_s,
               complete=bool(complete), episodes=eps, identical=doc["identical"], of=doc["of"],
               not_run=doc.get("not_run", []), differences=differences, passes=passes, pass_=passes,
               g9_ref=g9, sources=sources(), key_context=doc.get("key_context"),
               **provenance([record]), wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    return finish(out, record)


def g8a_lines(doc: dict) -> list[str]:
    """What the command prints of a G8a record: the count, then one line per episode that is not identical."""
    head = (f"G8a {doc['set']} {doc['body']} ({doc['form']}{'' if doc['complete'] else ', not the gate form'}): "
            f"{doc['identical']} of {doc['of']} episodes identical, {len(doc['not_run'])} conditions not run: "
            + ("PASS" if doc["pass_"] else "FAIL"))
    lines = [head]
    for d in doc["differences"]:
        lines.append(f"  {d['condition']} seed {d['seed']} k {d['k']} ({d['delay_ms']} ms): first difference at "
                     f"physics step {d['first_difference']}, ends alike {d['ends_alike']} (row fell at "
                     f"{d['reference_fell_step']}, env ended in env step {d['env_terminated_step']})")
    for n in doc["not_run"]:
        lines.append(f"  not run: {n['condition']} ({n.get('reason')})")
    return lines


# ------------------------------------------------------------------------------------------------
# G9-ref (rule Z2): its parts
# ------------------------------------------------------------------------------------------------
def _num_ok(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def four_numbers(read: dict) -> dict:
    """A reading's four numbers (bob, head pitch RMS, tail phase, toe-off) and whether all are finite."""
    tail = read.get("tail") or {}
    nums = dict(bob_pp_m=read.get("bob_pp_m"), head_world_pitch_rms_deg=read.get("head_world_pitch_rms_deg"),
                tail_phase_cycles=tail.get("phase_cycles"), tail_amp_rad=tail.get("amp_rad"),
                toe_off_rad=read.get("toe_off_rad"))
    return dict(nums, finite=all(_num_ok(nums[k]) for k in ("bob_pp_m", "head_world_pitch_rms_deg",
                                                             "tail_phase_cycles", "toe_off_rad")),
                steps=read.get("steps"), lift_offs=read.get("lift_offs"))


def toe_off_side(read: dict) -> dict:
    """The lead's ruling 7 of 16 Nov 2026: a reference's toe-off and its side of ``lookread.NOT_READ_RAD``."""
    from designlab.stage2 import lookread

    v = read.get("toe_off_rad")
    if not _num_ok(v):
        return dict(toe_off_rad=v, side="not computed", toe_off_term="not computed")
    under = float(v) < lookread.NOT_READ_RAD
    return dict(toe_off_rad=float(v), side=f"under {lookread.NOT_READ_RAD:g} rad" if under else
                f"at or over {lookread.NOT_READ_RAD:g} rad", toe_off_term="not read" if under else "read")


class _SetView:
    """A set (or the S1 view) opened for G9-ref: its directory, records, plant (kinematic) and LIPM frequency."""

    def __init__(self, body: str, set_name: str, root=None):
        from designlab.stage1 import control, plant
        from designlab.stage1.refs import Reference
        from designlab.stage2 import r5lib, setio

        self.body, self.set = body, set_name
        self.view_files = None
        if set_name == STOCK_SET:
            self.dir = setio._anchored(r5lib.s1_root(body)) / body
            view, docs = setio.stage1_view(self.dir.parent, body)
            self.view_files = view["files"]
            self.set_doc, self.crouch = {}, r5lib.S1_CROUCH
            self.docs = dict(docs, **{"set.json": None})
        else:
            self.dir = setio.set_dir(set_name, body, _set_root(root))
            self.docs = setio.load(self.dir)
            self.set_doc = self.docs["set.json"]
            if self.set_doc.get("set") != set_name:
                raise GateError(f"{self.dir}: set.json names the set {self.set_doc.get('set')!r}")
            self.crouch = float(self.set_doc.get("crouch_frac", 0.03))
        self.refs = self.docs["refs.json"]["references"]
        self.trs = self.docs["transitions.json"]["transitions"]
        self.p = plant.build(body, "hard", settle=False)
        self.omega = float(control.omega0(self.p, Reference.load(self.dir / "refs" / f"{setio.NOMINAL}.npz").q_home,
                                          crouch_frac=self.crouch))
        self.freeze = None if set_name == STOCK_SET else setio.frozen(self.set_doc)

    def walks(self) -> list[str]:
        return [n for n, e in self.refs.items() if isinstance(e, dict) and e.get("file") and e.get("available") is not False]

    def walk_track(self, name: str):
        """The walk alone, whole windows of its file to past the window ``WALK_TA_REF + 10 x 1.05`` s."""
        from designlab.stage2 import lookread, setio

        period = float(self.refs[name]["period_s"])
        end = WALK_TA_REF + lookread.RATE * WALK_WINDOW_S + 1.0
        k = max(2, int(math.ceil(end / period)))
        track = setio.build_track(self.p, self.dir, [("ref", name, k)], self.omega, self.trs)
        track.run_dir = self.dir
        return track

    def walk_reading(self, name: str) -> dict:
        from designlab.stage2 import lookread

        return lookread.playback(self.p, self.walk_track(name), self.dir, WALK_TA_REF,
                                 WALK_TA_REF + lookread.RATE * WALK_WINDOW_S)


def _freeze_files(v: _SetView) -> dict:
    """``{file: sha256}`` of the set's latest freeze (an entry's file under its kind's directory)."""
    from designlab.stage2 import setio

    return {f"{'refs' if e.get('kind') in setio.WALK_KINDS else 'transitions'}/{n}.npz": e.get("sha256")
            for n, e in ((v.freeze or {}).get("entries") or {}).items()}


def _refcheck_files_match(doc: dict | None, v: _SetView) -> bool:
    """Whether every file a refcheck record read has the sha256 the set's latest freeze holds for it."""
    files = ((doc or {}).get("inputs") or {}).get("files") or {}
    want = _freeze_files(v)
    return bool(files) and all(want.get(f) == (x or {}).get("sha256") for f, x in files.items())


def _refcheck_record(v: _SetView, record_dir, refcheck=None, run=True) -> tuple[dict | None, dict]:
    """Part (a)'s record and how it was found (``RULES["z2_ref"]``'s reading); ``files_match_freeze``: whether
    every file it read has the latest freeze's sha256 (for every source, a record given by path included)."""
    from designlab.stage2 import bar

    study = s2io.STUDY
    tried = []
    if refcheck is not None:
        p = Path(refcheck)
        doc = _read(p)
        return doc, dict(source="given", files_match_freeze=_refcheck_files_match(doc, v),
                         **(cite(p) if p.is_file() else dict(path=str(p), sha256=None)))
    last = (v.set_doc.get("freezes") or [{}])[-1] if v.set_doc.get("freezes") else {}
    c = last.get("refcheck") or {}
    if c.get("file"):
        p = study / c["file"]
        if p.is_file() and s2io.sha256_file(p) == c.get("sha256"):
            doc = _read(p)
            return doc, dict(source="cited by the set's latest freeze", files_match_freeze=_refcheck_files_match(doc, v),
                             **cite(p))
        tried.append(f"the latest freeze cites {c['file']} with another sha256 or it is not here")
    p = v.dir / "refcheck.json"
    if p.is_file():
        doc = _read(p) or {}
        if _refcheck_files_match(doc, v):
            return doc, dict(source="the set directory's refcheck.json (its files are the latest freeze's)",
                             files_match_freeze=True, **cite(p))
        tried.append("the set directory's refcheck.json read other files than the latest freeze's")
    if not run:
        return None, dict(source="none", tried=tried)
    out = None if record_dir is None else Path(record_dir) / "refcheck"
    doc = bar.refcheck(v.set, v.body, v.dir.parent.parent, out=out, log=lambda *a: None)
    match = _refcheck_files_match(doc, v)
    if out is None:
        return doc, dict(source="bar.refcheck, run by this gate (not written: no record)", path=None,
                         sha256=s2io.sha256_json(doc), files_match_freeze=match, tried=tried)
    p = out / v.set / v.body / "refcheck.json"
    return doc, dict(source="bar.refcheck, run by this gate", files_match_freeze=match, **cite(p), tried=tried)


def played_files(v: _SetView, items: bool = True) -> list[str]:
    """The files G9-ref's parts (b) and (c) play on a set: the nominal walk's (the LIPM frequency), every walk's,
    and (``items``) those of every walk and turn item at mu 0.9 the bar's planner resolves there."""
    from designlab.stage2 import bar, setio

    files = {f"refs/{setio.NOMINAL}.npz"} | {v.refs[n]["file"] for n in v.walks()}
    if items and v.set != STOCK_SET:
        planner = bar.BarPlanner(v.dir, v.refs, v.trs, v.set_doc)
        for c in bar.conditions(v.docs, every=True):
            if c.judge not in ("bar_walk", "bar_turn") or c.friction is not None or planner.missing(c) is not None:
                continue
            try:
                files.update(planner.resolve(c)["files"])
            except Exception:  # noqa: BLE001 - part (b) records that item's fault
                continue
    return sorted(files)


def inputs_of(v: _SetView, files) -> dict:
    """``{files: {file: {sha256, entry_sha256}}, freeze}`` of ``files`` on the set, each checked against the set's
    latest freeze (``setio.read_inputs``, ``require_frozen``); ``Refused`` for a set that is not frozen or a file
    that is not the freeze's.  On the Stage 1 view: its files as ``setio.stage1_view`` hashed them (no freeze)."""
    from designlab.stage2 import setio

    files = sorted(set(files))
    if v.set == STOCK_SET:
        missing = [f for f in files if f not in (v.view_files or {})]
        if missing:
            raise Refused(f"S1/{v.body}: {', '.join(missing)} not listed by the Stage 1 view")
        return dict(files={f: dict(v.view_files[f]) for f in files}, freeze=None)
    try:
        rec, _ = setio.read_inputs(v.dir, files, require_frozen=True)
    except setio.tune.InputError as exc:
        raise Refused(f"{v.set}/{v.body}: {exc}: G9-ref plays the set's latest freeze only (rule Z2: every frozen "
                      "reference)") from None
    return dict(files=rec["files"], freeze=rec["freeze"])


def still_available(v: _SetView, ref: str | None) -> dict:
    """Whether a failing item's reference (``refs/<name>.npz`` or ``transitions/<name>.npz``) is still available in
    the set (the lead's ruling 14 of 17 Nov 2026; ``RULES["z2_ref"]``'s reading): its entry is usable as the env and
    the bar read it (``r5lib.why_not``), or the set's latest freeze lists it.  A reference that cannot be named
    counts as still available."""
    from designlab.stage2 import r5lib

    if not ref or "/" not in ref:
        return dict(entry_available=None, in_latest_freeze=None, still_available=True)
    sub, fname = ref.split("/", 1)
    name = fname[:-len(".npz")] if fname.endswith(".npz") else fname
    entry = r5lib.why_not(v.refs, v.trs, v.set_doc, sub, name) is None
    frozen = name in ((v.freeze or {}).get("entries") or {})
    return dict(entry_available=bool(entry), in_latest_freeze=bool(frozen), still_available=bool(entry or frozen))


def part_a(v: _SetView, record_dir=None, refcheck=None) -> dict:
    """G9-ref (a): the bar's reference check, cited; what follows a failure (``RULES["z2_ref"]``).  ``still_available``
    names the failing items' references that are still available in the set: while it is not empty G9-ref fails
    (the lead's ruling 14 of 17 Nov 2026)."""
    if v.set == STOCK_SET:
        return dict(ok=None, applies=False, still_available=[],
                    reason="the Stage 1 view has no frozen set of the bar's references (no set.json): bar.refcheck "
                           "reads a set")
    doc, how = _refcheck_record(v, record_dir, refcheck)
    if doc is None:
        return dict(ok=False, applies=True, refcheck=how, still_available=[], blocks=True,
                    reason="no refcheck record could be read")
    items = doc.get("items") or {}
    failing = {k: x for k, x in items.items() if x.get("status") == "done" and not x.get("pass_")}
    refs_out = []
    for k, x in sorted(failing.items()):
        kind = k.split("/")[0]
        ref = (f"refs/{k.split('/', 1)[1]}.npz" if kind == "walk" else
               next((f for f in reversed(x.get("files") or []) if f.startswith("transitions/")), None))
        bad = sorted(t for t, ok in (x.get("ok") or {}).items() if not ok)
        refs_out.append(dict(item=k, reference=ref, failing=bad, **still_available(v, ref)))
    still = sorted({x["reference"] or f"the reference of {x['item']} (not named)" for x in refs_out
                    if x["still_available"]})
    blocked = sorted(k for k in failing if k in BLOCKING_ITEMS)
    same_set = (doc.get("set"), doc.get("body")) == (v.set, v.body)
    files_ok = how.get("files_match_freeze") is True
    ok = bool(same_set and files_ok and doc.get("pass_"))
    marks = (f"G9-ref fails while a failing item's reference is still available in the set ({', '.join(still)}): "
             "the lead marks them not available, freezes the set again and runs G9-ref again (the lead's ruling 14 "
             "of 17 Nov 2026)")
    followed = ("none: every reference passes its own item" if ok else
                "the reference check cited is not of this set and body or did not read the latest freeze's files: "
                "(a) is not read; it goes to the lead" if not (same_set and files_ok) else
                "blocked: a failing walk/nominal or walk/v0.4 item blocks the body and head; it goes to the lead"
                if blocked else marks if still else
                "the failing items' references are not available in the set and not in its latest freeze (rule T2, "
                "H3); they go to the lead")
    return dict(_doc=doc, ok=ok, applies=True, refcheck=how, schema=doc.get("schema"), refcheck_pass=doc.get("pass_"),
                same_set_and_body=same_set, files_match_freeze=files_ok,
                items={k: dict(status=x.get("status"), pass_=x.get("pass_"), ok=x.get("ok")) for k, x in items.items()},
                failing_references=refs_out, still_available=still if (same_set and files_ok) else [],
                blocked=blocked, blocks=bool(blocked or not same_set or not files_ok), followed=followed)


def _ladder() -> tuple[dict, dict, dict]:
    metrics = s2io.read_json(LOOK / "metrics.json")
    l1 = s2io.read_json(LOOK / "L1_FREEZE.json")
    return metrics, l1, dict(metrics=cite(LOOK / "metrics.json"), l1_freeze=cite(LOOK / "L1_FREEZE.json"))


def _rung_walk(v: _SetView, speed: str, variant: str) -> str | None:
    for n in v.walks():
        tw = v.refs[n].get("ladder_twin")
        if isinstance(tw, (list, tuple)) and len(tw) == 2 and tw[0] == speed:
            return n
    name = RUNG_WALKS.get(speed)
    return name if name in v.walks() else None


def part_b(v: _SetView, comparand: _SetView | None = None) -> dict:
    """G9-ref (b) (``RULES["z2_ref"]``).  ``comparand``: the plain body's view of the same set (an ``-X`` body) or
    the same body's V6 view (``V6noF5``)."""
    from designlab.stage2 import bar, lookread

    toe = v.set != NO_TOE_OFF_SET
    out = dict(toe_off="read" if toe else "not read (V6noF5: rule L11; Z2 as settled on 15 Nov 2026)", walks={},
               rungs={}, compared={}, items={}, faults=[])
    readings = {}
    for n in v.walks():
        try:
            r = v.walk_reading(n)
        except Exception as exc:  # noqa: BLE001 - a walk that cannot be played is a fault of the definition
            out["faults"].append(f"walk {n}: {type(exc).__name__}: {exc}")
            continue
        readings[n] = r
        nums = four_numbers(r)
        out["walks"][n] = dict(nums, window_ref=r.get("window_ref"), entry_look_sha256=v.refs[n].get("look_sha256"),
                               ladder_twin=v.refs[n].get("ladder_twin"),
                               ok=bool(nums["finite"] and (r.get("steps") or 0) >= lookread.MIN_STEPS))
    is_x = v.body.endswith("-X")
    if comparand is None and not is_x and toe and v.set != STOCK_SET:
        metrics, l1, cited = _ladder()
        out["ladder"] = cited
        frozen = (l1.get("rungs") or {}).get(v.body) or {}
        for speed, row in frozen.items():
            if not row.get("in_ladder"):
                out["rungs"][speed] = dict(in_ladder=False, note="not in the ladder (L1_FREEZE.json): a rung outside "
                                                                 "it reads finite numbers only")
                continue
            name = _rung_walk(v, speed, row["variant"])
            if name is None or name not in readings:
                out["rungs"][speed] = dict(in_ladder=True, walk=name, held=False, ok=speed not in REQUIRED_RUNGS,
                                           note="the set holds no walk of this rung" + (
                                               ": a required rung" if speed in REQUIRED_RUNGS else ": listed for the lead"))
                continue
            twin = v.refs[name].get("ladder_twin")
            variant = twin[1] if isinstance(twin, (list, tuple)) and len(twin) == 2 else row["variant"]
            lad = lookread.ladder_numbers(metrics["bodies"][v.body][speed][variant])
            a = lookread.against_ladder(readings[name], lad, toe_off=True)
            hash_ok = v.refs[name].get("look_sha256") == row.get("sha256")
            out["rungs"][speed] = dict(in_ladder=True, walk=name, held=True, ladder_variant=variant,
                                       l1_variant=row["variant"], against_ladder=a, look_sha256=v.refs[name].get(
                                           "look_sha256"), l1_sha256=row.get("sha256"), hash_ok=hash_ok,
                                       ok=bool(a["ok"] and hash_ok))
    elif comparand is not None:
        out["comparand"] = dict(set=comparand.set, body=comparand.body,
                                files={n: dict(cite(comparand.dir / comparand.refs[n]["file"]),
                                               entry_sha256=comparand.refs[n].get("sha256"))
                                       for n in comparand.walks()})
        for n in readings:
            if n not in comparand.walks():
                out["compared"][n] = dict(ok=False, note=f"{comparand.set}/{comparand.body} holds no walk {n}")
                continue
            try:
                other = comparand.walk_reading(n)
            except Exception as exc:  # noqa: BLE001
                out["compared"][n] = dict(ok=False, note=f"the comparand's walk: {type(exc).__name__}: {exc}")
                continue
            a = lookread.against_ladder(readings[n], other, toe_off=toe)
            out["compared"][n] = dict(against=a, comparand_numbers=four_numbers(other), ok=bool(a["ok"]))
    elif v.set == STOCK_SET:
        out["rungs_note"] = "the Stage 1 view holds no look-format walk: no ladder comparison"
    # every walk and turn item: its window's playback
    if v.set != STOCK_SET:
        ctx = bar.BarContext(v.body, v.dir, {"refs.json": v.docs["refs.json"],
                                             "transitions.json": v.docs["transitions.json"]}, v.set_doc)
        planner = ctx.bar_planner
        for c in bar.conditions(v.docs, every=True):
            if c.judge not in ("bar_walk", "bar_turn") or c.friction is not None:
                continue
            why = planner.missing(c)
            if why is not None:
                out["items"][c.key] = dict(status="not run", reason=why.get("reason"), ok=True)
                continue
            try:
                b = ctx.built(c)
                wr = lookread.window_ref(b)
                r = lookread.playback(b.p, b.track, b.track.run_dir, *wr)
            except Exception as exc:  # noqa: BLE001 - rule U5: a fault of the definition, never "no look reading"
                out["items"][c.key] = dict(status="fault", ok=False, fault=f"{type(exc).__name__}: {exc}")
                continue
            enough = (r.get("steps") or 0) >= lookread.MIN_STEPS and (r.get("lift_offs") or 0) >= lookread.MIN_STEPS
            out["items"][c.key] = dict(status="played", window_ref=[float(x) for x in wr], gated=bool(c.gated),
                                       steps=r.get("steps"), lift_offs=r.get("lift_offs"), numbers=four_numbers(r),
                                       **toe_off_side(r), ok=bool(enough))
        ctx.release()
    oks = ([w["ok"] for w in out["walks"].values()] + [r["ok"] for r in out["rungs"].values() if "ok" in r]
           + [c["ok"] for c in out["compared"].values()] + [i["ok"] for i in out["items"].values()])
    out["ok"] = bool(oks and all(oks) and not out["faults"])
    return out


def tail_tenth(p, poses) -> np.ndarray:
    """The fifth degraded copy of a playback's poses (``TRACKB_RULINGS6_16NOV.md`` ruling 2): the tail-yaw joint's
    angle scaled to ``TAIL_SCALE`` of its deviation from its mean over the poses; nothing else changes."""
    from designlab.stage2 import lookread

    q = np.array(poses, float)
    i = lookread._ids(p.m)["tail"]
    col = q[:, i]
    q[:, i] = col.mean() + TAIL_SCALE * (col - col.mean())
    return q


def controls(p, arrays: dict, flat_feet: bool = True) -> dict:
    """G9-ref (c) on one playback's arrays (``lookread.played``): ``lookread.controls`` (the four degraded copies,
    flat feet only with ``flat_feet``; the unchanged playback; the 3-step cut) and the fifth, ``tail_tenth``, which
    must fail the tail term and no other.  ``ok``."""
    from designlab.stage2 import lookread

    kinds = tuple(k for k in lookread.DEGRADED if flat_feet or lookread.DEGRADED[k] != "toe_off")
    w = arrays["window"]
    c = lookread.controls(p, arrays["poses"], arrays["loads"], arrays["dt"], arrays["every"], t0=arrays["t0"],
                          ta=w[0], tb=w[1], kinds=kinds)
    ser = lookread.series(p, arrays["poses"], arrays["loads"], arrays["dt"], arrays["every"], t0=arrays["t0"])
    ref = lookread.reading(ser, w[0], w[1])
    own = lookread.reading(lookread.series(p, tail_tenth(p, arrays["poses"]), arrays["loads"], arrays["dt"],
                                           arrays["every"], t0=arrays["t0"]), w[0], w[1])
    cmp = lookread.compare(own, ref)
    failed = [t for t in lookread.TERMS if not cmp["terms"][t]]
    tenth = dict(own=own, compare=cmp, failed=failed, expected=["tail"], ok=failed == ["tail"], scale=TAIL_SCALE,
                 own_amp_rad=(own.get("tail") or {}).get("amp_rad"), reference_amp_rad=(ref.get("tail") or {}).get("amp_rad"))
    brief = dict(unchanged=c["unchanged"]["ok"], cut=c["cut"]["ok"],
                 **{k: dict(failed=x["failed"], expected=x["expected"], ok=x["ok"]) for k, x in c["degraded"].items()},
                 tail_tenth=dict(failed=failed, expected=["tail"], ok=tenth["ok"]))
    return dict(kinds=list(kinds) + ["tail_tenth"], flat_feet=bool(flat_feet), lookread=c, tail_tenth=tenth,
                brief=brief, ok=bool(c["ok"] and tenth["ok"]))


def part_c(v: _SetView) -> dict:
    """G9-ref (c) on the playbacks of ``CONTROL_WALKS`` (flat feet on V6 only)."""
    from designlab.stage2 import lookread

    flat = v.set == "V6"
    out = dict(flat_feet=flat, flat_feet_rule="(c)'s flat-feet playback applies to V6 only (Z2 as settled)",
               walks={})
    for n in CONTROL_WALKS if v.set != STOCK_SET else ("nominal",):
        if n not in v.walks():
            out["walks"][n] = dict(ok=False, note="the set holds no such walk")
            continue
        try:
            a = lookread.played(v.p, v.walk_track(n), v.dir, WALK_TA_REF, WALK_TA_REF + lookread.RATE * WALK_WINDOW_S,
                                lead_s=lookread.LEAD_S)
            c = controls(v.p, a, flat_feet=flat)
        except Exception as exc:  # noqa: BLE001
            out["walks"][n] = dict(ok=False, fault=f"{type(exc).__name__}: {exc}")
            continue
        out["walks"][n] = dict(window=a["window"], brief=c["brief"], ok=c["ok"],
                               tail_tenth=dict(own_amp_rad=c["tail_tenth"]["own_amp_rad"],
                                               reference_amp_rad=c["tail_tenth"]["reference_amp_rad"]))
    out["ok"] = bool(out["walks"]) and all(x["ok"] for x in out["walks"].values())
    return out


def _segment_end(seg) -> np.ndarray:
    """The joint arrays a track segment gives at its own end (``control.Track.joint``'s formula at its full
    duration, on that segment)."""
    if seg.kind == "periodic":
        s = (seg.duration / seg.step) % seg.n
        k0 = int(s) % seg.n
        a = s - int(s)
        return (1 - a) * seg.S[k0] + a * seg.S[(k0 + 1) % seg.n]
    if seg.kind == "hold":
        return seg.S[0]
    return seg.S[-1]


def walk_steps(seg, nj: int) -> np.ndarray:
    """A walk's largest change between two consecutive frames, per joint (rad): over every two consecutive frames of
    its file's window as a periodic segment plays them, the last frame to the first included."""
    S = np.asarray(seg.S, float)[:, :nj]
    return np.max(np.abs(np.diff(np.vstack([S, S[:1]]), axis=0)), axis=0)


def _walk_left(segs, i: int) -> tuple:
    """The walk a join after segment ``i`` is held to (``RULES["z2_ref"]``'s reading): (segment or None, how)."""
    walks = [k for k, s in enumerate(segs) if s.kind == "periodic"]
    before = [k for k in walks if k <= i]
    after = [k for k in walks if k > i]
    if before and before[-1] == i:
        return segs[i], "the walk it leaves"
    if before:
        return segs[before[-1]], "the walk the track played last before the join (it left that walk through a move)"
    if after:
        return segs[after[0]], "the first walk the track plays after the join (no walk before it)"
    return None, "none: the track plays no walk (the join is held to 0 at every joint)"


def joins(track, nj: int, joints=None) -> list[dict]:
    """Every join of a track (G9-ref (f), the lead's ruling 15 of 17 Nov 2026): the segments' names and kinds, the
    walk it is held to, and per joint (``joints``: the names of the first ``nj`` columns) the jump of the joint
    target between segment i's end and segment i + 1's first frame with the largest change between two consecutive
    frames of that walk (rad); ``over`` names the joints whose jump exceeds it, and the join is ``ok`` when none
    does.  ``jump_rad``: the largest jump over the joints."""
    names = list(joints) if joints is not None else [str(j) for j in range(nj)]
    if len(names) != nj:
        raise GateError(f"{len(names)} joint names for {nj} joint columns")
    segs = track.segments
    out = []
    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        jump = np.abs(np.asarray(_segment_end(a), float)[:nj] - np.asarray(b.S[0], float)[:nj])
        w, how = _walk_left(segs, i)
        step = np.zeros(nj) if w is None else walk_steps(w, nj)
        within = jump <= step  # (a NaN on either side is not within)
        out.append(dict(at_ref_s=float(b.t0), before=f"{a.kind} {a.name}", after=f"{b.kind} {b.name}",
                        walk=None if w is None else f"{w.kind} {w.name}", walk_is=how, jump_rad=float(np.max(jump)),
                        joints={n: dict(jump_rad=float(jump[k]), walk_step_rad=float(step[k]))
                                for k, n in enumerate(names)},
                        over=[n for k, n in enumerate(names) if not within[k]], ok=bool(np.all(within))))
    return out


def part_f(v: _SetView, refcheck_doc: dict | None = None) -> dict:
    """G9-ref (f): the screen record cited for the references no reference check plays; every task's joins and
    2 s of its zero-action targets."""
    from designlab.stage2 import lookread, r5env, r5lib

    out = dict(unplayed={}, tasks={})
    if v.set != STOCK_SET:
        played = set()
        for x in ((refcheck_doc or {}).get("items") or {}).values():
            if x.get("status") == "done":
                played.update(x.get("files") or [])
        scr_path = v.dir / "screen.json"
        scr = _read(scr_path) if scr_path.is_file() else None
        out["screen"] = cite(scr_path) if scr is not None else None
        blocks = (scr or {}).get("blocks") or {}
        for sub, doc in (("refs", v.refs), ("transitions", v.trs)):
            for n, e in doc.items():
                if not (isinstance(e, dict) and e.get("file") and e.get("available") is not False):
                    continue
                f = f"{sub}/{n}.npz"
                if f in played:
                    continue
                blk = blocks.get(n)
                want = e.get("look_sha256") if sub == "refs" else e.get("sha256")
                got = ((blk or {}).get("arrays") or {}).get("sha256")
                verdict = (blk or {}).get("verdict") or {}
                out["unplayed"][n] = dict(file=f, block=blk is not None, block_sha256=got, entry_sha256=want,
                                          within_limits=verdict.get("within_limits"),
                                          blocks_freeze=verdict.get("blocks_freeze"), verdict=verdict.get("text"),
                                          ok=bool(blk is not None and got == want))
    else:
        out["screen"] = None
        out["unplayed_note"] = "the Stage 1 view has no screen record; its references are Stage 1's"
    lib = r5lib.Library(v.p, v.body, v.set, lookread.RATE, **({} if v.set == STOCK_SET else dict(root=v.dir.parent.parent)))
    nj = len(v.p.joints)
    dt = float(v.p.m.opt.timestep)
    for t, row in lib.tasks.items():
        if not row["available"]:
            out["tasks"][t] = dict(available=False, reason=row["reason"], ok=True)
            continue
        try:
            s = lib.script(t)
            js = joins(s.track, nj, v.p.joints)
            tk = r5env.R5Tracker(v.p, s.track, "R2", gains=None, terms=(), ankle_cap=(), rate=lookread.RATE, model=v.p.m)
            n = int(round(ZERO_ACTION_S / dt))
            finite = True
            for k in range(n + 1):
                q, qd, ff = tk.zero_command(lookread.RATE * k * dt)
                if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd)) and np.all(np.isfinite(ff))):
                    finite = False
                    break
        except Exception as exc:  # noqa: BLE001 - a task whose script cannot be built or played fails (f)
            out["tasks"][t] = dict(available=True, ok=False, fault=f"{type(exc).__name__}: {exc}")
            continue
        worst = max((j["jump_rad"] for j in js), default=0.0)
        out["tasks"][t] = dict(available=True, joins=js, max_jump_rad=worst, joins_ok=all(j["ok"] for j in js),
                               zero_action_s=ZERO_ACTION_S, zero_action_finite=finite,
                               ok=bool(finite and all(j["ok"] for j in js)))
    lin = lib.inputs
    out["library_inputs"] = dict(files=lin["files"], freeze=lin["freeze"])
    oks = [x["ok"] for x in out["unplayed"].values()] + [x["ok"] for x in out["tasks"].values()]
    out["max_jump_rad"] = max((x.get("max_jump_rad", 0.0) for x in out["tasks"].values()), default=0.0)
    out["join_limit"] = ("at each joint, the largest change between two consecutive frames of the walk the join "
                         "leaves (the lead's ruling 15 of 17 Nov 2026; changed after a result: as first written, 1e-6 "
                         "rad)")
    out["ok"] = bool(oks and all(oks))
    return out


def g9_ref(body: str, set_name: str, record=None, *, root=None, refcheck=None) -> dict:
    """Rule Z2's G9-ref record (parts (a), (b), (c), (f); module docstring).  Kinematic only: no episode runs."""
    s2io.require_one_thread()
    t0 = time.perf_counter()
    v = _SetView(body, set_name, root)
    if set_name != STOCK_SET and v.freeze is None:
        raise Refused(f"{set_name}/{body}: the set is not frozen (set.json has no freeze): G9-ref reads a frozen "
                      "set's references (rule Z2: every frozen reference)")
    comparand = None
    if set_name != STOCK_SET and (body.endswith("-X") or set_name == NO_TOE_OFF_SET):
        cb = body[:-2] if body.endswith("-X") else body
        cs = "V6" if set_name == NO_TOE_OFF_SET else set_name
        comparand = _SetView(cb, cs, root)
        if comparand.freeze is None:
            raise Refused(f"{cs}/{cb} (the comparand): the set is not frozen (set.json has no freeze)")
    # every file (b) and (c) play, and the comparand's walks, checked against the latest freeze before any is played
    inputs = inputs_of(v, played_files(v))
    comparand_inputs = None if comparand is None else inputs_of(comparand, played_files(comparand, items=False))
    rdir = None if record is None else Path(record).parent
    a = part_a(v, rdir, refcheck)
    b = part_b(v, comparand)
    c = part_c(v)
    f = part_f(v, a.pop("_doc", None))
    parts_ok = dict(a=a["ok"], b=b["ok"], c=c["ok"], f=f["ok"])
    applies = set_name != STOCK_SET
    # the lead's ruling 14 of 17 Nov 2026: G9-ref fails while a failing item's reference is still available in the set
    passes = bool(applies and not a.get("blocks") and not a.get("still_available") and a["ok"] is not None
                  and b["ok"] and c["ok"] and f["ok"])
    if not applies:
        followed = "the development form on the Stage 1 view: no record of G9-ref"
    elif passes:
        followed = ("none" if a["ok"] else a["followed"])
    else:
        followed = ("no closed-loop episode runs on this set and body; it goes to the lead"
                    + ("" if a["ok"] else f"; (a): {a['followed']}"))
    fz = v.freeze
    lib_inputs = f.pop("library_inputs", None)
    out = dict(schema=SCHEMAS["g9_ref"], gate="G9-ref", rule=RULES["z2_ref"], body=body, set=set_name,
               form="development (S1)" if not applies else ("V6noF5" if set_name == NO_TOE_OFF_SET else
                                                             "-X" if body.endswith("-X") else "plain"),
               set_freeze=None if fz is None else dict(index=fz["index"], manifest_sha256=fz["manifest_sha256"]),
               inputs=dict(played=inputs, library=lib_inputs, comparand=None if comparand is None else dict(
                   set=comparand.set, body=comparand.body, **comparand_inputs)),
               parts=dict(a=a, b=b, c=c, f=f), parts_ok=parts_ok, pass_=passes, followed=followed,
               kinematic_only=True, sources=sources(), **provenance([record, None if rdir is None else rdir / "refcheck"]),
               wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    return finish(out, record)


def g9_ref_lines(doc: dict) -> list[str]:
    p = doc["parts_ok"]
    b = doc["parts"]["b"]
    lines = [f"G9-ref {doc['set']} {doc['body']} ({doc['form']}): (a) {p['a']}, (b) {p['b']}, (c) {p['c']}, "
             f"(f) {p['f']}: {'PASS' if doc['pass_'] else 'FAIL'}; {doc['followed']}"]
    for x in doc["parts"]["a"].get("failing_references") or []:
        lines.append(f"  (a) item {x['item']} fails {', '.join(x['failing']) or '(no term named)'}: reference "
                     f"{x['reference']}, entry available {x['entry_available']}, in the latest freeze "
                     f"{x['in_latest_freeze']}")
    for k, x in sorted(b.get("items", {}).items()):
        if x.get("status") == "played":
            lines.append(f"  item {k}: {x['steps']} steps, {x['lift_offs']} lift-offs, reference toe-off "
                         f"{x['toe_off_rad']:.4f} rad ({x['side']})")
    for t, x in sorted(doc["parts"]["f"]["tasks"].items()):
        if x.get("available") and not x["ok"]:
            lines.append(f"  task {t}: joins ok {x.get('joins_ok')} (largest jump {x.get('max_jump_rad', math.nan):.3g} "
                         f"rad), zero action finite {x.get('zero_action_finite')}{' ' + x['fault'] if 'fault' in x else ''}")
            for j in x.get("joins") or []:
                for n in j["over"]:
                    lines.append(f"    join {j['before']} -> {j['after']} at {j['at_ref_s']:.4f} s, {n}: jump "
                                 f"{j['joints'][n]['jump_rad']:.3g} rad over {j['walk']}'s largest frame-to-frame "
                                 f"change {j['joints'][n]['walk_step_rad']:.3g} rad")
    return lines


# ------------------------------------------------------------------------------------------------
# G9-run (rule Z2)
# ------------------------------------------------------------------------------------------------
def _expected_conditions(body: str, set_name: str, root=None) -> list[str]:
    """The R0L records G9-run expects: every gated condition the set holds a reference for, and walk/nominal."""
    from designlab.stage2 import bar, setio

    if set_name == STOCK_SET:
        return []
    d = setio.set_dir(set_name, body, _set_root(root))
    docs = setio.load(d)
    planner = bar.BarPlanner(d, docs["refs.json"]["references"], docs["transitions.json"]["transitions"],
                             docs["set.json"])
    out = [c.key for c in bar.conditions(docs, every=True) if (c.gated or c.key == "walk/nominal")
           and planner.missing(c) is None]
    return list(dict.fromkeys(out))


def _toe_pair(e: dict) -> dict:
    m = ((e.get("measures") or {}).get("bar") or {})
    return dict(own_swing_m=m.get("toe_clearance_own_swing_m"), reference_clock_m=m.get("stage1_toe_clearance_m"),
                present=_num_ok(m.get("toe_clearance_own_swing_m")) and _num_ok(m.get("stage1_toe_clearance_m")))


def part_d(bar_out, body: str, set_name: str, block: str, root=None) -> dict:
    """G9-run (d) (``RULES["z2_run"]``'s reading)."""
    from designlab.stage2 import bar

    bdir = bar.block_dir(bar_out, set_name, body, block)
    edir, ldir = bdir / "eval", bdir / "look" / R0L_ROW
    out = dict(block_dir=s2io.portable(str(Path(bdir).resolve())), records={}, bar_records_10ms={}, problems=[])
    comps = sorted(q for q in ldir.glob("**/*.json") if not any(x.endswith(".superseded") for x in q.parts))
    for q in comps:
        doc = _read(q)
        rel = q.relative_to(ldir).as_posix()
        probs = []
        if doc is None or doc.get("schema") != "s2-look-1":
            out["problems"].append(f"{rel}: not a companion record")
            continue
        if not record_sha_ok(doc):
            probs.append("its sha256 is not its own")
        bpath = Path(bar_out) / doc["bar_record"]["path"]
        rec = _read(bpath) if bpath.is_file() else None
        if rec is None:
            probs.append(f"its bar record {doc['bar_record']['path']} is not here")
        elif s2io.sha256_file(bpath) != doc["bar_record"]["sha256"]:
            probs.append("its bar record's sha256 is not the one it names")
        if rec is not None and rec.get("status") != "done":
            probs.append(f"its bar record is {rec.get('status')!r}, not done")
        eps_bar = {int(e["seed"]): e for e in (rec or {}).get("episodes") or []}
        terms, terms3 = list((rec or {}).get("terms") or []), list((rec or {}).get("terms_but_A1") or [])
        judge = doc.get("judge")
        bad_eps, look_missing, outcomes = [], [], {}
        for ep in doc.get("episodes") or []:
            s = int(ep["seed"])
            outcomes[ep["outcome"]] = outcomes.get(ep["outcome"], 0) + 1
            e = eps_bar.get(s)
            if ep["outcome"] not in ("read", "fell"):
                bad_eps.append(f"seed {s}: {ep['outcome']} ({ep.get('error') or ep.get('reason')})")
                continue
            if e is None or e.get("error"):
                bad_eps.append(f"seed {s}: the bar episode did not run")
                continue
            if set(e.get("success") or {}) != set(terms) or not all(isinstance(x, bool) for x in (e.get("success") or {}).values()):
                bad_eps.append(f"seed {s}: its readings are not its item's terms")
            if terms3 and set(e.get("success_but_A1") or {}) != set(terms3):
                bad_eps.append(f"seed {s}: its reading 3 is not its item's terms")
            if judge in ("bar_walk", "bar_turn") and e.get("window_reached"):
                own = (ep.get("look") or {}).get("own") or {}
                if not four_numbers(own)["finite"]:
                    look_missing.append(s)
        if bad_eps:
            probs.append("; ".join(bad_eps))
        if look_missing:
            probs.append(f"seeds {look_missing}: the window was reached and the look reading is not finite")
        out["records"][doc["condition"]] = dict(companion=cite(q), bar_record=doc["bar_record"], judge=judge,
                                                episodes=len(doc.get("episodes") or []), outcomes=outcomes,
                                                problems=probs, ok=not probs)
    # both toe-clearance readings in every 10 ms bar record (every row of the run at 10 ms)
    for q in sorted(edir.glob(f"*@{RUN_MS}ms/**/*.json")) if edir.is_dir() else []:
        parts = q.relative_to(edir).parts
        if any(x.endswith(".superseded") for x in parts) or q.name == "summary.json" or "traces" in parts:
            continue
        rec = _read(q)
        if not rec or "episodes" not in rec:
            continue
        judge = (rec.get("definition") or {}).get("judge")
        if judge not in ("bar_walk", "bar_turn"):
            continue
        reached = [e for e in rec["episodes"] if e.get("window_reached") and not e.get("error")]
        missing = [int(e["seed"]) for e in reached if not _toe_pair(e)["present"]]
        out["bar_records_10ms"][q.relative_to(edir).as_posix()] = dict(
            row=rec.get("row"), condition=rec.get("key_name"), episodes=len(rec["episodes"]),
            reached_window=len(reached), missing_toe_readings=missing, no_window=not reached, ok=not missing)
    expected = _expected_conditions(body, set_name, root)
    out["expected"] = expected
    out["missing"] = [c for c in expected if c not in out["records"]]
    oks = [x["ok"] for x in out["records"].values()] + [x["ok"] for x in out["bar_records_10ms"].values()]
    out["ok"] = bool(out["records"] and all(oks) and not out["missing"] and not out["problems"])
    return out


def part_e(body: str, stage1b_reread=None) -> dict:
    """G9-run (e): round A's ``bar reread`` record of Stage 1b's 10 ms turn episodes of 60-N-T-10, cited."""
    if body not in STAGE1B_BODIES:
        return dict(applies=False, ok=True, reason="only the 60 cm body and head re-read Stage 1b's turn episodes")
    if stage1b_reread is None or not Path(stage1b_reread).is_file():
        return dict(applies=True, ok=False, reason="no reread record given (round A's `bar reread` of 60-N-T-10's "
                                                   "Stage 1b 10 ms turn episodes: --stage1b-reread PATH)")
    doc = _read(stage1b_reread) or {}
    rows, conds = doc.get("rows"), doc.get("conditions")
    checks = dict(schema=doc.get("schema") == "bar-reread-1", body=doc.get("body") == STAGE1B_BODY,
                  rows_10ms=bool(isinstance(rows, list) and rows and all(str(r).endswith(f"@{RUN_MS}ms") for r in rows)),
                  turn_conditions=bool(isinstance(conds, list) and conds
                                       and all(str(c).startswith("turn/") for c in conds)),
                  complete=doc.get("complete") is True, traces_equal=bool(doc.get("pass_") is True and doc.get(
                      "traces") and doc.get("traces_equal") == doc.get("traces")))
    return dict(applies=True, record=cite(stage1b_reread), cited_for=body, of_body=doc.get("body"),
                rows=doc.get("rows"), conditions=doc.get("conditions"), traces=doc.get("traces"),
                traces_equal=doc.get("traces_equal"), checks=checks, ok=all(checks.values()))


def g9_run(body: str, set_name: str, bar_out, stage1b_reread=None, record=None, *, block=None, root=None) -> dict:
    """Rule Z2's G9-run record (parts (d), (e)).  Reads records only: no episode runs."""
    from designlab.stage2 import bar

    t0 = time.perf_counter()
    block = block or bar.HEADLINE_BLOCK
    d = part_d(bar_out, body, set_name, block, root)
    e = part_e(body, stage1b_reread)
    passes = bool(d["ok"] and e["ok"])
    g9 = None
    if set_name in LOOK_SETS:
        p = gate_path(set_name, body, "g9_ref")
        g9 = dict(cite(p), pass_=(_read(p) or {}).get("pass_")) if p.is_file() else None
    out = dict(schema=SCHEMAS["g9_run"], gate="G9-run", rule=RULES["z2_run"], body=body, set=set_name, block=block,
               row=R0L_ROW, bar_out=s2io.portable(str(Path(bar_out).resolve())), set_freeze=_latest_freeze(set_name, body, root),
               parts=dict(d=d, e=e), parts_ok=dict(d=d["ok"], e=e["ok"]), pass_=passes,
               followed="none" if passes else "no training run of this body and head starts; it goes to the lead",
               g9_ref=g9, sources=sources(), **provenance([record]), wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    return finish(out, record)


# ------------------------------------------------------------------------------------------------
# Rates and memory
# ------------------------------------------------------------------------------------------------
def _rss_mb(pid=None) -> float | None:
    try:
        with open(f"/proc/{'self' if pid is None else int(pid)}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _rate_specs(body: str, set_name: str) -> tuple[list, list]:
    """The rate loop's 8 env configs (training's: ``r5env.env_configs``) and the pool's specs (the trainer's)."""
    from designlab.stage2 import r5env

    cfgs = [c.describe() for c in r5env.env_configs("R5", body, set_name, 8)]
    return cfgs, [("designlab.stage2.r5env:make", {"mode": "R5", "cfg": dict(c)}) for c in cfgs]


def _throughput(body: str, set_name: str, floor: str, steps: int) -> dict:
    from dataclasses import replace

    from designlab.stage2 import r5env

    cfg = replace(r5env.env_configs("R5", body, set_name, 1)[0], floor=floor)
    return r5env.throughput("R5", cfg, env_steps=int(steps), sigma=RATE_SIGMA, seed=0)


def rollout_rate(specs: list, workers: int, steps: int, n_steps: int | None = None, worker_cmd=None) -> dict:
    """``steps`` env steps of a fresh design-2 PPO policy (sigma 0.1) on 8 envs in ``workers`` pool workers,
    collected as PPO collects its rollouts (through ``VecNormalize``), with no update: the rate and the memory."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from designlab.stage1 import policy
    from designlab.stage2 import schedules, train2, vecpool

    torch.set_num_threads(1)
    d = policy.DESIGN2
    n_steps = int(n_steps or train2.N_STEPS)
    tb = time.perf_counter()
    venv = vecpool.PoolVecEnv(specs, int(workers), worker_cmd=worker_cmd or [sys.executable, *train2.WORKER_CMD])
    try:
        build_s = time.perf_counter() - tb
        layout = venv.env_method("describe", indices=[0])[0]["layout"]
        vecnorm = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=d["clip_obs"],
                               clip_reward=d["clip_reward"], gamma=d["gamma"])
        kw = policy.model_kwargs(int(layout["actor_dim"]), log_std_init=math.log(RATE_SIGMA))
        kw.update(learning_rate=schedules.ConstantSchedule(3e-5), n_steps=n_steps, clip_range=d["clip_range"],
                  batch_size=min(int(d["batch_size"]), n_steps * len(specs)))  # (no update runs: no effect on rates)
        model = PPO(env=vecnorm, seed=0, device="cpu", verbose=0, **kw)
        _, callback = model._setup_learn(int(steps), callback=None, reset_num_timesteps=True, progress_bar=False)
        callback.on_training_start(locals(), globals())
        me = os.getpid()

        def cpu():
            try:
                rows = dict(venv.env_method("process_cpu"))
            except Exception:  # noqa: BLE001 - an env without the method: the parent's alone
                rows = {}
            return time.process_time(), {int(p): float(c) for p, c in rows.items() if int(p) != me}

        c0, t0, rollouts = cpu(), time.perf_counter(), 0
        while model.num_timesteps < int(steps):
            if not model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps):
                break
            rollouts += 1
        wall = time.perf_counter() - t0
        c1 = cpu()
        pids = list(getattr(venv, "worker_pids", []) or [])
        n = int(model.num_timesteps)
        return dict(workers=int(workers), n_envs=len(specs), n_steps=n_steps, env_steps=n, rollouts=rollouts,
                    wall_s=wall, env_steps_per_wall_s=n / max(wall, 1e-9), build_s=build_s,
                    cpu_s_parent=c1[0] - c0[0],
                    cpu_s_workers=float(sum(c1[1].get(p, 0.0) - c0[1].get(p, 0.0) for p in c1[1])),
                    rss_mb=dict(parent=_rss_mb(), workers=[_rss_mb(p) for p in pids]), worker_pids=pids,
                    load_average_after=list(os.getloadavg()))
    finally:
        venv.close()


def rates(body: str, set_name: str, floors=("hard", "robot_floor_carpet"), workers=(1, 2, 3, 4), steps=16_384,
          record=None, *, g9_ref=None, root=None, n_steps=None) -> dict:
    """The validation record of rates and memory (``RULES["rates"]``)."""
    s2io.require_one_thread()
    g9 = require_g9_ref(body, set_name, g9_ref, root)
    from designlab.stage2 import train2

    load_before = list(os.getloadavg())
    t0 = time.perf_counter()
    per_floor = {f: _throughput(body, set_name, f, steps) for f in floors}
    cfgs, specs = _rate_specs(body, set_name)
    pool = [rollout_rate(specs, int(w), int(steps), n_steps) for w in workers]
    out = dict(schema=SCHEMAS["rates"], gate="rates", rule=RULES["rates"], body=body, set=set_name,
               floors=list(floors), workers=[int(w) for w in workers], steps=int(steps),
               load_average_before=load_before, provisional=not (load_before[0] < IDLE_LOAD),
               provisional_rule=f"provisional unless the 1-minute load average before the run was under {IDLE_LOAD:g}",
               vcpus=train2.vcpus(), throughput=per_floor, pool=pool,
               env_floors=[c.get("floor") for c in cfgs], set_freeze=_latest_freeze(set_name, body, root), g9_ref=g9,
               sources=sources(), **provenance([record]), wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    out["pass_"] = bool(pool and all(r["env_steps"] >= int(steps) for r in pool))
    return finish(out, record)


# ------------------------------------------------------------------------------------------------
# Kill and resume (rule K3)
# ------------------------------------------------------------------------------------------------
def _flat(prefix: str, x, out: dict) -> None:
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            _flat(f"{prefix}/{k}", x[k], out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _flat(f"{prefix}/{i}", v, out)
    elif hasattr(x, "detach") and hasattr(x, "numpy"):
        out[prefix] = x.detach().cpu().numpy().copy()
    else:
        out[prefix] = np.asarray(x)


def _blank_venv(obs_dim: int, action_dim: int):
    """A vec env of one gymnasium env with a checkpoint's spaces and nothing else: what ``VecNormalize.load``
    binds to (it is never reset or stepped here)."""
    import gymnasium
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _Blank(gymnasium.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (int(obs_dim),), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (int(action_dim),), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(self.observation_space.shape, np.float32), {}

        def step(self, action):
            return np.zeros(self.observation_space.shape, np.float32), 0.0, True, False, {}

    return DummyVecEnv([_Blank])


def checkpoint_items(job_dir, n: int) -> dict:
    """Everything training made that checkpoint ``n`` holds, as arrays by name: ``model/<member>/...`` (the
    archive's policy, optimizer and torch-variable tensors, read with ``torch.load(weights_only=True)``),
    ``vecnormalize/...`` (the statistics, through ``s2load.load_vecnormalize``) and ``fingerprint/<array>``."""
    import torch

    from designlab.stage2 import s2load

    files = s2load.checkpoint_files(job_dir, n)
    rec = s2io.read_json(files["record"])
    out: dict = {}
    with zipfile.ZipFile(files["model"]) as z:
        names = z.namelist()
        for member in ("policy.pth", "policy.optimizer.pth", "pytorch_variables.pth"):
            if member in names:
                state = torch.load(io.BytesIO(z.read(member)), map_location="cpu", weights_only=True)
                _flat(f"model/{member}", state, out)
    lay = rec["layout"]
    vn = s2load.load_vecnormalize(files["vecnormalize"], _blank_venv(lay["obs_dim"], lay["action_dim"]))
    for part in ("obs_rms", "ret_rms"):
        rms = getattr(vn, part)
        out[f"vecnormalize/{part}/mean"] = np.asarray(rms.mean)
        out[f"vecnormalize/{part}/var"] = np.asarray(rms.var)
        out[f"vecnormalize/{part}/count"] = np.asarray(rms.count)
    for key in ("clip_obs", "clip_reward", "gamma", "epsilon", "norm_obs", "norm_reward"):
        out[f"vecnormalize/{key}"] = np.asarray(getattr(vn, key))
    with np.load(files["fingerprint"], allow_pickle=False) as z:
        for k in sorted(z.files):
            out[f"fingerprint/{k}"] = z[k].copy()
    return out


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return a.tolist() == b.tolist()


def _first(a, b) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return dict(dtype=[str(a.dtype), str(b.dtype)], shape=[list(a.shape), list(b.shape)])
    if a.dtype.kind in "fc":
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    else:
        diff = a != b
    idx = np.argwhere(np.atleast_1d(diff))
    at = tuple(int(i) for i in idx[0]) if len(idx) else ()
    av, bv = (np.atleast_1d(a)[at], np.atleast_1d(b)[at]) if len(idx) else (None, None)
    out = dict(index=list(at), a=None if av is None else av.item(), b=None if bv is None else bv.item(),
               differing=int(np.count_nonzero(diff)), of=int(np.size(a)))
    if a.dtype.kind in "fc" and a.size:
        out["max_abs_diff"] = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return out


def compare_items(a: dict, b: dict) -> dict:
    """Two checkpoints' items (``checkpoint_items``) compared name by name, tolerance zero: ``{compared [{name,
    kind, dtype, shape, equal}], equal, first_difference}`` (the first in name order, with its index and both
    values)."""
    names = sorted(set(a) | set(b))
    rows, first = [], None
    for k in names:
        if k not in a or k not in b:
            rows.append(dict(name=k, kind=k.split("/")[0], equal=False, missing_in="a" if k not in a else "b"))
            if first is None:
                first = dict(name=k, missing_in="a" if k not in a else "b")
            continue
        eq = _same(a[k], b[k])
        rows.append(dict(name=k, kind=k.split("/")[0], dtype=str(np.asarray(a[k]).dtype),
                         shape=list(np.asarray(a[k]).shape), equal=eq))
        if not eq and first is None:
            first = dict(name=k, **_first(a[k], b[k]))
    kinds = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r["equal"])
    return dict(compared=rows, items=len(rows), equal=bool(rows and all(r["equal"] for r in rows)),
                by_kind={k: dict(items=len(v), equal=all(v)) for k, v in kinds.items()}, first_difference=first)


def compare_checkpoints(job_a, job_b, n: int) -> dict:
    """Checkpoint ``n`` of two jobs: each checked intact (``s2load.pair_problem``), then ``compare_items``."""
    from designlab.stage2 import s2load

    probs = {k: s2load.pair_problem(j, n) for k, j in (("a", job_a), ("b", job_b))}
    if any(probs.values()):
        return dict(steps=int(n), intact=probs, equal=False, compared=[], first_difference=dict(
            name="integrity", **{k: v for k, v in probs.items() if v}))
    out = compare_items(checkpoint_items(job_a, n), checkpoint_items(job_b, n))
    files = {k: {kind: s2io.sha256_file(p) for kind, p in s2load.checkpoint_files(j, n).items() if kind != "record"}
             for k, j in (("a", job_a), ("b", job_b))}
    return dict(steps=int(n), intact=probs, files_sha256=files,
                files_identical={kind: files["a"][kind] == files["b"][kind] for kind in files["a"]}, **out)


def _child_cmd(kwargs: dict) -> list:
    """kill-resume's child: this module's ``_train`` entry, naming the trainer it runs (so that every process of
    the run carries ``designlab.stage2.train2`` on its command line, as ``tools/memguard.sh``'s PROTECT reads)."""
    return [sys.executable, "-m", "designlab.stage2.gates2", "_train", TRAINER, json.dumps(kwargs, sort_keys=True)]


def _child_env(extra: dict | None = None) -> dict:
    env = tbenv.child_env("disable")
    env.setdefault("GIT_OPTIONAL_LOCKS", "0")
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def _group_gone(pgid: int, wait_s: float = GROUP_WAIT_S) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


def _start(cmd: list, log: Path, env: dict):
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a", encoding="utf-8")  # noqa: SIM115 - closed by _end
    fh.write(f"{s2io.now_utc()} {' '.join(cmd[:5])} ...\n")
    fh.flush()
    p = subprocess.Popen(cmd, cwd=str(s2io.STUDY), env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    return p, fh


def _end(p, fh, timeout_s: float) -> int:
    try:
        code = p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        fh.write(f"{s2io.now_utc()} timeout after {timeout_s:g} s: killed\n")
        fh.close()
        raise GateError(f"a training child ran past {timeout_s:g} s and was killed") from None
    fh.write(f"{s2io.now_utc()} exit {code}\n")
    fh.close()
    _group_gone(p.pid)
    return int(code)


def _progress_steps(path: Path) -> int | None:
    doc = _read(path)
    return None if doc is None or doc.get("steps") is None else int(doc["steps"])


def _running(job_dir: Path) -> int | None:
    """The process that a job's ``progress.json`` names when it is alive and is a Track B trainer (its command line
    names ``designlab.stage2``), else None: a run directory in use is never moved aside."""
    pid = (_read(job_dir / "progress.json") or {}).get("pid")
    if not pid:
        return None
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return None
    return int(pid) if "designlab.stage2" in cmd and int(pid) != os.getpid() else None


def k3_asked(s: dict, body: str, set_name: str, lr=None, config=None) -> dict:
    """The settings kill-resume asks of each of its runs, as ``train2`` records them in ``run.json``
    ``settings`` (``RULES["k3"]``'s reading): ``s`` (``K3``, ``K3_TINY`` or a test's), the body and set, and the
    starting rate and filter the trainer takes when they are not given (``train2._validation``: rules O3, O4).
    ``Refused`` when the trainer cannot run ``s``'s number of envs or rollout length."""
    from designlab.stage2 import train2

    ppo = dict(s.get("ppo") or {})
    tiny = bool(s["tiny"])
    n_steps = int(ppo.get("n_steps", train2.TINY["n_steps"] if tiny else train2.N_STEPS))
    if train2.N_ENVS != int(s["n_envs"]) or n_steps != int(s["n_steps"]):
        raise Refused(f"the settings ask {s['n_envs']} envs of {s['n_steps']} steps per rollout; the trainer runs "
                      f"{train2.N_ENVS} envs of {n_steps}: no run is made (rule K3 fixes both)")
    ck, mon = int(s["checkpoint_every"]), int(s["monitor_every"])
    if tiny:  # (as train2: the tiny run's intervals stand in for the trainer's defaults)
        ck = train2.TINY["checkpoint_every"] if ck == train2.CHECKPOINT_EVERY else ck
        mon = train2.TINY["monitor_every"] if mon == train2.MONITOR_EVERY else mon
    val = train2._validation(body, set_name)
    cfg = dict(config or {})
    if lr is not None:
        lr_start = float(lr)
    elif val.get("lr_probe"):
        lr_start = float(val["lr_probe"]["rate"])
    else:
        lr_start = float(train2.TINY["lr"]) if tiny else None
    hz = cfg.get("filter_hz", (val.get("filter") or {}).get("filter_hz"))
    rollout = train2.N_ENVS * n_steps
    return dict(mode=s["mode"], body=body, seed=int(s["seed"]), set=set_name, n_envs=int(s["n_envs"]), n_steps=n_steps,
                tiny=tiny, gates=bool(s["gates"]), checkpoint_every=ck, monitor_every=mon, config=s2io.plain(cfg),
                ppo_overrides=s2io.plain(ppo), budget=-(-int(s["steps"]) // rollout) * rollout, lr_start=lr_start,
                filter_hz=None if hz is None else float(hz))


def run_settings_problems(doc: dict | None, asked: dict) -> list[str]:
    """Every way a run's ``run.json`` ``settings`` differ from ``asked`` (``k3_asked``); [] when they are the ones
    asked.  A starting rate that cannot be asked (none given, no probe record, not tiny) is a difference."""
    s = (doc or {}).get("settings") or {}
    got = {k: s.get(k) for k in ("mode", "body", "seed", "set", "n_envs", "n_steps", "tiny", "gates",
                                 "checkpoint_every", "monitor_every", "budget")}
    got.update(config=s.get("config") or {}, ppo_overrides=s.get("ppo_overrides") or {},
               lr_start=(s.get("lr") or {}).get("start"))
    out = [f"{k}: {got[k]!r} (asked {asked[k]!r})" for k in got if got[k] != asked[k]]
    if asked["lr_start"] is None:
        out.append("lr_start: no starting rate is asked (no --lr, no lr_probe.json rate, not tiny)")
    if asked["filter_hz"] is not None:
        hz = sorted({c.get("filter_hz") for c in s.get("env_configs") or [] if isinstance(c, dict)}, key=str)
        if hz != [asked["filter_hz"]]:
            out.append(f"filter_hz: {hz!r} (asked {asked['filter_hz']!r})")
    return out


def _aside(path: Path) -> Path:
    k = 0
    while True:
        dest = path.with_name(f"{path.name}.discarded.{k}")
        if not dest.exists():
            path.rename(dest)
            return dest
        k += 1


def kill_resume(body: str, set_name: str, out, record=None, *, tiny=False, settings=None, lr=None, config=None,
                g9_ref=None, root=None, child_cmd=None, child_env=None, hold=False, poll_s=POLL_S,
                timeout_s=CHILD_TIMEOUT_S) -> dict:
    """Rule K3's validation record (``RULES["k3"]``).  ``tiny``: the smoke form ``K3_TINY`` (no record of K3);
    ``settings`` replaces entries of K3's (tests); ``child_cmd(kwargs) -> argv`` and ``child_env`` replace the
    child's command and add to its environment (tests: the stub env of the trainer's tests); ``hold``: the killed
    run waits at the kill step until it is killed (the stub's rollouts last milliseconds).  Resumable: a finished
    run is reused, an interrupted one is moved aside (``<run>.discarded.<k>``) and made again."""
    s2io.require_one_thread()
    g9 = require_g9_ref(body, set_name, g9_ref, root)
    from designlab.stage2 import train2

    s = dict(K3_TINY if tiny else K3, **(settings or {}))
    asked = k3_asked(s, body, set_name, lr, config)
    out = Path(out)
    roots = dict(uninterrupted=out / "uninterrupted", killed=out / "killed")
    job = train2.job_name(s["mode"], body, s["seed"])
    jobs = {k: r / "jobs" / job for k, r in roots.items()}
    cmd = child_cmd or _child_cmd
    env = _child_env(child_env)
    base = dict(mode=s["mode"], body=body, seed=int(s["seed"]), set_name=set_name, steps=int(s["steps"]),
                checkpoint_every=int(s["checkpoint_every"]), monitor_every=int(s["monitor_every"]), gates=bool(s["gates"]),
                tiny=bool(s["tiny"]))
    for k in ("ppo",):
        if s.get(k):
            base[k] = s[k]
    if lr is not None:
        base["lr"] = float(lr)
    if config:
        base["config"] = dict(config)
    t_all = time.perf_counter()
    runs: dict = {}
    w_a, w_b = (int(x) for x in s["workers"])

    def run_doc(name):
        p = jobs[name] / "run.json"
        return _read(p) if p.is_file() else None

    for name, jd in jobs.items():
        pid = _running(jd)
        if pid is not None:
            raise Refused(f"{jd}: its trainer (process {pid}) is still running: one kill-resume at a time")

    # --- the uninterrupted run (one session)
    d = run_doc("uninterrupted")
    why = [] if d is None else run_settings_problems(d, asked)
    if d is not None and (why or not (d.get("status") == "done" and len(d.get("sessions") or []) == 1)):
        runs["uninterrupted_discarded"] = s2io.portable(str(_aside(roots["uninterrupted"])))
        if why:
            runs["uninterrupted_discarded_settings"] = why
        d = None
    if d is None:
        p, fh = _start(cmd(dict(base, out=str(roots["uninterrupted"]), workers=w_a)), out / "uninterrupted.log", env)
        code = _end(p, fh, timeout_s)
        d = run_doc("uninterrupted")
        if code != 0 or d is None or d.get("status") != "done":
            raise GateError(f"the uninterrupted run ended with exit {code} ({out / 'uninterrupted.log'})")
        runs["uninterrupted_made"] = True
    # --- the killed run (two sessions: killed at kill_at, resumed with the other worker count)
    marker = roots["killed"] / "KILLED.json"
    d = run_doc("killed")
    kill = _read(marker) if marker.is_file() else None
    if d is not None:
        n_sess = len(d.get("sessions") or [])
        why = run_settings_problems(d, asked)
        good = not why and kill is not None and kill.get("as_ruled") and (
            (n_sess == 1 and d.get("status") != "done") or (n_sess == 2 and d.get("status") == "done"))
        if not good:
            runs["killed_discarded"] = s2io.portable(str(_aside(roots["killed"])))
            if why:
                runs["killed_discarded_settings"] = why
            d, kill = None, None
    if d is None:
        kw = dict(base, out=str(roots["killed"]), workers=w_a)
        if hold:
            kw["_hold_at"] = int(s["kill_at"])
        p, fh = _start(cmd(kw), out / "killed.log", env)
        prog = jobs["killed"] / "progress.json"
        read, t_kill = None, time.monotonic() + timeout_s
        try:
            while True:
                if p.poll() is not None:
                    break
                n = _progress_steps(prog)
                if n is not None and n >= int(s["kill_at"]):
                    read = n
                    os.killpg(p.pid, signal.SIGKILL)
                    break
                if time.monotonic() > t_kill:
                    os.killpg(p.pid, signal.SIGKILL)
                    break
                time.sleep(poll_s)
        finally:
            code = p.wait()
            fh.write(f"{s2io.now_utc()} exit {code} (progress read {read})\n")
            fh.close()
        gone = _group_gone(p.pid)
        if read is None:
            raise GateError(f"the killed run ended (exit {code}) before its progress read {s['kill_at']} steps "
                            f"({out / 'killed.log'})")
        kill = dict(progress_steps_read=int(read), kill_at=int(s["kill_at"]), as_ruled=read == int(s["kill_at"]),
                    signal="SIGKILL", whole_group=True, pgid=int(p.pid), exit_code=int(code), group_gone=gone,
                    held=bool(hold), at=s2io.now_utc())
        s2io.write_json(marker, kill)
        if not kill["as_ruled"]:
            raise GateError(f"the killed run's progress read {read}, past {s['kill_at']}: no kill as ruled; the next "
                            "call makes that run again")
        d = run_doc("killed")
    if d.get("status") != "done":
        p, fh = _start(cmd(dict(base, out=str(roots["killed"]), workers=w_b, resume=True)), out / "resumed.log", env)
        code = _end(p, fh, timeout_s)
        d = run_doc("killed")
        if code != 0 or d is None or d.get("status") != "done":
            raise GateError(f"the resumed run ended with exit {code} ({out / 'resumed.log'})")
    da, db = run_doc("uninterrupted"), run_doc("killed")
    sa, sb = da["sessions"], db["sessions"]
    resumed_from = sb[1].get("resumed_from") if len(sb) > 1 else None
    budget = int(da["budget"])
    from designlab.stage2 import s2load

    both = sorted(set(s2load.list_checkpoints(jobs["uninterrupted"])) & set(s2load.list_checkpoints(jobs["killed"])))
    steps = [n for n in both if resumed_from is not None and n > int(resumed_from)]
    if budget in both and budget not in steps:
        steps.append(budget)
    compared = [compare_checkpoints(jobs["uninterrupted"], jobs["killed"], n) for n in sorted(set(steps))]
    final = next((c for c in compared if c["steps"] == budget), None)
    first = next((dict(steps=c["steps"], **c["first_difference"]) for c in compared if not c["equal"]), None)
    problems = {k: run_settings_problems(x, asked) for k, x in (("uninterrupted", da), ("killed", db))}
    settings_ok = dict(
        uninterrupted_settings_as_asked=not problems["uninterrupted"], killed_settings_as_asked=not problems["killed"],
        same_settings=bool(da.get("settings_sha256") and da.get("settings_sha256") == db.get("settings_sha256")),
        uninterrupted_one_session=len(sa) == 1, uninterrupted_workers=sa[0].get("workers") == w_a,
        killed_two_sessions=len(sb) == 2, killed_workers=sb[0].get("workers") == w_a,
        resumed_workers=len(sb) == 2 and sb[1].get("workers") == w_b, kill_as_ruled=bool(kill and kill.get("as_ruled")),
        resumed_from_the_last_checkpoint_before_the_kill=resumed_from is not None and int(resumed_from) == (
            int(s["kill_at"]) // int(s["checkpoint_every"])) * int(s["checkpoint_every"]),
        same_budget=int(db["budget"]) == budget, final_compared=final is not None)
    equal = bool(final is not None and all(c["equal"] for c in compared))
    out_doc = dict(
        schema=SCHEMAS["kill_resume"], gate="K3", rule=RULES["k3"],
        form="K3" if (s == K3 and (body, set_name) == K3_ON) else K3_CHECK, body=body, set=set_name,
        settings=dict(s, workers=list(s["workers"])), lr_given=lr, config_given=config, settings_asked=asked,
        settings_problems=problems, job=job,
        out=s2io.portable(str(out.resolve())),
        runs={k: dict(run_uuid=x["run_uuid"], status=x["status"], num_timesteps=x.get("num_timesteps"),
                      budget=x.get("budget"), settings_sha256=x.get("settings_sha256"),
                      lr=(x.get("settings") or {}).get("lr"),
                      filter_hz=sorted({c.get("filter_hz") for c in (x.get("settings") or {}).get("env_configs") or []
                                        if isinstance(c, dict)}, key=str),
                      sessions=[dict(index=q.get("index"), workers=q.get("workers"), resumed_from=q.get("resumed_from"),
                                     fingerprint_max_abs_diff=q.get("fingerprint_max_abs_diff"),
                                     boundary_seeds=q.get("boundary_seeds"), started_at=q.get("started_at"),
                                     ended_at=q.get("ended_at"), error=q.get("error")) for q in x["sessions"]])
              for k, x in (("uninterrupted", da), ("killed", db))},
        made=runs, kill=kill, resumed_from=resumed_from, checkpoints_compared=[c["steps"] for c in compared],
        compared=compared, final_steps=budget, equal=equal, first_difference=first, settings_ok=settings_ok,
        pass_=bool(equal and all(settings_ok.values())),
        followed=None if equal else "a difference: the build stops and goes to the lead (no tolerance is set)",
        g9_ref=g9, set_freeze=_latest_freeze(set_name, body, root), sources=sources(),
        **provenance([record, out]), wall_s=time.perf_counter() - t_all, at=s2io.now_utc())
    return finish(out_doc, record)


def _train_child(argv: list) -> int:
    """kill-resume's child: ``_train <trainer module> <JSON of train's keyword arguments>``; ``_hold_at`` in the
    JSON makes the session wait at that step's rollout start, after its progress.json is written, until it is
    killed."""
    if len(argv) != 2 or argv[0] != TRAINER:
        print(f"usage: python -m designlab.stage2.gates2 _train {TRAINER} JSON", file=sys.stderr)
        return EXIT_ERROR
    kw = json.loads(argv[1])
    hold = kw.pop("_hold_at", None)
    t2 = importlib.import_module(argv[0])
    if hold is not None:
        inner = t2._Run._progress

        def progress(self, n, rate=None, last=None):
            inner(self, n, rate, last)
            if int(n) == int(hold) and self.cb is not None and self.session["ended_at"] is None:
                print(f"held at {n} steps", flush=True)
                time.sleep(CHILD_TIMEOUT_S)

        t2._Run._progress = progress
    try:
        doc = t2.train(log=lambda *x: print(*x, flush=True), **kw)
    except t2.TrainError as exc:
        print(f"train: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return int(exc.code)
    print(json.dumps(dict(status=doc["status"], steps=doc.get("num_timesteps"))), flush=True)
    return 0 if doc["status"] == "done" else 1


# ------------------------------------------------------------------------------------------------
# The training start with MUJOCO_GL set against it; R6's start
# ------------------------------------------------------------------------------------------------
def gl_start(out=None, record=None, *, timeout_s: float = 900.0) -> dict:
    """``RULES["gl_start"]``: ``train2 preflight`` in a child started with ``MUJOCO_GL=osmesa``."""
    t0 = time.perf_counter()
    env = _child_env(dict(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa"))
    cmd = [sys.executable, "-m", TRAINER, "preflight"]
    try:
        r = subprocess.run(cmd, cwd=str(s2io.STUDY), env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise GateError(f"train2 preflight ran past {timeout_s:g} s") from exc
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "gl_start.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
    res = None
    for line in reversed(r.stdout.strip().splitlines()):
        try:
            res = json.loads(line)
            break
        except ValueError:
            continue
    w = (res or {}).get("worker") or {}
    checks = dict(exit_0=r.returncode == 0, ok=bool((res or {}).get("ok")),
                  trainer_disable=(res or {}).get("mujoco_gl") == "disable",
                  trainer_no_platform=(res or {}).get("pyopengl_platform") is None,
                  worker_disable=w.get("mujoco_gl") == "disable", worker_no_platform=w.get("pyopengl_platform") is None,
                  worker_no_torch=w.get("torch") is False)
    doc = dict(schema=SCHEMAS["gl_start"], gate="gl-start", rule=RULES["gl_start"], command=cmd[1:],
               started_with=dict(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa"), exit_code=r.returncode,
               preflight=res, stderr_tail=r.stderr[-2000:] if r.returncode else None, checks=checks,
               pass_=all(checks.values()), sources=sources(), **provenance([record, out]),
               wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    return finish(doc, record)


def _r6_steps(e, actions, n: int) -> dict:
    lo, hi = e.core.lo, e.core.hi
    bad, ends, done = [], 0, 0
    for k in range(int(n)):
        a = actions(k)
        obs, r, term, trunc, _ = e.step(a)
        done += 1
        if not (np.all(e.tk.q_target >= lo) and np.all(e.tk.q_target <= hi)):
            bad.append(dict(step=k, what="a target outside its joint range"))
        if not (0.75 <= float(e.tk.rate_factor) <= 1.25):
            bad.append(dict(step=k, what=f"clock factor {float(e.tk.rate_factor)!r}"))
        if not (np.all(np.isfinite(obs)) and math.isfinite(float(r))):
            bad.append(dict(step=k, what="an observation or the reward not finite"))
        if term or trunc:
            ends += 1
            e.reset()
    return dict(steps=done, episodes_ended=ends, problems=bad[:20], problems_n=len(bad), ok=not bad)


def r6_run_problems(doc: dict | None, body: str, set_name: str, steps: int) -> list[str]:
    """Every way a run's ``run.json`` ``settings`` differ from r6-start's tiny R6 run of ``body`` on ``set_name``
    (``steps``: its budget); [] when they are its."""
    s = (doc or {}).get("settings") or {}
    want = dict(mode="R6", body=body, seed=42, set=set_name, tiny=True, gates=False)
    out = [f"{k}: {s.get(k)!r} (asked {v!r})" for k, v in want.items() if s.get(k) != v]
    budget = s.get("budget") if s.get("budget") is not None else s.get("budget_asked")
    if budget != int(steps):
        out.append(f"budget: {budget!r} (asked {int(steps)})")
    return out


def r6_training(body: str, set_name: str, work, workers: int = 2, train=None) -> dict:
    """r6-start's tiny R6 run (``RULES["r6_start"]``) under ``work``: a run already there is reused (resumed when it
    is not done) only when its settings are this one's (``r6_run_problems``), else it is moved aside
    (``<job>.discarded.<k>``) and made again; ``Refused`` while its trainer is alive.  ``train``: ``train2.train``
    (tests: a stand-in)."""
    from designlab.stage2 import s2load, train2

    train = train or train2.train
    work = Path(work)
    rollout = train2.N_ENVS * train2.TINY["n_steps"]
    steps = R6_ROLLOUTS * rollout
    job = train2.job_name("R6", body, 42)
    jd = work / "jobs" / job
    made: dict = {}
    if (jd / "run.json").is_file():
        pid = _running(jd)
        if pid is not None:
            raise Refused(f"{jd}: its trainer (process {pid}) is still running")
        why = r6_run_problems(_read(jd / "run.json"), body, set_name, steps)
        if why:
            made = dict(discarded=s2io.portable(str(_aside(jd))), discarded_settings=why)
    resume = (jd / "run.json").is_file()
    doc = (_read(jd / "run.json") or {}) if resume else {}
    if doc.get("status") != "done":
        doc = train("R6", body, 42, set_name=set_name, out=str(work), steps=steps, tiny=True, gates=False,
                    workers=int(workers), resume=resume, run_id="r6-start", log=lambda *x: None)
    n = int(doc.get("num_timesteps") or 0)
    ups = train2.read_updates(jd / "updates.jsonl") if (jd / "updates.jsonl").is_file() else []
    try:
        ck = s2load.load_checkpoint(jd, n)
        fp = float(ck["fingerprint_max_abs_diff"])
    except Exception as exc:  # noqa: BLE001
        fp = None
        ck_err = f"{type(exc).__name__}: {exc}"
    else:
        ck_err = None
    return dict(job=job, out=s2io.portable(str(work.resolve())), status=doc.get("status"), steps=n, budget=steps,
                rollouts=R6_ROLLOUTS, updates=len(ups), minibatch_steps=[u.get("minibatch_steps_run") for u in ups],
                fingerprint_max_abs_diff=fp, checkpoint_error=ck_err, made=made,
                settings_problems=r6_run_problems(doc, body, set_name, steps),
                ok=bool(doc.get("status") == "done" and n == steps and fp is not None and fp <= s2load.FINGERPRINT_TOL
                        and not r6_run_problems(doc, body, set_name, steps)))


def r6_start(body: str, set_name: str, record=None, *, out=None, workers: int = 2, g9_ref=None, root=None) -> dict:
    """``RULES["r6_start"]``."""
    s2io.require_one_thread()
    g9 = require_g9_ref(body, set_name, g9_ref, root)
    from designlab.stage2 import r5env

    t0 = time.perf_counter()
    cfg = r5env.env_configs("R6", body, set_name, 1)[0]
    e = r5env.make("R6", cfg, seed=0)
    nj = e.nj
    rng = np.random.default_rng(6)
    try:
        e.reset(seed=1)
        zero = _r6_steps(e, lambda k: np.zeros(nj + 1, np.float32), R6_STEPS)
        e.reset(seed=2)
        rand = _r6_steps(e, lambda k: rng.uniform(-1.0, 1.0, nj + 1).astype(np.float32), R6_STEPS)
        task = next((t for t in e.tasks if e.scripts[t].mid_spans()), None)
        mid = dict(task=task, ok=False)
        if task is not None:
            _, info = e.reset(seed=3, options=dict(task=task, start="mid"))
            t_s = float(info["t_start"])
            acts = e.core.reference_actions(t_s)
            want = np.clip(e.script.track.joint(t_s)[:nj], e.core.lo, e.core.hi)
            hist_ok = bool(np.array_equal(e.core.hist, acts) and np.array_equal(e.core.a_f, acts[0]))
            target_ok = bool(np.array_equal(e.tk.q_target, want))
            steps = _r6_steps(e, lambda k: np.zeros(nj + 1, np.float32), 50)
            mid = dict(task=task, start=info.get("start"), t_start=t_s, history_is_reference=hist_ok,
                       targets_are_reference=target_ok, steps=steps,
                       ok=bool(info.get("start") == "mid" and t_s > 0 and hist_ok and target_ok and steps["ok"]))
        span = [float(x) for x in e.span]
    finally:
        e.close()
    work = Path(out) if out is not None else WORK / "r6_start"
    training = r6_training(body, set_name, work, workers)
    checks = dict(zero=zero["ok"], random=rand["ok"], mid=mid["ok"], training=training["ok"])
    out_doc = dict(schema=SCHEMAS["r6_start"], gate="r6-start", rule=RULES["r6_start"], body=body, set=set_name,
                   env_config=cfg.describe(), span=span, zero_action=zero, random_actions=rand, mid_start=mid,
                   training=training, checks=checks, pass_=all(checks.values()), g9_ref=g9,
                   set_freeze=_latest_freeze(set_name, body, root), sources=sources(), **provenance([record, work]),
                   wall_s=time.perf_counter() - t0, at=s2io.now_utc())
    return finish(out_doc, record)


# ------------------------------------------------------------------------------------------------
# Rule X1's list of check episodes
# ------------------------------------------------------------------------------------------------
DEV_KEY = ("step", "set", "body")


def dev_episode(path, entry: dict) -> dict:
    """Rule X1: list one closed-loop step on a look set that is no record in ``path``
    (``stage2/tb/dev_episodes.json``), keyed by its step, set and body: a later call for the same step updates the
    entry (its state ``done`` after ``listed``), so a resumed chain adds no duplicate.  Whatever else the file holds
    is kept."""
    path = Path(path)
    doc = _read(path) if path.is_file() else None
    doc = doc or dict(schema=SCHEMAS["dev_episodes"], rule=RULES["x1"],
                      note="check episodes on the look sets that are no record (rule X1): each listed by "
                           "stage2/tb/run_trackb.sh before its step runs, and again when it is done", episodes=[])
    eps = doc.setdefault("episodes", [])
    key = tuple(entry.get(k) for k in DEV_KEY)
    row = dict(entry)
    stamp = "done_at" if row.get("state") == "done" else "listed_at"
    row[stamp] = s2io.now_utc()
    row.setdefault("commit", s2io.git_state().get("commit"))
    for i, e in enumerate(eps):
        if tuple(e.get(k) for k in DEV_KEY) == key:
            eps[i] = dict(e, **row)
            break
    else:
        eps.append(row)
    s2io.write_json(path, doc)
    return doc


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _csv(text) -> list[str] | None:
    return None if text is None else [x for x in (s.strip() for s in str(text).split(",")) if x]


def _ints(text) -> tuple | None:
    xs = _csv(text)
    return None if xs is None else tuple(int(x) for x in xs)


def _say(doc: dict, lines=()) -> None:
    for line in lines:
        print(line, flush=True)
    print(f"{doc['gate']}: {'PASS' if doc.get('pass_') else 'FAIL'} (record sha256 {doc['sha256']})", flush=True)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_train"]:
        return _train_child(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.gates2", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("g8a", "g9-ref", "g9-run", "rates", "kill-resume", "r6-start"):
        a = sub.add_parser(name)
        a.add_argument("--body", required=True)
        a.add_argument("--set", dest="set_name", required=True)
        a.add_argument("--record", default=None, help="the record's path (none: nothing is written)")
        a.add_argument("--set-root", default=None)
        if name in ("g8a", "rates", "kill-resume", "r6-start"):
            a.add_argument("--g9-ref", default=None, help="the G9-ref record (default stage2/tb/gates/<set>/<body>)")
        if name == "g8a":
            a.add_argument("--conditions", default=None)
            a.add_argument("--seeds", default=None)
            a.add_argument("--delays", default=None, help="physics steps (0,5)")
            a.add_argument("--cut-s", type=float, default=None)
        if name == "g9-ref":
            a.add_argument("--refcheck", default=None)
        if name == "g9-run":
            a.add_argument("--bar-out", required=True)
            a.add_argument("--stage1b-reread", default=None)
            a.add_argument("--block", default=None)
        if name == "rates":
            a.add_argument("--floors", default="hard,robot_floor_carpet")
            a.add_argument("--workers", default="1,2,3,4")
            a.add_argument("--steps", type=int, default=16_384)
        if name == "kill-resume":
            a.add_argument("--out", required=True)
            a.add_argument("--tiny", action="store_true")
            a.add_argument("--lr", type=float, default=None)
            a.add_argument("--config", default=None)
        if name == "r6-start":
            a.add_argument("--out", default=None)
            a.add_argument("--workers", type=int, default=2)
    a = sub.add_parser("gl-start")
    a.add_argument("--out", default=None)
    a.add_argument("--record", default=None)
    a = sub.add_parser("dev-episode")
    for k in ("--file", "--step", "--command", "--set", "--body", "--what", "--reason"):
        a.add_argument(k, required=True)
    a.add_argument("--count", type=int, required=True)
    a.add_argument("--state", default="listed", choices=("listed", "done"))
    args = ap.parse_args(argv)
    try:
        if args.cmd == "dev-episode":
            dev_episode(args.file, dict(step=args.step, command=args.command, set=args.set, body=args.body,
                                        rows_or_mode=args.what, count=args.count, reason=args.reason, state=args.state))
            print(f"dev episodes: {args.step} {args.set} {args.body} {args.state}", flush=True)
            return EXIT_PASS
        if args.cmd == "gl-start":
            doc = gl_start(args.out, args.record)
            _say(doc, [f"gl-start: {k} {v}" for k, v in doc["checks"].items() if not v])
        elif args.cmd == "g8a":
            doc = g8a(args.body, args.set_name, args.record, conditions=_csv(args.conditions), seeds=_ints(args.seeds),
                      delays=_ints(args.delays), cut_s=args.cut_s, root=args.set_root, g9_ref=args.g9_ref)
            _say(doc, g8a_lines(doc))
        elif args.cmd == "g9-ref":
            doc = g9_ref(args.body, args.set_name, args.record, root=args.set_root, refcheck=args.refcheck)
            _say(doc, g9_ref_lines(doc))
        elif args.cmd == "g9-run":
            doc = g9_run(args.body, args.set_name, args.bar_out, args.stage1b_reread, args.record, block=args.block,
                         root=args.set_root)
            d = doc["parts"]["d"]
            _say(doc, [f"G9-run {doc['set']} {doc['body']}: (d) {d['ok']} ({len(d['records'])} companion records, "
                       f"missing {d['missing']}), (e) {doc['parts']['e']['ok']}"])
        elif args.cmd == "rates":
            doc = rates(args.body, args.set_name, tuple(_csv(args.floors)), _ints(args.workers), args.steps, args.record,
                        g9_ref=args.g9_ref, root=args.set_root)
            _say(doc, [f"rates W={r['workers']}: {r['env_steps_per_wall_s']:.0f} env steps/s" for r in doc["pool"]]
                 + [f"provisional: {doc['provisional']} (load {doc['load_average_before'][0]:.2f})"])
        elif args.cmd == "kill-resume":
            doc = kill_resume(args.body, args.set_name, args.out, args.record, tiny=args.tiny, lr=args.lr,
                              config=json.loads(args.config) if args.config else None, g9_ref=args.g9_ref,
                              root=args.set_root)
            f = doc["first_difference"]
            _say(doc, [f"kill-resume ({doc['form']}): checkpoints {doc['checkpoints_compared']} equal {doc['equal']}"
                       + ("" if f is None else f"; first difference: {f}")]
                 + [f"  setting not met: {k}" for k, v in doc["settings_ok"].items() if not v])
        else:
            doc = r6_start(args.body, args.set_name, args.record, out=args.out, workers=args.workers,
                           g9_ref=args.g9_ref, root=args.set_root)
            _say(doc, [f"r6-start: {k} {v}" for k, v in doc["checks"].items()])
        return EXIT_PASS if doc.get("pass_") else EXIT_FAIL
    except Refused as exc:
        print(f"gates2 {args.cmd}: refused: {exc}", file=sys.stderr, flush=True)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - any fault is an error of the gate (exit 2), never a pass or a fail
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(f"gates2 {args.cmd}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return EXIT_ERROR


__all__ = ["G8A_CONDITIONS", "GateError", "K3", "K3_TINY", "LOOK_SETS", "RULES", "Refused", "SCHEMAS",
           "checkpoint_items", "compare_checkpoints", "compare_items", "controls", "dev_episode", "finish", "g8a",
           "g9_ref", "g9_run", "gl_start", "inputs_of", "joins", "k3_asked", "kill_resume", "main", "part_a", "part_b",
           "part_c", "part_d", "part_e", "part_f", "played_files", "r6_run_problems", "r6_start", "r6_training", "rates",
           "record_sha_ok", "require_g9_ref", "rollout_rate", "run_settings_problems", "sources", "still_available",
           "tail_tenth", "walk_steps"]

if __name__ == "__main__":
    from designlab.stage2 import gates2 as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
