"""Stage 1 tuning: the CMA-ES fit of the tracker's gains (``STAGE1_PLAN.md`` section 2, the control / estimator / tune
row; section 3 step 6, gate G5; section 6, budget and seeds).

What is fitted: R2 (the tracker on the estimated state, ``control``), 13 of the 14 gains of ``control.GAIN_SPEC``
(``SEARCHED``) from the tuning start (decision 16), each searched in its CMA-ES bounds scaled to [0, 1].  The
fourteenth, the capture step's ``capture_margin``, is held at its start value 0.0 (``FIXED_GAINS``, ``FIXED_RULE``):
the capture step acts only while standing and the mix has no standing push, so the objective has nothing to fit it on;
the gated push panels later use 0.0.

The objective (``OBJECTIVE``, recorded in every output; defined here before any run): design_1.md's tuning row sketches
an evaluation (47 simulated s on hard floor: the nominal walk with a 0.4x push, Fr 0.05, both turns, start and stop;
plus 8 s on low pile; "time survived first, then tracking and drive-duty penalties") without a formula, so the
objective takes its order (survival first, then tracking and the limits, drive duty among them) and plan section 6's
budget unit (about 55 simulated s) over a fixed mix of the behaviours G5 judges, plus start and stop.  Every evaluation
is three episodes on hard floor (mu 0.9), each from the settled stance (``EPISODES``, ``episode_plan``):

* ``walk``: start, the nominal walk ``WALK_STRIDES`` strides, stop;
* ``walking_turn``: start, the 0.35 rad/s walking turn's entry, ``TURN_STRIDES`` strides of it and its exit (at least
  180 deg of steady turning between the entry's ramp and the exit's), stop;
* ``in_place_turn``: the 0.5 rad/s in-place turn's start, ``IN_PLACE_STRIDES`` strides of it, its stop (at least 180
  deg of steady turning).

The two turns run in opposite directions, chosen by the episode seed's parity (even: the walking turn left, the
in-place turn right).  Every speed is commanded ``COMMAND_MARGIN`` (5 %) above its target, plan section 4's "commanded
5 % above" for the walks, applied to the turns' yaw rates as well (their references run exactly at the yaw-rate
targets, which section 4 asks to meet or beat): the track runs at the clock rate 1.05 (``control.COMMAND_RATE_RULE``).
The mix is the same number of strides on every body: 57 strides, 55.4 / 55.1 / 61.6 simulated s on 60-N-T-10 /
60-N-H-10 / 75-M-H-10 at the commanded rate (75-M-H-10's stride is 11 % longer).  Each episode's start is perturbed as
design_1.md's randomisation lists it, where the plant models it (``START_JOINT_NOISE_RAD``, ``START_ROOT_VELOCITY_
NOISE_M_S``; its trunk mass and latency are not modelled), and R2's sensors carry the default ``estimator.SensorNoise``;
both are drawn from seeds derived from the episode seed (``derived_seed``).

Score of an evaluation (lower is better; ``score``): J = S + P, where

* S = sum over the episodes that fell of 1 + the share of the episode not survived: a fall costs at least 1, more the
  earlier it comes;
* P = 0.5 T + 0.5 L in [0, 1]: T the mean of the tracking terms, L the mean over the behaviour groups (walk, walking
  turn, in-place turn, starts, stops) of (largest + mean) / 2 of the group's limit terms.  Every term lies in [0, 1]
  (``TERMS_RULE``): tracking terms are errors over a stated scale (the true DCM error against the reference, speed and
  yaw-rate errors against the command, heading change, in-place drift); limit terms are plan section 4's limits as
  hinges, 0 while a limit holds and 1 once it is broken by the limit's own size (or a stated scale), each read as G5's
  success reads it (``SUCCESS_RULE``: the walk and turn limits and the drives' motor-space RMS over the same window,
  the drives' peak speed and the joint range over the episode).  A fallen episode scores 1 on every term.

So a fall always outweighs any tracking or limit penalty.  The CMA-ES fitness adds the box repair: candidates are
evaluated at their projection into the bounds, and the squared distance to it (in the [0, 1] scaling) is added.

Failures (``FAILURE_RULE``): an evaluation that raises scores as the worst case (every episode a fall at its start,
J = 7) with its traceback in the evaluation log; an episode during which MuJoCo warned (``SIM_WARNINGS``: a bad
position, velocity, acceleration or control, after which MuJoCo resets the state and runs on; a full contact or
constraint buffer; a singular inertia) counts as a fall at its start; a worker process that dies or does not answer
within ``EVAL_TIMEOUT_S`` is replaced and its evaluation run once more, and one that fails twice scores as the worst
case with the reason (``Pool``).

CMA-ES (``CMA``, numpy only): (mu/mu_w, lambda) with ``POPSIZE`` 12 (the default 4 + 3 ln n is 11 for n = 13; 12 makes
720 an exact 60 generations), the standard weights and learning rates (Hansen 2016), the step size ``SIGMA0`` 0.1 of
each gain's bound range (a local refinement from a start that already passes G4).  Seeds 42 and 43 (plan section 6)
seed its sampling and the order of the training episodes: generation g is evaluated on training episode 1000 + perm[g]
(perm a permutation of 0-999 drawn from the seed), every member on the same episode (common random numbers).  The state
is checkpointed after every generation (``tune/<seed>/cma.json``, written atomically after the generation's evaluations
are appended to ``tune/<seed>/evals.jsonl``); ``--resume`` continues from it exactly and drops evaluations logged after
the last checkpoint.  A resume is refused if the settings, the libraries, the tuner's sources (``TUNER_SOURCES_RULE``:
the designlab/stage1 modules the tuner imports, and only those, so other files may be added while a fit runs) or its
inputs (``INPUTS_RULE``: every reference and transition file it loads, checked against the sha256 that refs.json and
transitions.json record, and those two files) differ from the run's first session.

Selection and G5 (``g5``): the candidates of a seed are the tuning start and the CMA-ES mean (projected into the
bounds) after each third of its generations; each is evaluated on the 20 validation episodes 2000-2019 and they are
ranked lexicographically (``SELECTION_RULE``): fewer falls first, then the lower mean S, then the lower mean P; the
lower mean J and then the earlier candidate break ties.  G5 (plan section 3 step 6): the selected gains succeed on at
least ``G5_MIN_SUCCESS`` (17) of the 20 validation episodes of each condition: the nominal walk, the walking turn and
the in-place turn, each judged by plan section 4's success criteria (``SUCCESS_RULE``).  A seed that fails is labelled
controller-limited.  G5 judges a seed only if its fit ran on the tuner sources and the inputs G5 runs on
(``sources_match``, ``inputs_match``).  Beside the selection it reports, as a diagnostic, the selected gains with the
four gains that start on their lower bound (the heading and phase-rate terms) set to zero (``ZEROED_RULE``).
``tune.json`` holds the result.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import wait as _wait
from pathlib import Path

import mujoco
import numpy as np

from . import BODIES, STUDY, control, metrics, plant
from .refs import Reference
from .transitions import Transition, stop_command_phase

RUN = STUDY / "stage1" / "runs" / "2026-11-08"
SCHEMA = 2
MODE = "R2"
BUDGET = 720
SEEDS = (42, 43)
TRAIN_EPISODES = (1000, 2000)  # [first, last)
VALIDATION_EPISODES = (2000, 2020)
POPSIZE = 12
SIGMA0 = 0.1
BOX_PENALTY = 1.0
COMMAND_MARGIN = 0.05
COMMAND_RATE = 1.0 + COMMAND_MARGIN
START_JOINT_NOISE_RAD = 0.01
START_ROOT_VELOCITY_NOISE_M_S = 0.02
WALK_STRIDES = 7
TURN_STRIDES = 5
IN_PLACE_STRIDES = 3
WALK_WINDOW_S = 10.0
LEGS_EVERY = 2  # feedback updates between the poses of the leg-to-leg check (50 Hz)
G5_MIN_SUCCESS = 17
G5_RUNS = 20
EPISODES = ("walk", "walking_turn", "in_place_turn")
CONDITIONS = {"walk": "nominal walk", "walking_turn": "0.35 rad/s walking turn", "in_place_turn": "0.5 rad/s in-place turn"}
TURN = {"walking_turn": "walk_0.35", "in_place_turn": "in_place_0.50"}

# ------------------------------------------------------------------------------------------------
# The searched gains
# ------------------------------------------------------------------------------------------------
#: Gains the fit holds at their tuning start (decision 16) instead of searching them (``FIXED_RULE``).
FIXED_GAINS = {"capture_margin": 0.0}
FIXED_RULE = (
    "capture_margin is held at its tuning start, 0.0 (decision 16), and not searched: the capture step acts only "
    "while the reference stands, and the fit's mix has no standing push (its only stands are the starts' first and "
    "the stops' last moments, under the start noise), so the objective has nothing to fit the margin on; the gated "
    "push panels later use 0.0")
SEARCHED = tuple(n for n in control.GAIN_NAMES if n not in FIXED_GAINS)
_SEARCHED_IX = np.array([control.GAIN_NAMES.index(n) for n in SEARCHED])
if any(FIXED_GAINS[n] != control.gains_dict()[n] for n in FIXED_GAINS):
    raise ValueError("FIXED_GAINS must hold the tuning start's values (decision 16)")
#: The searched gains whose tuning start is their lower bound (decision 16: the heading and phase-rate gains at zero).
LOWER_BOUND_START = tuple(n for n, i in zip(SEARCHED, _SEARCHED_IX, strict=True)
                          if control.START_GAINS[i] == control.GAIN_BOUNDS[i, 0])
ZEROED_RULE = (
    "a diagnostic beside the selection, not judged: the selected gains with the searched gains whose tuning start is "
    f"their lower bound ({', '.join(LOWER_BOUND_START)}: the heading and phase-rate terms, zero at the start, decision "
    "16) set to zero, on the same validation episodes; CMA-ES starts these on their bound (a sample below it is "
    "evaluated at the bound), so this shows what the fitted values of those terms add")

# ------------------------------------------------------------------------------------------------
# Plan section 4's limits and the objective's scales
# ------------------------------------------------------------------------------------------------
ROLL_PITCH_RMS_DEG = 4.0
ROLL_PITCH_PEAK_DEG = 8.0
DUTY = (0.55, 0.65)
CADENCE = (1.6, 2.2)
STRIDE_H = (0.5, 0.65)
GRF_BW = 1.6
TOE_H = 0.04
SLIP_MEDIAN_M = 0.005
SLIP_WORST_M = 0.015
DRIFT_M = 0.1
WALK_RANGE_RAD = 0.10
TURN_RANGE_SLACK_RAD = 0.02  # decision 14: the reference's own smallest margin less this
AT_LIMIT_RAD = 1e-3  # a joint this close to its range is at its limit (control's joint_range hardware limit)
START_WITHIN_STRIDES = 2.0
STOP_WITHIN_STRIDES = 1.5
ACCEL_M_S2 = 1.5
#: Scales of the objective's terms (each term is min(1, excess / scale)).
SCALES = dict(dcm_m=0.02, speed_frac=0.1, heading_rad=0.1, yaw_rate_frac=0.1, drift_m=0.1, flight_s=0.05,
              leg_interference_m=0.005, joint_range_rad=0.10, duty=0.05, cadence=0.3, stride_H=0.075,
              speed_limit_frac=0.05, yaw_rate_limit_frac=0.05)
TRACKING = {"walk": ("dcm", "speed", "heading"), "walking_turn": ("dcm", "yaw_rate"),
            "in_place_turn": ("dcm", "yaw_rate", "drift")}
WINDOW_LIMITS = ("roll_rms", "pitch_rms", "roll_peak", "pitch_peak", "duty_left", "duty_right", "flight", "grf",
                 "toe_clearance", "slip_median", "slip_worst")
EPISODE_LIMITS = ("drive_sustained", "drive_speed", "joint_range")
LIMITS = {"walk": WINDOW_LIMITS + ("cadence", "stride", "speed") + EPISODE_LIMITS,
          "walking_turn": WINDOW_LIMITS + ("yaw_rate", "leg_interference") + EPISODE_LIMITS,
          "in_place_turn": WINDOW_LIMITS + ("yaw_rate", "drift", "leg_interference") + EPISODE_LIMITS}
START_TERMS = ("speed_up", "acceleration")
STOP_TERMS = ("stopped", "acceleration")
WITH_START_STOP = ("walk", "walking_turn")
SUCCESS = {
    "walk": ("survived", "speed", "duty", "no_flight", "cadence", "stride", "grf", "roll_rms", "pitch_rms",
             "roll_peak", "pitch_peak", "toe_clearance", "slip_median", "slip_worst", "drive_sustained", "drive_speed",
             "joint_range"),
    "walking_turn": ("survived", "yaw_rate", "duty", "no_flight", "grf", "roll_rms", "pitch_rms", "roll_peak",
                     "pitch_peak", "toe_clearance", "slip_median", "slip_worst", "drive_sustained", "drive_speed",
                     "joint_range", "no_leg_interference"),
    "in_place_turn": ("survived", "yaw_rate", "drift", "duty", "no_flight", "grf", "roll_rms", "pitch_rms",
                      "roll_peak", "pitch_peak", "toe_clearance", "slip_median", "slip_worst", "drive_sustained",
                      "drive_speed", "joint_range", "no_leg_interference"),
}
TERMS_RULE = (
    "tracking: dcm = RMS of the true DCM error against the reference over the episode (relative to the anchor foot, "
    f"as the feedback's) / {SCALES['dcm_m']:g} m; walk speed = |10 s mean speed - command| / ({SCALES['speed_frac']:g} "
    f"command); walk heading = |heading change over the 10 s window| / {SCALES['heading_rad']:g} rad; turn yaw_rate = "
    f"|yaw rate over the 180 deg turn - command| / ({SCALES['yaw_rate_frac']:g} command); in-place drift = the root's "
    "displacement from the episode's first stand to its last (the in-place reference turns about the root, which "
    "stays put; its CoM, 34-67 mm ahead of the root, swings 66-131 mm round a 204 deg turn) / "
    f"{SCALES['drift_m']:g} m.  Limits (plan section 4) as hinges, 0 within, read over the window (the walk's 10 s "
    f"before the stop command, the turns' steady turning): roll and pitch RMS about home over {ROLL_PITCH_RMS_DEG:g} "
    f"deg and peaks over {ROLL_PITCH_PEAK_DEG:g} deg (scale: the limit); each foot's duty outside "
    f"{DUTY[0]:g}-{DUTY[1]:g} (scale {SCALES['duty']:g}); flight time (scale {SCALES['flight_s']:g} s); peak GRF over "
    f"{GRF_BW:g} BW (scale: the limit); toe clearance under {TOE_H:g} H (scale: the limit); slip median over "
    f"{1e3 * SLIP_MEDIAN_M:g} mm and worst over {1e3 * SLIP_WORST_M:g} mm (scale: the limit); walk cadence outside "
    f"{CADENCE[0]:g}-{CADENCE[1]:g} steps/s (scale {SCALES['cadence']:g}), stride outside "
    f"{STRIDE_H[0]:g}-{STRIDE_H[1]:g} H (scale {SCALES['stride_H']:g} H), 10 s mean speed under the target (scale "
    f"{SCALES['speed_limit_frac']:g} target); turn yaw rate under the target (scale "
    f"{SCALES['yaw_rate_limit_frac']:g} target); every drive's motor-space RMS over the window above its sustained "
    f"torque (scale: the limit).  Over the episode: in-place drift over {DRIFT_M:g} m (scale: the limit); leg "
    f"interference depth (scale {1e3 * SCALES['leg_interference_m']:g} mm); every drive's peak speed above "
    "speed_at(24 V) (scale: the limit); the joint range's shortfall under the required margin "
    f"(walks {WALK_RANGE_RAD:g} rad; turns decision 14: min({WALK_RANGE_RAD:g}, the episode track's own smallest margin "
    f"of that joint - {TURN_RANGE_SLACK_RAD:g}), at least {AT_LIMIT_RAD:g}; scale {SCALES['joint_range_rad']:g} rad); "
    f"start: strides to 90 % of the commanded speed over {START_WITHIN_STRIDES:g} (scale: the limit; not reached: 1); "
    f"stop: decision 18's strides to stopped over {STOP_WITHIN_STRIDES:g} (scale: the limit; not reached: 1); start "
    f"and stop forward acceleration low-passed at 1 Hz over {ACCEL_M_S2:g} m/s^2 (decision 10; scale: the limit)")
SUCCESS_RULE = (
    "plan section 4 on each validation episode: the nominal walk succeeds if it survives and over the 10 s before the "
    f"stop command its mean speed is at least the nominal speed, each foot's duty is {DUTY[0]:g}-{DUTY[1]:g}, there "
    f"is no flight, the cadence is {CADENCE[0]:g}-{CADENCE[1]:g} steps/s, the stride {STRIDE_H[0]:g}-{STRIDE_H[1]:g} "
    f"H, the peak GRF at most {GRF_BW:g} BW, roll and pitch about home at most {ROLL_PITCH_RMS_DEG:g} deg RMS and "
    f"{ROLL_PITCH_PEAK_DEG:g} deg peak, the toe clearance at least {TOE_H:g} H, the slip at most "
    f"{1e3 * SLIP_MEDIAN_M:g} mm median and {1e3 * SLIP_WORST_M:g} mm worst and every drive's motor-space RMS at most "
    "its sustained torque (the drive RMS is judged over the same window as the other limits), and over the whole "
    "episode every drive's peak speed is at most speed_at(24 V) and every joint keeps "
    f"{WALK_RANGE_RAD:g} rad of its range.  A turn succeeds if it survives, its yaw rate over the 180 deg turn is at "
    "least the target, the walk limits (duty, flight, GRF, roll and pitch, toe clearance, slip) and the drive RMS hold "
    "over its steady turning (the same window), the drives' peak speeds hold over the episode, every joint keeps "
    "decision 14's margin, the legs do not interfere (legs.LegGap at "
    f"50 Hz) and, in place, the drift (the root's displacement) is at most {DRIFT_M:g} m.  Cadence and stride are judged "
    "at nominal only (the Walk row's 'at nominal').  The measures are metrics' (its docstring): toe clearance on the "
    "reference's swing clock, flight on the 20 ms debounced contact, slip as the sole centre's largest displacement "
    "over each loaded stance, the drive RMS over the window's physics steps")


def hinge(x, limit: float, scale: float) -> float:
    """min(1, max(0, x - limit) / scale); None (not measured) scores 1."""
    if x is None or not math.isfinite(x):
        return 1.0
    return float(min(1.0, max(0.0, (x - limit) / scale)))


def hinge_below(x, limit: float, scale: float) -> float:
    if x is None or not math.isfinite(x):
        return 1.0
    return float(min(1.0, max(0.0, (limit - x) / scale)))


def band(x, lo: float, hi: float, scale: float) -> float:
    if x is None or not math.isfinite(x):
        return 1.0
    return float(min(1.0, max(0.0, lo - x, x - hi) / scale))


def error(x, scale: float) -> float:
    if x is None or not math.isfinite(x):
        return 1.0
    return float(min(1.0, abs(x) / scale))


def derived_seed(episode_seed: int, episode: int, purpose: int) -> int:
    """A seed for one episode of an evaluation (0: the start perturbation, 1: the sensor noise)."""
    return int(np.random.SeedSequence([int(episode_seed), int(episode), int(purpose)]).generate_state(1)[0])


def directions(episode_seed: int) -> dict:
    """The turns' directions of an episode seed: even, the walking turn left and the in-place turn right; odd, the
    reverse."""
    a, b = ("left", "right") if int(episode_seed) % 2 == 0 else ("right", "left")
    return {"walk": None, "walking_turn": a, "in_place_turn": b}


def episode_plan(kind: str, direction: str | None) -> list:
    """The track of an episode of the mix (module docstring)."""
    if kind == "walk":
        return [("transition", "start"), ("ref", "nominal", WALK_STRIDES), ("transition", "stop")]
    if kind == "walking_turn":
        c = f"{TURN[kind]}_{direction}"
        return [("transition", "start"), ("transition", f"enter_{c}"), ("ref", c, TURN_STRIDES),
                ("transition", f"exit_{c}"), ("transition", "stop")]
    if kind == "in_place_turn":
        c = f"{TURN[kind]}_{direction}"
        return [("transition", f"start_{c}"), ("ref", c, IN_PLACE_STRIDES), ("transition", f"stop_{c}")]
    raise ValueError(f"episode {kind!r}: choose from {', '.join(EPISODES)}")


# ------------------------------------------------------------------------------------------------
# Inputs: the reference and transition files the mix loads
# ------------------------------------------------------------------------------------------------
#: The run directory's records of its reference and transition files, with the key that holds their entries.
INPUT_RECORDS = (("refs.json", "references"), ("transitions.json", "transitions"))
INPUTS_RULE = (
    "before the evaluator loads them, every refs/*.npz and transitions/*.npz file the mix uses (the episodes' tracks, "
    "both directions of each turn, and refs/nominal.npz, which sets omega0) is hashed (sha256) and checked against "
    "the sha256 that refs.json or transitions.json records for it, and both JSON files are hashed; a mismatch stops "
    "the build; every session records {file: sha256} and the JSON files' hashes, --resume is refused on any difference "
    "from the run's first session, and G5 judges a seed only on the inputs its fit used")


class InputError(RuntimeError):
    """An input file is not the one its record names, or the inputs differ from the ones expected."""


def mix_files() -> list[str]:
    """The reference and transition files the evaluator loads, relative to the body's run directory
    (``INPUTS_RULE``)."""
    files = {"refs/nominal.npz"}
    for kind in EPISODES:
        for d in ((None,) if kind == "walk" else ("left", "right")):
            for item in episode_plan(kind, d):
                if item[0] == "ref":
                    files.add(f"refs/{item[1]}.npz")
                elif item[0] == "transition":
                    files.add(f"transitions/{item[1]}.npz")
    return sorted(files)


def read_inputs(run_dir: Path, files=None) -> tuple[dict, dict]:
    """``INPUTS_RULE`` on ``files`` (default ``mix_files``) of ``run_dir``: the record ({file: sha256} and the two
    JSON files' sha256) and the two JSON documents as read; ``InputError`` if a file is missing, is not listed or
    differs from its listed sha256."""
    run_dir = Path(run_dir)
    files = mix_files() if files is None else sorted(files)
    rec = dict(dir=str(run_dir), rule=INPUTS_RULE, files={}, records={})
    docs, listed, bad = {}, {}, []
    for name, key in INPUT_RECORDS:
        raw = (run_dir / name).read_bytes()
        rec["records"][name] = hashlib.sha256(raw).hexdigest()
        docs[name] = json.loads(raw)
        for e in docs[name][key].values():
            listed[e["file"]] = (name, e.get("sha256"))
    for f in files:
        if not (run_dir / f).is_file():
            bad.append(f"{f} is missing")
            continue
        got = plant.sha256(run_dir / f)
        rec["files"][f] = got
        where, want = listed.get(f, (None, None))
        if where is None:
            bad.append(f"{f} is not listed in {' or '.join(n for n, _ in INPUT_RECORDS)}")
        elif got != want:
            bad.append(f"{f} has sha256 {got[:12]}..., {where} records {str(want)[:12]}...")
    if bad:
        raise InputError(f"{run_dir}: " + "; ".join(bad))
    return rec, docs


def verify_inputs(run_dir: Path, files=None) -> dict:
    """The record of ``read_inputs``."""
    return read_inputs(run_dir, files)[0]


def input_differences(a: dict | None, b: dict | None) -> list[str]:
    """The files whose sha256 differs between two input records (the directory may differ); a missing record is a
    difference."""
    if not a or not b:
        return ["no record of the inputs"]
    out = []
    for key in ("files", "records"):
        x, y = a.get(key) or {}, b.get(key) or {}
        for f in sorted(set(x) | set(y)):
            if x.get(f) != y.get(f):
                out.append(f"{f}: {str(x.get(f))[:12]} -> {str(y.get(f))[:12]}")
    return out


# ------------------------------------------------------------------------------------------------
# The tuner's sources (what a resume and G5 compare)
# ------------------------------------------------------------------------------------------------
PACKAGE = __name__.rsplit(".", 1)[0]  # designlab.stage1
TUNER_ROOT = "tune.py"
TUNER_SOURCES_RULE = (
    "sha256 over the designlab/stage1 modules the tuner imports, in the form of plant.SOURCES_RULE (each file's name, "
    "a NUL byte, its bytes and a NUL byte, in path order): tune.py, the package's __init__.py and every "
    "designlab/stage1 module they reach through their imports, at module level or inside functions, followed "
    "transitively (read from the sources with ast); the module list is recorded beside the hash.  Other "
    "designlab/stage1 files (the command line __main__.py, gates.py, the later steps' modules) are not in it, so they "
    "may be added or changed while a fit runs; the designlab modules outside designlab/stage1 are named by each "
    "session's commit and dirty paths (plant.provenance), not hashed here")


def package_files() -> dict[str, bytes]:
    """designlab/stage1's ``*.py`` files as they are on disk: name -> bytes."""
    return {p.name: p.read_bytes() for p in plant.SOURCES.glob("*.py")}


def _imports(name: str, files: dict[str, bytes]) -> set[str]:
    """The designlab/stage1 modules one module imports anywhere in its source (``TUNER_SOURCES_RULE``)."""
    out = set()
    for node in ast.walk(ast.parse(files[name], filename=name)):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                mod = node.module
            elif node.level == 0 and node.module and (node.module + ".").startswith(PACKAGE + "."):
                mod = node.module[len(PACKAGE) + 1 :] or None
            else:
                continue  # the standard library, other packages, designlab outside designlab/stage1
            out.add("__init__.py")
            if mod:
                out.add(mod.split(".")[0] + ".py")
            else:
                out.update(f"{a.name}.py" for a in node.names if f"{a.name}.py" in files)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == PACKAGE or a.name.startswith(PACKAGE + "."):
                    out.add("__init__.py")
                    if a.name != PACKAGE:
                        out.add(a.name[len(PACKAGE) + 1 :].split(".")[0] + ".py")
    missing = out - set(files)
    if missing:
        raise ValueError(f"{name} imports {sorted(missing)}, which are not modules of designlab/stage1")
    return out


def tuner_modules(files: dict[str, bytes] | None = None, root: str = TUNER_ROOT) -> list[str]:
    """The designlab/stage1 modules the tuner imports (``TUNER_SOURCES_RULE``), from ``files`` (name -> bytes;
    default: the package on disk); ``root`` another module whose imports to follow the same way (the behaviour
    panels', ``panels.PANEL_SOURCES_RULE``)."""
    files = package_files() if files is None else files
    seen, todo = set(), ["__init__.py", root]
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo.extend(sorted(_imports(name, files) - seen))
    return sorted(seen)


def tuner_sources(files: dict[str, bytes] | None = None) -> dict:
    """The hash of the tuner's sources with its rule and module list (``TUNER_SOURCES_RULE``)."""
    files = package_files() if files is None else files
    mods = tuner_modules(files)
    return dict(sha256=plant.sources_sha256({m: files[m] for m in mods}), rule=TUNER_SOURCES_RULE, modules=mods)


# ------------------------------------------------------------------------------------------------
# Episodes
# ------------------------------------------------------------------------------------------------
@dataclass
class EpisodeSpec:
    """One episode of the mix on one body: its track and the reference quantities its measures need."""

    kind: str
    direction: str | None
    plan: list
    track: control.Track
    seconds: float
    period_s: float
    stride_s: float
    speed_cmd: float
    speed_target: float
    yaw_cmd: float
    yaw_target: float
    sign: float
    start_ref: float
    stop_ref: float
    window_ref: tuple | None
    ref_margins: np.ndarray
    required_margins: np.ndarray
    info: dict = field(default_factory=dict)


def _phase_time(seg: control.Segment, tr: Transition, phase: float) -> float:
    return float(seg.t0 + np.interp(phase, tr.phase, tr.t))


def build_spec(p: plant.Plant, run_dir: Path, kind: str, direction: str | None, omega: float, refs_doc: dict,
               tr_doc: dict, rate: float = COMMAND_RATE, plan: list | None = None, case: str | None = None,
               gait: str | None = None) -> EpisodeSpec:
    """The spec of one episode of the mix (``episode_plan``).  The behaviour panels (``panels``) may give their own
    track ``plan`` (items as ``control.build_track`` takes them; a walk's plan may end without a stop, its window is
    then given to ``measure``), the turn ``case`` (default ``TURN[kind]``) and the reference ``gait`` whose speed and
    period a walk or a walking turn targets (default the nominal walk).  Without them the spec is the mix's."""
    run_dir = Path(run_dir)
    custom = plan is not None
    plan = episode_plan(kind, direction) if plan is None else [tuple(x) for x in plan]
    track = control.build_track(p, run_dir, plan, omega)
    trs = {seg.name: Transition.load(run_dir / "transitions" / f"{seg.name}.npz") for seg in track.segments
           if seg.kind == "finite"}
    seg_of = {seg.name: seg for seg in track.segments}
    first = track.segments[0]
    duty_b = float(tr_doc[first.name]["gaits"]["b"]["duty"])
    start_ref = _phase_time(first, trs[first.name], stop_command_phase(duty_b))
    stops = [ev for ev in track.events if ev["kind"] == "stop"]
    if len(stops) != 1 and not (custom and kind == "walk" and not stops):
        raise ValueError(f"{kind}: expected one stop, the track has {len(stops)}")
    stop_ref = float(stops[0]["t_ref"]) if stops else None
    nominal = refs_doc[gait or "nominal"]
    period = float(nominal["period_s"])
    window = None
    if kind == "walk":
        steady = _phase_time(first, trs[first.name], 1.0)
        if stop_ref is not None and stop_ref - WALK_WINDOW_S * rate < steady:
            raise ValueError(f"walk: {WALK_WINDOW_S:g} s before the stop command reaches into the start's ramp")
        speed_cmd, target = float(nominal["speed_m_s"]) * rate, float(nominal["speed_m_s"])
        yaw_cmd, yaw_target, sign = 0.0, 0.0, 0.0
    else:
        c = f"{case or TURN[kind]}_{direction}"
        yr = float(refs_doc[c]["yaw_rate_rad_s"])
        yaw_cmd, yaw_target, sign = yr * rate, abs(yr), math.copysign(1.0, yr)
        if kind == "walking_turn":
            window = (_phase_time(seg_of[f"enter_{c}"], trs[f"enter_{c}"], 1.0),
                      _phase_time(seg_of[f"exit_{c}"], trs[f"exit_{c}"], 0.0))
            speed_cmd, target = float(nominal["speed_m_s"]) * rate, float(nominal["speed_m_s"])
        else:
            window = (_phase_time(first, trs[first.name], 1.0), stop_ref)
            speed_cmd, target = 0.0, 0.0
    m = p.m
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in p.joints]
    lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
    nj = len(p.joints)
    q = np.vstack([seg.S[:, :nj] for seg in track.segments])
    rest = plant.rest_ends(p.joints)  # an end a joint rests on by design is no range limit (plant.REST_ON_STOP_RULE)
    ref_margins = metrics.joint_margins(q, lo, hi, rest)
    if kind == "walk":
        required = np.full(nj, WALK_RANGE_RAD)
    else:
        required = np.maximum(np.minimum(WALK_RANGE_RAD, ref_margins - TURN_RANGE_SLACK_RAD), AT_LIMIT_RAD)
    info = dict(lo=lo, hi=hi, rest=rest, H=float(p.meta["H_m"]),
                weight=float(m.body_subtreemass[p.trunk]) * control.G, turned_ref_rad=None)
    if window is not None:  # the reference's own heading change over the steady turn (the track's yaw)
        y = [track.info(t)[1]["yaw"] for t in window]
        info["turned_ref_rad"] = float(sign * (y[1] - y[0]))
    return EpisodeSpec(kind=kind, direction=direction, plan=plan, track=track, seconds=track.t_end / rate,
                       period_s=period, stride_s=period / rate, speed_cmd=speed_cmd, speed_target=target,
                       yaw_cmd=yaw_cmd, yaw_target=yaw_target, sign=sign, start_ref=start_ref, stop_ref=stop_ref,
                       window_ref=window, ref_margins=ref_margins, required_margins=required, info=info)


def _terms_failed(kind: str) -> tuple[dict, dict, dict, dict]:
    tracking = {n: 1.0 for n in TRACKING[kind]}
    limits = {n: 1.0 for n in LIMITS[kind]}
    start = {n: 1.0 for n in START_TERMS} if kind in WITH_START_STOP else {}
    stop = {n: 1.0 for n in STOP_TERMS} if kind in WITH_START_STOP else {}
    return tracking, limits, start, stop


def measure(spec: EpisodeSpec, rec: metrics.Recorder, out: dict, lg=None, window: tuple | None = None) -> dict:
    """An episode's measures, its objective terms and its plan section 4 success (module docstring).  ``window``
    (simulated seconds, from, to) replaces the spec's window (the walk's 10 s before the stop command, a turn's
    steady turning): the behaviour panels judge a walk that ends without a stop over its last 10 s."""
    kind, inf = spec.kind, spec.info
    H = inf["H"]
    fell = bool(out["fell"])
    surv = 1.0 if not fell else min(1.0, float(out["fall_time_s"]) / spec.seconds)
    res = dict(kind=kind, direction=spec.direction, fell=fell, fall_time_s=out["fall_time_s"],
               fall_cause=out["fall_cause"], survived_frac=surv, seconds=float(out["seconds"]),
               planned_seconds=spec.seconds, wall_s=float(out["wall_s"]))
    if out.get("mujoco_warnings"):
        res["mujoco_warnings"] = out["mujoco_warnings"]
    stop = next((s for s in (out.get("stops") or []) if s["name"] in ("stop", f"stop_{TURN.get(kind)}_{spec.direction}")),
                None)
    heading = float(rec.col("yaw")[0]) if rec.rows else 0.0
    if window is not None:
        ta, tb = window
    elif kind == "walk":
        tb = stop.get("command_s") if stop and stop.get("reached") else None
        ta = None if tb is None else tb - WALK_WINDOW_S
    else:
        ta, tb = (rec.sim_time(t) for t in spec.window_ref)
    complete = (not fell) and ta is not None and tb is not None and tb > ta
    w = metrics.window(rec, ta, tb, heading, H, spec.sign) if complete else dict(complete=False)
    complete = complete and w.get("complete", False)
    a_lp = metrics.forward_acceleration(rec) if rec.rows > 3 else None
    if kind in WITH_START_STOP:
        t_start = rec.sim_time(spec.start_ref)
        st = (metrics.start_reading(rec, t_start, spec.speed_cmd, spec.stride_s, heading) if t_start is not None
              else dict(reached=False, strides=None))
        st["acceleration_peak_m_s2"] = (metrics.acceleration_peak(rec, t_start - 0.25 * spec.stride_s,
                                                                  t_start + START_WITHIN_STRIDES * spec.stride_s, a_lp)
                                        if t_start is not None and a_lp is not None else None)
        t_stop = stop.get("command_s") if stop and stop.get("reached") else None
        sp = dict(forward_stopped_strides=stop.get("forward_stopped_strides") if stop else None,
                  within=stop.get("within") if stop else None, at_rest_strides=stop.get("at_rest_strides") if stop
                  else None, rest_speed_m_s=stop.get("rest_speed_m_s") if stop else None,
                  acceleration_peak_m_s2=metrics.acceleration_peak(rec, t_stop, t_stop + 2.0 * spec.stride_s, a_lp)
                  if t_stop is not None and a_lp is not None else None)
    else:
        st = sp = None
        if stop:
            res["stop"] = dict(at_rest_strides=stop.get("at_rest_strides"), rest_speed_m_s=stop.get("rest_speed_m_s"))
    margins = (metrics.joint_margins(rec.Q, inf["lo"], inf["hi"], inf.get("rest")) if rec.rows
               else np.full(len(spec.ref_margins), -1.0))
    shortfall = float(np.max(spec.required_margins - margins))
    worst_j = int(np.argmax(spec.required_margins - margins))
    spd = float(out["worst_speed_ratio"][1])
    e = rec.U[:, [metrics.C["e_x"], metrics.C["e_y"]]] if rec.rows else np.zeros((1, 2))
    dcm = float(np.sqrt(np.mean(np.sum(e**2, axis=1))))
    drift = None
    if kind == "in_place_turn" and rec.rows:  # the turning centre: the reference turns about the root, at rest
        c = rec.U[:, [metrics.C["root_x"], metrics.C["root_y"]]]
        drift = float(np.linalg.norm(c[-1] - c[0]))
    legs = metrics.leg_gap(rec, lg) if (lg is not None and kind != "walk") else None
    res.update(window=w, start=st, stop_reading=sp, dcm_error_rms_m=dcm, drift_m=drift, legs=legs,
               joint_margin_min_rad=float(margins.min()), joint_shortfall_rad=shortfall,
               joint_worst=out_joint(spec, worst_j), worst_sustained_ratio=out["worst_sustained_ratio"],
               window_sustained_ratio=w.get("drive_sustained_worst"), worst_speed_ratio=out["worst_speed_ratio"],
               ankle_cap_residual=_cap_summary(out), phase_rate_range=out["phase_rate_range"],
               captures=len(out["captures"]), estimator=_est_summary(out))
    if not complete:
        res["tracking"], res["limits"], res["start_terms"], res["stop_terms"] = _terms_failed(kind)
        res["success"] = {n: False for n in SUCCESS[kind]}
        res["success"]["survived"] = not fell
        res["succeeded"] = False
        return res
    sus = float(w["drive_sustained_worst"][1])  # the drives' motor-space RMS over the window (SUCCESS_RULE)
    tr, lim = {}, {}
    tr["dcm"] = error(dcm, SCALES["dcm_m"])
    lim["roll_rms"] = hinge(w["roll_rms_deg"], ROLL_PITCH_RMS_DEG, ROLL_PITCH_RMS_DEG)
    lim["pitch_rms"] = hinge(w["pitch_rms_deg"], ROLL_PITCH_RMS_DEG, ROLL_PITCH_RMS_DEG)
    lim["roll_peak"] = hinge(w["roll_peak_deg"], ROLL_PITCH_PEAK_DEG, ROLL_PITCH_PEAK_DEG)
    lim["pitch_peak"] = hinge(w["pitch_peak_deg"], ROLL_PITCH_PEAK_DEG, ROLL_PITCH_PEAK_DEG)
    lim["duty_left"] = band(w["duty"][0], *DUTY, SCALES["duty"])
    lim["duty_right"] = band(w["duty"][1], *DUTY, SCALES["duty"])
    lim["flight"] = error(w["flight_s"], SCALES["flight_s"])
    lim["grf"] = hinge(w["grf_peak_bw"], GRF_BW, GRF_BW)
    lim["toe_clearance"] = hinge_below(w["toe_clearance_m"], TOE_H * H, TOE_H * H)
    lim["slip_median"] = hinge(w["slip_median_m"], SLIP_MEDIAN_M, SLIP_MEDIAN_M)
    lim["slip_worst"] = hinge(w["slip_worst_m"], SLIP_WORST_M, SLIP_WORST_M)
    ok = dict(survived=True, duty=bool(all(DUTY[0] <= x <= DUTY[1] for x in w["duty"])), no_flight=w["flight_s"] == 0.0,
              grf=w["grf_peak_bw"] <= GRF_BW, roll_rms=w["roll_rms_deg"] <= ROLL_PITCH_RMS_DEG,
              pitch_rms=w["pitch_rms_deg"] <= ROLL_PITCH_RMS_DEG, roll_peak=w["roll_peak_deg"] <= ROLL_PITCH_PEAK_DEG,
              pitch_peak=w["pitch_peak_deg"] <= ROLL_PITCH_PEAK_DEG,
              toe_clearance=w["toe_clearance_m"] is not None and w["toe_clearance_m"] >= TOE_H * H,
              slip_median=w["slip_median_m"] is not None and w["slip_median_m"] <= SLIP_MEDIAN_M,
              slip_worst=w["slip_worst_m"] is not None and w["slip_worst_m"] <= SLIP_WORST_M,
              drive_sustained=sus <= 1.0, drive_speed=spd <= 1.0, joint_range=shortfall <= 0.0)
    if kind == "walk":
        tr["speed"] = error(w["speed_m_s"] - spec.speed_cmd, SCALES["speed_frac"] * spec.speed_cmd)
        tr["heading"] = error(w["heading_change_rad"], SCALES["heading_rad"])
        lim["cadence"] = band(w["cadence_steps_s"], *CADENCE, SCALES["cadence"])
        lim["stride"] = band(w["stride_H"], *STRIDE_H, SCALES["stride_H"])
        lim["speed"] = hinge_below(w["speed_m_s"], spec.speed_target, SCALES["speed_limit_frac"] * spec.speed_target)
        ok.update(speed=w["speed_m_s"] >= spec.speed_target,
                  cadence=w["cadence_steps_s"] is not None and CADENCE[0] <= w["cadence_steps_s"] <= CADENCE[1],
                  stride=w["stride_H"] is not None and STRIDE_H[0] <= w["stride_H"] <= STRIDE_H[1])
    else:
        yr = w["yaw_rate_rad_s"]
        tr["yaw_rate"] = error(yr - abs(spec.yaw_cmd), SCALES["yaw_rate_frac"] * abs(spec.yaw_cmd))
        lim["yaw_rate"] = hinge_below(yr, spec.yaw_target, SCALES["yaw_rate_limit_frac"] * spec.yaw_target)
        gap = None if not legs or legs["min_gap_all_mm"] is None else 1e-3 * legs["min_gap_all_mm"]
        lim["leg_interference"] = 1.0 if gap is None else hinge(-gap, 0.0, SCALES["leg_interference_m"])
        ok.update(yaw_rate=yr >= spec.yaw_target, no_leg_interference=bool(legs and legs["interference_free"]))
        if kind == "in_place_turn":
            tr["drift"] = error(drift, SCALES["drift_m"])
            lim["drift"] = hinge(drift, DRIFT_M, DRIFT_M)
            ok["drift"] = drift is not None and drift <= DRIFT_M
    lim["drive_sustained"] = hinge(sus, 1.0, 1.0)
    lim["drive_speed"] = hinge(spd, 1.0, 1.0)
    lim["joint_range"] = hinge(shortfall, 0.0, SCALES["joint_range_rad"])
    start_t, stop_t = {}, {}
    if kind in WITH_START_STOP:
        start_t["speed_up"] = 1.0 if not st.get("reached") else hinge(st["strides"], START_WITHIN_STRIDES,
                                                                       START_WITHIN_STRIDES)
        start_t["acceleration"] = hinge(st["acceleration_peak_m_s2"], ACCEL_M_S2, ACCEL_M_S2)
        stop_t["stopped"] = hinge(sp["forward_stopped_strides"], STOP_WITHIN_STRIDES, STOP_WITHIN_STRIDES)
        stop_t["acceleration"] = hinge(sp["acceleration_peak_m_s2"], ACCEL_M_S2, ACCEL_M_S2)
    res["tracking"] = {n: tr[n] for n in TRACKING[kind]}
    res["limits"] = {n: lim[n] for n in LIMITS[kind]}
    res["start_terms"], res["stop_terms"] = start_t, stop_t
    res["success"] = {n: bool(ok[n]) for n in SUCCESS[kind]}
    res["succeeded"] = bool(all(res["success"].values()))
    return res


def out_joint(spec: EpisodeSpec, j: int) -> str:
    return spec.track.p.joints[j]


def _cap_summary(out: dict) -> dict | None:
    c = out.get("ankle_cap_residual")
    if not c:
        return None
    return {k: c[k] for k in ("box_frac", "box_max_excess_Nm", "pad_frac", "pad_max_excess_Nm", "heavy_pad_frac",
                              "heavy_pad_max_excess_Nm")}


def _est_summary(out: dict) -> dict | None:
    e = out.get("estimator")
    if not e:
        return None
    return {k: e[k] for k in ("com_vel_err_rms_m_s", "com_rel_feet_err_rms_mm", "tilt_err_rms_deg")}


def score(eps: dict) -> dict:
    """The objective of one evaluation from its measured episodes (module docstring)."""
    S = sum((1.0 + (1.0 - e["survived_frac"])) for e in eps.values() if e["fell"])
    tracking = [v for e in eps.values() for v in e["tracking"].values()]
    groups = {}
    for kind, e in eps.items():
        v = list(e["limits"].values())
        groups[kind] = 0.5 * (max(v) + float(np.mean(v)))
    for name, key in (("start", "start_terms"), ("stop", "stop_terms")):
        v = [x for e in eps.values() for x in e[key].values()]
        if v:
            groups[name] = 0.5 * (max(v) + float(np.mean(v)))
    T = float(np.mean(tracking))
    L = float(np.mean(list(groups.values())))
    P = 0.5 * T + 0.5 * L
    return dict(J=S + P, S=S, P=P, T=T, L=L, groups=groups, falls=int(sum(e["fell"] for e in eps.values())))


TERM_ORDER = tuple(f"{k}.tracking.{n}" for k in EPISODES for n in TRACKING[k]) + tuple(
    f"{k}.limits.{n}" for k in EPISODES for n in LIMITS[k]) + tuple(
    f"{k}.{g}.{n}" for k in WITH_START_STOP for g, names in (("start", START_TERMS), ("stop", STOP_TERMS))
    for n in names)


def term_vector(eps: dict) -> list:
    out = []
    for name in TERM_ORDER:
        kind, group, n = name.split(".")
        key = dict(tracking="tracking", limits="limits", start="start_terms", stop="stop_terms")[group]
        out.append(round(float(eps[kind][key][n]), 6))
    return out


# ------------------------------------------------------------------------------------------------
# Failures: evaluations that raise, MuJoCo warnings, lost workers
# ------------------------------------------------------------------------------------------------
#: MuJoCo's warnings (``mujoco.mjtWarning``): each means an episode's physics is not valid.  A bad position,
#: velocity, acceleration or control makes MuJoCo reset the data and run on from the model's default pose; a full
#: contact or constraint buffer drops contacts; a singular inertia breaks the dynamics.
SIM_WARNINGS = tuple(name for name in mujoco.mjtWarning.__members__ if name != "mjNWARNING")
#: Seconds a worker may take over one evaluation before it counts as lost (an evaluation takes 10-20 s here).
EVAL_TIMEOUT_S = 900.0
#: Seconds a worker may take to build its evaluator (about 10 s here).
START_TIMEOUT_S = 600.0
#: Attempts at an evaluation whose worker was lost before it is scored as the worst case.
ATTEMPTS = 2
FAILURE_RULE = (
    "an evaluation that raises scores as the worst case: every episode a fall at its start (survived share 0, S = 2 "
    "per episode, every term 1, J = 7), with its traceback in the evaluation log; an episode during which MuJoCo "
    f"warned ({', '.join(SIM_WARNINGS)}; counted after the episode) counts as a fall at its start, the warnings "
    "recorded; a worker process that dies or does not answer within "
    f"{EVAL_TIMEOUT_S:g} s is replaced by a fresh one (on the same inputs and tuner sources, else the run stops) and "
    f"its evaluation run again, up to {ATTEMPTS} attempts, after which it scores as the worst case with the reason; "
    "every such incident is kept in the session record")


def sim_warnings(d) -> dict:
    """The MuJoCo warnings raised on ``d`` since its last reset (``SIM_WARNINGS``): name -> count and last info."""
    out = {}
    for name in SIM_WARNINGS:
        w = d.warning[int(mujoco.mjtWarning.__members__[name])]
        if w.number:
            out[name.removeprefix("mjWARN_")] = dict(number=int(w.number), lastinfo=int(w.lastinfo))
    return out


def warned_fall(out: dict, warnings: dict) -> dict:
    """``control.run``'s outcome of an episode during which MuJoCo warned: a fall at its start (``FAILURE_RULE``)."""
    if not warnings:
        return out
    cause = "MuJoCo warning " + ", ".join(f"{k} (x{v['number']})" for k, v in sorted(warnings.items()))
    return dict(out, fell=True, fall_time_s=0.0, fall_cause=cause, mujoco_warnings=warnings)


def failed_episode(kind: str, direction: str | None, cause: str, planned_seconds: float | None = None) -> dict:
    """An episode scored as the worst case: a fall at its start, every term 1 (``FAILURE_RULE``)."""
    tracking, limits, start, stop = _terms_failed(kind)
    return dict(kind=kind, direction=direction, fell=True, fall_time_s=0.0, fall_cause=cause, survived_frac=0.0,
                seconds=0.0, planned_seconds=planned_seconds, wall_s=0.0, window=dict(complete=False), start=None,
                stop_reading=None, dcm_error_rms_m=None, drift_m=None, legs=None, joint_margin_min_rad=None,
                joint_shortfall_rad=None, joint_worst=None, worst_sustained_ratio=None, window_sustained_ratio=None,
                worst_speed_ratio=None, ankle_cap_residual=None, phase_rate_range=None, captures=0, estimator=None,
                tracking=tracking, limits=limits, start_terms=start, stop_terms=stop,
                success={n: False for n in SUCCESS[kind]}, succeeded=False)


def failed_evaluation(episode_seed: int, reason: str, planned: dict | None = None, wall_s: float = 0.0) -> dict:
    """An evaluation scored as the worst case (``FAILURE_RULE``) with the reason (a traceback, or what happened to its
    worker); ``planned`` maps each episode kind to its planned seconds (for the record)."""
    dirs = directions(episode_seed)
    eps = {k: failed_episode(k, dirs[k], "evaluation failed", (planned or {}).get(k)) for k in EPISODES}
    return dict(score(eps), episode_seed=int(episode_seed), directions=dirs, episodes=eps, terms=term_vector(eps),
                wall_s=float(wall_s), simulated_s=0.0, error=str(reason))


# ------------------------------------------------------------------------------------------------
# The evaluator
# ------------------------------------------------------------------------------------------------
class Evaluator:
    """One body's plant and episode tracks, built once; ``evaluate`` runs the mix for a gain vector and an episode
    seed.  Every episode gets a fresh tracker (and estimator) on the plant, reset to its settled stance, so an
    evaluation does not depend on the ones before it.  The reference and transition files are checked before they are
    loaded (``INPUTS_RULE``; ``inputs`` holds the record), and against ``expected_inputs`` when given."""

    def __init__(self, body: str, run_dir: Path | None = None, mode: str = MODE, rate: float = COMMAND_RATE,
                 legs: bool = True, expected_inputs: dict | None = None):
        from .legs import LegGap

        self.body, self.mode, self.rate = body, mode, float(rate)
        self.run_dir = Path(run_dir) if run_dir is not None else RUN / body
        self.inputs, docs = read_inputs(self.run_dir)
        if expected_inputs is not None:
            diff = input_differences(expected_inputs, self.inputs)
            if diff:
                raise InputError(f"{self.run_dir}: the inputs differ from the ones expected: " + "; ".join(diff))
        self.p = plant.build(body, "hard")
        refs_doc = docs["refs.json"]["references"]
        tr_doc = docs["transitions.json"]["transitions"]
        self.omega = control.omega0(self.p, Reference.load(self.run_dir / "refs" / "nominal.npz").q_home)
        self.specs = {}
        for kind in EPISODES:
            for d in ((None,) if kind == "walk" else ("left", "right")):
                self.specs[(kind, d)] = build_spec(self.p, self.run_dir, kind, d, self.omega, refs_doc, tr_doc,
                                                   self.rate)
        unverified = self.loaded_files() - set(self.inputs["files"])
        if unverified:
            raise InputError(f"the tracks hold files that were not checked: {sorted(unverified)}")
        self.lg = LegGap(body) if legs else None

    def loaded_files(self) -> set[str]:
        """The files the tracks were built from (each segment's) and refs/nominal.npz."""
        return {"refs/nominal.npz"} | {f"{'refs' if seg.kind == 'periodic' else 'transitions'}/{seg.name}.npz"
                                       for s in self.specs.values() for seg in s.track.segments if seg.kind != "hold"}

    def describe(self) -> dict:
        """The mix as built on this body (for the records)."""
        out = {}
        for (kind, d), s in self.specs.items():
            out[f"{kind}" + (f"_{d}" if d else "")] = dict(
                plan=[list(x) for x in s.plan], seconds=s.seconds, strides=s.seconds / s.stride_s,
                period_s=s.period_s, commanded_stride_s=s.stride_s, speed_command_m_s=s.speed_cmd,
                speed_target_m_s=s.speed_target, yaw_rate_command_rad_s=s.yaw_cmd, yaw_rate_target_rad_s=s.yaw_target,
                start_command_ref_s=s.start_ref, stop_command_ref_s=s.stop_ref, window_ref_s=s.window_ref,
                reference_turned_rad=s.info["turned_ref_rad"],
                reference_joint_margins_rad=dict(zip(self.p.joints, (float(x) for x in s.ref_margins), strict=True)),
                required_joint_margins_rad=dict(zip(self.p.joints, (float(x) for x in s.required_margins),
                                                    strict=True)))
        dirs = directions(0)
        out["simulated_s_per_evaluation"] = sum(self.specs[(k, dirs[k])].seconds for k in EPISODES)
        return out

    def episode(self, kind: str, direction: str | None, gains, episode_seed: int, j: int) -> dict:
        spec = self.specs[(kind, direction)]
        tk = control.Tracker(self.p, spec.track, self.mode, gains=np.asarray(gains, float), rate=self.rate)
        rec = metrics.Recorder(legs_every=LEGS_EVERY if (self.lg is not None and kind != "walk") else 0)
        out = control.run(tk, spec.seconds, start="stance", seed=derived_seed(episode_seed, j, 0),
                          noise_seed=derived_seed(episode_seed, j, 1), recorder=rec, joint_noise=START_JOINT_NOISE_RAD,
                          root_velocity_noise=START_ROOT_VELOCITY_NOISE_M_S)
        return measure(spec, rec, warned_fall(out, sim_warnings(self.p.d)), self.lg)

    def planned(self, episode_seed: int) -> dict:
        """Each episode kind's planned seconds on ``episode_seed``'s directions."""
        dirs, specs = directions(episode_seed), getattr(self, "specs", {})
        return {k: specs[(k, dirs[k])].seconds for k in EPISODES if (k, dirs[k]) in specs}

    def evaluate(self, gains, episode_seed: int) -> dict:
        """The mix for ``gains`` on ``episode_seed``; an evaluation that raises scores as the worst case with its
        traceback (``FAILURE_RULE``)."""
        t0 = time.perf_counter()
        dirs = directions(episode_seed)
        try:
            eps = {kind: self.episode(kind, dirs[kind], gains, episode_seed, j) for j, kind in enumerate(EPISODES)}
            sc = score(eps)
            return dict(sc, episode_seed=int(episode_seed), directions=dirs, episodes=eps,
                        terms=term_vector(eps), wall_s=time.perf_counter() - t0,
                        simulated_s=float(sum(e["seconds"] for e in eps.values())))
        except Exception:
            return failed_evaluation(episode_seed, traceback.format_exc(), self.planned(episode_seed),
                                     time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------------
# Worker processes (optional parallel evaluation within one run)
# ------------------------------------------------------------------------------------------------
class PoolError(RuntimeError):
    """A worker process could not be started: the run stops, its last checkpoint intact (``--resume`` continues)."""


def _evaluator(body: str, run_dir: str, mode: str, rate: float, expected_inputs: dict | None) -> Evaluator:
    return Evaluator(body, Path(run_dir), mode, rate, expected_inputs=expected_inputs)


def _worker_main(conn, factory, args: tuple) -> None:
    """A worker process: build the evaluator, report it ready (its inputs and mix), then evaluate the jobs it is sent
    until it gets None or the pipe closes."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        ev = factory(*args)
        conn.send(("ready", dict(inputs=ev.inputs, mix=ev.describe(), sources=tuner_sources()["sha256"])))
    except Exception:
        conn.send(("init_error", traceback.format_exc()))
        return
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        i, gains, seed = job
        r = ev.evaluate(gains, seed)
        try:
            conn.send(("result", i, r))
        except OSError:
            return


class _Worker:
    def __init__(self, ctx, factory, args: tuple):
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child, factory, args), daemon=True)
        self.proc.start()
        child.close()
        self.job = None  # (job index, monotonic start time)

    def ready(self, timeout: float) -> dict:
        if not self.conn.poll(timeout):
            self.stop(kill=True)
            raise PoolError(f"a worker did not build its evaluator within {timeout:g} s")
        try:
            kind, info = self.conn.recv()
        except (EOFError, OSError):
            raise PoolError(f"a worker died while building its evaluator (exit code {self.stop(kill=True)})") from None
        if kind != "ready":
            self.stop(kill=True)
            raise PoolError(f"a worker could not build its evaluator:\n{info}")
        return info

    def stop(self, kill: bool = False):
        """End the process (asked to stop, or killed); its exit code."""
        if kill:
            self.proc.kill()
        else:
            try:
                self.conn.send(None)
            except OSError:
                pass
        self.proc.join(5.0)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(5.0)
        self.conn.close()
        return self.proc.exitcode


class Pool:
    """Evaluations in this process (``workers`` 1, the default) or in worker processes (spawned), each with its own
    evaluator; results come back in job order, so the run does not depend on the worker count.  ``inputs`` and
    ``mix`` are the evaluator's and ``sources`` the tuner's sources it runs (``tuner_sources``; every worker must
    report the same, a replacement included, so a fit never runs on two versions of its code).  A worker that dies or
    does not answer within ``timeout_s`` is replaced by a fresh one (checked against the same inputs) and its
    evaluation run again, up to ``ATTEMPTS`` attempts, after which the evaluation scores as the worst case with the
    reason (``FAILURE_RULE``);
    each incident is appended to ``incidents``.  A worker that cannot be started raises ``PoolError`` instead of
    hanging.  ``factory`` builds a worker's evaluator from (body, run_dir, mode, rate, expected inputs)."""

    def __init__(self, body: str, run_dir: Path, workers: int = 1, mode: str = MODE, rate: float = COMMAND_RATE,
                 timeout_s: float = EVAL_TIMEOUT_S, factory=None):
        self.workers = max(1, int(workers))
        self.timeout_s = float(timeout_s)
        self.incidents: list[dict] = []
        self.local = None
        self._ws: list[_Worker] = []
        factory = factory or _evaluator
        if self.workers == 1:
            self.local = factory(body, str(run_dir), mode, rate, None)
            self.inputs, self.mix = self.local.inputs, self.local.describe()
            self.sources = tuner_sources()["sha256"]
            return
        self._ctx, self._factory = mp.get_context("spawn"), factory
        self._args = (body, str(run_dir), mode, float(rate))
        try:
            self._ws = [_Worker(self._ctx, factory, self._args + (None,)) for _ in range(self.workers)]
            infos = [w.ready(START_TIMEOUT_S) for w in self._ws]
            self.inputs, self.mix, self.sources = infos[0]["inputs"], infos[0]["mix"], infos[0]["sources"]
            for info in infos[1:]:
                diff = input_differences(self.inputs, info["inputs"])
                if diff:
                    raise PoolError("the workers read different inputs: " + "; ".join(diff))
                if info["sources"] != self.sources:
                    raise PoolError("the workers started on different tuner sources")
        except BaseException:
            self.close()
            raise

    def describe(self) -> dict:
        return self.mix

    def _replace(self, k: int):
        """A fresh worker in place of worker ``k``; the old one's exit code."""
        code = self._ws[k].stop(kill=True)
        self._ws[k] = _Worker(self._ctx, self._factory, self._args + (self.inputs,))
        if self._ws[k].ready(START_TIMEOUT_S)["sources"] != self.sources:
            raise PoolError("the tuner's sources changed during the session, so a replacement worker would run other "
                            "code: restore them and --resume")
        return code

    def _planned(self, seed: int) -> dict:
        mix = self.mix or {}
        return {k: (mix.get(k + (f"_{d}" if d else "")) or {}).get("seconds") for k, d in directions(seed).items()}

    def map(self, jobs: list, failed=None) -> list:
        """The results of ``jobs``, (gains, episode seed) pairs, in job order.  ``failed`` (the job, the reason) ->
        the result of a job whose worker was lost on every attempt; with it the jobs go to the evaluator as they are
        (the behaviour panels send their own jobs, ``panels``).  Without it each job is made a float array and an
        int and a lost one scores as the worst case (``failed_evaluation``)."""
        if self.local is not None:
            return [self.local.evaluate(g, s) for g, s in jobs]
        if failed is None:
            jobs = [(np.asarray(g, float), int(s)) for g, s in jobs]
        results = [None] * len(jobs)
        todo, tries = deque(range(len(jobs))), [0] * len(jobs)
        while todo or any(w.job is not None for w in self._ws):
            for k in range(len(self._ws)):
                w = self._ws[k]
                if w.job is None and todo:
                    i = todo.popleft()
                    try:
                        w.conn.send((i, *jobs[i]))
                    except OSError:  # it died while idle: a fresh worker, the job back to the front of the queue
                        todo.appendleft(i)
                        code = self._replace(k)
                        self.incidents.append(dict(at=_now(), job=i, episode=jobs[i][1], attempt=tries[i],
                                                   what=f"a worker had died while idle (exit code {code})",
                                                   then="replaced; the job was not lost"))
                        continue
                    tries[i] += 1
                    w.job = (i, time.monotonic())
            busy = [k for k, w in enumerate(self._ws) if w.job is not None]
            if not busy:
                continue
            due = min(self._ws[k].job[1] for k in busy) + self.timeout_s
            _wait([self._ws[k].conn for k in busy] + [self._ws[k].proc.sentinel for k in busy],
                  max(0.0, due - time.monotonic()))
            for k in busy:
                w = self._ws[k]
                i = w.job[0]
                lost = None
                if w.conn.poll():
                    try:
                        _, j, r = w.conn.recv()
                    except (EOFError, OSError):
                        lost = "died"
                    else:
                        if j != i:
                            raise PoolError(f"a worker answered job {j} for job {i}")
                        results[i], w.job = r, None
                        continue
                elif not w.proc.is_alive():
                    lost = "died"
                elif time.monotonic() - w.job[1] > self.timeout_s:
                    lost = f"gave no answer within {self.timeout_s:g} s"
                if lost is None:
                    continue
                code = self._replace(k)
                what = f"the worker {lost} (exit code {code})"
                again = tries[i] < ATTEMPTS
                if again:
                    todo.appendleft(i)
                elif failed is not None:
                    results[i] = failed(jobs[i], f"{what}, on each of {tries[i]} attempts")
                else:
                    results[i] = failed_evaluation(jobs[i][1], f"{what}, on each of {tries[i]} attempts",
                                                   self._planned(jobs[i][1]))
                self.incidents.append(dict(at=_now(), job=i, episode=jobs[i][1], attempt=tries[i], what=what,
                                           then="run again in a fresh worker" if again else
                                           "scored as the worst case (FAILURE_RULE)"))
        return results

    def close(self) -> None:
        for w in self._ws:
            w.stop()
        self._ws = []


# ------------------------------------------------------------------------------------------------
# CMA-ES (numpy)
# ------------------------------------------------------------------------------------------------
class CMA:
    """(mu/mu_w, lambda)-CMA-ES with the default weights and learning rates (Hansen, "The CMA Evolution Strategy: A
    Tutorial", 2016), the eigendecomposition refreshed every generation.  ``state``/``from_state`` round-trip it
    exactly through JSON (floats keep their bits; the generator its PCG64 state)."""

    def __init__(self, x0, sigma0: float, popsize: int, seed: int):
        x0 = np.asarray(x0, float)
        n = len(x0)
        self.n, self.lam = n, int(popsize)
        self.mu = self.lam // 2
        w = math.log((self.lam + 1) / 2) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = float(1.0 / np.sum(self.w**2))
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean, self.sigma = x0.copy(), float(sigma0)
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.gen = 0
        self.rng = np.random.default_rng(int(seed))

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * (z * self.D) @ self.B.T

    def tell(self, X: np.ndarray, f) -> None:
        X, f = np.asarray(X, float), np.asarray(f, float)
        n = self.n
        idx = np.argsort(f, kind="stable")[: self.mu]
        y = (X[idx] - self.mean) / self.sigma
        yw = self.w @ y
        self.mean = self.mean + self.sigma * yw
        c_inv_sqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (c_inv_sqrt @ yw)
        norm_ps = float(np.linalg.norm(self.ps))
        hsig = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chin < 1.4 + 2 / (n + 1)
        self.pc = (1 - self.cc) * self.pc + (math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw if hsig else 0.0)
        dh = (1 - hsig) * self.cc * (2 - self.cc)
        rank_mu = (y * self.w[:, None]).T @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + dh * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chin - 1))
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        ev, B = np.linalg.eigh(self.C)
        self.D, self.B = np.sqrt(np.maximum(ev, 1e-20)), B
        self.gen += 1

    def state(self) -> dict:
        return dict(n=self.n, popsize=self.lam, generation=self.gen, mean=self.mean.tolist(), sigma=self.sigma,
                    C=self.C.tolist(), B=self.B.tolist(), D=self.D.tolist(), pc=self.pc.tolist(), ps=self.ps.tolist(),
                    rng=self.rng.bit_generator.state)

    @classmethod
    def from_state(cls, s: dict, sigma0: float, seed: int) -> "CMA":
        c = cls(np.asarray(s["mean"], float), sigma0, s["popsize"], seed)
        c.mean, c.sigma, c.gen = np.asarray(s["mean"], float), float(s["sigma"]), int(s["generation"])
        c.C, c.B, c.D = (np.asarray(s[k], float) for k in ("C", "B", "D"))
        c.pc, c.ps = np.asarray(s["pc"], float), np.asarray(s["ps"], float)
        c.rng.bit_generator.state = s["rng"]
        return c


# ------------------------------------------------------------------------------------------------
# Gains in the search space
# ------------------------------------------------------------------------------------------------
LO, HI = control.GAIN_BOUNDS[_SEARCHED_IX, 0].copy(), control.GAIN_BOUNDS[_SEARCHED_IX, 1].copy()


def to_unit(gains) -> np.ndarray:
    """The searched gains (``SEARCHED``) of a full gain vector (``control.GAIN_NAMES`` order) in the [0, 1] scaling."""
    g = np.asarray(gains, float)
    if g.shape != control.START_GAINS.shape:
        raise ValueError(f"{g.shape}: a full gain vector has {len(control.START_GAINS)} gains")
    return (g[_SEARCHED_IX] - LO) / (HI - LO)


def from_unit(x) -> np.ndarray:
    """The full gain vector of a point of the [0, 1] search space: the searched gains projected into their bounds,
    the fixed ones at ``FIXED_GAINS``."""
    x = np.asarray(x, float)
    if x.shape != (len(SEARCHED),):
        raise ValueError(f"{x.shape}: the search space has {len(SEARCHED)} dimensions ({', '.join(SEARCHED)})")
    g = control.START_GAINS.copy()
    for n, v in FIXED_GAINS.items():
        g[control.GAIN_NAMES.index(n)] = v
    g[_SEARCHED_IX] = LO + np.clip(x, 0.0, 1.0) * (HI - LO)
    return g


def repair_penalty(x) -> float:
    x = np.asarray(x, float)
    return float(BOX_PENALTY * np.sum((x - np.clip(x, 0.0, 1.0)) ** 2))


def training_order(seed: int) -> np.ndarray:
    """The training episodes in the order a seed's generations use them."""
    lo, hi = TRAIN_EPISODES
    return lo + np.random.default_rng([int(seed), 1]).permutation(hi - lo)


# ------------------------------------------------------------------------------------------------
# Records
# ------------------------------------------------------------------------------------------------
OBJECTIVE = dict(
    name="stage1-tune-v2", defined="before any tuning run (design_1.md gives the order and the episode budget, no "
                                   "formula)",
    revisions=["v1 (8 Nov): the first build, run only in smoke checks",
               "v2 (9 Nov, before the fit, after an independent review of the tuner): capture_margin held at 0.0 "
               "(FIXED_RULE); the drives' motor-space RMS read over the same window as the other limits "
               "(SUCCESS_RULE); evaluations that fail scored by FAILURE_RULE; the mix, the other terms and the score "
               "unchanged"],
    controller=MODE, gains=list(SEARCHED), fixed_gains=dict(FIXED_GAINS), fixed_rule=FIXED_RULE,
    mix=dict(episodes=list(EPISODES), plans={k: [list(x) for x in episode_plan(k, "left" if k != "walk" else None)]
                                             for k in EPISODES},
             strides=dict(walk=WALK_STRIDES, walking_turn=TURN_STRIDES, in_place_turn=IN_PLACE_STRIDES),
             directions="episode seed even: walking turn left, in-place turn right; odd: the reverse",
             start="the settled stance, joint angles +-%g rad and root velocity +-%g m/s (uniform, plant.Plant.reset), "
                   "drawn from derived_seed(episode, j, 0); R2's sensor noise (estimator.SensorNoise defaults) from "
                   "derived_seed(episode, j, 1)" % (START_JOINT_NOISE_RAD, START_ROOT_VELOCITY_NOISE_M_S),
             floor="hard (mu 0.9)", command_margin=COMMAND_MARGIN, command_rule=control.COMMAND_RATE_RULE,
             walk_window_s=WALK_WINDOW_S),
    score="J = S + P; S = sum over fallen episodes of (1 + unsurvived share); P = 0.5 T + 0.5 L, T the mean tracking "
          "term, L the mean over behaviour groups (walk, walking_turn, in_place_turn, start, stop) of (max + mean) / 2 "
          "of the group's limit terms; a fallen episode scores 1 on every term; CMA-ES fitness J + %g x the squared "
          "distance of the sample to its projection into the [0, 1] box" % BOX_PENALTY,
    terms=TERMS_RULE, success=SUCCESS_RULE, term_order=list(TERM_ORDER), scales=SCALES, failures=FAILURE_RULE,
    sketch_in_design_1=("47 simulated s on hard floor (nominal walk with a 0.4x push, Fr 0.05, both turns, start and "
                        "stop) plus 8 s on low pile; score: time survived first, then tracking and drive-duty "
                        "penalties.  Kept: the order (survival, then tracking and limits, drive duty among them) and "
                        "the about 55 s budget.  Not kept: the push, Fr 0.05 and low pile, which G5 does not judge "
                        "(the task's fixed mix of G5's behaviours plus start and stop)"))
CMA_SETTINGS = dict(popsize=POPSIZE, sigma0=SIGMA0, searched=list(SEARCHED), fixed=dict(FIXED_GAINS),
                    x0="the tuning start (decision 16) of the searched gains in the [0, 1] scaling of their "
                       "control.GAIN_BOUNDS", bounds=np.column_stack([LO, HI]).tolist(),
                    box="evaluate the projection into [0, 1]; add the squared distance (repair penalty)",
                    weights="log((lambda + 1) / 2) - log(i), i = 1..mu, normalised", c_m=1.0,
                    train_episodes=list(TRAIN_EPISODES), validation_episodes=list(VALIDATION_EPISODES),
                    common_random_numbers="every member of a generation on the same training episode",
                    episode_order="1000 + permutation(1000) from default_rng([seed, 1])")
FIXED_RECORD = dict(gains=dict(FIXED_GAINS), rule=FIXED_RULE)


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def run_dir_of(out: Path, body: str, seed: int) -> Path:
    return Path(out) / body / "tune" / str(int(seed))


def _settings(body: str, seed: int) -> dict:
    return dict(body=body, seed=int(seed), mode=MODE, rate=COMMAND_RATE, cma=CMA_SETTINGS, objective=OBJECTIVE["name"])


def _episode_line(e: dict) -> dict:
    out = dict(dir=e["direction"], fell=e["fell"], t_fall=e["fall_time_s"], cause=e["fall_cause"],
               surv=round(e["survived_frac"], 6), ok=e["succeeded"])
    if e.get("mujoco_warnings"):
        out["warn"] = e["mujoco_warnings"]
    return out


def _eval_line(gen: int, i: int, x: np.ndarray, fit: float, r: dict) -> dict:
    line = dict(gen=gen, i=i, episode=r["episode_seed"], x=[float(v) for v in x], J=r["J"], fitness=fit, S=r["S"],
                P=r["P"], T=r["T"], L=r["L"], groups=r["groups"], falls=r["falls"],
                episodes={k: _episode_line(e) for k, e in r["episodes"].items()},
                terms=r["terms"], sim_s=round(r["simulated_s"], 3), wall_s=round(r["wall_s"], 3))
    if r.get("error"):
        line["error"] = r["error"]
    return line


def _warned(r: dict) -> int:
    return int(sum(bool(e.get("mujoco_warnings")) for e in r["episodes"].values()))


def tune(body: str, seed: int, evals: int = BUDGET, resume: bool = False, out: Path = RUN, workers: int = 1,
         inputs: Path | None = None, log=print) -> dict:
    """The CMA-ES run of one body and seed (module docstring) until ``evals`` evaluations (whole generations), into
    ``<out>/<body>/tune/<seed>/``; the references and transitions are read from ``<inputs>/<body>/`` (default: out)."""
    if body not in BODIES:
        raise KeyError(body)
    src = Path(inputs or out) / body
    d = run_dir_of(out, body, seed)
    path, elog = d / "cma.json", d / "evals.jsonl"
    prov = plant.provenance()
    tsrc = tuner_sources()
    settings = _settings(body, seed)
    if path.exists() and not resume:
        raise SystemExit(f"{path} exists: pass --resume to continue it (or remove it to start over)")
    doc = None
    if resume:
        if not path.exists():
            raise SystemExit(f"{path}: nothing to resume")
        doc = json.loads(path.read_text())
        if doc["settings"] != settings:
            raise SystemExit(f"{path}: its settings differ from this code's; refusing to resume")
        first = doc["sessions"][0]
        was = (first.get("tuner_sources") or {}).get("sha256")
        if was != tsrc["sha256"]:
            raise SystemExit(f"{path}: the tuner's sources changed since the run started ({str(was)[:12]} -> "
                             f"{tsrc['sha256'][:12]}; modules {', '.join(tsrc['modules'])}); a resumed run would not "
                             "continue the same computation")
        if first["provenance"]["versions"] != prov["versions"]:
            raise SystemExit(f"{path}: library versions changed since the run started; refusing to resume")
    pool = Pool(body, src, workers)
    try:
        if pool.sources != tsrc["sha256"]:
            raise SystemExit(f"{path}: the tuner's sources changed while the run was starting; run it again")
        session = dict(started_at=_now(), from_generation=0, workers=pool.workers, provenance=prov, tuner_sources=tsrc,
                       inputs=pool.inputs, incidents=[])
        if resume:
            diff = input_differences(doc["sessions"][0].get("inputs"), pool.inputs)
            if diff:
                raise SystemExit(f"{path}: its inputs changed since the run started ({'; '.join(diff)}); refusing to "
                                 "resume")
            cma = CMA.from_state(doc["state"], SIGMA0, seed)
            n_done = int(doc["evals"])
            lines = elog.read_text().splitlines() if elog.exists() else []
            if len(lines) < n_done:
                raise SystemExit(f"{elog}: {len(lines)} evaluations logged, the checkpoint has {n_done}")
            if len(lines) > n_done:  # evaluations after the last checkpoint: the resumed run repeats them exactly
                elog.write_text("".join(line + "\n" for line in lines[:n_done]))
                log(f"{body} seed {seed}: dropped {len(lines) - n_done} evaluations logged after the checkpoint")
            session["from_generation"] = cma.gen
            doc["sessions"].append(session)
        else:
            cma = CMA(to_unit(control.START_GAINS), SIGMA0, POPSIZE, seed)
            doc = dict(schema=SCHEMA, body=body, seed=int(seed), settings=settings, objective=OBJECTIVE,
                       fixed_gains=FIXED_RECORD, sessions=[session], history=[], evals=0, status="running",
                       state=cma.state())
            d.mkdir(parents=True, exist_ok=True)
            elog.write_text("")
            _write_json(path, doc)
        doc["budget"] = int(evals)
        doc["inputs"] = str(src)
        doc.setdefault("mix", pool.describe())
        order = training_order(seed)
        seen = 0
        while doc["evals"] + cma.lam <= evals:
            g = cma.gen
            ep = int(order[g % len(order)])
            X = cma.ask()
            t0 = time.perf_counter()
            res = pool.map([(from_unit(x), ep) for x in X])
            fit = [r["J"] + repair_penalty(x) for x, r in zip(X, res, strict=True)]
            with elog.open("a") as fh:
                for i, (x, f, r) in enumerate(zip(X, fit, res, strict=True)):
                    fh.write(json.dumps(_eval_line(g, i, x, f, r)) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            cma.tell(X, fit)
            J = np.array([r["J"] for r in res])
            best = int(np.argmin(fit))
            new = pool.incidents[seen:]
            seen = len(pool.incidents)
            doc["sessions"][-1]["incidents"].extend(new)
            doc["history"].append(dict(gen=g, episode=ep, evals=doc["evals"] + cma.lam, sigma=cma.sigma,
                                       mean=cma.mean.tolist(), mean_gains=from_unit(cma.mean).tolist(),
                                       J_min=float(J.min()), J_median=float(np.median(J)), J_max=float(J.max()),
                                       falls=int(sum(r["falls"] for r in res)), best=best,
                                       best_gains=from_unit(X[best]).tolist(), wall_s=time.perf_counter() - t0,
                                       sim_s=float(sum(r["simulated_s"] for r in res)),
                                       errors=int(sum(bool(r.get("error")) for r in res)),
                                       warned_episodes=int(sum(_warned(r) for r in res)), incidents=len(new)))
            doc["evals"] += cma.lam
            doc["state"] = cma.state()
            _write_json(path, doc)
            h = doc["history"][-1]
            extra = "".join(f", {h[k]} {k.replace('_', ' ')}" for k in ("errors", "warned_episodes", "incidents")
                            if h[k])
            log(f"{body} seed {seed} gen {g}: episode {ep}, J min/median {h['J_min']:.4f}/{h['J_median']:.4f}, falls "
                f"{h['falls']}{extra}, sigma {cma.sigma:.4f}, {h['wall_s']:.1f} s ({h['wall_s'] / cma.lam:.2f} s/eval)")
            for x in new:
                log(f"{body} seed {seed} gen {g}: job {x['job']} (episode {x['episode']}): {x['what']}; {x['then']}")
    finally:
        pool.close()
    doc["status"] = "done" if doc["evals"] >= BUDGET else "partial"
    s = doc["sessions"][-1]
    s["ended_at"] = _now()
    s["tuner_sources_unchanged_during_session"] = tuner_sources()["sha256"] == tsrc["sha256"]
    s["package_sources_unchanged_during_session"] = plant.sources_sha256() == prov["stage1_sources"]["sha256"]
    _write_json(path, doc)
    return doc


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------
# Selection and G5
# ------------------------------------------------------------------------------------------------
SELECTION_RULE = (
    "candidates: the tuning start and the CMA-ES mean (projected into the bounds) after each third of a seed's "
    "generations, each evaluated on the validation episodes; ranked lexicographically: fewer falls over the validation "
    "episodes first, then the lower mean S (the falls' cost, higher the earlier they come), then the lower mean P "
    "(tracking and limits); the lower mean J and then the earlier candidate break ties; the first is selected")


def candidate_generations(n_gen: int) -> list[int]:
    """The generations whose CMA-ES mean is a selection candidate: after each third of the run."""
    return sorted({max(1, int(round(n_gen * k / 3))) for k in (1, 2, 3)}) if n_gen else []


def selection_key(v: dict, index: int) -> tuple:
    """A candidate's rank key from its validation summary (``SELECTION_RULE``): lower is better."""
    return (int(v["falls"]), float(v["S_mean"]), float(v["P_mean"]), float(v["J_mean"]), int(index))


def zeroed(gains) -> np.ndarray:
    """``gains`` with the ``LOWER_BOUND_START`` gains set to zero (``ZEROED_RULE``)."""
    g = np.asarray(gains, float).copy()
    for n in LOWER_BOUND_START:
        g[control.GAIN_NAMES.index(n)] = 0.0
    return g


def _validation_summary(results: list[dict]) -> dict:
    J = [r["J"] for r in results]
    per = {k: dict(successes=int(sum(r["episodes"][k]["succeeded"] for r in results)),
                   survived=int(sum(not r["episodes"][k]["fell"] for r in results)), runs=len(results))
           for k in EPISODES}
    return dict(falls=int(sum(r["falls"] for r in results)), S_mean=float(np.mean([r["S"] for r in results])),
                P_mean=float(np.mean([r["P"] for r in results])), J_mean=float(np.mean(J)), J=J,
                errors=int(sum(bool(r.get("error")) for r in results)),
                warned_episodes=int(sum(_warned(r) for r in results)), conditions=per)


def _validation_record(r: dict) -> dict:
    eps = {}
    for k, e in r["episodes"].items():
        eps[k] = {key: e.get(key) for key in ("direction", "fell", "fall_time_s", "fall_cause", "survived_frac",
                                              "succeeded", "success", "dcm_error_rms_m", "drift_m", "legs",
                                              "joint_margin_min_rad", "joint_shortfall_rad", "joint_worst",
                                              "worst_sustained_ratio", "window_sustained_ratio", "worst_speed_ratio",
                                              "ankle_cap_residual", "estimator", "start", "stop_reading", "stop",
                                              "mujoco_warnings")}
        w = e.get("window") or {}
        eps[k]["window"] = {key: w.get(key) for key in (
            "complete", "t_start_s", "t_end_s", "speed_m_s", "lateral_m", "heading_change_rad", "yaw_rate_rad_s",
            "turned_180", "duty", "flight_s", "cadence_steps_s", "stride_H", "grf_peak_bw", "roll_rms_deg",
            "pitch_rms_deg", "roll_peak_deg", "pitch_peak_deg", "toe_clearance_m", "slip_median_m", "slip_worst_m",
            "slip_contact_path_median_m", "slip_contact_path_worst_m", "drive_sustained_worst", "swings", "stances",
            "unloaded_steps")}
    out = dict(episode=r["episode_seed"], J=r["J"], S=r["S"], P=r["P"], T=r["T"], L=r["L"], groups=r["groups"],
               falls=r["falls"], terms=r["terms"], episodes=eps, wall_s=r["wall_s"])
    if r.get("error"):
        out["error"] = r["error"]
    return out


def _conditions(summary: dict) -> dict:
    cond = summary["conditions"]
    return {k: dict(cond[k], condition=CONDITIONS[k], pass_=bool(cond[k]["successes"] >= G5_MIN_SUCCESS))
            for k in EPISODES}


def g5(body: str, out: Path = RUN, seeds=SEEDS, workers: int = 1, episodes: int = G5_RUNS, require_done: bool = True,
       inputs: Path | None = None, log=print) -> dict:
    """Selection on the validation episodes and gate G5 (module docstring) into ``<out>/<body>/tune.json``, the
    references read from ``<inputs>/<body>/`` (default: out).  ``episodes`` below 20 (a smoke check) records no G5
    verdict.  A seed whose fit ran on other tuner sources or other inputs than this code and directory is not judged
    (``sources_match``, ``inputs_match``)."""
    prov = plant.provenance()
    tsrc = tuner_sources()
    t0 = time.perf_counter()
    val = list(range(VALIDATION_EPISODES[0], VALIDATION_EPISODES[0] + int(episodes)))
    runs, cands = {}, {"start": dict(gains=control.START_GAINS.tolist(), source="the tuning start (decision 16)")}
    verdicts = {}
    pool = Pool(body, Path(inputs or out) / body, workers)
    try:
        if pool.sources != tsrc["sha256"]:
            raise SystemExit(f"{body}: the tuner's sources changed while G5 was starting; run it again")
        for seed in seeds:
            path = run_dir_of(out, body, seed) / "cma.json"
            if not path.exists():
                runs[seed] = dict(status="missing", sources_match=None, inputs_match=None)
                continue
            doc = json.loads(path.read_text())
            fit_src = [(s.get("tuner_sources") or {}).get("sha256") for s in doc["sessions"]]
            runs[seed] = dict(status=doc["status"], evals=doc["evals"], generations=len(doc["history"]),
                              budget=doc.get("budget"), sigma=doc["state"]["sigma"], sessions=len(doc["sessions"]),
                              tuner_sources_sha256=fit_src[0],
                              sources_match=bool(fit_src) and all(x == tsrc["sha256"] for x in fit_src),
                              inputs_match=all(not input_differences(s.get("inputs"), pool.inputs)
                                               for s in doc["sessions"]),
                              package_sources_sha256=doc["sessions"][0]["provenance"]["stage1_sources"]["sha256"])
            if not (runs[seed]["sources_match"] and runs[seed]["inputs_match"]):
                continue
            if require_done and doc["status"] != "done":
                continue
            gens = candidate_generations(len(doc["history"]))
            runs[seed]["candidates"] = [f"seed{seed}_gen{g}" for g in gens]
            for g in gens:
                h = doc["history"][g - 1]
                cands[f"seed{seed}_gen{g}"] = dict(
                    gains=h["mean_gains"], source=f"the CMA-ES mean after generation {g} of seed {seed} ({h['evals']} "
                                                  "evals)")
        names = list(cands)
        jobs = [(np.asarray(cands[n]["gains"], float), ep) for n in names for ep in val]
        log(f"{body}: {len(names)} candidates x {len(val)} validation episodes = {len(jobs)} evaluations")
        res = pool.map(jobs)
        by = {n: res[i * len(val) : (i + 1) * len(val)] for i, n in enumerate(names)}
        for n in names:
            cands[n]["gains_named"] = control.gains_dict(cands[n]["gains"])
            cands[n]["validation"] = _validation_summary(by[n])
            cands[n]["runs"] = [_validation_record(r) for r in by[n]]
        judged = len(val) == G5_RUNS
        for seed in seeds:
            r = runs[seed]
            base = dict(sources_match=r["sources_match"], inputs_match=r["inputs_match"], refused=False)
            if r["status"] == "missing":
                verdicts[seed] = dict(base, judged=False, reason="no run")
            elif not r["sources_match"]:
                verdicts[seed] = dict(base, judged=False, refused=True, reason=(
                    f"the fit ran on other tuner sources ({str(r['tuner_sources_sha256'])[:12]}) than G5 "
                    f"({tsrc['sha256'][:12]}): refusing to judge"))
            elif not r["inputs_match"]:
                verdicts[seed] = dict(base, judged=False, refused=True, reason=(
                    "the fit ran on other inputs than G5 (INPUTS_RULE): refusing to judge"))
            elif not r.get("candidates"):
                verdicts[seed] = dict(base, judged=False, reason=f"no finished run ({r['status']})")
            else:
                pool_names = ["start"] + r["candidates"]
                ranked = sorted(pool_names, key=lambda n: selection_key(cands[n]["validation"], pool_names.index(n)))
                sel = ranked[0]
                cond = _conditions(cands[sel]["validation"])
                passes = all(c["pass_"] for c in cond.values())
                verdicts[seed] = dict(
                    base, judged=judged, selected=sel, selected_gains=cands[sel]["gains_named"],
                    selection=[dict(candidate=n, rank=ranked.index(n) + 1,
                                    **{k: cands[n]["validation"][k] for k in ("falls", "S_mean", "P_mean", "J_mean")})
                               for n in pool_names],
                    conditions=cond, pass_=passes if judged else None,
                    all_survived=bool(all(c["survived"] == c["runs"] for c in cond.values())),
                    label=None if not judged else ("passes G5" if passes else "controller-limited (G5 fails)"))
        # the diagnostic: each selection with the four lower-bound-start gains at zero (ZEROED_RULE)
        todo = {}
        for seed, v in verdicts.items():
            if "selected" in v:
                g = np.asarray(cands[v["selected"]]["gains"], float)
                z = zeroed(g)
                v["zeroed_diagnostic"] = dict(rule=ZEROED_RULE, zeroed=list(LOWER_BOUND_START),
                                              gains=control.gains_dict(z),
                                              identical_to_selected=bool(np.array_equal(z, g)))
                if not v["zeroed_diagnostic"]["identical_to_selected"]:
                    todo.setdefault(tuple(z.tolist()), []).append(seed)
        if todo:
            zs = list(todo)
            log(f"{body}: zeroed diagnostic, {len(zs)} gain sets x {len(val)} validation episodes")
            res = pool.map([(np.asarray(z, float), ep) for z in zs for ep in val])
            for i, z in enumerate(zs):
                part = res[i * len(val) : (i + 1) * len(val)]
                summ = _validation_summary(part)
                for seed in todo[z]:
                    verdicts[seed]["zeroed_diagnostic"].update(validation=summ, conditions=_conditions(summ),
                                                               runs=[_validation_record(x) for x in part])
        for v in verdicts.values():
            zd = v.get("zeroed_diagnostic")
            if zd and zd["identical_to_selected"]:
                summ = cands[v["selected"]]["validation"]
                zd.update(validation=summ, conditions=_conditions(summ), runs="the selected candidate's runs")
        incidents, pool_inputs = list(pool.incidents), pool.inputs
    finally:
        pool.close()
    judged_seeds = [v for v in verdicts.values() if v.get("judged")]
    doc = dict(schema=SCHEMA, body=body, gate="G5", rule=(
        f"plan section 3 step 6: R2 with the selected gains succeeds on at least {G5_MIN_SUCCESS} of the "
        f"{G5_RUNS} validation episodes ({VALIDATION_EPISODES[0]}-{VALIDATION_EPISODES[1] - 1}) of each condition: "
        "the nominal walk, the 0.35 rad/s walking turn and the 0.5 rad/s in-place turn (the turns' directions "
        "alternate with the episode seed); a seed that fails is labelled controller-limited; a seed whose fit ran on "
        "other tuner sources or inputs is not judged; the body passes if a seed passes (seeds_passing; both seeds "
        "passing is plan section 4's 'shown')"),
               success_rule=SUCCESS_RULE, selection_rule=SELECTION_RULE, zeroed_rule=ZEROED_RULE,
               fixed_gains=FIXED_RECORD, objective=OBJECTIVE,
               settings=dict(cma=CMA_SETTINGS, mode=MODE, rate=COMMAND_RATE), validation_episodes=val,
               runs={str(s): v for s, v in runs.items()}, candidates=cands,
               seeds={str(s): v for s, v in verdicts.items()},
               pass_=bool(any(v["pass_"] for v in judged_seeds)) if judged_seeds and judged else None,
               seeds_passing=[s for s, v in verdicts.items() if v.get("pass_")],
               seeds_refused=[s for s, v in verdicts.items() if v.get("refused")], tuner_sources=tsrc,
               inputs=pool_inputs, incidents=incidents, failure_rule=FAILURE_RULE,
               seconds=time.perf_counter() - t0, provenance=prov)
    prov["stage1_sources"]["unchanged_during_run"] = plant.sources_sha256() == prov["stage1_sources"]["sha256"]
    doc["tuner_sources"]["unchanged_during_run"] = tuner_sources()["sha256"] == tsrc["sha256"]
    _write_json(Path(out) / body / "tune.json", doc)
    return doc


def g5_summary(doc: dict) -> str:
    lines = [f"G5 {doc['body']}: " + ("not judged" if doc["pass_"] is None else ("pass" if doc["pass_"] else "FAIL"))]
    for s, v in doc["seeds"].items():
        if "selected" not in v:
            lines.append(f"   seed {s}: {v.get('reason')}")
            continue
        conds = "; ".join(f"{c['condition']} {c['successes']}/{c['runs']} ({c['survived']} survive)"
                          for c in v["conditions"].values())
        sel = next(x for x in v["selection"] if x["candidate"] == v["selected"])
        lines.append(f"   seed {s}: selected {v['selected']} (falls {sel['falls']}, mean S {sel['S_mean']:.4f}, mean P "
                     f"{sel['P_mean']:.4f}); {conds}; {v['label'] or 'smoke check, no verdict'}")
        zd = v.get("zeroed_diagnostic")
        if zd and zd.get("validation"):
            zc = "; ".join(f"{c['condition']} {c['successes']}/{c['runs']}" for c in zd["conditions"].values())
            lines.append(f"      diagnostic, {', '.join(zd['zeroed'])} at zero"
                         + (" (identical to the selection)" if zd["identical_to_selected"] else "")
                         + f": falls {zd['validation']['falls']}, mean P {zd['validation']['P_mean']:.4f}; {zc}")
    if doc.get("incidents"):
        lines.append(f"   {len(doc['incidents'])} worker incidents (tune.json incidents)")
    return "\n".join(lines)
