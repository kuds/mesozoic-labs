"""Stage 2 reference sets: their definitions, and building, verifying and freezing a set directory.

Simulation only: prescribed references on the round-two model; nothing is adopted.

A *reference set* is a directory ``stage2/sets/<set>/<body>/`` in Stage 1's run-directory layout (``setio``), so that
Stage 1's planner, track and tuner load it unchanged: periodic walks in ``refs/<name>.npz`` (Stage 1's reference
format) with a twin in the look ladder's format in ``look/<name>.npz``, moves (a start, speed changes, stops, finite
turns) in ``transitions/<name>.npz``, and the records ``refs.json``, ``transitions.json`` and ``set.json``.  The arrays
are git-ignored and the records are committed.  This module works for any set of ``SETS`` on any body of
``designlab.stage1.BODIES``: nothing here names a body, and a set is its entry in ``SETS`` and nothing else.

1. Definitions (``STAGE2_75_PLAN.md`` A1 to A3)
-----------------------------------------------
``SETS``: ``V0`` (control: today's walk at every speed, ``v0_spec``), ``F25F9`` (control: the pull request's tuned
walk, ``tuned_f9_spec``, copied verbatim) and ``F25F9cl`` (candidate: that walk with duty 0.59 and lift 0.095 H).
``items(set, body)`` lists what a set holds, walks first: the nominal walk and one walk per speed of
``setio.SPEEDS``; ``start``; ``nominal_to_v<v>`` for every speed; the two-touchdown stop ``stop2_v0.4`` and Stage 1's
one-step stop ``stop1_v0.4`` (a labelled comparison); ``turn90_<rate>_<left|right>`` for every rate of
``setio.TURN_RATES``.  Names are made and parsed by ``setio`` only.  An item's ``definition`` is its identity as
asked: the specs, every builder argument, the body's model sha256 and the rules' constants.

2. Building (``build``; the plan's A4 to A12)
---------------------------------------------
One item is built by ``build_item`` in one process, inside ``s2io.r2()`` (accuracy r2 on entry and on exit), on a
body rebuilt for it (``look.source``; a move's arrays never depend on what the process built before):

* **A walk** (``walk_arrays``): ``look.generate``, ``gait.inverse_dynamics``, ``gait.summarize``, Stage 1's arrays
  (``refs.arrays``) and the look ladder's (``look.arrays``) from the one generation, with the look ladder's measures,
  price and flags (``look.measure``, ``look.price``, ``look.flags``).  Gates that stop the build: the two formats
  hold one motion (``q``, ``qd``, ``qdd`` equal); where Stage 1's run holds a walk of the same spec the Stage
  1-format file has its sha256; where the look ladder holds a rung of the same spec the look-format file has its.
* **A move** (``moves``): generated, summarised (the plan's A7 gates: ``pass_``) and screened at Stage 0 with the
  set's own screen (``setscreen.screen_move``: the plan's A12 gates, ``stage0.blocks_freeze``), so that the
  pre-registered consequences are applied where the move is built: a start, change or stop that fails is retried by
  ``RULES["S2S-fallback"]`` (A8) and no other way; a turn follows the stride rule (A11; ``moves.pick_stride_factor``
  once per rate, both directions at the chosen factor, ``moves.step_down`` on the built pair).
* **What has no reference** is an entry too: ``available: false`` with ``reason`` and ``numbers`` and no file (a walk
  over a Stage 0 gate is dropped from the ladder; a move outside its gates; a turn rate without a stride factor or
  outside the heading rule), listed in ``set.json`` ``not_available``.  No constant is changed for it.

``build`` holds the set's lock, sweeps what a killed run left, merges finished items found in ``partial/``, then
runs the missing items, walks before moves, each in a fresh spawned process (``--procs`` at a time).  A worker writes
its arrays into a scratch directory under ``partial/``, moves them into place, writes ``partial/<item>.json`` and
returns; the parent merges that into ``refs.json``, ``transitions.json`` and ``set.json`` (each written atomically)
and deletes the partial in the same step.  An item is kept when its entry holds the definition ``items`` gives now,
was built from the walks the set holds now, and its files have the recorded sha256.  A rebuild that reproduces an
entry's arrays leaves the entry, and its provenance, as they were.  Wall times go to ``logs/sets.log`` in the set
directory and into no record.  **On a frozen set** an entry of the latest manifest is never written again: a missing
array is rebuilt and put in place only when its sha256 is the manifest's, and a difference fails the build with
both hashes; entries the manifest does not hold may still be added.

3. ``verify``, ``freeze``, ``status``
-------------------------------------
``verify`` reads the set as it is (no array is rebuilt) and returns ``{ok, failures, notes}``: the files against
their entries and the latest manifest, names, definitions, the moves' gates, the stops' and turns' own readings, the
walks' two formats, the hash gates, the screen (``screen.json``: a current block for every available entry, none
blocking the freeze) and provenance.  ``freeze`` (the lead's act) appends a manifest of every available entry to
``set.json`` ``freezes``; it needs ``verify`` to pass, a passing ``stage2/sets/legacy_check.json`` of today's
``moves.py``, and one source state on a clean tree unless a flag says otherwise, and it never drops or changes an
entry an earlier freeze holds.

This module is in no record key.

Run from hardware/trex_form_study:
    python -m designlab.stage2.sets build  --set V0 --body 75-M-H-10 [--only NAME ...] [--procs 2] [--root DIR]
    python -m designlab.stage2.sets verify --set V0 --body 75-M-H-10 [--root DIR]
    python -m designlab.stage2.sets status --set V0 --body 75-M-H-10 [--root DIR]
    python -m designlab.stage2.sets freeze --set V0 --body 75-M-H-10 [--allow-mixed] [--allow-dirty]
        [--change-after-result REASON] [--root DIR]
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # before any numpy import: one core per process (bit-reproducible arrays)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402

from .. import envelope, gait, sizing, stage1, statics  # noqa: E402
from ..stage1 import plant, refs, transitions  # noqa: E402
from . import look, moves, s2io, setio, setscreen  # noqa: E402

STUDY = s2io.STUDY
SCHEMA = 1
STAGE1_RUN = STUDY / "stage1" / "runs" / "2026-11-08"
LOOK_REFS = STUDY / "stage2" / "look" / "look_refs.json"
LOOK_SCREEN = STUDY / "stage2" / "look" / "screen.json"
LEGACY_CHECK = setio.ROOT / "legacy_check.json"
#: The pull request's script, whose ``_tuned_f9_spec`` is copied here (``tuned_f9_spec``); read by ``verify`` only.
PR_SCRIPT = STUDY / "stage2" / "jetson_plem" / "jetson_plem_study.py"
#: The Stage 2 modules that make an entry (``s2io.built_by``).
MODULES = ("sets.py", "moves.py", "setscreen.py", "screen.py", "look.py", "setio.py", "s2io.py")
PARTIAL = "partial"
SCRATCH_PREFIX = "build-"
LOG_FILE = "logs/sets.log"
WALK_GENERATOR = "look.generate + gait.inverse_dynamics + refs.arrays"
#: The item kinds, and the builders of ``moves`` they call.
KINDS = ("walk", "start", "change", "stop_two_step", "stop_one_step", "turn")
#: The gates of a turn's entry that are the heading rule (the plan's A10): a turn that fails these and no other has
#: "no reference within the heading rule".
HEADING_GATES = ("heading_total", "heading_at_bar", "heading_at_slowest_clock")
NO_HEADING = "no reference within the heading rule"
NO_TWIST = "no reference within the thigh twist and the leg drives"
NO_GATES = "no transition within the gates"
#: A test's kill point (``STAGE2_SETS_FAIL_AT=<point>[:<item>]``): the process exits at once there, as if killed.
FAIL_AT = "STAGE2_SETS_FAIL_AT"
FAIL_POINTS = ("npz", "partial", "refs.json", "transitions.json", "set.json", "deleted")

RULES = {
    "S2S-sets": (
        "three reference sets per body, each a directory stage2/sets/<set>/<body>/ in Stage 1's run-directory layout: "
        "V0 (control: today's walk), F25F9 (control: the pull request's _tuned_f9_spec, verbatim) and F25F9cl "
        "(candidate: that spec with duty 0.59 and lift 0.095 H at every speed) (STAGE2_75_PLAN.md A1)"),
    "S2S-speeds": (
        "walks nominal (sizing.NOMINAL's Froude number) and v0.3, v0.35, v0.4, v0.45, v0.5 (A2).  V0: sizing.NOMINAL "
        "at the nominal speed; look.speed_spec between sizing.NOMINAL and sizing.BRISK below BRISK's speed and "
        "between sizing.BRISK and envelope.FAST from there to FAST's speed; above FAST's speed FAST's stride, duty "
        "and lift at the higher Froude number, labelled new.  F25F9cl: every rung other than 0.40, 0.45 and 0.50 m/s "
        "labelled new"),
    "S2S-margin": "duty and lift set after results on seeds 3042-3061 (JPS 2.3)",
    "S2S-crouch": (
        "every walk of a set and body has one crouch (a share of the hip height), read from its spec and recorded as "
        "set.json crouch_frac and in every walk entry; a move joins two walks of one crouch (one LIPM frequency)"),
    "S2S-process": (
        "every array is made inside s2io.r2() (gait accuracy r2, checked on entry and on exit) with one BLAS thread, "
        "each item in a fresh process on a body rebuilt for it (look.source); a walk's inverse dynamics, summary and "
        "both array formats come from the one generation in the one block"),
    "S2S-formats": (
        "a walk is written twice from one generation: refs/<name>.npz (Stage 1's reference format, which the bar "
        "plays and a freeze lists) and look/<name>.npz (the look ladder's format, which the Stage 0 screen and the "
        "clips read); the build stops unless q, qd and qdd of the two are equal"),
    "S2S-twins": (
        "where Stage 1's run (stage1/runs/2026-11-08/<body>) holds a walk of the same spec, or a transition between "
        "walks of the same specs made with Stage 1's own schedule, the set's file must have its sha256; where the "
        "look ladder (stage2/look/look_refs.json) holds a rung of the same spec, the look-format file must have its "
        "sha256; a difference stops the build with both hashes (STAGE2_75_PLAN.md A4)"),
    "S2S-stage0": (
        "the Stage 0 gates that block a freeze (STAGE2_75_PLAN.md A12) are read where a reference is built, with the "
        "set screen's own functions: a walk by setscreen.judge_walk on its look.price, a move by "
        "setscreen.screen_move on its file; the set's screen.json is made afterwards by setscreen and must agree "
        "(sets verify)"),
    "S2S-fallback": (
        "a start, change or stop that fails a gate of its entry (A7) or a blocking Stage 0 gate (A12) is retried in "
        "this order and no other (STAGE2_75_PLAN.md A8): (1) a change with a ramp of 2 strides instead of 1 (same "
        "name; the entry records ramp_strides 2 and this rule); (2) for a ladder rung other than 0.40 m/s, a change "
        "from the neighbouring rung toward 0.40 m/s (v0.4_to_v0.45, v0.45_to_v0.5, v0.4_to_v0.35, v0.35_to_v0.3), "
        "with one stride of ramp, the rung's plan then walking through 0.40 m/s; (3) the item is recorded 'no "
        "transition within the gates' with its numbers.  A start and the stops have no fallback and nominal_to_v0.4 "
        "has step (1) only.  A failed start or nominal_to_v0.4 stops the set and goes to the lead"),
    "S2S-turns": (
        "a turn rate is built once for both directions (STAGE2_75_PLAN.md A10, A11): moves.pick_stride_factor on the "
        "set's 0.40 m/s walk, both directions with all passes at the chosen factor, then moves.step_down on the "
        "built pair (ik_ok in both, the worse thigh-twist use, the tighter margin of another joint, a leg drive over "
        "a Stage 0 limit in either, a neck or tail limit in either).  No factor, or a leg drive over its limit: "
        f"'{NO_TWIST}'.  A built turn that fails another gate (A7's, A10's heading readings, another blocking gate "
        f"of A12) has no further retry and no reference; where only the heading readings fail: '{NO_HEADING}'.  A "
        "turn over a neck or tail limit alone is kept with a1_over true"),
    "S2S-not-available": (
        "an item with no reference is an entry with available false, a reason and its numbers, and no file; it is "
        "listed in set.json not_available; a walk that fails a blocking Stage 0 gate is dropped from the ladder in "
        "this way and no move is built from or to it (STAGE2_75_PLAN.md A8, A11, A12).  No ramp, grid, gate or "
        "constant is changed for it"),
    "S2S-freeze": (
        "a freeze lists every available entry: name -> kind, the file's sha256 and the sha256 of the entry's hashed "
        "subset (setio.entry_sha256), a walk's look-format sha256 beside them.  A later freeze may add entries and "
        "never drops or changes one.  It needs sets verify to pass, a passing stage2/sets/legacy_check.json made by "
        "today's moves.py, entries built from one source state and a clean tree (else the manifest says "
        "mixed_sources or dirty).  On a frozen set a build never writes a frozen entry again and puts an array in "
        "place only when its sha256 is the manifest's (STAGE2_75_PLAN.md A13)"),
    "S2S-omega": "control.omega0(p, q_home, crouch_frac=<the set's crouch_frac>): the track's LIPM frequency is the set's own",
}


class SetError(s2io.S2Error):
    """A set cannot be built, verified or frozen as asked."""


class GateError(SetError):
    """A built-in gate of the builder failed (two formats of one walk differ, a file that must hash to Stage 1's or
    to the look ladder's does not): the build stops."""


# ------------------------------------------------------------------------------------------------
# 1. Definitions
# ------------------------------------------------------------------------------------------------
def nominal_speed(H: float) -> float:
    """The nominal walk's speed on a body of hip height ``H`` (``sizing.NOMINAL``'s Froude number), m/s."""
    return float(sizing.NOMINAL.describe(H)["speed"])


def v0_spec(body: str, H: float, v: float | None = None) -> gait.GaitSpec:
    """Today's walk at speed ``v`` (None: the nominal walk) by ``RULES["S2S-speeds"]``."""
    if v is None:
        return sizing.NOMINAL
    brisk, fast = sizing.BRISK.describe(H)["speed"], envelope.FAST.describe(H)["speed"]
    if v < brisk:
        return look.speed_spec(H, v, sizing.NOMINAL, sizing.BRISK)
    if v <= fast:
        return look.speed_spec(H, v, sizing.BRISK, envelope.FAST)
    return replace(envelope.FAST, froude=v * v / (gait.G * H))


def tuned_f9_spec(body: str, H: float, v: float) -> gait.GaitSpec:
    """The pull request's ``_tuned_f9_spec`` (``stage2/jetson_plem/jetson_plem_study.py``), verbatim; ``verify`` and a
    test compare the two field by field at every rung."""
    fr = v * v / (gait.G * H)
    crouch = 0.07 if body.startswith("60") else 0.05
    duty = 0.58 if v < 0.36 else 0.55
    base_sr = 0.713 if body.startswith("60") else 0.863
    sr_cap = 0.90 if body.startswith("60") else 0.94
    sr = round(min(sr_cap, max(0.62, base_sr * ((max(fr, 0.025) / 0.05) ** 0.20))), 4)
    return gait.GaitSpec(
        froude=fr,
        duty=duty,
        lift_frac=0.081,
        head_amp=0.05,
        tail_amp=-0.125,
        sway_gain=1.0,
        stride_rel=sr,
        crouch=crouch,
        zmp_iters=6,
        zmp_gain=0.5,
        n=160,
    )


#: The candidate's margin (``RULES["S2S-margin"]``) and the rungs the pull request ran it at (m/s).
CL_DUTY, CL_LIFT = 0.59, 0.095
CL_RUNGS = (0.40, 0.45, 0.50)


def _f9(body: str, H: float, v: float | None) -> gait.GaitSpec:
    return tuned_f9_spec(body, H, nominal_speed(H) if v is None else v)


def _f9cl(body: str, H: float, v: float | None) -> gait.GaitSpec:
    return replace(_f9(body, H, v), duty=CL_DUTY, lift_frac=CL_LIFT)


def _v0_label(body: str, H: float, v: float | None) -> str | None:
    return "new" if v is not None and v > envelope.FAST.describe(H)["speed"] else None


def _cl_label(body: str, H: float, v: float | None) -> str | None:
    return None if v in CL_RUNGS else "new"


@dataclass(frozen=True)
class SetDef:
    """One reference set: its walk at a speed on a body, that walk's label, and what the set is."""

    name: str
    role: str
    spec_of: Callable[[str, float, float | None], gait.GaitSpec]  # (body, H, speed or None: nominal) -> spec
    label_of: Callable[[str, float, float | None], str | None]
    note: str
    margin_rule: str | None = None
    pr_spec: bool = False  # its walks come from tuned_f9_spec: verify compares that with the pull request's function


SETS: dict[str, SetDef] = {
    "V0": SetDef("V0", "control", v0_spec, _v0_label, "today's walk at every speed"),
    "F25F9": SetDef("F25F9", "control", _f9, lambda body, H, v: None,
                    "the pull request's tuned walk (_tuned_f9_spec), verbatim", pr_spec=True),
    "F25F9cl": SetDef("F25F9cl", "candidate", _f9cl, _cl_label,
                      f"the pull request's tuned walk with duty {CL_DUTY:g} and lift {CL_LIFT:g} H at every speed",
                      margin_rule=RULES["S2S-margin"], pr_spec=True),
}


@dataclass(frozen=True)
class Item:
    """One thing a set holds.  ``definition`` is its identity as asked (JSON-plain); items that are built together
    (the two directions of a turn rate) share ``definition["group"]``."""

    name: str
    kind: str
    depends: tuple
    definition: dict

    @property
    def group(self) -> str:
        return self.definition.get("group", self.name)


def _set(set_name: str) -> SetDef:
    if set_name not in SETS:
        raise SetError(f"{set_name!r} is not a reference set ({', '.join(SETS)})")
    return SETS[set_name]


def body_facts(body: str) -> tuple[float, str]:
    """(hip height H, the model's sha256) of a Stage 1 body from its committed record: what ``look.source`` rebuilds
    and checks (``build_item`` asserts both)."""
    if body not in stage1.BODIES:
        raise SetError(f"{body!r} is not a registered body ({', '.join(stage1.BODIES)})")
    rec = plant.record(body)
    return float(rec["H"]), str(rec["xml_sha256"])


def walk_names() -> list[str]:
    return [setio.NOMINAL] + [setio.walk_name(v) for v in setio.SPEEDS]


def walk_specs(set_name: str, body: str, H: float | None = None) -> dict:
    """name -> spec of every walk of a set on a body; ``SetError`` unless they have one crouch
    (``RULES["S2S-crouch"]``)."""
    st = _set(set_name)
    H = body_facts(body)[0] if H is None else H
    out = {setio.walk_name(v): st.spec_of(body, H, v) for v in (None, *setio.SPEEDS)}
    crouches = sorted({float(s.crouch) for s in out.values()})
    if len(crouches) != 1:
        raise SetError(f"{set_name} {body}: its walks have {len(crouches)} crouches ({crouches}); a set and body has one")
    return out


def neighbour(v: float) -> float | None:
    """The neighbouring rung of ``v`` toward the bar's speed (None for the bar's speed itself)."""
    i = setio.SPEEDS.index(v)
    if v == setio.BAR_SPEED:
        return None
    return setio.SPEEDS[i + (1 if v < setio.BAR_SPEED else -1)]


def stride_constants() -> dict:
    """The stride rule's constants, as a turn item's definition holds them."""
    return dict(grid=[float(f) for f in moves.STRIDE_GRID], limit_kinematic=float(moves.TWIST_KINEMATIC),
                limit_final=float(moves.TWIST_FINAL), limit_other_joints_rad=float(moves.OTHER_JOINT_MARGIN_RAD),
                steps_down_max=int(moves.STEPS_DOWN_MAX))


TURN_RULES = ("S2A-causal", "S2A-rule-F", "S2A-fade", "S2A-ramp", "S2A-appendage", "S2A-stride", "S2A-gates")


def items(set_name: str, body: str) -> list[Item]:
    """Everything the set holds on the body, in build order: the walks, then the moves (module docstring, section
    1).  No model is built: the body's height and model sha256 come from its record."""
    H, model = body_facts(body)
    specs = walk_specs(set_name, body, H)
    out = [Item(name, "walk", (), refs._plain(dict(kind="walk", spec=asdict(spec), body_model_sha256=model,
                                                    generator=WALK_GENERATOR)))
           for name, spec in specs.items()]
    gx = {name: moves.GaitX.of(name, spec, H) for name, spec in specs.items()}
    nom, bar = setio.NOMINAL, setio.walk_name(setio.BAR_SPEED)

    def of(kind: str, mv: moves.Move, depends: tuple, **more) -> Item:
        return Item(mv.name, kind, depends, refs._plain(dict(kind=kind, **mv.definition(), body_model_sha256=model,
                                                             **more)))

    out.append(of("start", moves.start(setio.START, gx[nom]), (nom,), fallback=None))
    for v in setio.SPEEDS:
        b, nb = setio.walk_name(v), neighbour(v)
        steps = [dict(step=1, ramp_strides=2.0)]
        if nb is not None:
            steps.append(dict(step=2, change=setio.change_name(setio.walk_name(nb), b), ramp_strides=1.0))
        out.append(of("change", moves.change(setio.change_name(nom, b), gx[nom], gx[b]),
                      (nom, b) + (() if nb is None else (setio.walk_name(nb),)),
                      fallback=dict(rule="S2S-fallback", steps=steps)))
    out.append(of("stop_two_step", moves.stop_two_step(setio.stop_name(2, bar), gx[bar]), (bar,), fallback=None))
    out.append(of("stop_one_step", moves.stop_one_step(setio.stop_name(1, bar), gx[bar]), (bar,), fallback=None,
                  label="comparison: Stage 1's one-step stop on the same walk (not causal)"))
    for label, rate in setio.TURN_RATES.items():
        for direction in setio.DIRECTIONS:
            sign = 1.0 if direction == "left" else -1.0
            out.append(Item(setio.turn_name(label, direction), "turn", (bar,), refs._plain(dict(
                kind="turn", builder="turn", group=f"turn{setio.TURN_ANGLE_DEG}_{label}",
                args=dict(angle=setio.TURN_ANGLE_RAD, rate=sign * rate, ramp_s=setio.RAMP_S),
                specs=dict(a=gx[bar].plain(), b=gx[bar].plain()), causal=True, laws=moves.TODAY.name,
                rules=list(TURN_RULES), stride_rule=stride_constants(), body_model_sha256=model))))
    return out


def groups(its: list[Item]) -> dict:
    """group -> its items, in build order."""
    out: dict = {}
    for it in its:
        out.setdefault(it.group, []).append(it)
    return out


# ------------------------------------------------------------------------------------------------
# 2. Twins: the files of Stage 1's run and of the look ladder that a set's file must equal
# ------------------------------------------------------------------------------------------------
def _read(path: Path):
    return s2io.read_json(path) if Path(path).is_file() else None


def stage1_walk_twin(body: str, spec: gait.GaitSpec) -> dict | None:
    """``{name, sha256}`` of Stage 1's reference of the same spec on the body (``refs.walk_specs``; the run's
    ``refs.json``), or None."""
    doc = _read(STAGE1_RUN / body / "refs.json")
    if doc is None:
        return None
    for name, s in refs.walk_specs(body).items():
        e = doc["references"].get(name)
        if s == spec and e is not None:
            return dict(name=name, sha256=e["sha256"], record=s2io.portable(str(STAGE1_RUN / body / "refs.json")))
    return None


def stage1_move_twin(body: str, mv: moves.Move, twin_a: dict | None, twin_b: dict | None) -> dict | None:
    """``{name, sha256}`` of Stage 1's transition that ``mv`` reproduces: a move of Stage 1's own schedule (a start,
    a change with Stage 1's ramp, the one-step stop) between walks that are Stage 1's own files (``twin_a``,
    ``twin_b``: their twins, None for the stand); None otherwise."""
    if mv.causal or mv.builder not in ("start", "change", "stop_one_step"):
        return None
    if mv.builder == "change" and mv.args.get("ramp_strides") != transitions.WALK_RAMP_STRIDES:
        return None
    ends = []
    for g, twin in ((mv.a, twin_a), (mv.b, twin_b)):
        if g.spec is None:
            ends.append("stand")
        elif twin is None:
            return None
        else:
            ends.append(twin["name"])
    doc = _read(STAGE1_RUN / body / "transitions.json")
    if doc is None:
        return None
    for name, pair in transitions.TRANSITIONS.items():
        e = doc["transitions"].get(name)
        if tuple(pair) == tuple(ends) and e is not None:
            return dict(name=name, sha256=e["sha256"], record=s2io.portable(str(STAGE1_RUN / body / "transitions.json")))
    return None


def ladder_twin(body: str, spec: gait.GaitSpec) -> dict | None:
    """``{speed, variant, sha256}`` of the look ladder's rung of the same spec on the body
    (``stage2/look/look_refs.json``), or None."""
    doc = _read(LOOK_REFS)
    want = look._plain(asdict(spec))
    for speed, rungs in ((doc or {}).get("bodies", {}).get(body) or {}).items():
        for variant, e in rungs.items():
            if e.get("spec") == want:
                return dict(speed=speed, variant=variant, sha256=e["sha256"], record=s2io.portable(str(LOOK_REFS)))
    return None


# ------------------------------------------------------------------------------------------------
# 3. One item
# ------------------------------------------------------------------------------------------------
def walk_arrays(src: dict, spec: gait.GaitSpec) -> tuple[dict, dict, dict]:
    """One periodic walk of ``src``'s body (``look.source``) from one generation, inside ``s2io.r2()``: (Stage
    1-format arrays, look-format arrays, parts).  ``parts``: ``info`` (the generator's, less its ``look`` block),
    ``summary`` (``gait.summarize``), ``frames``, ``ik_ok``, ``ik_fail_frames``, ``rolls``, ``measured``, ``price``
    and ``flags`` as ``look.write`` computes them, and ``seconds`` (for the log).  ``GateError`` unless ``q``, ``qd``
    and ``qdd`` of the two formats are equal (the largest difference is reported)."""
    if gait.look_active(spec):
        raise SetError("a walk with look features is built by the look phase's builder (moves_look.walk_arrays), not "
                       "by sets.walk_arrays")
    t0 = time.perf_counter()
    with s2io.r2():
        M = src["M"]
        g = look.generate(src, spec)
        idr = gait.inverse_dynamics(M, g)
        summary = gait.summarize(M, g, idr)
        s1 = refs.arrays(src, dict(name=None, traj=g, idr=idr, summary=summary))
        la = look.arrays(src, g)
        info = g["info"]
        rolls = {s: [dict(r) for r in g["look"]["rolls"][s] if 0 <= r["t_lift_off"] < info["window_s"]]
                 for s in statics.SIDES}
        measured = look._plain(look.measure(la, info, src["H"], crouch=spec.crouch))
        price = look._plain(look.price(src, la, info))
        flags = look.flags(spec, info, measured, price, rolls)
    worst = {}
    for key in ("q", "qd", "qdd"):
        a, b = np.asarray(s1[key]), np.asarray(la[key])
        if a.shape != b.shape:
            raise GateError(f"the two formats of one walk hold {key} of shapes {a.shape} and {b.shape}")
        if not np.array_equal(a, b):
            worst[key] = float(np.max(np.abs(a - b)))
    if worst:
        raise GateError("the two formats of one walk differ (" + ", ".join(f"{k}: largest difference {v:.3e}"
                                                                           for k, v in worst.items())
                        + "): the Stage 1-format file's inverse dynamics did not run at the look-format file's "
                          "accuracy; nothing is written")
    parts = dict(info=look._plain({k: v for k, v in info.items() if k != "look"}), summary=summary,
                 frames=len(g["q"]), ik_ok=bool(g["ik_ok"]), ik_fail_frames=len(g["ik_fail_frames"]),
                 rolls=look._plain(rolls), measured=measured, price=price, flags=flags,
                 seconds=time.perf_counter() - t0)
    return s1, la, parts


def walk_stage0(body: str, price: dict) -> dict:
    """A walk's Stage 0 verdict at build time (``RULES["S2S-stage0"]``): ``setscreen.judge_walk`` on the walk's own
    ``look.price`` (its drive ratios, ``over`` list, contact and joint-range margins)."""
    margins = (price.get("clearance") or {}).get("joint_range_margins")
    rng = setscreen.walk_range(margins, plant.usable_ranges(body))
    v = setscreen.judge_walk(dict(drives=price["drives"], over=price["over"], near=[], verdict=price["verdict"],
                                  contact=price["contact"], range=rng))
    worst = max(price["drives"].items(), key=lambda kv: kv[1]["worst"])
    kind = max(setscreen.KINDS, key=lambda k: worst[1][k])
    return s2io.plain(dict(source="setscreen.judge_walk on the walk's look.price (at build time)",
                           governing=[worst[0], kind.replace("_ratio", ""), float(worst[1][kind])],
                           within_limits=v["within_limits"], over=v["over"], blocks_freeze=v["blocks_freeze"],
                           a1_over=v["a1_over"], tightest_joint=rng["tightest"], text=v["text"]))


def _fail_at(point: str, name: str) -> None:
    """A test's kill point: with ``STAGE2_SETS_FAIL_AT=<point>`` or ``<point>:<item>`` the process exits here at
    once, with no cleanup, as a killed one does."""
    want = os.environ.get(FAIL_AT)
    if want and want in (point, f"{point}:{name}"):
        os._exit(86)


def _unavailable(name: str, kind: str, reason: str, numbers: dict, **more) -> dict:
    return dict(name=name, kind=kind, available=False, reason=reason, numbers=numbers, **more)


class _Job:
    """One group of items being built in this process: the set's records as they are, a scratch directory, and what
    the group makes (``out``)."""

    def __init__(self, set_name: str, body: str, group: str, root):
        self.set_name, self.body, self.group = set_name, body, group
        self.st = _set(set_name)
        self.dir = setio.set_dir(set_name, body, root)
        self.items = {i.name: i for i in items(set_name, body)}
        self.mine = [i for i in self.items.values() if i.group == group]
        if not self.mine:
            raise SetError(f"{set_name} {body}: no item or group {group!r}")
        self.H, self.model = body_facts(body)
        self.specs = walk_specs(set_name, body, self.H)
        self.scratch = self.dir / PARTIAL / f"{SCRATCH_PREFIX}{group}"
        refs_doc, tr_doc = _read(self.dir / "refs.json"), _read(self.dir / "transitions.json")
        self.refs = (refs_doc or {}).get("references", {})
        self.trans = (tr_doc or {}).get("transitions", {})
        self.out = dict(item=group, set=set_name, body=body, refs={}, transitions={}, look_entries={},
                        stride_rules={}, files={}, header=None, blocks_set=None, log=[])
        self._src = None

    # --- the body
    def source(self, fresh: bool = True) -> dict:
        """The body, rebuilt (``look.source``) unless ``fresh`` is false and one is at hand; checked against the
        record the definitions were made from."""
        if fresh or self._src is None:
            src = look.source(self.body)
            if float(src["H"]) != self.H or src["model"]["file_sha256"] != self.model:
                raise SetError(f"{self.body}: the rebuilt body (H {src['H']!r}, model {src['model']['file_sha256'][:12]}...) "
                               f"is not its record's (H {self.H!r}, {self.model[:12]}...)")
            self._src = src
            if self.out["header"] is None:
                self.out["header"] = dict(model=src["model"], H_m=float(src["H"]),
                                          q_home=[float(v) for v in src["q_home"]], joints=list(src["M"].names))
        return self._src

    def built_by(self) -> dict:
        return s2io.built_by(list(MODULES), own_outputs=[str(self.dir)])

    # --- the walks a move reads
    def walk(self, name: str) -> tuple[moves.GaitX, refs.Reference, dict, dict | None] | None:
        """(gait, reference, entry, Stage 1 twin) of a walk the set holds, its file checked against its entry and
        its entry against the definition made now; None where the set records it as not available."""
        e = self.refs.get(name)
        if e is None:
            raise SetError(f"{self.set_name} {self.body}: the walk {name!r} is not built yet")
        if e.get("available") is False:
            return None
        spec = self.specs[name]
        if e.get("definition") != self.items[name].definition:
            raise SetError(f"{self.set_name} {self.body}: the walk {name!r} in refs.json is not the one defined now; "
                           "build it again first")
        path = self.dir / e["file"]
        if not path.is_file() or s2io.sha256_file(path) != e["sha256"]:
            raise SetError(f"{path} is missing or is not the file its entry records")
        return (moves.GaitX.of(name, spec, self.H), refs.Reference.load(path), e,
                stage1_walk_twin(self.body, spec))

    def inputs(self, names) -> dict:
        """walk -> the sha256 of its Stage 1-format file as the set records it (None: not available)."""
        return {n: (None if self.refs[n].get("available") is False else self.refs[n]["sha256"]) for n in names}

    def missing_walk(self, item: Item, names) -> dict | None:
        """The entry of a move one of whose walks the set does not hold as a rung, or None."""
        gone = [n for n in names if self.refs.get(n, {}).get("available") is False]
        if not gone:
            return None
        why = "; ".join(f"{n}: {self.refs[n].get('reason')}" for n in gone)
        return _unavailable(item.name, item.kind,
                            f"the walk {', '.join(gone)} is not a rung of the set ({why})",
                            dict(walks={n: self.refs[n].get("numbers") for n in gone}), item=item.definition,
                            group=item.group, inputs=self.inputs(names), set=self.set_name, pass_=None)

    # --- arrays and entries
    def put(self, sub: str, name: str, arrays: dict) -> str:
        rel = f"{sub}/{name}.npz"
        sha = s2io.write_npz(self.scratch / rel, arrays)
        self.out["files"][rel] = sha
        return sha

    def drop(self, sub: str, name: str) -> None:
        rel = f"{sub}/{name}.npz"
        self.out["files"].pop(rel, None)
        (self.scratch / rel).unlink(missing_ok=True)

    def note(self, what: str, seconds: float, text: str) -> None:
        self.out["log"].append(dict(what=what, seconds=float(seconds), text=text))

    def move(self, item: Item, mv: moves.Move, ref_a, ref_b, e_a: dict | None, e_b: dict | None,
             twin_a: dict | None, twin_b: dict | None, depends) -> tuple[dict, dict]:
        """Generate one move on a body rebuilt for it, write its file into the scratch directory and screen it at
        Stage 0: (entry, flags).  ``flags``: ``ok`` (the entry's gates pass and nothing blocks the freeze),
        ``over_leg_drive`` and ``a1_over`` (``setscreen.drive_flags``), ``tightest_other``."""
        t0 = time.perf_counter()
        src = self.source()
        with s2io.r2():
            g = moves.generate(src, mv, ref_a)
            entry = moves.summary(g, ref_a, ref_b, src["q_home"])
            arrs = moves.arrays(g)
            sha = self.put("transitions", mv.name, arrs)
            other = moves.tightest_other(src, g["q"])
            prov = self.built_by()
        entry.update(file=f"transitions/{mv.name}.npz", sha256=sha)
        twin = stage1_move_twin(self.body, mv, twin_a, twin_b)
        if twin is not None and twin["sha256"] != sha:
            raise GateError(f"{self.set_name} {self.body} {mv.name}: its file hashes to {sha}, Stage 1's "
                            f"{twin['name']} ({twin['record']}) to {twin['sha256']}: the move is Stage 1's own "
                            "transition and must reproduce it")
        block = setscreen.screen_move(src, self.scratch, mv.name, entry, None)
        with s2io.r2():
            df = setscreen.drive_flags(src, arrs["tau"], arrs["omega"], arrs["t"])
        v = block["verdict"]
        if bool(v["a1_over"]) != bool(df["a1_over"]):
            raise SetError(f"{mv.name}: the screen's a1_over ({v['a1_over']}) is not setscreen.drive_flags' ({df['a1_over']})")
        entry.update(
            available=True, set=self.set_name, group=item.group, item=item.definition,
            definition=refs._plain(dict(mv.definition(), body_model_sha256=self.model,
                                        ref_a_sha256=None if e_a is None else e_a["sha256"],
                                        ref_b_sha256=None if e_b is None else e_b["sha256"])),
            inputs=self.inputs(depends), a1_over=bool(v["a1_over"]), fallback=None, tightest_other=other,
            stage1_sha256=None if twin is None else twin["sha256"], equals_stage1=None if twin is None else True,
            stage1_twin=None if twin is None else twin["name"],
            stage0=dict(source="setscreen.screen_move on the built file (at build time)", governing=block["governing"],
                        within_limits=v["within_limits"], over=v["over"], blocks_freeze=v["blocks_freeze"],
                        a1_over=v["a1_over"], over_leg_drive=bool(df["over_leg_drive"]), text=v["text"],
                        tightest_joint=block["range"].get("tightest"),
                        accel_forward_lp1hz_peak_m_s2=block["accel"].get("com_forward_lp1hz_peak")),
            built_by=prov)
        entry = refs._plain(entry)
        ok = bool(entry["pass_"] and not v["blocks_freeze"])
        failed = sorted(k for k, x in entry["gates"].items() if not x)
        self.note(mv.name, time.perf_counter() - t0,
                  f"pass_ {entry['pass_']}" + (f" (failing: {', '.join(failed)})" if failed else "")
                  + f"; Stage 0: {v['text']}; governing {block['governing'][0]} {block['governing'][1]} "
                    f"{block['governing'][2]:.4f}; thigh twist {entry['hip_yaw_range_use']:.4f}"
                  + ("" if other is None else f"; tightest other joint {other['joint']} {other['end']} "
                                              f"{other['margin_rad']:.4f} rad"))
        return entry, dict(ok=ok, over_leg_drive=bool(df["over_leg_drive"]), a1_over=bool(df["a1_over"]),
                           tightest_other=other, failed=failed, blocks=v["blocks_freeze"])


def _numbers(entry: dict, flags: dict, **more) -> dict:
    """What a failed build leaves in the record: its gates, joins, histories and what blocked at Stage 0."""
    keep = ("gates", "pass_", "ik_ok", "ik_fail_frames", "join_a", "join_b", "join_command", "zmp_error_history_m",
            "zmp_consistency_history_m", "post_window_zmp_error_m", "zmp_consistency_all_m", "cop_margin_single_min",
            "min_cop_margin_single_m", "hip_yaw_range_use", "tightest_other", "reference", "turn", "frames",
            "duration_s")
    out = {k: entry[k] for k in keep if k in entry}
    if isinstance(out.get("turn"), dict):
        out["turn"] = {k: v for k, v in out["turn"].items() if k != "stride_rule"}
    out.update(failed_gates=flags["failed"], stage0=entry.get("stage0"), built_sha256=entry.get("sha256"), **more)
    return out


def _to_unavailable(job: _Job, item: Item, entry: dict, flags: dict, reason: str, **more) -> dict:
    """A built move that has no reference: its file is not kept; its numbers are."""
    job.drop("transitions", entry["name"])
    return refs._plain(_unavailable(
        entry["name"], item.kind, reason, _numbers(entry, flags), item=entry["item"], group=item.group,
        inputs=entry["inputs"], set=job.set_name, pass_=entry["pass_"], gates=entry["gates"],
        definition=entry["definition"], built_by=entry["built_by"], **more))


def _produce_walk(job: _Job, item: Item) -> None:
    name, body, st = item.name, job.body, job.st
    spec = job.specs[name]
    v = setio.parse_name(name)["speed_m_s"]
    src = job.source()
    s1, la, parts = walk_arrays(src, spec)
    with s2io.r2():
        sha = job.put("refs", name, s1)
        sha_look = job.put("look", name, la)
        prov = job.built_by()
    twin, ladder = stage1_walk_twin(body, spec), ladder_twin(body, spec)
    if twin is not None and twin["sha256"] != sha:
        raise GateError(f"{job.set_name} {body} {name}: its Stage 1-format file hashes to {sha}, Stage 1's "
                        f"{twin['name']} ({twin['record']}) to {twin['sha256']}")
    if ladder is not None and ladder["sha256"] != sha_look:
        raise GateError(f"{job.set_name} {body} {name}: its look-format file hashes to {sha_look}, the look ladder's "
                        f"{ladder['speed']} {ladder['variant']} ({ladder['record']}) to {ladder['sha256']}")
    info, summ, price = parts["info"], parts["summary"], parts["price"]
    le = dict(body=body, speed=name, variant=job.set_name, features=[], file=f"look/{name}.npz", sha256=sha_look,
              frames=parts["frames"], spec=look._plain(asdict(spec)), info=info, ik_ok=parts["ik_ok"],
              ik_fail_frames=parts["ik_fail_frames"], zmp_error_final_mm=1e3 * info["zmp_error_history_m"][-1],
              zmp_margin_min_mm=1e3 * info["zmp_margin_min_m"], rolls=parts["rolls"], measured=parts["measured"],
              price=price, flags=parts["flags"])
    stage0 = walk_stage0(body, price)
    entry = dict(
        name=name, file=f"refs/{name}.npz", sha256=sha, frames=parts["frames"], dt_s=float(s1["dt"]),
        period_s=float(info["period"]), speed_m_s=float(info["speed"]), stride_m=float(info["stride"]),
        lift_m=float(info["lift"]), cadence_steps_s=float(info["cadence_hz"]), yaw_rate_rad_s=0.0,
        stride_motion=[float(x) for x in s1["stride_motion"]],
        min_cop_margin_single_m=float(summ["min_cop_margin_single"]),
        max_friction_ratio=float(summ["max_friction_ratio"]), ik_ok=bool(summ["ik_ok"]),
        generator="look.generate (gait.generate at accuracy r2)", dynamics="gait.inverse_dynamics",
        spec=look._plain(asdict(spec)), info=info,
        set=job.set_name, kind="walk", label=st.label_of(body, job.H, v), margin_rule=st.margin_rule,
        crouch_frac=float(spec.crouch), strides=int(info.get("strides", 1)), window_s=float(info["window_s"]),
        look_file=f"look/{name}.npz", look_sha256=sha_look,
        stage1_sha256=None if twin is None else twin["sha256"], equals_stage1=None if twin is None else True,
        stage1_twin=None if twin is None else twin["name"],
        ladder_sha256=None if ladder is None else ladder["sha256"], equals_ladder=None if ladder is None else True,
        ladder_twin=None if ladder is None else [ladder["speed"], ladder["variant"]],
        stage0=stage0, available=True, group=item.group, definition=item.definition, built_by=prov)
    if entry["strides"] != 1 or entry["window_s"] != entry["period_s"]:
        raise SetError(f"{name}: a walk of today's laws is one stride (strides {entry['strides']}, window "
                       f"{entry['window_s']!r} s, period {entry['period_s']!r} s)")
    text = (f"ik {entry['ik_ok']}; governing {stage0['governing'][0]} {stage0['governing'][1]} "
            f"{stage0['governing'][2]:.4f}; Stage 0: {stage0['text']}"
            + ("" if twin is None else f"; equals Stage 1's {twin['name']}")
            + ("" if ladder is None else f"; equals the look ladder's {ladder['speed']} {ladder['variant']}"))
    job.note(name, parts["seconds"], text)
    if stage0["blocks_freeze"]:
        job.drop("refs", name)
        job.drop("look", name)
        entry = _unavailable(
            name, "walk", "dropped from the ladder: " + stage0["text"],
            dict(stage0=stage0, speed_m_s=entry["speed_m_s"], period_s=entry["period_s"], ik_ok=entry["ik_ok"],
                 min_cop_margin_single_m=entry["min_cop_margin_single_m"], built_sha256=sha,
                 built_look_sha256=sha_look, price=price),
            set=job.set_name, label=entry["label"], margin_rule=entry["margin_rule"], crouch_frac=entry["crouch_frac"],
            spec=entry["spec"], dropped=True, group=item.group, definition=item.definition, built_by=prov)
    else:
        job.out["look_entries"][name] = look._plain(le)
    job.out["refs"][name] = refs._plain(entry)


def _produce_straight(job: _Job, item: Item) -> None:
    """A start, a speed change or a stop, with the fallbacks of ``RULES["S2S-fallback"]``."""
    name, kind = item.name, item.kind
    gone = job.missing_walk(item, item.depends[: 2 if kind == "change" else 1])
    if gone is not None:
        job.out["transitions"][name] = gone
        job.note(name, 0.0, "not available: " + gone["reason"])
        return
    nom, bar = setio.NOMINAL, setio.walk_name(setio.BAR_SPEED)
    blocks = kind == "start" or name == setio.change_name(nom, bar)
    if kind == "change":
        p = setio.parse_name(name)
        (ga, ra, ea, ta), (gb, rb, eb, tb) = job.walk(p["a"]), job.walk(p["b"])
        tried = []
        for step, ramp in ((0, 1.0), (1, 2.0)):
            mv = moves.change(name, ga, gb, ramp_strides=ramp)
            entry, fl = job.move(item, mv, ra, rb, ea, eb, ta, tb, (p["a"], p["b"]))
            entry["ramp_strides"] = ramp
            if fl["ok"]:
                if step:
                    entry["fallback"] = dict(rule="S2S-fallback", text=RULES["S2S-fallback"], step=1, ramp_strides=ramp,
                                             tried=tried)
                job.out["transitions"][name] = entry
                return
            tried.append(_numbers(entry, fl, ramp_strides=ramp, change=name))
            job.drop("transitions", name)
        v = setio.parse_name(p["b"])["speed_m_s"]
        nb = None if v not in setio.SPEEDS else neighbour(v)
        if nb is not None:
            nbn = setio.walk_name(nb)
            other = setio.change_name(nbn, p["b"])
            got = job.walk(nbn)
            if got is None:
                tried.append(dict(change=other, not_built=f"the walk {nbn} is not a rung of the set"))
            else:
                gn, rn, en, tn = got
                e2, fl2 = job.move(item, moves.change(other, gn, gb), rn, rb, en, eb, tn, tb, (nbn, p["b"]))
                e2["ramp_strides"] = 1.0
                if fl2["ok"]:
                    e2["fallback"] = dict(rule="S2S-fallback", text=RULES["S2S-fallback"], step=2, of=name, tried=tried)
                    job.out["transitions"][other] = e2
                    job.out["transitions"][name] = refs._plain(_unavailable(
                        name, kind, f"no direct transition within the gates; the rung is reached from {nbn} "
                                    f"({other}; fallback step 2)", dict(tried=tried), item=item.definition,
                        group=item.group, inputs=job.inputs((p["a"], p["b"], nbn)), set=job.set_name, pass_=False,
                        fallback=dict(rule="S2S-fallback", step=2, by=other)))
                    return
                tried.append(_numbers(e2, fl2, ramp_strides=1.0, change=other))
                job.drop("transitions", other)
        job.out["transitions"][name] = refs._plain(_unavailable(
            name, kind, NO_GATES, dict(tried=tried), item=item.definition, group=item.group,
            inputs=job.inputs(item.depends), set=job.set_name, pass_=False,
            fallback=dict(rule="S2S-fallback", step=3, tried=len(tried))))
        if blocks:
            job.out["blocks_set"] = f"{name}: {NO_GATES} ({RULES['S2S-fallback']})"
        return
    if kind == "start":
        gb, rb, eb, tb = job.walk(nom)
        entry, fl = job.move(item, moves.start(name, gb), None, rb, None, eb, None, tb, (nom,))
    else:
        ga, ra, ea, ta = job.walk(bar)
        make = moves.stop_two_step if kind == "stop_two_step" else moves.stop_one_step
        entry, fl = job.move(item, make(name, ga), ra, None, ea, None, ta, None, (bar,))
        if item.definition.get("label"):
            entry["label"] = item.definition["label"]
    if fl["ok"]:
        job.out["transitions"][name] = entry
        return
    job.out["transitions"][name] = _to_unavailable(job, item, entry, fl, NO_GATES,
                                                   fallback=dict(rule="S2S-fallback", step=3, tried=1))
    if blocks:
        job.out["blocks_set"] = f"{name}: {NO_GATES} ({RULES['S2S-fallback']})"


def _produce_turns(job: _Job) -> None:
    """One turn rate, both directions (``RULES["S2S-turns"]``)."""
    its = {setio.parse_name(i.name)["direction"]: i for i in job.mine}
    first = job.mine[0]
    p = setio.parse_name(first.name)
    label, rate, angle = p["rate"], p["rate_rad_s"], p["angle_rad"]
    bar = first.depends[0]
    gone = [job.missing_walk(i, (bar,)) for i in job.mine]
    if gone[0] is not None:
        for e in gone:
            job.out["transitions"][e["name"]] = e
        job.out["stride_rules"][label] = dict(outcome="none", chosen=None, reason=gone[0]["reason"], references={
            e["name"]: False for e in gone}, no_reference=gone[0]["reason"])
        return
    ga, ra, ea, ta = job.walk(bar)
    t0 = time.perf_counter()
    c = first.definition["stride_rule"]
    with s2io.r2():
        rec = moves.pick_stride_factor(job.source(), ga, ra, angle, rate, grid=tuple(c["grid"]),
                                       limit=c["limit_kinematic"], other_margin=c["limit_other_joints_rad"])
    rec = refs._plain(dict(rec, walk=bar, walk_sha256=ea["sha256"]))
    job.note(f"stride rule {label}", time.perf_counter() - t0, "tried " + "; ".join(
        f"f {r['f']}: twist {r['left']} / {r['right']}, other "
        + " / ".join("-" if (r.get("tightest_other") or {}).get(d) is None else
                     f"{r['tightest_other'][d]['joint']} {r['tightest_other'][d]['margin_rad']:.4f}"
                     for d in setio.DIRECTIONS) + f", ik {r['ik_ok']}, pass {r['pass_']}" for r in rec["tried"])
             + f"; chosen {rec['chosen']}")
    built: dict = {}
    while rec["chosen"] is not None and rec["outcome"] in ("picked", "rebuild"):
        f = rec["chosen"]
        built = {}
        for direction, item in its.items():
            sign = 1.0 if direction == "left" else -1.0
            mv = moves.turn(item.name, ga, angle, sign * rate, f, ramp_s=item.definition["args"]["ramp_s"])
            built[direction] = job.move(item, mv, ra, ra, ea, ea, ta, ta, (bar,))
        others = {d: fl["tightest_other"] for d, (_, fl) in built.items()}
        rec = refs._plain(moves.step_down(rec, dict(
            ik_ok=bool(all(e["ik_ok"] for e, _ in built.values())),
            hip_yaw_range_use=float(max(e["hip_yaw_range_use"] for e, _ in built.values())),
            tightest_other_rad=min(o["margin_rad"] for o in others.values()) if all(others.values()) else None,
            tightest_other=others,
            over_leg_drive=bool(any(fl["over_leg_drive"] for _, fl in built.values())),
            a1_over=bool(any(fl["a1_over"] for _, fl in built.values())),
            pass_={d: bool(e["pass_"]) for d, (e, _) in built.items()},
            stage0_blocks={d: fl["blocks"] for d, (_, fl) in built.items()})))
        job.note(f"stride rule {label}", 0.0, f"built at f {f:g}: {rec['outcome']}"
                 + ("" if rec["reason"] is None else f" ({rec['reason']})"))
    references, why = {}, None
    for direction, item in its.items():
        name = item.name
        if rec["outcome"] != "frozen":
            why = f"{NO_TWIST}: {rec['reason']}"
            if direction in built:
                entry, fl = built[direction]
                job.out["transitions"][name] = _to_unavailable(job, item, entry, fl, why)
            else:
                job.out["transitions"][name] = refs._plain(_unavailable(
                    name, item.kind, why, dict(stride_rule="set.json stride_rules[" + label + "]"), item=item.definition,
                    group=item.group, inputs=job.inputs((bar,)), set=job.set_name, pass_=None))
            references[name] = False
            continue
        entry, fl = built[direction]
        if fl["ok"]:
            references[name] = True
            job.out["transitions"][name] = entry
            continue
        only_heading = bool(fl["failed"]) and set(fl["failed"]) <= set(HEADING_GATES) and not fl["blocks"]
        why = NO_HEADING if only_heading else (
            "no reference within the gates (" + "; ".join(
                ([f"entry gates: {', '.join(fl['failed'])}"] if fl["failed"] else [])
                + ([f"Stage 0: {entry['stage0']['text']}"] if fl["blocks"] else [])) + ")")
        job.out["transitions"][name] = _to_unavailable(job, item, entry, fl, why)
        references[name] = False
    rec = dict(rec, references=references, no_reference=None if all(references.values()) else why)
    job.out["stride_rules"][label] = rec
    for name in references:
        e = job.out["transitions"][name]
        if e.get("available") is not False:
            e["turn"]["stride_rule"] = rec
        else:
            e["numbers"]["stride_rule"] = {k: v for k, v in rec.items() if k != "text"}


def produce(set_name: str, body: str, name: str, root=setio.ROOT) -> dict:
    """Everything of one item, or of the group it is built with, made in this process: the arrays written into the
    group's scratch directory (``partial/build-<group>/``), and the result ``{item, set, body, refs, transitions,
    look_entries, stride_rules, files {relative path: sha256}, header, blocks_set, log}``.  Nothing of the set is
    changed."""
    its = {i.name: i for i in items(set_name, body)}
    group = its[name].group if name in its else name
    job = _Job(set_name, body, group, root)
    shutil.rmtree(job.scratch, ignore_errors=True)
    job.scratch.mkdir(parents=True)
    kind = job.mine[0].kind
    if kind == "walk":
        _produce_walk(job, job.mine[0])
    elif kind == "turn":
        _produce_turns(job)
    else:
        _produce_straight(job, job.mine[0])
    return job.out  # (header None: nothing was generated, as for a move whose walk is not a rung; no body was built)


def partial_path(d: Path, group: str) -> Path:
    return d / PARTIAL / f"{group}.json"


def _frozen_rows(d: Path) -> dict:
    """name -> row of the set's latest manifest ({} for a set that is not frozen, or not built)."""
    doc = _read(d / "set.json")
    fz = None if doc is None else setio.frozen(doc)
    return {} if fz is None else fz["entries"]


def _expected(rows: dict, entries: dict, rel: str) -> str | None:
    """The sha256 a frozen set holds for the file ``rel`` (``refs/x.npz``, ``transitions/x.npz``, ``look/x.npz``),
    or None where the latest manifest does not list its name."""
    sub, stem = rel.split("/")[0], Path(rel).stem
    row = rows.get(stem)
    if row is None:
        return None
    if sub == "look":
        return row.get("look_sha256") or (entries.get(stem) or {}).get("look_sha256")
    return row["sha256"]


def build_item(set_name: str, body: str, name: str, root=setio.ROOT) -> dict:
    """One item (or its group) in this process: ``produce``, then its arrays moved from the scratch directory into
    place and ``partial/<group>.json`` written.  Returns that partial's content.  On a frozen set a file of an entry
    the latest manifest holds is put in place only when its sha256 is the manifest's; the others are listed under
    ``mismatch`` and left out."""
    s2io.require_one_thread()
    out = produce(set_name, body, name, root)
    d = setio.set_dir(set_name, body, root)
    group = out["item"]
    scratch = d / PARTIAL / f"{SCRATCH_PREFIX}{group}"
    rows = _frozen_rows(d)
    entries = ((_read(d / "refs.json") or {}).get("references", {}) if rows else {})
    out["mismatch"] = []
    for name_, e in {**out["refs"], **out["transitions"]}.items():
        if name_ in rows and e.get("available") is False:
            out["mismatch"].append(dict(name=name_, frozen=rows[name_]["sha256"], built=None,
                                        why=f"built now it has no reference: {e.get('reason')}"))
    for rel, sha in sorted(out["files"].items()):
        want = _expected(rows, entries, rel)
        if want is not None and want != sha:
            out["mismatch"].append(dict(name=Path(rel).stem, file=rel, frozen=want, built=sha,
                                        why="the rebuilt file is not the frozen one"))
            continue
        dest = d / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch / rel, dest)
    shutil.rmtree(scratch, ignore_errors=True)
    _fail_at("npz", group)
    s2io.write_json(partial_path(d, group), out)
    _fail_at("partial", group)
    return out


def _worker(args) -> str:
    """A pool worker: one group in a fresh process; the group's name."""
    set_name, body, group, root = args
    s2io.require_one_thread()
    build_item(set_name, body, group, root)
    return group


# ------------------------------------------------------------------------------------------------
# 4. The set's records, and merging a finished item into them
# ------------------------------------------------------------------------------------------------
def _new_docs(set_name: str, body: str, header: dict) -> dict:
    st = _set(set_name)
    H, _ = body_facts(body)
    specs = walk_specs(set_name, body, H)
    crouch = float(next(iter(specs.values())).crouch)
    speeds = {setio.walk_name(v): (nominal_speed(H) if v is None else float(v)) for v in (None, *setio.SPEEDS)}
    return {
        "refs.json": dict(
            body=body, set=set_name, schema=SCHEMA, accuracy=s2io.ACCURACY, model=header["model"], H_m=header["H_m"],
            q_home=header["q_home"], joints=header["joints"],
            feedforward="inverse-dynamics joint torque, contact-consistent; MJCF joint damping and friction not "
                        "included",
            rules={k: RULES[k] for k in ("S2S-sets", "S2S-speeds", "S2S-margin", "S2S-crouch", "S2S-process",
                                         "S2S-formats", "S2S-twins", "S2S-stage0", "S2S-not-available")},
            references={}),
        "transitions.json": dict(
            body=body, set=set_name, schema=SCHEMA, transitions={},
            method="designlab.stage2.moves: Stage 1's transition (ZMP preview control on the designlab model) with "
                   "its schedule passed as data; a commanded move (a stop, a turn) is causal",
            preview=dict(qe=transitions.PREVIEW_QE, r=transitions.PREVIEW_R, preview_s=transitions.PREVIEW_S),
            shift_strides=transitions.SHIFT_STRIDES, settle_strides=transitions.SETTLE_STRIDES,
            path_half_width=transitions.PATH_HALF_WIDTH, walk_ramp_strides=transitions.WALK_RAMP_STRIDES,
            windows=transitions.WINDOWS, g3_com_mm=transitions.G3_COM_MM, g3_zmp_mm=transitions.G3_ZMP_MM,
            rules=dict(moves.RULES, **{k: RULES[k] for k in ("S2S-process", "S2S-twins", "S2S-stage0", "S2S-fallback",
                                                              "S2S-turns", "S2S-not-available")})),
        "set.json": dict(
            set=set_name, body=body, schema=SCHEMA, role=st.role, note=st.note, spec_rule=RULES["S2S-speeds"],
            margin_rule=st.margin_rule, speeds=speeds, crouch_frac=crouch, omega_rule=RULES["S2S-omega"], rules=RULES,
            look_entries={}, stride_rules={}, not_available={}, blocked=None, freezes=[]),
    }


def load_docs(d: Path) -> dict | None:
    """The set's three records as they are, or None while the set has none (a set never built)."""
    got = {n: _read(d / n) for n in ("refs.json", "transitions.json", "set.json")}
    have = [n for n, v in got.items() if v is not None]
    if not have:
        return None
    if len(have) != 3:
        raise SetError(f"{d}: it holds {', '.join(have)} and not the other record(s) of a set")
    return got


def _files_ok(d: Path, entry: dict, look_too: bool) -> bool:
    for key_f, key_s in (("file", "sha256"),) + ((("look_file", "look_sha256"),) if look_too else ()):
        p = d / entry[key_f]
        if not p.is_file() or s2io.sha256_file(p) != entry[key_s]:
            return False
    return True


def _entry_of(docs: dict, item_kind: str, name: str) -> dict | None:
    if item_kind == "walk":
        return docs["refs.json"]["references"].get(name)
    return docs["transitions.json"]["transitions"].get(name)


def _same_inputs(docs: dict, entry: dict) -> bool:
    """Whether the walks an entry was built from are the ones the set holds now (their sha256, or their being no
    rung)."""
    refs_ = docs["refs.json"]["references"]
    for walk, sha in (entry.get("inputs") or {}).items():
        e = refs_.get(walk)
        now = None if e is None or e.get("available") is False else e.get("sha256")
        if e is None or now != sha:
            return False
    return True


def current(d: Path, docs: dict | None, item: Item, rows: dict) -> bool:
    """Whether an item needs no build: its entry holds the definition made now, was built from the walks the set
    holds now, and its files have the recorded sha256 (an entry with no reference has no file).  An entry of the
    latest manifest (``rows``) is current when its files are in place: it is never built into the records again."""
    if docs is None:
        return False
    e = _entry_of(docs, item.kind, item.name)
    if e is None:
        return False
    if item.name in rows:
        return _files_ok(d, e, item.kind == "walk")
    if e.get("item" if item.kind != "walk" else "definition") != item.definition or not _same_inputs(docs, e):
        return False
    if e.get("available") is False:
        return True
    return _files_ok(d, e, item.kind == "walk")


def _write_if_changed(path: Path, doc: dict, old: dict | None, point: str, group: str) -> bool:
    if old is not None and s2io.canonical(doc) == s2io.canonical(old):  # (canonical: a stand's NaN equals itself)
        return False
    s2io.write_json(path, doc)
    _fail_at(point, group)
    return True


def merge(d: Path, part: dict, log=print) -> dict:
    """Merge one finished item (a ``partial/<group>.json``'s content) into the set's records and delete the partial.
    An entry the latest manifest holds is never written; an entry whose arrays were reproduced (same definition,
    same inputs, same sha256) is left as it was, provenance included.  Returns ``{written [...], kept [...],
    frozen [...]}``."""
    set_name, body, group = part["set"], part["body"], part["item"]
    docs = load_docs(d)
    old = None if docs is None else {k: json.loads(json.dumps(v)) for k, v in docs.items()}
    if docs is None:
        if part.get("header") is None:
            raise SetError(f"{d}: the set has no record yet and {group} built no body to make one from")
        docs = _new_docs(set_name, body, part["header"])
    rows = setio.frozen(docs["set.json"])
    rows = {} if rows is None else rows["entries"]
    res = dict(written=[], kept=[], frozen=[])
    set_doc = docs["set.json"]
    for key, target in (("refs", docs["refs.json"]["references"]), ("transitions", docs["transitions.json"]["transitions"])):
        made = part[key]
        if key == "transitions":  # an entry this group made before and does not make now (a fallback's change)
            for name in [n for n, e in target.items() if e.get("group") == group and n not in made and n not in rows]:
                del target[name]
                set_doc["not_available"].pop(name, None)
                res["written"].append(name)
        for name, e in made.items():
            if name in rows:
                res["frozen"].append(name)
                continue
            was = target.get(name)
            same = (was is not None and was.get("available") is not False and e.get("available") is not False
                    and was.get("sha256") == e.get("sha256") and was.get("item") == e.get("item")
                    and was.get("definition") == e.get("definition") and was.get("inputs") == e.get("inputs")
                    and was.get("look_sha256") == e.get("look_sha256"))
            if same:
                res["kept"].append(name)
            else:
                target[name] = e
                res["written"].append(name)
            now = target[name]  # what set.json says of it follows the entry (also when this merge is a repeated one)
            if now.get("available") is False:
                set_doc["not_available"][name] = dict(reason=now["reason"], numbers=now.get("numbers"))
                set_doc["look_entries"].pop(name, None)
            else:
                set_doc["not_available"].pop(name, None)
                if key == "refs" and (set_doc["look_entries"].get(name) or {}).get("sha256") != now["look_sha256"]:
                    set_doc["look_entries"][name] = part["look_entries"][name]
    for label, rec in part.get("stride_rules", {}).items():
        names = [setio.turn_name(label, x) for x in setio.DIRECTIONS]
        if not any(n in rows for n in names):
            set_doc["stride_rules"][label] = rec
    if res["written"] and part.get("header") is not None:
        for k, v in part["header"].items():
            docs["refs.json"][k] = v
    if part.get("blocks_set"):
        set_doc["blocked"] = dict(by=group, reason=part["blocks_set"])
    elif (set_doc.get("blocked") or {}).get("by") == group:
        set_doc["blocked"] = None
    d.mkdir(parents=True, exist_ok=True)
    for name in ("refs.json", "transitions.json", "set.json"):
        _write_if_changed(d / name, docs[name], None if old is None else old[name], name, group)
    partial_path(d, group).unlink(missing_ok=True)
    _fail_at("deleted", group)
    for line in part.get("log", []):
        _log_time(d, set_name, body, line["what"], line["seconds"], line["text"])
        log(f"{set_name} {body} {line['what']}: {line['text']}"
            + (f"; {line['seconds']:.0f} s" if line["seconds"] else ""))
    return res


def _log_time(d: Path, set_name: str, body: str, what: str, seconds: float, text: str) -> None:
    path = d / LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{s2io.now_utc()} {set_name} {body} {what} seconds={seconds:.1f} {text}\n")


def _partial_ok(d: Path, part: dict, set_name: str, body: str, names: set) -> bool:
    """Whether a partial found at the start of a build is a finished item of this set as it is defined now: every
    file it made is in place with its sha256, and every entry holds the definition made now."""
    if part.get("set") != set_name or part.get("body") != body:
        return False
    its = {i.name: i for i in items(set_name, body)}
    made = {**part.get("refs", {}), **part.get("transitions", {})}
    if not made or part.get("item") not in names:
        return False
    for name, e in made.items():
        it = its.get(name)
        if it is not None and e.get("item" if it.kind != "walk" else "definition") != it.definition:
            return False
        if it is None and e.get("group") not in names:
            return False
    wrong = {m.get("file") for m in part.get("mismatch", [])}
    for rel, sha in part.get("files", {}).items():
        if rel in wrong:
            continue
        p = d / rel
        if not p.is_file() or s2io.sha256_file(p) != sha:
            return False
    return True


def _require_models(body: str) -> None:
    """The body's round-two model files must be here: the Stage 0 clearance gates read them through
    ``plant.model_files``, which would otherwise rebuild them into ``stage1/models/<body>/``."""
    xml = stage1.case_dir(body) / "models" / f"{stage1.candidate_id(body)}.xml"
    if not (xml.is_file() and xml.with_suffix(".json").is_file()):
        raise SetError(f"{body}: its round-two model files are not here ({s2io.portable(str(xml))} and its ledger "
                       "JSON); restore them first (python tools/restore_inputs.py restore, or copy-from DIR)")


def build(set_name: str, body: str, root=setio.ROOT, only=None, procs: int = 2, log=print) -> dict:
    """Build what the set lacks (module docstring, section 2), resumably, under the set's lock.  ``only``: item or
    group names (their walks must be there).  ``procs`` spawned workers at a time, each item in a fresh process (0:
    the items run in this process, one after another, for tests).  Returns ``{dir, ran, kept, recovered, restored,
    skipped, mismatch, blocked, not_available, seconds}``; ``SetError`` after the run if a frozen entry could not be
    reproduced; ``GateError`` if a built-in gate stopped it."""
    s2io.require_one_thread()
    its = items(set_name, body)
    by_group = groups(its)
    known = set(by_group) | {i.name for i in its}
    for n in only or ():
        if n not in known:
            raise SetError(f"{set_name} {body}: no item or group {n!r} ({', '.join(by_group)})")
    wanted = None if not only else {next(i.group for i in its if n in (i.name, i.group)) for n in only}
    _require_models(body)
    d = setio.set_dir(set_name, body, root)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = dict(dir=str(d), ran=[], kept=[], recovered=[], restored=[], skipped=[], mismatch=[], blocked=None,
               not_available={}, seconds=None)
    with s2io.Lock(d / ".lock", log=log):
        for f in s2io.sweep_tmp(d):
            log(f"removed a stale temporary file: {f}")
        pdir = d / PARTIAL
        for p in sorted(pdir.glob(f"{SCRATCH_PREFIX}*")) if pdir.is_dir() else []:
            shutil.rmtree(p, ignore_errors=True)
            log(f"removed a stale scratch directory: {PARTIAL}/{p.name}")

        def take(part: dict, how: str) -> None:
            group = part["item"]
            m = merge(d, part, log=log)
            res["mismatch"] += part.get("mismatch", [])
            res[how].append(group)
            if m["frozen"] and not part.get("mismatch"):
                res["restored"] += m["frozen"]
            if part.get("blocks_set"):
                res["blocked"] = dict(by=group, reason=part["blocks_set"])

        # finished items a killed run left in partial/
        for p in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
            if p.name.startswith("screen__"):  # the set screen's own partials
                continue
            try:
                part = s2io.read_json(p)
                ok = p == partial_path(d, part["item"]) and _partial_ok(d, part, set_name, body, set(by_group))
            except (ValueError, KeyError, TypeError):
                ok = False
            if ok:
                take(part, "recovered")
            else:
                p.unlink(missing_ok=True)
                log(f"removed a partial item that is not of the set as it is now: {p.name}")

        def run(phase_groups: list) -> None:
            if not phase_groups or res["blocked"]:
                return
            jobs = [(set_name, body, g, str(root)) for g in phase_groups]
            log(f"{set_name} {body}: building {len(jobs)} item(s) with {min(max(1, procs), len(jobs))} process(es): "
                + ", ".join(phase_groups))
            if procs < 1:
                for j in jobs:
                    if res["blocked"]:
                        res["skipped"].append(j[2])
                        continue
                    take(build_item(*j[:3], root=j[3]), "ran")
                return
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=min(procs, len(jobs)), mp_context=mp.get_context("spawn"),
                                     max_tasks_per_child=1) as pool:
                futures = {pool.submit(_worker, j): j[2] for j in jobs}
                try:
                    for fut in as_completed(futures):
                        if fut.cancelled():
                            continue
                        group = fut.result()
                        take(s2io.read_json(partial_path(d, group)), "ran")
                        if res["blocked"]:
                            for other in futures:
                                if other.cancel():
                                    res["skipped"].append(futures[other])
                except BaseException:
                    for other in futures:
                        other.cancel()
                    raise

        for phase in (True, False):  # the walks, then the moves
            docs = load_docs(d)
            rows = {} if docs is None else (setio.frozen(docs["set.json"]) or {}).get("entries", {})
            todo = []
            for group, members in by_group.items():
                if (members[0].kind == "walk") != phase or (wanted is not None and group not in wanted):
                    continue
                if all(current(d, docs, i, rows) for i in members):
                    if group not in res["recovered"]:
                        res["kept"].append(group)
                    continue
                todo.append(group)
            if not phase:  # the start and the change into the bar's walk first: a failure of either stops the set
                first = [setio.START, setio.change_name(setio.NOMINAL, setio.walk_name(setio.BAR_SPEED))]
                todo.sort(key=lambda g: (first.index(g) if g in first else len(first)))
            run(todo)
        docs = load_docs(d)
        if docs is not None:
            res["not_available"] = {n: v.get("reason") for n, v in docs["set.json"].get("not_available", {}).items()}
            if res["blocked"] is None:
                res["blocked"] = docs["set.json"].get("blocked")
    res["seconds"] = time.perf_counter() - t0
    if res["mismatch"]:
        raise SetError(f"{d}: a frozen entry was not reproduced; nothing of it was written: " + "; ".join(
            f"{m['name']}: {m['why']} (frozen sha256 {m['frozen']}, built {m['built']})" for m in res["mismatch"]))
    return res


# ------------------------------------------------------------------------------------------------
# 5. verify
# ------------------------------------------------------------------------------------------------
def pr_function():
    """The pull request's ``_tuned_f9_spec``, imported from its script by path (None where the script is not
    here)."""
    if not PR_SCRIPT.is_file():
        return None
    spec = importlib.util.spec_from_file_location("jetson_plem_study_for_sets", PR_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    saved = {v: os.environ.get(v) for v in (*s2io.THREAD_VARS, "MUJOCO_GL")}
    try:
        spec.loader.exec_module(mod)
    finally:  # the script sets defaults for these at import
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod._tuned_f9_spec


def pr_differences(body: str, H: float) -> list[str] | None:
    """Where ``tuned_f9_spec`` differs from the pull request's function at a rung of the body (field by field), or
    None where the script is not here."""
    fn = pr_function()
    if fn is None:
        return None
    out = []
    for v in (nominal_speed(H), *setio.SPEEDS):
        a, b = asdict(tuned_f9_spec(body, H, v)), asdict(fn(body, H, v))
        out += [f"{body} at {v:g} m/s, {k}: {a.get(k)!r} here, {b.get(k)!r} in the pull request's function"
                for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return out


def source_hash(built_by: dict) -> str:
    """One hash of what built an entry: its Stage 2 modules' and the panels' source hashes."""
    return s2io.sha256_json(dict(stage2_modules=built_by.get("stage2_modules"),
                                 panel_sources_sha256=built_by.get("panel_sources_sha256")))


def modules_now() -> dict:
    return {s2io.portable(str(s2io.module_path(m))): s2io.sha256_file(s2io.module_path(m)) for m in MODULES}


def _check_name(name: str, kind: str, e: dict, item: Item | None, bad: list) -> None:
    """A name against its entry: it parses back (``setio.parse_name``) to the values its definition holds."""
    if kind != "walk" and (name == "stop" or name.startswith("stop_in_place_")):
        bad.append(f"{name}: a transition may not carry a name by which Stage 1 registers its own stop command")
        return
    try:
        p = setio.parse_name(name)
    except ValueError as exc:
        bad.append(f"{name}: not a name setio makes ({exc})")
        return
    if kind == "walk":
        if p["kind"] != "walk":
            bad.append(f"{name}: a walk's entry under a {p['kind']}'s name")
            return
        v, got = p["speed_m_s"], e.get("speed_m_s")
        if v is not None and got is not None and abs(float(got) - v) > 1e-9 * v:
            bad.append(f"{name}: its entry's speed is {got!r} m/s, its name holds {v:g}")
        return
    d = e.get("definition") or (item.definition if item is not None else {})
    builder = d.get("builder")
    want = {"start": "start", "change": "change", "turn": "turn"}.get(p["kind"])
    if p["kind"] == "stop":
        want = {2: "stop_two_step", 1: "stop_one_step"}[p["steps"]]
    if p["kind"] == "walk" or builder != want:
        bad.append(f"{name}: its name reads as a {p['kind']}, its definition's builder is {builder!r}")
        return
    a, b = (d.get("specs") or {}).get("a") or {}, (d.get("specs") or {}).get("b") or {}
    if p["kind"] == "change" and (a.get("name"), b.get("name")) != (p["a"], p["b"]):
        bad.append(f"{name}: its definition joins {a.get('name')!r} to {b.get('name')!r}")
    if p["kind"] == "stop" and a.get("name") != p["walk"]:
        bad.append(f"{name}: its definition stops from {a.get('name')!r}")
    if p["kind"] == "turn":
        args = d.get("args") or {}
        if args.get("rate") != p["sign"] * p["rate_rad_s"] or args.get("angle") != p["angle_rad"]:
            bad.append(f"{name}: its definition turns {args.get('angle')!r} rad at {args.get('rate')!r} rad/s, its "
                       f"name {p['angle_rad']!r} rad at {p['sign'] * p['rate_rad_s']!r} rad/s")


def verify(set_name: str, body: str, root=setio.ROOT) -> dict:
    """The set as it is against its rules (module docstring, section 3; the round-A design's 4.3): ``{ok, failures,
    notes, set, body, frozen, available, not_available}``.  No array is rebuilt.  A reference the set records as not
    available (a rung over a limit, a turn rate without a reference, a move outside its gates) is a result, listed
    in ``notes`` with its reason; a difference between the set and its records or rules is a failure."""
    d = setio.set_dir(set_name, body, root)
    bad: list = []
    notes: list = []
    out = dict(ok=False, failures=bad, notes=notes, set=set_name, body=body, frozen=None, available=[],
               not_available={})
    try:
        docs = setio.load(d)
    except Exception as exc:  # tune.InputError: a record is missing
        bad.append(f"the set cannot be read: {exc}")
        return out
    refs_doc, tr_doc, set_doc, screen = (docs[k] for k in ("refs.json", "transitions.json", "set.json", "screen.json"))
    its = {i.name: i for i in items(set_name, body)}
    group_names = {i.group for i in its.values()}
    H, model = body_facts(body)
    specs = walk_specs(set_name, body, H)
    st = _set(set_name)
    fz = setio.frozen(set_doc)
    rows = {} if fz is None else fz["entries"]
    out["frozen"] = None if fz is None else dict(index=fz["index"], manifest_sha256=fz["manifest_sha256"],
                                                 entries=len(rows))
    for doc, name in ((refs_doc, "refs.json"), (tr_doc, "transitions.json"), (set_doc, "set.json")):
        if doc.get("set") != set_name or doc.get("body") != body:
            bad.append(f"{name} is of {doc.get('set')!r} {doc.get('body')!r}")
    entries = {n: ("walk", e) for n, e in refs_doc["references"].items()}
    for n, e in tr_doc["transitions"].items():
        if n in entries:
            bad.append(f"{n}: names a walk and a move")
        entries[n] = ("move", e)
    blocks = (screen or {}).get("blocks", {})
    if screen is None:
        bad.append("screen.json is missing: the set is not screened (python -m designlab.stage2.setscreen run)")
    now_modules = modules_now()
    sources, dirty_entries = {}, []
    for name in its:
        if name not in entries:
            bad.append(f"{name}: not built (python -m designlab.stage2.sets build)")
    for name, (kind, e) in entries.items():
        item = its.get(name)
        if item is None and e.get("group") not in group_names:
            notes.append(f"{name}: an entry of no item defined now")
        _check_name(name, kind, e, item, bad)
        by = e.get("built_by")
        if by is not None:
            if by.get("blas_threads") != 1 or by.get("accuracy") != s2io.ACCURACY:
                bad.append(f"{name}: built with blas_threads {by.get('blas_threads')!r} at accuracy "
                           f"{by.get('accuracy')!r} (1 and {s2io.ACCURACY!r} are required)")
        if item is not None and (e.get("item" if kind != "walk" else "definition") != item.definition
                                 or not _same_inputs(docs, e)):
            bad.append(f"{name}: its entry is not of the definition made now, or was built from another file of its "
                       "walk than the set holds now (python -m designlab.stage2.sets build)")
        if e.get("available") is False:  # ---- an item with no reference
            out["not_available"][name] = e.get("reason")
            if not e.get("reason") or "numbers" not in e:
                bad.append(f"{name}: recorded as not available without a reason and its numbers")
            if e.get("file"):
                bad.append(f"{name}: recorded as not available and names a file")
            if (set_doc.get("not_available") or {}).get(name, {}).get("reason") != e.get("reason"):
                bad.append(f"{name}: set.json not_available does not hold its reason")
            if name in rows:
                bad.append(f"{name}: the latest freeze lists it and its entry is not available")
            notes.append(f"{name}: not available: {e.get('reason')}")
            continue
        out["available"].append(name)
        if by is None:
            bad.append(f"{name}: no built_by")
        else:
            sources.setdefault(source_hash(by), []).append(name)
            if by.get("dirty"):
                dirty_entries.append(name)
            if by.get("stage2_modules") != now_modules:
                changed = sorted({Path(k).name for k in set(now_modules) | set(by.get("stage2_modules") or {})
                                  if now_modules.get(k) != (by.get("stage2_modules") or {}).get(k)})
                notes.append(f"{name}: built by other code, arrays identical ({', '.join(changed)} moved since)")
        # ---- item 1: files against entries and the latest manifest
        files = [("file", "sha256")] + ([("look_file", "look_sha256")] if kind == "walk" else [])
        here = True
        for key_f, key_s in files:
            p = d / e[key_f]
            if not p.is_file():
                bad.append(f"{name}: {e[key_f]} is missing (python -m designlab.stage2.sets build puts it back)")
                here = False
            elif s2io.sha256_file(p) != e[key_s]:
                bad.append(f"{name}: {e[key_f]} has sha256 {s2io.sha256_file(p)}, its entry records {e[key_s]}")
                here = False
        if name in rows:
            row = rows[name]
            esha = setio.entry_sha256("walk" if kind == "walk" else "move", e)
            if row.get("sha256") != e["sha256"]:
                bad.append(f"{name}: its entry records sha256 {e['sha256']}, the freeze {row.get('sha256')}")
            if row.get("entry_sha256") != esha:
                bad.append(f"{name}: its entry's hashed subset is {esha}, the freeze holds {row.get('entry_sha256')}")
            if (row.get("kind") in setio.WALK_KINDS) != (kind == "walk"):
                bad.append(f"{name}: the freeze lists it as a {row.get('kind')}")
            if kind == "walk" and row.get("look_sha256") not in (None, e["look_sha256"]):
                bad.append(f"{name}: its look-format file is {e['look_sha256']}, the freeze holds {row['look_sha256']}")
        elif fz is not None:
            notes.append(f"{name}: not in the latest freeze (index {fz['index']})")
        if kind == "walk":
            _verify_walk(d, name, e, specs.get(name), set_doc, body, here, bad)
        else:
            _verify_move(d, name, e, body, bad, notes)
        # ---- item 5: the screen
        if screen is not None:
            b = blocks.get(name)
            if b is None:
                bad.append(f"{name}: no block in screen.json (python -m designlab.stage2.setscreen run)")
                continue
            want = e["look_sha256"] if kind == "walk" else e["sha256"]
            if (b.get("arrays") or {}).get("sha256") != want:
                bad.append(f"{name}: its screen block is of another file ({(b.get('arrays') or {}).get('sha256')})")
            if kind == "walk" and (b.get("reference") or {}).get("sha256") != e["sha256"]:
                bad.append(f"{name}: its screen block names another Stage 1-format file")
            v = b.get("verdict") or {}
            if v.get("blocks_freeze"):
                bad.append(f"{name}: the screen blocks its freeze and the set holds it as a reference: "
                           + "; ".join(str(o.get("limit")) + ("" if o.get("value") is None else f" {o['value']}")
                                       for o in v["blocks_freeze"]))
            over_a1 = any(float(((b.get("drives") or {}).get(g) or {}).get(k, 0.0)) > 1.0
                          for g in setscreen.A1_GROUPS for k in setscreen.KINDS)
            if kind != "walk":
                is_turn = (e.get("move") or {}).get("builder") == "turn"
                if bool(v.get("a1_over")) != bool(e.get("a1_over")):
                    bad.append(f"{name}: its entry's a1_over is {e.get('a1_over')!r}, the screen's {v.get('a1_over')!r}")
                if over_a1 and not (is_turn and e.get("a1_over") is True
                                    and "S2A-appendage" in ((e.get("move") or {}).get("rules") or [])):
                    bad.append(f"{name}: a neck or tail ratio is above 1 and it is not a turn with a1_over true and "
                               "the S2A-appendage rule")
                if is_turn:
                    use = (b.get("range") or {}).get("hip_yaw_range_use")
                    if use is None or use > moves.TWIST_FINAL:
                        bad.append(f"{name}: the screen reads a thigh-twist use of {use!r} (at most {moves.TWIST_FINAL:g})")
            elif over_a1:
                bad.append(f"{name}: a neck or tail ratio is above 1 on a walk the set holds as a reference")
            if kind == "walk" and e.get("ladder_twin"):
                speed, variant = e["ladder_twin"]
                rung = (((_read(LOOK_SCREEN) or {}).get("rungs", {}).get(body) or {}).get(speed) or {}).get(variant)
                if rung is None:
                    notes.append(f"{name}: the look ladder's screen holds no block of {speed} {variant}")
                else:
                    diff = setscreen.ladder_differences(b, rung)
                    if diff:
                        bad.append(f"{name}: its screen block differs from the look ladder's {speed} {variant}: "
                                   + "; ".join(diff[:4]) + (f"; ... {len(diff) - 4} more" if len(diff) > 4 else ""))
    # ---- screen blocks of what the set does not hold
    for name in blocks:
        if name not in out["available"]:
            bad.append(f"{name}: screen.json holds a block of a reference the set does not hold")
    # ---- item 4: one crouch; the pull request's function
    crouches = {float(s.crouch) for s in specs.values()}
    if set_doc.get("crouch_frac") is None or {float(set_doc["crouch_frac"])} != crouches:
        bad.append(f"set.json records crouch_frac {set_doc.get('crouch_frac')!r}, the set's specs {sorted(crouches)}")
    if st.pr_spec:
        diff = pr_differences(body, H)
        if diff is None:
            notes.append(f"the pull request's script is not here ({s2io.portable(str(PR_SCRIPT))}): tuned_f9_spec was "
                         "not compared with it")
        else:
            bad += [f"tuned_f9_spec is not the pull request's function: {x}" for x in diff]
    # ---- item 3: the stride rule's record of every turn rate
    for label in setio.TURN_RATES:
        names = [setio.turn_name(label, x) for x in setio.DIRECTIONS]
        if not any(n in entries for n in names):
            continue
        rec = (set_doc.get("stride_rules") or {}).get(label)
        need = ("outcome", "chosen", "references")
        if rec is None or any(k not in rec for k in need):
            bad.append(f"turn rate {label}: set.json stride_rules holds no complete record")
            continue
        if "tried" in rec:
            for k in ("grid", "limit_kinematic", "limit_final", "limit_other_joints_rad", "steps_down", "builds", "text"):
                if k not in rec:
                    bad.append(f"turn rate {label}: the stride rule's record lacks {k}")
            for r in rec["tried"]:
                if "error" in r:
                    continue
                for x in setio.DIRECTIONS:
                    if r.get(x) is None or not (r.get("tightest_other") or {}).get(x):
                        bad.append(f"turn rate {label}: the stride rule's row at f {r.get('f')!r} lacks the {x} "
                                   "turn's twist use or tightest other joint")
        held = {n: entries[n][1].get("available") is not False for n in names if n in entries}
        if rec["references"] != held:
            bad.append(f"turn rate {label}: the stride rule's record says {rec['references']}, the entries {held}")
        if not all(held.values()):
            notes.append(f"turn rate {label}: {rec.get('no_reference') or rec.get('reason')}")
        else:
            notes.append(f"turn rate {label}: stride factor {rec['chosen']:g}"
                         + (f", {rec['steps_down']} step(s) down" if rec.get("steps_down") else "")
                         + (", a1_over" if rec.get("a1_over") else ""))
    if set_doc.get("blocked"):
        bad.append(f"the set is stopped: {set_doc['blocked'].get('reason')} (it goes to the lead)")
    for name, rec in (set_doc.get("not_available") or {}).items():
        if name not in entries or entries[name][1].get("available") is not False:
            bad.append(f"{name}: set.json not_available lists it and its entry does not say so")
    for name in set_doc.get("look_entries") or {}:
        if name not in out["available"]:
            bad.append(f"{name}: set.json holds a look entry of a walk the set does not hold")
    # ---- item 6: provenance
    mixed = len(sources) > 1
    if fz is not None:
        last = set_doc["freezes"][-1]
        frozen_sources = {source_hash(entries[n][1]["built_by"]) for n in rows if n in entries
                          and entries[n][1].get("built_by")}
        frozen_dirty = [n for n in rows if n in dirty_entries]
        if len(frozen_sources) > 1 and not last.get("mixed_sources"):
            bad.append("the frozen entries were built from differing sources and the freeze does not say mixed_sources")
        if frozen_dirty and not last.get("dirty"):
            bad.append(f"frozen entries were built on a dirty tree and the freeze does not say dirty ({', '.join(frozen_dirty)})")
        if s2io.sha256_json(last.get("manifest")) != last.get("manifest_sha256"):
            bad.append("the latest freeze's manifest_sha256 is not its manifest's")
        for n in rows:
            if n not in entries:
                bad.append(f"{n}: the latest freeze lists it and the set has no such entry")
    else:
        if mixed:
            notes.append(f"the entries were built from {len(sources)} differing source states: a freeze refuses them "
                         "without --allow-mixed")
        if dirty_entries:
            notes.append(f"{len(dirty_entries)} entries were built on a dirty tree: a freeze refuses them without "
                         "--allow-dirty")
    out["ok"] = not bad
    return out


def _verify_walk(d: Path, name: str, e: dict, spec, set_doc: dict, body: str, here: bool, bad: list) -> None:
    """Item 4 of the round-A design's 4.3 for one walk: the two formats hold one motion; the entry's spec is the
    set's; its crouch is the set's; the twins' hashes."""
    if spec is not None and e.get("spec") != look._plain(asdict(spec)):
        bad.append(f"{name}: the spec in its entry is not the set's at this speed")
    if e.get("crouch_frac") is None or float(e["crouch_frac"]) != float(set_doc.get("crouch_frac", math.nan)):
        bad.append(f"{name}: its entry's crouch_frac is {e.get('crouch_frac')!r}, set.json's {set_doc.get('crouch_frac')!r}")
    le = (set_doc.get("look_entries") or {}).get(name)
    if le is None or le.get("sha256") != e.get("look_sha256") or le.get("file") != e.get("look_file"):
        bad.append(f"{name}: set.json holds no look entry of its look-format file")
    if here:
        with np.load(d / e["file"], allow_pickle=False) as a, np.load(d / e["look_file"], allow_pickle=False) as b:
            for key in ("q", "qd", "qdd"):
                if not np.array_equal(a[key], b[key]):
                    bad.append(f"{name}: {key} of the Stage 1-format file is not the look-format file's")
    if spec is not None:
        twin, ladder = stage1_walk_twin(body, spec), ladder_twin(body, spec)
        if (None if twin is None else twin["sha256"]) != e.get("stage1_sha256"):
            bad.append(f"{name}: Stage 1's file of the same spec is {None if twin is None else twin['sha256']}, its "
                       f"entry records {e.get('stage1_sha256')}")
        elif twin is not None and (twin["sha256"] != e["sha256"] or e.get("equals_stage1") is not True):
            bad.append(f"{name}: it does not hash to Stage 1's {twin['name']} ({e['sha256']} against {twin['sha256']})")
        if (None if ladder is None else ladder["sha256"]) != e.get("ladder_sha256"):
            bad.append(f"{name}: the look ladder's rung of the same spec is "
                       f"{None if ladder is None else ladder['sha256']}, its entry records {e.get('ladder_sha256')}")
        elif ladder is not None and (ladder["sha256"] != e["look_sha256"] or e.get("equals_ladder") is not True):
            bad.append(f"{name}: its look-format file does not hash to the look ladder's {ladder['speed']} "
                       f"{ladder['variant']} ({e['look_sha256']} against {ladder['sha256']})")


def _verify_move(d: Path, name: str, e: dict, body: str, bad: list, notes: list) -> None:
    """Items 2 and 3 of the round-A design's 4.3 for one move the set holds as a reference."""
    if e.get("pass_") is not True:
        failed = sorted(k for k, x in (e.get("gates") or {}).items() if not x)
        bad.append(f"{name}: held as a reference with pass_ {e.get('pass_')!r} (failing: {', '.join(failed) or '-'})")
    if (e.get("stage0") or {}).get("blocks_freeze"):
        bad.append(f"{name}: held as a reference although a Stage 0 gate blocked it where it was built")
    if e.get("a1_over") not in (True, False):
        bad.append(f"{name}: its entry's a1_over is {e.get('a1_over')!r}")
    move, ref = e.get("move") or {}, e.get("reference") or {}
    if e.get("stage1_sha256") is not None and (e["stage1_sha256"] != e["sha256"] or e.get("equals_stage1") is not True):
        bad.append(f"{name}: it does not hash to Stage 1's {e.get('stage1_twin')} ({e['sha256']} against "
                   f"{e['stage1_sha256']})")
    if e.get("fallback"):
        notes.append(f"{name}: built by fallback step {e['fallback'].get('step')} ({RULES['S2S-fallback'][:60]}...)")
    if move.get("builder") == "stop_two_step":
        fwd = ref.get("forward_stopped_strides")
        if ref.get("touchdowns_after_command") != moves.STOP_TOUCHDOWNS:
            bad.append(f"{name}: {ref.get('touchdowns_after_command')!r} touchdowns after the command, not "
                       f"{moves.STOP_TOUCHDOWNS}")
        if fwd is None or fwd > transitions.STOP_WITHIN_STRIDES:
            bad.append(f"{name}: stopped {fwd!r} strides after the command (at most {transitions.STOP_WITHIN_STRIDES:g})")
        lp = ref.get("accel_forward_lp1hz_peak_m_s2")
        if lp is None or lp > moves.STOP_ACCEL_LP_LIMIT_M_S2:
            bad.append(f"{name}: 1 Hz forward acceleration {lp!r} m/s^2 (at most {moves.STOP_ACCEL_LP_LIMIT_M_S2:g})")
    if move.get("builder") == "turn":
        t = e.get("turn") or {}
        total = t.get("heading_total_rad")
        if total is None or abs(abs(total) - setio.TURN_ANGLE_RAD) > moves.TURN_CLOSURE_RAD:
            bad.append(f"{name}: total heading {total!r} rad (within {moves.TURN_CLOSURE_RAD:g} of "
                       f"{setio.TURN_ANGLE_RAD!r})")
        for key in ("heading_at_bar_deg", "heading_at_slowest_clock_deg"):
            if ref.get(key) is None or ref[key] < moves.TURN_HEADING_MIN_DEG:
                bad.append(f"{name}: {key} {ref.get(key)!r} (at least {moves.TURN_HEADING_MIN_DEG:g})")
        if e.get("hip_yaw_range_use") is None or e["hip_yaw_range_use"] > moves.TWIST_FINAL:
            bad.append(f"{name}: thigh-twist use {e.get('hip_yaw_range_use')!r} (at most {moves.TWIST_FINAL:g})")
        other = e.get("tightest_other")
        if not other or other["margin_rad"] < moves.OTHER_JOINT_MARGIN_RAD:
            bad.append(f"{name}: tightest other joint {other!r} (at least {moves.OTHER_JOINT_MARGIN_RAD:g} rad)")
        if "S2A-appendage" not in (move.get("rules") or []):
            bad.append(f"{name}: S2A-appendage is not among its rules")
        if not isinstance(t.get("stride_rule"), dict) or t["stride_rule"].get("outcome") != "frozen":
            bad.append(f"{name}: its entry holds no stride rule record with the outcome 'frozen'")
        if t.get("ramp_s") != setio.RAMP_S:
            bad.append(f"{name}: its ramps last {t.get('ramp_s')!r} s, not {setio.RAMP_S:g}")
        if e.get("a1_over"):
            notes.append(f"{name}: a1_over (a neck or tail limit over at Stage 0): read under the third reading only")


# ------------------------------------------------------------------------------------------------
# 6. freeze, status
# ------------------------------------------------------------------------------------------------
def legacy_state() -> dict:
    """``stage2/sets/legacy_check.json`` against today's ``moves.py``: ``{ok, why, sha256}``."""
    doc = _read(LEGACY_CHECK)
    if doc is None:
        return dict(ok=False, why=f"{s2io.portable(str(LEGACY_CHECK))} is missing", sha256=None)
    why = []
    if doc.get("moves_sha256") != s2io.sha256_file(s2io.module_path("moves.py")):
        why.append("it was made by another moves.py")
    for b in look.BODIES:
        rec = (doc.get("bodies") or {}).get(b)
        if rec is None or not rec.get("complete") or not all(x.get("equal") for x in rec.get("names", {}).values()):
            why.append(f"{b} is not complete with every transition equal")
    return dict(ok=not why, why="; ".join(why) or None, sha256=s2io.sha256_file(LEGACY_CHECK))


def manifest_of(docs: dict) -> dict:
    """name -> ``{kind, sha256, entry_sha256}`` (a walk: ``look_sha256`` beside them) of every available entry."""
    out = {}
    for name, e in docs["refs.json"]["references"].items():
        if e.get("available") is not False:
            out[name] = dict(kind="walk", sha256=e["sha256"], entry_sha256=setio.entry_sha256("walk", e),
                             look_sha256=e["look_sha256"])
    for name, e in docs["transitions.json"]["transitions"].items():
        if e.get("available") is not False:
            kind = (e.get("item") or {}).get("kind") or (e.get("move") or {}).get("builder") or "move"
            out[name] = dict(kind=kind, sha256=e["sha256"], entry_sha256=setio.entry_sha256("move", e))
    return out


def freeze(set_name: str, body: str, root=setio.ROOT, allow_mixed: bool = False, allow_dirty: bool = False,
           change_after_result: str | None = None, log=print) -> dict:
    """Append a freeze to ``set.json`` (``RULES["S2S-freeze"]``) and return it; ``SetError`` when the set cannot be
    frozen (the reasons, all of them).  A freeze that would add nothing appends nothing and returns the latest."""
    d = setio.set_dir(set_name, body, root)
    with s2io.Lock(d / ".lock", log=log):
        v = verify(set_name, body, root)
        why = [f"sets verify: {x}" for x in v["failures"]]
        docs = setio.load(d)
        screen = docs["screen.json"]
        if screen is not None and screen.get("refused"):
            why.append(f"the screen refused {', '.join(screen['refused'])}: screen.json is not complete")
        leg = legacy_state()
        if not leg["ok"]:
            why.append(f"the legacy check is not current: {leg['why']} (python -m designlab.stage2.moves legacy --body "
                       "<pick> --record stage2/sets/legacy_check.json, both picks)")
        manifest = manifest_of(docs)
        set_doc = docs["set.json"]
        freezes = list(set_doc.get("freezes") or [])
        prev = freezes[-1]["manifest"] if freezes else {}
        for name, row in prev.items():
            now = manifest.get(name)
            if now is None:
                why.append(f"{name}: an earlier freeze holds it and the set no longer does (a frozen entry is never dropped)")
            elif (now["sha256"], now["entry_sha256"]) != (row["sha256"], row["entry_sha256"]):
                why.append(f"{name}: an earlier freeze holds sha256 {row['sha256']} (entry {row['entry_sha256'][:12]}...), "
                           f"the set now {now['sha256']} (entry {now['entry_sha256'][:12]}...): a changed reference "
                           "takes a new name or a new set version")
        entries = {**docs["refs.json"]["references"], **docs["transitions.json"]["transitions"]}
        mixed = len({source_hash(entries[n]["built_by"]) for n in manifest if entries[n].get("built_by")}) > 1
        git = s2io.git_state(own_outputs=[str(d)])
        dirty_entries = sorted(n for n in manifest if (entries[n].get("built_by") or {}).get("dirty"))
        dirty = bool(git["dirty"] or dirty_entries)
        if mixed and not allow_mixed:
            why.append("the entries were built from differing source states (--allow-mixed records it and freezes)")
        if dirty and not allow_dirty:
            why.append("the tree is dirty" + (f" ({'; '.join((git['dirty_paths'] or [])[:5])})" if git["dirty"] else "")
                       + (f"; built on a dirty tree: {', '.join(dirty_entries[:6])}" if dirty_entries else "")
                       + " (--allow-dirty records it and freezes)")
        if why:
            raise SetError(f"{d} is not frozen: " + "; ".join(why))
        if freezes and manifest == prev:
            log(f"{set_name} {body}: nothing to add to freeze {freezes[-1]['index']}")
            return freezes[-1]
        rec = dict(index=len(freezes), at=s2io.now_utc(), commit=git["commit"], manifest=manifest,
                   manifest_sha256=s2io.sha256_json(manifest), legacy_check_sha256=leg["sha256"],
                   mixed_sources=bool(mixed), dirty=bool(dirty), added=sorted(set(manifest) - set(prev)))
        if change_after_result:
            rec["change_after_result"] = str(change_after_result)
        set_doc["freezes"] = freezes + [rec]
        s2io.write_json(d / "set.json", set_doc)
        log(f"{set_name} {body}: freeze {rec['index']}: {len(manifest)} entries, manifest {rec['manifest_sha256'][:12]}...")
        return rec


def status(set_name: str, body: str, root=setio.ROOT) -> dict:
    """What the set holds, read now: ``{dir, built, frozen, items {name: state}, missing, not_available, blocked}``;
    an item's state is ``built``, ``built (frozen)``, ``array missing``, ``stale``, ``missing`` or ``not available:
    <reason>``."""
    d = setio.set_dir(set_name, body, root)
    its = items(set_name, body)
    docs = load_docs(d) if d.is_dir() else None
    out = dict(dir=str(d), built=docs is not None, frozen=None, items={}, missing=[], not_available={}, blocked=None)
    if docs is None:
        out["missing"] = [i.name for i in its]
        out["items"] = {i.name: "missing" for i in its}
        return out
    fz = setio.frozen(docs["set.json"])
    rows = {} if fz is None else fz["entries"]
    out["frozen"] = None if fz is None else dict(index=fz["index"], manifest_sha256=fz["manifest_sha256"],
                                                 entries=len(rows))
    out["blocked"] = docs["set.json"].get("blocked")
    for i in its:
        e = _entry_of(docs, i.kind, i.name)
        if e is None:
            state = "missing"
            out["missing"].append(i.name)
        elif e.get("available") is False:
            state = f"not available: {e.get('reason')}"
            out["not_available"][i.name] = e.get("reason")
        elif not _files_ok(d, e, i.kind == "walk"):
            state = "array missing"
        elif i.name in rows:
            state = "built (frozen)"
        else:
            state = "built" if current(d, docs, i, rows) else "stale"
        out["items"][i.name] = state
    for name, e in docs["transitions.json"]["transitions"].items():
        if name not in out["items"]:
            out["items"][name] = ("built (a fallback's change)" if e.get("available") is not False
                                  else f"not available: {e.get('reason')}")
    return out


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    s2io.require_one_thread()
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.sets", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("build", "verify", "freeze", "status"):
        p = sub.add_parser(cmd)
        p.add_argument("--set", required=True, choices=list(SETS))
        p.add_argument("--body", required=True, choices=list(stage1.BODIES))
        p.add_argument("--root", default=str(setio.ROOT), help="the directory that holds the sets (default: stage2/sets)")
        if cmd == "build":
            p.add_argument("--only", action="extend", nargs="+", metavar="NAME", help="items or groups to build")
            p.add_argument("--procs", type=int, default=2)
        if cmd == "freeze":
            p.add_argument("--allow-mixed", action="store_true")
            p.add_argument("--allow-dirty", action="store_true")
            p.add_argument("--change-after-result", metavar="REASON")
    a = ap.parse_args(argv)

    def say(line: str) -> None:
        print(line, flush=True)

    try:
        if a.cmd == "build":
            r = build(a.set, a.body, a.root, a.only, a.procs, log=say)
            say(f"{a.set} {a.body}: built {len(r['ran'])}, kept {len(r['kept'])}, recovered {len(r['recovered'])}"
                + (f", arrays of frozen entries put back: {', '.join(r['restored'])}" if r["restored"] else "")
                + f"; {r['seconds']:.0f} s")
            for name, why in r["not_available"].items():
                say(f"  not available: {name}: {why}")
            if r["blocked"]:
                say(f"  the set is stopped: {r['blocked']['reason']}"
                    + (f"; not built: {', '.join(r['skipped'])}" if r["skipped"] else ""))
                return 1
            return 0
        if a.cmd == "verify":
            r = verify(a.set, a.body, a.root)
            for line in r["notes"]:
                say(f"  note: {line}")
            for line in r["failures"]:
                say(f"  FAILS: {line}")
            say(f"{a.set} {a.body}: verify {'ok' if r['ok'] else 'FAILED'}: {len(r['available'])} references, "
                f"{len(r['not_available'])} not available, {len(r['failures'])} failures"
                + ("" if r["frozen"] is None else f"; freeze {r['frozen']['index']}"))
            return 0 if r["ok"] else 1
        if a.cmd == "freeze":
            r = freeze(a.set, a.body, a.root, a.allow_mixed, a.allow_dirty, a.change_after_result, log=say)
            say(json.dumps({k: v for k, v in r.items() if k != "manifest"}, indent=1))
            return 0
        r = status(a.set, a.body, a.root)
        for name, state in r["items"].items():
            say(f"  {name}: {state}")
        say(f"{a.set} {a.body}: " + ("not built" if not r["built"] else
                                     ("not frozen" if r["frozen"] is None else f"freeze {r['frozen']['index']}"))
            + (f"; missing {len(r['missing'])}" if r["missing"] else "")
            + (f"; stopped: {r['blocked']['reason']}" if r["blocked"] else ""))
        return 0
    except SetError as exc:
        say(f"{type(exc).__name__}: {exc}")
        return 2 if isinstance(exc, GateError) else 1
    except s2io.S2Error as exc:
        say(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    from designlab.stage2 import sets as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
