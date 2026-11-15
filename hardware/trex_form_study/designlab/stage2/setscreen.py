"""Stage 2 round A: the Stage 0 screen of every reference of a set (``STAGE2_75_PLAN.md`` A12; ``STAGE2_PLAN.md``
4 A.3: every reference is priced before anything is simulated on it).

Simulation only: prescribed references that the joints follow exactly, priced on the round-two model; nothing is
adopted.

A reference set is a directory ``stage2/sets/<set>/<body>/`` in Stage 1's run-directory layout (``setio``): periodic
walks in ``refs/<name>.npz`` with a twin in the look ladder's format in ``look/<name>.npz``, moves (starts, speed
changes, stops, finite turns) in ``transitions/<name>.npz``.  The look ladder has a Stage 0 screen
(``designlab/stage2/screen.py``) for periodic walks only, written under ``stage2/look/``.  This module screens a
set, walks and moves, into the set's own ``screen.json`` and ``SCREEN.md``.  It writes nothing under
``stage2/look/`` and changes no entry of ``refs.json``, ``transitions.json`` or ``set.json``.  It is in no record
key.

1. A walk (``screen_walk``)
---------------------------
``screen.screen_rung`` on the walk's look-format file and its entry of ``look.write``'s shape (``set.json``
``look_entries[name]``; its ``file`` is relative to the set directory), inside ``s2io.r2()``.  The block is the look
ladder's rung block (``speed`` holds the walk's name, ``variant`` the set's) with these changes:

* ``seconds`` is dropped (wall times go to the log);
* ``name`` and ``kind`` (``"walk"``) are added;
* ``reference`` is added: ``{file, sha256}`` of the Stage 1-format file the bar plays (``refs/<name>.npz``, the
  walk's entry of ``refs.json``), checked against the file on disk; None where no entry was given;
* ``range`` is added from the entry's ``price.clearance.joint_range_margins`` (``screen_rung`` leaves that key
  out): per joint ``{walk_min, walk_max, usable, margin_min, margin_max}``, the tightest gated end, ``rule_rad``
  (0.10), ``ok``, ``hip_yaw_range_use`` and ``usable_differs_from_plant``;
* ``verdict`` becomes ``{text, ladder_text, within_limits, over, near, a1_over, blocks_freeze}``: the shape every
  block of a set has, and ``ladder_text``, which is ``screen_rung``'s sentence (the look ladder's: "keeps its
  feature and loses that speed rung").  ``text`` is this screen's (the plan's A12: a walk over a gate is not
  frozen and its rung is dropped from the ladder with its numbers).

A walk's block names two files.  ``arrays.{file, sha256}`` is the look-format file (``look/<name>.npz``;
``refs.json``'s ``look_sha256``, the look entry's ``sha256``): every number of the block is read on it.
``reference.{file, sha256}`` is the Stage 1-format file (``refs.json``'s ``file`` and ``sha256``), which the bar
plays and a freeze lists.  That the two hold one motion (``q``, ``qd``, ``qdd`` equal) is checked where the set is
built and verified, not here.  ``ladder_differences`` compares a walk's block with a rung block of
``stage2/look/screen.json``: for the ladder's own V0 walks the two are equal on every key ``screen_rung`` makes
but ``speed``, ``variant``, ``seconds`` and ``arrays.file``.

2. A move (``screen_move``)
---------------------------
From the transition file's arrays (``q, qd, qdd, tau, omega, swing, t, phase, com, feet, heading, path``), checked
against the entry's sha256, on the move's own frames with no wrap:

* ``drives``: ``sizing._motor_space`` -> ``envelope._ratios`` on the file's ``tau`` and ``omega`` at the record's
  bus, as ``look.price`` reads a walk: per group 1.25 x peak / planning peak, 1.15 x speed / speed at the bus, and
  the sustained ratio, RMS / sustained.  A move's RMS is over time: each frame is weighted by its duration
  (``frame_weights``: half the time between its two neighbours, ``np.gradient(t)``), because the frames of a move
  are not uniform in time (a stride takes the same number of frames whatever its period).  ``rms`` and
  ``cont_ratio`` are that reading and are what the gates read; ``rms_frames`` and ``cont_ratio_frames`` are
  ``envelope._ratios``' own (every frame the same weight).  Where the frames are uniform in time (to 1e-9 of the
  interval: a walk, Stage 1's start) the two are the same number, not two computations.  ``arrays.frames_uniform``
  and ``arrays.frame_s`` say which case a file is.  ``governing``.
* ``passive``: ``screen.passive_loads`` (the neck and tail with the plant's joint damping and friction loss); its
  torque RMS values and sustained ratios (``rms_id_nm``, ``rms_passive_alone_nm``, ``rms_total_nm``,
  ``cont_ratio_id``, ``cont_ratio_total``, ``rms_total_over_id``) are over time as the drives', with
  ``rms_id_frames_nm``, ``rms_total_frames_nm``, ``cont_ratio_id_frames`` and ``cont_ratio_total_frames`` beside
  them; its speed RMS, moving fraction and mean powers are statistics of the frames.
* ``summary``: the entry's ``min_cop_margin_single_m``, ``max_friction_ratio``, ``ik_ok``, ``ik_fail_frames``.
* ``contact`` and ``friction``: the contact wrenches of every frame are solved again on the planned support
  (``contact_pass``: ``turning.contact_torques(..., "exact")`` in single support, where the wrench is unique; the
  entry's ``split`` in double support, as ``transitions.idyn``).  ``contact``: the single-support CoP margin with
  its frame and foot against the rule that applies (Stage 0's -10 mm; +10 mm for a causal move), the largest
  friction ratio, the double-support frames whose split did not solve (``split_failed_frames``).  ``friction``:
  per single-support frame the formulas of ``turning.inverse_dynamics`` (the stance pad's free moment about its
  CoP, ``mu_req``, ``mu_req_cop``, the utilisation at mu 0.8), per foot ``turning._foot_stats``; in double support
  ``turning.ds_min_mu``; ``grips`` at mu 0.9 and 0.54 (the look screen's ``straight_run`` keys).  ``recheck``
  compares the solve with the file's own ``tau``, ``cop_margin`` and ``friction`` and with the entry's two numbers.
* ``sway``: the model CoM's lateral excursion in the path frame, peak to peak per stride of the move, / H.
* ``accel``: the model CoM's acceleration (second differences on the file's frames, the ends left out): the
  horizontal peak, the part along the heading, the vertical range, the feet's vertical peak; and the part along
  the heading low-passed at 1 Hz (``stage1.metrics.lowpass``) on a uniform 500 Hz resample of the frames
  (``com_forward_lp1hz_peak``: the reading of ``readings.reference_readings``, and the one this screen gates).
  The filter sees the file alone: at a file's end that joins a walk its value is that of the unfiltered series
  there, not of a filter that had the walk before or after it.  ``com_forward_lp1hz_peak_at_end`` says that the
  peak lies within 1 s of an end of the file, and ``com_forward_lp1hz_peak_interior`` is the largest value at least
  1 s from both ends (None for a file under 2 s).  The builder's own reading of the move
  (``reference.accel_forward_lp1hz_peak_m_s2``: second differences on the frames' own times, the filter given the
  walk or the stand around the move as its history; the plan's A7) is another number and is recorded beside it
  (``com_forward_lp1hz_peak_entry``, ``com_forward_lp1hz_peak_minus_entry``).
* ``clearance``: ``look._clearance_gates`` (``clearance.screen``, gates r2, no interpolated pose: ``substeps`` 0)
  with the move standing in for the nominal walk and the support read from the planned ``swing``.  The swing
  runs of that screen are cyclic and its mid-swing reading is the middle half of each run, so a file is refused
  (``Refused``) when a foot is in swing on its last frame (the run would join the file's first frames, or be a cut
  swing) or when a swing on its first frame does not begin there (``swing_u`` of that frame is not 0, or the file
  has no ``swing_u``): the middle half of a cut swing is not the swing's.
* ``range``: per joint the margins against the usable range as that clearance screen reads them (its usable
  ranges are checked against ``plant.usable_ranges(body)``), with a walk's keys; ``hip_yaw_range_use`` (the worse
  leg's use of the usable thigh twist on the file's ``q``, ``turning.summarize``'s formula on
  ``plant.usable_ranges``; the entry's own value beside it).
* ``vs_walk``: each headline number against the block of the move's own walk (gait a; gait b for a start).
* ``verdict``: ``{text, within_limits, over, near, a1_over, blocks_freeze}``.

3. Which gate blocks a freeze (``RULES``; the plan's A12, nothing else blocks)
------------------------------------------------------------------------------
======================================================  ========  ====================  =====================
gate                                                    walk      start, change, stop   turn
======================================================  ========  ====================  =====================
a drive's cont, peak or speed ratio above 1, the        blocks    blocks                blocks
neck's and the tail's apart
a neck or tail drive's ratio above 1 (``a1_over``)      blocks    blocks                does not block
``ik_ok``; the single-support CoP rule                  blocks    blocks                blocks
a planned swing's mid-swing pad clearance under 0.04 H  blocks    blocks                blocks
interference, component overlap, tail clearance         blocks    blocks                blocks
joint range                                             0.10 rad  0.10 rad              twist use at most 0.87;
                                                                                        other joints 0.10 rad
1 Hz forward acceleration above 1.5 m/s^2               reported  blocks                reported
a gate's reading is missing                             blocks    blocks                blocks
======================================================  ========  ====================  =====================

"The neck's and the tail's apart": the appendage groups are the neck and the tail alone (``A1_GROUPS``); the
expressive head's two drives (neck yaw, jaw), on a body that has them, are gated like the legs'.  A gate that
is not read is not passed: no single-support CoP margin, no joint-range margins (or none for a joint other than
a turn's thigh twist, or no thigh-twist use), a contact split that did not solve (a walk's rolling-contact split;
a move's double-support split), a clearance gate that is not true, usable ranges that are not
``plant.usable_ranges(body)``'s: each is named in ``verdict.blocks_freeze`` ("not read: ...").

Grip need, sway, passive loads, ``recheck`` and a walk's ``reprice`` are reported.  ``a1_over`` is true when a
neck or tail group's cont, peak or speed ratio exceeds 1; a turn with ``a1_over`` is frozen and read under the
plan's third reading only.  ``drive_flags`` gives a builder the two flags of a move's arrays without the rest of
the screen.

4. ``screen.json`` and ``run``
------------------------------
``{set, body, schema, what, rules, limits, margins, grips, specs, blocks {name: block}, over, a1_over,
blocks_freeze {name: [..]}, not_available, refused, sessions, provenance}``; no wall time.  ``sets verify`` reads of
a block: ``arrays.sha256`` (a walk: against ``refs.json``'s ``look_sha256``; a move: against its entry's
``sha256``), a walk's ``reference.sha256`` (against ``refs.json``'s ``sha256``), ``verdict.blocks_freeze``,
``verdict.a1_over``, ``drives.<group>.{cont_ratio, peak_ratio, speed_ratio}``, ``range.hip_yaw_range_use``.

``run`` holds the set's lock.  A reference is screened when it is available and its block is missing or stale (a
walk: ``arrays.sha256`` is not the look entry's or ``reference.sha256`` not its ``refs.json`` entry's; a move:
``arrays.sha256`` is not the entry's, or its ``vs_walk`` was read against another file of its walk).  Walks
first, then moves; each job runs in a spawned worker that builds the body itself, writes
``partial/screen__<name>.json`` and returns; the parent merges the block into ``screen.json`` (atomic) and deletes
the partial in the same step.  A run on a complete screen starts no job and rewrites no file.  ``provenance`` is
``s2io.built_by`` of the last session that screened a block; ``sessions`` lists which session made which block.

Run from hardware/trex_form_study:
    python -m designlab.stage2.setscreen run --set V0 --body 75-M-H-10 [--procs 2] [--root DIR]
    python -m designlab.stage2.setscreen md  --set V0 --body 75-M-H-10 [--root DIR]     # SCREEN.md from screen.json
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # one BLAS thread per process, set before numpy loads
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .. import clearance, envelope, sizing, statics, turning  # noqa: E402
from ..stage1 import metrics, plant, tune  # noqa: E402
from . import look, s2io, screen, setio  # noqa: E402

LOG = logging.getLogger(__name__)
SIDES = statics.SIDES
G = turning.G
SCHEMA = 1
SCREEN_JSON = "screen.json"
SCREEN_MD = "SCREEN.md"
PARTIAL = "partial"
LOG_FILE = "logs/setscreen.log"
#: The Stage 2 modules that make a block (``s2io.built_by``).
MODULES = ("setscreen.py", "screen.py", "look.py", "setio.py", "s2io.py")

GRIPS = screen.GRIPS
NEAR_BAND = screen.NEAR_BAND
KINDS = screen.KINDS
#: The appendage groups (the plan's A12, "a neck or tail drive"): the drives reading 3 sets aside ("shown but for
#: A1").  The expressive head's two drives (``sizing.HEAD_X_GROUPS``) are gated like every other drive.
A1_GROUPS = ("neck", "tail")
#: Single-support CoP margin (m): Stage 0's rule (walks and non-causal moves) and a causal move's (twice
#: ``control.COP_INSET``: the tracker's CoP feedback works inside the pad inset).
COP_RULE_M = look.COP_RULE_M
COP_RULE_CAUSAL_M = 0.010
SWING_MIN_FRAC = clearance.SWING_MIN_FRAC
RANGE_RULE_RAD = clearance.RANGE_MARGIN_RAD
#: A turn's thigh twist: the largest use of the usable range on the built reference (the stride rule's final limit).
TWIST_USE_MAX = 0.87
#: Decision 10's limit on the 1 Hz low-passed forward acceleration (m/s^2) and the resample it is read on (Hz).
ACCEL_RULE_M_S2 = screen.ACCEL_SPEC
RESAMPLE_HZ = 500.0
#: A 1 Hz reading within this time of a file's end (s; the filter's corner period) is marked as an end value.
ACCEL_EDGE_S = 1.0
#: A file's frames are uniform in time when their durations differ by at most this fraction of their mean.
UNIFORM_TOL = 1e-9
#: A swing on a file's first frame begins there when its ``swing_u`` is at most this (a swing cut by one frame has at
#: least 1 / its number of frames; the builders' phase tolerance leaves under 1e-8).
SWING_START_TOL = 1e-6
#: The first words of a ``verdict`` item that says a gate's reading is missing.
NOT_READ = "not read: "
#: The kinds of a block; a start, change or stop is a straight move.
WALK, TURN = "walk", "turn"
STRAIGHT = ("start", "change", "stop")
KIND_OF = {"start": "start", "change": "change", "walk": "change", "stop": "stop", "stop_one_step": "stop",
           "stop_two_step": "stop", "turn": "turn"}
#: The clearance gates that block a freeze, beside the joint-range rule (read from ``range``).
CLEARANCE_GATES = ("swing_clearance", "interference_free_home", "interference_free_nominal_gait",
                   "component_overlap_free", "tail_clearance")

RULES = dict(
    source="STAGE2_75_PLAN.md A12 (the round-A design's section 3.7); nothing but the gates below blocks a freeze",
    constants=dict(drive_ratio_max=1.0, near_band=NEAR_BAND, cop_rule_m=COP_RULE_M,
                   cop_rule_causal_m=COP_RULE_CAUSAL_M, swing_min_frac_H=SWING_MIN_FRAC,
                   joint_range_rad=RANGE_RULE_RAD, twist_use_max=TWIST_USE_MAX, accel_lp1hz_m_s2=ACCEL_RULE_M_S2,
                   accel_resample_hz=RESAMPLE_HZ, accel_edge_s=ACCEL_EDGE_S, frames_uniform_tol=UNIFORM_TOL,
                   swing_start_tol=SWING_START_TOL, grips=list(GRIPS), a1_groups=list(A1_GROUPS)),
    gates=[
        dict(gate="a drive's cont, peak or speed ratio above 1, the neck's and the tail's apart (the legs' six "
                  "drives; on a body with the expressive head, its two as well)", walk="blocks", straight="blocks",
             turn="blocks"),
        dict(gate="a neck or tail drive's ratio above 1 (inverse dynamics, as Stage 0 reads it): a1_over",
             walk="blocks", straight="blocks",
             turn="does not block: frozen with a1_over and read under the plan's third reading only"),
        dict(gate="ik_ok; the single-support CoP margin (walks and non-causal moves: Stage 0's -10 mm; causal moves: "
                  "+10 mm)", walk="blocks", straight="blocks", turn="blocks"),
        dict(gate="each planned swing's mid-swing minimum of the pads under 0.04 H", walk="blocks", straight="blocks",
             turn="blocks"),
        dict(gate="interference at home or along the frames; component overlap; tail clearance", walk="blocks",
             straight="blocks", turn="blocks"),
        dict(gate="joint range", walk="0.10 rad on every gated joint", straight="0.10 rad on every gated joint",
             turn="thigh twist: use of the usable range at most 0.87; every other gated joint 0.10 rad"),
        dict(gate="forward CoM acceleration low-passed at 1 Hz above 1.5 m/s^2 (decision 10)", walk="reported",
             straight="blocks", turn="reported"),
        dict(gate="a gate's reading is missing (no single-support CoP margin; no joint-range margins, none for a "
                  "joint other than a turn's thigh twist, or no thigh-twist use; a contact split that did not "
                  "solve; a clearance gate that is not true; usable ranges that are not plant.usable_ranges(body)'s)",
             walk="blocks", straight="blocks", turn="blocks"),
        dict(gate="grip need at mu 0.9 and 0.54; sway against 0.12 H; passive neck and tail loads; the re-priced "
                  "walk and the re-solved contact against the record", walk="reported", straight="reported",
             turn="reported"),
    ],
    a1_over="true when a neck or tail group's cont, peak or speed ratio exceeds 1",
    not_read="a gate that is not read is not passed: the item in verdict.blocks_freeze begins 'not read: ' and names "
             "the reading",
    sustained="a move's sustained (cont) ratio is its torque's RMS over time / sustained torque: each frame is "
              "weighted by its duration (half the time between its two neighbours; the first and the last frame by "
              "their one interval). The value with every frame the same weight is kept beside it (rms_frames, "
              "cont_ratio_frames). On a walk, whose frames are uniform in time, the two are the same number: a file "
              "whose frame durations differ by at most 1e-9 of their mean is given the frame-weighted value as both",
    accel_end="the 1 Hz reading is the file's own, with no history before its first or after its last frame: a peak "
              "within 1 s of an end is marked (com_forward_lp1hz_peak_at_end) and the largest value at least 1 s "
              "from both ends is given beside it; the gate reads the peak wherever it lies",
    cyclic_swing="a move is refused when a foot is in swing on its last frame, or when a swing on its first frame "
                 "does not begin there (swing_u above 1e-6, or no swing_u): clearance.screen reads swings as cyclic "
                 "runs and takes the middle half of each run, which on a cut swing is not the swing's",
)

GROUP_LABEL = screen.GROUP_LABEL
#: The headline numbers of a move compared with its walk (label -> path in both blocks).
HEADLINES = {
    "governing ratio": ("governing", 2),
    **{f"{GROUP_LABEL[g]} worst": ("drives", g, "worst") for g in sizing.GROUPS},
    "knee speed": ("drives", "knee", "speed_ratio"),
    "hip pitch speed": ("drives", "hip_pitch", "speed_ratio"),
    "neck cont": ("drives", "neck", "cont_ratio"),
    "tail cont": ("drives", "tail", "cont_ratio"),
    "tail peak": ("drives", "tail", "peak_ratio"),
    "neck cont with passive": ("passive", "neck", "cont_ratio_total"),
    "neck peak with passive": ("passive", "neck", "peak_ratio_total"),
    "tail cont with passive": ("passive", "tail", "cont_ratio_total"),
    "tail peak with passive": ("passive", "tail", "peak_ratio_total"),
    "single-support CoP margin (mm)": ("contact", "single_support_cop_margin_min_mm"),
    "friction ratio": ("contact", "friction_ratio_max"),
    "friction need mu_req": ("friction", "single_support", "mu_req_max"),
    "stance-pad peak |Mz| (N m)": ("friction", "single_support", "mz_abs_max_nm"),
    "double-support min mu": ("friction", "double_support", "min_mu_max"),
    "sway CoM per stride (H)": ("sway", "com_pp_per_stride_max_H"),
    "CoM horizontal accel (m/s^2)": ("accel", "com_horizontal_peak"),
    "CoM forward accel at 1 Hz (m/s^2)": ("accel", "com_forward_lp1hz_peak"),
    "CoM vertical accel max (m/s^2)": ("accel", "com_vertical_max"),
    "CoM vertical accel min (m/s^2)": ("accel", "com_vertical_min"),
    "foot vertical accel (m/s^2)": ("accel", "foot_vertical_peak"),
    "mid-swing clearance (H)": ("clearance", "mid_swing_min_frac_H"),
    "tightest joint margin (rad)": ("range", "tightest", 2),
    "thigh twist use": ("range", "hip_yaw_range_use"),
}
#: Keys a walk's block has and a look-ladder rung block has not; keys never compared; the ladder's own addition.
WALK_ADDED = ("name", "kind", "reference", "range")
LADDER_NOT_COMPARED = ("speed", "variant", "seconds")
LADDER_ADDED = ("vs_V0",)


class Refused(s2io.S2Error):
    """A reference this screen cannot read (a file that ends inside a swing, arrays of another model)."""


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------
def _get(d, path):
    for p in path:
        if d is None:
            return None
        try:
            d = d[p]
        except (KeyError, IndexError, TypeError):
            return None
    return d


def _finite(x) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    return x[np.isfinite(x)]


def _fmax(x):
    x = _finite(x)
    return float(x.max()) if len(x) else None


def _drive_rows(drives: dict) -> list[tuple[str, str, float]]:
    return [(g, k.replace("_ratio", ""), float(v[k])) for g, v in drives.items() for k in KINDS]


def _item(o: dict) -> str:
    """One ``{limit, value}`` item in words: four decimals, the precision the margins are read to (a margin of 0.0997
    rad is not printed as its 0.10 rad gate)."""
    v = o.get("value")
    return f"{o['limit']} {v:.4f}" if isinstance(v, float) else (f"{o['limit']} {v}" if v is not None else o["limit"])


def _not_read(what: str, value=None) -> dict:
    return dict(limit=NOT_READ + what, value=value)


def _text(over: list, blocks: list, a1: bool, kind: str) -> str:
    """A verdict's sentence: what is over a limit or not read, and what that does to the freeze (the plan's A12)."""
    if not over:
        return "within the Stage 0 limits"
    read = [o for o in over if not o["limit"].startswith(NOT_READ)]
    items = "; ".join((["over " + "; ".join(_item(o) for o in read)] if read else [])
                      + [_item(o) for o in over if o["limit"].startswith(NOT_READ)])
    if blocks:
        return (f"{items}: blocks the freeze (" + "; ".join(o["limit"] for o in blocks) + ")"
                + ("; a walk that is not frozen is dropped from the ladder with its numbers" if kind == WALK else ""))
    if a1 and kind == TURN:
        return f"{items}: does not block the freeze; a1_over, read under the plan's third reading only"
    return f"{items}: reported, does not block the freeze"


def load_arrays(set_dir, entry: dict) -> dict:
    """A move's arrays, checked against the sha256 its entry records."""
    if entry.get("available") is False or not entry.get("file"):
        raise s2io.S2Error(f"{entry.get('name')!r} is recorded as not available: it has no file to screen")
    p = Path(set_dir) / entry["file"]
    sha = s2io.sha256_file(p)
    if sha != entry["sha256"]:
        raise s2io.S2Error(f"{p} is not the file its entry records (sha256 {sha[:12]}..., recorded "
                           f"{str(entry['sha256'])[:12]}...)")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def move_kind(entry: dict) -> tuple[str, bool]:
    """(kind, causal) of a ``transitions.json`` entry: ``"turn"`` for a finite turn (``move.kind`` or ``move.builder``
    ``"turn"``, or a ``turn`` block), else ``"start"``, ``"change"`` or ``"stop"`` from ``move.builder`` or, for an
    entry of Stage 1's own schema, from ``kind`` (its ``"walk"`` is a change between two gaits)."""
    move = entry.get("move") or {}
    causal = bool(move.get("causal"))
    if move.get("kind") == TURN or move.get("builder") == TURN or isinstance(entry.get("turn"), dict):
        return TURN, causal
    for key in (move.get("builder"), move.get("kind"), entry.get("kind")):
        if key in KIND_OF:
            return KIND_OF[key], causal
    raise s2io.S2Error(f"{entry.get('name')!r}: no kind can be read from its entry (move.builder {move.get('builder')!r}, "
                       f"kind {entry.get('kind')!r})")


def support_phases(swing) -> list[tuple]:
    """The support tuples ``clearance.screen`` reads, from the planned ``swing`` flags (a foot not in swing stands)."""
    sw = np.asarray(swing, bool)
    return [tuple(s for j, s in enumerate(SIDES) if not sw[k, j]) for k in range(len(sw))]


def check_swing_window(name: str, swing, swing_u=None) -> None:
    """``Refused`` if a foot is in swing on the file's last frame, or is part of the way through a swing on its first.

    ``clearance._swing_runs`` reads swings as cyclic runs and ``clearance._swing_report`` takes the middle half of
    each run: a foot in swing on the first and on the last frame would have the two parts joined into one swing, and
    the middle half of a swing cut by either end of the file is not the swing's.  Every move built for a set ends at
    a stride boundary in double support or in a stand, and begins in a stand or at a lift-off (a causal move's
    command).  ``swing_u`` is the file's swing fraction per frame and foot: a swing on the first frame begins there
    when its ``swing_u`` is at most ``SWING_START_TOL``; with no ``swing_u`` that cannot be shown and the file is
    refused."""
    sw = np.asarray(swing, bool)
    for j, side in enumerate(SIDES):
        if sw[-1, j]:
            both = " and on its first" if sw[0, j] else ""
            raise Refused(f"{name}: the {side} foot is in swing on the file's last frame{both}; the clearance screen "
                          "reads swings as cyclic runs, so a window that ends inside a swing is not screened")
    for j, side in enumerate(SIDES):
        if not sw[0, j]:
            continue
        if swing_u is None:
            raise Refused(f"{name}: the {side} foot is in swing on the file's first frame and the file has no swing_u "
                          "to show that the swing begins there; a window that begins inside a swing is not screened")
        u = float(np.asarray(swing_u, float)[0, j])
        if not u <= SWING_START_TOL:  # also a swing_u that is not a number
            raise Refused(f"{name}: the {side} foot is {u:g} of the way through a swing on the file's first frame; "
                          "the clearance screen reads the middle half of each swing run, so a window that begins "
                          "inside a swing is not screened")


# ------------------------------------------------------------------------------------------------
# Readings of a move
# ------------------------------------------------------------------------------------------------
def frame_weights(t) -> tuple[np.ndarray, bool]:
    """Each frame's duration (s) and whether the frames are uniform in time.

    A frame's duration is half the time between its two neighbours; the first and the last frame take their one
    interval (``np.gradient(t)``), so K frames at one interval last K intervals, as a periodic window does.  The
    frames are uniform when the durations differ by at most ``UNIFORM_TOL`` of their mean.  ``Refused`` for fewer
    than two frames or a time that does not increase."""
    t = np.asarray(t, float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0.0):
        raise Refused("the file's t is not an increasing series of at least two frame times")
    w = np.gradient(t)
    return w, bool(np.ptp(w) <= UNIFORM_TOL * float(np.mean(w)))


def rms_over_time(x, w) -> float:
    """The RMS of ``x`` with each sample weighted by ``w``: sqrt(sum(w x^2) / sum(w))."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float(np.sqrt(np.sum(w * x**2) / np.sum(w)))


def _frame_of(names, K: int) -> dict:
    """Per drive group, the frame of each sample of ``sizing._motor_space``'s stacked series (both sides of a leg
    drive in one series), read from that function itself on a series that holds the frame's number."""
    probe = np.repeat(np.arange(K, dtype=float)[:, None], len(names), axis=1)
    return {g: np.rint(v[0]).astype(int) for g, v in sizing._motor_space(names, probe, probe, "").items()}


def drive_flags(src: dict, tau, omega, t) -> dict:
    """The drive ratios of a ``(tau, omega)`` series at the frame times ``t``, as Stage 0 reads them
    (``sizing._motor_space`` -> ``envelope._ratios`` at the record's bus), and what they decide: ``{drives, governing,
    over, near, over_leg_drive, a1_over, frames_uniform}``.

    The sustained ratio is the RMS over time (``RULES["sustained"]``): per group ``rms`` and ``cont_ratio`` weight
    each frame by its duration (``frame_weights``), ``rms_frames`` and ``cont_ratio_frames`` are
    ``envelope._ratios``' own, and ``worst`` is the largest of the three gated ratios.  On frames that are uniform
    in time the two readings are the same number (the frame-weighted one, bit for bit).  ``over`` lists every ratio
    above 1 as ``{limit, value}``; ``a1_over`` is true when one of them is a neck or tail drive's;
    ``over_leg_drive`` when one is any other drive's (a leg's, or the expressive head's on a body that has it): no
    reference is frozen."""
    M, design = src["M"], src["design"]
    bus = src["record"]["selection"]["bus"]
    tau, omega = np.asarray(tau, float), np.asarray(omega, float)
    w, uniform = frame_weights(t)
    if len(w) != len(tau):
        raise Refused(f"t holds {len(w)} frames, tau {len(tau)}")
    ms = sizing._motor_space(M.names, tau, omega, src["layout"])
    drives = envelope._ratios(design, bus, ms)
    frame_of = None if uniform else _frame_of(M.names, len(tau))
    for g, d in drives.items():
        d["rms_frames"], d["cont_ratio_frames"] = d["rms"], d["cont_ratio"]
        if not uniform:
            d["rms"] = rms_over_time(ms[g][0], w[frame_of[g]])
            d["cont_ratio"] = d["rms"] / design.act(g).sustained
            d["worst"] = max(d["cont_ratio"], d["peak_ratio"], d["speed_ratio"])
    rows = _drive_rows(drives)
    over = [dict(limit=f"{g} {k}", value=r) for g, k, r in rows if r > 1.0]
    near = [dict(limit=f"{g} {k}", value=r) for g, k, r in rows if 1.0 - NEAR_BAND < r <= 1.0]
    return s2io.plain(dict(drives=drives, governing=screen.governing(drives), over=over, near=near,
                           over_leg_drive=any(g not in A1_GROUPS and r > 1.0 for g, _, r in rows),
                           a1_over=any(g in A1_GROUPS and r > 1.0 for g, _, r in rows), frames_uniform=uniform))


def passive_over_time(M: statics.Model, passive: dict, tau, omega, t) -> dict:
    """``screen.passive_loads``' block with its torque RMS values and sustained ratios read over time, as the
    drives' (``drive_flags``): ``rms_id_nm``, ``rms_passive_alone_nm``, ``rms_total_nm``, ``cont_ratio_id``,
    ``cont_ratio_total`` and ``rms_total_over_id`` weight each frame by its duration; the frame-weighted values stay
    beside them (``rms_id_frames_nm``, ``rms_total_frames_nm``, ``cont_ratio_id_frames``,
    ``cont_ratio_total_frames``).  On frames uniform in time nothing is computed again.  The peaks need no weight;
    the speed RMS, the moving fraction and the mean powers stay statistics of the frames."""
    tau, omega = np.asarray(tau, float), np.asarray(omega, float)
    w, uniform = frame_weights(t)
    out = {}
    for grp, p in passive.items():
        p = dict(p)
        p.update(rms_id_frames_nm=p["rms_id_nm"], rms_total_frames_nm=p["rms_total_nm"],
                 cont_ratio_id_frames=p["cont_ratio_id"], cont_ratio_total_frames=p["cont_ratio_total"])
        if not uniform:
            i = M.names.index(p["joint"])
            t_id, om = tau[:, i], omega[:, i]
            t_p = p["damping"] * om + p["frictionloss"] * np.sign(om)  # passive_loads' terms, in its order
            t_tot = t_id + p["damping"] * om + p["frictionloss"] * np.sign(om)
            rid, rtot = rms_over_time(t_id, w), rms_over_time(t_tot, w)
            p.update(rms_id_nm=rid, rms_passive_alone_nm=rms_over_time(t_p, w), rms_total_nm=rtot,
                     cont_ratio_id=rid / p["sustained_nm"], cont_ratio_total=rtot / p["sustained_nm"],
                     rms_total_over_id=rtot / max(rid, 1e-12))
        out[grp] = p
    return out


def contact_pass(M: statics.Model, q, qd, qdd, swing, split: str, mu: float = turning.MU) -> dict:
    """The contact wrenches of every frame on the planned support, solved again from the file's ``q, qd, qdd``.

    As ``transitions.idyn``: joint torques, per-foot CoP margin and normal force, the largest friction ratio per
    frame; the wrench is unique in single support (``turning.contact_torques(..., "exact")``) and split by
    ``split`` in double support (``"exact"``: ``turning.contact_torques``; else ``statics.contact_torques``).  For
    the stance foot of a single-support frame the formulas of ``turning.inverse_dynamics``: the free moment about
    the CoP, the pressure radius, the utilisation at ``mu``, ``mu_req``, ``mu_req_cop``, ``mu_req_centroid``.  For a
    double-support frame ``turning.ds_min_mu``.  Returns ``tau, cop_margin, normal, friction, feet, r_eff, ds,
    how, phases``."""
    K = len(q)
    swing = np.asarray(swing, bool)
    tau = np.zeros((K, len(M.act_dofs)))
    cop_margin = np.full((K, 2), np.nan)
    normal = np.zeros((K, 2))
    friction = np.zeros(K)
    keys = ("fn", "ft", "mz", "r_cop", "util", "util_cop", "mu_req", "mu_req_cop", "mu_req_centroid", "margin")
    feet = {s: {k: np.full(K, np.nan) for k in keys} for s in SIDES}
    M.forward(q[0])
    r_eff = {s: turning.uniform_radius(*turning.pad_samples(M, s)) for s in SIDES}
    ds = dict(min_mu=np.full(K, np.nan), min_mu_lower=np.full(K, np.nan), feasible=np.ones(K, bool))
    how, phases = [], []
    for k in range(K):
        sup = tuple(s for j, s in enumerate(SIDES) if not swing[k, j]) or SIDES
        phases.append(sup)
        if split == "exact" or len(sup) == 1:
            tk, wr, hw = turning.contact_torques(M, q[k], qd[k], qdd[k], sup, split="exact")
        else:
            tk, wr = statics.contact_torques(M, q[k], qd[k], qdd[k], support=sup)
            hw = "statics"
        how.append(hw)
        tau[k] = tk
        cops = {}
        for j, side in enumerate(SIDES):  # transitions.idyn's lines, in its order
            if side in wr:
                f = wr[side][:3]
                normal[k, j] = f[2]
                friction[k] = max(friction[k], float(np.linalg.norm(f[:2]) / max(f[2], 1e-6)))
                c = statics.cop(M, side, wr[side])
                M.forward(q[k])
                cop_margin[k, j] = statics.support_margin(c, M.pad_polygon(side))
                cops[side] = c
        M.forward(q[k])
        cen = {}
        for side in SIDES:
            pts, wts = turning.pad_samples(M, side)
            cen[side] = (wts[:, None] * pts).sum(0) / wts.sum()  # pad-area centroid (world xy)
            if len(sup) != 1 or side not in wr:
                continue
            f, mom = wr[side][:3], wr[side][3:]
            p, _ = M.foot_pose(side)
            F = feet[side]
            fn, ft = float(f[2]), float(np.linalg.norm(f[:2]))
            F["fn"][k], F["ft"][k] = fn, ft
            c = cops[side]
            if not np.all(np.isfinite(c)):
                continue
            mz = float(mom[2] + np.cross(p - np.array([c[0], c[1], 0.0]), f)[2])
            mz_cen = float(mom[2] + (p[0] - cen[side][0]) * f[1] - (p[1] - cen[side][1]) * f[0])
            margin = float(cop_margin[k, SIDES.index(side)])
            r_cop = turning.pressure_radius(pts, wts, c) if margin >= 0 else 0.0
            F["mz"][k], F["margin"][k], F["r_cop"][k] = mz, margin, r_cop
            F["util"][k] = abs(mz) / (mu * fn * r_eff[side])
            F["util_cop"][k] = abs(mz) / (mu * fn * r_cop) if r_cop > 0 else np.inf
            F["mu_req"][k] = math.hypot(ft / fn, mz / (fn * r_eff[side]))
            F["mu_req_cop"][k] = math.hypot(ft / fn, mz / (fn * r_cop)) if r_cop > 0 else np.inf
            F["mu_req_centroid"][k] = math.hypot(ft / fn, mz_cen / (fn * r_eff[side]))
        if len(sup) == 2:
            r = turning.ds_min_mu(M, q[k], qd[k], qdd[k], r_eff, cen)
            ds["feasible"][k] = bool(r["feasible"])
            if r["mu"] is not None:
                ds["min_mu"][k] = r["mu"]
            if r.get("mu_lower") is not None:
                ds["min_mu_lower"][k] = r["mu_lower"]
    return dict(tau=tau, cop_margin=cop_margin, normal=normal, friction=friction, feet=feet, r_eff=r_eff, ds=ds,
                how=how, phases=phases, mu=mu)


def _contact_block(M, cp: dict, t, causal: bool, split: str) -> dict:
    phases, cm = cp["phases"], cp["cop_margin"]
    single = np.array([k for k, p in enumerate(phases) if len(p) == 1], int)
    double = np.array([k for k, p in enumerate(phases) if len(p) == 2], int)
    weight = float(M.mass() * G)
    rule = COP_RULE_CAUSAL_M if causal else COP_RULE_M
    out = dict(single_support_cop_margin_min_mm=None, single_support_cop_margin_min_at=None, cop_rule_mm=1e3 * rule,
               cop_rule=("causal move: twice the tracker's pad inset inside the pad" if causal else
                         "Stage 0's envelope rule: the loads are read while the CoP is within 10 mm of the pad"),
               single_support_frames=len(single), double_support_frames=len(double),
               double_support_cop_margin_min_mm=None, friction_ratio_max=float(np.max(cp["friction"])), split=split,
               split_solved={h: sum(1 for x in cp["how"] if x == h) for h in sorted(set(cp["how"])) if h != "unique"},
               split_failed_frames=sum(1 for x in cp["how"] if x == "infeasible"),
               total_normal_frac_weight=[float(np.min(cp["normal"].sum(axis=1)) / weight),
                                         float(np.max(cp["normal"].sum(axis=1)) / weight)])
    if len(single) and np.isfinite(cm[single]).any():
        m = cm[single]
        i, j = np.unravel_index(int(np.nanargmin(m)), m.shape)
        out["single_support_cop_margin_min_mm"] = float(1e3 * m[i, j])
        out["single_support_cop_margin_min_at"] = dict(frame=int(single[i]), t_s=float(t[single[i]] - t[0]),
                                                       foot=SIDES[j])
    if len(double) and np.isfinite(cm[double]).any():
        out["double_support_cop_margin_min_mm"] = float(1e3 * np.nanmin(cm[double]))
    return out


def _friction_block(M, cp: dict, t) -> dict:
    """The look screen's ``straight_run`` keys on a move: the single-support friction need, per foot
    ``turning._foot_stats``, the double-support requirement and the grips."""
    phases = cp["phases"]
    weight = float(M.mass() * G)
    feet = {}
    for side in SIDES:
        idx = np.array([k for k, p in enumerate(phases) if p == (side,)], int)
        feet[side] = turning._foot_stats(cp["feet"][side], idx, phases)
    ss = np.array([k for k, p in enumerate(phases) if len(p) == 1], int)
    ds = np.array([k for k, p in enumerate(phases) if len(p) == 2], int)
    D = cp["ds"]
    single = dict(frames=len(ss), mz_abs_max_nm=None, mz_rms_nm=None, mz_peak_frame=None, mz_peak_stance=None,
                  mu_req_max=None, mu_req_max_frame=None, mu_req_max_t_s=None, mu_req_max_stance=None,
                  mu_req_cop_max=None, util_max=None, fn_min_frac_weight=None, mu_req_peak_frame_is_mz_peak_frame=None)
    mu_max = None
    if len(ss):
        stance = [phases[k][0] for k in ss]

        def per_frame(key):
            return np.array([cp["feet"][s][key][k] for s, k in zip(stance, ss, strict=True)], float)

        mz, mu, mu_cop, util, fn = (per_frame(k) for k in ("mz", "mu_req", "mu_req_cop", "util", "fn"))
        if np.isfinite(mz).any() and np.isfinite(mu).any():
            k_mz = int(ss[int(np.nanargmax(np.abs(mz)))])
            k_mu = int(ss[int(np.nanargmax(mu))])
            mu_max = float(np.nanmax(mu))
            single.update(mz_abs_max_nm=float(np.nanmax(np.abs(mz))), mz_rms_nm=float(np.sqrt(np.nanmean(mz**2))),
                          mz_peak_frame=k_mz, mz_peak_stance=phases[k_mz][0], mu_req_max=mu_max, mu_req_max_frame=k_mu,
                          mu_req_max_t_s=float(t[k_mu] - t[0]), mu_req_max_stance=phases[k_mu][0],
                          mu_req_cop_max=_fmax(mu_cop), util_max=_fmax(util),
                          fn_min_frac_weight=float(np.nanmin(fn) / weight),
                          mu_req_peak_frame_is_mz_peak_frame=bool(k_mz == k_mu))
    return dict(single_support=single, feet=feet,
                double_support=dict(frames=len(ds), read=len(ds), min_mu_max=_fmax(D["min_mu"][ds]) if len(ds) else None,
                                    min_mu_lower_max=_fmax(D["min_mu_lower"][ds]) if len(ds) else None,
                                    infeasible_frames=int(np.sum(~D["feasible"][ds])) if len(ds) else 0),
                grips={f"{g:g}": (None if mu_max is None else bool(mu_max <= g)) for g in GRIPS},
                mu_assumed=cp["mu"])


def move_sway(t, phase, com, path, heading, H: float) -> dict:
    """The model CoM's lateral excursion in the path frame (m across the heading, from the planned path point of
    each frame), peak to peak per stride of the move (strides by the whole part of the gait phase) and over the
    whole file, / H; against BEHAVIOR_SPEC's 0.12 H (reported)."""
    com, path, heading = np.asarray(com, float), np.asarray(path, float), np.asarray(heading, float)
    d = com[:, :2] - path
    lat = -np.sin(heading) * d[:, 0] + np.cos(heading) * d[:, 1]
    idx = np.floor(np.asarray(phase, float) + 1e-9).astype(int)
    per = [dict(stride=int(i), pp_H=float(np.ptp(lat[idx == i]) / H)) for i in np.unique(idx)
           if np.count_nonzero(idx == i) > 1]
    vals = [p["pp_H"] for p in per]
    return dict(com_pp_H=float(np.ptp(lat) / H), com_pp_per_stride_max_H=float(np.max(vals)) if vals else None,
                com_pp_per_stride_mean_H=float(np.mean(vals)) if vals else None, per_stride=per, strides=len(per),
                spec_H=screen.SWAY_SPEC_H, frame="path frame: across the planned heading, from the planned path point")


def _second_difference(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The three-point second difference on the file's own (not uniform) frame times, the interior frames only: the
    look screen's periodic second difference where the two intervals are equal, with no wrap."""
    h1, h2 = (t[1:-1] - t[:-2]), (t[2:] - t[1:-1])
    shape = (-1,) + (1,) * (x.ndim - 1)
    h1, h2 = h1.reshape(shape), h2.reshape(shape)
    return 2.0 * ((x[2:] - x[1:-1]) / h2 - (x[1:-1] - x[:-2]) / h1) / (h1 + h2)


def forward_acceleration_lp(t, com_xy, heading, fs: float = RESAMPLE_HZ) -> np.ndarray:
    """The model CoM's acceleration along the per-frame heading, low-passed at 1 Hz (``stage1.metrics.lowpass``,
    decision 10's filter) on a uniform ``fs`` resample of the file's own frames (linear, no wrap, second
    differences): the reading of ``readings.reference_readings`` and of the move's own entry."""
    t, c = np.asarray(t, float), np.asarray(com_xy, float)
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    x, y, h = np.interp(tu, t, c[:, 0]), np.interp(tu, t, c[:, 1]), np.interp(tu, t, np.asarray(heading, float))
    ax, ay = np.gradient(np.gradient(x, tu), tu), np.gradient(np.gradient(y, tu), tu)
    return metrics.lowpass(ax * np.cos(h) + ay * np.sin(h), fs)


def move_accelerations(t, com, heading, feet) -> dict:
    """The model CoM's accelerations over a move (m/s^2), on its own frames with no wrap: the horizontal peak, the
    peaks along and across the heading, the vertical range and the feet's (origins') vertical peak from second
    differences on the file's frames (the first and last frame left out); the part along the heading low-passed at
    1 Hz on a uniform 500 Hz resample (``forward_acceleration_lp``), which decision 10's 1.5 m/s^2 is read on.

    The filter has the file alone, so its value at an end that joins a walk is the unfiltered series' there:
    ``com_forward_lp1hz_peak_at_end`` is true when the peak lies within ``ACCEL_EDGE_S`` of the file's first or last
    sample, and ``com_forward_lp1hz_peak_interior`` is the largest value at least that far from both (None for a file
    shorter than twice that)."""
    t, com, heading = np.asarray(t, float), np.asarray(com, float), np.asarray(heading, float)
    acc = _second_difference(com, t)
    h = heading[1:-1]
    fwd = acc[:, 0] * np.cos(h) + acc[:, 1] * np.sin(h)
    lat = -acc[:, 0] * np.sin(h) + acc[:, 1] * np.cos(h)
    fa = _second_difference(np.asarray(feet, float)[:, :, 2], t)
    lp = forward_acceleration_lp(t, com[:, :2], heading)
    k = int(np.argmax(np.abs(lp)))
    edge = int(round(ACCEL_EDGE_S * RESAMPLE_HZ))
    inner = np.abs(lp[edge : len(lp) - edge])
    return dict(com_horizontal_peak=float(np.max(np.linalg.norm(acc[:, :2], axis=1))),
                com_forward_peak=float(np.max(np.abs(fwd))), com_lateral_peak=float(np.max(np.abs(lat))),
                com_forward_lp1hz_peak=float(np.max(np.abs(lp))), com_forward_lp1hz_peak_t_s=float(k / RESAMPLE_HZ),
                com_forward_lp1hz_at_peak=float(lp[k]),
                com_forward_lp1hz_peak_at_end=bool(k < edge or k > len(lp) - 1 - edge),
                com_forward_lp1hz_peak_interior=float(inner.max()) if len(inner) else None,
                com_forward_lp1hz_edge_s=ACCEL_EDGE_S, com_vertical_min=float(acc[:, 2].min()),
                com_vertical_max=float(acc[:, 2].max()), foot_vertical_peak=float(np.max(np.abs(fa))),
                foot_vertical_up_peak=float(fa.max()), spec_start_stop=ACCEL_RULE_M_S2, resample_hz=RESAMPLE_HZ)


def hip_yaw_use(joints: dict) -> float | None:
    """The worse leg's thigh-twist use of the usable range (``turning.summarize``'s formula: the largest fraction of
    a limit reached, 1.0 = at the limit) from ``clearance.screen``'s ``joint_range_margins`` (its rounded
    excursions)."""
    vals = []
    for jn, r in joints.items():
        if jn.endswith("_hip_yaw"):
            lo, hi = r["usable"]
            if lo and hi:
                vals.append(max(r["walk_max"] / hi, r["walk_min"] / lo))
    return float(max(vals)) if vals else None


def _tightest(joints: dict, leave_out=(), only=()) -> list | None:
    gated = [g for g in clearance.gated_margins(joints) if not g[0].endswith(tuple(leave_out) or ("\0",))
             and (not only or g[0].endswith(tuple(only)))]
    return list(min(gated, key=lambda x: x[2])) if gated else None


def usable_differences(joints: dict, usable: dict) -> list[str]:
    """The joints whose usable range in ``clearance.screen``'s margins is not ``plant.usable_ranges(body)``'s, or
    that only one of the two holds (sorted).  The margins and the thigh-twist use are read on one set of ranges only
    when this is empty."""
    return sorted(jn for jn in set(joints) | set(usable)
                  if jn not in joints or jn not in usable
                  or [float(v) for v in joints[jn]["usable"]] != [float(v) for v in usable[jn]])


def walk_range(margins: dict, usable: dict | None = None) -> dict:
    """A walk's ``range`` block from ``clearance.screen``'s ``joint_range_margins`` (rad, as that screen rounds
    them): the joints, the tightest gated end ``[joint, "min" | "max", margin]``, Stage 0's 0.10 rad rule (``ok``:
    false also when no margin was read).  ``usable``: ``plant.usable_ranges(body)``; ``usable_differs_from_plant``
    lists the joints whose range in the margins is another (None where ``usable`` was not given)."""
    joints = {jn: dict(r) for jn, r in (margins or {}).items()}
    tight = _tightest(joints)
    return dict(joints=joints, tightest=tight, rule_rad=RANGE_RULE_RAD,
                ok=bool(tight is not None and tight[2] >= RANGE_RULE_RAD), hip_yaw_range_use=hip_yaw_use(joints),
                usable_differs_from_plant=None if usable is None else usable_differences(joints, usable),
                source="the look entry's price.clearance.joint_range_margins (clearance.screen, gates r2)")


def move_range(src: dict, q, kind: str, entry: dict, margins: dict) -> dict:
    """A move's ``range`` block.  ``margins`` are ``clearance.screen``'s ``joint_range_margins`` of the move (rad,
    as that screen rounds them; its usable ranges are checked against ``plant.usable_ranges(body)``): Stage 0's 0.10
    rad rule is read on them, as on a walk (``stage0_rule_ok``).  ``hip_yaw_range_use`` is the worse leg's use of the
    usable thigh twist on the file's ``q`` at full precision (``turning.summarize``'s formula on
    ``plant.usable_ranges``, the builder's reading; the entry's own value beside it).  ``ok`` is the rule that
    applies to ``kind``: 0.10 rad on every gated joint, or for a turn a twist use of at most 0.87 and 0.10 rad on
    every other gated joint (``tightest_other``; ``tightest_twist`` is the twist's own tightest margin).  A margin
    or a twist use that was not read fails its rule."""
    m = src["M"].m
    q = np.asarray(q, float)
    joints = {jn: dict(r) for jn, r in (margins or {}).items()}
    usable = plant.usable_ranges(src["body"])
    differs = usable_differences(joints, usable)
    exact, uses = {}, []
    for jn, (lo, hi) in usable.items():
        qv = q[:, m.jnt_qposadr[statics.jid(m, jn)]]
        exact[jn] = dict(usable=[float(lo), float(hi)], margin_min=float(qv.min() - lo), margin_max=float(hi - qv.max()))
        if jn in clearance.REST_ON_STOP:
            exact[jn]["rests_on_stop"] = clearance.REST_ON_STOP[jn]
        if jn.endswith("_hip_yaw") and lo and hi:
            uses.append(max(float(qv.max()) / hi, float(qv.min()) / lo))
    use = float(max(uses)) if uses else None
    tight = _tightest(joints)
    other = _tightest(joints, leave_out=("_hip_yaw",))
    stage0 = bool(tight is not None and tight[2] >= RANGE_RULE_RAD)
    recorded = entry.get("hip_yaw_range_use")
    out = dict(joints=joints, tightest=tight, tightest_unrounded=_tightest(exact), rule_rad=RANGE_RULE_RAD,
               stage0_rule_ok=stage0, hip_yaw_range_use=use, hip_yaw_range_use_entry=recorded,
               hip_yaw_range_use_equals_entry=None if recorded is None or use is None else bool(use == recorded),
               usable_differs_from_plant=differs,
               source="clearance.screen's joint_range_margins of the move (gates r2); the thigh-twist use from the "
                      "file's q against plant.usable_ranges(body)")
    if kind == TURN:
        twist_ok = use is not None and use <= TWIST_USE_MAX
        other_ok = other is not None and other[2] >= RANGE_RULE_RAD
        out.update(rule=f"a turn: thigh twist use at most {TWIST_USE_MAX:g} of the usable range; every other gated "
                        f"joint {RANGE_RULE_RAD:g} rad", twist_use_max=TWIST_USE_MAX, twist_ok=bool(twist_ok),
                   tightest_other=other, tightest_twist=_tightest(joints, only=("_hip_yaw",)),
                   ok=bool(twist_ok and other_ok))
    else:
        out.update(rule=f"{RANGE_RULE_RAD:g} rad on every gated joint", ok=stage0)
    return out


def _passive_lists(passive: dict) -> tuple[list, list]:
    def rows(test):
        return [dict(limit=f"{grp} {kind.replace('_ratio_total', '')} with the plant's damping and friction",
                     value=passive[grp][kind]) for grp in ("neck", "tail")
                for kind in ("cont_ratio_total", "peak_ratio_total") if test(passive[grp][kind])]

    return rows(lambda r: r > 1.0), rows(lambda r: 1.0 - NEAR_BAND < r <= 1.0)


def judge_move(kind: str, causal: bool, flags: dict, ik_ok: bool, ik_fail_frames, contact: dict, clear: dict,
               rng: dict, accel: dict) -> dict:
    """A move's verdict by ``RULES``: ``{text, within_limits, over, near, a1_over, blocks_freeze}``.  ``over`` lists
    every limit of the table that the move is over, as ``{limit, value}``, and every gate whose reading is missing
    (``"not read: ..."``); ``blocks_freeze`` those that block its freeze.  A gate that is only reported for the kind
    (a turn's 1 Hz acceleration, Stage 0's 0.10 rad on a turn's thigh twist) is in neither."""
    over, blocks = [], []

    def add(item: dict, blocking: bool = True):
        over.append(item)
        if blocking:
            blocks.append(item)

    for o in flags["over"]:
        add(o, blocking=not (o["limit"].split()[0] in A1_GROUPS and kind == TURN))
    if not ik_ok:
        add(dict(limit="IK", value=ik_fail_frames))
    cop = contact["single_support_cop_margin_min_mm"]
    if cop is None:
        add(_not_read("single-support CoP margin"))
    elif cop < contact["cop_rule_mm"]:
        add(dict(limit=f"single-support CoP margin under {contact['cop_rule_mm']:g} mm", value=cop))
    if contact.get("split_failed_frames"):
        add(_not_read("the contact split did not solve in double-support frames", contact["split_failed_frames"]))
    for gate in CLEARANCE_GATES:
        if clear["gates"].get(gate) is not True:
            add(dict(limit=f"clearance {gate}",
                     value=clear["mid_swing_min_frac_H"] if gate == "swing_clearance" else None))
    tight = rng.get("tightest_other") if kind == TURN else rng.get("tightest")
    if kind == TURN:
        if rng.get("hip_yaw_range_use") is None:
            add(_not_read("thigh twist use"))
        elif not rng["twist_ok"]:
            add(dict(limit=f"thigh twist use above {TWIST_USE_MAX:g}", value=rng["hip_yaw_range_use"]))
    if tight is None:
        add(_not_read("joint range margins" + (" of a joint other than the thigh twist" if kind == TURN else "")))
    elif tight[2] < RANGE_RULE_RAD:
        add(dict(limit=f"joint range margin under {RANGE_RULE_RAD:g} rad ({tight[0]} {tight[1]})", value=tight[2]))
    if rng.get("usable_differs_from_plant"):
        add(_not_read("the usable ranges are not plant.usable_ranges' (" + ", ".join(rng["usable_differs_from_plant"])
                      + ")"))
    if kind in STRAIGHT and accel["com_forward_lp1hz_peak"] > ACCEL_RULE_M_S2:
        add(dict(limit=f"1 Hz forward acceleration above {ACCEL_RULE_M_S2:g} m/s^2",
                 value=accel["com_forward_lp1hz_peak"]))
    a1 = bool(flags["a1_over"])
    return dict(text=_text(over, blocks, a1, kind), within_limits=not over, over=over, near=flags["near"],
                a1_over=a1, blocks_freeze=blocks)


def judge_walk(block: dict) -> dict:
    """A walk's verdict from its ``screen_rung`` block with ``range`` added: ``{text, ladder_text, within_limits,
    over, near, a1_over, blocks_freeze}``.

    Every item of ``look.price``'s ``over`` list that is a gate of ``RULES`` blocks the freeze: a drive ratio, IK,
    the CoP rule, a clearance gate (the 0.10 rad joint range among them; a gate that is not true is listed, read or
    not) and a rolling-contact split that did not solve (the drive loads of those frames are then not read).  A gate
    whose reading is missing is added to ``over`` here and blocks: no single-support CoP margin, no joint-range
    margins, usable ranges that are not ``plant.usable_ranges``'.  ``a1_over`` when a neck or tail ratio exceeds 1
    (on a walk it blocks).  ``text`` is this screen's sentence (the plan's A12); ``ladder_text`` is ``screen_rung``'s,
    the look ladder's."""
    groups = set(block["drives"])
    over = list(block["over"])
    blocks = [o for o in over
              if o["limit"].split()[0] in groups or o["limit"] == "IK" or o["limit"].startswith("single-support CoP")
              or o["limit"].startswith("clearance ") or o["limit"].startswith("rolling-contact split")]

    def add(item: dict):
        over.append(item)
        blocks.append(item)

    if _get(block, ("contact", "single_support_cop_margin_min_mm")) is None:
        add(_not_read("single-support CoP margin"))
    rng = block.get("range") or {}
    tight = rng.get("tightest")
    if tight is None:
        add(_not_read("joint range margins"))
    elif tight[2] < RANGE_RULE_RAD and not any(o["limit"] == "clearance joint_range_margin" for o in over):
        add(dict(limit=f"joint range margin under {RANGE_RULE_RAD:g} rad ({tight[0]} {tight[1]})", value=tight[2]))
    if rng.get("usable_differs_from_plant"):
        add(_not_read("the usable ranges are not plant.usable_ranges' (" + ", ".join(rng["usable_differs_from_plant"])
                      + ")"))
    a1 = bool(any(g in A1_GROUPS and r > 1.0 for g, _, r in _drive_rows(block["drives"])))
    return dict(text=_text(over, blocks, a1, WALK), ladder_text=block["verdict"], within_limits=not over, over=over,
                near=block["near"], a1_over=a1, blocks_freeze=blocks)


def vs_walk(block: dict, walk_block: dict | None) -> dict | None:
    """Each headline number of a move against its walk's block: ``{walk {name, sha256}, headlines {label: [walk,
    move, move - walk]}}``; None without a walk's block."""
    if walk_block is None:
        return None
    out = {}
    for label, path in HEADLINES.items():
        a, b = _get(walk_block, path), _get(block, path)
        num = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
        out[label] = [a, b, (b - a) if num else None]
    return dict(walk=dict(name=walk_block.get("name"), sha256=walk_block["arrays"]["sha256"]), headlines=out)


# ------------------------------------------------------------------------------------------------
# The two screens
# ------------------------------------------------------------------------------------------------
def walk_reference(set_dir, name: str, entry: dict) -> dict:
    """``{file, sha256}`` of the Stage 1-format file of a walk (its ``refs.json`` entry), the file the bar plays,
    checked against the file on disk."""
    if not entry.get("file") or not entry.get("sha256"):
        raise s2io.S2Error(f"{name}: its refs.json entry names no file with a sha256")
    path = Path(set_dir) / entry["file"]
    sha = s2io.sha256_file(path)
    if sha != entry["sha256"]:
        raise s2io.S2Error(f"{path} is not the file its entry records (sha256 {sha[:12]}..., recorded "
                           f"{str(entry['sha256'])[:12]}...)")
    return dict(file=entry["file"], sha256=entry["sha256"])


def screen_walk(src: dict, set_dir, name: str, look_entry: dict, reference: dict | None = None) -> dict:
    """One periodic walk's block (module docstring, section 1).  ``src`` is ``look.source(body)``; ``look_entry`` the
    walk's entry of ``look.write``'s shape (``set.json`` ``look_entries[name]``), its ``file`` relative to
    ``set_dir``.  ``reference`` is the walk's ``refs.json`` entry: the block then names the Stage 1-format file the
    bar plays (``reference.{file, sha256}``, checked on disk) beside the look-format file it was read on
    (``arrays.{file, sha256}``); without it ``reference`` is None.  The set's name is the directory's
    (``<set>/<body>``)."""
    set_dir = Path(set_dir)
    ref = None if reference is None else walk_reference(set_dir, name, reference)
    with s2io.r2():
        block = screen.screen_rung(src, name, set_dir.parent.name, look_entry, out_dir=set_dir)
    seconds = block.pop("seconds")
    LOG.info("screen_walk %s %s: %.1f s", set_dir, name, seconds)
    margins = _get(look_entry, ("price", "clearance", "joint_range_margins"))
    if margins is None:
        raise s2io.S2Error(f"{name}: the look entry holds no price.clearance.joint_range_margins")
    block.update(name=name, kind=WALK, reference=ref,
                 range=walk_range(margins, plant.usable_ranges(src["body"])))
    block["verdict"] = judge_walk(block)
    return s2io.plain(block)


def screen_move(src: dict, set_dir, name: str, entry: dict, walk_block: dict | None) -> dict:
    """One move's block (module docstring, section 2).  ``entry`` is its ``transitions.json`` entry (Stage 1's own
    entries are read too); ``walk_block`` the block of its walk in the same screen, or None (no ``vs_walk``)."""
    t0 = time.perf_counter()
    M, design = src["M"], src["design"]
    arrs = load_arrays(set_dir, entry)
    kind, causal = move_kind(entry)
    if [str(j) for j in arrs["joints"]] != list(M.names):
        raise Refused(f"{name}: the file's joints are not {src['body']}'s")
    q, qd, qdd, tau, omega, t = (np.asarray(arrs[k], float) for k in ("q", "qd", "qdd", "tau", "omega", "t"))
    swing = np.asarray(arrs["swing"], bool)
    for key, a in (("q", q), ("qd", qd), ("qdd", qdd), ("tau", tau), ("omega", omega), ("t", t)):
        if not np.all(np.isfinite(a)):
            raise Refused(f"{name}: {key} holds a value that is not finite")
    check_swing_window(name, swing, arrs.get("swing_u"))
    try:
        weights, uniform = frame_weights(t)
    except Refused as exc:
        raise Refused(f"{name}: {exc}") from None
    split = entry["split"]
    ik_ok = bool(entry["ik_ok"])
    with s2io.r2():
        flags = drive_flags(src, tau, omega, t)
        passive = passive_over_time(M, screen.passive_loads(M, design, tau, omega), tau, omega, t)
        cp = contact_pass(M, q, qd, qdd, swing, split)
        clear = look._clearance_gates(src, dict(q=q, phases=support_phases(swing), ik_ok=ik_ok))
        M.forward(src["q_home"])
    contact = _contact_block(M, cp, t, causal, split)
    rng = move_range(src, q, kind, entry, clear.pop("joint_range_margins", None))
    accel = move_accelerations(t, arrs["com"], arrs["heading"], arrs["feet"])
    verdict = judge_move(kind, causal, flags, ik_ok, entry.get("ik_fail_frames"), contact, clear, rng, accel)
    over_p, near_p = _passive_lists(passive)
    single = np.array([len(p) == 1 for p in cp["phases"]])
    rec_cop, rec_fr = entry.get("min_cop_margin_single_m"), entry.get("max_friction_ratio")
    got_cop = float(np.nanmin(cp["cop_margin"][single])) if single.any() else None
    got_fr = float(np.max(cp["friction"]))
    file_cop = np.asarray(arrs["cop_margin"], float) if "cop_margin" in arrs else None
    recheck = dict(tau_max_abs_diff_nm=float(np.max(np.abs(cp["tau"] - tau))),
                   cop_margin_max_abs_diff_m=None if file_cop is None else _fmax(np.abs(cp["cop_margin"] - file_cop)),
                   cop_margin_same_feet=(None if file_cop is None else
                                         bool(np.array_equal(np.isnan(cp["cop_margin"]), np.isnan(file_cop)))),
                   friction_max_abs_diff=(float(np.max(np.abs(cp["friction"] - np.asarray(arrs["friction"], float))))
                                          if "friction" in arrs else None),
                   min_cop_margin_single_m=got_cop, min_cop_margin_single_m_recorded=rec_cop,
                   max_friction_ratio=got_fr, max_friction_ratio_recorded=rec_fr)
    recheck["equal_to_recorded"] = bool(
        recheck["tau_max_abs_diff_nm"] == 0.0 and recheck["cop_margin_max_abs_diff_m"] in (None, 0.0)
        and recheck["cop_margin_same_feet"] in (None, True) and recheck["friction_max_abs_diff"] in (None, 0.0)
        and got_cop == rec_cop and got_fr == rec_fr)
    move = entry.get("move") or {}
    ref_lp = _get(entry, ("reference", "accel_forward_lp1hz_peak_m_s2"))
    accel.update(com_forward_lp1hz_peak_entry=ref_lp,  # the builder's own reading (the plan's A7): another number
                 com_forward_lp1hz_peak_minus_entry=(None if ref_lp is None else
                                                     float(accel["com_forward_lp1hz_peak"] - ref_lp)))
    block = dict(
        body=src["body"], name=name, kind=kind,
        move=dict(kind=move.get("kind", entry.get("kind")), builder=move.get("builder"), causal=causal,
                  command_phase=move.get("command_phase"), from_=entry.get("from_"), to=entry.get("to"), split=split,
                  rules=move.get("rules")),
        arrays=dict(file=entry["file"], sha256=entry["sha256"], frames=len(q), duration_s=float(t[-1] - t[0]),
                    frames_uniform=uniform, frame_s=[float(weights.min()), float(weights.max())]),
        drives=flags["drives"], governing=flags["governing"], passive=passive,
        summary=dict(min_cop_margin_single_m=rec_cop, max_friction_ratio=rec_fr, ik_ok=ik_ok,
                     ik_fail_frames=entry.get("ik_fail_frames")),
        contact=contact, friction=_friction_block(M, cp, t),
        sway=move_sway(t, arrs["phase"], arrs["com"], arrs["path"], arrs["heading"], src["H"]), accel=accel,
        clearance=clear, range=rng, over=verdict["over"], near=verdict["near"], over_with_passive=over_p,
        near_with_passive=near_p, within_limits=verdict["within_limits"], verdict=verdict, recheck=recheck)
    block["vs_walk"] = vs_walk(block, walk_block)
    LOG.info("screen_move %s %s: %.1f s", set_dir, name, time.perf_counter() - t0)
    return s2io.plain(block)


def _same(a, b, path: str, out: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in the {'ladder' if k in b else 'set'}'s block")
            else:
                _same(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items against {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                _same(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    elif type(a) is not type(b) or a != b:
        out.append(f"{path}: {a!r} against {b!r}")


def ladder_differences(block: dict, rung: dict) -> list[str]:
    """Where a walk's block differs from a rung block of the look ladder's ``screen.json``: every key
    ``screen_rung`` makes is compared exactly but ``speed``, ``variant``, ``seconds`` and ``arrays.file``; the
    block's added keys (``name``, ``kind``, ``reference``, ``range``) and the ladder's ``vs_V0`` (added by
    ``screen.assemble``) are left out, and the block's ``verdict.ladder_text`` is compared with the rung's
    ``verdict`` sentence.  An empty list: the walk is the ladder's rung, priced to the same numbers."""
    out: list = []
    skip = set(WALK_ADDED) | set(LADDER_NOT_COMPARED) | set(LADDER_ADDED) | {"verdict", "arrays"}
    a = {k: v for k, v in block.items() if k not in skip}
    b = {k: v for k, v in rung.items() if k not in skip}
    _same(a, b, "", out)
    _same({k: v for k, v in block["arrays"].items() if k != "file"},
          {k: v for k, v in rung["arrays"].items() if k != "file"}, ".arrays", out)
    _same(block["verdict"].get("ladder_text"), rung["verdict"], ".verdict", out)
    return out


# ------------------------------------------------------------------------------------------------
# A set's screen
# ------------------------------------------------------------------------------------------------
def _partial(d: Path, name: str) -> Path:
    return d / PARTIAL / f"screen__{name}.json"


def _entries(docs: dict) -> dict:
    """{name: (kind, entry)} of the two records, walks first, in the records' order; ``S2Error`` for a name that is
    in both."""
    out = {}
    for name, e in docs["refs.json"]["references"].items():
        out[name] = (WALK, e)
    for name, e in docs["transitions.json"]["transitions"].items():
        if name in out:
            raise s2io.S2Error(f"{name!r} names a walk and a move of the set")
        out[name] = ("move", e)
    return out


def _available(e: dict) -> bool:
    return isinstance(e, dict) and e.get("available") is not False and bool(e.get("file"))


def _look_entry(docs: dict, name: str) -> dict:
    le = (docs["set.json"].get("look_entries") or {}).get(name)
    if le is None:
        raise s2io.S2Error(f"set.json holds no look entry for the walk {name!r}")
    e = docs["refs.json"]["references"][name]
    if e.get("look_sha256") not in (None, le["sha256"]):
        raise s2io.S2Error(f"{name}: refs.json records the look-format file {str(e['look_sha256'])[:12]}..., the "
                           f"look entry {le['sha256'][:12]}...")
    return le


def walk_of(entry: dict) -> str | None:
    """The name of a move's own walk: gait a (``from_``), or gait b (``to``) for a start; None if neither is a walk."""
    a, b = entry.get("from_"), entry.get("to")
    return a if a not in (None, "stand") else (b if b not in (None, "stand") else None)


def _current(kind: str, name: str, entry: dict, docs: dict, blocks: dict) -> bool:
    """Whether ``blocks[name]`` is the screen of the entry's file as it is now."""
    b = blocks.get(name)
    if b is None:
        return False
    if kind == WALK:  # read on the look-format file; names the Stage 1-format file the bar plays
        return (b["arrays"]["sha256"] == _look_entry(docs, name)["sha256"]
                and _get(b, ("reference", "sha256")) == entry["sha256"])
    if b["arrays"]["sha256"] != entry["sha256"]:
        return False
    w = blocks.get(walk_of(entry) or "")
    seen = _get(b, ("vs_walk", "walk", "sha256"))
    return seen == (w["arrays"]["sha256"] if w is not None and w.get("kind") == WALK else None)


def _lists(doc: dict) -> None:
    blocks = doc["blocks"]
    doc["over"] = [dict(name=n, kind=b["kind"], over=b["verdict"]["over"]) for n, b in blocks.items()
                   if b["verdict"]["over"]]
    doc["a1_over"] = [n for n, b in blocks.items() if b["verdict"]["a1_over"]]
    doc["blocks_freeze"] = {n: b["verdict"]["blocks_freeze"] for n, b in blocks.items() if b["verdict"]["blocks_freeze"]}


def status(set_name: str, body: str, root=setio.ROOT) -> dict:
    """What a run would do, read now: ``{current, missing, stale, not_available, extra}`` (names; ``extra``: blocks
    of names the set no longer lists).  A move whose walk will be screened again is stale with it."""
    d = setio.set_dir(set_name, body, root)
    docs = setio.load(d)
    blocks = dict((docs["screen.json"] or {}).get("blocks", {}))
    ents = _entries(docs)
    out = dict(current=[], missing=[], stale=[], not_available=[], extra=[n for n in blocks if n not in ents])
    for phase in (WALK, "move"):
        for name, (kind, e) in ents.items():
            if kind != phase:
                continue
            if not _available(e):
                out["not_available"].append(name)
                continue
            key = "current" if _current(kind, name, e, docs, blocks) else ("stale" if name in blocks else "missing")
            out[key].append(name)
            if kind == WALK and key != "current":  # after the run its block is of the look entry's file
                blocks[name] = dict(kind=WALK, arrays=dict(sha256=_look_entry(docs, name)["sha256"]))
    return out


def _job(args) -> tuple:
    """One reference in this process: the block to ``partial/screen__<name>.json``; (name, seconds, refusal)."""
    set_name, body, name, root = args
    s2io.require_one_thread()
    d = setio.set_dir(set_name, body, root)
    docs = setio.load(d)
    kind, entry = _entries(docs)[name]
    src = look._source_cached(body)
    t0 = time.perf_counter()
    try:
        if kind == WALK:
            block = screen_walk(src, d, name, _look_entry(docs, name), reference=entry)
        else:
            w = (docs["screen.json"] or {}).get("blocks", {}).get(walk_of(entry) or "")
            block = screen_move(src, d, name, entry, w if w is not None and w.get("kind") == WALK else None)
    except Refused as exc:
        return name, time.perf_counter() - t0, str(exc)
    seconds = time.perf_counter() - t0
    s2io.write_json(_partial(d, name), dict(name=name, block=block, seconds=seconds))
    return name, seconds, None


def _start_worker() -> None:
    s2io.require_one_thread()


def _session(prov: dict) -> dict:
    return dict(at=prov["at"], commit=prov["commit"], dirty=prov["dirty"],
                stage2_modules_sha256=s2io.sha256_json(prov["stage2_modules"]),
                panel_sources_sha256=prov["panel_sources_sha256"], blocks=[])


def run(set_name: str, body: str, root=setio.ROOT, procs: int = 2, log=print) -> dict:
    """Screen every reference of a set that has no current block and merge the blocks into the set's
    ``screen.json`` (module docstring, section 4); ``SCREEN.md`` is written with it.  Resumable per reference: a
    block a worker finished is kept in ``partial/`` until it is merged.  ``procs`` spawned workers (0: the jobs run
    in this process, for tests).  Returns ``{path, doc, ran, kept, recovered, dropped, refused, seconds}``; ``doc``
    is the document as it stands."""
    s2io.require_one_thread()
    d = setio.set_dir(set_name, body, root)
    if not d.is_dir():
        raise tune.InputError(f"{d}: no such set directory")
    with s2io.Lock(d / ".lock", log=log):
        for f in s2io.sweep_tmp(d):
            log(f"removed a stale temporary file: {f}")
        docs = setio.load(d)
        ents = _entries(docs)
        old = docs["screen.json"]
        if old is not None and (old.get("set"), old.get("body"), old.get("schema")) != (set_name, body, SCHEMA):
            raise s2io.S2Error(f"{d / SCREEN_JSON} is the screen of {old.get('set')} {old.get('body')} (schema "
                               f"{old.get('schema')}), not of {set_name} {body} (schema {SCHEMA})")
        blocks = dict((old or {}).get("blocks", {}))
        sessions = [dict(s) for s in (old or {}).get("sessions", [])]
        res = dict(path=str(d / SCREEN_JSON), doc=old, ran=[], kept=[], recovered=[], dropped=[], refused={},
                   seconds={})
        st = dict(changed=False, prov=None, session=None, src=None)
        for name in [n for n in blocks if n not in ents or not _available(ents[n][1])]:
            del blocks[name]
            res["dropped"].append(name)
            st["changed"] = True

        def provenance() -> dict:
            if st["prov"] is None:
                with s2io.r2():
                    st["prov"] = s2io.built_by(list(MODULES), own_outputs=[str(d / SCREEN_JSON), str(d / SCREEN_MD)])
            return st["prov"]

        def write() -> None:
            if st["src"] is None:
                st["src"] = look._source_cached(body)
            doc = dict(
                set=set_name, body=body, schema=SCHEMA,
                what="Stage 2 round A: the Stage 0 screen of every reference of the set, walks and moves "
                     "(STAGE2_75_PLAN.md A12)",
                note="prescribed references that the joints follow exactly, on the round-two model; nothing adopted",
                rules=RULES, limits=screen.limits(st["src"]),
                margins=dict(peak=sizing.PEAK_MARGIN, speed=sizing.SPEED_MARGIN, cont=sizing.CONT_MARGIN,
                             near_band=NEAR_BAND),
                grips=list(GRIPS), specs=dict(sway_H=screen.SWAY_SPEC_H, start_stop_accel_ms2=ACCEL_RULE_M_S2),
                blocks={n: blocks[n] for n in ents if n in blocks})
            _lists(doc)
            doc["not_available"] = {n: e.get("reason") for n, (_, e) in ents.items() if not _available(e)}
            doc["refused"] = dict(res["refused"])
            live = [dict(s, blocks=[n for n in s["blocks"] if n in blocks]) for s in sessions]
            doc["sessions"] = [s for s in live if s["blocks"]]
            doc["provenance"] = st["prov"] or (old or {}).get("provenance") or provenance()
            s2io.write_json(d / SCREEN_JSON, doc)
            res["doc"] = s2io.plain(doc)

        def merge(name: str, part: dict, how: str) -> None:
            if st["session"] is None:
                st["session"] = _session(provenance())
                sessions.append(st["session"])
            for s in sessions:
                s["blocks"] = [n for n in s["blocks"] if n != name]
            st["session"]["blocks"].append(name)
            blocks[name] = part["block"]
            write()
            _partial(d, name).unlink(missing_ok=True)
            st["changed"] = True
            res[how].append(name)
            block, seconds = part["block"], part.get("seconds")
            if seconds is not None:
                res["seconds"][name] = float(seconds)
                _log_time(d, set_name, body, name, block, float(seconds), how)
            g = block["governing"]
            log(f"{set_name} {body} {name} ({block['kind']}): governing {g[0]} {g[1]} {g[2]:.4f}; "
                f"{_verdict_cell(block)}" + ("" if seconds is None else f"; {seconds:.0f} s")
                + ("" if how == "ran" else " (recovered from partial/)"))

        # blocks a worker finished before an earlier run was cut off
        pdir = d / PARTIAL
        for p in sorted(pdir.glob("screen__*.json")) if pdir.is_dir() else []:
            try:
                part = s2io.read_json(p)
                name = part["name"]
                ok = name in ents and _available(ents[name][1]) and p == _partial(d, name)
                ok = ok and _current(ents[name][0], name, ents[name][1], docs, {**blocks, name: part["block"]})
            except (ValueError, KeyError, TypeError, s2io.S2Error):
                ok = False
            if ok:
                merge(name, part, "recovered")
            else:
                p.unlink(missing_ok=True)
                log(f"removed a partial block that is not of the set as it is now: {p.name}")
        for phase in (WALK, "move"):
            todo = []
            for name, (kind, e) in ents.items():
                if kind != phase or not _available(e):
                    continue
                if _current(kind, name, e, docs, blocks):
                    if name not in res["recovered"]:
                        res["kept"].append(name)
                    continue
                if name in blocks:  # stale: its file, or its walk's, is another now
                    del blocks[name]
                    res["dropped"].append(name)
                    st["changed"] = True
                todo.append((set_name, body, name, str(root)))
            if not todo:
                continue
            if st["changed"]:
                write()  # a move's worker reads its walk's block from screen.json: no stale block may be there
            log(f"{set_name} {body}: screening {len(todo)} {'walk' if phase == WALK else 'move'}(s) with "
                f"{min(max(1, procs), len(todo))} process(es): " + ", ".join(j[2] for j in todo))
            if procs >= 1:
                import multiprocessing as mp

                with mp.get_context("spawn").Pool(min(procs, len(todo)), initializer=_start_worker) as pool:
                    for name, seconds, refusal in pool.imap_unordered(_job, todo, chunksize=1):
                        _finish(d, name, seconds, refusal, res, merge, log)
            else:
                for j in todo:
                    _finish(d, *_job(j), res, merge, log)
        if st["changed"] or res["refused"] != (old or {}).get("refused", {}):
            write()
        doc = res["doc"]
        if doc is not None:
            text = markdown(doc)
            md = d / SCREEN_MD
            if not md.is_file() or md.read_text() != text:
                _write_text(md, text)
        return res


def _finish(d: Path, name: str, seconds: float, refusal: str | None, res: dict, merge, log) -> None:
    if refusal is not None:
        res["refused"][name] = refusal
        log(f"refused: {refusal}")
        return
    merge(name, s2io.read_json(_partial(d, name)), "ran")


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _log_time(d: Path, set_name: str, body: str, name: str, block: dict, seconds: float, how: str) -> None:
    """Wall times go to the set's ``logs/`` (git-ignored), never into ``screen.json``."""
    path = d / LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{s2io.now_utc()} {set_name} {body} {name} kind={block['kind']} frames={block['arrays']['frames']} "
                 f"seconds={seconds:.1f} {how}\n")


# ------------------------------------------------------------------------------------------------
# SCREEN.md
# ------------------------------------------------------------------------------------------------
def _f(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{nd}f}"


def _verdict_cell(b: dict) -> str:
    v = b["verdict"]
    if v["blocks_freeze"]:
        return "blocks the freeze: " + "; ".join(_item(o) for o in v["blocks_freeze"])
    if v["a1_over"]:
        return "a1_over: " + "; ".join(_item(o) for o in v["over"])
    if v["over"]:
        return "over, reported: " + "; ".join(_item(o) for o in v["over"])
    return "within the Stage 0 limits"


def _worst(b: dict, groups) -> tuple[str, str, float] | None:
    rows = [r for r in _drive_rows(b["drives"]) if r[0] in groups]
    return max(rows, key=lambda r: r[2]) if rows else None


def _ratio_cell(row) -> str:
    return "-" if row is None else f"{row[2]:.4f} ({GROUP_LABEL.get(row[0], row[0])} {row[1]})"


def _triple(b: dict, g: str) -> str:
    d = b["drives"].get(g)
    return "-" if d is None else " / ".join(f"{d[k]:.4f}" for k in KINDS)


def _cont_pair(b: dict, g: str) -> str:
    """A group's sustained ratio over time and by frames (a walk's block holds the one number: its frames are
    uniform)."""
    d = b["drives"].get(g)
    return "-" if d is None else f"{d['cont_ratio']:.4f} / {d.get('cont_ratio_frames', d['cont_ratio']):.4f}"


def _margin_cell(tight) -> str:
    return "-" if tight is None else f"{tight[2]:.4f} ({tight[0]} {tight[1]})"


def _range_cells(b: dict) -> tuple[str, str]:
    """The table's two joint-range cells: the margin the 0.10 rad gate reads (a turn: the tightest joint other than
    the thigh twist) and the thigh-twist use (a turn: with the twist's own tightest margin beside it)."""
    rng = b["range"]
    use = _f(rng.get("hip_yaw_range_use"))
    if b["kind"] != TURN:
        return _margin_cell(rng.get("tightest")), use
    twist = rng.get("tightest_twist")
    return _margin_cell(rng.get("tightest_other")), use + ("" if twist is None else
                                                           f" (margin {twist[2]:.4f} rad, {twist[0]} {twist[1]})")


def _accel_cell(b: dict) -> str:
    """The 1 Hz forward acceleration, marked where the peak is the filter's value at an end of the file."""
    a = b["accel"]
    if not a.get("com_forward_lp1hz_peak_at_end"):
        return _f(a.get("com_forward_lp1hz_peak"))
    inner = a.get("com_forward_lp1hz_peak_interior")
    return (f"{_f(a['com_forward_lp1hz_peak'])} (at the file's end; "
            + ("the file is under 2 s" if inner is None else f"{inner:.4f} at least {a['com_forward_lp1hz_edge_s']:g} "
               "s from both ends") + ")")


def markdown(doc: dict) -> str:
    """``SCREEN.md`` from a ``screen.json`` document: one table of every reference, then what blocks a freeze, the
    turns with ``a1_over`` and what is reported, each with its numbers."""
    blocks = doc["blocks"]
    L = [f"# Stage 0 screen of set {doc['set']} on {doc['body']}", "",
         "Simulation only: every reference is priced as a prescribed motion that the joints follow exactly, on the "
         "round-two model. Made by "
         f"`python -m designlab.stage2.setscreen md --set {doc['set']} --body {doc['body']}` from `screen.json`; "
         "the rules are in its `rules` block (STAGE2_75_PLAN.md A12).", "",
         "A ratio is a load over its limit: cont = RMS torque over time / sustained torque (each frame weighted by "
         "its duration; on a walk, whose frames are uniform in time, that is the RMS over the frames), peak = "
         f"{doc['margins']['peak']:g} x peak torque / planning peak, speed = {doc['margins']['speed']:g} x peak speed "
         "/ speed at the bus. A ratio above 1 is over the limit. Ratios and margins are printed to four decimals; "
         "the verdict is read on the unrounded numbers in `screen.json`.", "",
         "## Limits", "",
         "| drive | actuator | sustained (N m) | planning peak (N m) | speed at the bus (rad/s) | bus (V) |",
         "|---|---|---|---|---|---|"]
    for g, r in doc["limits"].items():
        L.append(f"| {GROUP_LABEL.get(g, g)} | {r['actuator']} | {r['sustained_nm']:.4f} | {r['peak_nm']:.4f} | "
                 f"{r['speed_rad_s']:.4f} | {_f(r['bus_v'], 1)} |")
    leg = [g for g in doc["limits"] if g not in A1_GROUPS]
    L += ["", "## Every reference", "",
          "Columns: the governing ratio (largest of all drives); the worst ratio of the drives other than the neck's "
          "and the tail's (the legs'; on a body with the expressive head, its two drives as well); neck and tail "
          "ratios (cont / peak / speed); the single-support CoP margin (mm inside the pad; negative: outside) and "
          "the rule it is read against (mm); the single-support grip need mu_req and whether it is within mu 0.9 and "
          "mu 0.54; the mid-swing pad clearance (H; the gate is 0.04 H); the joint-range margin that the 0.10 rad "
          "gate reads (rad): the tightest gated joint, and for a turn the tightest joint other than the thigh twist; "
          "the thigh-twist use of the usable range (a turn's gate is 0.87; for a turn the twist's own tightest margin "
          "is beside it); the forward CoM acceleration low-passed at 1 Hz (m/s^2; the gate of a start, change or "
          "stop is 1.5 m/s^2), read on the file alone: a peak within 1 s of the file's first or last frame is the "
          "filter's value at that end and is marked, with the largest value away from the ends beside it.", "",
          "| reference | kind | frames | governing ratio | worst ratio but neck and tail | neck c / p / s | tail c / "
          "p / s | CoP margin (mm) | CoP rule (mm) | grip need mu_req | within mu 0.9 | within mu 0.54 | mid-swing "
          "clearance (H) | gated joint margin (rad) | thigh twist use | 1 Hz forward accel (m/s^2) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in blocks.items():
        g = b["governing"]
        grips = b["friction"]["grips"]
        margin, twist = _range_cells(b)
        L.append(
            f"| `{name}` | {b['kind']} | {b['arrays']['frames']} | {g[2]:.4f} ({GROUP_LABEL.get(g[0], g[0])} {g[1]}) | "
            f"{_ratio_cell(_worst(b, leg))} | {_triple(b, 'neck')} | {_triple(b, 'tail')} | "
            f"{_f(b['contact']['single_support_cop_margin_min_mm'])} | {_f(b['contact']['cop_rule_mm'], 0)} | "
            f"{_f(b['friction']['single_support']['mu_req_max'])} | {_f(grips.get('0.9'))} | {_f(grips.get('0.54'))} | "
            f"{_f(b['clearance']['mid_swing_min_frac_H'])} | {margin} | {twist} | {_accel_cell(b)} | "
            f"{_verdict_cell(b)} |")
    L += ["", "## What blocks a freeze", ""]
    if doc["blocks_freeze"]:
        for name, items in doc["blocks_freeze"].items():
            L.append(f"- `{name}` ({blocks[name]['kind']}): " + "; ".join(_item(o) for o in items))
    else:
        L.append("No screened reference is over a gate that blocks its freeze.")
    L += ["", "## Turns over a neck or tail limit (`a1_over`)", ""]
    turns = [n for n in doc["a1_over"] if blocks[n]["kind"] == TURN]
    if turns:
        L.append("Frozen with `a1_over` and read under the plan's third reading only:")
        L.append("")
        for name in turns:
            b = blocks[name]
            a1 = [o for o in b["over"] if o["limit"].split()[0] in A1_GROUPS]
            L.append(f"- `{name}`: " + "; ".join(_item(o) for o in a1) + f"; neck c / p / s {_triple(b, 'neck')}; "
                     f"tail c / p / s {_triple(b, 'tail')}; thigh twist use {_range_cells(b)[1]}; gated joint margin "
                     f"{_range_cells(b)[0]}")
    else:
        L.append("No screened turn is over a neck or tail planning limit.")
    other = [n for n in doc["a1_over"] if blocks[n]["kind"] != TURN]
    if other:
        L += ["", "Walks and straight moves over a neck or tail limit (it blocks their freeze): "
              + ", ".join(f"`{n}`" for n in other) + "."]
    L += ["", "## Reported, not gated", "",
          "The frames of a move are not uniform in time where its stride period changes; the cont ratio by frames "
          "(every frame the same weight) is beside the gated one, which is over time.", "",
          "| reference | frames uniform in time | neck cont over time / by frames | tail cont over time / by frames | "
          "neck with passive c / p | tail with passive c / p | double-support min mu | sway per stride "
          "(H; spec 0.12 H) | CoM horizontal accel peak (m/s^2) | near a limit (within 0.02 below 1) | record "
          "re-check equal |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in blocks.items():
        p = b["passive"]
        chk = b["reprice"]["equal_to_recorded"] if b["kind"] == WALK else b["recheck"]["equal_to_recorded"]
        L.append(f"| `{name}` | {_f(True if b['kind'] == WALK else b['arrays'].get('frames_uniform'))} | "
                 f"{_cont_pair(b, 'neck')} | {_cont_pair(b, 'tail')} | "
                 f"{p['neck']['cont_ratio_total']:.4f} / {p['neck']['peak_ratio_total']:.4f} | "
                 f"{p['tail']['cont_ratio_total']:.4f} / {p['tail']['peak_ratio_total']:.4f} | "
                 f"{_f(b['friction']['double_support']['min_mu_max'])} | {_f(b['sway']['com_pp_per_stride_max_H'])} | "
                 f"{_f(b['accel']['com_horizontal_peak'])} | "
                 f"{'; '.join(_item(o) for o in b['near'] + b['near_with_passive']) or '-'} | {_f(chk)} |")
    over_p = {n: b["over_with_passive"] for n, b in blocks.items() if b["over_with_passive"]}
    if over_p:
        L += ["", "Over a neck or tail planning limit once the plant's joint damping and friction are added (outside "
              "Stage 0's rule):", ""]
        L += [f"- `{n}`: " + "; ".join(_item(o) for o in items) for n, items in over_p.items()]
    if doc.get("not_available"):
        L += ["", "## Not available", ""]
        L += [f"- `{n}`: {r or 'no reason recorded'}" for n, r in doc["not_available"].items()]
    if doc.get("refused"):
        L += ["", "## Not screened", ""]
        L += [f"- {r}" for r in doc["refused"].values()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    s2io.require_one_thread()
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.setscreen",
                                 description="The Stage 0 screen of every reference of a set")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, text in (("run", "screen what has no current block; screen.json and SCREEN.md"),
                      ("md", "SCREEN.md again from screen.json")):
        p = sub.add_parser(cmd, help=text)
        p.add_argument("--set", required=True, dest="set_name")
        p.add_argument("--body", required=True)
        p.add_argument("--root", type=Path, default=setio.ROOT, help="the sets' directory (default stage2/sets)")
        if cmd == "run":
            p.add_argument("--procs", type=int, default=2)
    a = ap.parse_args(argv)
    d = setio.set_dir(a.set_name, a.body, a.root)
    try:
        if a.cmd == "md":
            path = d / SCREEN_JSON
            if not path.is_file():
                print(f"error: {path} does not exist (run the screen first)", file=sys.stderr)
                return 2
            _write_text(d / SCREEN_MD, markdown(s2io.read_json(path)))
            print(f"wrote {d / SCREEN_MD}")
            return 0
        res = run(a.set_name, a.body, root=a.root, procs=a.procs, log=lambda line: print(line, flush=True))
    except (s2io.S2Error, tune.InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    doc = res["doc"] or dict(blocks={}, blocks_freeze={}, a1_over=[])
    print(f"{a.set_name} {a.body}: {len(res['ran'])} screened, {len(res['recovered'])} recovered, "
          f"{len(res['kept'])} kept, {len(res['dropped'])} dropped, {len(res['refused'])} refused; "
          f"{len(doc['blocks'])} blocks; {len(doc['blocks_freeze'])} block a freeze; a1_over: "
          f"{', '.join(doc['a1_over']) or 'none'}")
    return 1 if res["refused"] else 0


if __name__ == "__main__":
    from designlab.stage2 import setscreen as _named  # the importable copy owns every function and all module state

    sys.exit(_named.main())
