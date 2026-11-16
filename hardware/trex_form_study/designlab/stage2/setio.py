"""Reading a Stage 2 reference set: names, entries, freezes, command events and the track of a plan.

A reference set is a directory ``stage2/sets/<set>/<body>/`` in Stage 1's run-directory layout (``refs/*.npz``,
``transitions/*.npz``, ``refs.json``, ``transitions.json``) plus ``set.json`` (definition, rules, freezes) and
``screen.json``.  This module only reads one; ``sets.py`` builds it.  It is hashed into every bar record's key, so it
holds nothing a record does not execute:

* **Names** are made and parsed here only, numbers with ``%g``: walks ``nominal`` and ``v0.4``; the start ``start``
  (``start_v0.4``: a start from standing straight into another walk, for a set whose nominal walk is no rung);
  changes ``nominal_to_v0.4``; stops ``stop2_v0.4`` (two touchdowns after the command) and ``stop1_v0.4``; turns
  ``turn90_<rate>_<left|right>`` with the rate's label from ``TURN_RATES`` (``turn90_pi4_left`` is the quick turn at
  pi/4 rad/s).  No name holds a rounded value: ``parse_name`` maps every name back to its values exactly, and a
  maker refuses a value its name would round.
* **Entry subsets.**  A bar record's key holds, per input file, the file's sha256 and ``entry_sha256``: the hash of
  the entry's keys an episode reads (``WALK_SUBSET``, ``MOVE_SUBSET``; a tuple names a nested key).  A label, a rule
  text, ``built_by`` or ``look_sha256`` is not among them and changes no key.
* **``read_inputs``** checks the files a run will load against their entries and the set's latest freeze
  (``set.json`` ``freezes[-1].manifest``) and returns ``tune.read_inputs``' two values, so that ``panels.Planner``,
  ``panels.Context`` and ``tune.build_spec`` take the documents unchanged.  ``stage1_view`` gives the same for a
  Stage 1 run directory (the pseudo-set ``S1``: no freeze).
* **Commands.**  A causal move's file begins at its command instant, so the command of a move on a track is its
  segment's ``t0``.  ``add_events`` appends one event per causal move to ``track.events``, in the shape of Stage 1's
  own stop event (``control.Track.add_transition``); ``control.run`` reads the events of kind ``"stop"`` for decision
  18's reading.  No transition of a set may carry a name by which Stage 1 registers its own stop event
  (``stop``, ``stop_in_place_*``): ``commands`` refuses one.
* **``min_planned_swing_s``**: the shortest planned swing of a track, which sets the bar's air-gap threshold.

Imports ``designlab.stage1.control``, ``designlab.stage1.tune`` and ``s2io`` only; never ``sets``, ``moves`` or
``bar``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..stage1 import control, tune
from . import s2io

ROOT = s2io.STUDY / "stage2" / "sets"
#: The speed ladder's rungs above the nominal walk (m/s), the bar's speed and the turn ramps' length (s of
#: reference time, for every set, body and rate).
SPEEDS = (0.30, 0.35, 0.40, 0.45, 0.50)
BAR_SPEED = 0.40
RAMP_S = 0.375
#: Turn rates by their label in a name (rad/s); the quick turn's label.
TURN_RATES = {"0.35": 0.35, "0.5": 0.5, "pi4": math.pi / 4, "1": 1.0}
QUICK = "pi4"

NOMINAL = "nominal"
START = "start"
DIRECTIONS = ("left", "right")
STOP_STEPS = (1, 2)
TURN_ANGLE_DEG = 90
TURN_ANGLE_RAD = math.pi / 2

#: The keys of an entry that ``entry_sha256`` hashes, where the entry has them.
WALK_SUBSET = ("file", "sha256", "frames", "dt_s", "period_s", "speed_m_s", "stride_m", "yaw_rate_rad_s", "strides",
               "window_s", "crouch_frac")
MOVE_SUBSET = ("file", "sha256", "frames", "duration_s", "kind", ("gaits", "a"), ("gaits", "b"), ("move", "causal"),
               ("move", "command_phase"), ("move", "kind"), ("turn", "angle_rad"), ("turn", "rate_rad_s"),
               ("turn", "direction"), "a1_over")
#: The entry kinds ``entry_subset`` takes: a periodic walk (``refs.json``) or a move (``transitions.json``).
WALK_KINDS = ("walk", "ref")
MOVE_KINDS = ("move", "transition", "start", "change", "stop", "stop_one_step", "stop_two_step", "turn")
#: (JSON file, key of its entries, subdirectory of its arrays, entry kind).
RECORDS = (("refs.json", "references", "refs", "walk"), ("transitions.json", "transitions", "transitions", "move"))


# ------------------------------------------------------------------------------------------------
# Names
# ------------------------------------------------------------------------------------------------
def _walk_speed(name: str) -> float | None:
    """The speed a walk's name holds (None: the nominal walk); ``ValueError`` for any other string."""
    if name == NOMINAL:
        return None
    if isinstance(name, str) and re.fullmatch(r"v\d+(\.\d+)?", name):
        v = float(name[1:])
        if v > 0 and f"v{v:g}" == name:
            return v
    raise ValueError(f"{name!r} is not a walk's name")


def walk_name(v: float | None) -> str:
    """``"nominal"`` for None, else ``f"v{v:g}"`` (``v0.4``).  A speed that ``%g`` would round is refused."""
    if v is None:
        return NOMINAL
    v = float(v)
    name = f"v{v:g}"
    try:
        ok = _walk_speed(name) == v
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"speed {v!r} has no exact name ({name!r} does not map back to it)")
    return name


def start_name(walk: str = NOMINAL) -> str:
    """The start from standing into ``walk``: ``start`` into the nominal walk, ``start_v0.4`` into another (a set
    whose nominal walk is no rung starts straight into its 0.40 m/s walk: ``STAGE2_75_PLAN.md`` A8 as amended on
    15 Nov, ``DECISION_LOG.md`` P34)."""
    return START if _walk_speed(walk) is None else f"{START}_{walk}"


def change_name(a: str, b: str) -> str:
    """A speed change from walk ``a`` to walk ``b``: ``nominal_to_v0.4``."""
    _walk_speed(a), _walk_speed(b)
    if a == b:
        raise ValueError(f"a change from {a!r} to itself")
    return f"{a}_to_{b}"


def stop_name(steps: int, walk: str) -> str:
    """The stop from ``walk`` with ``steps`` touchdowns after its command: ``stop2_v0.4``, ``stop1_v0.4``."""
    _walk_speed(walk)
    if isinstance(steps, bool) or steps not in STOP_STEPS:
        raise ValueError(f"a stop has {' or '.join(str(s) for s in STOP_STEPS)} steps, not {steps!r}")
    return f"stop{int(steps)}_{walk}"


def turn_name(rate: str, direction: str) -> str:
    """The 90 degree turn at the rate labelled ``rate`` in ``TURN_RATES``: ``turn90_pi4_left``."""
    if rate not in TURN_RATES:
        raise ValueError(f"turn rate label {rate!r}: choose from {', '.join(TURN_RATES)}")
    if direction not in DIRECTIONS:
        raise ValueError(f"turn direction {direction!r}: choose from {', '.join(DIRECTIONS)}")
    return f"turn{TURN_ANGLE_DEG}_{rate}_{direction}"


def parse_name(name: str) -> dict:
    """The exact inverse of the makers: ``{kind: "walk", speed_m_s}`` (None: nominal), ``{kind: "start"}``
    (``start_v0.4``: with ``walk``), ``{kind: "change", a, b}``, ``{kind: "stop", steps, walk}`` or ``{kind: "turn",
    angle_deg, angle_rad, rate, rate_rad_s, direction, sign}`` (sign +1 left, -1 right), each with ``name``.
    ``ValueError`` for a name the
    makers did not make."""
    if not isinstance(name, str):
        raise ValueError(f"{name!r} is not a name")
    out = None
    if name == START:
        out = dict(kind="start", name=name)
    elif name.startswith(START + "_") and "_to_" not in name:
        out = dict(kind="start", name=name, walk=name[len(START) + 1:])
        made = start_name(out["walk"])
    elif "_to_" in name:
        a, _, b = name.partition("_to_")
        out = dict(kind="change", name=name, a=a, b=b)
        made = change_name(a, b)
    elif m := re.fullmatch(r"stop(\d)_(.+)", name):
        out = dict(kind="stop", name=name, steps=int(m.group(1)), walk=m.group(2))
        made = stop_name(out["steps"], out["walk"])
    elif m := re.fullmatch(rf"turn{TURN_ANGLE_DEG}_([^_]+)_([a-z]+)", name):
        rate, direction = m.group(1), m.group(2)
        made = turn_name(rate, direction)
        out = dict(kind="turn", name=name, angle_deg=TURN_ANGLE_DEG, angle_rad=TURN_ANGLE_RAD,
                   rate=rate, rate_rad_s=TURN_RATES[rate], direction=direction, sign=1.0 if direction == "left" else -1.0)
    else:
        out = dict(kind="walk", name=name, speed_m_s=_walk_speed(name))
    if (out["kind"] in ("change", "stop", "turn") or "walk" in out and out["kind"] == "start") and made != name:
        raise ValueError(f"{name!r} is not a name this module makes")
    return out


# ------------------------------------------------------------------------------------------------
# Entries and their hashed subsets
# ------------------------------------------------------------------------------------------------
def entry_subset(kind: str, entry: dict) -> dict:
    """The keys of ``WALK_SUBSET`` (``kind`` in ``WALK_KINDS``) or ``MOVE_SUBSET`` (``MOVE_KINDS``) that ``entry``
    has, as a plain copy; a tuple names a nested key and is returned nested (``{"gaits": {"a": ..}}``)."""
    if kind in WALK_KINDS:
        keys = WALK_SUBSET
    elif kind in MOVE_KINDS:
        keys = MOVE_SUBSET
    else:
        raise ValueError(f"entry kind {kind!r}: choose from {', '.join(WALK_KINDS + MOVE_KINDS)}")
    out: dict = {}
    for key in keys:
        if isinstance(key, tuple):
            node = entry
            for part in key:
                if not (isinstance(node, dict) and part in node):
                    break
                node = node[part]
            else:
                out.setdefault(key[0], {})[key[1]] = s2io.plain(node)
        elif key in entry:
            out[key] = s2io.plain(entry[key])
    return out


def entry_sha256(kind: str, entry: dict) -> str:
    """``s2io.sha256_json`` of ``entry_subset``: what a bar record's key holds of an entry."""
    return s2io.sha256_json(entry_subset(kind, entry))


# ------------------------------------------------------------------------------------------------
# Reading a set
# ------------------------------------------------------------------------------------------------
def _anchored(path) -> Path:
    """A relative path is relative to the study directory, never to the working directory."""
    p = Path(path)
    return p if p.is_absolute() else s2io.STUDY / p


def set_dir(set_name: str, body: str, root=ROOT) -> Path:
    for part in (set_name, body):
        if not part or Path(part).name != part or part in (".", ".."):
            raise ValueError(f"{part!r} is not a set or body name")
    return _anchored(root) / set_name / body


def load(set_dir) -> dict:
    """The set's documents as read: ``{"set.json", "refs.json", "transitions.json", "screen.json"}``;
    ``screen.json`` is None when the set has not been screened yet.  ``tune.InputError`` if one of the other three
    is missing."""
    d = _anchored(set_dir)
    out = {}
    for name in ("set.json", "refs.json", "transitions.json", "screen.json"):
        path = d / name
        if path.is_file():
            out[name] = s2io.read_json(path)
        elif name == "screen.json":
            out[name] = None
        else:
            raise tune.InputError(f"{d}: {name} is missing")
    return out


def frozen(set_doc: dict) -> dict | None:
    """The latest freeze of a ``set.json``: ``{index, manifest_sha256, entries {name: {kind, sha256,
    entry_sha256}}}``, or None for a set that was never frozen."""
    freezes = set_doc.get("freezes") or []
    if not freezes:
        return None
    last = freezes[-1]
    return dict(index=last["index"], manifest_sha256=last["manifest_sha256"], entries=last["manifest"])


def _listed(docs: dict) -> dict:
    """{file: (JSON file, entry kind, name, entry)} of every entry of the two records that names a file."""
    out = {}
    for record, key, _, kind in RECORDS:
        for name, e in docs[record][key].items():
            if isinstance(e, dict) and e.get("file"):
                out[e["file"]] = (record, kind, name, e)
    return out


def _check_file(d: Path, f: str, listed: dict, bad: list) -> tuple | None:
    """One input file against its entry: (kind, name, entry, sha256), or None with the reason appended to ``bad``."""
    where = listed.get(f)
    if where is None:
        bad.append(f"{f} is not listed in refs.json or transitions.json")
        return None
    record, kind, name, e = where
    sub = next(s for r, _, s, _ in RECORDS if r == record)
    if f != f"{sub}/{name}.npz":
        bad.append(f"{f} is listed by {record} under the name {name!r} (expected {sub}/{name}.npz)")
        return None
    if e.get("available") is False:
        bad.append(f"{f}: {record} records {name!r} as not available")
        return None
    if not (d / f).is_file():
        bad.append(f"{f} is missing")
        return None
    got = s2io.sha256_file(d / f)
    if got != e.get("sha256"):
        bad.append(f"{f} has sha256 {got[:12]}..., {record} records {str(e.get('sha256'))[:12]}...")
        return None
    return kind, name, e, got


def read_inputs(set_dir, files, require_frozen: bool = True) -> tuple[dict, dict]:
    """Check ``files`` (paths relative to the set directory: ``refs/v0.4.npz``, ``transitions/start.npz``) and
    return ``({"dir", "files": {f: {"sha256", "entry_sha256"}}, "freeze": {"index", "manifest_sha256"} | None},
    {"refs.json": <document>, "transitions.json": <document>})``: the second value has ``tune.read_inputs``' shape.

    Every file must be there, be listed under its own name and have its entry's sha256.  Against the set's latest
    freeze: a file the manifest lists must have the manifest's ``sha256`` and ``entry_sha256`` (in every case: a
    frozen entry never changes).  With ``require_frozen`` the set must have a freeze and the manifest must list
    every file.  Without it, a set with no freeze, or files the manifest does not list, are accepted and
    ``"freeze"`` is None: it names a freeze only when that freeze covers every file read.  ``tune.InputError`` lists
    every difference."""
    d = _anchored(set_dir)
    docs = load(d)
    fz = frozen(docs["set.json"])
    if fz is None and require_frozen:
        raise tune.InputError(f"{d}: the set is not frozen (set.json has no freeze)")
    listed = _listed(docs)
    bad, out, covered = [], {}, fz is not None
    for f in sorted(files):
        chk = _check_file(d, f, listed, bad)
        if chk is None:
            continue
        kind, name, e, got = chk
        esha = entry_sha256(kind, e)
        m = fz["entries"].get(name) if fz is not None else None
        if m is None:
            covered = False
            if require_frozen:
                bad.append(f"{f} is not in the latest freeze (index {fz['index']})")
        else:
            if (m.get("kind") in WALK_KINDS) != (kind == "walk"):
                bad.append(f"{f}: the freeze lists {name!r} as a {m.get('kind')}")
            if m.get("sha256") != got:
                bad.append(f"{f} has sha256 {got[:12]}..., the freeze (index {fz['index']}) holds "
                           f"{str(m.get('sha256'))[:12]}...")
            if m.get("entry_sha256") != esha:
                bad.append(f"{f}: its entry's hashed subset is {esha[:12]}..., the freeze (index {fz['index']}) holds "
                           f"{str(m.get('entry_sha256'))[:12]}...")
        out[f] = dict(sha256=got, entry_sha256=esha)
    if bad:
        raise tune.InputError(f"{d}: " + "; ".join(bad))
    freeze = dict(index=fz["index"], manifest_sha256=fz["manifest_sha256"]) if covered else None
    return (dict(dir=str(d), files=out, freeze=freeze),
            {"refs.json": docs["refs.json"], "transitions.json": docs["transitions.json"]})


def stage1_view(run_dir, body) -> tuple[dict, dict]:
    """The pseudo-set ``S1``: ``read_inputs``' two values for the Stage 1 run directory ``run_dir/body``
    (``stage1/runs/2026-11-08/75-M-H-10``).  There is no freeze (``"freeze"`` is None); every file ``refs.json``
    and ``transitions.json`` list is hashed and must have its entry's sha256 (``tune.InputError`` otherwise, or for
    a missing file); ``entry_sha256`` is of the same subsets."""
    d = _anchored(run_dir) / body
    docs = {}
    for record, _, _, _ in RECORDS:
        if not (d / record).is_file():
            raise tune.InputError(f"{d}: {record} is missing")
        docs[record] = s2io.read_json(d / record)
    listed = _listed(docs)
    bad, out = [], {}
    for f in sorted(listed):
        chk = _check_file(d, f, listed, bad)
        if chk is not None:
            kind, _, e, got = chk
            out[f] = dict(sha256=got, entry_sha256=entry_sha256(kind, e))
    if bad:
        raise tune.InputError(f"{d}: " + "; ".join(bad))
    return dict(dir=str(d), files=out, freeze=None), docs


# ------------------------------------------------------------------------------------------------
# Commands and tracks
# ------------------------------------------------------------------------------------------------
def commands(plan, tr_doc: dict) -> list[dict]:
    """The causal moves of a plan (``control.build_track``'s items), in order: ``{index, name, kind,
    command_phase, speed_m_s, period_s}`` for each ``("transition", name)`` whose entry in ``tr_doc`` (the
    ``transitions`` of a ``transitions.json``) has ``move.causal``.  ``index`` is the item's place in the plan,
    which is its segment's on the track; ``kind`` is the entry's ``move.kind`` (``"stop"`` or ``"turn"``); speed and
    period are those of the gait the move leaves (``gaits.a``)."""
    out = []
    for i, item in enumerate(plan):
        if item[0] != "transition":
            continue
        name = item[1]
        if name not in tr_doc:
            raise tune.InputError(f"plan item {i}: transition {name!r} is not in transitions.json")
        e = tr_doc[name]
        move = e.get("move") or {}
        if not move.get("causal"):
            continue
        if control.TRANSITIONS.get(name, (None, None))[1] == "stand":
            raise tune.InputError(f"causal move {name!r} carries a name by which Stage 1 registers its own stop "
                                  "command; a set may not use it")
        a = e["gaits"]["a"]
        out.append(dict(index=i, name=name, kind=move["kind"], command_phase=float(move["command_phase"]),
                        speed_m_s=float(a["speed"]), period_s=float(a["period"])))
    return out


def add_events(track: control.Track, plan, tr_doc: dict) -> list[dict]:
    """Append one command event per causal move of ``plan`` to ``track.events`` and return them: ``dict(kind, name,
    t_ref, command_phase, speed_m_s, period_s)``, Stage 1's shape, with ``t_ref`` the move's segment's ``t0`` (a
    causal move's file begins at its command).  ``track`` must be the track of ``plan``; an event that is already
    there is refused (a second call on one track)."""
    plan = list(plan)
    if len(track.segments) != len(plan):
        raise ValueError(f"the track has {len(track.segments)} segments, the plan {len(plan)} items")
    out = []
    for c in commands(plan, tr_doc):
        seg = track.segments[c["index"]]
        if seg.kind != "finite" or seg.name != c["name"]:
            raise ValueError(f"segment {c['index']} of the track is {seg.kind} {seg.name!r}, the plan has the "
                             f"transition {c['name']!r} there")
        ev = dict(kind=c["kind"], name=c["name"], t_ref=float(seg.t0), command_phase=c["command_phase"],
                  speed_m_s=c["speed_m_s"], period_s=c["period_s"])
        if any(x.get("name") == ev["name"] and x.get("t_ref") == ev["t_ref"] for x in track.events):
            raise ValueError(f"the track already holds an event of {ev['name']!r} at {ev['t_ref']:g} s")
        out.append(ev)
    track.events.extend(out)
    return out


def build_track(p, set_dir, plan, omega: float, tr_doc: dict) -> control.Track:
    """``control.build_track(p, set_dir, plan, omega)`` with the plan's command events added (``add_events``)."""
    track = control.build_track(p, _anchored(set_dir), plan, omega)
    add_events(track, plan, tr_doc)
    return track


def planned_swings(track: control.Track) -> list[tuple[int, float, float]]:
    """(foot, start, end) of every planned swing of a track, on the reference clock: the maximal runs of frames
    whose ``swing`` is that foot, each frame lasting from its own time to the next frame's as ``Track.info`` holds
    it, joined across segments.  A run still open at the track's end is left out (it is cut, not planned so)."""
    ends, vals = [], []
    for seg in track.segments:
        sw = np.asarray(seg.info["swing"], int)
        if seg.kind == "periodic":
            frames = int(math.ceil(seg.duration / seg.step - 1e-9))
            k = np.arange(frames)
            ends.append(seg.t0 + np.minimum((k + 1) * seg.step, seg.duration))
            vals.append(sw[k % seg.n])
        elif seg.kind == "finite":
            ends.append(seg.t0 + np.asarray(seg.t, float)[1 : len(sw) + 1])
            vals.append(sw)
        else:
            ends.append(np.array([seg.t0 + seg.duration]))
            vals.append(sw[:1])
    if not ends:
        return []
    ends, vals = np.concatenate(ends), np.concatenate(vals)
    starts = np.r_[track.segments[0].t0, ends[:-1]]
    change = np.flatnonzero(vals[1:] != vals[:-1]) + 1
    a, b = np.r_[0, change], np.r_[change, len(vals)]
    return [(int(vals[i]), float(starts[i]), float(ends[j - 1])) for i, j in zip(a, b, strict=True)
            if vals[i] >= 0 and j < len(vals)]


def min_planned_swing_s(track: control.Track) -> float:
    """The shortest planned swing of the track's segments, in reference seconds (``planned_swings``).  The bar's
    air-gap threshold is half of it on the simulated clock.  ``S2Error`` for a track that plans no swing."""
    swings = planned_swings(track)
    if not swings:
        raise s2io.S2Error("the track plans no complete swing")
    return min(t1 - t0 for _, t0, t1 in swings)
