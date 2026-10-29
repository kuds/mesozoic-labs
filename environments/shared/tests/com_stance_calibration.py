"""Re-measure one species' CoM stance pins (``com_stance_helpers``) and print them as Python.

    python -m environments.shared.tests.com_stance_calibration \\
        environments.tyrannosaurus_rex.tests.test_com_stance_quality:TestComStanceQuality

Runs, one after the other in this process, at the class's horizon: the noise-free statue, the calibration
spawns (seeds 0-29) and the held-out spawns (seeds 100-149) at reset noise 0.02, and the three scripted poor
stances.  It prints the settle time (where the window starts), the statue's ``reference`` and the
``spread_width`` beside the class's current values (so a re-pin shows what moved), the held-out check (how much
of each width every held-out spawn uses, and how many break a bar), the two noisy seeds, and the scripted
stances' detection ratios.  About 85 episodes: one to four minutes per model on one core.  Nothing here runs
under pytest; ``test_stance_track_reduction.py`` checks the pure functions.

The rules, in order:

1. Settle time: the first whole second, from 2 s on, after which the horizontal CoM speed's RMS in every
   whole second to the episode's end stays under 1 mm/s, for the statue and every calibration spawn.  A
   window (horizon less settle) under :data:`MIN_WINDOW_S` calls for twice the ``horizon_steps``.
2. Reference: the statue's window value of every key mirror symmetry does not fix, rounded to 0.01 mm,
   0.01 mm/s, 1e-4 deg or 1e-4 of the load (under 1 % of a bar's width).
3. Spread width: twice the worst deviation (on the bad side of the bar) of the calibration spawns from the
   reference, never under twice the resolution floor, rounded up to two significant figures.
4. Held-out check: a held-out spawn that breaks a width is a finding to report (a stance the calibration
   spawns never showed), not a reason to widen it; the test's two seeds are, among the held-out spawns that
   hold every width, the ones with the largest turn from the spawn and the largest lateral offset.

The ratchet.  ``--diff`` prints, on stdout, a unified diff of the species file with the class's pins
(``physics_revision``, ``window_start_s``, ``reference``, ``spread_width``, ``noisy_seeds``) replaced by the
re-measured ones and their provenance comments re-read; ``git apply`` takes it as it is.  ``--write`` makes the
same change in place.  Each change is classed (:func:`pin_changes`): ``tighter`` (a narrower width, an earlier
window, a one-sided reference moved toward its good side, a new bar), ``looser`` (the opposite, or a bar
dropped) or ``moved`` (a two-sided reference, the plant revision, the noisy seeds).  ``--write`` refuses a
looser change unless ``--allow-loosen`` is given, so after a model fix the committed bars only tighten unless a
reviewer says otherwise.  Both refuse when the running MuJoCo is not ``configs/plant_versions.toml``'s
canonical one or the window is under :data:`MIN_WINDOW_S`.  A pin whose value is unchanged keeps its lines as
they are, so on an unchanged plant the diff is empty.  Exit status: 0 nothing to change, 1 changes printed or
written, 2 a looser change (refused by ``--write`` without ``--allow-loosen``) or a refusal.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import difflib
import importlib
import inspect
import math
import re
import sys
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .com_stance_helpers import (
    BAR_KINDS,
    LEAN_DETECTION_KEYS,
    PIVOT_DETECTION_KEYS,
    REPO_ROOT,
    ROCK_DETECTION_KEYS,
    SYMMETRIC_VALUES,
    Bar,
    ComStanceQualityBase,
    TrackedEpisode,
    canonical_mujoco_version,
    fore_aft_rock,
    hip_roll_lean_action,
    manifest_physics_revision,
    resolution_floor,
    root_yaw_torque,
)
from .stance_track_helpers import StanceTrack, stance_track_quality, stance_track_series

CALIBRATION_SEEDS: tuple[int, ...] = tuple(range(30))
HELDOUT_SEEDS: tuple[int, ...] = tuple(range(100, 150))
#: The settle rule's speed: the horizontal CoM speed's RMS over a whole second, m/s.
SETTLE_SPEED_MPS = 1e-3
#: The settle rule's earliest window start, s (the reset's settle onto the floor and its pop stay outside).
EARLIEST_WINDOW_S = 2.0
#: The shortest window, s: a few periods of the slowest rocking (0.5 Hz) and room for the end keys' END_S.
MIN_WINDOW_S = 5.0


def settle_time(tracks: Iterable[StanceTrack], *, speed_mps: float = SETTLE_SPEED_MPS) -> float:
    """The first whole second from :data:`EARLIEST_WINDOW_S` on after which every track's per-second horizontal
    CoM speed RMS stays under *speed_mps* to its end (``inf`` when some track never settles)."""
    start = EARLIEST_WINDOW_S
    for track in tracks:
        series = stance_track_series(track)
        t = series["t"] - series["t"][0]
        speed = series["com_speed"]
        last_bad = -1.0
        for second in range(int(math.floor(t[-1]))):
            inside = (t >= second - 1e-9) & (t < second + 1 - 1e-9)
            if inside.any() and float(np.sqrt(np.mean(speed[inside] ** 2))) >= speed_mps:
                last_bad = float(second)
        if last_bad + 1.0 >= math.floor(t[-1]):
            return math.inf
        start = max(start, last_bad + 1.0)
    return start


def round_reference(key: str, value: float) -> float:
    """A statue value rounded for pinning: 1e-5 in SI for lengths and speeds, 1e-4 for degrees and load shares."""
    step = 1e-5 if key.endswith(("_m", "_mps")) else 1e-4
    return round(round(value / step) * step, 8)


def round_up_2sf(value: float) -> float:
    """*value* rounded up to two significant figures."""
    if not value > 0.0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(value)) - 1
    return float(f"{math.ceil(value / 10.0**exponent - 1e-9) * 10.0**exponent:.2g}")


def references(base: ComStanceQualityBase, statue: Mapping[str, float]) -> dict[str, float]:
    """The statue's rounded value of every key of *base*'s model that symmetry does not fix."""
    return {key: round_reference(key, float(statue[key])) for key in base.keys() if not base.symmetric(key)}


def _bar(base: ComStanceQualityBase, key: str, reference: Mapping[str, float], width: float) -> Bar:
    """*key*'s bar of *width* about the symmetric value or *reference*'s."""
    value = SYMMETRIC_VALUES[key] if base.symmetric(key) else float(reference[key])
    return Bar(key, BAR_KINDS[key], width, value)


def spread_widths(
    base: ComStanceQualityBase, reference: Mapping[str, float], calibration: Sequence[Mapping[str, float]]
) -> dict[str, float]:
    """Twice the worst bad-side deviation of the calibration spawns from the reference, never under twice the
    floor, rounded up to two significant figures."""
    out = {}
    for key in base.keys():
        unit = _bar(base, key, reference, 1.0)
        worst = max(max(unit.badness(float(metrics[key])), 0.0) for metrics in calibration)
        out[key] = round_up_2sf(max(2.0 * worst, 2.0 * resolution_floor(key)))
    return out


def usage(
    base: ComStanceQualityBase,
    reference: Mapping[str, float],
    widths: Mapping[str, float],
    metrics: Mapping[str, float],
) -> tuple[float, str]:
    """The largest share of a width *metrics* uses (over 1 breaks the bar), and its key."""
    return max((_bar(base, key, reference, widths[key]).ratio(float(metrics[key])), key) for key in base.keys())


@dataclass
class Calibration:
    """What :func:`derive` reads off one model's episodes."""

    settle_s: float
    reference: dict[str, float]
    spread_width: dict[str, float]
    heldout_usage: dict[int, tuple[float, str]]
    noisy_seeds: tuple[int, int]
    statue: dict[str, float]
    heldout: dict[int, dict[str, float]] = field(default_factory=dict)
    calibration: dict[int, dict[str, float]] = field(default_factory=dict)


def derive(
    base: ComStanceQualityBase,
    statue: StanceTrack,
    calibration: Mapping[int, StanceTrack],
    heldout: Mapping[int, StanceTrack],
    *,
    window_start_s: float | None = None,
) -> Calibration:
    """Rules 1-4 on one model's statue, calibration and held-out tracks (``window_start_s`` overrides rule 1)."""
    settle = settle_time([statue, *calibration.values()]) if window_start_s is None else window_start_s
    if not math.isfinite(settle):
        raise ValueError("a track never settles under the settle rule's speed: lengthen horizon_steps")
    statue_numbers = stance_track_quality(statue, window_start_s=settle)
    reference = references(base, statue_numbers)
    cal = {seed: stance_track_quality(track, window_start_s=settle) for seed, track in calibration.items()}
    widths = spread_widths(base, reference, list(cal.values()))
    held = {seed: stance_track_quality(track, window_start_s=settle) for seed, track in heldout.items()}
    used = {seed: usage(base, reference, widths, metrics) for seed, metrics in held.items()}
    holding = [seed for seed in held if used[seed][0] <= 1.0]
    by_turn = sorted(holding, key=lambda seed: (-abs(held[seed]["root_turn_from_spawn_deg"]), seed))
    by_lateral = sorted(holding, key=lambda seed: (-abs(held[seed]["com_lateral_mean_m"]), seed))
    first = by_turn[0]
    second = next(seed for seed in by_lateral if seed != first)
    return Calibration(settle, reference, widths, used, (first, second), statue_numbers, held, cal)


# ── the ratchet ──────────────────────────────────────────────────────────────

#: The pins the ratchet rewrites, as a species class assigns them.
PIN_NAMES: tuple[str, ...] = ("physics_revision", "window_start_s", "reference", "spread_width", "noisy_seeds")
#: The first words of the comment directly above a pin that states its measured numbers, which the ratchet
#: re-reads when it rewrites (a comment that starts otherwise is someone's own and stays as it is).  The comment
#: above ``reference`` states only the provenance, which :data:`_PROVENANCE` re-reads in place.
PIN_COMMENT_STARTS: dict[str, str] = {
    "spread_width": "Twice the worst of calibration seeds",
    "noisy_seeds": "Held-out seeds: the largest turn",
}
_PROVENANCE = re.compile(r"physics r\d+, mujoco \d[\d.]*\d, read on \d{4}-\d{2}-\d{2}")
_MISSING = object()


@dataclass(frozen=True)
class PinChange:
    """One pin a re-measurement moves, and which way a bar reads it: ``tighter``, ``looser`` or ``moved``."""

    pin: str
    key: str | None
    old: Any
    new: Any
    direction: str

    def describe(self) -> str:
        name = self.pin if self.key is None else f"{self.pin}[{self.key}]"
        return f"{self.direction} {name}: {self.old!r} -> {self.new!r}"


def _direction(more_is_looser: bool, old: float, new: float) -> str:
    return "looser" if (new > old) == more_is_looser else "tighter"


def pin_changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[PinChange]:
    """Each pin of *new* that differs from *old*'s, classed as the bars read it.

    ``spread_width``: wider is looser.  ``reference``: on an ``upper`` key (bar: the statue's value plus a width)
    a larger value is looser, on a ``lower`` key a smaller one; a ``dev`` key's bar is two-sided, so its move is
    ``moved``.  A key in *new* only is a new bar (tighter), one in *old* only a dropped bar (looser).
    ``window_start_s``: later is looser (more of the settle goes unread).  ``physics_revision`` and
    ``noisy_seeds``: ``moved``.
    """
    out: list[PinChange] = []
    for pin in PIN_NAMES:
        if pin not in new:
            continue
        before, after = old.get(pin), new[pin]
        if pin in ("reference", "spread_width"):
            before = dict(before or {})
            for key in [*before, *(key for key in after if key not in before)]:
                if key in before and key in after and before[key] == after[key]:
                    continue
                if key not in after:
                    direction = "looser"
                elif key not in before:
                    direction = "tighter"
                elif pin == "spread_width":
                    direction = _direction(True, before[key], after[key])
                elif BAR_KINDS[key] in ("upper", "lower"):
                    direction = _direction(BAR_KINDS[key] == "upper", before[key], after[key])
                else:
                    direction = "moved"
                out.append(PinChange(pin, key, before.get(key), after.get(key), direction))
        elif before != after:
            if pin == "window_start_s" and before is not None:
                direction = _direction(True, float(before), float(after))
            else:
                direction = "moved"
            out.append(PinChange(pin, None, before, after, direction))
    return out


def measured_pins(result: Calibration, physics_revision: int) -> dict[str, Any]:
    """The pins of a :class:`Calibration`, as plain Python numbers (so they print as literals)."""
    return {
        "physics_revision": int(physics_revision),
        "window_start_s": float(result.settle_s),
        "reference": {key: float(value) for key, value in result.reference.items()},
        "spread_width": {key: float(value) for key, value in result.spread_width.items()},
        "noisy_seeds": tuple(int(seed) for seed in result.noisy_seeds),
    }


def _class_node(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"no class {class_name} at the module's top level")


def _pin_nodes(class_node: ast.ClassDef) -> dict[str, ast.Assign | ast.AnnAssign]:
    out: dict[str, ast.Assign | ast.AnnAssign] = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name = node.target.id
        else:
            continue
        if name in PIN_NAMES:
            out[name] = node
    return out


def read_pins(source: str, class_name: str) -> dict[str, Any]:
    """The pins *class_name* assigns in its own body in *source*, as literals."""
    nodes = _pin_nodes(_class_node(ast.parse(source), class_name))
    return {name: ast.literal_eval(node.value) for name, node in nodes.items() if node.value is not None}


def render_pin(name: str, value: Any, indent: str) -> list[str]:
    """``name = value`` as class-body lines: a dict one key a line with a trailing comma, as ruff formats it."""
    if isinstance(value, Mapping):
        items = [f'{indent}    "{key}": {item!r},' for key, item in value.items()]
        return [f"{indent}{name} = {{", *items, f"{indent}}}"]
    return [f"{indent}{name} = {value!r}"]


def rewrite_pins(
    source: str,
    class_name: str,
    pins: Mapping[str, Any],
    *,
    provenance: str | None = None,
    comments: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """*source* with *class_name*'s pins set to *pins*.

    When no pin changes, *source* comes back as it is; a pin whose value is unchanged always keeps its lines.
    Otherwise each changed pin's assignment is re-rendered (:func:`render_pin`), every
    ``physics rN, mujoco X, read on YYYY-MM-DD`` in the class body becomes *provenance*, and each comment block
    directly above a pin of *comments* whose first line starts with that pin's :data:`PIN_COMMENT_STARTS` words
    is replaced by *comments*' lines unless it already says the same words (however they are wrapped).  A changed
    pin that the class body does not assign raises ``ValueError``.
    """
    class_node = _class_node(ast.parse(source), class_name)
    nodes = _pin_nodes(class_node)
    current = {name: ast.literal_eval(node.value) for name, node in nodes.items() if node.value is not None}
    changed = [name for name in PIN_NAMES if name in pins and current.get(name, _MISSING) != pins[name]]
    if not changed:
        return source
    missing = [name for name in changed if name not in nodes]
    if missing:
        raise ValueError(f"{class_name} does not assign {missing} in its own body: add them by hand first")
    lines = source.splitlines(keepends=True)
    if provenance is not None:
        for index in range(class_node.lineno - 1, int(class_node.end_lineno or len(lines))):
            if lines[index].lstrip().startswith("#"):
                lines[index] = _PROVENANCE.sub(provenance, lines[index])
    edits: list[tuple[int, int, list[str]]] = []
    for name in changed:
        node = nodes[name]
        edits.append(
            (node.lineno - 1, int(node.end_lineno or node.lineno), render_pin(name, pins[name], " " * node.col_offset))
        )
    for name, text in (comments or {}).items():
        if name not in nodes:
            continue
        indent = " " * nodes[name].col_offset
        end = start = nodes[name].lineno - 1
        while start > class_node.lineno and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        said = " ".join(line.strip().removeprefix("#").strip() for line in lines[start:end])
        if start < end and said.startswith(PIN_COMMENT_STARTS[name]) and said.split() != " ".join(text).split():
            edits.append((start, end, [f"{indent}# {line}" for line in text]))
    for start, end, new_lines in sorted(edits, reverse=True):
        lines[start:end] = [f"{line}\n" for line in new_lines]
    return "".join(lines)


def pin_comments(base: ComStanceQualityBase, result: Calibration) -> dict[str, list[str]]:
    """The comments stating a calibration's numbers above its pins (:data:`PIN_COMMENT_STARTS`), wrapped for a
    class body."""
    held = result.heldout_usage
    broken = sorted(seed for seed, (ratio, _) in held.items() if ratio > 1.0)
    holding = {seed: used for seed, used in held.items() if seed not in broken}
    worst = max(holding, key=lambda seed: holding[seed][0])
    use = f"{holding[worst][0]:.2f} ({holding[worst][1]}, seed {worst})"
    if broken:
        heldout = (
            f"{len(broken)} break a width ({', '.join(map(str, broken))}: see known_poor_spawns); the largest use "
            f"among the rest is {use}."
        )
    else:
        heldout = f"none breaks a width; the largest use is {use}."
    first, second = result.noisy_seeds
    paragraphs = {
        "spread_width": [
            f"Twice the worst of calibration seeds {CALIBRATION_SEEDS[0]}-{CALIBRATION_SEEDS[-1]} at reset noise "
            f"{base.noisy_reset_noise_scale:g}, never under twice the floor (SI, deg).",
            f"Held-out seeds {HELDOUT_SEEDS[0]}-{HELDOUT_SEEDS[-1]}: {heldout}",
        ],
        "noisy_seeds": [
            f"Held-out seeds: the largest turn from the spawn ({result.heldout[first]['root_turn_from_spawn_deg']:.2f} "
            f"deg) and the largest lateral offset ({1e3 * result.heldout[second]['com_lateral_mean_m']:.3f} mm) among "
            f"those holding every width; they use {held[first][0]:.2f} and {held[second][0]:.2f} of a width."
        ],
    }
    return {
        name: [line for text in texts for line in textwrap.wrap(text, width=112, break_on_hyphens=False)]
        for name, texts in paragraphs.items()
    }


def ratchet(
    path: Path,
    class_name: str,
    pins: Mapping[str, Any],
    *,
    write: bool,
    allow_loosen: bool,
    provenance: str | None = None,
    comments: Mapping[str, Sequence[str]] | None = None,
    display_path: str | None = None,
) -> tuple[int, str, list[PinChange]]:
    """Compare *pins* with *class_name*'s in *path*: (exit status, unified diff, the changes).

    Status 0: nothing changes (the diff is empty).  1: a non-empty diff with no looser change, or one that
    ``allow_loosen`` let ``write`` apply.  2: a looser change, which ``write`` refuses without ``allow_loosen``
    (the file is not touched).  ``write`` rewrites *path* in place otherwise.
    """
    source = path.read_text()
    changes = pin_changes(read_pins(source, class_name), pins)
    updated = rewrite_pins(source, class_name, pins, provenance=provenance, comments=comments)
    name = display_path or str(path)
    diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True), updated.splitlines(keepends=True), f"a/{name}", f"b/{name}"
        )
    )
    if not changes:
        return 0, diff, changes
    looser = any(change.direction == "looser" for change in changes)
    if write and (allow_loosen or not looser):
        path.write_text(updated)
        return 1, diff, changes
    return (2 if looser else 1), diff, changes


def known_poor_seeds(cls: type) -> list[int]:
    """The seeds of a class's ``test_known_poor_spawn_seed<N>_...`` reproducers (strict expected failures)."""
    found = (re.match(r"test_known_poor_spawn_seed(\d+)_", name) for name in dir(cls))
    return sorted({int(match.group(1)) for match in found if match})


def _detections(base: ComStanceQualityBase, reference: Mapping[str, float], metrics: Mapping[str, float], keys) -> str:
    parts = []
    for key in (key for key in keys if key in base.keys()):
        bar = _bar(base, key, reference, 2.0 * resolution_floor(key))
        parts.append(f"{key} {bar.ratio(float(metrics.get(key, math.nan))):.2f}x")
    return ", ".join(parts)


def _python_dict(name: str, values: Mapping[str, float], current: Mapping[str, float]) -> list[str]:
    lines = [f"    {name} = {{"]
    for key, value in values.items():
        old = current.get(key)
        note = "" if old is None or old == value else f"  # was {old!r}"
        lines.append(f'        "{key}": {value!r},{note}')
    lines.append("    }")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("target", help="the species test class, as <module>:<class>")
    parser.add_argument(
        "--diff", action="store_true", help="print a unified diff of the species file with the re-measured pins"
    )
    parser.add_argument("--write", action="store_true", help="rewrite the species file's pins in place")
    parser.add_argument("--allow-loosen", action="store_true", help="let --write apply a change that loosens a bar")
    args = parser.parse_args(argv)
    module_name, _, class_name = args.target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    base: ComStanceQualityBase = cls()
    log: Callable[[str], None] = lambda text: print(text, file=sys.stderr, flush=True)  # noqa: E731
    ratcheting = args.diff or args.write
    # In the ratchet modes stdout carries the diff alone, so the report goes to stderr.
    say: Callable[[str], None] = log if ratcheting else print
    canonical = canonical_mujoco_version()
    if ratcheting and mujoco.__version__ != canonical:
        log(f"refused: running mujoco {mujoco.__version__}; pins are read on the canonical {canonical}")
        return 2

    def run(seed: int, noisy: bool) -> TrackedEpisode:
        episode = base.run_episode(seed=seed, reset_noise_scale=base.noisy_reset_noise_scale if noisy else None)
        if not episode.stood:
            raise SystemExit(f"seed {seed} terminated at step {episode.steps}: {episode.termination_reason}")
        return episode

    log(f"{base.species_label}: the statue, {len(CALIBRATION_SEEDS)} calibration and {len(HELDOUT_SEEDS)} held-out")
    statue = run(0, noisy=False)
    calibration = {seed: run(seed, noisy=True).track for seed in CALIBRATION_SEEDS}
    heldout = {seed: run(seed, noisy=True).track for seed in HELDOUT_SEEDS}
    result = derive(base, statue.track, calibration, heldout)
    horizon_s = statue.horizon * statue.dt
    say(
        f"# {base.species_label}: settle {result.settle_s:g} s of a {horizon_s:g} s horizon (class: "
        f"window_start_s = {base.window_start_s:g}, horizon_steps = {base.horizon_steps})"
    )
    short_window = horizon_s - result.settle_s < MIN_WINDOW_S
    if short_window:
        say(f"# the window is under {MIN_WINDOW_S:g} s: double horizon_steps and re-run")
    say("\n".join(_python_dict("reference", result.reference, base.reference)))
    say("\n".join(_python_dict("spread_width", result.spread_width, base.spread_width)))
    broken = sorted(seed for seed, (ratio, _) in result.heldout_usage.items() if ratio > 1.0)
    worst_seed = max(result.heldout_usage, key=lambda seed: result.heldout_usage[seed][0])
    ratio, key = result.heldout_usage[worst_seed]
    say(
        f"# held-out: {len(broken)} of {len(HELDOUT_SEEDS)} spawns break a width {broken}; the largest use is "
        f"{ratio:.2f} ({key}, seed {worst_seed})"
    )
    for seed in broken:
        over = sorted(
            (
                (_bar(base, k, result.reference, result.spread_width[k]).ratio(result.heldout[seed][k]), k)
                for k in base.keys()
            ),
            reverse=True,
        )
        say(f"#   seed {seed} (a finding): " + ", ".join(f"{k} {r:.1f}x" for r, k in over if r > 1.0))
    for seed in known_poor_seeds(cls):
        if seed not in result.heldout_usage:
            say(f"# known poor seed {seed}: not a held-out seed here; run its test")
            continue
        ratio, key = result.heldout_usage[seed]
        verdict = (
            "still breaks a width: keep its strict expected failure"
            if ratio > 1.0
            else "now holds every width: its strict expected failure will fail, so remove it"
        )
        say(f"# known poor seed {seed}: uses {ratio:.2f} of a width ({key}): {verdict}")
    say(f"    noisy_seeds = {result.noisy_seeds}")
    for seed in result.noisy_seeds:
        say(
            f"#   seed {seed}: uses at most {result.heldout_usage[seed][0]:.2f} of a width "
            f"({result.heldout_usage[seed][1]})"
        )
    scripted = {
        "lean": (dict(action=lambda env, m: hip_roll_lean_action(env, m, base.lean_degrees)), LEAN_DETECTION_KEYS),
        "pivot": (dict(pre_step=lambda env, m: root_yaw_torque(env, m, base.yaw_torque_k)), PIVOT_DETECTION_KEYS),
        "rock": (
            dict(pre_step=lambda env, m: fore_aft_rock(env, m, base.rock_force_bw, base.rock_hz)),
            ROCK_DETECTION_KEYS,
        ),
    }
    for name, (hooks, keys) in scripted.items():
        episode = base.run_episode(seed=0, **hooks)  # type: ignore[arg-type]
        if not episode.stood:
            say(f"# scripted {name}: terminated at step {episode.steps} ({episode.termination_reason})")
            continue
        metrics = stance_track_quality(episode.track, window_start_s=result.settle_s)
        say(f"# scripted {name}: {_detections(base, result.reference, metrics, keys)}")
    if not ratcheting:
        return 0
    if short_window:
        log("# ratchet: refused: the window is under the shortest; double horizon_steps by hand and re-run")
        return 2
    revision = manifest_physics_revision(base.model_id)
    path = Path(inspect.getsourcefile(cls) or "")
    try:
        display = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        display = str(path)
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    status, diff, changes = ratchet(
        path,
        class_name,
        measured_pins(result, revision),
        write=args.write,
        allow_loosen=args.allow_loosen,
        provenance=f"physics r{revision}, mujoco {mujoco.__version__}, read on {today}",
        comments=pin_comments(base, result),
        display_path=display,
    )
    if args.diff:
        sys.stdout.write(diff)
    for change in changes:
        log(f"# ratchet: {change.describe()}")
    looser = sum(change.direction == "looser" for change in changes)
    if status == 0:
        log(f"# ratchet: the re-measured pins equal {class_name}'s: nothing to change in {display}")
    elif args.write and status == 2:
        log(f"# ratchet: refused: {looser} looser change(s) above; {display} not written (--allow-loosen accepts them)")
    elif args.write:
        log(f"# ratchet: wrote {display}; its docstring's measured numbers are re-read by hand (the report above)")
    else:
        log(f"# ratchet: {len(changes)} change(s), {looser} looser; --write applies them")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
