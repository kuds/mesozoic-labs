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
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .com_stance_helpers import (
    BAR_KINDS,
    LEAN_DETECTION_KEYS,
    PIVOT_DETECTION_KEYS,
    ROCK_DETECTION_KEYS,
    SYMMETRIC_VALUES,
    Bar,
    ComStanceQualityBase,
    TrackedEpisode,
    fore_aft_rock,
    hip_roll_lean_action,
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
    args = parser.parse_args(argv)
    module_name, _, class_name = args.target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    base: ComStanceQualityBase = cls()
    log: Callable[[str], None] = lambda text: print(text, file=sys.stderr, flush=True)  # noqa: E731

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
    print(
        f"# {base.species_label}: settle {result.settle_s:g} s of a {horizon_s:g} s horizon (class: "
        f"window_start_s = {base.window_start_s:g}, horizon_steps = {base.horizon_steps})"
    )
    if horizon_s - result.settle_s < MIN_WINDOW_S:
        print(f"# the window is under {MIN_WINDOW_S:g} s: double horizon_steps and re-run")
    print("\n".join(_python_dict("reference", result.reference, base.reference)))
    print("\n".join(_python_dict("spread_width", result.spread_width, base.spread_width)))
    broken = sorted(seed for seed, (ratio, _) in result.heldout_usage.items() if ratio > 1.0)
    worst_seed = max(result.heldout_usage, key=lambda seed: result.heldout_usage[seed][0])
    ratio, key = result.heldout_usage[worst_seed]
    print(
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
        print(f"#   seed {seed} (a finding): " + ", ".join(f"{k} {r:.1f}x" for r, k in over if r > 1.0))
    print(f"    noisy_seeds = {result.noisy_seeds}")
    for seed in result.noisy_seeds:
        print(
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
            print(f"# scripted {name}: terminated at step {episode.steps} ({episode.termination_reason})")
            continue
        metrics = stance_track_quality(episode.track, window_start_s=result.settle_s)
        print(f"# scripted {name}: {_detections(base, result.reference, metrics, keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
