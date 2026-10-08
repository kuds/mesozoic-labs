"""Per-episode stance metrics on a floor-truth :class:`.recorder.EpisodeTrace` (pure numpy).

:func:`episode_stance_metrics` turns one recorded episode into a
:class:`StanceEpisodeMetrics` row: every per-episode quantity a
``stance_quality/v2`` threshold key reads, plus the report-only ones.  No
mujoco import (the trace class is imported for its type only), so the pure
gate module can import it.

NaN means UNMEASURED, and the gate treats it as a failure
(``curriculum/gate_schema.finite_gate_metric``): a window with no steps
(``length <= settle_steps``), a sole metric on a foot with no sole
(velociraptor), a box-corner metric on a non-box sole (brachiosaurus'
ellipsoids), and the saturation metrics when a window step has no action.
The whole-number counts (``*_substeps``, ``*_steps``, ``*_events``,
``settle_touchdowns``) are floats for that reason.

Windows (``g = spawn_grace_steps(dt)``, ``s = settle_steps``, ``T`` steps)
--------------------------------------------------------------------------
* spawn ``[0, g)`` -- the reset transient, a PLANT property: the spawn
  metrics are for plant acceptance, not for the policy gate;
* settle ``[g, s)`` for the hop and impact checks, after the spawn grace;
  the whole of ``[0, s)`` for the settle touchdowns and the settle geometry;
* window ``[s, T)`` for everything else.

Conventions: a leg is down on a step when ``leg_down_frac >=
DOWN_SUBSTEP_FRACTION``; a share is a force ratio (a foot's summed floor
force over the window divided by every foot's); a "pair distance" is the
HORIZONTAL distance between two feet's reference sites, over every pair (1
on bipeds, 6 on quadrupeds), so it is free of yaw.

Statue-relative keys (``min_foot_load_share_statue_ratio``,
``min_avg_reward_statue_ratio``) are panel-level: the gate divides these
per-episode values (``foot_load_share``, ``reward``) by a statue panel's
means, rolled through the same recorder.  This module only supplies them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from . import events
from .constants import (
    CONTACT_THRESHOLD_N,
    DEBOUNCE_S,
    LOAD_WINDOW_S,
    SATURATION_ABS,
    SOLE_ROLLED_DEG,
    SPAWN_GRACE_S,
    SUPPORT_GEOM_DOWN_FRACTION,
)

if TYPE_CHECKING:
    from .recorder import EpisodeTrace

NAN = float("nan")


def spawn_grace_steps(dt: float) -> int:
    """The spawn grace in control steps at step ``dt`` (10 at dt 0.01, 5 at dt 0.02).

    A ``settle_steps`` at or below it leaves the settle window empty and every
    settle metric NaN; the gate's threshold validation should refuse one.
    """
    return events.steps_for(SPAWN_GRACE_S, dt)


@dataclass(frozen=True)
class StanceEpisodeMetrics:
    """One episode's stance metrics; the module docstring defines the windows."""

    length: int
    terminated: bool
    #: The episode's summed reward, NaN when any step's is not finite.
    reward: float
    settle_steps: int
    # --- post-settle window [s, T) ---
    #: Fraction of window steps with every leg down / with no leg down.
    all_feet_support: float
    flight_fraction: float
    #: Window substeps with every leg unloaded, and the max substep animal floor force over the
    #: window / body weight: the settle window's hop and impact detectors, read over the window.  A
    #: light plant's flight phase can last fewer substeps than DOWN_SUBSTEP_FRACTION of a step, so a
    #: hop that leaves the floor on every step can still read every leg down on every step.
    window_airborne_substeps: float
    window_peak_floor_force_bw: float
    #: Debounced touchdowns in the window, summed over feet, per foot per second.
    touchdown_rate: float
    #: Min over feet of the window load share; and of the share over each whole LOAD_WINDOW_S
    #: block of the window (a block with no foot load scores 0).
    min_foot_load_share: float
    min_foot_load_share_windowed: float
    #: Horizontal root travel from the last settle step to the last step (m).
    window_displacement_m: float
    #: |unwrapped yaw change| over the window, and over the episode from spawn: report-only (the trex
    #: statue turns up to 12.6 degrees, the velociraptor one 15.7).
    window_yaw_change_deg: float
    episode_yaw_change_deg: float
    #: Max over feet of the horizontal reference-site travel across the window (m).
    max_foot_slide_m: float
    #: Mean over the window of the fraction of substeps with leg-on-leg contact (foot-on-foot).
    foot_on_foot_fraction: float
    #: Window steps on which some foot's touch reads > CONTACT_THRESHOLD_N while the floor says it is
    #: up; and on which touch and floor agree on every foot (report-only).
    phantom_support_fraction: float
    touch_floor_agreement: float
    #: Window floor force on non-leg geoms (trunk, tail, head, arms), and on leg geoms outside the
    #: support registry, as fractions of the animal's window floor force.  0 on every statue.
    nonfoot_load_fraction: float
    offsupport_load_fraction: float
    #: Mean over the window of the fraction of substeps with a leg against the rest of the body.
    leg_body_contact_fraction: float
    #: Min over support geoms of the fraction of window steps the geom is loaded (floor force above
    #: the threshold on >= SUPPORT_GEOM_DOWN_FRACTION of the substeps); min over feet of the mean
    #: fraction of the foot's support geoms loaded.
    min_support_geom_duty: float
    min_support_geom_coverage: float
    #: Max over sole feet of the window-mean sole tilt; of that minus the keyframe's own tilt; of
    #: |window-mean roll|.  NaN when no foot has a sole.
    max_sole_tilt_deg: float
    max_sole_tilt_excess_deg: float
    max_sole_abs_roll_deg: float
    #: Window steps with some sole tilted more than SOLE_ROLLED_DEG beyond its keyframe tilt.
    sole_rolled_fraction: float
    #: Box soles: max over feet of the window-mean (highest - lowest bottom corner) (m).
    max_sole_corner_lift_m: float
    #: Min over sole feet of the window-mean loaded contact points on the sole (box: 4 flat, 2 edge,
    #: 1 corner).
    min_sole_contacts: float
    #: Max over sole feet of the window-mean |outer CoP| / half-width: report-only.
    max_sole_cop_outer: float
    # --- spawn [0, g) and settle [g, s) ---
    #: Max substep animal floor force / body weight in the spawn grace; substeps there with every
    #: leg unloaded.  The reset pop: plant acceptance, not the policy gate.
    spawn_peak_floor_force_bw: float
    spawn_airborne_substeps: float
    #: Settle steps with no leg down, and their runs.
    settle_unsupported_steps: float
    settle_unsupported_events: float
    #: Settle substeps with every leg unloaded: the substep-level hop detector.
    settle_airborne_substeps: float
    #: Max substep animal floor force over the settle window / body weight: a hop's or stomp's impact.
    settle_peak_floor_force_bw: float
    #: Debounced landings in [0, s); a foot that spawns in the air counts its first landing.
    #: Report-only (a trex statue scores 1 when one foot spawns airborne).
    settle_touchdowns: float
    #: Max over pairs of |pair distance at step s-1 - at spawn|, and max over feet of the horizontal
    #: reference-site shift from spawn to step s-1 (m).
    settle_stance_width_change_m: float
    settle_max_foot_shift_m: float
    #: Max over every step and pair of |pair distance - at spawn| (m).
    max_stance_width_change_m: float
    # --- actions, post-settle window ---
    #: Max / mean over actuators of the fraction of window steps with |action| >= SATURATION_ABS.
    max_actuator_saturation_fraction: float
    mean_actuator_saturation_fraction: float
    # --- per-foot and per-geom detail (report, and the statue-relative ratios) ---
    #: Per foot: the window load share.
    foot_load_share: tuple[float, ...] = ()
    #: Per support geom (foot-major, as EpisodeTrace.support_geom_names): the loaded-step fraction, and
    #: the geom's share of its foot's window load.
    support_geom_duty: tuple[float, ...] = ()
    support_geom_share: tuple[float, ...] = ()
    #: Per foot: window means of the sole tilt, roll and pitch, and of the corner lift.
    mean_sole_tilt_deg: tuple[float, ...] = ()
    mean_sole_roll_deg: tuple[float, ...] = ()
    mean_sole_pitch_deg: tuple[float, ...] = ()
    mean_sole_corner_lift_m: tuple[float, ...] = ()

    def as_row(self) -> dict[str, Any]:
        """Every field by name, the per-foot tuples as lists: a JSON-ready (and ``csv.DictWriter``-ready) row."""
        row = asdict(self)
        for name in STANCE_METRIC_FOOT_FIELDS:
            row[name] = list(row[name])
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "StanceEpisodeMetrics":
        """Rebuild from :meth:`as_row` output or from its CSV rendering (every cell a string).

        Strict: every field must be present (extra keys such as a seed column
        are ignored), except the :data:`STANCE_METRIC_LATER_FIELDS` a panel
        recorded before they existed lacks, which read as NaN -- unmeasured,
        so a gate that declares their key fails such a panel and one that does
        not is unaffected.  An empty or ``None`` cell is NaN -- unmeasured -- for a
        float field and refused for the others; ``"True"`` / ``"False"`` read
        as booleans; a per-foot cell is a list or its ``str()`` (``"[0.5,
        nan]"``).  Floats written by ``str`` / ``repr`` round-trip exactly.
        """
        missing = [
            spec.name for spec in fields(cls) if spec.name not in row and spec.name not in STANCE_METRIC_LATER_FIELDS
        ]
        if missing:
            raise ValueError(f"stance metrics row lacks {missing}")
        values: dict[str, Any] = {}
        for spec in fields(cls):
            value = row.get(spec.name)
            if spec.name in _INT_FIELDS:
                values[spec.name] = _parse_int(spec.name, value)
            elif spec.name in _BOOL_FIELDS:
                values[spec.name] = _parse_bool(spec.name, value)
            elif spec.name in STANCE_METRIC_FOOT_FIELDS:
                values[spec.name] = _parse_floats(spec.name, value)
            else:
                values[spec.name] = _parse_float(spec.name, value)
        return cls(**values)


_INT_FIELDS = frozenset({"length", "settle_steps"})
_BOOL_FIELDS = frozenset({"terminated"})

#: Float fields added after the panel CSV contract shipped (with the window
#: hop pair, D-D26): a row or a panel CSV recorded before then lacks them, and
#: :meth:`StanceEpisodeMetrics.from_row` and
#: ``curriculum.stance_gate_v2.read_stance_v2_panel`` read them as unmeasured,
#: so every report recorded before the pair keeps its verdict under a gate
#: that does not declare it.
STANCE_METRIC_LATER_FIELDS: frozenset[str] = frozenset({"window_airborne_substeps", "window_peak_floor_force_bw"})

#: The per-foot / per-geom tuple fields, in declaration order.
STANCE_METRIC_FOOT_FIELDS: tuple[str, ...] = (
    "foot_load_share",
    "support_geom_duty",
    "support_geom_share",
    "mean_sole_tilt_deg",
    "mean_sole_roll_deg",
    "mean_sole_pitch_deg",
    "mean_sole_corner_lift_m",
)

#: The scalar fields, in declaration (CSV column) order.
STANCE_METRIC_FIELDS: tuple[str, ...] = tuple(
    spec.name for spec in fields(StanceEpisodeMetrics) if spec.name not in STANCE_METRIC_FOOT_FIELDS
)


def _parse_float(name: str, value: Any) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return NAN
    if isinstance(value, bool):
        raise ValueError(f"stance metric {name} is a boolean, not a number: {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stance metric {name} is not a number: {value!r}") from exc


def _parse_int(name: str, value: Any) -> int:
    number = _parse_float(name, value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"stance metric {name} is not a whole number: {value!r}")
    return int(number)


def _parse_bool(name: str, value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"stance metric {name} is not a boolean: {value!r}")


def _parse_floats(name: str, value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[(" and text[-1:] in "])":
            text = text[1:-1]
        items: list[Any] = [item for item in text.split(",") if item.strip()]
    elif isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
    else:
        raise ValueError(f"stance metric {name} is not a per-foot list: {value!r}")
    return tuple(_parse_float(name, item) for item in items)


def _mean(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()) if array.size else NAN


def _finite_column_means(values: np.ndarray) -> np.ndarray:
    """Per-column means over the finite entries; NaN (without a warning) for a column with none."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _max_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else NAN


def _yaw(quat: np.ndarray) -> np.ndarray:
    """Heading of (w, x, y, z) quaternions, radians."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    heading: np.ndarray = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return heading


def _pair_distances(positions: np.ndarray) -> np.ndarray:
    """(..., F, 3) -> (..., P): the horizontal distance of every foot pair (P = 0 with one foot)."""
    pairs = list(combinations(range(positions.shape[-2]), 2))
    if not pairs:
        return np.zeros((*positions.shape[:-2], 0))
    return np.stack([np.linalg.norm(positions[..., i, :2] - positions[..., j, :2], axis=-1) for i, j in pairs], axis=-1)


def episode_stance_metrics(trace: "EpisodeTrace", *, settle_steps: int) -> StanceEpisodeMetrics:
    """The stance metrics of one recorded episode, with the window starting at ``settle_steps``."""
    if isinstance(settle_steps, bool) or int(settle_steps) != settle_steps or settle_steps < 0:
        raise ValueError(f"settle_steps must be a non-negative whole number of steps, got {settle_steps!r}")
    steps, n_feet = trace.leg_floor_mean.shape
    if steps == 0:
        raise ValueError("the trace has no steps")
    dt, body_weight = float(trace.dt), float(trace.body_weight_n)
    settle = int(settle_steps)
    window = slice(settle, steps)
    window_steps = max(steps - settle, 0)
    down = events.down_mask(trace.leg_down_frac)
    debounce_steps = events.steps_for(DEBOUNCE_S, dt)
    debounced = np.stack([events.debounce(down[:, foot], debounce_steps) for foot in range(n_feet)], axis=1)
    support_foot = np.asarray(trace.support_geom_foot, dtype=np.int64)
    n_support = support_foot.size
    yaw = np.unwrap(np.concatenate([[_yaw(trace.spawn_root_quat)], _yaw(trace.root_quat)]))  # [spawn, step 0, ...]

    window_values: dict[str, Any]
    if window_steps:
        # The last settle step is the reference of the window's displacement and slides; the
        # spawn sample is when there is no settle window.
        root_before = trace.root_pos[settle - 1, :2] if settle >= 1 else trace.spawn_root_pos[:2]
        feet_before = trace.foot_pos[settle - 1, :, :2] if settle >= 1 else trace.spawn_foot_pos[:, :2]
        down_window = down[window]
        touchdowns = sum(int(np.sum(events.rising_edges(debounced[:, foot]) >= settle)) for foot in range(n_feet))
        leg_window = trace.leg_floor_mean[window]
        leg_total = float(leg_window.sum())
        shares = leg_window.sum(axis=0) / leg_total if leg_total > 0 else np.full(n_feet, NAN)
        block = events.steps_for(LOAD_WINDOW_S, dt)
        block_minima = []
        for start in range(0, window_steps - block + 1, block):
            loads = leg_window[start : start + block]
            block_total = float(loads.sum())
            block_minima.append(float((loads.sum(axis=0) / block_total).min()) if block_total > 0 else 0.0)
        animal_total = float(trace.total_floor_mean[window].sum())
        loaded = trace.support_loaded_frac[window] >= SUPPORT_GEOM_DOWN_FRACTION  # (W, G)
        geom_duty = loaded.mean(axis=0) if n_support else np.zeros(0)
        geom_share = np.full(n_support, NAN)
        coverage = []
        for foot in range(n_feet):
            mine = support_foot == foot
            coverage.append(float(loaded[:, mine].mean()) if mine.any() else NAN)
            foot_total = float(leg_window[:, foot].sum())
            if foot_total > 0:
                geom_share[mine] = trace.support_floor_mean[window][:, mine].sum(axis=0) / foot_total
        touch_up = trace.touch[window] > CONTACT_THRESHOLD_N
        if animal_total > 0:
            nonfoot = float(trace.nonleg_floor_mean[window].sum()) / animal_total
            offsupport = float(trace.offsupport_floor_mean[window].sum()) / animal_total
        else:
            nonfoot = offsupport = NAN
        window_values = {
            "all_feet_support": _mean(np.all(down_window, axis=1)),
            "flight_fraction": _mean(~np.any(down_window, axis=1)),
            "window_airborne_substeps": float(np.sum(trace.feet_airborne_substeps[window])),
            "window_peak_floor_force_bw": float(np.max(trace.total_floor_max[window])) / body_weight,
            "touchdown_rate": touchdowns / (n_feet * window_steps * dt),
            "min_foot_load_share": float(shares.min()) if np.isfinite(shares).all() else NAN,
            "min_foot_load_share_windowed": min(block_minima) if block_minima else NAN,
            "window_displacement_m": float(np.linalg.norm(trace.root_pos[steps - 1, :2] - root_before)),
            "window_yaw_change_deg": float(np.degrees(abs(yaw[steps] - yaw[settle]))),
            "max_foot_slide_m": float(np.max(np.linalg.norm(trace.foot_pos[steps - 1, :, :2] - feet_before, axis=-1))),
            "foot_on_foot_fraction": _mean(trace.interleg_substep_frac[window]),
            "phantom_support_fraction": _mean(np.any(touch_up & ~down_window, axis=1)),
            "touch_floor_agreement": _mean(np.all(touch_up == down_window, axis=1)),
            "nonfoot_load_fraction": nonfoot,
            "offsupport_load_fraction": offsupport,
            "leg_body_contact_fraction": _mean(trace.leg_body_substep_frac[window]),
            "min_support_geom_duty": float(geom_duty.min()) if n_support else NAN,
            "min_support_geom_coverage": min(coverage) if all(map(math.isfinite, coverage)) else NAN,
            **_sole_metrics(trace, window),
            **_saturation_metrics(trace.action[window]),
        }
        foot_load_share = shares
    else:
        window_values = {name: NAN for name in _WINDOW_FIELDS}
        foot_load_share = np.full(n_feet, NAN)
        geom_duty = np.full(n_support, NAN)
        geom_share = np.full(n_support, NAN)

    # --- spawn [0, g) and settle [g, s): hops and impacts after the spawn grace ---
    grace = min(spawn_grace_steps(dt), settle, steps)
    settle_end = min(settle, steps)
    settle_window = slice(grace, settle_end)
    if settle_end > grace:
        unsupported = ~np.any(down[settle_window], axis=1)
        settle_unsupported_steps = float(unsupported.sum())
        settle_unsupported_events = float(events.count_runs(unsupported))
        settle_airborne_substeps = float(np.sum(trace.feet_airborne_substeps[settle_window]))
        settle_peak = float(np.max(trace.total_floor_max[settle_window])) / body_weight
    else:
        settle_unsupported_steps = settle_unsupported_events = settle_airborne_substeps = settle_peak = NAN
    if grace:
        spawn_peak = float(np.max(trace.total_floor_max[:grace])) / body_weight
        spawn_airborne = float(np.sum(trace.feet_airborne_substeps[:grace]))
    else:
        spawn_peak = spawn_airborne = NAN
    if settle_end:
        # Prepending "down" makes a foot that spawns in the air count its first landing.
        settle_touchdowns = float(
            sum(
                events.rising_edges(np.concatenate([[True], debounced[:settle_end, foot]])).size
                for foot in range(n_feet)
            )
        )
    else:
        settle_touchdowns = NAN
    spawn_pairs = _pair_distances(trace.spawn_foot_pos[None])[0]
    step_pairs = _pair_distances(trace.foot_pos)
    if 1 <= settle <= steps and spawn_pairs.size:
        settle_width_change = float(np.max(np.abs(step_pairs[settle - 1] - spawn_pairs)))
    else:
        settle_width_change = NAN
    if 1 <= settle <= steps:
        settle_shift = float(
            np.max(np.linalg.norm(trace.foot_pos[settle - 1, :, :2] - trace.spawn_foot_pos[:, :2], axis=-1))
        )
    else:
        settle_shift = NAN
    max_width_change = float(np.max(np.abs(step_pairs - spawn_pairs))) if spawn_pairs.size else NAN

    reward = float(np.sum(trace.reward)) if np.all(np.isfinite(trace.reward)) else NAN
    sole_tilt = trace.sole_tilt_deg[window]
    return StanceEpisodeMetrics(
        length=int(steps),
        terminated=bool(trace.terminated),
        reward=reward,
        settle_steps=settle,
        episode_yaw_change_deg=float(np.degrees(abs(yaw[-1] - yaw[0]))),
        spawn_peak_floor_force_bw=spawn_peak,
        spawn_airborne_substeps=spawn_airborne,
        settle_unsupported_steps=settle_unsupported_steps,
        settle_unsupported_events=settle_unsupported_events,
        settle_airborne_substeps=settle_airborne_substeps,
        settle_peak_floor_force_bw=settle_peak,
        settle_touchdowns=settle_touchdowns,
        settle_stance_width_change_m=settle_width_change,
        settle_max_foot_shift_m=settle_shift,
        max_stance_width_change_m=max_width_change,
        foot_load_share=_floats(foot_load_share),
        support_geom_duty=_floats(geom_duty),
        support_geom_share=_floats(geom_share),
        mean_sole_tilt_deg=_floats(_window_column_means(sole_tilt, n_feet)),
        mean_sole_roll_deg=_floats(_window_column_means(trace.sole_roll_deg[window], n_feet)),
        mean_sole_pitch_deg=_floats(_window_column_means(trace.sole_pitch_deg[window], n_feet)),
        mean_sole_corner_lift_m=_floats(_window_column_means(trace.sole_corner_lift[window], n_feet)),
        **window_values,
    )


def _floats(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=np.float64))


def _window_column_means(values: np.ndarray, n_feet: int) -> np.ndarray:
    return _finite_column_means(values) if values.shape[0] else np.full(n_feet, NAN)


def _sole_metrics(trace: "EpisodeTrace", window: slice) -> dict[str, float]:
    """The flatness metrics over the window, from the feet that have a sole (all NaN when none does)."""
    tilt = trace.sole_tilt_deg[window]
    has_sole = np.isfinite(tilt).all(axis=0) & np.isfinite(trace.home_sole_tilt_deg)
    if not has_sole.any():
        return {
            "max_sole_tilt_deg": NAN,
            "max_sole_tilt_excess_deg": NAN,
            "max_sole_abs_roll_deg": NAN,
            "sole_rolled_fraction": NAN,
            "max_sole_corner_lift_m": NAN,
            "min_sole_contacts": NAN,
            "max_sole_cop_outer": NAN,
        }
    home = trace.home_sole_tilt_deg[has_sole]
    mean_tilt = tilt[:, has_sole].mean(axis=0)
    mean_roll = _finite_column_means(trace.sole_roll_deg[window][:, has_sole])
    mean_lift = _finite_column_means(trace.sole_corner_lift[window][:, has_sole])
    mean_cop_outer = _finite_column_means(np.abs(trace.sole_cop[window][:, has_sole, 1]))
    return {
        "max_sole_tilt_deg": float(mean_tilt.max()),
        "max_sole_tilt_excess_deg": float((mean_tilt - home).max()),
        "max_sole_abs_roll_deg": _max_finite(np.abs(mean_roll)),
        "sole_rolled_fraction": _mean(np.any(tilt[:, has_sole] - home > SOLE_ROLLED_DEG, axis=1)),
        "max_sole_corner_lift_m": _max_finite(mean_lift),
        "min_sole_contacts": float(trace.sole_contacts_mean[window][:, has_sole].mean(axis=0).min()),
        "max_sole_cop_outer": _max_finite(mean_cop_outer),
    }


def _saturation_metrics(actions: np.ndarray) -> dict[str, float]:
    """Per-actuator saturated-step fractions over the window: NaN when any window step has no action."""
    actions = np.asarray(actions, dtype=np.float64)
    if not actions.size or not np.all(np.isfinite(actions)):
        return {"max_actuator_saturation_fraction": NAN, "mean_actuator_saturation_fraction": NAN}
    saturated = (np.abs(actions) >= SATURATION_ABS).mean(axis=0)
    return {
        "max_actuator_saturation_fraction": float(saturated.max()),
        "mean_actuator_saturation_fraction": float(saturated.mean()),
    }


#: The scalar fields computed over the post-settle window (all NaN when it is empty).
_WINDOW_FIELDS: tuple[str, ...] = (
    "all_feet_support",
    "flight_fraction",
    "window_airborne_substeps",
    "window_peak_floor_force_bw",
    "touchdown_rate",
    "min_foot_load_share",
    "min_foot_load_share_windowed",
    "window_displacement_m",
    "window_yaw_change_deg",
    "max_foot_slide_m",
    "foot_on_foot_fraction",
    "phantom_support_fraction",
    "touch_floor_agreement",
    "nonfoot_load_fraction",
    "offsupport_load_fraction",
    "leg_body_contact_fraction",
    "min_support_geom_duty",
    "min_support_geom_coverage",
    "max_sole_tilt_deg",
    "max_sole_tilt_excess_deg",
    "max_sole_abs_roll_deg",
    "sole_rolled_fraction",
    "max_sole_corner_lift_m",
    "min_sole_contacts",
    "max_sole_cop_outer",
    "max_actuator_saturation_fraction",
    "mean_actuator_saturation_fraction",
)
