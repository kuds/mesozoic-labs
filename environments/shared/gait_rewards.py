"""Gait reward kit: contact terms that pay a walk, inert at their legacy values.

GAIT_QUALITY_PLAN_2026_09 §5.  The 2026-09 audit replayed the certified T. rex
walkers and found two-footed hops: the locomotion reward paid the alive bonus
and the forward speed whether the feet were on the floor or not (alive 164 and
205, forward 261 and 475 per episode paid while airborne for seeds 42 and 44),
and nothing in it rewarded alternation.  This module holds the terms that price
that, generic over the foot registry of :mod:`environments.shared.gait.morphology`
(bipeds now; the four-legged phase templates are not implemented and refuse).
A species env wires it through constructor kwargs (T. rex only, for now; see
``TRexEnv``), one per knob of :class:`GaitRewardConfig`:

``support_source`` (``"touch"``)
    The force the support terms read for each foot: ``"touch"`` is the species'
    touch-sensor sum (the per-foot MIN over the control step's substeps, the
    legacy reading); ``"floor"`` is the floor normal force on the foot's
    registered geometries (``FOOT_GEOMETRIES``), MIN over the same substeps,
    which is the floor truth the gait gate measures.  It feeds the species'
    bilateral terms (on T. rex ``bilateral_support_weight``,
    ``foot_contact_saturation_force``, ``support_conditioned_alive_fraction``
    and the foot load balance) and the feet-down flags below.  The touch info
    keys (``r_foot_contact``) stay touch either way: the stance gate reads them.
``gait_phase_weight`` (0), ``gait_phase_step_over_leg`` (0.64)
    A clock-free event reward at each step of foot i, a debounced floor-truth
    touchdown whose stance bears load:
    ``w * min(T_i, 5 T_sw) / (feet * dt) * P * A * S``.

    * ``T_i`` is the time since foot i's previous step (its stride period) and
      ``dt`` the control step.  Summed over a stride the events of all feet pay
      ``w * P * A * S`` per control step whatever the cadence, so tapping fast
      earns no more than walking slowly; a stride is credited at most five
      reference swings (1.95 s on T. rex), so a pause between steps is not paid
      as stepping time.
    * ``P``, alternation: the other foot's latest step must fall strictly
      inside foot i's stride, at relative phase ``phi`` in (0, 1);
      ``P = exp(-((phi - 0.5) / 0.15)^2)``, and 0 when ``phi`` is within 0.1 of
      0 or 1 (the two feet landed within a tenth of a stride of each other, a
      hop) or when the other foot has not stepped since foot i's previous step
      (one foot hopping, or tapping beside a standing foot).
    * ``A``, swing: ``min(swing, T_sw) / T_sw`` with ``T_sw = 1.3 sqrt(L / g)``
      (0.39 s on T. rex), the swing being the time from the foot's lift-off to
      this touchdown: a tap or an impact bounce earns almost nothing.
    * ``S``, step-through: ``clip(min(advance, stride / 2) / l, 0, 1)``.
      ``advance`` is the landing footprint's advance past the other foot's
      latest footprint along the trunk heading, as the walk-first checker
      measures a step (a footprint is the load-weighted centre of the stance's
      foot-site positions; the heading is the root body's x axis on the floor,
      averaged over the two steps' moments of loading), and ``stride`` the
      foot's own advance along the same heading from its previous step's
      touchdown to this one (the checker's in-place stride test); in a walk
      ``advance`` is half the ``stride``.  ``l = gait_phase_step_over_leg * L``
      (0.56 m on T. rex).  A foot that lands level with or behind the other (a
      step-to's trailing foot, a staggered hop's trailing foot) earns 0, and
      so does a foot that goes nowhere: marching in place, even in a split
      stance whose front foot lands ahead of the other at every step, and feet
      sliding back under a body that stays put.

    A step is a touchdown (a load above 0.01 body weight per foot lasting at
    least 6 ms; unloads of up to 10 ms are chatter inside the stance: the
    checker's segmentation values) whose stance has borne half of the foot's
    share of body weight for 30 ms in all; it is paid in the substep that
    completes the 30 ms, about 30 ms after first contact in a walk.  A tap is
    not a step, for itself or as the other foot of P, even when its impact
    transient reaches that load for a substep or two (33 of the 34 contacts
    shorter than 30 ms in the certified T. rex hop runs reach it, for 9 ms at
    the median).  The stance the feet hold through the reset is not a step and
    the first step of each foot only starts its stride clock, so a statue earns
    0 and the reset never counts as a touchdown.  Statue, two-footed hop,
    staggered hop (lag under a tenth of a stride), one-legged hop whose other
    foot only taps, tapping beside a standing foot, marching in place (level or
    in a split stance) and feet sliding back in place earn 0; a step-to earns
    its leading foot's credit for half its stride (a quarter of a walk's rate
    at lead steps of ``l``), and taps that creep forward a few percent of a
    walk.
``flight_penalty_weight`` (0), ``flight_min_feet`` (1)
    ``-w`` on control steps with fewer than ``flight_min_feet`` feet down (1
    for a biped walk, 2 for a quadruped walk).  A foot is down for a control
    step when its support force (``support_source``) stays at or above a
    tenth of its share of body weight on every substep (the checker's
    light-stance bound, on the per-foot MIN the support terms read).  So every
    control step with an airborne substep is a flight step, and so is one that
    hands the load from foot to foot with less than a control step of double
    support; a walk's double support spans several control steps.
``foot_slip_penalty_weight`` (0)
    ``-w * sum(|v_slip|) / sqrt(g L)`` over the feet down, with ``v_slip`` the
    force-weighted tangential velocity of the foot's floor contacts at the
    control step's last substep (the gait recorder's slip arithmetic; the
    terrain is static).  A foot rolling over its planted toes does not slip.
``foot_collision_penalty_weight`` (0)
    ``-w`` on control steps with a contact between two feet in any substep.
``leg_contact_penalty_weight`` (0), ``terminate_on_leg_contact`` (False)
    ``-w`` on control steps in which a registered thigh or shank geometry
    (:data:`LEG_GEOMETRIES`) touches the terrain in any substep, and, with
    ``terminate_on_leg_contact``, the episode ends there (reason
    ``leg_contact``, after the species' own checks).

Inert at legacy values (§5.2).  With every knob at its value in
:data:`GAIT_REWARD_KIT_LEGACY` the kit resolves nothing, installs no substep
hook, adds no info key and leaves the reward sum and its arithmetic untouched,
so every stage's reward, info and termination values are bit for bit the
pre-kit ones.  The task fingerprint and ``stage_config.json`` leave out every
knob a stage does not set whose default equals its legacy value
(:func:`drop_inert_kit_keys`), so a stage that sets none keeps its
``task_sha256`` and recorded config.  Setting a knob in a stage TOML is the
task revision (``gait-r1``).  The legacy values are pinned: changing one, or a
constructor default away from it, moves the digest of every stage that relies
on it.

Measurement.  The floor force and the contact flags come from the env's
per-substep reward hook (``BaseDinoEnv._substep_reward_hook``), installed only
when a knob needs them: a read-only scan of the solved contacts after each
``mj_step``, with no extra dynamics call, no random draw and no change to the
state, so every knob except ``terminate_on_leg_contact`` leaves the trajectory
of a given action sequence unchanged (the plan's §5.5 re-scoring relies on
it).  The kit's state resets with ``BaseDinoEnv._reset_gait_state``, which
every species' ``_spawn_target`` calls.  Scored without a step (a hand-posed
state), the terms read the current contacts and pay no step event.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

SUPPORT_SOURCES = ("touch", "floor")

#: Thigh and shank geometries per species: their floor contact is a knee, a
#: shin or a hip on the ground.  Registered explicitly, like the foot registry;
#: a species without an entry refuses the leg-contact knobs.
LEG_GEOMETRIES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "trex": ("r_thigh_geom", "r_tibia_geom", "l_thigh_geom", "l_tibia_geom"),
    }
)

# Contact segmentation, from the gait checker's GaitProtocol (v2) defaults and
# pinned here: the reward's arithmetic must not move when the measurement's
# protocol does.
#: A foot touches the floor above this fraction of body weight per foot.
CONTACT_FORCE_BW_PER_FOOT = 0.01
#: A load shorter than this is a blip, not a touchdown (seconds).
BLIP_S = 0.006
#: An unload no longer than this is chatter inside a stance (seconds).
CHATTER_FILL_S = 0.010

#: A touchdown is a step once its stance has borne this fraction of the foot's share of body weight ...
STEP_LOAD_BW_PER_FOOT = 0.5
#: ... for this long in all (seconds): an impact spike of a tap reaches the load for a substep or two.
STEP_HOLD_S = 0.03
#: A foot is down for a control step when its support force stays at or above this fraction of its share.
SUPPORT_LOAD_BW_PER_FOOT = 0.1
#: Width of the alternation factor P around anti-phase, in cycles (the checker's template tolerance).
PHASE_TOLERANCE = 0.15
#: P is 0 when the other foot stepped within this fraction of the stride of either step of the landing foot.
SIMULTANEOUS_FRACTION = 0.1
#: The reference swing ``T_sw = SWING_REFERENCE_FACTOR * sqrt(L / g)``.
SWING_REFERENCE_FACTOR = 1.3
#: A stride is credited at most this many reference swings.
STRIDE_CREDIT_SWINGS = 5.0
#: Qualified steps remembered per foot (P looks for the other foot's latest step inside a stride).
_STEP_HISTORY = 4


@dataclass(frozen=True)
class GaitRewardConfig:
    """The kit's knobs; the defaults are the legacy values (:data:`GAIT_REWARD_KIT_LEGACY`)."""

    support_source: str = "touch"
    gait_phase_weight: float = 0.0
    gait_phase_step_over_leg: float = 0.64
    flight_penalty_weight: float = 0.0
    flight_min_feet: int = 1
    foot_slip_penalty_weight: float = 0.0
    foot_collision_penalty_weight: float = 0.0
    leg_contact_penalty_weight: float = 0.0
    terminate_on_leg_contact: bool = False

    def __post_init__(self) -> None:
        if self.support_source not in SUPPORT_SOURCES:
            raise ValueError(f"support_source must be one of {SUPPORT_SOURCES}, got {self.support_source!r}")
        for name in (
            "gait_phase_weight",
            "flight_penalty_weight",
            "foot_slip_penalty_weight",
            "foot_collision_penalty_weight",
            "leg_contact_penalty_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
        step = self.gait_phase_step_over_leg
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
            raise ValueError(f"gait_phase_step_over_leg must be a finite positive number, got {step!r}")
        if isinstance(self.flight_min_feet, bool) or not isinstance(self.flight_min_feet, int):
            raise ValueError(f"flight_min_feet must be an integer, got {self.flight_min_feet!r}")
        if self.flight_min_feet < 1:
            raise ValueError("flight_min_feet must be at least 1")
        if not isinstance(self.terminate_on_leg_contact, bool):
            raise ValueError(f"terminate_on_leg_contact must be a bool, got {self.terminate_on_leg_contact!r}")


#: Every kit knob and its legacy value, the value that reproduces the pre-kit
#: reward arithmetic bit for bit (GAIT_QUALITY_PLAN_2026_09 §5.2 item 1).
#: Pinned: :func:`drop_inert_kit_keys` leaves a knob out of a stage's task
#: fingerprint and recorded config only while it is unset and equal to this
#: value, so the table is never retuned.
GAIT_REWARD_KIT_LEGACY: MappingProxyType[str, Any] = MappingProxyType(asdict(GaitRewardConfig()))


def drop_inert_kit_keys(effective: MutableMapping[str, Any], explicit: Collection[str]) -> None:
    """Pop each kit knob absent from *explicit* (the stage's ``[env]``) and equal to its legacy value.

    The carve-out of GAIT_QUALITY_PLAN_2026_09 §5.2 item 2, applied by the
    task fingerprint's effective config and by ``save_stage_config`` alike: a
    stage that sets no knob records the pre-kit config, while a knob the stage
    sets, even to its legacy value, and a default retuned away from it stay in.
    Species without the kit have no such keys, so it is a no-op for them.
    """
    for name, legacy in GAIT_REWARD_KIT_LEGACY.items():
        if name in effective and name not in explicit:
            value = effective[name]
            if type(value) is type(legacy) and value == legacy:
                del effective[name]


@dataclass
class _Stance:
    """One debounced stance of one foot: its touchdown, the swing before it and its load-weighted footprint.

    The footprint accumulates about the stance's first sample, so a foot that
    does not move during its stance has exactly that position as footprint.
    """

    start_s: float | None
    swing_s: float | None
    origin: tuple[float, float] | None = None
    load: float = 0.0
    load_dx: float = 0.0
    load_dy: float = 0.0
    #: Samples at or above the step load.
    held: int = 0
    step: bool = False
    axis: tuple[float, float] = (0.0, 0.0)

    def add(self, force: float, x: float, y: float, step_load_n: float) -> None:
        if self.origin is None:
            self.origin = (x, y)
        self.load += force
        self.load_dx += force * (x - self.origin[0])
        self.load_dy += force * (y - self.origin[1])
        if force >= step_load_n:
            self.held += 1

    def footprint(self) -> tuple[float, float]:
        assert self.origin is not None
        return self.origin[0] + self.load_dx / self.load, self.origin[1] + self.load_dy / self.load


@dataclass
class _Foot:
    """Per-foot segmentation state."""

    stance: _Stance | None = None
    #: Loaded samples ``(time, force, x, y)`` of a load not yet past the blip debounce.
    pending: list[tuple[float, float, float, float]] = field(default_factory=list)
    #: Unloaded samples ``(force, x, y)`` inside a stance, not yet a lift-off.
    unloaded: list[tuple[float, float, float]] = field(default_factory=list)
    unload_start_s: float = 0.0
    liftoff_s: float | None = None
    steps: list[_Stance] = field(default_factory=list)


class GaitPhaseTracker:
    """Floor-truth steps and the gait-phase event reward of a biped, one substep sample at a time.

    Pure bookkeeping, no simulator: :meth:`sample` takes each substep's per-foot
    floor force and foot-site position, in the registry's foot order, and
    returns what the steps completed at that sample pay (see the module
    docstring for the terms).  :meth:`reset` starts a new episode; the first
    sample after it only records which feet are down.
    """

    def __init__(
        self,
        *,
        feet: int,
        substep_s: float,
        control_dt: float,
        body_weight_n: float,
        leg_length_m: float,
        gravity: float,
        weight: float,
        step_over_leg: float,
    ):
        if feet != 2:
            raise ValueError(
                "gait_phase_weight is implemented for bipeds only; the quadruped templates "
                "(lateral_walk, trot) of GAIT_QUALITY_PLAN_2026_09 §5.3 are not"
            )
        for name, value in (
            ("substep_s", substep_s),
            ("control_dt", control_dt),
            ("body_weight_n", body_weight_n),
            ("leg_length_m", leg_length_m),
            ("gravity", gravity),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        self.feet = feet
        self.substep_s = substep_s
        self.contact_n = CONTACT_FORCE_BW_PER_FOOT * body_weight_n / feet
        self.step_load_n = STEP_LOAD_BW_PER_FOOT * body_weight_n / feet
        self.swing_reference_s = SWING_REFERENCE_FACTOR * math.sqrt(leg_length_m / gravity)
        self.step_length_m = step_over_leg * leg_length_m
        self.credit_cap_s = STRIDE_CREDIT_SWINGS * self.swing_reference_s
        self.scale = weight / (feet * control_dt)
        # Sample counts of the checker's debounce: a load lasting k samples is
        # held for k * substep_s; an unload of k samples spans (k + 1) *
        # substep_s from the last loaded sample to the next one.
        self._blip_samples = max(1, math.ceil(BLIP_S / substep_s - 1e-9))
        self._liftoff_samples = max(1, math.floor(CHATTER_FILL_S / substep_s + 1e-9))
        self._hold_samples = max(1, math.ceil(STEP_HOLD_S / substep_s - 1e-9))
        self._feet: list[_Foot] | None = None
        self._samples = 0

    def reset(self) -> None:
        self._feet = None
        self._samples = 0

    def sample(
        self,
        forces: Sequence[float] | np.ndarray,
        foot_xy: Any,
        trunk_axis: Callable[[], tuple[float, float]],
    ) -> tuple[float, int, float]:
        """Advance one substep: returns ``(reward, steps, quality)`` of the steps made at this sample.

        *trunk_axis* is called only when a step completes; it returns the root
        body's x axis projected on the floor (any length; zero when undefined).
        """
        self._samples += 1
        time_s = self._samples * self.substep_s
        if self._feet is None:
            # Whatever bears load now has stood since the reset: a stance, but no step.
            self._feet = [_Foot() for _ in range(self.feet)]
            for foot, force, (x, y) in zip(self._feet, forces, foot_xy, strict=True):
                force = float(force)
                if force > self.contact_n:
                    foot.stance = _Stance(start_s=None, swing_s=None)
                    foot.stance.add(force, float(x), float(y), self.step_load_n)
            return 0.0, 0, 0.0
        completed: list[int] = []
        for index, (foot, force, (x, y)) in enumerate(zip(self._feet, forces, foot_xy, strict=True)):
            force, x, y = float(force), float(x), float(y)
            loaded = force > self.contact_n
            stance = foot.stance
            if stance is None:
                if not loaded:
                    foot.pending.clear()
                    continue
                foot.pending.append((time_s, force, x, y))
                if len(foot.pending) < self._blip_samples:
                    continue
                start = foot.pending[0][0]
                stance = _Stance(start_s=start, swing_s=None if foot.liftoff_s is None else start - foot.liftoff_s)
                for _, pending_force, pending_x, pending_y in foot.pending:
                    stance.add(pending_force, pending_x, pending_y, self.step_load_n)
                foot.pending.clear()
                foot.stance = stance
            elif loaded:
                # An unload that ended within the chatter fill stays inside the stance.
                for unloaded_force, unloaded_x, unloaded_y in foot.unloaded:
                    stance.add(unloaded_force, unloaded_x, unloaded_y, self.step_load_n)
                foot.unloaded.clear()
                stance.add(force, x, y, self.step_load_n)
            else:
                if not foot.unloaded:
                    foot.unload_start_s = time_s
                foot.unloaded.append((force, x, y))
                if len(foot.unloaded) >= self._liftoff_samples:
                    foot.liftoff_s = foot.unload_start_s
                    foot.unloaded.clear()
                    foot.stance = None
                continue
            if not stance.step and stance.start_s is not None and stance.held >= self._hold_samples:
                stance.step = True
                completed.append(index)
        if not completed:
            return 0.0, 0, 0.0
        axis = trunk_axis()
        reward = 0.0
        quality_sum = 0.0
        for index in completed:
            step_reward, quality = self._step(index, axis)
            reward += step_reward
            quality_sum += quality
        return reward, len(completed), quality_sum

    def _step(self, index: int, axis: tuple[float, float]) -> tuple[float, float]:
        """Record foot *index*'s completed step and return ``(reward, P * A * S)``."""
        assert self._feet is not None
        foot = self._feet[index]
        stance = foot.stance
        assert stance is not None and stance.start_s is not None
        stance.axis = (float(axis[0]), float(axis[1]))
        previous = foot.steps[-1] if foot.steps else None
        foot.steps.append(stance)
        del foot.steps[:-_STEP_HISTORY]
        if previous is None or previous.start_s is None:
            return 0.0, 0.0  # the first step after the reset starts the stride clock
        period = stance.start_s - previous.start_s
        partner = None
        for candidate in reversed(self._feet[1 - index].steps):
            assert candidate.start_s is not None
            if previous.start_s < candidate.start_s <= stance.start_s:
                partner = candidate
                break
        if partner is None or partner.start_s is None or period <= 0.0:
            return 0.0, 0.0
        phase = (partner.start_s - previous.start_s) / period
        if phase < SIMULTANEOUS_FRACTION or phase > 1.0 - SIMULTANEOUS_FRACTION:
            return 0.0, 0.0
        p = math.exp(-(((phase - 0.5) / PHASE_TOLERANCE) ** 2))
        swing = 0.0 if stance.swing_s is None else stance.swing_s
        a = min(swing, self.swing_reference_s) / self.swing_reference_s
        heading_x = stance.axis[0] + partner.axis[0]
        heading_y = stance.axis[1] + partner.axis[1]
        length = math.hypot(heading_x, heading_y)
        if length <= 1e-9:
            return 0.0, 0.0
        own_x, own_y = stance.footprint()
        other_x, other_y = partner.footprint()
        advance = ((own_x - other_x) * heading_x + (own_y - other_y) * heading_y) / length
        # Capped at half the foot's own stride, touchdown to touchdown: a foot
        # marking time in a split stance lands ahead of the other every step
        # but goes nowhere, and feet sliding back under a body that stays put
        # land where they landed before.
        assert stance.origin is not None and previous.origin is not None
        stride_x, stride_y = stance.origin[0] - previous.origin[0], stance.origin[1] - previous.origin[1]
        half_stride = 0.5 * (stride_x * heading_x + stride_y * heading_y) / length
        s = min(max(min(advance, half_stride) / self.step_length_m, 0.0), 1.0)
        quality = p * a * s
        return self.scale * min(period, self.credit_cap_s) * quality, quality


def flight_term(support_n: Sequence[float], down_n: float, min_feet: int, weight: float) -> tuple[float, int, bool]:
    """``(reward, feet down, flight)``: ``-weight`` when fewer than *min_feet* feet carry *down_n*."""
    feet_down = sum(1 for force in support_n if force >= down_n)
    flight = feet_down < min_feet
    return (0.0 - weight) if flight else 0.0, feet_down, flight


def slip_term(
    slip_mps: Sequence[float] | np.ndarray,
    support_n: Sequence[float],
    down_n: float,
    speed_scale_mps: float,
    weight: float,
) -> tuple[float, float]:
    """``(reward, slip)``: ``slip`` sums the slip speed of the feet down over *speed_scale_mps* (``sqrt(g L)``)."""
    total = 0.0
    for speed, force in zip(slip_mps, support_n, strict=True):
        if force >= down_n:
            total += float(speed)
    slip = total / speed_scale_mps
    return 0.0 - weight * slip, slip


class GaitRewardKit:
    """The kit bound to one species env (see the module docstring).

    Construct it after the env's ``super().__init__`` (it reads the foot
    registry, the touch-sensor groups and the terrain); the env calls
    :meth:`support_forces` and :meth:`step_terms` from its reward,
    :meth:`leg_contact_terminates` from its termination, and the base env
    resets it.  At the legacy config :attr:`active` is False and every call
    returns its input or nothing.
    """

    def __init__(self, env: Any, species: str, config: GaitRewardConfig):
        self.config = config
        self.active = config != GaitRewardConfig()
        self._env = env
        self._floor_support = config.support_source == "floor"
        self._track_steps = config.gait_phase_weight > 0.0
        self._track_floor = self._floor_support or self._track_steps
        self._track_collision = config.foot_collision_penalty_weight > 0.0
        self._track_legs = config.leg_contact_penalty_weight > 0.0 or config.terminate_on_leg_contact
        self._tracker: GaitPhaseTracker | None = None
        self._step_tag = -1
        self._min_floor: np.ndarray | None = None
        self._foot_collision = False
        self._leg_contact = False
        self._phase_reward = 0.0
        self._phase_steps = 0
        self._phase_quality = 0.0
        if not self.active:
            return
        from .gait.morphology import GaitMorphology

        morphology = GaitMorphology.from_env(env, species)
        model = env.model
        self.foot_names = morphology.foot_names
        self._feet = len(morphology.foot_names)
        if config.flight_min_feet > self._feet:
            raise ValueError(f"flight_min_feet {config.flight_min_feet} exceeds the {self._feet} feet of {species}")
        self._foot_sites = np.asarray(morphology.foot_site_ids, dtype=np.int64)
        # Lookup tables indexed by geom id, one entry longer than ngeom so the
        # geom id -1 of a flex contact reads the trailing "none" entry.
        self._foot_of = np.append(np.asarray(morphology.geom_foot, dtype=np.int64), -1)
        self._root_qpos = int(morphology.root_qpos_address)
        gravity = float(np.linalg.norm(model.opt.gravity))
        self.body_weight_n = morphology.body_weight_n
        self.leg_length_m = morphology.leg_length_m
        self.down_n = SUPPORT_LOAD_BW_PER_FOOT * self.body_weight_n / self._feet
        self._speed_scale = math.sqrt(gravity * self.leg_length_m)
        self._is_leg = np.zeros(model.ngeom + 1, dtype=bool)
        self._terrain_ids: np.ndarray | None = None
        self._is_terrain = np.zeros(model.ngeom + 1, dtype=bool)
        if self._track_legs:
            if species not in LEG_GEOMETRIES:
                raise ValueError(f"no leg geometry registry for {species}; the leg-contact knobs need one")
            for name in LEG_GEOMETRIES[species]:
                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom < 0:
                    raise ValueError(f"{species} is missing registered leg geometry {name!r}")
                if self._foot_of[geom] >= 0:
                    raise ValueError(f"registered leg geometry {name!r} is also a foot geometry")
                self._is_leg[geom] = True
        if self._track_steps:
            self._tracker = GaitPhaseTracker(
                feet=self._feet,
                substep_s=float(model.opt.timestep),
                control_dt=float(env.dt),
                body_weight_n=self.body_weight_n,
                leg_length_m=self.leg_length_m,
                gravity=gravity,
                weight=float(config.gait_phase_weight),
                step_over_leg=float(config.gait_phase_step_over_leg),
            )
        self._jacobian = np.zeros((3, model.nv), dtype=np.float64)
        self._wrench = np.zeros(6, dtype=np.float64)
        self.reset()
        if self._track_floor or self._track_collision or self._track_legs:
            env._substep_reward_hook = self._record_substep

    # ------------------------------------------------------------------
    # episode state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Start a new episode: no aggregates, no steps."""
        self.invalidate()
        if self._tracker is not None:
            self._tracker.reset()

    def invalidate(self) -> None:
        """Drop the current control step's aggregates (a hand-posed state is then read live)."""
        self._step_tag = -1
        self._min_floor = None
        self._foot_collision = False
        self._leg_contact = False
        self._phase_reward = 0.0
        self._phase_steps = 0
        self._phase_quality = 0.0

    def _fresh(self) -> bool:
        return bool(self._step_tag == self._env._step_count)

    # ------------------------------------------------------------------
    # measurement
    # ------------------------------------------------------------------

    def _terrain_lookup(self) -> np.ndarray:
        """Which geoms are terrain: the env's static floor geoms (the behavior env swaps them with its model)."""
        ids = self._env._static_floor_geoms()
        if ids is not self._terrain_ids:
            self._is_terrain[:] = False
            self._is_terrain[ids] = True
            self._terrain_ids = ids
        return self._is_terrain

    def _scan(self, floor: bool, flags: bool) -> tuple[np.ndarray | None, bool, bool]:
        """Read the solved contacts: per-foot floor force, foot-on-foot contact, leg-on-terrain contact."""
        env = self._env
        model, data = env.model, env.data
        forces = np.zeros(self._feet) if floor else None
        collision = leg = False
        if data.ncon == 0:
            return forces, collision, leg
        pairs = data.contact.geom
        g1, g2 = pairs[:, 0], pairs[:, 1]
        foot1, foot2 = self._foot_of[g1], self._foot_of[g2]
        terrain = self._terrain_lookup()
        on1, on2 = terrain[g1], terrain[g2]
        if flags and self._track_collision:
            collision = bool(((foot1 >= 0) & (foot2 >= 0) & (foot1 != foot2)).any())
        if flags and self._track_legs:
            leg = bool(((on1 & self._is_leg[g2]) | (on2 & self._is_leg[g1])).any())
        if forces is not None:
            selected = ((on1 & (foot2 >= 0)) | (on2 & (foot1 >= 0))) & (data.contact.efc_address >= 0)
            for ci in np.flatnonzero(selected).tolist():
                mujoco.mj_contactForce(model, data, ci, self._wrench)
                force = float(self._wrench[0])
                if force > 0.0:
                    forces[foot2[ci] if on1[ci] else foot1[ci]] += force
        return forces, collision, leg

    def _record_substep(self) -> None:
        """The env's per-substep hook: aggregate this control step's contacts and advance the steps."""
        env = self._env
        step = env._step_count + 1  # the control step the substep belongs to (counted after the loop)
        if self._step_tag != step:
            self.invalidate()
            self._step_tag = step
        forces, collision, leg = self._scan(self._track_floor, True)
        self._foot_collision |= collision
        self._leg_contact |= leg
        if forces is None:
            return
        self._min_floor = forces if self._min_floor is None else np.minimum(self._min_floor, forces)
        if self._tracker is not None:
            reward, steps, quality = self._tracker.sample(
                forces, env.data.site_xpos[self._foot_sites, :2], self._trunk_axis
            )
            self._phase_reward += reward
            self._phase_steps += steps
            self._phase_quality += quality

    def _trunk_axis(self) -> tuple[float, float]:
        """The root body's x axis on the floor (the walk-first checker's trunk axis), unnormalised."""
        w, x, y, z = (float(v) for v in self._env.data.qpos[self._root_qpos + 3 : self._root_qpos + 7])
        norm = w * w + x * x + y * y + z * z
        if norm <= 0.0:
            return 0.0, 0.0
        return 1.0 - 2.0 * (y * y + z * z) / norm, 2.0 * (x * y + w * z) / norm

    def _floor_forces(self) -> np.ndarray:
        if self._fresh() and self._min_floor is not None:
            return self._min_floor
        forces, _, _ = self._scan(True, False)
        assert forces is not None
        return forces

    def _flags(self) -> tuple[bool, bool]:
        if self._fresh():
            return self._foot_collision, self._leg_contact
        _, collision, leg = self._scan(False, True)
        return collision, leg

    def _slip_speeds(self) -> np.ndarray:
        """Force-weighted tangential slip speed of each foot's floor contacts now (the gait recorder's arithmetic)."""
        env = self._env
        model, data = env.model, env.data
        weighted = np.zeros(self._feet)
        load = np.zeros(self._feet)
        if data.ncon == 0:
            return weighted
        pairs = data.contact.geom
        g1, g2 = pairs[:, 0], pairs[:, 1]
        foot1, foot2 = self._foot_of[g1], self._foot_of[g2]
        terrain = self._terrain_lookup()
        on1, on2 = terrain[g1], terrain[g2]
        selected = ((on1 & (foot2 >= 0)) | (on2 & (foot1 >= 0))) & (data.contact.efc_address >= 0)
        for ci in np.flatnonzero(selected).tolist():
            contact = data.contact[ci]
            animal = int(g2[ci] if on1[ci] else g1[ci])
            mujoco.mj_contactForce(model, data, ci, self._wrench)
            force = float(self._wrench[0])
            if force <= 0.0:
                continue
            point = np.asarray(contact.pos)
            mujoco.mj_jac(model, data, self._jacobian, None, point, int(model.geom_bodyid[animal]))
            velocity = self._jacobian @ data.qvel
            normal = np.asarray(contact.frame[:3])
            tangential = velocity - normal * float(velocity @ normal)
            foot = int(self._foot_of[animal])
            weighted[foot] += force * float(tangential @ tangential)
            load[foot] += force
        return np.sqrt(np.divide(weighted, load, out=np.zeros(self._feet), where=load > 0.0))

    # ------------------------------------------------------------------
    # the env's calls
    # ------------------------------------------------------------------

    def support_forces(self, touch: tuple[float, ...]) -> tuple[float, ...]:
        """The per-foot force the support terms read: *touch* itself under ``"touch"``, floor force under ``"floor"``."""
        if not self._floor_support:
            return touch
        return tuple(float(value) for value in self._floor_forces())

    def step_terms(self, info: dict[str, Any], support: tuple[float, ...]) -> float:
        """Add the kit's terms and their diagnostics to *info*; return their sum (0.0 and nothing added when inert).

        *support* is what :meth:`support_forces` returned for this control step.
        """
        if not self.active:
            return 0.0
        config = self.config
        for name, force in zip(self.foot_names, support, strict=True):
            info[f"gait_support_force_{name}"] = float(force)
        reward_flight, feet_down, flight = flight_term(
            support, self.down_n, config.flight_min_feet, config.flight_penalty_weight
        )
        info["gait_feet_down"] = float(feet_down)
        info["gait_flight"] = 1.0 if flight else 0.0
        info["reward_flight"] = reward_flight
        if config.foot_slip_penalty_weight > 0.0:
            reward_slip, slip = slip_term(
                self._slip_speeds(), support, self.down_n, self._speed_scale, config.foot_slip_penalty_weight
            )
        else:
            reward_slip, slip = 0.0, 0.0
        info["gait_foot_slip"] = slip
        info["reward_foot_slip"] = reward_slip
        collision, leg = self._flags() if (self._track_collision or self._track_legs) else (False, False)
        reward_collision = (0.0 - config.foot_collision_penalty_weight) if collision else 0.0
        reward_leg = (0.0 - config.leg_contact_penalty_weight) if leg else 0.0
        info["gait_foot_collision"] = 1.0 if collision else 0.0
        info["reward_foot_collision"] = reward_collision
        info["gait_leg_contact"] = 1.0 if leg else 0.0
        info["reward_leg_contact"] = reward_leg
        fresh = self._fresh()
        reward_phase = self._phase_reward if fresh else 0.0
        info["gait_phase_steps"] = float(self._phase_steps if fresh else 0)
        info["gait_phase_quality"] = self._phase_quality if fresh else 0.0
        info["reward_gait_phase"] = reward_phase
        return reward_flight + reward_slip + reward_collision + reward_leg + reward_phase

    def leg_contact_terminates(self) -> bool:
        """True when ``terminate_on_leg_contact`` is set and a leg touched the terrain this control step."""
        if not self.config.terminate_on_leg_contact:
            return False
        return self._flags()[1]
