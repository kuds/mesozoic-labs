"""The ``command_tracking/v1`` panel: which seeds, and the rollout that reduces them for the gate.

The panel
---------
A panel is a fixed list of reset seeds for a stage, rolled one episode per
seed with deterministic actions in the stage's own environment
(``env_kwargs``, its live ``[commands]`` table included).  The seeds come
from the publication seed family that the stance and recovery panels start
from (:data:`~environments.shared.constants.PUBLICATION_SEED_START`, 3042):
training resets the run's ``seed + rank`` and evaluates on ``seed + 1000``,
and a panel reset at a seed of this family replays no reset sequence the
policy fitted.

Under a live command the episode's command draw is a pure function of the
reset seed: ``reset(seed=S)`` gives ``np_random_seed = S`` and episode index
0, and ``BaseDinoEnv._command_rng`` seeds the controller from
``SeedSequence([S, 0, COMMAND_STREAM_DOMAIN])`` (decision D-D60).  The walk
profile's families are therefore fixed BY CONSTRUCTION, not by luck:
:func:`plan_command_panel` scans the seeds upward from the first panel seed,
reads each seed's reset draw through the same controller code
(:func:`reset_command_family`), and keeps the first ``min_cruise_episodes``
cruise seeds and the first ``min_stand_episodes`` stand seeds.  Both
families keep the stage's own distribution: a stand episode is one the
stage's ``stop_probability`` draws, never a command forced through
``set_direction``, so the panel judges exactly the episodes training sees.
A schedule that cannot fill a family within :data:`MAX_PANEL_SCAN` seeds
(``stop_probability`` 0 has no stand episode) leaves the family short, and
the gate fails closed on it.  The roll re-checks every episode's family
against the env's own draw and raises :class:`CommandPanelError` on a
mismatch: that is a defect in this replica, never a policy outcome.

The follow profile's panel is the first ``min_follow_episodes`` seeds of the
family, contiguous, as the stance panel's are.

The rollout
-----------
:func:`roll_command_panel` rolls any action callable ``predict(obs) ->
action`` (a loaded policy, the zero-action statue of
:func:`zero_action_predict`, a scripted controller), one fresh env for the
whole panel, closed in a ``finally``.  Each step's info is kept as scalars
with the step's end time, reward, termination flags and the root's radial
course progress, and the episode is reduced by
:func:`~environments.shared.curriculum.command_tracking_gate.reduce_command_episode`
under the stage's :class:`~environments.shared.curriculum.command_tracking_gate.PanelMeasurement`.
The floor-truth recorder (``gait.recorder.SubstepContactRecorder``) rides
the env's substep hook for the whole panel, and each episode's trace is
reduced to the gait clause's metrics
(``gait.locomotion_metrics.episode_locomotion_metrics``), which the record
carries in ``gait_metrics``.  The recorder only reads the physics, so a
recorded panel rolls the same trajectories as an unrecorded one.  Rolling the
same plan twice gives the same records.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .command_frame import COMMAND_MODE_NONE
from .constants import PUBLICATION_SEED_START
from .curriculum.command_evaluation import episode_command_family
from .curriculum.command_tracking_gate import (
    COMMAND_TRACKING_GATE_KIND,
    CRUISE_FAMILY,
    FOLLOW_PROFILE,
    STAND_FAMILY,
    CommandTrackingEpisode,
    CommandTrackingThresholds,
    FollowThresholds,
    PanelMeasurement,
    command_config_from,
    panel_measurement,
    reduce_command_episode,
    thresholds_from_curriculum,
)
from .direction_commands import DirectionCommandConfig

#: The most seeds :func:`plan_command_panel` examines before it leaves a family short.
MAX_PANEL_SCAN = 10_000


class CommandPanelError(RuntimeError):
    """The panel cannot be rolled as planned: a stage with no live command, or a reset that disagrees with its plan."""


def reset_command_family(command_config: DirectionCommandConfig, seed: int) -> str:
    """The walk family of the command ``reset(seed=seed)`` draws: ``stand`` for a zero requested speed, else ``cruise``.

    The env's own reset draw, replayed: the controller seeded from
    ``SeedSequence([seed, 0, COMMAND_STREAM_DOMAIN])`` as ``BaseDinoEnv``'s
    command stream seeds it on a seeded reset
    (:func:`~environments.shared.curriculum.command_evaluation.episode_command_family`
    at episode 0).  The requested speed does not depend on the spawn
    heading, so the heading passed here is immaterial.
    """
    return episode_command_family(command_config, int(seed), 0)


@dataclass(frozen=True)
class CommandPanelPlan:
    """The seeds a panel rolls, in order, each with the family its reset draws.

    ``scanned`` is how many seeds the planner examined from ``seed_start``
    (the follow panel's is its size).
    """

    profile: str
    seed_start: int
    seeds: tuple[int, ...]
    families: tuple[str, ...]
    scanned: int

    def family_counts(self) -> dict[str, int]:
        return {family: self.families.count(family) for family in (CRUISE_FAMILY, STAND_FAMILY)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "seed_start": self.seed_start,
            "seeds": list(self.seeds),
            "families": list(self.families),
            "scanned": self.scanned,
        }


def plan_command_panel(
    thresholds: CommandTrackingThresholds,
    command_config: DirectionCommandConfig,
    *,
    seed_start: int = PUBLICATION_SEED_START,
    max_scan: int = MAX_PANEL_SCAN,
) -> CommandPanelPlan:
    """The panel's seeds for a stage: walk families filled by construction, follow seeds contiguous.

    Walk: the seeds from *seed_start* upward, keeping each seed whose reset
    draw (:func:`reset_command_family`) is in a family still short of its
    minimum, until both are full or *max_scan* seeds were examined; the plan
    is in seed order.  Follow: ``seed_start`` to ``seed_start +
    min_follow_episodes - 1``.
    """
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise ValueError(f"seed_start must be a non-negative integer, not {seed_start!r}")
    if isinstance(thresholds, FollowThresholds):
        seeds = tuple(range(seed_start, seed_start + thresholds.min_follow_episodes))
        return CommandPanelPlan(
            profile=FOLLOW_PROFILE,
            seed_start=seed_start,
            seeds=seeds,
            families=tuple(reset_command_family(command_config, seed) for seed in seeds),
            scanned=len(seeds),
        )
    wanted = {CRUISE_FAMILY: thresholds.min_cruise_episodes, STAND_FAMILY: thresholds.min_stand_episodes}
    kept = {CRUISE_FAMILY: 0, STAND_FAMILY: 0}
    seeds_out: list[int] = []
    families_out: list[str] = []
    scanned = 0
    for seed in range(seed_start, seed_start + max_scan):
        if kept == wanted:
            break
        scanned += 1
        family = reset_command_family(command_config, seed)
        if kept[family] < wanted[family]:
            kept[family] += 1
            seeds_out.append(seed)
            families_out.append(family)
    return CommandPanelPlan(
        profile=thresholds.profile,
        seed_start=seed_start,
        seeds=tuple(seeds_out),
        families=tuple(families_out),
        scanned=scanned,
    )


@dataclass(frozen=True)
class CommandPanelRollout:
    """A rolled panel: the plan, the reduced episodes in plan order, the measurement and, on request, the step rows.

    ``traces`` holds each episode's floor-truth ``gait.recorder.EpisodeTrace``
    when the roll was asked to keep them (``keep_traces``).
    """

    plan: CommandPanelPlan
    episodes: tuple[CommandTrackingEpisode, ...]
    measurement: PanelMeasurement
    steps: tuple[tuple[dict[str, Any], ...], ...] | None = None
    traces: tuple[Any, ...] | None = None


def zero_action_predict(action_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """The statue: a predict that commands ``action = 0`` (the named home control) every step."""
    zero = np.zeros(int(action_dim), dtype=np.float32)

    def predict(_obs: np.ndarray) -> np.ndarray:
        return zero

    return predict


def _scalar_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """The step info's scalars as plain Python values (vectors and the terminal observation dropped)."""
    row: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            row[key] = value
    return row


def _live_command_kwargs(env_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], DirectionCommandConfig]:
    """*env_kwargs* with the command config as the dataclass, and that config; refused under ``"none"``."""
    kwargs = dict(env_kwargs)
    mode = kwargs.get("command_mode", COMMAND_MODE_NONE)
    try:
        config = command_config_from(kwargs.get("command_config"))
    except (TypeError, ValueError) as exc:
        raise CommandPanelError(f"the stage's command config is unreadable: {exc}") from exc
    if mode == COMMAND_MODE_NONE or config is None:
        raise CommandPanelError(
            f'{COMMAND_TRACKING_GATE_KIND} rolls a live command, and the stage runs command_mode "none" '
            "(it declares no [commands] table)"
        )
    kwargs["command_config"] = config
    return kwargs, config


def roll_command_panel(
    model_id: Any,
    stage_config: Mapping[str, Any],
    *,
    predict: Callable[[np.ndarray], Any],
    thresholds: CommandTrackingThresholds | None = None,
    plan: CommandPanelPlan | None = None,
    seed_start: int = PUBLICATION_SEED_START,
    plant_identity: Any = None,
    keep_steps: bool = False,
    keep_traces: bool = False,
) -> CommandPanelRollout:
    """Roll *predict* over the stage's panel and reduce every episode for the gate.

    *stage_config* is a loaded or recorded stage config (``env_kwargs`` and
    ``curriculum_kwargs``; a recorded ``command_config`` object is rebuilt).
    *thresholds* defaults to the stage's own ``[curriculum]`` block, which
    must then declare ``command_tracking/v1``; *plan* defaults to
    :func:`plan_command_panel` from *seed_start*.  *predict* maps an
    observation to an action; one with a ``reset`` attribute is reset before
    every episode, so a stateful controller carries nothing between episodes.
    With *plant_identity* the env is validated against it first.
    *keep_steps* keeps every episode's step rows (for charts), and
    *keep_traces* its floor-truth trace.

    The gait clause's metrics are measured on every episode: a
    ``gait.recorder.SubstepContactRecorder`` rides the env's substep hook for
    the whole panel (it reads the physics and never writes it, so the rollout
    is the one an unrecorded env would produce), and each episode's trace is
    reduced by ``gait.locomotion_metrics.episode_locomotion_metrics`` into the
    record's ``gait_metrics``.

    Raises :class:`CommandPanelError` for a stage with no live command, a
    plan that repeats a seed, pairs its seeds and families unevenly or was
    planned for the other profile, and a reset whose family disagrees with
    the plan.
    """
    env_kwargs, command_config = _live_command_kwargs(stage_config.get("env_kwargs") or {})
    if thresholds is None:
        curriculum = stage_config.get("curriculum_kwargs") or {}
        if curriculum.get("gate_kind") != COMMAND_TRACKING_GATE_KIND:
            raise CommandPanelError(
                f"the stage declares gate_kind {curriculum.get('gate_kind')!r}, not {COMMAND_TRACKING_GATE_KIND}; "
                "pass the thresholds to roll its panel"
            )
        thresholds = thresholds_from_curriculum(curriculum)
    measurement = panel_measurement(command_config, thresholds)
    if plan is None:
        plan = plan_command_panel(thresholds, command_config, seed_start=seed_start)
    if len(set(plan.seeds)) != len(plan.seeds):
        raise CommandPanelError("the panel plan repeats a reset seed")
    if len(plan.families) != len(plan.seeds):
        raise CommandPanelError(f"the panel plan names {len(plan.families)} families for {len(plan.seeds)} seeds")
    if plan.profile != thresholds.profile:
        raise CommandPanelError(
            f"the panel plan is a {plan.profile} panel and the thresholds a {thresholds.profile} gate"
        )

    # Imported here: the recorder and the morphology import mujoco and walk the model.
    from .gait.locomotion_metrics import episode_locomotion_metrics
    from .gait.morphology import Morphology
    from .gait.recorder import SubstepContactRecorder
    from .species_registry import env_class as model_env_class

    env = model_env_class(model_id)(**env_kwargs)
    episodes: list[CommandTrackingEpisode] = []
    kept: list[tuple[dict[str, Any], ...]] = []
    traces: list[Any] = []
    recorder: Any = None
    try:
        if plant_identity is not None:
            from .plant_contract import validate_environment_plant

            validate_environment_plant(env, plant_identity, artifact=f"{model_id} command-tracking panel environment")
        recorder = SubstepContactRecorder(env, Morphology.from_env(env, model_id)).attach()
        raw = env.unwrapped
        dt = float(raw.dt)
        horizon = int(raw.max_episode_steps)
        if not (math.isfinite(dt) and dt > 0.0 and horizon > 0):
            raise CommandPanelError("the environment's control step and horizon must be positive")
        reset_predict = getattr(predict, "reset", None)
        for index, (seed, family) in enumerate(zip(plan.seeds, plan.families)):
            obs, _ = env.reset(seed=int(seed))
            if callable(reset_predict):
                reset_predict()
            controller = raw.direction_controller
            # The heading the controller was reset at (no physics step since): the first event's
            # initial heading error is measured from it.
            initial_heading = float(raw._heading())
            planner = copy.deepcopy(controller)
            planner.update(horizon * dt, initial_heading)
            planned = [event for event in planner.events if event["start_time_s"] < horizon * dt - 1e-9]
            drawn = STAND_FAMILY if float(planned[0]["desired_speed"]) == 0.0 else CRUISE_FAMILY
            if drawn != family:
                raise CommandPanelError(
                    f"seed {seed}: the env's reset drew a {drawn} command and the plan expected {family}; the "
                    "planner's replica of the command stream disagrees with the env"
                )
            start_xy = np.asarray(raw.data.qpos[:2], dtype=np.float64).copy()
            rows: list[dict[str, Any]] = []
            terminated = truncated = False
            reason = ""
            for step in range(horizon):
                action = np.asarray(predict(obs), dtype=np.float32).ravel()
                obs, reward, terminated, truncated, info = env.step(action)
                recorder.end_step(action, float(reward))
                reason = str(info.get("termination_reason") or "")
                row = _scalar_info(info)
                row.update(
                    episode=index,
                    episode_seed=int(seed),
                    step=step + 1,
                    time_s=(step + 1) * dt,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    course_progress_m=float(np.linalg.norm(np.asarray(raw.data.qpos[:2]) - start_xy)),
                )
                rows.append(row)
                if terminated or truncated:
                    break
            trace = recorder.end_episode(
                terminated=bool(terminated), truncated=bool(truncated), termination_reason=reason
            )
            episodes.append(
                reduce_command_episode(
                    rows,
                    episode=index,
                    seed=int(seed),
                    dt=dt,
                    horizon=horizon,
                    planned_events=planned,
                    initial_heading=initial_heading,
                    measurement=measurement,
                    gait_metrics=episode_locomotion_metrics(trace).as_gait_metrics(),
                )
            )
            if keep_steps:
                kept.append(tuple(rows))
            if keep_traces:
                traces.append(trace)
    finally:
        try:
            if recorder is not None:
                recorder.detach()
        finally:
            env.close()
    return CommandPanelRollout(
        plan=plan,
        episodes=tuple(episodes),
        measurement=measurement,
        steps=tuple(kept) if keep_steps else None,
        traces=tuple(traces) if keep_traces else None,
    )
