"""The in-training evaluation of a live-command stage: one command panel, scored by family (decision D-D62).

A walk's episodes come in two families, fixed at reset by the command
stream: cruise (a forward command) and stand (an exactly zero command, with
probability ``stop_probability``).  A stand episode earns several times a
cruise episode (the stand block pays the stance's reward), so two readings
of the in-training evaluation that are sound on a command-free stage go
wrong on a walk:

* **The composition.**  The evaluation env is seeded once, and an unseeded
  reset advances the command stream, so each evaluation of 30 episodes
  would draw its own number of stand episodes, Binomial(30, 0.15), and SB3's
  ``best_model`` (the highest 30-episode mean) would rank that draw before
  the policy.  :class:`~environments.shared.eval_diagnostics.StageAwareEvalCallback`
  therefore reseeds the evaluation env with :attr:`CommandEvaluationPlan.seed`
  before every evaluation of a live stage: the seeded reset restarts the
  command stream at episode 0, so every evaluation rolls the same commands
  (:attr:`CommandEvaluationPlan.families`) from the same reset draws, and
  the mean of one evaluation is comparable with the next.  The run's stand
  count is then fixed by its evaluation seed, not drawn anew each time.
* **The spread.**  ``robust_best_model`` is scored ``mean - std`` over an
  evaluation's returns, and pooled over both families the family gap reads
  as risk: a checkpoint that stands worse narrows the gap and scores higher.
  On a live stage :class:`~environments.shared.curriculum.checkpoints.RobustBestModelCallback`
  scores each family apart, ``mean - risk_coef * std`` within the family,
  and combines them at the families' probabilities (cruise ``1 - p``, stand
  ``p``), renormalized over the families the evaluation holds
  (:func:`family_robust_score`).

Under command_mode "none" neither applies: the evaluation env is not
reseeded and the robust score is the pooled one, exactly as before.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..command_frame import COMMAND_STREAM_DOMAIN
from ..direction_commands import DirectionCommandConfig, DirectionCommandController
from .command_tracking_gate import CRUISE_FAMILY, STAND_FAMILY

#: The families an evaluation episode of a live stage is read as, in the order the logs and records list them.
COMMAND_EVALUATION_FAMILIES: tuple[str, ...] = (CRUISE_FAMILY, STAND_FAMILY)


def episode_command_family(command_config: DirectionCommandConfig, seed: int, episode_index: int) -> str:
    """The walk family of episode *episode_index* after ``reset(seed=seed)``: ``stand`` at a zero requested speed.

    ``BaseDinoEnv`` draws each episode's command from
    ``SeedSequence([seed, episode_index, COMMAND_STREAM_DOMAIN])``: the
    seeded reset is episode 0 and each unseeded reset after it the next.
    The requested speed does not depend on the spawn heading, so the
    heading passed to the controller is immaterial.
    """
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError(f"episode_index must be a non-negative integer, not {episode_index!r}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), episode_index, COMMAND_STREAM_DOMAIN]))
    state = DirectionCommandController(command_config).reset(rng, 0.0)
    return STAND_FAMILY if state.desired_speed == 0.0 else CRUISE_FAMILY


def info_command_family(info: Mapping[str, object]) -> str:
    """The family of the command a live step's ``info`` reports: ``stand`` at an exactly zero ``desired_speed``.

    Read from an episode's first step, it is the family of the reset's
    draw.  Raises ``KeyError`` for an info with no command (a stage under
    command_mode "none").
    """
    speed = float(info["desired_speed"])  # type: ignore[arg-type]
    return STAND_FAMILY if speed == 0.0 else CRUISE_FAMILY


@dataclass(frozen=True)
class CommandEvaluationPlan:
    """The command panel a live stage's in-training evaluation rolls at every evaluation.

    ``seed`` is the evaluation env's seed (``train_base.eval_env_seed``),
    applied before every evaluation; ``families`` is the family of each of
    its episodes in order (:func:`episode_command_family`);
    ``stop_probability`` is the stage's, the stand family's weight in the
    robust score.
    """

    seed: int
    families: tuple[str, ...]
    stop_probability: float

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, not {self.seed!r}")
        unknown = sorted(set(self.families) - set(COMMAND_EVALUATION_FAMILIES))
        if unknown:
            raise ValueError(f"unknown command families {unknown}")
        if not self.families:
            raise ValueError("a command evaluation plan needs at least one episode")
        p = float(self.stop_probability)
        if not math.isfinite(p) or not 0.0 <= p < 1.0:
            raise ValueError(f"stop_probability must lie in [0, 1), not {self.stop_probability!r}")

    def family_weights(self) -> dict[str, float]:
        """Each family's weight in the robust score: its probability under the stage's command distribution."""
        p = float(self.stop_probability)
        return {CRUISE_FAMILY: 1.0 - p, STAND_FAMILY: p}

    def family_counts(self) -> dict[str, int]:
        return {family: self.families.count(family) for family in COMMAND_EVALUATION_FAMILIES}


def command_evaluation_plan(
    command_config: DirectionCommandConfig, seed: int, n_episodes: int
) -> CommandEvaluationPlan:
    """The plan of an evaluation env reseeded with *seed*: the families of its first *n_episodes* episodes."""
    if isinstance(n_episodes, bool) or not isinstance(n_episodes, int) or n_episodes < 1:
        raise ValueError(f"n_episodes must be a positive integer, not {n_episodes!r}")
    families = tuple(episode_command_family(command_config, seed, index) for index in range(n_episodes))
    return CommandEvaluationPlan(
        seed=int(seed), families=families, stop_probability=float(command_config.stop_probability)
    )


def family_statistics(
    returns: "Sequence[float] | np.ndarray", families: Sequence[str]
) -> dict[str, tuple[int, float, float]]:
    """Each family's ``(episodes, mean, std)`` over the returns it holds, in :data:`COMMAND_EVALUATION_FAMILIES` order.

    ``std`` is the population standard deviation (``numpy.std``), as the
    pooled robust score uses.  A family the evaluation does not hold is
    absent; a family label outside the walk's two, or *families* of another
    length than *returns*, is refused.
    """
    values = np.asarray(returns, dtype=float).reshape(-1)
    labels = list(families)
    if len(labels) != values.size:
        raise ValueError(f"{values.size} returns but {len(labels)} command families")
    unknown = sorted(set(labels) - set(COMMAND_EVALUATION_FAMILIES))
    if unknown:
        raise ValueError(f"unknown command families {unknown}")
    out: dict[str, tuple[int, float, float]] = {}
    for family in COMMAND_EVALUATION_FAMILIES:
        part = values[[label == family for label in labels]]
        if part.size:
            out[family] = (int(part.size), float(part.mean()), float(part.std()))
    return out


def family_robust_score(
    returns: "Sequence[float] | np.ndarray",
    families: Sequence[str],
    weights: Mapping[str, float],
    *,
    risk_coef: float = 1.0,
) -> float:
    """The family-aware robust score: each family's ``mean - risk_coef * std``, combined at *weights*.

    The weights are renormalized over the families present, so an
    evaluation that holds one family is scored on it alone; the plan of a
    reseeded evaluation holds the same families at every evaluation, so the
    scores of one run stay comparable.  Refused when the families present
    carry no weight.
    """
    stats = family_statistics(returns, families)
    weight_sum = sum(float(weights[family]) for family in stats)
    if not weight_sum > 0.0:
        raise ValueError(f"the families present ({sorted(stats)}) carry no weight in {dict(weights)}")
    total = sum(float(weights[family]) * (mean - risk_coef * std) for family, (_, mean, std) in stats.items())
    return float(total / weight_sum)


__all__ = [
    "COMMAND_EVALUATION_FAMILIES",
    "CommandEvaluationPlan",
    "command_evaluation_plan",
    "episode_command_family",
    "family_robust_score",
    "family_statistics",
    "info_command_family",
]
