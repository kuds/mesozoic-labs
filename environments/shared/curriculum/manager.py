"""Stage thresholds and the backend-independent curriculum manager.

:class:`CurriculumManager` owns the advancement decision and per-stage config
loading; it has no SB3 dependency, so it can be driven from a JAX loop or a
notebook just as well as from :class:`~.advancement.CurriculumCallback`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from environments.shared.config import load_all_stages

from .gate_schema import validate_gate_config
from .stance_gate import (
    STANCE_GATE_KIND,
    StanceGateThresholds,
    StancePanel,
    evaluate_stance_gate,
)

logger = logging.getLogger(__name__)


@dataclass
class StageThreshold:
    """Performance thresholds that must be met to advance past a stage.

    Carries the union of every gate kind's fields; ``gate_kind`` selects which
    subset :meth:`CurriculumManager.should_advance` actually evaluates. The
    defaults are the permissive ones only for ``reward_and_length/v1``, which
    is why :func:`thresholds_from_configs` runs the fail-closed schema check
    before any of them is populated.
    """

    gate_kind: str = "reward_and_length/v1"

    # reward_and_length/v1
    min_avg_reward: float = -np.inf
    min_avg_episode_length: float = 0.0
    min_avg_forward_vel: float = 0.0
    min_success_rate: float = 0.0

    # stance_quality/v1.  Ceilings default to +inf and the floor to 0.0 so a
    # field the schema failed to populate cannot silently tighten the gate;
    # the schema requires all three of the real ones, so these defaults are
    # only ever seen by a stage using a different kind.
    min_full_horizon_fraction: float = 0.0
    max_unsupported_duty: float = np.inf
    max_unsupported_duty_ucb: float = np.inf
    settle_steps: int = 0

    # Shared
    min_eval_episodes: int = 10
    required_consecutive: int = 3


class CurriculumManager:
    """Manages automated progression through curriculum stages.

    Tracks evaluation results and determines when a training stage's
    performance thresholds have been met, signalling that it is time
    to advance to the next stage.

    Args:
        species: Species name used to load TOML configs.
        stage_thresholds: Mapping from stage number to threshold dict.
            Keys in each dict should match ``StageThreshold`` fields.
        start_stage: Initial curriculum stage (default 1).
        total_stages: Total number of stages (default 3).
    """

    def __init__(
        self,
        species: str,
        stage_thresholds: dict[int, dict[str, Any]] | None = None,
        start_stage: int = 1,
        total_stages: int = 3,
    ):
        self.species = species
        self.total_stages = total_stages
        self._current_stage = start_stage
        self._configs = load_all_stages(species)

        # Build threshold objects per stage
        self._thresholds: dict[int, StageThreshold] = {}
        for stage in range(1, total_stages + 1):
            if stage_thresholds and stage in stage_thresholds:
                self._thresholds[stage] = StageThreshold(**stage_thresholds[stage])
            else:
                self._thresholds[stage] = StageThreshold()

        # History of evaluation results per stage
        self._eval_history: dict[int, list[dict[str, float]]] = {s: [] for s in range(1, total_stages + 1)}
        self._consecutive_passes: dict[int, int] = {s: 0 for s in range(1, total_stages + 1)}

        logger.info(
            "CurriculumManager initialized for %s: stage %d/%d",
            species,
            start_stage,
            total_stages,
        )

    @property
    def current_stage(self) -> int:
        """Current curriculum stage number."""
        return self._current_stage

    @property
    def current_threshold(self) -> StageThreshold:
        """Effective threshold for the current stage, including overrides."""

        return self._thresholds[self._current_stage]

    @property
    def is_final_stage(self) -> bool:
        """Whether the manager is on the last stage."""
        return self._current_stage >= self.total_stages

    def current_config(self) -> dict[str, Any]:
        """Return the TOML config dict for the current stage."""
        return self._configs[self._current_stage]

    def record_eval(
        self,
        rewards: list[float],
        episode_lengths: list[float],
        forward_velocities: list[float] | None = None,
        success_rates: list[float] | None = None,
        stance_panel: StancePanel | None = None,
    ) -> dict[str, float]:
        """Record evaluation results for the current stage.

        Args:
            rewards: List of episode total rewards from evaluation.
            episode_lengths: List of episode lengths from evaluation.
            forward_velocities: Optional list of mean forward velocities
                per episode (m/s). Used for locomotion stage gating.
            success_rates: Optional list of per-episode success flags
                (1.0 if prey contact / food reached, 0.0 otherwise).
            stance_panel: Optional stance summary for ``stance_quality/v1``.
                Both backends must build this with
                :func:`~environments.shared.curriculum.stance_gate.summarize_stance_panel`
                rather than reducing per-episode traces themselves, so the
                settling window and the failed-episode rule cannot drift
                between SB3 and JAX — which is the whole point of the
                versioned schema.

        Returns:
            Summary dict with mean/std statistics.
        """
        summary = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
            "n_episodes": len(rewards),
        }
        if forward_velocities is not None:
            summary["mean_forward_vel"] = float(np.mean(forward_velocities))
        if success_rates is not None:
            summary["mean_success_rate"] = float(np.mean(success_rates))
        if stance_panel is not None:
            # The bound is stored as a scalar rather than the raw duties: it is
            # a pure function of them, and the history is serialized into run
            # summaries where per-episode traces would not belong.
            summary["full_horizon_fraction"] = stance_panel.full_horizon_fraction
            summary["mean_unsupported_duty"] = stance_panel.mean_unsupported_duty
            summary["unsupported_duty_ucb"] = stance_panel.unsupported_duty_ucb
            summary["n_duty_episodes"] = stance_panel.n_duty_episodes
        self._eval_history[self._current_stage].append(summary)

        vel_str = ""
        if "mean_forward_vel" in summary:
            vel_str = f", fwd_vel={summary['mean_forward_vel']:.2f} m/s"
        success_str = ""
        if "mean_success_rate" in summary:
            success_str = f", success={summary['mean_success_rate']:.0%}"
        stance_str = ""
        if "full_horizon_fraction" in summary:
            stance_str = (
                f", horizon={summary['full_horizon_fraction']:.1%}"
                f", duty={summary['mean_unsupported_duty']:.4f}"
                f" (ucb {summary['unsupported_duty_ucb']:.4f}, n={summary['n_duty_episodes']:.0f})"
            )
        logger.info(
            "Stage %d eval: reward=%.2f +/- %.2f, length=%.1f +/- %.1f%s%s%s (%d eps)",
            self._current_stage,
            summary["mean_reward"],
            summary["std_reward"],
            summary["mean_length"],
            summary["std_length"],
            vel_str,
            success_str,
            stance_str,
            summary["n_episodes"],
        )
        return summary

    def should_advance(
        self,
        rewards: list[float] | None = None,
        episode_lengths: list[float] | None = None,
        forward_velocities: list[float] | None = None,
        success_rates: list[float] | None = None,
        stance_panel: StancePanel | None = None,
    ) -> bool:
        """Check whether performance thresholds are met for advancement.

        If ``rewards`` and ``episode_lengths`` are provided they are
        recorded first via :meth:`record_eval`.

        Args:
            rewards: Per-episode total rewards.
            episode_lengths: Per-episode step counts.
            forward_velocities: Per-episode mean forward velocities (m/s).
            success_rates: Per-episode success flags (1.0 if prey contact
                / food reached, 0.0 otherwise).
            stance_panel: Stance summary, required by ``stance_quality/v1``.
                A stage declaring that kind without one fails closed.

        Returns:
            True if the current stage thresholds have been met for the
            required number of consecutive evaluations.
        """
        if rewards is not None and episode_lengths is not None:
            self.record_eval(rewards, episode_lengths, forward_velocities, success_rates, stance_panel)

        threshold = self._thresholds[self._current_stage]
        history = self._eval_history[self._current_stage]

        if not history:
            return False

        latest = history[-1]

        if threshold.gate_kind == "none/v1":
            # A declared non-advancing pilot. Previously this stage produced no
            # threshold entry at all and fell through to StageThreshold's
            # permissive defaults, which pass on any evaluation — the schema
            # rejects "none/v1" whenever advancement is enabled, so nothing
            # reached that path, but it was fail-open one config change away.
            passes = False
        elif threshold.gate_kind == STANCE_GATE_KIND:
            passes = self._stance_gate_passes(latest)
        else:
            passes = self._reward_and_length_gate_passes(latest, threshold)

        if passes:
            self._consecutive_passes[self._current_stage] += 1
        else:
            self._consecutive_passes[self._current_stage] = 0

        met = self._consecutive_passes[self._current_stage] >= threshold.required_consecutive

        if met:
            logger.info(
                "Stage %d thresholds met (%d consecutive passes). Ready to advance.",
                self._current_stage,
                threshold.required_consecutive,
            )

        return met

    def _reward_and_length_gate_passes(self, latest: dict[str, float], threshold: StageThreshold) -> bool:
        """Evaluate ``reward_and_length/v1`` against one evaluation."""
        passes = (
            latest["mean_reward"] >= threshold.min_avg_reward
            and latest["mean_length"] >= threshold.min_avg_episode_length
            and latest["n_episodes"] >= threshold.min_eval_episodes
        )

        # Forward velocity gate (only checked when threshold is > 0)
        if threshold.min_avg_forward_vel > 0.0:
            mean_vel = latest.get("mean_forward_vel", 0.0)
            passes = passes and mean_vel >= threshold.min_avg_forward_vel

        # Success rate gate (only checked when threshold is > 0)
        if threshold.min_success_rate > 0.0:
            mean_success = latest.get("mean_success_rate", 0.0)
            passes = passes and mean_success >= threshold.min_success_rate

        return passes

    def _stance_gate_passes(self, latest: dict[str, float]) -> bool:
        """Evaluate ``stance_quality/v1`` against one evaluation.

        Reconstructs a :class:`StancePanel` from the recorded summary so the
        shared criterion logic in
        :func:`~environments.shared.curriculum.stance_gate.evaluate_stance_gate`
        is the single implementation both backends run.

        A stage declaring this kind whose evaluation carried no stance panel
        fails closed and says so once per evaluation, rather than falling
        through to the reward criteria — advancing on evidence nobody
        collected is the exact failure mode the versioned schema exists to
        prevent.
        """
        threshold = self._thresholds[self._current_stage]

        if "full_horizon_fraction" not in latest:
            logger.warning(
                "Stage %d declares gate_kind %s but the evaluation carried no stance panel; "
                "refusing to advance. The eval path must build one with "
                "stance_gate.summarize_stance_panel().",
                self._current_stage,
                STANCE_GATE_KIND,
            )
            return False

        panel = StancePanel(
            n_episodes=int(latest["n_episodes"]),
            full_horizon_fraction=latest["full_horizon_fraction"],
            mean_reward=latest["mean_reward"],
            n_duty_episodes=int(latest["n_duty_episodes"]),
            mean_unsupported_duty=latest["mean_unsupported_duty"],
            unsupported_duty_ucb=latest["unsupported_duty_ucb"],
        )
        passed, failures = evaluate_stance_gate(
            panel,
            StanceGateThresholds(
                min_full_horizon_fraction=threshold.min_full_horizon_fraction,
                max_unsupported_duty=threshold.max_unsupported_duty,
                max_unsupported_duty_ucb=threshold.max_unsupported_duty_ucb,
                settle_steps=threshold.settle_steps,
                min_eval_episodes=threshold.min_eval_episodes,
                min_avg_reward=threshold.min_avg_reward,
                required_consecutive=threshold.required_consecutive,
            ),
        )
        if not passed:
            logger.info("Stage %d stance gate not met: %s", self._current_stage, "; ".join(failures))
        return passed

    def advance(self) -> int:
        """Advance to the next curriculum stage.

        Returns:
            The new stage number.

        Raises:
            RuntimeError: If already on the final stage.
        """
        if self.is_final_stage:
            raise RuntimeError(f"Cannot advance past final stage {self.total_stages}")

        prev = self._current_stage
        self._current_stage += 1

        logger.info(
            "Advanced from stage %d to stage %d (%s -> %s)",
            prev,
            self._current_stage,
            self._configs[prev]["name"],
            self._configs[self._current_stage]["name"],
        )

        return self._current_stage

    def summary(self) -> dict[str, Any]:
        """Return a summary of the curriculum state for logging/serialization."""
        return {
            "species": self.species,
            "current_stage": self._current_stage,
            "total_stages": self.total_stages,
            "eval_history": dict(self._eval_history),
            "consecutive_passes": dict(self._consecutive_passes),
        }


def thresholds_from_configs(
    configs: dict[int, dict[str, Any]],
    *,
    advancement_enabled: bool = True,
) -> dict[int, dict[str, Any]]:
    """Extract stage thresholds from loaded TOML configs.

    Reads the ``curriculum_kwargs`` section from each stage config and
    returns a dict suitable for passing to ``CurriculumManager``.

    Args:
        configs: Dict mapping stage number to config dict (from ``load_all_stages``).
        advancement_enabled: Whether this run may advance between stages. Passed
            through to :func:`~environments.shared.curriculum.gate_schema.validate_gate_config`,
            which tolerates a missing gate declaration only when it is ``False``.

    Returns:
        Dict mapping stage number to threshold kwargs.

    Raises:
        GateSchemaError: If any stage's gate declaration is missing, unknown,
            or malformed.
    """
    thresholds: dict[int, dict[str, Any]] = {}
    for stage, cfg in configs.items():
        cur = cfg.get("curriculum_kwargs", {})
        # Fail closed. Silently dropping an unrecognised key here is what let a
        # composite-only gate config produce no thresholds at all, after which
        # StageThreshold's permissive defaults advanced the stage on any
        # evaluation whatsoever. See gate_schema for the full failure mode.
        gate_kind = validate_gate_config(stage, cur, advancement_enabled=advancement_enabled)
        # The kind is carried onto the threshold so ``should_advance`` selects
        # the same criteria the schema validated, rather than inferring the
        # gate from which fields happen to be present — inference is how a
        # dropped field used to become a disabled criterion.
        threshold_fields: dict[str, Any] = {"gate_kind": gate_kind}
        for key in (
            "min_avg_reward",
            "min_avg_episode_length",
            "min_avg_forward_vel",
            "min_success_rate",
            "min_full_horizon_fraction",
            "max_unsupported_duty",
            "max_unsupported_duty_ucb",
            "settle_steps",
            "min_eval_episodes",
            "required_consecutive",
        ):
            if key in cur:
                threshold_fields[key] = cur[key]
        thresholds[stage] = threshold_fields
    return thresholds
