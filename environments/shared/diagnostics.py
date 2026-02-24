"""
Training diagnostics callback for Stable-Baselines3.

Provides a species-agnostic callback that logs per-component reward
breakdowns, environment state metrics, observation/action statistics,
VecNormalize running variance, termination reason breakdown, and
plateau detection to TensorBoard.

When ``log_dir`` is provided, per-rollout averages of ``info_keys`` are
persisted to ``diagnostics.npz`` for offline plotting.

Usage::

    from environments.shared.diagnostics import DiagnosticsCallback

    diag = DiagnosticsCallback(
        reward_keys=["reward_forward", "reward_alive", "reward_energy"],
        info_keys=["forward_vel", "tilt_angle"],
        log_dir=str(stage_dir),
    )

    model.learn(total_timesteps=500_000, callback=diag)
"""

import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    BaseCallback = object  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class DiagnosticsCallback(BaseCallback):
    """Logs per-component reward breakdowns and training diagnostics to TensorBoard.

    Tracked metrics (under ``diagnostics/`` in TensorBoard):
      - Per-component rewards from ``reward_keys``
      - Environment state metrics from ``info_keys``
      - Observation statistics: mean, std, max absolute value
      - Action statistics: mean, std
      - VecNormalize running variance for observations and returns
      - Termination reason breakdown (fraction per reason)
      - Reward plateau detection with console warnings

    When *log_dir* is provided, per-rollout averages of *info_keys* are also
    saved to ``diagnostics.npz`` so they can be plotted alongside the
    evaluation curves produced by :func:`plot_training_curves`.

    Args:
        reward_keys: Info-dict keys containing per-component reward values.
        info_keys: Info-dict keys containing environment state metrics.
        plateau_window: Number of recent rollouts to check for plateau.
        plateau_threshold: Minimum reward variation to avoid a warning.
        log_dir: Directory for ``diagnostics.npz`` persistence.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        reward_keys: List[str],
        info_keys: List[str],
        plateau_window: int = 10,
        plateau_threshold: float = 1.0,
        log_dir: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.reward_keys = list(reward_keys)
        self.info_keys = list(info_keys)
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._all_keys = self.reward_keys + self.info_keys
        self._step_infos: dict[str, list[float]] = {k: [] for k in self._all_keys}
        self._rollout_ep_rewards: list[float] = []
        self._rollout_terminations: Counter = Counter()
        # Per-rollout history for npz persistence
        self._history_timesteps: list[int] = []
        self._history: dict[str, list[float]] = {k: [] for k in self.info_keys}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self._all_keys:
                if key in info:
                    self._step_infos[key].append(float(info[key]))
            # Track termination reasons from completed episodes
            if "termination_reason" in info:
                self._rollout_terminations[info["termination_reason"]] += 1
        return True

    def _on_rollout_end(self) -> None:
        # Per-component reward breakdown
        for key, values in self._step_infos.items():
            if values:
                self.logger.record(f"diagnostics/{key}", np.mean(values))

        # Persist rollout averages for info_keys
        has_info = any(self._step_infos[k] for k in self.info_keys)
        if has_info:
            self._history_timesteps.append(self.num_timesteps)
            for key in self.info_keys:
                vals = self._step_infos[key]
                self._history[key].append(float(np.mean(vals)) if vals else float("nan"))
            self._save_diagnostics()

        self._step_infos = {k: [] for k in self._all_keys}

        # Termination reason breakdown (fraction of terminated episodes)
        total_terminations = sum(self._rollout_terminations.values())
        if total_terminations > 0:
            for reason, count in self._rollout_terminations.items():
                self.logger.record(
                    f"terminations/{reason}",
                    count / total_terminations,
                )
            self.logger.record("terminations/total_count", total_terminations)
        self._rollout_terminations.clear()

        # Observation statistics from rollout buffer
        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.observations is not None:
            obs = self.model.rollout_buffer.observations
            self.logger.record("diagnostics/obs_mean", float(np.mean(obs)))
            self.logger.record("diagnostics/obs_std", float(np.std(obs)))
            self.logger.record("diagnostics/obs_max_abs", float(np.max(np.abs(obs))))

        # Action statistics from rollout buffer
        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.actions is not None:
            acts = self.model.rollout_buffer.actions
            self.logger.record("diagnostics/action_mean", float(np.mean(acts)))
            self.logger.record("diagnostics/action_std", float(np.std(acts)))

        # VecNormalize running statistics
        env = self.training_env
        if hasattr(env, "obs_rms"):
            self.logger.record("diagnostics/vecnorm_obs_var_mean", float(np.mean(env.obs_rms.var)))
        if hasattr(env, "ret_rms"):
            self.logger.record("diagnostics/vecnorm_ret_var", float(np.mean(env.ret_rms.var)))

        # Plateau detection from completed episodes
        ep_rewards = [info["episode"]["r"] for info in self.locals.get("infos", []) if "episode" in info]
        if ep_rewards:
            self._rollout_ep_rewards.append(np.mean(ep_rewards))
            if len(self._rollout_ep_rewards) >= self.plateau_window:
                recent = self._rollout_ep_rewards[-self.plateau_window :]
                variation = max(recent) - min(recent)
                self.logger.record("diagnostics/reward_variation", variation)
                if variation < self.plateau_threshold:
                    logger.warning(
                        "PLATEAU WARNING: Reward variation over last %d rollouts is only %.4f. "
                        "Consider adjusting learning rate or stopping.",
                        self.plateau_window,
                        variation,
                    )

    def _save_diagnostics(self) -> None:
        """Persist accumulated diagnostics to an npz file in the stage dir."""
        if self._log_dir is None:
            return
        save_dict: dict[str, np.ndarray] = {"timesteps": np.array(self._history_timesteps)}
        for key in self.info_keys:
            save_dict[key] = np.array(self._history[key])
        np.savez(self._log_dir / "diagnostics.npz", **save_dict)
