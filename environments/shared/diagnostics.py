"""DiagnosticsCallback for logging per-component reward breakdowns to TensorBoard.

Extracted from ``train_base.py`` for maintainability.  The callback is
self-contained and depends only on NumPy and the SB3 BaseCallback interface.
"""

import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]

_SB3_AVAILABLE = False
try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback

    _SB3_AVAILABLE = True
except ImportError:
    _BaseCallback = object  # type: ignore[misc,assignment]


def _sanitize(value: float) -> float:
    """Replace NaN/Inf with 0.0 to prevent tensorboardX x2num warnings."""
    if _np is not None and not _np.isfinite(value):
        return 0.0
    return value


def _safe_mean(vals: list) -> float:
    """Return the mean of *vals*, or NaN if the list is empty."""
    if vals:
        return float(_np.mean(vals))
    return float("nan")


class DiagnosticsCallback(_BaseCallback):
    """Logs per-component reward breakdowns and training diagnostics to TensorBoard.

    Tracked metrics (under ``diagnostics/`` in TensorBoard):
      - Per-component rewards: reward_forward, reward_alive, reward_energy, etc.
      - Environment state: forward_vel, prey_distance, pelvis_height, tilt_angle
      - Observation statistics: mean, std, max absolute value
      - Action statistics: mean, std
      - VecNormalize running variance for observations and returns
      - Termination reason breakdown (fraction per reason)
      - Reward plateau detection with console warnings

    When *log_dir* is provided, per-rollout averages of INFO_KEYS are also
    saved to ``diagnostics.npz`` so they can be plotted alongside the
    evaluation curves produced by the notebook's ``plot_training_curves``.
    """

    REWARD_KEYS = [
        "reward_forward",
        "reward_backward",
        "reward_drift",
        "reward_alive",
        "reward_energy",
        "reward_tail",
        "reward_posture",
        "reward_nosedive",
        "reward_smoothness",
        "reward_strike",
        "reward_approach",
        "reward_gait",
        "reward_heading",
        "reward_lateral",
        "reward_spin",
        "reward_speed",
        "reward_idle",
        # Species-specific reward components
        "reward_bite",  # T-Rex
        "reward_food",  # Brachiosaurus
        "reward_height",  # T-Rex, Brachiosaurus
        "reward_proximity",  # Velociraptor
        "reward_claw_proximity",  # Velociraptor
        "reward_head_proximity",  # T-Rex, Brachiosaurus
        "reward_gait_symmetry",  # Brachiosaurus
    ]
    INFO_KEYS = [
        "forward_vel",
        "prey_distance",
        "pelvis_height",
        "tilt_angle",
        "tail_instability",
        "contact_asymmetry",
        "action_delta",
        "heading_alignment",
        "lateral_vel",
        "forward_z",
        "approach_delta",
        "strike_success",
        "bite_success",
        "r_foot_contact",
        "l_foot_contact",
        "rr_foot_contact",
        "rl_foot_contact",
        "pelvis_angular_vel",
        "pelvis_yaw_vel",
        "drift_distance",
        "spin_instability",
        "distance_traveled",
        "abs_speed",
        "food_reached",
    ]

    def __init__(self, plateau_window=10, plateau_threshold=1.0, log_dir=None, verbose=0):
        if _SB3_AVAILABLE:
            super().__init__(verbose)
        else:
            super().__init__()
            self.verbose = verbose
            self.locals: dict = {}
            self.model = None
            self.logger = None
            self.num_timesteps = 0
            self.training_env = None
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._step_infos = {k: [] for k in self.REWARD_KEYS + self.INFO_KEYS}
        self._rollout_ep_rewards: list[float] = []
        self._rollout_terminations: Counter = Counter()
        self._history_timesteps: list[int] = []
        self._history = {k: [] for k in self.INFO_KEYS}
        self._history_rewards = {k: [] for k in self.REWARD_KEYS}
        self._history_heading_std: list[float] = []
        self._history_terminations: dict[str, list[float]] = {}
        self._history_term_timesteps: list[int] = []

    if not _SB3_AVAILABLE:

        def init_callback(self, model) -> None:
            """Minimal stand-in for ``BaseCallback.init_callback`` when SB3 is absent."""
            self.model = model
            self.logger = getattr(model, "logger", None)
            self.training_env = getattr(model, "get_env", lambda: None)()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self.REWARD_KEYS + self.INFO_KEYS:
                if key in info:
                    self._step_infos[key].append(float(info[key]))
            if "termination_reason" in info:
                self._rollout_terminations[info["termination_reason"]] += 1
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._step_infos.items():
            if values:
                self.logger.record(f"diagnostics/{key}", _sanitize(float(_np.mean(values))))

        has_info = any(self._step_infos[k] for k in self.INFO_KEYS)
        if has_info:
            self._history_timesteps.append(self.num_timesteps)
            for key in self.INFO_KEYS:
                vals = self._step_infos[key]
                self._history[key].append(_safe_mean(vals))
            for key in self.REWARD_KEYS:
                vals = self._step_infos[key]
                self._history_rewards[key].append(_safe_mean(vals))
            # Track heading alignment std to distinguish spinning from stable heading
            heading_vals = self._step_infos.get("heading_alignment", [])
            self._history_heading_std.append(float(_np.std(heading_vals)) if len(heading_vals) > 1 else float("nan"))
            self._save_diagnostics()

        self._step_infos = {k: [] for k in self.REWARD_KEYS + self.INFO_KEYS}

        total_terminations = sum(self._rollout_terminations.values())
        if total_terminations > 0:
            self._history_term_timesteps.append(self.num_timesteps)
            for reason, count in self._rollout_terminations.items():
                frac = count / total_terminations
                self.logger.record(f"terminations/{reason}", _sanitize(frac))
                self._history_terminations.setdefault(reason, []).append(frac)
            self.logger.record("terminations/total_count", total_terminations)
            self._save_diagnostics()
        self._rollout_terminations.clear()

        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.observations is not None:
            obs = self.model.rollout_buffer.observations
            self.logger.record("diagnostics/obs_mean", _sanitize(float(_np.mean(obs))))
            self.logger.record("diagnostics/obs_std", _sanitize(float(_np.std(obs))))
            self.logger.record("diagnostics/obs_max_abs", _sanitize(float(_np.max(_np.abs(obs)))))

        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.actions is not None:
            acts = self.model.rollout_buffer.actions
            self.logger.record("diagnostics/action_mean", _sanitize(float(_np.mean(acts))))
            self.logger.record("diagnostics/action_std", _sanitize(float(_np.std(acts))))

        env = self.training_env
        if hasattr(env, "obs_rms"):
            self.logger.record("diagnostics/vecnorm_obs_var_mean", _sanitize(float(_np.mean(env.obs_rms.var))))
        if hasattr(env, "ret_rms"):
            self.logger.record("diagnostics/vecnorm_ret_var", _sanitize(float(_np.mean(env.ret_rms.var))))

        ep_rewards = [info["episode"]["r"] for info in self.locals.get("infos", []) if "episode" in info]
        if ep_rewards:
            self._rollout_ep_rewards.append(_np.mean(ep_rewards))
            if len(self._rollout_ep_rewards) >= self.plateau_window:
                recent = self._rollout_ep_rewards[-self.plateau_window :]
                variation = max(recent) - min(recent)
                self.logger.record("diagnostics/reward_variation", _sanitize(variation))
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
        save_dict = {"timesteps": _np.array(self._history_timesteps)}
        for key in self.INFO_KEYS:
            save_dict[key] = _np.array(self._history[key])
        for key in self.REWARD_KEYS:
            arr = self._history_rewards[key]
            if len(arr) == len(self._history_timesteps):
                save_dict[key] = _np.array(arr)
        if len(self._history_heading_std) == len(self._history_timesteps):
            save_dict["heading_alignment_std"] = _np.array(self._history_heading_std)
        if self._history_term_timesteps:
            save_dict["term_timesteps"] = _np.array(self._history_term_timesteps)
            for reason, fracs in self._history_terminations.items():
                save_dict[f"term_{reason}"] = _np.array(fracs)
        _np.savez(str(self._log_dir / "diagnostics.npz"), **save_dict)  # type: ignore[arg-type]
