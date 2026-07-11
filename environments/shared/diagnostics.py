"""DiagnosticsCallback for logging per-component reward breakdowns to TensorBoard.

Extracted from ``train_base.py`` for maintainability.  The callback is
self-contained and depends only on NumPy and the SB3 BaseCallback interface.
"""

import logging
import math
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


def _global_grad_norm(parameters) -> float | None:
    """Global L2 norm over the ``.grad`` tensors of *parameters*.

    Returns ``None`` when no parameter carries a gradient yet (e.g. before the
    first optimizer step).  Used to surface PPO's post-update (clipped)
    gradient norm: it saturates at ``max_grad_norm`` when clipping binds — a
    useful "is the clip ceiling always active?" signal — and collapses toward
    0 if the policy stops learning.
    """
    total_sq = 0.0
    found = False
    for p in parameters:
        grad = getattr(p, "grad", None)
        if grad is None:
            continue
        total_sq += float((grad.detach() ** 2).sum().item())
        found = True
    if not found:
        return None
    return math.sqrt(total_sq)


class DiagnosticsCallback(_BaseCallback):
    """Logs per-component reward breakdowns and training diagnostics to TensorBoard.

    Tracked metrics (under ``diagnostics/`` in TensorBoard):
      - Per-component rewards: reward_forward, reward_alive, reward_energy, etc.
      - Environment state: forward_vel, prey_distance, pelvis_height, tilt_angle
      - Observation statistics: mean, std, max absolute value (PPO)
      - Action statistics: mean, std, abs_max, and saturation fraction
        (fraction of action components pinned at the control limit) — for
        both PPO and SAC
      - PPO post-update gradient norm
      - Algorithm-specific train/* metrics captured from SB3's logger
        (PPO: clip_fraction, approx_kl, explained_variance, …; SAC:
        critic_loss, actor_loss, ent_coef, …)
      - VecNormalize running variance for observations and returns
      - Termination reason breakdown (fraction per reason)
      - Reward plateau detection with console warnings

    When *log_dir* is provided, per-rollout averages of INFO_KEYS, the
    captured algorithm metrics (under ``algo_*`` keys, on their own
    ``algo_timesteps`` axis), and termination fractions are saved to
    ``diagnostics.npz`` so they can be plotted alongside the evaluation curves
    produced by the notebook's ``plot_training_curves``.
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
        "reward_total",
    ]
    INFO_KEYS = [
        "forward_vel",
        "backward_vel",
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
        "head_food_distance",  # Brachiosaurus
        "torso_height",  # Brachiosaurus
        "jaw_distance",  # T-Rex
    ]

    def __init__(
        self,
        plateau_window=10,
        plateau_threshold=1.0,
        log_dir=None,
        verbose=0,
        action_saturation_threshold=0.99,
        save_interval_seconds=60.0,
    ):
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
        # |action| at/above this (in the normalized [-1, 1] control space) counts
        # as "saturated" for the diagnostics/action_saturation metric.
        self.action_saturation_threshold = action_saturation_threshold
        self._log_dir = Path(log_dir) if log_dir is not None else None
        self._step_infos = {k: [] for k in self.REWARD_KEYS + self.INFO_KEYS}
        self._rollout_ep_rewards: list[float] = []
        # Episode returns + actions seen during the CURRENT rollout (flushed at
        # rollout end).  Accumulating per-step avoids the old bias of only
        # sampling episodes that ended on the rollout's final step.
        self._rollout_ep_returns: list[float] = []
        self._step_actions: list = []
        self._rollout_terminations: Counter = Counter()
        self._history_timesteps: list[int] = []
        self._history = {k: [] for k in self.INFO_KEYS}
        self._history_rewards = {k: [] for k in self.REWARD_KEYS}
        self._history_heading_std: list[float] = []
        self._history_terminations: dict[str, list[float]] = {}
        self._history_term_timesteps: list[int] = []
        # Algorithm-specific metrics (PPO clip_fraction/approx_kl/…,
        # SAC critic_loss/ent_coef/…) captured from SB3's logger.
        self._history_algo_timesteps: list[int] = []
        self._history_algo: dict[str, list[float]] = {}
        # Throttle for diagnostics.npz rewrites (see _on_rollout_end).
        self.save_interval_seconds = save_interval_seconds
        self._last_save_time: float | None = None

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
            # Capture EVERY completed episode's return, not just those ending on
            # the rollout's final step (Monitor sets info["episode"] on done).
            if "episode" in info and "r" in info["episode"]:
                self._rollout_ep_returns.append(float(info["episode"]["r"]))

        # Accumulate the actions taken this step. self.locals["actions"] is
        # populated by both the on-policy (PPO) and off-policy (SAC) collection
        # loops, so this works uniformly for both algorithms.
        actions = self.locals.get("actions")
        if actions is not None and _np is not None:
            self._step_actions.append(_np.asarray(actions, dtype=float).reshape(-1))
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

        self._step_infos = {k: [] for k in self.REWARD_KEYS + self.INFO_KEYS}

        total_terminations = sum(self._rollout_terminations.values())
        if total_terminations > 0:
            self._history_term_timesteps.append(self.num_timesteps)
            n_term_points = len(self._history_term_timesteps)
            for reason, count in self._rollout_terminations.items():
                frac = count / total_terminations
                self.logger.record(f"terminations/{reason}", _sanitize(frac))
                # Back-fill newly-seen reasons with 0.0 for prior rollouts so
                # every series stays aligned with term_timesteps (otherwise
                # plots shift whenever a reason skips a rollout).
                series = self._history_terminations.setdefault(reason, [0.0] * (n_term_points - 1))
                series.append(frac)
            # Reasons absent from THIS rollout get an explicit 0.0
            for reason, series in self._history_terminations.items():
                if len(series) < n_term_points:
                    series.append(0.0)
            self.logger.record("terminations/total_count", total_terminations)
        self._rollout_terminations.clear()

        if hasattr(self.model, "rollout_buffer") and self.model.rollout_buffer.observations is not None:
            obs = self.model.rollout_buffer.observations
            self.logger.record("diagnostics/obs_mean", _sanitize(float(_np.mean(obs))))
            self.logger.record("diagnostics/obs_std", _sanitize(float(_np.std(obs))))
            self.logger.record("diagnostics/obs_max_abs", _sanitize(float(_np.max(_np.abs(obs)))))

        # Action statistics + saturation from the actions actually taken this
        # rollout (covers PPO and SAC alike — SAC has no rollout_buffer, so the
        # old buffer-based path logged nothing for it).  Saturation is the
        # fraction of action components pinned at the control limit, the key
        # early-warning of a degenerate bang-bang policy in continuous control.
        if self._step_actions and _np is not None:
            acts = _np.concatenate(self._step_actions)
            self.logger.record("diagnostics/action_mean", _sanitize(float(_np.mean(acts))))
            self.logger.record("diagnostics/action_std", _sanitize(float(_np.std(acts))))
            self.logger.record("diagnostics/action_abs_max", _sanitize(float(_np.max(_np.abs(acts)))))
            saturation = float(_np.mean(_np.abs(acts) >= self.action_saturation_threshold))
            self.logger.record("diagnostics/action_saturation", _sanitize(saturation))
        self._step_actions = []

        # PPO post-update (clipped) gradient norm.  Scoped to PPO: SAC runs
        # several separate optimizers, so a single global norm is meaningless.
        if hasattr(self.model, "rollout_buffer") and hasattr(self.model, "policy"):
            try:
                gn = _global_grad_norm(self.model.policy.parameters())
                if gn is not None:
                    self.logger.record("diagnostics/grad_norm", _sanitize(gn))
            except (TypeError, AttributeError):
                logger.debug("grad_norm logging skipped (parameters unavailable)", exc_info=True)

        # Algorithm-specific train/* metrics → durable diagnostics.npz.
        self._record_algo_metrics()

        env = self.training_env
        if hasattr(env, "obs_rms"):
            self.logger.record("diagnostics/vecnorm_obs_var_mean", _sanitize(float(_np.mean(env.obs_rms.var))))
        if hasattr(env, "ret_rms"):
            self.logger.record("diagnostics/vecnorm_ret_var", _sanitize(float(_np.mean(env.ret_rms.var))))

        # Plateau detection over the mean episode return of each rollout.  Uses
        # ALL episodes that completed during the rollout (accumulated in
        # _on_step), not just those ending on the final step.
        if self._rollout_ep_returns:
            self._rollout_ep_rewards.append(float(_np.mean(self._rollout_ep_returns)))
            self._rollout_ep_returns = []
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

        # Persist once per rollout-end, throttled: _save_diagnostics rewrites
        # the ENTIRE accumulated history, and SAC fires on_rollout_end every
        # train_freq steps (~hundreds of times more often than PPO), which
        # made per-event saves O(n^2) I/O on a growing file — brutal on GCS
        # FUSE mounts.  A final flush happens in _on_training_end.
        import time as _time

        now = _time.monotonic()
        if self._last_save_time is None or now - self._last_save_time >= self.save_interval_seconds:
            self._last_save_time = now
            self._save_diagnostics()

    def _on_training_end(self) -> None:
        """Final flush so the npz always contains the complete history."""
        self._save_diagnostics()

    def _collect_algo_metrics(self) -> dict:
        """Snapshot SB3's algorithm-specific ``train/*`` metrics.

        SB3 PPO writes ``train/clip_fraction``, ``train/approx_kl``,
        ``train/explained_variance``, ``train/entropy_loss`` … and SAC writes
        ``train/critic_loss``, ``train/actor_loss``, ``train/ent_coef`` … into
        the logger's ``name_to_value`` dict during ``train()``.  These reach
        TensorBoard but never the durable ``diagnostics.npz``; capturing them
        here gives BOTH algorithms first-class, offline-plottable diagnostics.

        Returns the ``train/``-prefixed keys (prefix stripped) that currently
        hold a finite scalar.  Empty when the logger is unavailable (before the
        first update, or under a test mock where ``name_to_value`` is not a
        real dict).

        Note: SB3 fires ``on_rollout_end`` *before* the iteration's
        ``train()``, so the snapshot holds the **previous** update's values
        tagged with the current ``num_timesteps`` — the ``algo_*`` series are
        shifted one rollout to the right.  Acceptable for offline plots.
        """
        logger_obj = getattr(self.model, "logger", None)
        name_to_value = getattr(logger_obj, "name_to_value", None)
        if not isinstance(name_to_value, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in name_to_value.items():
            if not isinstance(key, str) or not key.startswith("train/"):
                continue
            # Skip bools (isinstance(True, int) is True) and anything that is
            # not a real scalar.  float() coerces numpy float32/float64 scalars,
            # which SB3 commonly stores, while rejecting strings/arrays/None.
            if isinstance(value, bool):
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            if _np is not None and not _np.isfinite(fval):
                continue
            out[key[len("train/") :]] = fval
        return out

    def _record_algo_metrics(self) -> None:
        """Append the latest algorithm ``train/*`` metrics to the npz history."""
        metrics = self._collect_algo_metrics()
        if not metrics:
            return
        self._history_algo_timesteps.append(self.num_timesteps)
        n = len(self._history_algo_timesteps)
        # Back-fill any newly-seen key with NaN for prior rollouts so every
        # series stays length-aligned with algo_timesteps.
        for key in metrics:
            if key not in self._history_algo:
                self._history_algo[key] = [float("nan")] * (n - 1)
        for key, series in self._history_algo.items():
            series.append(float(metrics[key]) if key in metrics else float("nan"))

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
        if self._history_algo_timesteps:
            save_dict["algo_timesteps"] = _np.array(self._history_algo_timesteps)
            for key, series in self._history_algo.items():
                if len(series) == len(self._history_algo_timesteps):
                    save_dict[f"algo_{key}"] = _np.array(series)
        # Atomic write: diagnostics.npz is rewritten on every save and often
        # lives on a Drive/GCS FUSE mount where an in-place rewrite can be
        # left truncated if the runtime dies mid-flush.
        from .file_io import atomic_savez

        atomic_savez(self._log_dir / "diagnostics.npz", **save_dict)
